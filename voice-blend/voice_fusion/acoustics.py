"""음향 특징 추출 — torch/librosa 없이 numpy만으로 실제 wav를 측정한다.

여기서 나오는 값이 매니페스트의 특징 축(9개)이 된다. 즉 클러스터링과 병합 매칭은
**실제 음성 측정값** 위에서 돌아간다. (합성 오디오로도 검증 가능: 220Hz 사인파를
넣으면 f0_center가 220Hz 근처로 나와야 한다 — tests/test_acoustics.py 참고)

지원 형식: PCM 16/24/32-bit WAV, 모노/스테레오. 리샘플링은 하지 않으므로
같은 샘플레이트로 맞춰 넣는 것을 권장한다(다르면 경고만 한다).
"""
from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np

C3_HZ = 130.8128  # f0 축의 기준(반음 0)
FRAME_MS = 40.0
HOP_MS = 20.0
VOICING_THRESHOLD = 0.30   # 정규화 자기상관 최대값이 이보다 낮으면 무성음
F0_MIN, F0_MAX = 60.0, 900.0
VIBRATO_BAND = (3.0, 9.0)


# ------------------------------------------------------------------ 오디오 로드
def read_wav(path: str) -> tuple[np.ndarray, int]:
    """모노 float32(-1..1)과 샘플레이트를 돌려준다."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints / float(2 ** 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(2 ** 31)
    else:
        raise ValueError(f"지원하지 않는 샘플 폭: {width * 8}bit (16/24/32bit만)")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if data.size == 0:
        raise ValueError("빈 오디오")
    return data.astype(np.float64), rate


# --------------------------------------------------------------------- STFT
def stft_frames(signal: np.ndarray, rate: int, frame_ms: float = FRAME_MS,
                hop_ms: float = HOP_MS) -> tuple[np.ndarray, np.ndarray]:
    """(프레임 수, 주파수 빈) 크기 스펙트로그램과 프레임별 RMS."""
    frame = int(rate * frame_ms / 1000)
    hop = int(rate * hop_ms / 1000)
    if frame < 16:
        frame, hop = 16, 8
    if signal.size < frame:
        signal = np.pad(signal, (0, frame - signal.size))
    count = 1 + (signal.size - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = signal[index] * np.hanning(frame)[None, :]
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return spectrum, rms


def _freqs(rate: int, frame: int) -> np.ndarray:
    return np.fft.rfftfreq(frame, 1.0 / rate)


def f0_track(signal: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """자기상관 기반 f0 트랙과 프레임별 주기성(0~1). 무성 프레임은 NaN."""
    frame = int(rate * FRAME_MS / 1000)
    hop = int(rate * HOP_MS / 1000)
    if signal.size < frame:
        signal = np.pad(signal, (0, frame - signal.size))
    count = 1 + (signal.size - frame) // hop
    lag_min = max(2, int(rate / F0_MAX))
    lag_max = min(frame - 2, int(rate / F0_MIN))
    total = 1 << int(np.ceil(np.log2(2 * frame)))
    f0 = np.full(count, np.nan)
    periodicity = np.zeros(count)
    for i in range(count):
        start = i * hop
        chunk = signal[start:start + frame]
        if chunk.size < frame:
            chunk = np.pad(chunk, (0, frame - chunk.size))
        chunk = chunk - chunk.mean()
        energy = float(np.sqrt((chunk ** 2).mean()))
        if energy < 1e-4:
            continue
        spec = np.fft.rfft(chunk, n=total)
        ac = np.fft.irfft(spec * np.conj(spec))[:frame]
        ac = ac / (ac[0] + 1e-12)
        window = ac[lag_min:lag_max + 1]
        if window.size == 0:
            continue
        peak = int(np.argmax(window))
        confidence = float(window[peak])
        periodicity[i] = max(0.0, min(1.0, confidence))
        if confidence < VOICING_THRESHOLD:
            continue
        # 포물선 보간으로 서브샘플 정밀도
        lag = lag_min + peak
        if 0 < lag < ac.size - 1:
            a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
            denom = a - 2 * b + c
            offset = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c) / denom
            lag = lag + float(np.clip(offset, -0.5, 0.5))
        f0[i] = rate / lag if lag > 0 else np.nan
    return f0, periodicity


def _hz_to_semitones(hz: float) -> float:
    return 12.0 * float(np.log2(max(hz, 1e-6) / C3_HZ))


def _band_db(spectrum: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -120.0
    power = (spectrum[:, mask] ** 2).mean()
    return float(10.0 * np.log10(max(power, 1e-12)))


def vibrato_rate(f0: np.ndarray, times: np.ndarray) -> float:
    """f0 트랙의 저주파 변조 성분의 지배 주파수(Hz). 부족하면 NaN."""
    voiced = np.isfinite(f0)
    if voiced.sum() < 16:
        return float("nan")
    values = f0[voiced]
    stamps = times[voiced]
    semis = 12.0 * np.log2(values / C3_HZ)
    # 선형 추세(글라이드) 제거
    coeffs = np.polyfit(stamps - stamps[0], semis, 1)
    detrended = semis - np.polyval(coeffs, stamps - stamps[0])
    if detrended.size < 8:
        return float("nan")
    # 균일 리샘플(1/FRAME_HOP)
    grid = np.linspace(stamps[0], stamps[-1], detrended.size)
    interp = np.interp(grid, stamps, detrended)
    interp = interp - interp.mean()
    if np.allclose(interp, 0):
        return float("nan")
    window = np.hanning(interp.size)
    spec = np.abs(np.fft.rfft(interp * window))
    freqs = np.fft.rfftfreq(interp.size, d=float(np.mean(np.diff(grid))))
    mask = (freqs >= VIBRATO_BAND[0]) & (freqs <= VIBRATO_BAND[1])
    if not mask.any() or spec[mask].max() <= 0:
        return float("nan")
    return float(freqs[mask][int(np.argmax(spec[mask]))])


def describe_audio(signal: np.ndarray, rate: int) -> dict[str, float]:
    """9개 특징 축을 실제 신호에서 측정한다."""
    spectrum, rms = stft_frames(signal, rate)
    frame = int(rate * FRAME_MS / 1000) or 16
    freqs = _freqs(rate, frame)
    f0, periodicity = f0_track(signal, rate)
    times = np.arange(f0.size) * (HOP_MS / 1000.0)

    voiced = np.isfinite(f0)
    if voiced.sum() >= 5:
        f0_center = float(np.median(f0[voiced]))
        f0_lo, f0_hi = np.percentile(f0[voiced], [5, 95])
        f0_range = _hz_to_semitones(f0_hi) - _hz_to_semitones(f0_lo)
    else:
        f0_center, f0_range = float("nan"), float("nan")

    # 밝기: 스펙트럼 중심(Hz)을 200Hz~6kHz 로그 구간에 사상(0=어두움, 1=밝음).
    # 나이퀴스트로 나누면 대부분 0.7~0.9에 몰려 변별력이 없다(실측으로 확인).
    mag = spectrum.sum(axis=0) + 1e-12
    centroid = float((freqs * mag).sum() / mag.sum())
    brightness = float(np.clip(np.log(centroid / 200.0) / np.log(6000.0 / 200.0), 0.0, 1.0))

    # 거칠기: 스펙트럼 평탄도 평균(0=토널, 1=노이즈)
    flatness = np.exp(np.mean(np.log(spectrum + 1e-12), axis=1)) / \
        (np.mean(spectrum, axis=1) + 1e-12)
    roughness = float(np.clip(np.mean(flatness), 0.0, 1.0))

    # 어택: RMS 상승 기울기(양수만) 중앙값을 dB/프레임 → 0~1 정규화
    db = 20 * np.log10(rms + 1e-9)
    rise = np.diff(db)
    positive = rise[rise > 0.5]
    attack = float(np.clip(np.median(positive) / 12.0, 0.0, 1.0)) if positive.size else 0.0

    # 비음: 2.5~4kHz 에너지 대비 0.5~1kHz 에너지(dB)
    nasality = _band_db(spectrum, freqs, 2500, 4000) - _band_db(spectrum, freqs, 500, 1000)

    # 숨소리: 조용한 프레임(하위 40%)에서 4~8kHz 상대 에너지
    threshold = np.percentile(rms, 40)
    quiet = rms <= threshold
    high = _band_db(spectrum[quiet], freqs, 4000, 8000) if quiet.any() else -120.0
    low = _band_db(spectrum[quiet], freqs, 0, 2000) if quiet.any() else -120.0
    breathiness = float(np.clip((high - low + 60.0) / 60.0, 0.0, 1.0))

    # 표현력: 음량 변동 폭(dB, 5~95퍼센타일)
    expressiveness = float(np.percentile(db, 95) - np.percentile(db, 5))

    vibrato = vibrato_rate(f0, times)
    if not np.isfinite(vibrato):
        vibrato = 0.0

    return {
        "f0_center_st": 0.0 if not np.isfinite(f0_center) else _hz_to_semitones(f0_center),
        "f0_range_st": 0.0 if not np.isfinite(f0_range) else float(f0_range),
        "f0_vibrato_hz": float(vibrato),
        "brightness": brightness,
        "roughness": roughness,
        "attack": attack,
        "nasality": float(nasality),
        "breathiness": breathiness,
        "expressiveness": expressiveness,
        "voiced_ratio": float(np.mean(periodicity >= VOICING_THRESHOLD)),
        "seconds": float(signal.size / rate),
        "sample_rate": int(rate),
    }


@dataclass
class AudioDescriptor:
    path: str
    features: dict[str, float]

    def to_row(self) -> dict:
        return {"path": self.path, **{k: round(v, 6) for k, v in self.features.items()}}


def describe_file(path: str) -> AudioDescriptor:
    signal, rate = read_wav(path)
    return AudioDescriptor(path=str(path), features=describe_audio(signal, rate))
