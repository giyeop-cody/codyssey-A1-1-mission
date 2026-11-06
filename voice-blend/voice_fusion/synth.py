"""데모/테스트 전용 합성 음성 생성기.

⚠ **여기는 실제 가수가 아니다.** 파이프라인 배선(특징 추출 → 군집 → 레시피 →
게이트 → 리포트)을 GPU 없이 검증하기 위한 신호 합성기다. 이 출력물로 만든
설명글에는 반드시 `synthetic` 표기가 붙고, 판매 대상이 아니다.

각 "화자"는 다음 축을 다르게 만들어 실제 파이프라인에서 구분되게 한다.
- f0 중심/음역, 비브라토 깊이·속도
- 배음 기울기(밝기), 스펙트럼 평탄도(거칠기)
- 음절 엔벨로프(어택), 저역/고역 비(비음·숨소리)
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

SR = 22050


def _envelope(t: np.ndarray, syllable_hz: float, attack: float) -> np.ndarray:
    phase = (t * syllable_hz) % 1.0
    rise = np.clip(phase / max(attack, 1e-3), 0.0, 1.0)
    fall = np.clip((1.0 - phase) / 0.45, 0.0, 1.0)
    return rise * fall


def synthesize(path: Path, *, f0_hz: float, semitone_range: float,
               vibrato_hz: float, vibrato_st: float, tilt: float,
               noise: float, syllable_hz: float, attack: float,
               seconds: float = 4.0, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    # 음역 폭: 저속 삼각파로 음높이를 흔든다(5~95퍼센타일이 대략 semitone_range가 되게)
    glide = (np.sin(2 * np.pi * 0.21 * t + rng.uniform(0, 6.28)) * semitone_range / 2.4)
    vib = vibrato_st * np.sin(2 * np.pi * vibrato_hz * t)
    f0 = f0_hz * 2 ** ((glide + vib) / 12.0)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    signal = np.zeros(n)
    for harmonic in range(1, 26):
        freq = f0 * harmonic
        if freq.mean() > SR / 2.2:
            break
        amp = harmonic ** (-tilt)
        signal += amp * np.sin(harmonic * phase + rng.uniform(0, 6.28))
    signal /= np.max(np.abs(signal)) + 1e-9
    signal *= _envelope(t, syllable_hz, attack)
    signal += noise * rng.standard_normal(n) * 0.05
    signal *= 0.7 / (np.max(np.abs(signal)) + 1e-9)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((signal * 32767).astype("<i2").tobytes())
    return path


# name → 파라미터(전부 합성). speech 4개 + song 4개.
SYNTH_VOICES: dict[str, dict] = {
    "syn-low-speech": dict(f0_hz=118, semitone_range=8, vibrato_hz=4.5, vibrato_st=0.15,
                           tilt=1.05, noise=0.35, syllable_hz=4.2, attack=0.06, seed=11),
    "syn-mid-speech": dict(f0_hz=176, semitone_range=6, vibrato_hz=5.2, vibrato_st=0.12,
                           tilt=0.95, noise=0.5, syllable_hz=4.8, attack=0.04, seed=12),
    "syn-high-speech": dict(f0_hz=252, semitone_range=7, vibrato_hz=5.8, vibrato_st=0.14,
                            tilt=0.8, noise=0.55, syllable_hz=5.1, attack=0.035, seed=13),
    "syn-airy-speech": dict(f0_hz=214, semitone_range=9, vibrato_hz=4.0, vibrato_st=0.10,
                            tilt=1.25, noise=0.95, syllable_hz=3.9, attack=0.10, seed=14),
    "syn-low-song": dict(f0_hz=132, semitone_range=14, vibrato_hz=5.6, vibrato_st=0.55,
                         tilt=1.0, noise=0.30, syllable_hz=2.9, attack=0.05, seed=21),
    "syn-mid-song": dict(f0_hz=198, semitone_range=16, vibrato_hz=6.1, vibrato_st=0.6,
                         tilt=0.9, noise=0.40, syllable_hz=3.2, attack=0.045, seed=22),
    "syn-high-song": dict(f0_hz=286, semitone_range=18, vibrato_hz=6.5, vibrato_st=0.65,
                          tilt=0.75, noise=0.45, syllable_hz=3.4, attack=0.04, seed=23),
    "syn-whisper-song": dict(f0_hz=230, semitone_range=12, vibrato_hz=4.2, vibrato_st=0.25,
                             tilt=1.35, noise=1.1, syllable_hz=2.6, attack=0.12, seed=24),
}

KIND = {name: ("speech" if name.endswith("speech") else "song") for name in SYNTH_VOICES}


def build_demo_audio(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    made = {}
    for name, params in SYNTH_VOICES.items():
        made[name] = synthesize(directory / f"{name}.wav", **params)
    return made
