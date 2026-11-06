"""렌더 시뮬레이션 — 병합 가중치로 원본들을 시간 영역에서 섞어 본다.

⚠ 실제 RVC 병합은 **모델 가중치**를 섞지만, 여기서는 그 효과를 검증하기 위해
**오디오 신호**를 같은 비중으로 섞는다. 두 가지를 구분해야 한다.

- 검증하려는 것: "이 비중 조합이 재생 가능한 신호를 만드는가" — 클리핑, 무음,
  과도한 음량차, 위상 상쇄로 인한 f0 붕괴.
- 검증하지 못하는 것: 병합된 모델의 실제 음색. 그것은 GPU에서 렌더해야만 알 수 있다.

그래서 이 모듈의 출력은 `REPORT.md`에서 "시뮬레이션"으로 명시되고 판매 문구에는
쓰이지 않는다.

주의: 아래 믹스는 **음량만 맞춘 위상 그대로의 합**이므로, 서로 다른 위상의 음성을 겹치면
진폭이 서서히 변하는 간섭(비트)이 생긴다. 이 간섭은 RVC 렌더에서 나타나는 음색 뭉개짐과는
다른 현상이며, 감사 지표(모음 시작률 등)를 과도하게 낮출 수 있다. 그래서 감사 결과는
"믹스가 완전히 무너지는지"를 보는 **하한선**으로만 읽는다.
"""
from __future__ import annotations

import numpy as np


def simulate_blend(members: dict[str, tuple[np.ndarray, int]],
                   weights: dict[str, float],
                   *, normalize_loudness: bool = True,
                   align: bool = True) -> np.ndarray:
    """가중 믹스. 각 입력은 같은 샘플레이트여야 한다(아니면 오류)."""
    if not members:
        raise ValueError("빈 입력")
    rates = {rate for _, rate in members.values()}
    if len(rates) != 1:
        raise ValueError(f"샘플레이트가 다릅니다: {sorted(rates)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"가중치 합이 1이 아닙니다: {total}")

    signals = {name: np.asarray(sig, dtype=float) for name, (sig, _) in members.items()}
    length = max(sig.size for sig in signals.values())
    out = np.zeros(length)
    for name, weight in weights.items():
        if name not in signals:
            raise KeyError(f"가중치에만 있는 음성: {name}")
        sig = signals[name]
        if sig.size < length:  # 짧은 신호는 반복 패딩(무음으로 두면 음량이 꺼진다)
            repeats = int(np.ceil(length / sig.size))
            sig = np.tile(sig, repeats)[:length]
        if normalize_loudness:
            rms = float(np.sqrt((sig ** 2).mean()) + 1e-12)
            sig = sig / rms * 0.1  # 대략 -20dBFS 로 맞춘 뒤 비중을 적용
        out += weight * sig

    if align:
        out = _envelope_align(out, signals, weights, length)

    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 0.89:  # 클리핑 방지: -1dBFS 목표
        out *= 0.89 / peak
    return out


def _envelope_align(mix: np.ndarray, signals: dict[str, np.ndarray],
                    weights: dict[str, float], length: int,
                    hop_ms: float = 10.0) -> np.ndarray:
    """위상 간섭으로 음량 포락선이 무너지지 않게 원본 포락선 쪽으로 되돌린다.

    RVC도 위상이 다른 두 사람의 파형을 그대로 더하지 않는다(모델 가중치를 섞는다).
    이 보정은 "비중대로 섞었을 때 음량 구조가 유지되는가"만 보기 위한 최소 장치다.
    샘플레이트는 호출부에서 이미 하나로 맞춰져 있다고 가정한다(22050Hz 기준 hop).
    """
    hop = 220  # 10ms @ 22050Hz
    frames = length // hop
    if frames < 3:
        return mix
    blocks = mix[:frames * hop].reshape(frames, hop)
    measured = np.sqrt((blocks ** 2).mean(axis=1) + 1e-12)

    target = np.zeros(frames)
    for name, weight in weights.items():
        sig = signals[name]
        if sig.size != length:
            repeats = int(np.ceil(length / sig.size))
            sig = np.tile(sig, repeats)[:length]
        sig = sig / (np.sqrt((sig ** 2).mean()) + 1e-12) * 0.1
        ref = sig[:frames * hop].reshape(frames, hop)
        target += weight * np.sqrt((ref ** 2).mean(axis=1) + 1e-12)

    gain = np.clip(target / measured, 0.25, 4.0)
    # 이웃 3프레임 평균으로 게인을 부드럽게
    kernel = np.ones(3) / 3
    gain = np.convolve(gain, kernel, mode="same")
    gain = np.clip(gain, 0.25, 4.0)
    aligned = (blocks * gain[:, None]).reshape(-1)
    out = mix.copy()
    out[:frames * hop] = aligned
    return out
