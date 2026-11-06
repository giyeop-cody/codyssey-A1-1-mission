"""체크포인트 → 특징 벡터(proxy). 실제 음향 측정이 불가능할 때만 쓰는 보조 경로.

정직하게 말하면, RVC 가중치 통계(텐서별 평균·표준편차)는 **음색의 직접 측정값이 아니다.**
그래서 여기서 만든 값은
- 매니페스트에 `feature_source="ckpt_stats(proxy)"`로 표시되고,
- `selling.describe_product`에 `allow_proxy=False`를 주면 판매 문구 생성이 **거부**된다.

proxy는 (1) 모델 간 중복 탐지, (2) 대략적인 클러스터링의 초기값으로만 쓴다.
운영 경로는 `feature_source="audio_measured"` — 실제 렌더/원본 wav를
`acoustics.describe_file`로 측정한 값이다.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .features import FEATURE_NAMES
from .rvc import CheckpointInfo

# 축별 (base, scale) — proxy 스케일을 사람이 읽을 수 있는 범위로 옮긴다.
_PROXY_SCALE = {
    "f0_center_st": (4.0, 2.0),
    "f0_range_st": (10.0, 3.0),
    "f0_vibrato_hz": (5.0, 1.5),
    "brightness": (0.40, 0.08),
    "roughness": (0.16, 0.04),
    "attack": (0.35, 0.08),
    "nasality": (-10.0, 3.0),
    "breathiness": (0.40, 0.10),
    "expressiveness": (18.0, 4.0),
}


def describe_checkpoint(info: CheckpointInfo) -> dict[str, float]:
    """가중치 통계 → 9축 proxy. 결정론적(같은 ckpt면 같은 벡터)."""
    vector = np.asarray(info.signature_vector, dtype=float)
    if vector.size == 0:
        raise ValueError(f"{info.path}: 텐서 통계가 비었습니다")
    # 지문 벡터를 고정 시드로 접어 축별 잡음을 만든다(모델마다 안정적).
    digest = hashlib.sha256((info.signature + "|axes").encode()).digest()
    seed_vec = np.frombuffer(digest, dtype=np.uint8)[:len(FEATURE_NAMES)]
    normal = (seed_vec.astype(float) - 127.5) / 74.0  # 대략 -1..1
    # 전역 통계도 반영: 평균 절댓값이 큰 모델은 밝기·거칠기가 조금 더 크게.
    magnitude = float(np.mean(np.abs(vector)))
    tilt = float(np.clip(magnitude * 50.0, -1.0, 1.0))

    features = {}
    for i, name in enumerate(FEATURE_NAMES):
        base, scale = _PROXY_SCALE[name]
        jitter = float(normal[i]) * scale
        if name in ("brightness", "roughness", "attack"):
            jitter += 0.25 * tilt * scale
        features[name] = base + jitter
    return features


def is_measurable(feature_source: str) -> bool:
    """판매 문구에 써도 되는 출처인지(실측만 허용)."""
    return feature_source.startswith("audio_measured")


def proxy_warning(feature_source: str) -> str | None:
    if "proxy" in feature_source:
        return ("특징값이 체크포인트 통계 기반 proxy입니다. 판매 문구에 쓰려면 "
                "실제 오디오 측정(audio_measured)으로 교체하세요.")
    return None
