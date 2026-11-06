"""음색 특징 벡터 — 클러스터링과 병합 매칭에 쓰는 저차원 기술자(descriptor).

설계 원칙
1. **정직한 입력.** 여기 들어가는 값은 (a) 실제 오디오에서 뽑은 음향 특징 또는
   (b) 학습된 RVC 모델의 가중치 통계에서 뽑은 값이어야 한다. 손으로 찍은 숫자를
   "측정값"처럼 넣지 않는다. `voice_fusion.descriptors`가 실제 ckpt를 읽어 이 값을 만든다.
2. **해석 가능.** 각 축은 사람이 읽을 수 있는 의미(밝기, 거칠기, 비음, 진폭 변동…)를
   가지므로, 병합 후 "왜 이런 음색이 나왔는지"를 설명할 수 있다.
3. **스케일 불변.** 모든 축은 균등 스케일에 맞춰 로버스트 표준화한다(`zscore`).
   병합 비중(0~1)과 같은 단위로 다루기 위해 클러스터링 전에 표준화한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 축 이름과 사람이 읽는 설명. 순서가 곧 벡터 순서다.
FEATURE_SPEC: tuple[tuple[str, str], ...] = (
    ("f0_center_st", "발성 중심 음높이(반음, C3=0)"),
    ("f0_range_st", "발성 음역 폭(반음, 5~95퍼센타일)"),
    ("f0_vibrato_hz", "비브라토/흔들림 속도(Hz)"),
    ("brightness", "스펙트럼 중심(밝기, 0~1 정규화)"),
    ("roughness", "스펙트럼 평탄도(거칠기/속삭임 성분)"),
    ("attack", "어택/자음 과도성(에너지 상승 기울기)"),
    ("nasality", "비음 성분(500Hz~1kHz 대비 2.5~4kHz 비)" ),
    ("breathiness", "숨소리 비율(무성 구간 고주파 에너지)"),
    ("expressiveness", "진폭 변동 폭(dB, 5~95퍼센타일)"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in FEATURE_SPEC)
FEATURE_LABELS: dict[str, str] = dict(FEATURE_SPEC)

# 각 축의 "정상" 범위. 표준화 후 이상치를 자르는 데 쓴다(합성/파손 값 방지).
CLIP_Z = 3.0


def as_vector(mapping: dict[str, float]) -> np.ndarray:
    """이름→값 dict를 정해진 축 순서의 벡터로 바꾼다. 누락 축은 오류."""
    missing = [name for name in FEATURE_NAMES if name not in mapping]
    if missing:
        raise KeyError(f"특징 축 누락: {missing}")
    vector = np.array([float(mapping[name]) for name in FEATURE_NAMES], dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError("특징 벡터에 NaN/Inf가 있습니다")
    return vector


def as_mapping(vector: np.ndarray) -> dict[str, float]:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.shape[0] != len(FEATURE_NAMES):
        raise ValueError(f"벡터 길이 {vector.shape[0]} != {len(FEATURE_NAMES)}")
    return {name: float(value) for name, value in zip(FEATURE_NAMES, vector)}


@dataclass(frozen=True)
class Standardizer:
    """로버스트 표준화(중앙값/IQR). 병합 레시피를 파일로 옮겨도 같은 좌표계를 쓴다."""

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Standardizer":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("matrix must be 2-D and non-empty")
        center = np.median(matrix, axis=0)
        q75, q25 = np.percentile(matrix, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349  # IQR → 표준편차 근사
        # 한 축이 거의 상수면 표준편차로 대체하고, 그것도 0이면 1로 둔다.
        fallback = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.ones_like(center)
        scale = np.where(scale > 1e-9, scale, np.where(fallback > 1e-9, fallback, 1.0))
        return cls(center=center, scale=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        return np.clip((matrix - self.center) / self.scale, -CLIP_Z, CLIP_Z)

    def to_dict(self) -> dict:
        return {
            "center": self.center.round(8).tolist(),
            "scale": self.scale.round(8).tolist(),
            "features": list(FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Standardizer":
        if list(payload.get("features", [])) != list(FEATURE_NAMES):
            raise ValueError("표준화 파일의 특징 축 순서가 코드와 다릅니다")
        return cls(center=np.array(payload["center"], dtype=float),
                   scale=np.array(payload["scale"], dtype=float))


def centroid(matrix: np.ndarray) -> np.ndarray:
    """데이터셋 중심(축별 평균)."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    return matrix.mean(axis=0)


def centered_cosine(a: np.ndarray, b: np.ndarray, center: np.ndarray) -> float:
    """중심을 뺀 코사인 = Pearson 상관. 표준화 공간에서 두 음성의 '모양' 유사도.

    중심을 빼지 않은 원시 코사인은 9축 합이 공통으로 크면 모두 1에 가까워져
    아무 것도 구분하지 못한다(실제로 겪은 함정). 비교 대상이 2개뿐일 때는
    중심을 빼면 상관이 ±1로 고정되므로, 호출부에서 `center=None`(원시 코사인)을
    함께 보는 것이 안전하다.
    """
    return cosine_similarity(np.asarray(a) - center, np.asarray(b) - center)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)
