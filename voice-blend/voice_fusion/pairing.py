"""대사↔노래 짝 정렬 — "대사 음성 + 노래 음성"을 병합 재료 쌍으로 묶는다.

왜 필요한가
-----------
RVC는 변환기다. 음색 모델만 있으면 무엇을 말하든 그 목소리로 바뀐다. 하지만
**노래로 학습한 모델**은 발성·호흡·비브라토가 노래 쪽에 맞춰져 있고, **대사로 학습한
모델**은 자음 과도성과 말의 리듬에 맞춰져 있다. 둘을 섞으면 "노래하는 목소리"가
되지만, 아무 조합이나 섞으면 음역이 튀고 피치가 깨진다.

그래서 다음 우선순위로 짝을 만든다.
1. 음역(f0_center·f0_range)이 가까운 것 — 병합 시 피치 분포가 갈라지지 않게
2. 밝기/거칠기가 가까운 것 — 음색이 "두 사람"으로 들리지 않게
3. 그래도 남으면 거리 순(최선 노력)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifest import Manifest

PAIR_AXES = ("f0_center_st", "f0_range_st", "brightness", "roughness", "breathiness")
# 음역 축은 병합에서 가장 치명적이므로 가중치를 준다.
AXIS_WEIGHTS = {"f0_center_st": 1.6, "f0_range_st": 1.4, "brightness": 1.0,
                "roughness": 1.0, "breathiness": 0.8}


@dataclass(frozen=True)
class Pair:
    speech: str
    song: str
    distance: float
    shared_axes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"speech": self.speech, "song": self.song,
                "distance": round(self.distance, 4), "close_axes": list(self.shared_axes)}


def pair_speech_song(manifest: Manifest, z_matrix: np.ndarray,
                     max_pairs: int | None = None) -> list[Pair]:
    """1:1 최선 짝(탐욕적, 결정론적). 표준화 공간 거리 사용."""
    idx = {e.voice_id: i for i, e in enumerate(manifest.entries)}
    speeches = [e.voice_id for e in manifest.entries if e.kind == "speech"]
    songs = [e.voice_id for e in manifest.entries if e.kind == "song"]
    if not speeches or not songs:
        return []
    axes = [i for i, name in enumerate(
        ("f0_center_st", "f0_range_st", "brightness", "roughness", "breathiness"))]
    weight = np.array([AXIS_WEIGHTS[name] for name in PAIR_AXES])

    candidates = []
    for s in speeches:
        for g in songs:
            delta = (z_matrix[idx[s], axes] - z_matrix[idx[g], axes]) * weight
            dist = float(np.sqrt((delta ** 2).sum()))
            close = tuple(name for name, value in zip(PAIR_AXES, delta / weight)
                          if abs(value) <= 0.5)
            candidates.append((dist, s, g, close))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_s, used_g, pairs = set(), set(), []
    for dist, s, g, close in candidates:
        if s in used_s or g in used_g:
            continue
        used_s.add(s)
        used_g.add(g)
        pairs.append(Pair(speech=s, song=g, distance=dist, shared_axes=close))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def pool_from_pairs(pairs: list[Pair], per_pool: int = 5) -> list[list[str]]:
    """각 풀은 "대사1 + 노래1"로 시작해 같은 음역대 음성으로 채운다.

    풀 안에 speech/song이 모두 있어야 "노래 4~5곡 + 대사 5~6개"를 한 모델로 뽑을 수 있다.
    """
    pools = []
    for pair in pairs:
        base = [pair.speech, pair.song]
        pools.append(base[:per_pool])
    return pools
