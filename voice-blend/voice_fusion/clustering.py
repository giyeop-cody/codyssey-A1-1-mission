"""비슷한 특징끼리 묶기 — 결정론적 계층 응집 클러스터링 + 실루엣으로 k 선택.

요구사항 "비슷한 특징끼리 묶고"를 재현 가능하게 만족시키기 위해
- 거리: 표준화 공간의 유클리드(평균 연결법)
- k 선택: 실루엣 계수 최대, 동점이면 작은 k (설명이 단순한 쪽)
- 난수 없음: 같은 입력이면 항상 같은 군집

구현은 "병합 기록(dendrogram)"을 남기고 k마다 그 기록을 잘라 라벨을 만든다.
단계별 상태를 변형하는 방식보다 버그가 적고, k별 결과를 몇 번이든 다시 만들 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray            # 각 항목의 군집 번호(0..k-1)
    k: int
    silhouette: float | None      # k>=2 일 때만 계산
    score_table: list[dict]       # k별 점수(리포트용)


def pairwise_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def agglomerative_merges(distances: np.ndarray) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """평균 연결 응집 병합 기록. n-1개의 (군집A 구성원, 군집B 구성원) 튜플."""
    dist = np.asarray(distances, dtype=float)
    n = dist.shape[0]
    members: dict[int, tuple[int, ...]] = {i: (i,) for i in range(n)}
    active = list(range(n))
    next_id = n
    merges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def average(a: int, b: int) -> float:
        rows = [dist[x, y] for x in members[a] for y in members[b]]
        return float(np.mean(rows))

    while len(active) > 1:
        best, pair = np.inf, (active[0], active[1])
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                value = average(a, b)
                if value < best - 1e-12:  # 결정론적: 동률이면 먼저 나온 쌍
                    best, pair = value, (a, b)
        a, b = pair
        merges.append((members[a], members[b]))
        members[next_id] = members.pop(a) + members.pop(b)
        active = [x for x in active if x not in (a, b)]
        active.append(next_id)
        next_id += 1
    return merges


def labels_at(merges: list[tuple[tuple[int, ...], tuple[int, ...]]], n: int, k: int) -> np.ndarray:
    """병합 기록을 앞에서 (n-k)개만 적용한 라벨."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group_a, group_b in merges[: max(0, n - k)]:
        ra, rb = find(group_a[0]), find(group_b[0])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        root = find(i)
        labels[i] = roots.setdefault(root, len(roots))
    return labels


def _silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    n = matrix.shape[0]
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= n:
        return -1.0
    dist = pairwise_distance(matrix)
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == other].mean() for other in unique if other != labels[i])
        denom = max(a, b)
        scores.append(0.0 if denom < 1e-12 else (b - a) / denom)
    return float(np.mean(scores))


def cluster_voices(matrix: np.ndarray, max_k: int = 8, min_size: int = 1) -> ClusterResult:
    """표준화된 특징 행렬을 군집화한다. 행 순서는 입력과 같다."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("빈 입력")
    if n == 1:
        return ClusterResult(labels=np.array([0]), k=1, silhouette=None,
                             score_table=[{"k": 1, "silhouette": None, "min_size": 1,
                                           "rejected": None}])
    max_k = int(min(max(max_k, 1), n))

    merges = agglomerative_merges(pairwise_distance(matrix))
    table, best = [], None
    for k in range(1, max_k + 1):
        labels = labels_at(merges, n, k)
        sizes = np.bincount(labels)
        if min_size > 1 and sizes.min() < min_size:
            table.append({"k": k, "silhouette": None, "min_size": int(sizes.min()),
                          "rejected": f"최소 군집 크기 {min_size} 미만"})
            continue
        score = None if k == 1 else _silhouette(matrix, labels)
        table.append({"k": k, "silhouette": None if score is None else round(score, 4),
                      "min_size": int(sizes.min()), "rejected": None})
        if score is not None:
            key = (score, -k)
            if best is None or key > best[0]:
                best = (key, k, score)

    k, score = (1, None) if best is None else (best[1], best[2])
    return ClusterResult(labels=labels_at(merges, n, k), k=k, silhouette=score,
                         score_table=table)
