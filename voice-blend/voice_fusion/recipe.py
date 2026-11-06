"""병합 레시피 뼈대 — 가중치 트리, 풀 선택, 랜덤 경로, 게이트, 설명글.

핵심 모델
---------
RVC의 ckpt merge는 텐서별 선형 결합이다(Weighted Sum). 두 모델을 w:(1-w)로 섞으면
결과 가중치(c1, c2)는 합이 1인 확률분포가 된다. 세 개 이상을 "3개를 1:3:2로 섞고,
그 결과와 남은 1개를 다시 섞는" 식으로 이으면 **가중치 트리**가 되고, 트리 루트의
최종 가중치는 재귀식으로 정확히 계산된다:

    g_leaf(i)   = 1
    g_path(원소) = 부모경로의 누적 → 각 단계에서 (1-w)를 곱해 내려온다

이 모듈은 그 계산을 **실제 타깃 모델 없이** 수행해, 어떤 음색이 어떻게 섞였는지
미리 예측·검증·설명할 수 있게 한다. (`torch` 불필요, numpy만 사용)
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_LABELS, FEATURE_NAMES


# --------------------------------------------------------------------------- RNG
class DeterministicRNG:
    """SHA-256 기반 난수 — 플랫폼/파이썬 버전과 무관하게 같은 시드=같은 레시피."""

    def __init__(self, seed: str):
        self.seed = str(seed)
        self._counter = 0
        self._buffer = b""

    def _bytes(self, n: int) -> bytes:
        while len(self._buffer) < n:
            block = hashlib.sha256(f"{self.seed}|{self._counter}".encode()).digest()
            self._counter += 1
            self._buffer += block
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def random(self) -> float:
        return int.from_bytes(self._bytes(8), "big") / 2**64

    def randint(self, low: int, high: int) -> int:
        """[low, high] 양끝 포함."""
        if high < low:
            raise ValueError("high < low")
        return low + int(self.random() * (high - low + 1)) if high > low else low

    def choice(self, seq):
        if not seq:
            raise ValueError("빈 시퀀스에서 고를 수 없습니다")
        return seq[self.randint(0, len(seq) - 1)]

    def weights(self, count: int) -> list[int]:
        """1..3 정수 비중을 만들되 최소 2종류가 되게(완전 동률 병합 방지)."""
        if count < 2:
            raise ValueError("가중치는 2개 이상이어야 합니다")
        for _ in range(50):
            draw = [self.randint(1, 3) for _ in range(count)]
            if len(set(draw)) > 1:
                return draw
        draw[0], draw[1] = 1, 3
        return draw


def normalize(weights) -> list[float]:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or arr.min() <= 0:
        raise ValueError("가중치는 양수여야 합니다")
    return (arr / arr.sum()).round(10).tolist()


# ------------------------------------------------------------------ 병합 트리
@dataclass
class MergeNode:
    """가중치 트리의 한 노드. leaf면 voice_id, 아니면 children+weights."""

    voice_id: str | None = None
    children: list["MergeNode"] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    label: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.voice_id is not None

    def leaves(self) -> list[str]:
        if self.is_leaf:
            return [self.voice_id]
        out: list[str] = []
        for child in self.children:
            out.extend(child.leaves())
        return out

    def to_dict(self) -> dict:
        if self.is_leaf:
            return {"leaf": self.voice_id}
        return {
            "label": self.label,
            "weights": list(self.weights),
            "children": [child.to_dict() for child in self.children],
        }


def final_weights(node: MergeNode, depth: int = 0, carry: float = 1.0) -> dict[str, float]:
    """루트부터 (1-w)를 곱해 내려가며 각 원본 음성의 트리 루트 가중치를 계산한다."""
    if node.is_leaf:
        return {node.voice_id: round(carry, 10)}
    if len(node.weights) != len(node.children):
        raise ValueError("노드의 children과 weights 길이가 다릅니다")
    out: dict[str, float] = {}
    for child, weight in zip(node.children, node.weights):
        sub = final_weights(child, depth + 1, carry * float(weight))
        for key, value in sub.items():
            out[key] = round(out.get(key, 0.0) + value, 10)
    total = sum(out.values())
    # 하위 노드는 carry만큼만 합산되므로, 기대 합은 carry다(루트에서 1.0).
    if abs(total - carry) > 1e-6:
        raise AssertionError(f"가중치 합이 carry({carry})와 다릅니다: {total}")
    return out


def blend_features(vetors: dict[str, np.ndarray], node: MergeNode) -> np.ndarray:
    """선형 결합 가정에서 병합 음색의 특징 벡터를 예측한다(1차 근사)."""
    if node.is_leaf:
        return vetors[node.voice_id]
    acc = np.zeros_like(next(iter(vetors.values())))
    for child, weight in zip(node.children, node.weights):
        acc = acc + float(weight) * blend_features(vetors, child)
    return acc


# --------------------------------------------------------------- 랜덤 레시피
def sample_recipe(
    rng: DeterministicRNG,
    pool_ids: list[str],
    *,
    pick_min: int = 3,
    pick_max: int = 5,
    first_group=(2, 3),
    max_depth: int = 5,
) -> tuple[list[str], MergeNode, list[dict]]:
    """요구사항의 예시를 그대로 구현한다.

    ex) 5개 뽑음 → 처음 3개를 1:3:2로 섞고 → 그 결과와 남은 1개 → 그 후 또 1개.
    즉 **매 단계마다 아직 안 쓴 음성 중 (1개 또는 남은 전부)를 골라 누적 결과와 병합**하고,
    그때의 비중도 매번 새로 뽑는다.

    반환하는 각 단계는 `items`(무엇을 어떤 순서로 놓았는지)와 `weights`(같은 순서의
    비중, `누적본`은 `"carry"`로 표기)를 함께 담아, 사람이 경로를 그대로 읽을 수 있다.
    """
    pool = list(dict.fromkeys(pool_ids))
    if len(pool) < 2:
        raise ValueError("병합에는 서로 다른 음성이 2개 이상 필요합니다")
    pick_max = min(max(pick_max, 2), len(pool))
    pick_min = min(max(pick_min, 2), pick_max)
    n_pick = rng.randint(pick_min, pick_max)

    order = list(pool)
    for i in range(len(order) - 1, 0, -1):  # 자체 Fisher–Yates(결정론적)
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]

    picked = order[:n_pick]  # 뽑힌 것만 병합 재료로 쓴다(나머지는 다음 병합본의 몫)

    gmin, gmax = first_group
    take = min(rng.randint(gmin, gmax), len(picked))
    first = picked[:take]
    first_weights = normalize(rng.weights(take))
    steps: list[dict] = [{
        "step": 1, "kind": "seed",
        "items": list(first),
        "weights": first_weights,
    }]
    current = MergeNode(children=[MergeNode(voice_id=v) for v in first],
                        weights=first_weights, label="seed")

    rest, step_no = picked[take:], 2
    while rest and step_no <= max_depth:
        candidates = rest
        if len(candidates) == 1 or step_no == max_depth:
            add_count = len(candidates)
        else:
            add_count = len(candidates) if rng.random() < 0.35 else 1
        adds = candidates[:add_count]
        weights = normalize(rng.weights(add_count + 1))  # 누적본 + 새 음성들
        carry_first = rng.random() < 0.5  # 누적본을 앞/뒤 어디에 둘지도 랜덤
        if carry_first:
            items = ["carry"] + list(adds)
            children = [current] + [MergeNode(voice_id=v) for v in adds]
        else:
            items = list(adds) + ["carry"]
            children = [MergeNode(voice_id=v) for v in adds] + [current]
        current = MergeNode(children=children, weights=weights, label=f"step{step_no}")
        steps.append({"step": step_no, "kind": "chain", "adds": list(adds),
                      "items": items, "weights": list(weights)})
        rest = [v for v in rest if v not in adds]
        step_no += 1
    return picked, current, steps


# --------------------------------------------------------------------- 게이트
@dataclass
class GateConfig:
    max_leaves: int = 6                 # 6개 넘으면 "대표 음색"이 뭉개지고 재현도 어려움
    max_share: float = 0.62             # 한 음성이 62%를 넘으면 사실상 그 음성의 재탕
    min_share: float = 0.03             # 3% 미만은 사실상 잡음(설명·재현 비용만 늘림)
    warn_share: float = 0.08            # 8% 미만이면 "주의" 표시(실패는 아님)
    max_pair_correlation: float = 0.995  # 중심 제거 코사인(=상관)이 이보다 크면 사실상 동일 음색
    max_pair_cosine: float = 0.9995      # 원시 코사인 동시 확인(중심 제거가 과하게 엄격할 때 완화)
    min_pair_distance: float = 0.25     # 표준화 공간 최소 거리(같은 모델 이중 사용 방지)


@dataclass
class GateReport:
    passed: bool
    checks: list[dict]
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reason": self.reason,
                "warnings": self.warnings, "checks": self.checks}


def check_recipe(
    node: MergeNode,
    features: dict[str, np.ndarray],
    config: GateConfig = GateConfig(),
    center: np.ndarray | None = None,
) -> GateReport:
    """병합 결과를 "팔 수 있는가" 기준으로 검사한다.

    - 구조: 중복 리프, 리프 수
    - 분포: 최대/최소 지분, 사실상 단일 음성 하나인지
    - 다양성: 서로 너무 비슷한 음성만 섞였는지(중심 제거 코사인)

    `center`는 전체 음성의 표준화 중심이다. 주지 않으면 `features`의 평균을 쓴다.
    """
    from .features import centered_cosine

    weights = final_weights(node)
    leaves = list(weights)
    checks: list[dict] = []
    warnings: list[str] = []
    # center=None → 원시 코사인만 사용(중심을 알 수 없는 소규모 비교용)

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("중복 없는 원본", len(leaves) == len(set(leaves)), f"{len(leaves)}개 원본 사용")
    add("원본 개수 상한", len(leaves) <= config.max_leaves,
        f"{len(leaves)} ≤ {config.max_leaves}")
    top = max(weights.values())
    add("최대 지분", top <= config.max_share, f"최대 {top:.3f} ≤ {config.max_share}")
    low = min(weights.values())
    add("최소 지분", low >= config.min_share, f"최소 {low:.3f} ≥ {config.min_share}")
    if config.min_share <= low < config.warn_share:
        warnings.append(f"최소 지분 {low:.1%} < {config.warn_share:.0%} — "
                        "들리는 차이가 작아 설명·재현 비용이 남을 수 있음")
    add("사실상 단일 음성 아님", top < 0.95, f"최대 {top:.3f} < 0.95")

    worst_corr, pair = float("-inf"), ("", "")
    near_duplicates: list[str] = []
    too_close: list[str] = []
    for a, b in itertools.combinations(leaves, 2):
        if a not in features or b not in features:
            continue
        za, zb = np.asarray(features[a], dtype=float), np.asarray(features[b], dtype=float)
        corr = centered_cosine(za, zb, center) if center is not None else float("nan")
        cos = float(np.dot(za, zb) / (np.linalg.norm(za) * np.linalg.norm(zb) + 1e-12))
        dist = float(np.linalg.norm(za - zb))
        if np.isfinite(corr) and corr > worst_corr:
            worst_corr, pair = corr, (a, b)
        # 두 신호가 동시에 "같다"고 말할 때만 중복으로 본다.
        if np.isfinite(corr) and corr > config.max_pair_correlation and cos > config.max_pair_cosine:
            near_duplicates.append(f"{a}~{b} (상관 {corr:.4f}, 코사인 {cos:.5f})")
        if dist < config.min_pair_distance:
            too_close.append(f"{a}~{b} (거리 {dist:.4f})")
    add("유사 음성쌍 없음", not near_duplicates,
        "통과" if not near_duplicates else "; ".join(near_duplicates))
    add("최소 거리", not too_close,
        f"기준 {config.min_pair_distance}" if not too_close else "; ".join(too_close))
    detail = (f"최대 상관 {worst_corr:.3f} ({pair[0]}~{pair[1]})"
              if np.isfinite(worst_corr) else "중심 미지정 → 원시 코사인만 확인")
    add("음색 다양성", not near_duplicates, detail)

    failed = [c for c in checks if not c["ok"]]
    return GateReport(passed=not failed, checks=checks,
                      reason="; ".join(f"{c['check']}: {c['detail']}" for c in failed),
                      warnings=warnings)


# ------------------------------------------------------------------ 설명 생성
def dominant(delta: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(-np.abs(delta))[:k]
    return [(FEATURE_NAMES[i], float(delta[i])) for i in order]


def describe_blend(
    weights: dict[str, float],
    z_features: dict[str, np.ndarray],
    blended_z: np.ndarray,
    voices_meta: dict[str, dict],
) -> dict:
    """가중 평균 대비 병합 결과의 편차로 "어떤 성질이 도드라졌는지"를 문장화한다.

    선형 결합만으로 예측한 값이므로, 실제 렌더 결과와의 차이는
    `voice_fusion.analysis`의 실측 비교가 담당한다.
    """
    expected = np.zeros_like(blended_z)
    for voice, w in weights.items():
        expected = expected + w * z_features[voice]
    delta = blended_z - expected
    top = dominant(delta, 3)

    sentences = []
    for name, value in top:
        if abs(value) < 0.15:
            continue
        direction = "강해짐" if value > 0 else "약해짐"
        sentences.append(f"{FEATURE_LABELS[name]}이(가) 가중평균보다 {direction} (Δz {value:+.2f})")

    top_voice, top_share = max(weights.items(), key=lambda kv: kv[1])
    meta = voices_meta.get(top_voice, {})
    lead = f"{meta.get('source', top_voice)} 계열이 {top_share:.0%}로 중심"
    if sentences:
        lead += ", " + ", ".join(sentences)
    else:
        lead += ", 각 축은 가중평균과 거의 동일(선형 근사)"
    return {
        "headline": lead,
        "top_deviations": [{"feature": n, "label": FEATURE_LABELS[n], "delta_z": round(v, 3)}
                           for n, v in top],
        "predicted": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, blended_z)},
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
    }
