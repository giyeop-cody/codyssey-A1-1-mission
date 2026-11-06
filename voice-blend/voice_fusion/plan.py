"""전체 계획 수립 — 클러스터링 → 풀 선택 → 랜덤 레시피 → 게이트 → 산출물.

한 번의 `build_plan()` 호출로 다음이 결정되고 JSON으로 고정된다.
1. 어떤 음성들이 어떤 군집에 속하는가(비슷한 특징끼리 묶기)
2. 어떤 (대사,노래) 짝을 기본 재료로 쓰는가
3. 각 병합본의 **랜덤 경로**(순서·개수·비중)와 최종 가중치
4. 각 병합본으로 만들 노래 4~5곡 / 대사 5~6개
5. 게이트 통과 여부와 실패 사유

`plan.json`은 그대로 실행 스크립트의 입력이 되므로, 계획과 렌더가 어긋나지 않는다.
"""
from __future__ import annotations

import itertools
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

from .analysis import lint_copy
from .clustering import cluster_voices
from .features import FEATURE_NAMES, Standardizer, as_mapping
from .manifest import Manifest, VoiceEntry, no_leakage
from .pairing import Pair, pair_speech_song
from .recipe import (DeterministicRNG, GateConfig, MergeNode, blend_features,
                     check_recipe, final_weights, sample_recipe)

DIALOGUE_RANGE = (5, 6)
SONG_RANGE = (4, 5)


@dataclass
class PlanConfig:
    seed: str = "voice-blend-v1"
    n_blends: int = 3
    recipes_per_blend: int = 60      # 병합본당 시도 횟수(랜덤 경로 탐색)
    gate: GateConfig = field(default_factory=GateConfig)
    max_k: int = 6
    min_pool: int = 3
    max_pool: int = 5
    song_range: tuple[int, int] = SONG_RANGE
    dialogue_range: tuple[int, int] = DIALOGUE_RANGE
    # RVC 권장 최소 학습 길이(10분). 데모(합성 4초 클립)에서는 낮춰서 쓴다.
    min_train_seconds: float = 600.0
    # "이어붙이는 단계 수"를 병합본마다 다르게 해서 한 런에 짧은 경로/긴 경로가
    # 모두 나오게 한다(사용자 예시: 1단계 ~ 여러 단계 체인).
    route_preferences: tuple[int, ...] = (2, 4, 3, 5)


@dataclass
class BlendPlan:
    recipe_id: str
    name: str
    voice_ids: list[str]
    final_weights: dict[str, float]
    steps: list[dict]
    tree: dict
    cluster: int
    predicted_features: dict[str, float]
    gate: dict
    songs: list[dict]
    dialogues: list[dict]
    checkpoint_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "recipe_id": self.recipe_id, "name": self.name,
            "voice_ids": self.voice_ids, "final_weights": self.final_weights,
            "steps": self.steps, "tree": self.tree, "cluster": self.cluster,
            "predicted_features": {k: round(v, 6) for k, v in self.predicted_features.items()},
            "gate": self.gate, "songs": self.songs, "dialogues": self.dialogues,
            "checkpoints": self.checkpoint_paths,
        }


def _pick_assets(rng: DeterministicRNG, library: list[dict], count: int,
                 kind: str) -> list[dict]:
    if not library:
        return []
    index = list(library)
    for i in range(len(index) - 1, 0, -1):
        j = rng.randint(0, i)
        index[i], index[j] = index[j], index[i]
    chosen = index[:max(1, min(count, len(index)))]
    return [{**item, "kind": kind, "render_id": f"{kind}-{pos + 1:02d}-{item['id']}"}
            for pos, item in enumerate(chosen)]


def _diversity_score(weights: dict[str, float], z: dict[str, np.ndarray]) -> float:
    """리프들 사이 최소 거리(표준화 공간). 클수록 음색 대비가 큼."""
    ids = list(weights)
    if len(ids) < 2:
        return 0.0
    return float(min(np.linalg.norm(z[a] - z[b]) for a, b in itertools.combinations(ids, 2)))


def _kind_balance(weights: dict[str, float], manifest: Manifest) -> float:
    kinds: dict[str, float] = {}
    for voice, weight in weights.items():
        kinds[manifest.by_id(voice).kind] = kinds.get(manifest.by_id(voice).kind, 0.0) + weight
    return float(min(kinds.get("song", 0.0), kinds.get("speech", 0.0)) * 2)  # 0~1


def build_plan(
    manifest: Manifest,
    *,
    config: PlanConfig,
    dialogue_library: list[dict],
    song_library: list[dict],
    checkpoint_paths: dict[str, str] | None = None,
) -> dict:
    """계획 전체를 만들고 JSON 직렬화 가능한 dict를 돌려준다."""
    problems = no_leakage(manifest, min_seconds=config.min_train_seconds)
    if problems:
        raise ValueError("매니페스트 사전 점검 실패:\n - " + "\n - ".join(problems))

    matrix = np.array(manifest.matrix(), dtype=float)
    standardizer = Standardizer.fit(matrix)
    z_matrix = standardizer.transform(matrix)
    z = {entry.voice_id: z_matrix[i] for i, entry in enumerate(manifest.entries)}
    raw = {entry.voice_id: entry.vector() for entry in manifest.entries}

    clusters = cluster_voices(z_matrix, max_k=config.max_k)
    cluster_of = {vid: int(clusters.labels[i]) for i, vid in enumerate(manifest.ids)}
    pairs = pair_speech_song(manifest, z_matrix)
    if not pairs:
        raise ValueError("대사(speech)와 노래(song) 음성이 각각 1개 이상 필요합니다")

    rng = DeterministicRNG(f"{config.seed}|plan")
    gate = config.gate
    plans: list[BlendPlan] = []

    for slot in range(config.n_blends):
        base_pair: Pair = pairs[slot % len(pairs)]
        # 풀: 기본 짝 + 같은 군집에서 가까운 순으로 채운다.
        target_cluster = cluster_of[base_pair.song]
        same = [vid for vid in manifest.ids if cluster_of[vid] == target_cluster]
        same.sort(key=lambda vid: float(np.linalg.norm(z[vid] - z[base_pair.song])))
        pool = [base_pair.speech, base_pair.song]
        for vid in same:
            if len(pool) >= config.max_pool:
                break
            if vid not in pool:
                pool.append(vid)
        if len(pool) < config.min_pool:
            for vid in manifest.ids:  # 같은 군집이 작으면 전체에서 보충
                if len(pool) >= config.min_pool:
                    break
                if vid not in pool:
                    pool.append(vid)

        pool_source = "같은 군집" if len(pool) == len(same) else "같은 군집 우선 + 전체 보충"
        best = None
        for attempt in range(config.recipes_per_blend):
            local = DeterministicRNG(f"{config.seed}|{slot}|{attempt}")
            picked, tree, steps = sample_recipe(local, pool,
                                                pick_min=config.min_pool,
                                                pick_max=min(config.max_pool, len(pool)))
            report = check_recipe(tree, z, gate, center=np.zeros(len(FEATURE_NAMES)))
            if not report.passed:
                continue
            weights = final_weights(tree)
            chain_len = sum(1 for step in steps if step["kind"] == "chain")
            target = config.route_preferences[slot % len(config.route_preferences)]
            score = (_diversity_score(weights, z) + _kind_balance(weights, manifest)
                     - max(0.0, max(weights.values()) - 0.45)
                     - 0.12 * abs(chain_len - target))
            if best is None or score > best[0]:
                best = (score, picked, tree, steps, weights, report)
        if best is None:
            raise RuntimeError(f"병합본 {slot + 1}: 게이트를 통과하는 레시피를 찾지 못했습니다. "
                               "풀 크기를 늘리거나 게이트 기준을 확인하세요.")

        score, picked, tree, steps, weights, report = best
        blended_z = blend_features(z, tree)
        # 예측 특징은 표준화 공간이 아니라 실제 단위로 돌려준다(문구에 쓰기 위해).
        blended_raw = as_mapping(blended_z * standardizer.scale + standardizer.center)

        n_song = rng.randint(*config.song_range)
        n_dialogue = rng.randint(*config.dialogue_range)
        songs = _pick_assets(rng, song_library, n_song, "song")
        dialogues = _pick_assets(rng, dialogue_library, n_dialogue, "dialogue")
        recipe_id = f"blend-{slot + 1:02d}-{_short_hash(weights)}"
        name = _name_voice(slot, weights, manifest, recipe_id)
        steps = [{"pool": pool, "pool_source": pool_source}] + steps
        paths = {vid: (checkpoint_paths or {}).get(vid, f"models/{vid}.pth") for vid in weights}
        plans.append(BlendPlan(
            recipe_id=recipe_id, name=name, voice_ids=list(weights),
            final_weights=weights, steps=steps, tree=tree.to_dict(),
            cluster=int(target_cluster), predicted_features=blended_raw,
            gate=report.to_dict(), songs=songs, dialogues=dialogues,
            checkpoint_paths=paths))

    return {
        "version": 1,
        "seed": config.seed,
        "features": list(FEATURE_NAMES),
        "standardizer": standardizer.to_dict(),
        "clusters": {
            "k": clusters.k,
            "silhouette": None if clusters.silhouette is None else round(clusters.silhouette, 4),
            "assignment": cluster_of,
            "score_table": clusters.score_table,
        },
        "pairs": [pair.to_dict() for pair in pairs],
        "blends": [plan.to_dict() for plan in plans],
        "manifest": [{"voice_id": e.voice_id, "kind": e.kind, "source": e.source,
                      "consent": e.consent, "feature_source": e.feature_source}
                     for e in manifest.entries],
    }


def _short_hash(weights: dict[str, float]) -> str:
    import hashlib
    payload = "|".join(f"{k}:{v:.6f}" for k, v in sorted(weights.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:6]


_ADJ = ("흐릿한", "서늘한", "젖은", "따뜻한", "각진", "부드러운", "야릇한", "맑은",
        "거친", "고요한", "번지는", "날카로운")


def _name_voice(slot: int, weights: dict[str, float], manifest: Manifest,
                recipe_id: str) -> str:
    import hashlib
    digest = hashlib.sha256(recipe_id.encode()).digest()
    adj = _ADJ[digest[0] % len(_ADJ)]
    noun = ("목소리", "음색", "톤")[digest[1] % 3]
    lead = max(weights.items(), key=lambda kv: kv[1])[0]
    return f"{adj} {noun} #{slot + 1} ({lead} 중심)"


def write_plan(payload: dict, out_dir: str | pathlib.Path) -> dict[str, pathlib.Path]:
    """계획을 디스크에 남긴다(JSON + 병합 스크립트 + 실행용 레시피 파일)."""
    from .rvc import write_infer_script, write_merge_script

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    recipe_dir = out / "recipes"
    recipe_dir.mkdir(exist_ok=True)
    for blend in payload["blends"]:
        (recipe_dir / f"{blend['recipe_id']}.json").write_text(
            json.dumps(blend, ensure_ascii=False, indent=2), encoding="utf-8")

    merge = write_merge_script(out)
    infer = write_infer_script(out)
    return {"plan": plan_path, "merge_script": merge, "infer_script": infer,
            "recipes": recipe_dir}


def lint_plan(payload: dict) -> list[dict]:
    text = json.dumps(payload, ensure_ascii=False)
    return lint_copy(text)


def summary_markdown(payload: dict) -> str:
    clusters = payload["clusters"]
    lines = ["# 병합 계획 요약", "",
             f"- 시드: `{payload['seed']}`",
             f"- 군집: k={clusters['k']} (실루엣 {clusters['silhouette']})",
             f"- 대사↔노래 짝: {len(payload['pairs'])}쌍",
             f"- 병합본: {len(payload['blends'])}개", ""]
    for blend in payload["blends"]:
        lines += [f"## {blend['name']}", "",
                  f"- ID `{blend['recipe_id']}` · 군집 {blend['cluster']} · "
                  f"게이트 {'통과' if blend['gate']['passed'] else '실패'}",
                  "- 풀: " + ", ".join(blend["steps"][0].get("pool", [])),
                  f"  ({blend['steps'][0].get('pool_source', '')})",
                  "- 랜덤 경로:"]
        for step in blend["steps"]:
            if "step" not in step:
                continue
            parts = [f"{item}={weight:.3f}" if item == "carry"
                     else f"{item}={weight:.3f}"
                     for item, weight in zip(step["items"], step["weights"])]
            lines.append(f"  {step['step']}) " + " + ".join(parts))
        lines.append("- 최종 비중: " + ", ".join(
            f"{k} {v:.1%}" for k, v in blend["final_weights"].items()))
        lines.append(f"- 산출물: 노래 {len(blend['songs'])}곡 / 대사 {len(blend['dialogues'])}개")
        lines.append("")
    return "\n".join(lines)
