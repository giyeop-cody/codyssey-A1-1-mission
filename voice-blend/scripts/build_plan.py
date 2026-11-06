"""계획을 만들고 JSON/스크립트로 고정한다(실데이터용).

    python scripts/build_plan.py --manifest data/manifests/real.csv \\
        --dialogues data/libraries/dialogues.json --songs data/libraries/songs.json \\
        --out plan --seed voice-blend-v1 --blends 3

렌더 목록(`songs`/`dialogues`)은 라이브러리 JSON에서 **시드 기반으로 무작위 선택**된다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from voice_fusion.analysis import lint_violations  # noqa: E402
from voice_fusion.manifest import Manifest  # noqa: E402
from voice_fusion.plan import PlanConfig, build_plan, summary_markdown, write_plan  # noqa: E402


def load_library(path: pathlib.Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("items", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path.name}: 항목 리스트가 아닙니다")
    for item in payload:
        if "id" not in item:
            raise ValueError(f"{path.name}: 'id' 없는 항목")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--dialogues", type=pathlib.Path, required=True)
    parser.add_argument("--songs", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("plan"))
    parser.add_argument("--seed", default="voice-blend-v1")
    parser.add_argument("--blends", type=int, default=3)
    parser.add_argument("--max-leaves", type=int, default=6)
    parser.add_argument("--max-share", type=float, default=0.62)
    parser.add_argument("--checkpoints", type=pathlib.Path,
                        help="선택: voice_id→ckpt 경로 JSON. 없으면 plan에서 경로를 비워 둔다")
    args = parser.parse_args(argv)

    manifest = Manifest.read(args.manifest)
    ckpt_map = None
    if args.checkpoints:
        ckpt_map = json.loads(args.checkpoints.read_text(encoding="utf-8"))

    config = PlanConfig(seed=args.seed, n_blends=args.blends)
    config.gate.max_leaves = args.max_leaves
    config.gate.max_share = args.max_share
    payload = build_plan(manifest, config=config,
                         dialogue_library=load_library(args.dialogues),
                         song_library=load_library(args.songs),
                         checkpoint_paths=ckpt_map)
    hits = lint_violations(json.dumps(payload, ensure_ascii=False))
    if hits:
        print("⚠ 위험 표현이 계획에 포함되어 있습니다:", hits)

    paths = write_plan(payload, args.out)
    (args.out / "SUMMARY.md").write_text(summary_markdown(payload), encoding="utf-8")
    print(f"계획 저장: {paths['plan']}")
    print(f"병합 스크립트: {paths['merge_script']}")
    print(f"추론 스크립트: {paths['infer_script']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
