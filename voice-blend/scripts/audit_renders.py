"""렌더 결과 감사 — 노래 4~5곡/대사 5~6개를 실제로 검사하고 판매 문구를 확정한다.

    python scripts/audit_renders.py --plan plan/plan.json --renders renders/ \\
        --out reports/audit

`renders/` 안의 파일 이름은 계획의 `render_id`와 같아야 한다(예: `dialogue-01-L03.wav`).
빠진 렌더는 실패로 집계한다 — 없는 산출물을 있는 것처럼 팔지 않기 위해서다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from voice_fusion import acoustics, analysis, selling  # noqa: E402
from voice_fusion.features import FEATURE_NAMES, Standardizer  # noqa: E402
from voice_fusion.manifest import Manifest  # noqa: E402
from voice_fusion.recipe import MergeNode  # noqa: E402


def node_from(payload: dict) -> MergeNode:
    if "leaf" in payload:
        return MergeNode(voice_id=payload["leaf"])
    return MergeNode(children=[node_from(child) for child in payload["children"]],
                     weights=payload["weights"], label=payload.get("label", ""))


def blend_features(node: MergeNode, measured: dict[str, dict[str, float]]) -> dict[str, float]:
    """트리 구조대로 가중 평균한 특징(가중치 합이 1이므로 스케일이 유지된다)."""
    if node.is_leaf:
        return measured[node.voice_id]
    acc: dict[str, float] = {}
    for child, weight in zip(node.children, node.weights):
        for name, value in blend_features(child, measured).items():
            acc[name] = acc.get(name, 0.0) + float(weight) * value
    return acc


def find_render(renders: pathlib.Path, render_id: str) -> pathlib.Path | None:
    for suffix in (".wav", ".WAV", ".flac", ".aiff"):
        candidate = renders / f"{render_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(renders.glob(f"{render_id}*"))
    return matches[0] if matches else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--renders", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reports/audit"))
    parser.add_argument("--manifest", type=pathlib.Path,
                        help="선택: 원본 특징(정체성 일관성 계산용)")
    parser.add_argument("--min-songs", type=int, default=4)
    parser.add_argument("--min-dialogues", type=int, default=5)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"blends": [], "problems": []}

    for blend in plan["blends"]:
        rows, missing = [], []
        for item in blend["songs"] + blend["dialogues"]:
            path = find_render(args.renders, item["render_id"])
            if path is None:
                missing.append(item["render_id"])
                continue
            try:
                signal, rate = acoustics.read_wav(path)
            except ValueError as exc:  # flac 등 미지원
                missing.append(f"{item['render_id']} ({exc})")
                continue
            audit = analysis.audit_render(item["render_id"], item["kind"], signal, rate)
            measured = acoustics.describe_audio(signal, rate)
            rows.append({**audit.to_dict(),
                         "features": {k: round(measured[k], 6) for k in FEATURE_NAMES}})

        counts = {
            "songs": sum(1 for r in rows if r["kind"] == "song"),
            "dialogues": sum(1 for r in rows if r["kind"] == "dialogue"),
        }
        passed = [r for r in rows if r["passed"]]
        block = {
            "recipe_id": blend["recipe_id"], "name": blend["name"],
            "counts": counts, "missing": missing,
            "audited": rows,
            "passed": len(passed), "failed": len(rows) - len(passed),
        }
        if counts["songs"] < args.min_songs:
            summary["problems"].append(f"{blend['recipe_id']}: 노래 {counts['songs']}곡 "
                                       f"< {args.min_songs}")
        if counts["dialogues"] < args.min_dialogues:
            summary["problems"].append(f"{blend['recipe_id']}: 대사 {counts['dialogues']}개 "
                                       f"< {args.min_dialogues}")
        if missing:
            summary["problems"].append(f"{blend['recipe_id']}: 렌더 누락 {missing}")
        summary["blends"].append(block)

    if args.manifest:
        manifest = Manifest.read(args.manifest)
        standardizer = Standardizer.from_dict(plan["standardizer"])
        matrix = standardizer.transform([e.vector() for e in manifest.entries])
        import hashlib
        # 타깃은 voice_id 해시 기반 의사 라벨: 같은 음성인지 구분하는 정보가
        # 특징 안에 얼마나 남아 있는지를 재는 용도(라벨 누출 없음).
        target = [int.from_bytes(
            hashlib.sha256(e.voice_id.encode()).digest()[:2], "big") % 17
            for e in manifest.entries]
        summary["consistency"] = analysis.consistency(
            matrix, target, list(plan["features"]))
        # 판매 문구는 "렌더 통과분"만 반영해 다시 생성
        for blend in plan["blends"]:
            block = next(b for b in summary["blends"] if b["recipe_id"] == blend["recipe_id"])
            product = selling.describe_product(
                name=blend["name"], node=node_from(blend["tree"]), manifest=manifest,
                standardizer=standardizer, blended_raw=blend["predicted_features"],
                consistency=summary["consistency"], audited=block["audited"],
                price_krw=selling.price_tiers(block["counts"]["songs"],
                                              block["counts"]["dialogues"])["personal"],
                status="READY" if (not block["missing"] and block["failed"] == 0) else "NEEDS-FIX",
            )
            (args.out / f"product_{blend['recipe_id']}.md").write_text(
                product["markdown"] + "\n", encoding="utf-8")
            block["lint"] = product["lint"]

    report = args.out / "audit.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 렌더 감사", "",
             f"- 병합본 {len(summary['blends'])}개",
             f"- 문제 {len(summary['problems'])}건"]
    lines += [f"  - {item}" for item in summary["problems"]] or ["  - 없음"]
    lines += ["", "| blend | 노래 | 대사 | 통과 | 실패 | 누락 |", "|---|---:|---:|---:|---:|---:|"]
    for block in summary["blends"]:
        lines.append(f"| {block['recipe_id']} | {block['counts']['songs']} | "
                     f"{block['counts']['dialogues']} | {block['passed']} | "
                     f"{block['failed']} | {len(block['missing'])} |")
    if summary.get("consistency"):
        cons = summary["consistency"]
        lines += ["", "## 정체성 일관성(kNN 교차검증)",
                  f"- 전체 R²: {cons['r2_full']}",
                  f"- 결정적 지표군: {', '.join(cons['critical_families']) or '없음'}"]
        for row in cons["knockouts"]:
            lines.append(f"  - {row['family']} 제거 → R² {row['r2_without']} "
                         f"({row['drop_pct']:+.1f}%){' ★' if row['critical'] else ''}")
    (args.out / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"감사 저장: {report} / {args.out / 'AUDIT.md'}")
    return 0 if not summary["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
