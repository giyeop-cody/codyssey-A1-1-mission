"""데모: 합성 음성으로 전 과정을 실제 계산해 리포트를 만든다(GPU·torch 불필요).

    python scripts/demo_plan.py

하는 일(전부 실제 실행)
1. 합성 음성 8개 생성(대사 4 / 노래 4) — **실제 가수가 아님**
2. wav에서 9축 특징을 실측 → 매니페스트 CSV
3. 군집화(k 선택 포함) → 대사↔노래 짝 → 병합 풀
4. 병합본 3개: 랜덤 순서·개수·비중 경로 + 게이트 통과 여부
5. 산출물 목록(노래 4~5곡 / 대사 5~6개)과 판매 설명 초안 생성
6. 렌더 품질 게이트를 원본 음성에 적용해 임계값이 실제로 동작하는지 확인
7. artifacts/demo/REPORT.md 로 저장
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice_fusion import acoustics, analysis, selling  # noqa: E402
from voice_fusion.features import FEATURE_LABELS, FEATURE_NAMES, Standardizer  # noqa: E402
from voice_fusion.manifest import Manifest, no_leakage  # noqa: E402
from voice_fusion.plan import PlanConfig, build_plan, summary_markdown, write_plan  # noqa: E402
from voice_fusion.synth import SYNTH_VOICES, build_demo_audio  # noqa: E402


def mix_rate(members: dict) -> int:
    rates = {rate for _, rate in members.values()}
    if len(rates) != 1:
        raise ValueError(f"샘플레이트가 섞여 있습니다: {sorted(rates)}")
    return rates.pop()

DIALOGUE_LIB = [
    {"id": "L01", "text": "\"여기까지야. 더는 못 가.\"", "seconds": 3.2, "tone": "체념"},
    {"id": "L02", "text": "\"네가 그렇게 말할 줄 알았어.\"", "seconds": 2.7, "tone": "비꼼"},
    {"id": "L03", "text": "\"문 열어. 나야.\"", "seconds": 2.1, "tone": "조급"},
    {"id": "L04", "text": "\"이번엔 진짜 끝이야.\"", "seconds": 2.4, "tone": "단호"},
    {"id": "L05", "text": "\"웃지 마. 진지하니까.\"", "seconds": 2.9, "tone": "발끈"},
    {"id": "L06", "text": "\"다시 시작할 수 있을까.\"", "seconds": 3.5, "tone": "회복"},
    {"id": "L07", "text": "\"비가 오네.\"", "seconds": 1.8, "tone": "담담"},
    {"id": "L08", "text": "\"나 여기 있을게.\"", "seconds": 2.2, "tone": "다정"},
]
SONG_LIB = [
    {"id": "S01", "text": "[Verse] 새벽 세 시의 골목 / 네 이름을 부르다", "seconds": 42, "bpm": 78},
    {"id": "S02", "text": "[Chorus] 다시 봄이 오면 / 너의 자리에 서 있을게", "seconds": 38, "bpm": 92},
    {"id": "S03", "text": "[Verse] 지워지지 않는 장면 / 필름처럼 번져", "seconds": 45, "bpm": 70},
    {"id": "S04", "text": "[Bridge] 말하지 못한 것들이 / 노래가 되어", "seconds": 33, "bpm": 84},
    {"id": "S05", "text": "[Chorus] 우리는 여기까지 / 하지만 노래는 남아", "seconds": 40, "bpm": 96},
]


def write_template(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["voice_id", "source", "kind", "shard",
                                                    "consent", "audio"])
        writer.writeheader()
        for name in SYNTH_VOICES:
            writer.writerow({
                "voice_id": name,
                "source": f"합성 {name.split('-')[1].upper()} ({name.split('-')[2]})",
                "kind": "speech" if name.endswith("speech") else "song",
                "shard": f"synthetic/{name}",
                "consent": "synthetic",
                "audio": f"{name}.wav",
            })


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - 단계별 리포트 생성
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "artifacts" / "demo")
    parser.add_argument("--seed", default="voice-blend-v1")
    parser.add_argument("--blends", type=int, default=3)
    args = parser.parse_args(argv)

    out = args.out
    # 이전 실행 산출물(특히 product_*.md)이 남아 있으면 리포트와 어긋난다.
    if out.exists():
        for stale in list(out.glob("product_*.md")) + list(out.glob("*.csv")):
            stale.unlink()
        plan_dir = out / "plan" / "recipes"
        if plan_dir.exists():
            for stale in plan_dir.glob("*.json"):
                stale.unlink()
    audio_dir = out / "audio"
    print("[1/6] 합성 음성 생성 …")
    build_demo_audio(audio_dir)

    print("[2/6] 특징 실측 → 매니페스트 …")
    template = out / "voices_template.csv"
    write_template(template)
    from scripts.make_manifest import build_from_audio  # noqa: PLC0415
    manifest = build_from_audio(template, audio_dir)
    manifest.write(out / "voices_measured.csv")
    # 데모 클립은 4초라 RVC 권장 600초 기준을 그대로 걸 수 없다. 동의/누출 검사는
    # 그대로 수행하고, 길이 기준만 낮춰서 통과시킨다(리포트에 명시).
    demo_min_seconds = 1.0
    problems = no_leakage(manifest, min_seconds=demo_min_seconds)
    if problems:
        raise SystemExit("매니페스트 점검 실패:\n - " + "\n - ".join(problems))

    print("[3/6] 군집화 · 짝짓기 · 랜덤 경로 · 게이트 …")
    payload = build_plan(manifest, config=PlanConfig(seed=args.seed, n_blends=args.blends,
                                                     min_train_seconds=demo_min_seconds),
                         dialogue_library=DIALOGUE_LIB, song_library=SONG_LIB)
    paths = write_plan(payload, out / "plan")

    print("[4/6] 판매 설명 초안 …")
    standardizer = Standardizer.from_dict(payload["standardizer"])
    from voice_fusion.recipe import MergeNode  # noqa: PLC0415

    def node_from(dict_node: dict) -> MergeNode:
        if "leaf" in dict_node:
            return MergeNode(voice_id=dict_node["leaf"])
        return MergeNode(children=[node_from(c) for c in dict_node["children"]],
                         weights=dict_node["weights"], label=dict_node.get("label", ""))

    products = []
    for blend in payload["blends"]:
        preview = selling.describe_product(
            name=blend["name"], node=node_from(blend["tree"]), manifest=manifest,
            standardizer=standardizer, blended_raw=blend["predicted_features"],
            price_krw=selling.price_tiers(len(blend["songs"]), len(blend["dialogues"]))["personal"],
            status="PREVIEW(synthetic)",
        )
        path = out / f"product_{blend['recipe_id']}.md"
        path.write_text(preview["markdown"] + "\n", encoding="utf-8")
        products.append((blend, preview, path))

    print("[5/6] 렌더 시뮬레이션 + 품질 게이트 검증 …")
    from voice_fusion.recipe import final_weights  # noqa: PLC0415
    from voice_fusion.simulate import simulate_blend  # noqa: PLC0415

    simulated = []
    for blend in payload["blends"]:
        node = node_from(blend["tree"])
        weights = final_weights(node)
        members = {vid: acoustics.read_wav(audio_dir / f"{vid}.wav") for vid in weights}
        # 실제 RVC 렌더는 곡마다 대상(노래/대사)이 다르므로, 비중 검증은
        # "노래 모드"와 "대사 모드" 두 기준으로 같은 믹스를 감사한다.
        rate = mix_rate(members)
        mix = simulate_blend(members, weights)
        # 파형 단순 가산은 운율 지표가 의미를 갖지 않으므로 구조 검사만 한다.
        audits = [analysis.audit_render(f"{blend['recipe_id']}-sim", "song", mix, rate,
                                        prosody=False)]
        simulated.append((blend["recipe_id"], audits))
    audits = []
    for name in SYNTH_VOICES:
        signal, rate = acoustics.read_wav(audio_dir / f"{name}.wav")
        kind = "dialogue" if name.endswith("speech") else "song"
        audits.append(analysis.audit_render(name, kind, signal, rate).to_dict())

    print("[6/6] 리포트 작성 …")
    lines = ["# voice-blend 데모 리포트", "",
             "> ⚠ 모든 음성은 **합성 신호**입니다(가수 아님). 이 리포트의 목적은",
             "> 파이프라인이 실제로 동작함을 수치로 보이는 것입니다.",
             f"> 길이 기준: 학습 권장 600초를 데모에서만 {demo_min_seconds:.0f}초로 낮췄습니다.", ""]
    lines += ["## 1. 매니페스트(실측 특징)", "",
              "| voice | kind | f0중심(st) | 음역(st) | 밝기 | 거칠기 | 어택 | 비음(dB) | 숨소리 | 기복(dB) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in manifest.entries:
        f = entry.features
        lines.append(f"| {entry.voice_id} | {entry.kind} | {f['f0_center_st']:.2f} | "
                     f"{f['f0_range_st']:.2f} | {f['brightness']:.3f} | {f['roughness']:.3f} | "
                     f"{f['attack']:.3f} | {f['nasality']:.2f} | {f['breathiness']:.3f} | "
                     f"{f['expressiveness']:.2f} |")
    lines += ["", "## 2. 군집", "",
              f"- k = {payload['clusters']['k']} (실루엣 {payload['clusters']['silhouette']})"]
    for k_row in payload["clusters"]["score_table"]:
        lines.append(f"  - k={k_row['k']}: 실루엣 {k_row['silhouette']}"
                     + (f" · 제외({k_row['rejected']})" if k_row.get("rejected") else ""))
    groups: dict[int, list[str]] = {}
    for voice, label in payload["clusters"]["assignment"].items():
        groups.setdefault(label, []).append(voice)
    for label, members in sorted(groups.items()):
        lines.append(f"- 군집 {label}: {', '.join(members)}")
    lines += ["", "## 3. 대사↔노래 짝", ""]
    for pair in payload["pairs"]:
        lines.append(f"- {pair['speech']} ↔ {pair['song']} (거리 {pair['distance']}, "
                     f"가까운 축: {', '.join(pair['close_axes']) or '없음'})")
    lines += ["", "## 4. 병합 레시피와 산출물", "", summary_markdown(payload), ""]
    lines += ["## 5. 판매 설명 초안", ""]
    for blend, preview, path in products:
        lines += [f"### {blend['name']}", "",
                  f"- 파일: `{path.relative_to(out)}`",
                  "- 문구 lint: " + ("위험 표현 없음" if not preview["lint"] else str(preview["lint"]))
                  + (f" (면책 문맥 {len(preview['lint_all']) - len(preview['lint'])}건 제외)"
                     if preview.get("lint_all") else ""),
                  f"- 게이트: {'통과' if preview['gate']['passed'] else preview['gate']['reason']}", ""]
        lines += ["```text", *preview["markdown"].splitlines()[:18], "…", "```", ""]
    lines += ["## 6a. 병합 비중 시뮬레이션(파형 가중 믹스 · 모델 병합 아님)", "",
              "구조 검사만 수행합니다(클리핑·무음·길이). 서로 다른 음성의 파형을 그대로 더하면",
              "운율 지표가 무의미해지므로, 운율·f0 검사는 실제 RVC 렌더(6b 방식)에서만 합니다.", "",
              "| blend | 길이(s) | peak(dBFS) | 무음비 | 판정 |",
              "|---|---:|---:|---:|---|"]
    for recipe_id, simulated_audits in simulated:
        for row in simulated_audits:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            lines.append(f"| {recipe_id} | {row['duration_s']} | {row['peak_dbfs']} | "
                         f"{row['silence_ratio']} | "
                         f"{'통과' if row['passed'] else '; '.join(row['flags'])} |")
    lines += ["", "## 6b. 품질 게이트 검증(원본 음성)", "",
              "| render | kind | 길이(s) | peak(dBFS) | 무음비 | 모음시작/s | f0점프 | 판정 |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for audit in audits:
        row = audit.to_dict() if hasattr(audit, "to_dict") else audit
        lines.append(f"| {row['render_id']} | {row['kind']} | {row['duration_s']} | "
                     f"{row['peak_dbfs']} | {row['silence_ratio']} | {row['vowel_rate']} | "
                     f"{row['f0_jump_ratio']} | {'통과' if row['passed'] else '; '.join(row['flags'])} |")
    lines += ["", "## 7. 다음 단계(실데이터)", "",
              "1. `data/raw/<voice_id>.wav`에 학습에 쓸 음성(10분 이상, 권리 확보) 배치",
              "2. `scripts/make_manifest.py --audio-dir data/raw --template ...`로 실측 매니페스트 생성",
              "3. `scripts/build_plan.py` → `plan/plan.json` 확정",
              "4. `plan/merge_recipe.py`로 RVC ckpt 병합(GPU 환경)",
              "5. `plan/render_track.py`로 노래 4~5곡·대사 5~6개 렌더",
              "6. `scripts/audit_renders.py`로 품질 게이트 → 통과분만 판매 문구 생성", ""]
    report = out / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"완료: {report}")
    print(f"  매니페스트 {out / 'voices_measured.csv'}")
    print(f"  계획 {paths['plan']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
