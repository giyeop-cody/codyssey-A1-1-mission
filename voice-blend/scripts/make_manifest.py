"""매니페스트 만들기 — 오디오 폴더(실측) 또는 체크포인트 폴더(proxy)에서 생성.

실사용(권장, 실측):
    python scripts/make_manifest.py --audio-dir data/raw --template data/manifests/voices.csv \\
        --out data/manifests/voices.measured.csv

`--template`은 voice_id·source·kind·shard·seconds·consent를 사람이 채운 CSV이고,
특징 9축 열은 여기서 측정값으로 **덮어쓴다**.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from voice_fusion.acoustics import describe_file  # noqa: E402
from voice_fusion.features import FEATURE_NAMES  # noqa: E402
from voice_fusion.manifest import MANIFEST_COLUMNS, Manifest  # noqa: E402

TEMPLATE_COLUMNS = ("voice_id", "source", "kind", "shard", "consent", "audio")


def build_from_audio(template_path: pathlib.Path, audio_dir: pathlib.Path) -> Manifest:
    rows = list(csv.DictReader(template_path.open(encoding="utf-8")))
    entries = []
    for row in rows:
        audio_path = pathlib.Path(row.get("audio") or (audio_dir / f"{row['voice_id']}.wav"))
        if not audio_path.is_absolute():
            audio_path = (template_path.parent / audio_path).resolve()
        if not audio_path.exists():
            audio_path = (audio_dir / audio_path.name).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"{row['voice_id']}: 오디오 없음 ({audio_path})")
        measured = describe_file(str(audio_path)).features
        entries.append(dict(
            voice_id=row["voice_id"], source=row["source"], kind=row["kind"],
            shard=row.get("shard") or row["voice_id"],
            seconds=float(row.get("seconds") or measured["seconds"]),
            consent=row.get("consent", "self"),
            feature_source="audio_measured",
            **{name: measured[name] for name in FEATURE_NAMES},
        ))
    return Manifest([_entry(e) for e in entries])


def _entry(payload: dict):
    from voice_fusion.manifest import VoiceEntry
    return VoiceEntry(
        voice_id=payload["voice_id"], source=payload["source"], kind=payload["kind"],
        shard=payload["shard"], seconds=payload["seconds"], consent=payload["consent"],
        feature_source=payload["feature_source"],
        features={name: payload[name] for name in FEATURE_NAMES},
    )


def build_from_checkpoints(ckpt_dir: pathlib.Path, consent: str = "self") -> Manifest:
    from voice_fusion.descriptors import describe_checkpoint
    from voice_fusion.rvc import inspect_checkpoint

    entries = []
    for ckpt in sorted(ckpt_dir.rglob("*.pth")):
        if ckpt.name in ("f0G40k.pth",):
            continue
        info = inspect_checkpoint(str(ckpt))
        features = describe_checkpoint(info)
        entries.append(_entry({
            "voice_id": ckpt.parent.parent.name if ckpt.parent.name == "weights" else ckpt.stem,
            "source": ckpt.stem, "kind": "song" if "song" in ckpt.stem else "speech",
            "shard": str(ckpt), "seconds": 900.0, "consent": consent,
            "feature_source": "ckpt_stats(proxy)", **features,
        }))
    if not entries:
        raise FileNotFoundError(f"{ckpt_dir}: .pth 없음")
    return Manifest(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=pathlib.Path, help="원본 wav 폴더")
    parser.add_argument("--template", type=pathlib.Path, help="voice_id/kind/shard/consent CSV")
    parser.add_argument("--checkpoints", type=pathlib.Path, help="RVC .pth 폴더(proxy 경로)")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    if args.checkpoints:
        manifest = build_from_checkpoints(args.checkpoints)
    elif args.audio_dir and args.template:
        manifest = build_from_audio(args.template, args.audio_dir)
    else:
        parser.error("--audio-dir + --template 또는 --checkpoints 중 하나가 필요합니다")

    manifest.write(args.out)
    print(f"wrote {args.out} ({len(manifest.entries)} voices, {len(MANIFEST_COLUMNS)} cols)")
    for entry in manifest.entries:
        print(f"  {entry.voice_id:<18} {entry.kind:<6} {entry.feature_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
