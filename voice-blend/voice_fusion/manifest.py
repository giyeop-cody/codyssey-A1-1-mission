"""매니페스트 — 어떤 음성을 썼는지, 어떤 음향 특징을 가졌는지 CSV로 고정한다.

CSV를 쓰는 이유: 스프레드시트로도 열 수 있고, 커밋 diff로 "무엇이 바뀌었는지"를
사람이 읽을 수 있다. 특징값은 반드시 실측(또는 ckpt 통계)에서 나온 것이어야 하며,
그 출처를 `feature_source` 열에 남긴다.
"""
from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass

from .features import FEATURE_NAMES, Standardizer, as_vector

MANIFEST_COLUMNS = ("voice_id", "source", "kind", "shard", "seconds",
                    "consent", "feature_source", *FEATURE_NAMES)


@dataclass(frozen=True)
class VoiceEntry:
    voice_id: str
    source: str
    kind: str          # speech | song | game_dialogue ...
    shard: str         # 학습 데이터 조각의 출처(같은 원본 중복 방지)
    seconds: float
    consent: str
    feature_source: str
    features: dict[str, float]

    def vector(self):
        return as_vector(self.features)

    def to_row(self) -> dict:
        return {
            "voice_id": self.voice_id, "source": self.source, "kind": self.kind,
            "shard": self.shard, "seconds": self.seconds, "consent": self.consent,
            "feature_source": self.feature_source,
            **{name: self.features[name] for name in FEATURE_NAMES},
        }


@dataclass
class Manifest:
    entries: list[VoiceEntry]

    def __post_init__(self) -> None:
        ids = [entry.voice_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("voice_id 중복")
        for entry in self.entries:
            if entry.kind not in ("speech", "song"):
                raise ValueError(f"{entry.voice_id}: kind는 speech|song (받은 값 {entry.kind!r})")
            if entry.seconds <= 0:
                raise ValueError(f"{entry.voice_id}: 길이가 0 이하")

    def by_id(self, voice_id: str) -> VoiceEntry:
        for entry in self.entries:
            if entry.voice_id == voice_id:
                return entry
        raise KeyError(voice_id)

    @property
    def ids(self) -> list[str]:
        return [entry.voice_id for entry in self.entries]

    def matrix(self):
        return [entry.vector() for entry in self.entries]

    def meta(self) -> dict[str, dict]:
        return {e.voice_id: {"source": e.source, "kind": e.kind, "shard": e.shard,
                             "consent": e.consent, "seconds": e.seconds} for e in self.entries}

    # ------------------------------------------------------------- IO
    @classmethod
    def read(cls, path: str | pathlib.Path) -> "Manifest":
        path = pathlib.Path(path)
        entries = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path.name}: 열 누락 {missing}")
            for row in reader:
                entries.append(VoiceEntry(
                    voice_id=row["voice_id"].strip(),
                    source=row["source"].strip(),
                    kind=row["kind"].strip(),
                    shard=row["shard"].strip(),
                    seconds=float(row["seconds"]),
                    consent=row["consent"].strip(),
                    feature_source=row["feature_source"].strip(),
                    features={name: float(row[name]) for name in FEATURE_NAMES},
                ))
        return cls(entries)

    def write(self, path: str | pathlib.Path, standardizer: Standardizer | None = None) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(entry.to_row())


def no_leakage(manifest: Manifest, min_seconds: float = 600.0) -> list[str]:
    """학습 최소 길이·중복 출처·동의 표기 등 "팔기 전에" 확인할 것들을 점검한다.

    RVC 계열은 10분 이상의 저잡음 음성을 권장한다. 그보다 짧은 화자는
    병합 재료로 쓰면 결과가 불안정해지므로 사유를 반환한다.
    """
    problems = []
    seen_shards: dict[str, str] = {}
    for entry in manifest.entries:
        if entry.seconds < min_seconds:
            problems.append(f"{entry.voice_id}: {entry.seconds:.0f}s < 권장 {min_seconds:.0f}s")
        if entry.shard in seen_shards:
            problems.append(f"{entry.voice_id}: shard '{entry.shard}'가 "
                            f"{seen_shards[entry.shard]}와 겹침(누출 위험)")
        else:
            seen_shards[entry.shard] = entry.voice_id
        if entry.consent not in ("self", "licensed", "synthetic"):
            problems.append(f"{entry.voice_id}: consent='{entry.consent}' — "
                            "self|licensed|synthetic 중 하나여야 판매 가능")
    return problems


def decontaminate(manifest: Manifest) -> tuple[Manifest, list[str]]:
    """대사/노래가 같은 원본 조각을 공유하면 특징이 서로 새어든다.

    shard가 겹치는 항목은 뒤에 나온 것을 제외하고 사유를 남긴다(결정론적).
    """
    kept, dropped, seen = [], [], {}
    for entry in manifest.entries:
        if entry.shard in seen:
            dropped.append(f"{entry.voice_id} (shard={entry.shard} ← {seen[entry.shard]})")
            continue
        seen[entry.shard] = entry.voice_id
        kept.append(entry)
    return Manifest(kept), dropped
