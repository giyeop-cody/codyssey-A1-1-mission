"""계획 생성·렌더 감사·문구 검사·커뮤니케이션 규칙 테스트."""
from __future__ import annotations

import json

import numpy as np
import pytest

from voice_fusion import analysis, selling
from voice_fusion.features import FEATURE_NAMES, Standardizer
from voice_fusion.manifest import Manifest, VoiceEntry
from voice_fusion.plan import PlanConfig, build_plan, summary_markdown, write_plan
from voice_fusion.recipe import MergeNode
from voice_fusion.simulate import simulate_blend
from voice_fusion.synth import build_demo_audio

DIALOGUES = [{"id": f"L{i:02d}", "text": f"line {i}"} for i in range(1, 9)]
SONGS = [{"id": f"S{i:02d}", "text": f"song {i}"} for i in range(1, 8)]


def _features(center: float, bright: float, rough: float) -> dict[str, float]:
    payload = {name: 0.0 for name in FEATURE_NAMES}
    payload.update({"f0_center_st": center, "f0_range_st": 8.0, "brightness": bright,
                    "roughness": rough, "attack": 0.4, "expressiveness": 15.0,
                    "breathiness": 0.4, "nasality": -12.0, "f0_vibrato_hz": 5.0})
    return payload


def _manifest() -> Manifest:
    specs = [  # (id, kind, f0center, brightness, roughness)
        ("s1", "speech", 0.0, 0.30, 0.10), ("s2", "speech", 6.0, 0.55, 0.18),
        ("s3", "speech", 12.0, 0.80, 0.26), ("s4", "speech", 3.0, 0.38, 0.40),
        ("g1", "song", 1.0, 0.32, 0.12), ("g2", "song", 7.0, 0.58, 0.20),
        ("g3", "song", 13.0, 0.82, 0.28), ("g4", "song", 4.0, 0.40, 0.42),
    ]
    return Manifest([VoiceEntry(
        voice_id=vid, source=f"src-{vid}", kind=kind, shard=f"shard/{vid}",
        seconds=900.0, consent="self", feature_source="audio_measured",
        features=_features(center, bright, rough)) for vid, kind, center, bright, rough in specs])


def test_build_plan_shape_and_constraints():
    manifest = _manifest()
    payload = build_plan(manifest, config=PlanConfig(seed="t", n_blends=3,
                                                     min_train_seconds=10.0),
                         dialogue_library=DIALOGUES, song_library=SONGS)
    assert payload["clusters"]["k"] >= 2
    assert len(payload["blends"]) == 3
    for blend in payload["blends"]:
        assert blend["gate"]["passed"], blend["gate"]["reason"]
        shares = blend["final_weights"].values()
        assert sum(shares) == pytest.approx(1.0)
        assert max(shares) <= 0.62 + 1e-9
        # 요구사항: 노래 4~5곡, 대사 5~6개
        assert 4 <= len(blend["songs"]) <= 5
        assert 5 <= len(blend["dialogues"]) <= 6
        assert len({item["render_id"] for item in blend["songs"]}) == len(blend["songs"])
        assert blend["steps"][0]["pool"], "풀 구성이 기록되어야 한다"


def test_build_plan_is_reproducible():
    manifest = _manifest()
    kwargs = dict(config=PlanConfig(seed="same", n_blends=2, min_train_seconds=10.0),
                  dialogue_library=DIALOGUES, song_library=SONGS)
    a = build_plan(manifest, **kwargs)
    b = build_plan(manifest, **kwargs)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_build_plan_requires_speech_and_song():
    only_speech = Manifest([e for e in _manifest().entries if e.kind == "speech"])
    with pytest.raises(ValueError):
        build_plan(only_speech, config=PlanConfig(min_train_seconds=10.0),
                   dialogue_library=DIALOGUES, song_library=SONGS)


def test_build_plan_rejects_short_or_unlicensed_manifest():
    manifest = Manifest([e if e.voice_id != "s1" else VoiceEntry(
        voice_id="s1", source="x", kind="speech", shard="shard/s1", seconds=5.0,
        consent="self", feature_source="audio_measured", features=e.features)
        for e in _manifest().entries])
    with pytest.raises(ValueError, match="권장"):
        build_plan(manifest, config=PlanConfig(min_train_seconds=600.0),
                   dialogue_library=DIALOGUES, song_library=SONGS)


def test_write_plan_creates_runnable_artifacts(tmp_path):
    payload = build_plan(_manifest(), config=PlanConfig(seed="w", n_blends=1,
                                                        min_train_seconds=10.0),
                         dialogue_library=DIALOGUES, song_library=SONGS)
    paths = write_plan(payload, tmp_path)
    assert paths["plan"].exists() and paths["merge_script"].exists()
    assert paths["infer_script"].exists()
    recipes = sorted(paths["recipes"].glob("*.json"))
    assert len(recipes) == 1
    assert "final_weights" in json.loads(recipes[0].read_text())
    assert "summary" not in paths["plan"].read_text()
    assert "병합 계획 요약" in summary_markdown(payload)


def test_simulate_blend_normalizes_and_avoids_clipping():
    rate = 16000
    t = np.arange(rate) / rate
    members = {"a": (0.9 * np.sin(2 * np.pi * 200 * t), rate),
               "b": (0.9 * np.sin(2 * np.pi * 310 * t), rate)}
    mix = simulate_blend(members, {"a": 0.6, "b": 0.4})
    assert mix.size == rate
    assert np.max(np.abs(mix)) <= 0.9 + 1e-9


def test_simulate_blend_validates_input():
    with pytest.raises(KeyError):  # 가중치에만 있는 음성
        simulate_blend({"a": (np.zeros(10), 16000)}, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError):  # 가중치 합 != 1
        simulate_blend({"a": (np.zeros(10), 16000)}, {"a": 0.5})
    with pytest.raises(ValueError):
        simulate_blend({"a": (np.zeros(10), 16000), "b": (np.zeros(10), 22050)},
                       {"a": 0.5, "b": 0.5})


def test_audit_flags_silence_and_clipping():
    rate = 16000
    silent = np.zeros(rate * 3)
    audit = analysis.audit_render("x", "dialogue", silent, rate)
    assert not audit.passed
    loud = np.clip(np.sin(2 * np.pi * 200 * np.arange(rate * 3) / rate) * 1.5, -1, 1)
    loud_audit = analysis.audit_render("y", "song", loud, rate)
    assert not loud_audit.passed
    assert any("클리핑" in flag for flag in loud_audit.flags)


def test_audit_without_prosody_skips_rhythm_checks():
    rate = 16000
    signal = 0.3 * np.sin(2 * np.pi * 200 * np.arange(rate * 3) / rate)
    audit = analysis.audit_render("flat", "dialogue", signal, rate, prosody=False)
    assert audit.vowel_rate is None and audit.f0_jump_ratio is None
    assert not any("모음 시작" in flag for flag in audit.flags)


def test_audit_rejects_unknown_kind():
    with pytest.raises(ValueError):
        analysis.audit_render("x", "rap", np.zeros(1000), 16000)


def test_lint_separates_negated_disclaimers():
    disclaimers = ("본 상품은 AI 음성 모델입니다. 실제 인물의 목소리가 아니며, "
                   "사칭을 금지합니다.")
    assert analysis.lint_violations(disclaimers) == []
    bad = "이 모델은 실제 가수의 목소리를 완벽 재현했습니다."
    hits = analysis.lint_violations(bad)
    assert hits and any("실존 인물" in hit["label"] for hit in hits)


def test_price_tiers_scale_with_assets():
    cheap = selling.price_tiers(4, 5)
    rich = selling.price_tiers(5, 6)
    assert rich["personal"] > cheap["personal"]
    assert rich["commercial"] == pytest.approx(rich["personal"] * 3.5, rel=0.02)


def test_describe_product_refuses_proxy_features(tmp_path):
    manifest = _manifest()
    entries = [VoiceEntry(voice_id=e.voice_id, source=e.source, kind=e.kind, shard=e.shard,
                          seconds=e.seconds, consent=e.consent,
                          feature_source="ckpt_stats(proxy)", features=e.features)
               for e in manifest.entries]
    proxy = Manifest(entries)
    std = Standardizer.fit(np.array(proxy.matrix()))
    node = MergeNode(children=[MergeNode(voice_id="s1"), MergeNode(voice_id="g1")],
                     weights=[0.6, 0.4])
    with pytest.raises(ValueError, match="proxy"):
        selling.describe_product(name="t", node=node, manifest=proxy, standardizer=std,
                                 blended_raw=_features(1.0, 0.4, 0.2))
    preview = selling.describe_product(name="t", node=node, manifest=proxy,
                                       standardizer=std, allow_proxy=True,
                                       blended_raw=_features(1.0, 0.4, 0.2))
    assert "라이선스" in preview["markdown"]
    assert preview["lint"] == [], preview["lint"]


def test_describe_product_contains_compliance_text():
    manifest = _manifest()
    std = Standardizer.fit(np.array(manifest.matrix()))
    node = MergeNode(children=[MergeNode(voice_id=s) for s in ("s1", "g1", "s2")],
                     weights=[0.5, 0.3, 0.2])
    product = selling.describe_product(name="테스트", node=node, manifest=manifest,
                                       standardizer=std,
                                       blended_raw=_features(2.0, 0.45, 0.2),
                                       price_krw=15000)
    body = product["markdown"]
    assert "AI 생성" in body and "재판매" in body and "15,000원" in body
    assert "상태: PREVIEW" in body


def test_render_audit_passes_on_demo_song(tmp_path):
    audio = build_demo_audio(tmp_path)["syn-mid-song"]
    from voice_fusion.acoustics import read_wav
    signal, rate = read_wav(audio)
    audit = analysis.audit_render("demo-song", "song", signal, rate)
    assert audit.duration_s == pytest.approx(4.0, rel=0.02)
    assert audit.peak_dbfs < -0.1
