"""음향 측정·매니페스트·클러스터링·비교 로직 테스트(합성 신호 기반)."""
from __future__ import annotations

import numpy as np
import pytest

from voice_fusion.acoustics import describe_audio, f0_track, read_wav
from voice_fusion.analysis import vowel_onsets
from voice_fusion.clustering import agglomerative_merges, cluster_voices, labels_at
from voice_fusion.features import FEATURE_NAMES, Standardizer
from voice_fusion.manifest import Manifest, VoiceEntry, decontaminate, no_leakage
from voice_fusion.pairing import pair_speech_song
from voice_fusion.synth import build_demo_audio, synthesize


def _features(**overrides) -> dict[str, float]:
    payload = {name: 0.0 for name in FEATURE_NAMES}
    payload.update(overrides)
    return payload


def _entry(voice_id: str, kind: str = "speech", **kwargs) -> VoiceEntry:
    features = kwargs.pop("features", _features())
    payload = dict(source=voice_id, kind=kind, shard=f"shard/{voice_id}", seconds=900.0,
                   consent="self", feature_source="audio_measured")
    payload.update(kwargs)
    return VoiceEntry(voice_id=voice_id, features=features, **payload)


def test_read_wav_roundtrip(tmp_path):
    audio = synthesize(tmp_path / "a.wav", f0_hz=220, semitone_range=4, vibrato_hz=5,
                       vibrato_st=0.2, tilt=1.0, noise=0.2, syllable_hz=4, attack=0.05,
                       seconds=2.0)
    signal, rate = read_wav(audio)
    assert rate == 22050
    assert signal.size == 2 * rate
    assert np.max(np.abs(signal)) <= 1.0


def test_f0_track_recovers_synthesized_pitch(tmp_path):
    audio = synthesize(tmp_path / "b.wav", f0_hz=220, semitone_range=0.0, vibrato_hz=0.0,
                       vibrato_st=0.0, tilt=1.2, noise=0.05, syllable_hz=3, attack=0.02,
                       seconds=3.0)
    signal, rate = read_wav(audio)
    f0, periodicity = f0_track(signal, rate)
    voiced = f0[np.isfinite(f0)]
    assert voiced.size > 20
    assert np.median(voiced) == pytest.approx(220.0, rel=0.04)
    assert np.mean(periodicity >= 0.3) > 0.5


def test_describe_audio_ranges(tmp_path):
    audio = synthesize(tmp_path / "c.wav", f0_hz=180, semitone_range=10, vibrato_hz=5.5,
                       vibrato_st=0.4, tilt=0.9, noise=0.4, syllable_hz=4.0, attack=0.04,
                       seconds=3.0)
    signal, rate = read_wav(audio)
    features = describe_audio(signal, rate)
    assert set(FEATURE_NAMES) <= set(features)
    assert 0.0 <= features["brightness"] <= 1.0
    assert 0.0 <= features["roughness"] <= 1.0
    assert 0.0 <= features["attack"] <= 1.0
    assert 0.0 <= features["breathiness"] <= 1.0
    assert features["expressiveness"] > 0
    assert features["seconds"] == pytest.approx(3.0, rel=0.01)


def test_vowel_onset_rate_tracks_syllable_rate():
    rate = 22050
    t = np.arange(rate * 4) / rate
    envelope = (np.sin(2 * np.pi * 4 * t) > 0).astype(float)  # 4Hz on/off
    signal = 0.5 * np.sin(2 * np.pi * 200 * t) * envelope
    _, per_second = vowel_onsets(signal, rate)
    assert 2.5 <= per_second <= 5.5


def test_clustering_is_deterministic_and_partitions():
    rng = np.random.default_rng(1)
    matrix = np.vstack([rng.normal(0, 0.2, (4, len(FEATURE_NAMES))),
                        rng.normal(4, 0.2, (4, len(FEATURE_NAMES)))])
    a = cluster_voices(matrix, max_k=5)
    b = cluster_voices(matrix, max_k=5)
    assert a.k == b.k
    assert np.array_equal(a.labels, b.labels)
    assert len(set(a.labels)) == a.k
    # 두 덩어리는 실제로 분리되어야 한다
    assert len(set(a.labels[:4])) == 1
    assert len(set(a.labels[4:])) == 1
    assert a.labels[0] != a.labels[4]


def test_clustering_single_item_and_merges():
    single = cluster_voices(np.zeros((1, len(FEATURE_NAMES))), max_k=4)
    assert single.k == 1 and single.silhouette is None
    matrix = np.array([[0.0, 0], [0.1, 0], [5.0, 0], [5.1, 0]])
    merges = agglomerative_merges(np.array([[np.linalg.norm(a - b) for b in matrix]
                                            for a in matrix]))
    assert len(merges) == 3
    assert np.array_equal(labels_at(merges, 4, 2), np.array([0, 0, 1, 1]))


def test_manifest_rejects_bad_kind_and_duplicates():
    with pytest.raises(ValueError):
        Manifest([_entry("a", kind="rap")])
    with pytest.raises(ValueError):
        Manifest([_entry("a"), _entry("a")])


def test_manifest_csv_roundtrip(tmp_path):
    manifest = Manifest([_entry("voice-a"), _entry("voice-b", kind="song")])
    path = tmp_path / "voices.csv"
    manifest.write(path)
    again = Manifest.read(path)
    assert again.ids == manifest.ids
    assert again.entries[0].features == pytest.approx(manifest.entries[0].features)


def test_no_leakage_reports_short_shard_and_consent():
    manifest = Manifest([
        _entry("a", seconds=120.0),
        _entry("b", shard="shard/a"),
        _entry("c", consent="unknown"),
    ])
    problems = no_leakage(manifest)
    assert any("권장 600s" in p for p in problems)
    assert any("겹침" in p for p in problems)
    assert any("consent" in p for p in problems)


def test_decontaminate_drops_later_shard_copy():
    manifest = Manifest([_entry("a"), _entry("b", shard="shard/a"), _entry("c")])
    kept, dropped = decontaminate(manifest)
    assert kept.ids == ["a", "c"]
    assert dropped and "b" in dropped[0]


def test_pairing_pairs_by_range():
    speech_low = _entry("s-low", features=_features(f0_center_st=0.0, brightness=0.3))
    speech_high = _entry("s-high", features=_features(f0_center_st=12.0, brightness=0.8))
    song_low = _entry("g-low", kind="song", features=_features(f0_center_st=1.0, brightness=0.32))
    song_high = _entry("g-high", kind="song", features=_features(f0_center_st=11.0, brightness=0.79))
    manifest = Manifest([speech_low, speech_high, song_low, song_high])
    matrix = Standardizer.fit(np.array(manifest.matrix())).transform(manifest.matrix())
    pairs = {pair.speech: pair.song for pair in pair_speech_song(manifest, matrix)}
    assert pairs == {"s-low": "g-low", "s-high": "g-high"}


def test_pairing_returns_empty_without_both_kinds():
    manifest = Manifest([_entry("a"), _entry("b")])
    matrix = Standardizer.fit(np.array(manifest.matrix())).transform(manifest.matrix())
    assert pair_speech_song(manifest, matrix) == []


def test_demo_audio_builder(tmp_path):
    made = build_demo_audio(tmp_path)
    assert len(made) == 8
    assert all(path.exists() for path in made.values())
