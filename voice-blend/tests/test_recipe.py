"""핵심 계산(가중치 트리·결정론적 난수·게이트) 단위 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from voice_fusion.features import FEATURE_NAMES, Standardizer, as_mapping, as_vector
from voice_fusion.recipe import (DeterministicRNG, GateConfig, MergeNode, blend_features,
                                 check_recipe, final_weights, normalize, sample_recipe)


def test_final_weights_two_models():
    node = MergeNode(children=[MergeNode(voice_id="a"), MergeNode(voice_id="b")],
                     weights=[0.7, 0.3])
    assert final_weights(node) == {"a": 0.7, "b": 0.3}


def test_final_weights_tree_matches_sequential_merge():
    """1:3:2 → 그 결과와 1개를 0.4:0.6으로 → 남은 1개를 0.25로.

    손으로 계산한 값과 트리 재귀 계산이 같아야 한다(수학적으로 Weighted Sum과 동일).
    """
    a, b, c, d, e = (MergeNode(voice_id=v) for v in "abcde")
    seed = MergeNode(children=[a, b, c], weights=normalize([1, 3, 2]))
    step2 = MergeNode(children=[seed, d], weights=[0.4, 0.6])
    step3 = MergeNode(children=[step2, e], weights=[0.6, 0.4])
    weights = final_weights(step3)

    first = np.array([1, 3, 2]) / 6
    expected = {
        "a": first[0] * 0.4 * 0.6, "b": first[1] * 0.4 * 0.6,
        "c": first[2] * 0.4 * 0.6, "d": 0.6 * 0.6, "e": 0.4,
    }
    for key, value in expected.items():
        assert weights[key] == pytest.approx(value, abs=1e-9)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_final_weights_rejects_bad_tree():
    bad = MergeNode(children=[MergeNode(voice_id="a"), MergeNode(voice_id="b")],
                    weights=[0.5, 0.9])
    with pytest.raises(AssertionError):
        final_weights(bad)


def test_rng_is_deterministic_and_seed_sensitive():
    seq1 = [DeterministicRNG("seed-1").randint(1, 100) for _ in range(1)]
    seq2 = [DeterministicRNG("seed-1").randint(1, 100) for _ in range(1)]
    assert seq1 == seq2
    a = DeterministicRNG("x")
    b = DeterministicRNG("y")
    assert [a.randint(1, 10**6) for _ in range(20)] != [b.randint(1, 10**6) for _ in range(20)]


def test_rng_choice_and_weights():
    rng = DeterministicRNG("t")
    assert rng.choice(["only"]) == "only"
    for _ in range(30):
        draw = rng.weights(3)
        assert len(draw) == 3
        assert set(draw) <= {1, 2, 3}
        assert len(set(draw)) > 1  # 완전 동률은 나오지 않는다


def test_sample_recipe_respects_pool_and_weights_sum():
    pool = [f"v{i}" for i in range(5)]
    for attempt in range(50):
        rng = DeterministicRNG(f"r|{attempt}")
        picked, tree, steps = sample_recipe(rng, pool, pick_min=3, pick_max=5)
        assert 3 <= len(picked) <= 5
        assert len(set(picked)) == len(picked)
        assert set(final_weights(tree)) == set(picked)
        assert sum(final_weights(tree).values()) == pytest.approx(1.0)
        assert steps[0]["kind"] == "seed" and len(steps[0]["items"]) >= 2


def test_sample_recipe_is_reproducible():
    a = sample_recipe(DeterministicRNG("same"), ["a", "b", "c", "d"])
    b = sample_recipe(DeterministicRNG("same"), ["a", "b", "c", "d"])
    assert a[1].to_dict() == b[1].to_dict()
    assert a[2] == b[2]


def test_sample_recipe_needs_two_voices():
    with pytest.raises(ValueError):
        sample_recipe(DeterministicRNG("x"), ["a"])


def test_gate_flags_single_voice_dominance():
    features = {v: np.array([i * 1.0] + [0, 0, 0, 0, 0, 0, 0, 0]) for i, v in enumerate("ab")}
    node = MergeNode(children=[MergeNode(voice_id="a"), MergeNode(voice_id="b")],
                     weights=[0.9, 0.1])
    report = check_recipe(node, features, GateConfig())
    assert not report.passed
    assert "최대 지분" in report.reason


def test_gate_flags_near_duplicates():
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    z = {"a": base, "b": base * 1.0001}  # 거의 같은 '모양' → 상관·코사인 모두 1에 근접
    node = MergeNode(children=[MergeNode(voice_id="a"), MergeNode(voice_id="b")],
                     weights=[0.5, 0.5])
    report = check_recipe(node, z, GateConfig(), center=np.zeros(len(FEATURE_NAMES)))
    assert not report.passed
    assert any("유사 음성쌍" in check["check"] for check in report.checks if not check["ok"])


def test_gate_warns_on_tiny_share_but_passes():
    features = {"a": np.array([1.0, -2, 3, -4, 5, -6, 7, -8, 9]),
                "b": np.array([-3.0, 4, -5, 6, -7, 8, -9, 10, -11]),
                "c": np.array([2.0, 5, -1, 0, 3, -2, 1, 4, -3])}
    node = MergeNode(children=[MergeNode(voice_id="a"), MergeNode(voice_id="b"),
                               MergeNode(voice_id="c")],
                     weights=normalize([60, 36, 4]))
    report = check_recipe(node, features, GateConfig(), center=np.zeros(len(FEATURE_NAMES)))
    assert report.passed
    assert report.warnings and "최소 지분" in report.warnings[0]


def test_blend_features_is_weighted_average():
    feats = {"a": np.array([2.0] + [0] * (len(FEATURE_NAMES) - 1)),
             "b": np.array([6.0] + [0] * (len(FEATURE_NAMES) - 1))}
    node = MergeNode(children=[MergeNode(voice_id="a"), MergeNode(voice_id="b")],
                     weights=[0.25, 0.75])
    assert blend_features(feats, node)[0] == pytest.approx(5.0)


def test_standardizer_roundtrip_and_scaling():
    matrix = np.random.default_rng(0).normal(5, 2, size=(12, len(FEATURE_NAMES)))
    std = Standardizer.fit(matrix)
    z = std.transform(matrix)
    assert z.shape == matrix.shape
    assert np.all(np.abs(z) <= 3.0 + 1e-9)
    assert np.abs(np.median(z, axis=0)).max() < 0.5
    again = Standardizer.from_dict(std.to_dict()).transform(matrix)
    assert np.allclose(z, again)


def test_as_vector_rejects_missing_axis():
    payload = {name: 1.0 for name in FEATURE_NAMES}
    payload.pop("brightness")
    with pytest.raises(KeyError):
        as_vector(payload)
    ok = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    mapping = as_mapping(as_vector(ok))
    assert all(mapping[name] == pytest.approx(ok[name]) for name in FEATURE_NAMES)
