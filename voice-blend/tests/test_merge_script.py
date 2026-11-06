"""생성된 RVC 병합 스크립트를 **가짜 torch**로 실제 실행해 수학을 검증한다.

GPU/torch 없이도 "트리 → 최종 가중치 → 텐서 선형 결합"이 맞는지 확인할 수 있다.
가짜 텐서는 numpy 배열을 감싸고, 스크립트가 쓰는 최소 인터페이스만 구현한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types

import numpy as np
import pytest

from voice_fusion.rvc import write_merge_script


class FakeTensor:
    def __init__(self, array, dtype="float32"):
        self.array = np.asarray(array, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def float(self):
        return FakeTensor(self.array.astype(np.float32), "float32")

    def to(self, dtype):
        return FakeTensor(self.array.astype(np.float32), dtype)

    def __mul__(self, other):
        return FakeTensor(self.array * other, "float32")

    def __add__(self, other):
        return FakeTensor(self.array + other.array, "float32")


def _install_fake_torch(monkeypatch):
    module = types.ModuleType("torch")

    def save(payload, path):
        with open(path, "wb") as handle:
            np.save(handle, np.array([1]))  # 자리표시자
        with open(str(path) + ".json", "w", encoding="utf-8") as handle:
            json.dump({key: value.array.tolist() for key, value in payload["model"].items()},
                      handle)

    def load(path, map_location=None):
        with open(str(path) + ".json", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {"model": {key: FakeTensor(np.array(value)) for key, value in raw.items()}}

    module.save = save
    module.load = load
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def _write_ckpt(directory, name, value):
    (directory / f"{name}.pth.json").write_text(
        json.dumps({"enc.weight": [[value, value + 1]], "dec.bias": [value]}),
        encoding="utf-8")
    (directory / f"{name}.pth").write_text("placeholder", encoding="utf-8")
    return str(directory / f"{name}.pth")


def _load_script(path):
    spec = importlib.util.spec_from_file_location("merge_recipe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_script_applies_final_weights(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    script = _load_script(write_merge_script(tmp_path / "run"))
    a = _write_ckpt(tmp_path, "a", 0.0)
    b = _write_ckpt(tmp_path, "b", 10.0)
    recipe = {
        "recipe_id": "blend-test",
        "name": "test",
        "final_weights": {"a": 0.75, "b": 0.25},
        "checkpoints": {"a": a, "b": b},
        "gate": {"passed": True},
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    target = script.merge(recipe_path, tmp_path / "out", tmp_path / "out")
    with open(str(target) + ".json", encoding="utf-8") as handle:
        merged = json.load(handle)
    assert merged["enc.weight"][0][0] == pytest.approx(0.75 * 0.0 + 0.25 * 10.0, rel=1e-6)
    assert merged["dec.bias"][0] == pytest.approx(2.5, rel=1e-6)


def test_merge_script_rejects_unbalanced_weights(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    script = _load_script(write_merge_script(tmp_path / "run"))
    a = _write_ckpt(tmp_path, "a", 0.0)
    recipe = {"recipe_id": "bad", "final_weights": {"a": 0.9}, "checkpoints": {"a": a}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(SystemExit):
        script.merge(path, tmp_path / "out", tmp_path / "out")


def test_merge_script_rejects_key_mismatch(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    script = _load_script(write_merge_script(tmp_path / "run"))
    a = _write_ckpt(tmp_path, "a", 0.0)
    b = tmp_path / "b.pth"
    b.write_text("placeholder", encoding="utf-8")
    (tmp_path / "b.pth.json").write_text(json.dumps({"other.weight": [[1.0]]}), encoding="utf-8")
    recipe = {"recipe_id": "mismatch", "final_weights": {"a": 0.5, "b": 0.5},
              "checkpoints": {"a": a, "b": str(b)}}
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(SystemExit):
        script.merge(path, tmp_path / "out", tmp_path / "out")
