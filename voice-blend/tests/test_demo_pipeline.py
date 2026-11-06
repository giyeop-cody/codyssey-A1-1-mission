"""엔드투엔드: 데모 스크립트가 실제로 리포트·계획·상품 문구를 만드는지 확인한다."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_demo_script_produces_report_and_plan(tmp_path):
    from scripts import demo_plan

    exit_code = demo_plan.main(["--out", str(tmp_path), "--blends", "2"])
    assert exit_code == 0

    report = tmp_path / "REPORT.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "합성 신호" in text          # 데모임을 숨기지 않는다
    assert "랜덤 경로" in text
    assert "품질 게이트" in text

    plan = json.loads((tmp_path / "plan" / "plan.json").read_text(encoding="utf-8"))
    assert len(plan["blends"]) == 2
    for blend in plan["blends"]:
        assert blend["gate"]["passed"]
        assert 4 <= len(blend["songs"]) <= 5
        assert 5 <= len(blend["dialogues"]) <= 6
        assert (tmp_path / "plan" / "recipes" / f"{blend['recipe_id']}.json").exists()

    products = list(tmp_path.glob("product_*.md"))
    assert len(products) == 2
    body = products[0].read_text(encoding="utf-8")
    assert "AI" in body and "라이선스" in body and "실제 인물" in body


def test_demo_script_is_reproducible(tmp_path):
    from scripts import demo_plan

    first, second = tmp_path / "a", tmp_path / "b"
    assert demo_plan.main(["--out", str(first), "--blends", "1"]) == 0
    assert demo_plan.main(["--out", str(second), "--blends", "1"]) == 0
    a = json.loads((first / "plan" / "plan.json").read_text(encoding="utf-8"))
    b = json.loads((second / "plan" / "plan.json").read_text(encoding="utf-8"))
    assert a["blends"][0]["final_weights"] == b["blends"][0]["final_weights"]
    assert a["clusters"] == b["clusters"]
