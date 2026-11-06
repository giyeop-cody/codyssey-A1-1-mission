#!/usr/bin/env bash
# 제출 전 자체 점검: 테스트 + 데모 재생성 + 산출물 존재 확인
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1) 테스트 =="
python3 -m pytest -q

echo "== 2) 데모 재생성 =="
python3 scripts/demo_plan.py > /dev/null

echo "== 3) 산출물 확인 =="
test -f artifacts/demo/REPORT.md
test -f artifacts/demo/plan/plan.json
ls artifacts/demo/product_*.md > /dev/null
python3 - <<'PY'
import json, pathlib
plan = json.loads(pathlib.Path("artifacts/demo/plan/plan.json").read_text(encoding="utf-8"))
for blend in plan["blends"]:
    shares = blend["final_weights"].values()
    assert abs(sum(shares) - 1) < 1e-6, blend["recipe_id"]
    assert max(shares) <= 0.62 + 1e-9, blend["recipe_id"]
    assert 4 <= len(blend["songs"]) <= 5 and 5 <= len(blend["dialogues"]) <= 6
    assert blend["gate"]["passed"]
print(f"OK — 병합본 {len(plan['blends'])}개, 군집 k={plan['clusters']['k']}, "
      f"실루엣 {plan['clusters']['silhouette']}")
PY
