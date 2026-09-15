"""게이트 테스트의 실행 전략.

- 기본: 구현이 없으면 **skip**(레포가 초록으로 보이지만 README가 '몇 건이 skip인지'를 말한다).
- `MISSION_STRICT=1 pytest`: skip → fail. 제출 직전에는 반드시 이 모드(미구현을 남기지 않기 위한 장치).
- '통과하지만 무의미한 테스트'를 피하려고, 모든 게이트는 **측정 대상 파일**이 있을 때만 실행된다.
"""
import os
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STRICT = os.environ.get("MISSION_STRICT") == "1"


def implemented() -> bool:
    """미션 코드가 한 줄이라도 있는가. 없으면 게이트 검사는 판정할 대상이 없다."""
    return (ROOT / "reports" / "gates.json").exists() or any((ROOT / d).exists() for d in ("src", "figures", "notebooks"))


def gate_record(gid: str):
    """게이트 판정은 '이 레포가 **재서**한 값'으로만 내린다.

    reports/gates.json 형식:  {"G1": {"value": "IoU 52.4%", "pass": true, "how": "evaluate.py"}}
    이 파일이 없거나 항목이 없으면 실패다. '검사했는데 아무것도 말하지 않는' 상태를 막는 계약이다.
    """
    import json
    f = ROOT / "reports" / "gates.json"
    if not f.exists():
        raise AssertionError(f"{gid}: reports/gates.json 이 없다 → 게이트별 실측값을 먼저 기록하라")
    d = json.loads(f.read_text(encoding="utf-8"))
    if gid not in d:
        raise AssertionError(f"{gid}: gates.json 에 항목이 없다(키 목록: {sorted(d)})")
    rec = d[gid]
    assert set(("value", "pass")) <= set(rec), f"{gid}: {{'value':…,'pass':…}} 형식이어야 한다"
    assert rec["pass"] is True, f"{gid}: 통과 실패 — 측정값 {rec['value']}"
    return rec


def needs(*rel: str):
    """게이트의 증거 파일 기준 마크.

    - 미구현 + 기본        → skip  (초록이지만 README가 skip 수를 말한다)
    - 미구현 + STRICT      → 실행 → 본문 assert 가 **실패**한다(미구현을 남기지 않음)
    - 구현됨 + 증거 없음   → skip / STRICT이면 실패
    - 구현됨 + 증거 있음   → 실행
    """
    missing = [r for r in rel if not (ROOT / r).exists()]
    if not implemented():
        return pytest.mark.skipif(not STRICT, reason="미션 미구현 — 구현 후 실행")
    if missing:
        return pytest.mark.skipif(not STRICT, reason="산출물 없음: " + ", ".join(missing))
    return pytest.mark.skipif(False, reason="")
