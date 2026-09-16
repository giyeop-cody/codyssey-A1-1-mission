"""요구 검수 스크립트(scripts/check_requirements.py)를 그대로 테스트로 실행한다.

동일 로직을 두 번 만들지 않기 위해, 통과 조건은 스크립트의 exit code 로만 판정한다.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


# bootstrap 은 conftest.ensure_inputs() 가 autouse 로 이미 한다(아래 호출은 자기 문서화용).
from conftest import ensure_inputs as _ensure_inputs  # noqa: E402


def test_check_requirements_ALL_PASS():
    _ensure_inputs()
    r = subprocess.run([sys.executable, "scripts/check_requirements.py"], cwd=ROOT,
                       capture_output=True, text=True)
    lines = r.stdout.splitlines()
    assert r.returncode == 0, "자가채점 실패:\n" + "\n".join(lines[-12:])
    assert any(l.startswith("---- ") and "PASS" in l for l in lines), "채점 요약이 없다 = 검사가 죽은 것"
