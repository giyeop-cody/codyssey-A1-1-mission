"""요구 검수 스크립트(scripts/check_requirements.py)를 그대로 테스트로 실행한다.

동일 로직을 두 번 만들지 않기 위해, 통과 조건은 스크립트의 exit code 로만 판정한다.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ensure_inputs():
    """클론 직후에는 data/·reports/metrics.json 이 없다(둘 다 gitignore: 재생 가능한 산출물).
    그래서 검사 전에 필요한 것만 만들어 실행 순서 의존을 없앤다. 없으면 만들 수 없는 레포(미구현)는 skip."""
    if not (ROOT / "src" / "data_gen.py").exists():
        pytest.skip("src/data_gen.py 가 없는 레포(미구현) — 데이터 생성 불가")
    if not (ROOT / "data" / "products.csv").exists():
        subprocess.run([sys.executable, "src/data_gen.py", "--seed", "42"], cwd=ROOT, check=True,
                       capture_output=True, text=True)
    if (ROOT / "scripts" / "run_analysis.py").exists() and not (ROOT / "reports" / "metrics.json").exists():
        subprocess.run([sys.executable, "scripts/run_analysis.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)
    if (ROOT / "scripts" / "record_gates.py").exists():
        subprocess.run([sys.executable, "scripts/record_gates.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)


def test_check_requirements_ALL_PASS():
    _ensure_inputs()
    r = subprocess.run([sys.executable, "scripts/check_requirements.py"], cwd=ROOT,
                       capture_output=True, text=True)
    lines = r.stdout.splitlines()
    assert r.returncode == 0, "자가채점 실패:\n" + "\n".join(lines[-12:])
    assert any(l.startswith("---- ") and "PASS" in l for l in lines), "채점 요약이 없다 = 검사가 죽은 것"
