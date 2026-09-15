"""레포 꼴 검사 — 항상 실행된다(구현 여부와 무관하게 지켜야 할 것)."""
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_READM_존재하고_실행법포함():
    p = ROOT / "README.md"
    assert p.exists(), "README.md 필요"
    t = p.read_text(encoding="utf-8")
    assert "실행" in t, "README에 실행 방법이 있어야 한다"


def test_학습맵_존재():
    p = ROOT / "docs" / "learning-map.md"
    assert p.exists() and "게이트" in p.read_text(encoding="utf-8")


def test_데이터를_커밋하지_않는다():
    """요구: csv/그림 산출물은 재생 가능해야 하므로 git 추적 대상이 아니다(A5-1 규정의 일반화)."""
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    assert "*.csv" in gi or "data/" in gi, ".gitignore 에 데이터 제외 규칙이 필요하다"


def test_시드_상수():
    """시드 리터럴은 한 곳(CLI 기본값 또는 config)에만 있어야 한다 — 재현성의 최소 조건."""
    import re
    roots = [d for d in (ROOT / "src", ROOT / "scripts") if d.exists()]
    if not roots:
        pytest.skip("코드가 아직 없다")
    bad = []
    for f in [p for d in roots for p in d.rglob("*.py")]:
        for ln in f.read_text(encoding="utf-8").splitlines():
            if re.search(r"(?<![\w_])seed\s*=\s*\d", ln) and not re.search(r"default_rng|add_argument", ln):
                bad.append(f"{f.name}: {ln.strip()[:48]}")
    assert not bad, "시드가 여러 곳에 리터럴로 박혔다: " + "; ".join(bad[:3])
