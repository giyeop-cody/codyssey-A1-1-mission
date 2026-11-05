"""실데이터 준비부터 분석·노트북·검사까지 순서대로 실행한다."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def execute(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--download-mirror", action="store_true",
                        help="커밋된 15일 결과와 같은 고정 공개 미러에서 준비(실행 검증 완료)")
    source.add_argument("--download-original", action="store_true",
                        help="Kaggle 브라우저 OAuth 로그인으로 전체 원본 CSV·이미지 다운로드 후 준비(대용량)")
    parser.add_argument("--n-customers", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args(argv)
    required = [ROOT / "data/products.csv", ROOT / "data/transactions.csv", ROOT / "data/images.npy"]
    if args.download_original:
        original = ROOT / "data" / "_kaggle"
        execute("fetch_kaggle_original.py", "--out", str(original), "--include-images")
        execute("prepare_hm_data.py", "--source-dir", str(original), "--image-dir", str(original / "images"),
                "--n-customers", str(args.n_customers), "--seed", str(args.seed))
    elif args.download_mirror or not all(path.exists() for path in required):
        if not args.download_mirror:
            parser.error("data가 없습니다. --download-mirror 또는 --download-original을 사용하세요.")
        execute("prepare_hm_data.py", "--download-mirror", "--n-customers", str(args.n_customers), "--seed", str(args.seed))
    execute("run_analysis.py")
    execute("spec_curve.py")
    execute("build_notebook.py")
    execute("check_requirements.py")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
