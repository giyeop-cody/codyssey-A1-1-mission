"""Kaggle 원본 H&M 데이터를 사용자의 Kaggle CLI 인증으로 다운로드한다.

사전 조건:
  1. Kaggle competition 규칙에 사용자가 직접 동의
  2. pip install -r requirements-original.txt
  3. 이 스크립트가 실행하는 브라우저 OAuth `kaggle auth login --force` 완료

API 토큰을 요구하거나 사용하지 않는다. 다운로드 직전에 공식 CLI 로그인을 실행하며,
토큰용 환경변수는 subprocess에서 제거해 OAuth 로그인으로만 인증한다.

전체 멀티모달 분석(원본 CSV + 이미지):
  python scripts/fetch_kaggle_original.py --out data/_kaggle --include-images

CSV만 받으면 이미지 분석은 실행할 수 없다. 인증 정보는 인자·파일·로그에 저장하지
않고 공식 Kaggle CLI의 브라우저 OAuth에 맡긴다. data/는 Git에 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPETITION = "h-and-m-personalized-fashion-recommendations"
SOURCE_URL = f"https://www.kaggle.com/competitions/{COMPETITION}"
RULES_URL = f"{SOURCE_URL}/rules"
CSV_FILES = ("articles.csv", "customers.csv", "transactions_train.csv")
TOKEN_ENV_VARS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_download_command(executable: str, download_dir: pathlib.Path,
                           filename: str | None = None) -> list[str]:
    # Kaggle CLI 2.2.5는 competition을 위치 인자로 받는다(과거 `-c` 옵션 아님).
    command = [executable, "competitions", "download", COMPETITION, "-p", str(download_dir)]
    if filename is not None:
        command += ["-f", filename]
    return command


def run_kaggle(command: list[str]) -> None:
    """공식 CLI만 실행한다. 토큰 환경변수는 제거하고 인증 값을 로그에 남기지 않는다."""
    oauth_env = os.environ.copy()
    for name in TOKEN_ENV_VARS:
        oauth_env.pop(name, None)
    subprocess.run(command, check=True, env=oauth_env)


def build_login_command(executable: str) -> list[str]:
    # Kaggle CLI 2.2.5는 기존 로그인이 있으면 --force 없이 종료 코드 1을 반환한다.
    # 재현 명령을 항상 같은 OAuth 흐름으로 시작하기 위해 브라우저 로그인을 다시 확인한다.
    return [executable, "auth", "login", "--force"]


def require_kaggle_login() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "Kaggle CLI가 없습니다. `pip install -r requirements-original.txt` 후 다시 실행하세요. "
            "브라우저 OAuth 로그인과 competition 규칙 동의가 필요합니다."
        )
    run_kaggle(build_login_command(executable))
    return executable


def validate_zip_members(archive: zipfile.ZipFile, destination: pathlib.Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe zip member path: {member.filename}")


def extract_zip(zip_path: pathlib.Path, destination: pathlib.Path, *,
                only_file: str | None = None, test_zip: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        validate_zip_members(archive, destination)
        if test_zip:
            damaged = archive.testzip()
            if damaged is not None:
                raise ValueError(f"CRC verification failed: {damaged}")
        if only_file is None:
            archive.extractall(destination)
            return
        matches = [name for name in archive.namelist()
                   if name == only_file or name.endswith(f"/{only_file}")]
        if len(matches) != 1:
            raise ValueError(f"expected one {only_file} in {zip_path}, found {matches}")
        target = destination / only_file
        with archive.open(matches[0]) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def downloaded_zip(download_dir: pathlib.Path, filename: str) -> pathlib.Path:
    expected = download_dir / f"{filename}.zip"
    if expected.exists():
        return expected
    candidates = sorted(download_dir.glob(f"{filename}*.zip"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Kaggle download zip not found for {filename}: {candidates}")
    return candidates[0]


def fetch_csv_files(out: pathlib.Path, *, force: bool, keep_zip: bool,
                    test_zip: bool) -> list[pathlib.Path]:
    executable = require_kaggle_login()
    download_dir = out / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for filename in CSV_FILES:
        target = out / filename
        if target.exists() and not force:
            extracted.append(target)
            continue
        zip_path = download_dir / f"{filename}.zip"
        if force or not zip_path.exists():
            run_kaggle(build_download_command(executable, download_dir, filename))
        zip_path = downloaded_zip(download_dir, filename)
        extract_zip(zip_path, out, only_file=filename, test_zip=test_zip)
        if not target.exists():
            raise FileNotFoundError(f"압축 해제 후에도 원본 파일이 없습니다: {target}")
        if not keep_zip:
            zip_path.unlink()
        extracted.append(target)
    return extracted


def count_images(image_dir: pathlib.Path) -> int:
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.rglob("*.jpg") if path.is_file())


def fetch_full_bundle(out: pathlib.Path, *, force: bool, keep_zip: bool,
                      test_zip: bool) -> list[pathlib.Path]:
    executable = require_kaggle_login()
    download_dir = out / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"{COMPETITION}.zip"
    complete = all((out / name).exists() for name in CSV_FILES) and count_images(out / "images") > 0
    if force or not complete:
        if force or not zip_path.exists():
            run_kaggle(build_download_command(executable, download_dir))
            if not zip_path.exists():
                candidates = sorted(download_dir.glob("*.zip"))
                if len(candidates) != 1:
                    raise FileNotFoundError(f"전체 competition zip을 찾지 못했습니다: {candidates}")
                zip_path = candidates[0]
        extract_zip(zip_path, out, test_zip=test_zip)
        if not keep_zip:
            zip_path.unlink()
    missing = [name for name in CSV_FILES if not (out / name).exists()]
    if missing:
        raise FileNotFoundError(f"원본 CSV 누락: {missing}")
    if count_images(out / "images") < 1000:
        raise RuntimeError("원본 images/에 분석 가능한 JPG가 1,000장 미만입니다.")
    return [out / name for name in CSV_FILES]


def write_manifest(out: pathlib.Path, files: list[pathlib.Path], *,
                   include_images: bool, test_zip: bool) -> dict:
    image_count = count_images(out / "images") if include_images else 0
    manifest = {
        "dataset": "H&M Personalized Fashion Recommendations — Kaggle original user download",
        "source": SOURCE_URL, "rules": RULES_URL, "competition": COMPETITION,
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "authentication": "Kaggle CLI browser OAuth: kaggle auth login --force; no API token arguments",
        "include_images": include_images, "zip_crc_tested": test_zip,
        "license_scope": "Non-Commercial Purposes & Academic Research; do not commit or redistribute raw data/images",
        "files": {},
        "image_count": image_count,
        "notes": [
            "This manifest is local provenance, not an independent official Kaggle checksum.",
            "prepare_hm_data.py verifies these hashes again before constructing an analysis subset.",
            "Full-source results differ from the committed 15-day GitHub-mirror results.",
        ],
    }
    for path in files:
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    target = out / "kaggle_original_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "_kaggle")
    parser.add_argument("--include-images", action="store_true",
                        help="전체 competition bundle(약 34.56GB)과 원본 이미지까지 받는다")
    parser.add_argument("--keep-zip", action="store_true", help="압축 해제 후 다운로드 zip도 보존")
    parser.add_argument("--test-zip", action="store_true", help="압축 해제 전 전체 CRC 검사(대용량에서는 시간 증가)")
    parser.add_argument("--force", action="store_true", help="기존 파일이 있어도 다시 다운로드·압축 해제")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.include_images:
        free_gb = shutil.disk_usage(out).free / 1e9
        if free_gb < 80:
            print(f"WARNING: 여유 디스크 {free_gb:.1f}GB. zip+압축 해제 작업 공간은 80GB 이상을 권장합니다.", file=sys.stderr)
    try:
        if args.include_images:
            files = fetch_full_bundle(out, force=args.force, keep_zip=args.keep_zip, test_zip=args.test_zip)
        else:
            files = fetch_csv_files(out, force=args.force, keep_zip=args.keep_zip, test_zip=args.test_zip)
        manifest = write_manifest(out, files, include_images=args.include_images, test_zip=args.test_zip)
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: Kaggle 다운로드 실패(exit {exc.returncode}). 브라우저 OAuth 로그인, competition 규칙 동의, "
            "네트워크·할당량을 확인하세요. 로그인은 공식 Kaggle CLI에서 직접 완료해야 합니다.", file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\n다음 명령으로 실제 원본에서 분석 입력을 만드세요:")
    if args.include_images:
        print(f"python scripts/prepare_hm_data.py --source-dir {out} --image-dir {out / 'images'}")
    else:
        print("CSV만 있으므로 RFM은 가능하지만 전체 멀티모달 분석에는 --include-images 다운로드가 필요합니다.")


if __name__ == "__main__":
    main()
