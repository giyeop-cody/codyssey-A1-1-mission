"""H&M 실제 공개 데이터에서 A1-1 입력을 재현 가능하게 만든다.

권장 입력은 이용자가 Kaggle 규칙에 동의하고 직접 다운로드한 원본이다. 이 실행 환경에서는
Kaggle/Hugging Face 직접 연결이 차단되어 동일 원본의 공개 GitHub 미러를 사용했다.
원본 CSV·이미지와 파생 입력 data/는 Git에 커밋하지 않는다.

예시(공개 미러, 교육·비상업 용도):
    python scripts/prepare_hm_data.py --download-mirror --n-customers 1000

예시(Kaggle 원본):
    python scripts/prepare_hm_data.py --source-dir /path/to/hm --image-dir /path/to/hm/images

표본 규칙: SHA-256(f"{seed}:{customer_id}")의 오름차순으로 고객을 선택하고,
그 고객들의 기간 내 모든 구매 행을 유지한다. 이미지가 실제로 존재하는 상품만 남긴다.
가격은 선택된 실거래의 article_id별 중앙값이며 원화로 변환하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIRROR_REPO = "Gott-Data/H_and_M_Recommendation_Streamlit"
MIRROR_COMMIT = "a3184bf24dbb78c67dd9863dd0aebb93e1b9b462"
MIRROR_URL = f"https://github.com/{MIRROR_REPO}"
KAGGLE_URL = "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations"
RULES_URL = f"{KAGGLE_URL}/rules"
EXPECTED_SHA256 = {
    "articles.csv": "25f21882053517f7684ea43f524225a9481a7bb9aab9cf0abfc7fe8c4576a72b",
    "new_transactions.csv": "c0098b8215693aa1bcae386e86e27299b41aa3f7ffada685345e509b7eb3a44e",
}
IMAGE_RE = re.compile(r"^images_256_256/[0-9]{3}/[0-9]{10}\.jpg$")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def customer_key(customer_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{customer_id}".encode("utf-8")).hexdigest()


def run(command: list[str], *, cwd: pathlib.Path | None = None,
        stdout=None, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE if stdout is None else stdout,
        stderr=subprocess.PIPE, text=True, input=input_text, check=True,
    )


def download_mirror_csv(source_dir: pathlib.Path) -> None:
    """공개 미러 CSV를 고정 커밋에서 받고 알려진 SHA-256과 대조한다."""
    if shutil.which("gh") is None:
        raise RuntimeError("미러 다운로드에는 GitHub CLI(gh)가 필요합니다. Kaggle 원본 --source-dir를 사용할 수도 있습니다.")
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED_SHA256.items():
        target = source_dir / name
        if target.exists() and sha256_file(target) == expected:
            continue
        temp = target.with_suffix(target.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            endpoint = f"repos/{MIRROR_REPO}/contents/{name}?ref={MIRROR_COMMIT}"
            run(["gh", "api", endpoint, "-H", "Accept: application/vnd.github.raw+json"], stdout=handle)
        actual = sha256_file(temp)
        if actual != expected:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"{name} SHA-256 불일치: expected={expected}, actual={actual}")
        temp.replace(target)


def ensure_image_mirror(repo: pathlib.Path) -> None:
    """이미지 blob은 선택 전까지 받지 않는 partial clone을 만든다."""
    if not (repo / ".git").exists():
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(f"이미지 저장소 경로가 비어 있지 않지만 Git 저장소가 아닙니다: {repo}")
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", "--no-tags", MIRROR_URL, str(repo)])
    # upstream main이 나중에 움직여도 고정 커밋만 사용한다. HEAD 전체를 checkout하면
    # 모든 이미지 blob을 받게 되므로 commit 객체 존재만 확인하고 선택 checkout은 뒤에서 한다.
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{MIRROR_COMMIT}^{{commit}}"], cwd=repo,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0
    if not exists:
        run(["git", "fetch", "origin", MIRROR_COMMIT], cwd=repo)


def mirror_image_paths(repo: pathlib.Path) -> dict[str, str]:
    listing = run(["git", "ls-tree", "-r", "--name-only", MIRROR_COMMIT, "--", "images_256_256"], cwd=repo).stdout
    paths = sorted(path for path in listing.splitlines() if IMAGE_RE.fullmatch(path))
    result = {pathlib.PurePosixPath(path).stem: path for path in paths}
    if len(result) < 1000:
        raise RuntimeError(f"사용 가능한 실제 이미지가 부족합니다: {len(result)}")
    return result


def local_image_paths(image_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for path in image_dir.rglob("*.jpg"):
        if re.fullmatch(r"[0-9]{10}", path.stem):
            result[path.stem] = path
    if len(result) < 1000:
        raise RuntimeError(f"사용 가능한 실제 이미지가 부족합니다: {image_dir} ({len(result)}장)")
    return result


def checkout_selected_images(repo: pathlib.Path, selected: list[str]) -> None:
    """선택한 파일만 lazy-fetch 한다. 저장소의 나머지 10만여 blob은 받지 않는다."""
    run(["git", "sparse-checkout", "init", "--no-cone"], cwd=repo)
    run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"], cwd=repo,
        input_text="\n".join(selected) + "\n")
    run(["git", "read-tree", "-mu", MIRROR_COMMIT], cwd=repo)


def read_transactions(source_dir: pathlib.Path) -> tuple[pd.DataFrame, str, str]:
    mirror_file = source_dir / "new_transactions.csv"
    original_file = source_dir / "transactions_train.csv"
    if mirror_file.exists():
        path, source_kind = mirror_file, "github_mirror_tail_window"
        actual = sha256_file(path)
        if actual != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"{path.name} SHA-256 불일치: {actual}")
    elif original_file.exists():
        path, source_kind = original_file, "kaggle_original_user_supplied"
    else:
        raise FileNotFoundError(f"{source_dir}에 new_transactions.csv 또는 transactions_train.csv가 없습니다.")
    frame = pd.read_csv(path, dtype={"customer_id": "string", "article_id": "string"})
    required = {"t_dat", "customer_id", "article_id", "price", "sales_channel_id"}
    if not required <= set(frame.columns):
        raise ValueError(f"거래 필수 컬럼 부족: {required - set(frame.columns)}")
    if "Unnamed: 0" in frame:
        frame["source_row_id"] = pd.to_numeric(frame["Unnamed: 0"], downcast="integer")
    else:
        frame["source_row_id"] = np.arange(len(frame), dtype=np.int64)
    frame["article_id"] = frame["article_id"].str.zfill(10)
    frame["t_dat"] = pd.to_datetime(frame["t_dat"], errors="raise")
    frame["price"] = pd.to_numeric(frame["price"], errors="raise")
    if frame[["customer_id", "article_id", "price"]].isna().any().any():
        raise ValueError("거래의 customer_id/article_id/price에 결측이 있어 표본 규칙을 적용할 수 없습니다.")
    if (frame["price"] <= 0).any():
        raise ValueError("H&M 정규화 가격은 양수여야 합니다.")
    return frame, source_kind, path.name


def prepare(args: argparse.Namespace) -> dict:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir.resolve()
    if args.download_mirror:
        download_mirror_csv(source_dir)
        if args.image_repo is None:
            args.image_repo = source_dir / "image_mirror"

    tx, source_kind, transaction_filename = read_transactions(source_dir)
    transaction_hash = sha256_file(source_dir / transaction_filename)
    articles_path = source_dir / "articles.csv"
    if not articles_path.exists():
        raise FileNotFoundError(articles_path)
    articles_hash = sha256_file(articles_path)
    if source_kind == "github_mirror_tail_window" and articles_hash != EXPECTED_SHA256["articles.csv"]:
        raise RuntimeError(f"articles.csv SHA-256 불일치: {articles_hash}")
    original_manifest_path = source_dir / "kaggle_original_manifest.json"
    if source_kind == "kaggle_original_user_supplied" and original_manifest_path.exists():
        original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
        expected_files = original_manifest.get("files", {})
        for filename, actual_hash in [(transaction_filename, transaction_hash), ("articles.csv", articles_hash)]:
            expected = expected_files.get(filename, {}).get("sha256")
            if expected is not None and expected != actual_hash:
                raise RuntimeError(f"Kaggle 다운로드 manifest와 {filename} 해시가 다릅니다: {actual_hash}")
    articles = pd.read_csv(articles_path, dtype={"article_id": "string"})
    articles["article_id"] = articles["article_id"].str.zfill(10)
    if articles["article_id"].duplicated().any():
        raise ValueError("articles.csv의 article_id가 중복됩니다.")

    image_repo = args.image_repo.resolve() if args.image_repo else None
    if image_repo is not None:
        ensure_image_mirror(image_repo)
        available = mirror_image_paths(image_repo)
        local_images = None
    else:
        if args.image_dir is None:
            raise ValueError("--image-repo 또는 --image-dir 중 하나가 필요합니다.")
        local_images = local_image_paths(args.image_dir.resolve())
        available = {key: str(path) for key, path in local_images.items()}

    ranked_customers = sorted(
        (str(value) for value in tx["customer_id"].unique()),
        key=lambda value: (customer_key(value, args.seed), value),
    )
    if len(ranked_customers) < args.n_customers:
        raise ValueError(f"요청 고객 {args.n_customers}명 > 원본 고객 {len(ranked_customers)}명")
    selected_customers = set(ranked_customers[: args.n_customers])
    sampled = tx[tx["customer_id"].astype(str).isin(selected_customers)].copy()
    before_rows = len(sampled)
    before_products = sampled["article_id"].nunique()
    sampled = sampled[sampled["article_id"].isin(available)].copy()
    if len(sampled) < 1000 or sampled["article_id"].nunique() < 1000:
        raise RuntimeError(
            f"실데이터 표본이 요구보다 작습니다: {len(sampled)}거래/{sampled['article_id'].nunique()}상품. "
            "--n-customers를 늘리세요."
        )

    sampled["txn_id"] = sampled["source_row_id"].astype(np.int64) + 1
    sampled["product_id"] = sampled["article_id"]
    sampled["amount"] = sampled["price"].astype(float)
    sampled["order_date"] = sampled["t_dat"]
    sampled["txn_date"] = sampled["order_date"]
    sampled = sampled.sort_values(["order_date", "txn_id", "product_id"]).reset_index(drop=True)

    grouped = sampled.groupby("product_id", sort=True)
    products = grouped.agg(
        price=("amount", "median"),
        price_min=("amount", "min"),
        price_max=("amount", "max"),
        transaction_count=("txn_id", "size"),
        customer_count=("customer_id", "nunique"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
    ).reset_index()
    metadata_columns = [
        "article_id", "prod_name", "product_type_name", "product_group_name",
        "colour_group_name", "index_name", "department_name", "detail_desc",
    ]
    metadata = articles[metadata_columns].rename(columns={
        "article_id": "product_id", "prod_name": "name",
        "product_group_name": "category", "detail_desc": "description",
    })
    products = products.merge(metadata, on="product_id", how="left", validate="one_to_one")
    if products["name"].isna().any():
        raise ValueError("선택 거래와 상품 메타데이터의 article_id 조인에 실패했습니다.")
    products["image_path"] = products["product_id"].map(
        lambda value: f"images/{value[:3]}/{value}.jpg"
    )
    products = products.sort_values("product_id").reset_index(drop=True)

    selected_paths = [available[value] for value in products["product_id"]]
    if image_repo is not None:
        checkout_selected_images(image_repo, selected_paths)

    images = np.empty((len(products), 32, 32, 3), dtype=np.uint8)
    manifest_rows = []
    for index, (article_id, source_path) in enumerate(zip(products["product_id"], selected_paths)):
        path = image_repo / source_path if image_repo is not None else pathlib.Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"선택 이미지가 체크아웃되지 않았습니다: {path}")
        source_hash = sha256_file(path)
        with Image.open(path) as opened:
            original_size = list(opened.size)
            converted = ImageOps.exif_transpose(opened).convert("RGB")
            resized = converted.resize((32, 32), Image.Resampling.LANCZOS)
            images[index] = np.asarray(resized, dtype=np.uint8)
        manifest_rows.append({
            "product_id": article_id,
            "source_image_path": str(source_path),
            "source_image_sha256": source_hash,
            "source_width": original_size[0], "source_height": original_size[1],
        })

    product_columns = [
        "product_id", "name", "category", "product_type_name", "colour_group_name",
        "index_name", "department_name", "description", "price", "price_min", "price_max",
        "transaction_count", "customer_count", "first_order_date", "last_order_date", "image_path",
    ]
    transaction_columns = [
        "txn_id", "source_row_id", "product_id", "customer_id", "amount",
        "order_date", "txn_date", "sales_channel_id",
    ]
    products[product_columns].to_csv(out / "products.csv", index=False, date_format="%Y-%m-%d")
    sampled[transaction_columns].to_csv(out / "transactions.csv", index=False, date_format="%Y-%m-%d")
    np.save(out / "images.npy", images)
    pd.DataFrame(manifest_rows).to_csv(out / "image_manifest.csv", index=False)

    source_window_limit = (
        "RFM frequency counts transaction lines, not unique baskets. The source mirror covers only the last 15 days, so this is short-window RFM."
        if source_kind == "github_mirror_tail_window" else
        "RFM frequency counts transaction lines, not unique baskets. This selected-customer original-source sample is still observational and not a future-churn label."
    )
    population_limit = (
        "Product prices and image-price correlations describe this selected-customer subset, not the full 31.8M-row H&M population."
        if source_kind == "github_mirror_tail_window" else
        "Product prices and image-price correlations describe this selected-customer subset, not all H&M customers or products."
    )
    image_provenance = (
        "256x256 resized mirrors of the original H&M article images; this script only RGB-converts and resizes to 32x32"
        if image_repo is not None else
        "User-downloaded Kaggle original H&M article images; this script RGB-converts and resizes to 32x32"
    )
    meta = {
        "dataset": "H&M Personalized Fashion Recommendations — real public data, derived subset",
        "synthetic": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_source": KAGGLE_URL,
        "original_rules": RULES_URL,
        "license_scope": "Non-Commercial Purposes & Academic Research; raw data/images are not committed or redistributed here",
        "source_kind": source_kind,
        "mirror": {
            "repository": MIRROR_URL if source_kind == "github_mirror_tail_window" else None,
            "commit": MIRROR_COMMIT if image_repo is not None else None,
            "images": image_provenance,
        },
        "source_files": {
            "articles.csv": {"sha256": articles_hash, "rows": int(len(articles))},
            transaction_filename: {
                "sha256": transaction_hash, "rows": int(len(tx)),
                "date_min": str(tx["t_dat"].min().date()), "date_max": str(tx["t_dat"].max().date()),
                "tail_source_row_min": int(tx["source_row_id"].min()),
                "tail_source_row_max": int(tx["source_row_id"].max()),
            },
        },
        "sampling": {
            "seed": args.seed,
            "method": "lowest SHA-256(seed:customer_id), then all transactions for selected customers; retain products with a real available image",
            "requested_customers": args.n_customers,
            "source_customers": int(tx["customer_id"].nunique()),
            "before_image_filter_transactions": int(before_rows),
            "before_image_filter_products": int(before_products),
            "removed_transactions_without_image": int(before_rows - len(sampled)),
            "removed_products_without_image": int(before_products - sampled["article_id"].nunique()),
            "price_aggregation": "median observed transaction-line price per article_id within this selected-customer subset",
        },
        "result": {
            "n_products": int(len(products)), "n_transactions": int(len(sampled)),
            "n_customers": int(sampled["customer_id"].nunique()), "images_shape": list(images.shape),
            "date_min": str(sampled["order_date"].min().date()), "date_max": str(sampled["order_date"].max().date()),
            "missing_descriptions": int(products["description"].isna().sum()),
            "missing_prices": int(products["price"].isna().sum()),
            "transaction_article_join_coverage": float(sampled["product_id"].isin(set(products["product_id"])).mean()),
        },
        "units_and_limits": [
            "H&M prices are normalized/scaled values, not KRW or another identifiable currency; amount is a relative sales-value index.",
            "Each source row is one transaction line. Duplicate rows represent repeated purchases; no order/basket ID is supplied.",
            source_window_limit,
            "Customer histories are complete within the selected source window except rows whose product has no available image; image-filter counts are reported above.",
            population_limit,
            "No impressions, clicks, sessions, returns, stock, ratings, or product creation dates are present; no such values are fabricated.",
        ],
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path, default=ROOT / "data" / "_source")
    parser.add_argument("--image-repo", type=pathlib.Path, default=None,
                        help="공개 GitHub 이미지 미러의 partial clone 경로")
    parser.add_argument("--image-dir", type=pathlib.Path, default=None,
                        help="Kaggle 원본 images/ 디렉터리 (GitHub 미러 대신)")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "data")
    parser.add_argument("--n-customers", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--download-mirror", action="store_true",
                        help="고정 GitHub 미러에서 CSV와 선택 이미지를 받는다(비상업·교육 용도)")
    args = parser.parse_args()
    if args.n_customers < 1:
        parser.error("--n-customers는 양수여야 합니다.")
    if args.image_repo is not None and args.image_dir is not None:
        parser.error("--image-repo와 --image-dir은 동시에 지정할 수 없습니다.")
    try:
        meta = prepare(args)
    except (subprocess.CalledProcessError, OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            message = exc.stderr or str(exc)
        else:
            message = str(exc)
        print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
