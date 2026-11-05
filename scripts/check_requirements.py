"""실제 H&M 실행 산출물에 대한 A1-1 요구·provenance 자동 검사.

데이터 준비와 run_analysis/spec_curve/build_notebook을 실행한 뒤 사용한다.
고객별 원본이 아닌 집계 결과만 Git에 남는지도 함께 검사한다.
"""
from __future__ import annotations

import inspect
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


def main() -> int:
    product_path = ROOT / "data/products.csv"
    transaction_path = ROOT / "data/transactions.csv"
    image_path = ROOT / "data/images.npy"
    meta_path = ROOT / "data/dataset_meta.json"
    for name, path in [("products", product_path), ("transactions", transaction_path),
                       ("images", image_path), ("metadata", meta_path)]:
        # 커밋되는 JSON이 실행 위치와 무관하게 동일하도록 저장소 기준 상대 경로를 기록한다.
        check(f"실데이터 입력 {name} 존재", path.exists(), str(path.relative_to(ROOT)))
    if not all(path.exists() for path in [product_path, transaction_path, image_path, meta_path]):
        return report()

    products = pd.read_csv(product_path, dtype={"product_id": str})
    transactions = pd.read_csv(transaction_path, dtype={"product_id": str, "customer_id": str})
    images = np.load(image_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    spec = pd.read_csv(ROOT / "reports/spec_curve.csv")

    check("상품 1000개 이상", len(products) >= 1000, f"{len(products):,}")
    check("거래 1000건 이상", len(transactions) >= 1000, f"{len(transactions):,}")
    check("상품 8컬럼 이상", products.shape[1] >= 8, str(products.shape[1]))
    check("CSV dtype 3종 이상", len({str(value) for value in products.dtypes}) >= 3,
          str(sorted({str(value) for value in products.dtypes})))
    check("거래 필수 스키마", {"txn_id", "customer_id", "product_id", "amount", "order_date"} <= set(transactions.columns))
    check("상품 ID 고유", products["product_id"].is_unique)
    check("거래행 ID 고유", transactions["txn_id"].is_unique)
    check("거래–상품 article join 100%", set(transactions["product_id"]) <= set(products["product_id"]))
    check("실제 이미지 shape·dtype", images.shape == (len(products), 32, 32, 3) and images.dtype == np.uint8,
          f"{images.shape}/{images.dtype}")
    manifest = pd.read_csv(ROOT / "data/image_manifest.csv", dtype={"product_id": str})
    check("이미지 manifest가 상품과 1:1", len(manifest) == len(products) and manifest["product_id"].is_unique)
    check("원본 이미지 SHA-256 기록", manifest["source_image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())

    check("합성 데이터 아님 명시", meta["synthetic"] is False and metrics["data_provenance"]["synthetic"] is False)
    is_mirror = meta["source_kind"] == "github_mirror_tail_window"
    check("실제 source kind 기록", meta["source_kind"] in
          {"github_mirror_tail_window", "kaggle_original_user_supplied"}, meta["source_kind"])
    mirror_commit = meta["mirror"]["commit"]
    check("미러 커밋 또는 원본 다운로드 경로 구분",
          bool(re.fullmatch(r"[0-9a-f]{40}", mirror_commit)) if is_mirror else mirror_commit is None,
          str(mirror_commit))
    transaction_name = next(name for name in meta["source_files"] if name != "articles.csv")
    source_transactions = meta["source_files"][transaction_name]
    check("미러 행 수 또는 원본 전체 행 수 확인",
          source_transactions["rows"] == 531967 if is_mirror else source_transactions["rows"] >= 531967,
          str(source_transactions["rows"]))
    check("원본 행 위치 확인", source_transactions["tail_source_row_max"] == 31788323 if is_mirror
          else source_transactions["tail_source_row_max"] >= 31788323)
    articles_hash = meta["source_files"]["articles.csv"]["sha256"]
    check("articles SHA-256 기록·검증", articles_hash ==
          "25f21882053517f7684ea43f524225a9481a7bb9aab9cf0abfc7fe8c4576a72b" if is_mirror
          else bool(re.fullmatch(r"[0-9a-f]{64}", articles_hash)))
    transaction_hash = source_transactions["sha256"]
    check("transactions SHA-256 기록·검증", transaction_hash ==
          "c0098b8215693aa1bcae386e86e27299b41aa3f7ffada685345e509b7eb3a44e" if is_mirror
          else bool(re.fullmatch(r"[0-9a-f]{64}", transaction_hash)))
    check("고객 수가 실제 표본과 일치", transactions["customer_id"].nunique() == meta["result"]["n_customers"] ==
          metrics["schema"]["n_customers"], str(transactions["customer_id"].nunique()))
    check("원본 관찰 기간 명시", (source_transactions["date_min"] == "2020-09-08" and
          source_transactions["date_max"] == "2020-09-22") if is_mirror else
          source_transactions["date_min"] <= source_transactions["date_max"])
    check("가격 단위를 원화로 오인하지 않음", "not KRW" in metrics["data_provenance"]["price_unit"])
    check("합성 생성기 제거", not (ROOT / "src/data_gen.py").exists())
    check("이미지 필터 수 계산 일치", meta["sampling"]["removed_transactions_without_image"] ==
          meta["sampling"]["before_image_filter_transactions"] - len(transactions))

    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import DataAnalyzer
    for method in ["load_data", "handle_missing_values", "detect_outliers", "calculate_rfm"]:
        check(f"DataAnalyzer.{method} 구현", callable(getattr(DataAnalyzer, method, None)))
    signature = inspect.signature(DataAnalyzer.detect_outliers)
    check("IQR k·policy 인자화", {"k", "policy"} <= set(signature.parameters), str(signature))
    missing_signature = inspect.signature(DataAnalyzer.handle_missing_values)
    check("결측 전략·fallback 인자화", {"strategy", "min_group", "fallback"} <= set(missing_signature.parameters),
          str(missing_signature))
    check("실제 설명 결측 처리 완료", metrics["actual_missing"]["description_after"] == 0)
    check("실제 가격 홀드아웃 20건 이상", metrics["imputation"]["n_evaluated"] >= 20 and
          "observed real prices" in metrics["imputation"]["method"], str(metrics["imputation"]["n_evaluated"]))
    check("두 대치 MAE 유한값", np.isfinite([
        metrics["imputation"]["global_median_MAE"], metrics["imputation"]["groupby_median_MAE"],
    ]).all())
    expected_pct = round(metrics["outliers"]["n_flagged"] / len(products) * 100, 2)
    check("IQR 플래그 비율 재계산 일치", metrics["outliers"]["price_outlier_pct"] == expected_pct,
          f"{expected_pct}%")
    check("실제 상관 유효 표본·유한값", metrics["correlation"]["n_pairs"] == len(products) and
          np.isfinite(metrics["correlation"]["price__img_mean"]))
    check("상관 2쌍 이상 계산", all(key in metrics["correlation"] and np.isfinite(metrics["correlation"][key])
          for key in ["price__img_mean", "price__transaction_count"]))
    check("스펙 커브 60개·유한값", len(spec) == 60 and np.isfinite(spec["r"]).all(), f"{len(spec)}")
    check("RFM 4개 세그먼트", len(metrics["rfm"]["segments"]) == 4, str(list(metrics["rfm"]["segments"])))
    check("VIP Recency 방향 게이트", metrics["rfm"]["gate_VIP_recency_lower"] is True,
          f"VIP={metrics['rfm']['VIP_mean_recency']:.3f}")
    check("R 역순 실패 모드 검출", metrics["rfm"]["gate_naive_reversal_is_worse"] is True,
          f"naive={metrics['rfm']['naive_VIP_mean_recency']:.3f}")
    check("벡터화 3경로 측정", all(key in metrics["vectorization"] for key in
          ["pure_python_s_per_image", "rowloop_s_per_image", "vectorized_s_per_image"]))

    figure_names = [
        "01_hist_price.png", "02_box_price_before_after.png", "03_bar_segment.png", "04_heatmap_corr.png",
        "05_scatter_price_vs_imgmean.png", "06_line_daily_revenue.png", "07_spec_curve.png",
    ]
    for name in figure_names:
        path = ROOT / "figures" / name
        check(f"그림 {name} 존재·10KB", path.exists() and path.stat().st_size > 10000,
              str(path.stat().st_size) if path.exists() else "missing")
    summary = pd.read_csv(ROOT / "reports/summary.csv", index_col=0)
    check("기술통계에 사분위수 포함", {"25%", "75%", "mean", "median", "std"} <= set(summary.columns))
    analysis = (ROOT / "reports/analysis.md").read_text(encoding="utf-8")
    check("리포트 가격 단위·비인과 한계", "원화가 아닌" in analysis and "인과효과를 뜻하지 않는다" in analysis)
    check("기술통계 수치 해석·시사점", "### 기술통계 해석" in analysis and
          "평균만으로 대표 가격을 말하지 않는다" in analysis and "`transaction_count`를 함께 확인한다" in analysis)
    check("상관 2쌍 각각 해석", "가격–이미지 평균" in analysis and "가격–거래행 수" in analysis and
          "가격만으로 판매량을 예측" in analysis)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for keyword in ["근거", "실행", "검증"]:
        check(f"README 사업 제안 '{keyword}' 3회", readme.count(keyword) >= 3, str(readme.count(keyword)))
    business = readme.split("## 비즈니스 인사이트", 1)[1].split("\n## 실행", 1)[0]
    business_blocks = re.findall(r"(?ms)^\d+\. \*\*.*?(?=^\d+\. \*\*|\Z)", business)
    check("README 사업 제안 3개 이상", len(business_blocks) >= 3, str(len(business_blocks)))
    for index, block in enumerate(business_blocks[:3], start=1):
        check(f"사업 제안 {index} 대상·기대효과·반증", all(term in block for term in
              ["근거", "대상 세그먼트", "기대 효과", "검증"]) and ("반증" in block or "추가로" in block))
    check("고객별 RFM을 reports에서 제외", not (ROOT / "reports/rfm.csv").exists())
    notebook_path = ROOT / "notebooks/analysis_report.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    outputs = [output for cell in notebook["cells"] for output in cell.get("outputs", [])]
    check("노트북 실제 실행 출력 포함", bool(outputs) and not any(output.get("output_type") == "error" for output in outputs),
          f"{len(outputs)} outputs")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    png_outputs = sum("image/png" in output.get("data", {}) for output in outputs)
    check("노트북 연속 실행·6종 PNG", execution_counts == list(range(1, len(code_cells) + 1)) and png_outputs >= 6,
          f"{execution_counts}/{png_outputs} PNG")
    notebook_text = "\n".join("".join(cell.get("source", [])) if isinstance(cell.get("source"), list)
                              else cell.get("source", "") for cell in notebook["cells"])
    check("노트북 탐색·통계·2쌍 해석", all(term in notebook_text for term in
          ["products.info()", "products.describe", ".head(3)", "구조 탐색 해석", "통계·두 변수 쌍의 상관 해석",
           "가격–거래행 수", "시사점"]))
    check("노트북 RFM 4그룹 특징 해석", "그룹별 특징과 시사점" in notebook_text and
          all(name in notebook_text for name in ["대체고객", "성장고객", "우수고객", "VIP"]))
    fetch_source = (ROOT / "scripts/fetch_kaggle_original.py").read_text(encoding="utf-8")
    check("Kaggle 브라우저 OAuth 로그인 강제", '"auth", "login", "--force"' in fetch_source and
          fetch_source.count("require_kaggle_login()") >= 3)
    check("Kaggle 토큰 환경변수 미사용", "TOKEN_ENV_VARS" in fetch_source and "oauth_env.pop(name, None)" in fetch_source)
    original_requirements = (ROOT / "requirements-original.txt").read_text(encoding="utf-8")
    check("OAuth 지원 Kaggle CLI 버전 고정", "kaggle>=2.2.5" in original_requirements and
          "six>=1.16" in original_requirements)
    check("Kaggle 2.2.5 competition 위치 인자", '"competitions", "download", COMPETITION' in fetch_source and
          '"-c", COMPETITION' not in fetch_source)
    run_all_source = (ROOT / "scripts/run_all.py").read_text(encoding="utf-8")
    check("run_all이 pytest까지 실행", '"-m", "pytest", "-q"' in run_all_source)
    repo_root = pathlib.Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve()
    data_relative = (ROOT / "data").resolve().relative_to(repo_root)
    tracked = subprocess.run(["git", "ls-files", "--", str(data_relative)], cwd=repo_root,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    check("원본·고객 단위 data/ Git 미추적", tracked == [], str(tracked))
    return report()


def report() -> int:
    width = max(len(name) for name, _, _ in RESULTS) + 1
    passed = sum(condition for _, condition, _ in RESULTS)
    payload = {
        "all_pass": passed == len(RESULTS), "passed": passed, "total": len(RESULTS),
        "checks": [{"name": name, "passed": condition, "detail": detail} for name, condition, detail in RESULTS],
    }
    (ROOT / "reports/requirements_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    for name, condition, detail in RESULTS:
        print(f"[{'PASS' if condition else 'FAIL'}] {name:<{width}} {detail}")
    print(f"---- {passed}/{len(RESULTS)} PASS")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
