"""실제 H&M 부분집합으로 EDA·이미지 피처·6종 시각화·RFM을 실행한다.

    python scripts/prepare_hm_data.py --download-mirror
    python scripts/run_analysis.py
    python scripts/spec_curve.py

모든 수치는 실행 결과에서 계산한다. 원화 환산, 평점·재고·노출·클릭 생성은 하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import SEGMENT_LABELS, DataAnalyzer, tied_quantile_scores  # noqa: E402

DATA, FIG, REP = ROOT / "data", ROOT / "figures", ROOT / "reports"
FIG.mkdir(exist_ok=True, parents=True)
REP.mkdir(exist_ok=True, parents=True)
plt.rcParams["axes.unicode_minus"] = False


def save(fig: plt.Figure, name: str) -> None:
    if not fig.axes or not fig.axes[0].get_title():
        raise AssertionError(f"{name}: title/axes missing")
    real = [axis for axis in fig.axes if getattr(axis, "_colorbar", None) is None]
    if not any(axis.get_xlabel() for axis in real) or not any(axis.get_ylabel() for axis in real):
        raise AssertionError(f"{name}: axis label missing")
    fig.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def masked_by_hash(product_id: str, seed: int, modulo: int = 10) -> bool:
    digest = hashlib.sha256(f"price-holdout:{seed}:{product_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo == 0


def finite_float(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite metric: {value}")
    return result


def render_analysis_report() -> None:
    """저장된 실행 지표만으로 리포트를 다시 쓴다(spec_curve 실행 후에도 호출)."""
    metrics_path = REP / "metrics.json"
    if not metrics_path.exists():
        return
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    schema, imp, out, corr, rfm = m["schema"], m["imputation"], m["outliers"], m["correlation"], m["rfm"]
    meta = m["data_provenance"]
    is_mirror = meta["source_kind"] == "github_mirror_tail_window"
    source_reference = (f"원본 미러 고정 커밋: `{meta['mirror_commit']}`" if is_mirror else
                        "입력 경로: 사용자가 Kaggle 규칙에 동의하고 직접 다운로드한 전체 원본")
    window_caution = ("15일 단면으로 캠페인 효과를 주장하지 않는다." if is_mirror else
                      "이 관찰 기간의 상관·RFM만으로 캠페인 효과를 주장하지 않는다.")
    source_window_limit = ("GitHub 미러의 거래는 원본 마지막 15일 부분집합이다. 장기 충성도, 계절성, 진짜 VIP를 식별하는 분석이 아니다."
                           if is_mirror else
                           "Kaggle 전체 원본에서 고객 해시 표본을 뽑았어도 관찰 기간의 구매 이력만 사용한다. 미래 이탈·캠페인 효과는 별도 검증이 필요하다.")
    image_limit = ("이미지는 미러가 256×256으로 축소한 실제 상품 사진이며 이 코드가 32×32 RGB로 다시 축소했다."
                   if is_mirror else "Kaggle 원본 상품 사진을 이 코드가 32×32 RGB로 축소했다.") + " 원본 해상도의 질감·선명도 평가는 불가능하다."
    lines = [
        "# H&M 실데이터 A1-1 분석 리포트",
        "",
        "> 자동 생성: `scripts/run_analysis.py`; 60개 분석 스펙은 `scripts/spec_curve.py` 실행 후 갱신.",
        "",
        "## 데이터와 단위",
        "",
        f"- 실제 H&M 공개 데이터의 부분집합: **상품 {schema['n_products']:,}개 · 거래행 {schema['n_transactions']:,}건 · 고객 {schema['n_customers']:,}명 · 실제 이미지 {schema['n_images']:,}장**.",
        f"- 관찰 기간: **{schema['date_min']}~{schema['date_max']}**. 고객은 SHA-256 순위로 선택하고 그 고객의 기간 내 거래를 유지했다.",
        f"- {source_reference}. 원본 파일 해시와 표본 규칙은 `data/dataset_meta.json`에 기록했다.",
        "- **가격·amount는 원화가 아닌 H&M 정규화 값**이다. 합산은 상대 매출지표로만 해석한다. 평점·재고·주문 ID·노출·클릭·세션은 원본에 없으므로 만들지 않았다.",
        f"- 원본 거래 파일 {meta['source_transaction_rows']:,}행에서 고객을 선택한 뒤, 이미지가 없는 상품의 {meta['removed_transactions_without_image']:,}개 선택 거래행을 제외했다. 상품 가격은 선택 고객 표본 안의 article_id별 실거래 중앙값이다.",
        "",
        "## 실제 결측과 가격 홀드아웃 대치 평가",
        "",
        f"- 실제 상품 설명 결측은 {m['actual_missing']['description_before']}건이며 `unknown` 처리 후 {m['actual_missing']['description_after']}건이다. 실제 가격 결측은 0건이다.",
        f"- 대치법 비교를 위해 관측된 실제 가격의 결정론적 10%({imp['n_evaluated']}건)를 **평가용 사본에서만** 가리고, 남은 값으로 대치한 뒤 원래 실제 가격과 비교했다.",
        f"- 전체 중앙값 MAE: **{imp['global_median_MAE']:.10f}** / 카테고리별 중앙값 MAE: **{imp['groupby_median_MAE']:.10f}** (정규화 가격 단위).",
        "- 이 수치는 합성 정답지가 아니라 실제 관측값 홀드아웃이다. 메인 분석의 가격은 가리지 않았다.",
        "",
        "## 기술통계와 IQR",
        "",
    ]
    summary = pd.read_csv(REP / "summary.csv", index_col=0)
    lines += ["| 변수 | 평균 | 중앙값 | 표준편차 | Q1 | Q3 |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["price", "transaction_count", "word_count", "img_mean", "img_std", "img_saturation"]:
        row = summary.loc[name]
        lines.append(f"| {name} | {row['mean']:.6f} | {row['median']:.6f} | {row['std']:.6f} | {row['25%']:.6f} | {row['75%']:.6f} |")
    price_stats = summary.loc["price"]
    transaction_stats = summary.loc["transaction_count"]
    image_stats = summary.loc["img_mean"]
    lines += [
        "",
        "### 기술통계 해석",
        "",
        f"- 가격 평균 **{price_stats['mean']:.6f}**이 중앙값 **{price_stats['median']:.6f}**보다 크고 Q3는 **{price_stats['75%']:.6f}**다. 소수 고가 상품이 있는 오른쪽 꼬리를 고려해 평균만으로 대표 가격을 말하지 않는다(정규화 가격 단위).",
        f"- 상품별 거래행 수는 평균 **{transaction_stats['mean']:.3f}행**, 중앙값 **{transaction_stats['median']:.1f}행**, Q3 **{transaction_stats['75%']:.1f}행**이다. 절반 이상의 상품은 거래가 적으므로 상품 가격·상관을 읽을 때 `transaction_count`를 함께 확인한다.",
        f"- 이미지 평균 밝기는 평균 **{image_stats['mean']:.6f}**, 중앙값 **{image_stats['median']:.6f}**, 표준편차 **{image_stats['std']:.6f}**다. 이 값은 픽셀 요약일 뿐 사진 품질이나 선호도를 뜻하지 않는다.",
        "",
        f"- 실제 가격의 IQR(k=1.5) 플래그: **{out['n_flagged']}건({out['price_outlier_pct']}%)**; Q1={out['q1']:.8f}, Q3={out['q3']:.8f}, 상한={out['upper_fence']:.8f}.",
        "- 플래그는 오류 판정이 아니다. 고가 상품을 삭제하지 않고 기본 `flag`로 유지한다.",
        "",
        "## 실제 이미지 피처와 NumPy 축 연산",
        "",
        f"- `(N,32,32,3) uint8` 실제 이미지에서 RGB/255 평균·표준편차·채도 범위를 계산했다. 이미지를 가격에 맞춰 밝히지 않았다.",
        f"- 전체 벡터화 {m['vectorization']['vectorized_s']}s / 이미지별 NumPy 루프 {m['vectorization']['loop_s']}s. 1장당 순수 Python {m['vectorization']['pure_python_s_per_image']}s, 벡터화 {m['vectorization']['vectorized_s_per_image']}s (순수 Python 표본 {m['vectorization']['n_pure_sample']}장).",
        "- 시간은 하드웨어·배열 크기에 따라 바뀌므로 절대 배율보다 세 경로의 측정 조건을 함께 본다.",
        "",
        "## 상관: 잣대를 함께 보고",
        "",
        f"- 가격–이미지 평균: Pearson 원본 **r={corr['price__img_mean']:.4f}**, n={corr['n_pairs']:,}.",
        f"- 가격–거래행 수: 상품별 Pearson **r={corr['price__transaction_count']:.4f}**, n={corr['n_pairs']:,}.",
        "",
        "| 방법 | 이상치 | 스케일 | 카테고리 통제 | r | n |",
        "|---|---|---|---|---:|---:|",
    ]
    variants = pd.read_csv(REP / "correlation_variants.csv")
    for row in variants.itertuples():
        lines.append(f"| {row.method} | {row.outlier} | {row.scale} | {row.control} | {row.r:.4f} | {row.n:,} |")
    lines += [
        "",
        "이 값들은 밝기와 가격의 **선형/순위 관측 관계**다. 이미지 품질, CTR, 구매 전환, 인과효과를 뜻하지 않는다.",
        f"가격–거래행 수는 r={corr['price__transaction_count']:.4f}로 0에 가까워, 이 선택 표본에서는 상품 가격과 거래량의 선형 관계가 약하다. 따라서 가격만으로 판매량을 예측하거나 가격 변경의 인과효과를 주장하지 않는다.",
        "",
    ]
    spec_path = REP / "spec_curve_summary.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        lines += [
            "### 60개 스펙 커브",
            "",
            f"계수 5종×이상치 3종×스케일 2종×카테고리 통제 2종의 r 범위는 **{spec['r_min']:.4f}~{spec['r_max']:.4f}**, 중위 **{spec['r_median']:.4f}**다.",
            f"중위값과 부호가 같은 비율은 {spec['sign_agreement']*100:.1f}%, |r|<0.1 비율은 {spec['near_zero_share']*100:.1f}%다. 어느 한 스펙만 골라 시장 일반론을 만들지 않는다.",
            "",
        ]
    lines += [
        "## RFM 세분화",
        "",
        "기준일은 최종 거래 다음 날이다. R만 역전하고, 짧은 기간의 정수 빈도 동률은 평균순위로 같은 점수를 부여했다. F는 주문/바구니가 아니라 **고유 거래행 수**다.",
        "",
        "| 세그먼트 | 고객 수 | 매출지표 점유 | 평균 Recency(일) | 평균 거래행 수 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in rfm["segments"].items():
        lines.append(f"| {name} | {row['n']:,} | {row['revenue_share_pct']:.1f}% | {row['mean_recency_days']:.2f} | {row['mean_frequency_lines']:.2f} |")
    lines += [
        "",
        f"- VIP 평균 Recency **{rfm['VIP_mean_recency']:.2f}일**, 비VIP **{rfm['non_VIP_mean_recency']:.2f}일**, 전체 중앙값 {rfm['all_recency_median']:.2f}일; 방향 게이트={rfm['gate_VIP_recency_lower']}.",
        f"- R을 역전하지 않은 실패 모드의 VIP 평균 Recency는 **{rfm['naive_VIP_mean_recency']:.2f}일**이다.",
        "",
        "## 사업 제안 (근거 → 실행 → 검증)",
        "",
        "1. **단기 VIP를 후보로 하되 실제 발송 전 긴 이력으로 재계산한다.**",
        f"   - 근거: VIP {rfm['VIP_n']:,}명이 정규화 매출지표의 {rfm['VIP_revenue_share_pct']:.1f}%를 차지한다.",
        f"   - 실행: **대상 세그먼트는 VIP {rfm['VIP_n']:,}명**이다. 최소 6~12개월 이력으로 RFM을 다시 계산한 뒤 재구매 캠페인 대상을 시뮬레이션한다. **기대 효과는 재구매율·유지율 개선 여부를 확인하는 것**이며 효과를 미리 단정하지 않는다.",
        f"   - 검증: 사전 정한 대조군과 이후 재구매율·유지율·1인당 매출을 비교하고, 차이가 없으면 VIP 캠페인 가설을 반증된 것으로 본다. {window_caution}",
        "2. **상품 설명 누락과 가격 누락을 분리한 품질 게이트를 둔다.**",
        f"   - 근거: 실제 설명 결측 {m['actual_missing']['description_before']}건, 가격 홀드아웃 그룹 대치 MAE {imp['groupby_median_MAE']:.10f}.",
        "   - 실행: **대상 세그먼트는 VIP·우수·성장·대체고객 전체의 상품 탐색에 필요한 설명·가격 누락 레코드**다. 텍스트는 `unknown`으로 표시하고 가격은 카테고리별 중앙값과 소그룹 전체 중앙값 fallback을 적용한다. **기대 효과는 결측으로 인한 분석 제외와 대치 오차 감소**다.",
        "   - 검증: 추가로 확보한 실제 가격을 같은 해시 규칙으로 홀드아웃해 MAE를 재측정한다. 그룹 대치가 전체 중앙값보다 악화되면 현재 선택이 반증되므로 전체 중앙값으로 되돌린다.",
        "3. **이미지 피처는 가격 결론이 아니라 참여 실험의 입력으로 제한한다.**",
        f"   - 근거: 실제 이미지–가격 Pearson 원본 r={corr['price__img_mean']:.4f}; 노출·클릭·세션은 0개다.",
        "   - 실행: **대상 세그먼트는 VIP·우수고객**이며, 이들에게 동일 상품의 원본/보정 이미지를 가격과 노출위치가 같은 조건으로 무작위 배정하는 별도 실험을 설계한다. **기대 효과는 CTR·구매 전환 개선 여부를 검증하는 것**이다.",
        "   - 검증: 추가로 노출·클릭·세션·구매와 무작위 배정 로그를 수집해 CTR과 구매 전환을 나누어 비교한다. 대조 차이가 없으면 이미지 가설은 반증되며, 현재 상관을 실험 효과로 인용하지 않는다.",
        "",
        "## 한계",
        "",
        f"- {source_window_limit}",
        "- 고객 표본 내 가격 중앙값이며 일부 상품은 거래 수가 적다. `transaction_count`를 함께 보고 전체 H&M 가격으로 일반화하지 않는다.",
        f"- {image_limit}",
        "- 원본 규칙은 비상업·학술/교육 용도다. 원본 CSV, 고객별 RFM, 원본/파생 이미지 배열은 Git에 올리지 않는다.",
    ]
    (REP / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for name in ["products.csv", "transactions.csv", "images.npy", "dataset_meta.json"]:
        if not (DATA / name).exists():
            raise FileNotFoundError(f"{DATA/name}가 없습니다. 먼저 scripts/prepare_hm_data.py를 실행하세요.")
    meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
    if meta.get("synthetic") is not False:
        raise ValueError("실데이터 메타데이터가 아닙니다.")
    products = pd.read_csv(DATA / "products.csv", dtype={"product_id": str})
    txns = pd.read_csv(DATA / "transactions.csv", dtype={"product_id": str, "customer_id": str}, parse_dates=["order_date"])
    images = np.load(DATA / "images.npy")
    if images.shape != (len(products), 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError(f"unexpected images: {images.shape}/{images.dtype}")
    if not set(txns["product_id"]) <= set(products["product_id"]):
        raise ValueError("transactions contain products absent from products.csv")

    m: dict = {
        "data_provenance": {
            "dataset": meta["dataset"], "synthetic": False,
            "original_source": meta["original_source"], "source_kind": meta["source_kind"],
            "mirror_commit": meta["mirror"]["commit"],
            "source_transaction_rows": meta["source_files"][next(k for k in meta["source_files"] if k != "articles.csv")]["rows"],
            "source_transaction_sha256": meta["source_files"][next(k for k in meta["source_files"] if k != "articles.csv")]["sha256"],
            "articles_sha256": meta["source_files"]["articles.csv"]["sha256"],
            "sampling_method": meta["sampling"]["method"], "sampling_seed": meta["sampling"]["seed"],
            "removed_transactions_without_image": meta["sampling"]["removed_transactions_without_image"],
            "license_scope": meta["license_scope"], "price_unit": "H&M normalized/scaled value; not KRW",
        }
    }
    m["schema"] = {
        "n_products": len(products), "n_transactions": len(txns), "n_product_columns": products.shape[1],
        "n_dtypes": len({str(dtype) for dtype in products.dtypes}), "n_customers": txns["customer_id"].nunique(),
        "n_images": len(images), "image_shape": list(images.shape),
        "date_min": str(txns["order_date"].min().date()), "date_max": str(txns["order_date"].max().date()),
    }

    actual_description_missing = int(products["description"].isna().sum())
    clean = DataAnalyzer(products, txns).handle_missing_values(columns=["description"])
    m["actual_missing"] = {
        "description_before": actual_description_missing,
        "description_after": int(clean["description"].isna().sum()),
        "price_before": int(products["price"].isna().sum()),
    }
    ana = DataAnalyzer(products=clean, transactions=txns, images=images).load_data()

    seed = int(meta["sampling"]["seed"])
    holdout_mask = products["product_id"].map(lambda value: masked_by_hash(value, seed)).to_numpy()
    masked = products.copy()
    heldout_truth = masked.loc[holdout_mask, "price"].copy()
    masked.loc[holdout_mask, "price"] = np.nan
    masked_ana = DataAnalyzer(masked, txns)
    filled_global = masked_ana.handle_missing_values(strategy="median", columns=["price"])
    filled_group = masked_ana.handle_missing_values(strategy="groupby_median", min_group=30, columns=["price"])
    error_global = (filled_global.loc[holdout_mask, "price"] - heldout_truth).abs()
    error_group = (filled_group.loc[holdout_mask, "price"] - heldout_truth).abs()
    if len(heldout_truth) < 20:
        raise ValueError(f"price holdout too small: {len(heldout_truth)}")
    m["imputation"] = {
        "method": "deterministic 10% holdout of observed real prices; masked only in an evaluation copy",
        "n_evaluated": int(holdout_mask.sum()),
        "global_median_MAE": finite_float(error_global.mean()),
        "groupby_median_MAE": finite_float(error_group.mean()),
        "group_minus_global_MAE": finite_float(error_group.mean() - error_global.mean()),
        "min_group": 30,
        "MAE_by_category": {
            "global": {str(k): finite_float(v) for k, v in error_global.groupby(products.loc[holdout_mask, "category"]).mean().items()},
            "group": {str(k): finite_float(v) for k, v in error_group.groupby(products.loc[holdout_mask, "category"]).mean().items()},
        },
    }

    outlier_df = ana.detect_outliers("price", k=1.5, policy="flag")
    q1, q3 = products["price"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    lower = q1 - 1.5 * (q3 - q1)
    m["outliers"] = {
        "policy": "flag_only", "k": 1.5, "n_flagged": int(outlier_df["price_is_outlier"].sum()),
        "price_outlier_pct": ana.log["price_outlier_pct"], "q1": float(q1), "q3": float(q3),
        "lower_fence": float(lower), "upper_fence": float(upper),
        "p99_price": float(products["price"].quantile(0.99)), "max_price": float(products["price"].max()),
    }

    feats = ana.add_multimodal_features()
    n_pure = min(150, len(images))
    started = time.perf_counter()
    for image in images[:n_pure]:
        total = 0
        for row in image.tolist():
            for pixel in row:
                for channel in pixel:
                    total += channel
        _ = total / image.size
    pure_seconds = (time.perf_counter() - started) / n_pure
    vector_per = ana.log["img_time_vectorized_s"] / len(images)
    loop_per = ana.log["img_time_loop_s"] / len(images)
    m["vectorization"] = {
        "loop_s": ana.log["img_time_loop_s"], "vectorized_s": ana.log["img_time_vectorized_s"],
        "speedup_loop_vs_vec": ana.log["img_speedup_loop_vs_vec"],
        "pure_python_s_per_image": pure_seconds, "vectorized_s_per_image": vector_per,
        "rowloop_s_per_image": loop_per, "speedup_pure_vs_vectorized": pure_seconds / max(vector_per, 1e-12),
        "speedup_pure_vs_rowloop": pure_seconds / max(loop_per, 1e-12), "n_pure_sample": n_pure,
    }

    summary = ana.summarize(feats)
    summary.to_csv(REP / "summary.csv")
    corr_columns = ["price", "transaction_count", "customer_count", "word_count", "img_mean", "img_std", "img_saturation"]
    corr = feats[corr_columns].corr()
    corr.to_csv(REP / "correlation.csv")
    pair = feats[["price", "img_mean"]].dropna().copy()
    clipped = pair["price"].clip(*np.percentile(pair["price"], [1, 99]))
    centered = pair.copy()
    categories = feats.loc[pair.index, "category"]
    centered["price"] = pair["price"] - pair["price"].groupby(categories).transform("mean")
    centered["img_mean"] = pair["img_mean"] - pair["img_mean"].groupby(categories).transform("mean")
    variants = pd.DataFrame([
        {"method": "Pearson", "outlier": "none", "scale": "raw", "control": "none",
         "r": pair["price"].corr(pair["img_mean"]), "n": len(pair)},
        {"method": "Pearson", "outlier": "1/99 clip", "scale": "raw", "control": "none",
         "r": clipped.corr(pair["img_mean"]), "n": len(pair)},
        {"method": "Pearson", "outlier": "none", "scale": "log(price),log1p(img)", "control": "none",
         "r": np.log(pair["price"]).corr(np.log1p(pair["img_mean"])), "n": len(pair)},
        {"method": "Spearman", "outlier": "none", "scale": "ranks", "control": "none",
         "r": pair["price"].corr(pair["img_mean"], method="spearman"), "n": len(pair)},
        {"method": "Pearson", "outlier": "none", "scale": "category-centered", "control": "category",
         "r": centered["price"].corr(centered["img_mean"]), "n": len(pair)},
    ])
    if not np.isfinite(variants["r"]).all():
        raise ValueError("non-finite correlation variant")
    variants.to_csv(REP / "correlation_variants.csv", index=False)
    m["correlation"] = {
        "price__img_mean": finite_float(variants.loc[0, "r"]),
        "price__transaction_count": finite_float(corr.loc["price", "transaction_count"]),
        "n_pairs": len(pair),
        "variants": {f"{row.method}|{row.outlier}|{row.scale}|{row.control}": finite_float(row.r) for row in variants.itertuples()},
        "caution": ("Observed association in a selected-customer, 15-day mirror subset; not image quality, CTR, conversion, or causality."
                    if meta["source_kind"] == "github_mirror_tail_window" else
                    "Observed association in a selected-customer Kaggle-original sample; not image quality, CTR, conversion, or causality."),
    }

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(np.log10(products["price"]), bins=40, color="#4477aa")
    ax.set_title("Real H&M normalized price distribution")
    ax.set_xlabel("log10(normalized price)"); ax.set_ylabel("Products")
    save(fig, "01_hist_price.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.boxplot([
        np.log10(products.loc[~outlier_df["price_is_outlier"], "price"]),
        np.log10(products["price"]),
    ], tick_labels=["IQR-flag excluded", "All products"])
    ax.set_title("IQR flag comparison (rows are not deleted)")
    ax.set_xlabel("Displayed subset"); ax.set_ylabel("log10(normalized price)")
    save(fig, "02_box_price_before_after.png")

    category = txns.merge(products[["product_id", "category"]], on="product_id", how="left").groupby("category").agg(
        sales_value=("amount", "sum"), products=("product_id", "nunique")
    ).sort_values("sales_value", ascending=False).head(12).sort_values("sales_value")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    category["products"].plot.barh(ax=ax, color="#4477aa")
    ax.set_title("Products by category (top sales-value categories)")
    ax.set_xlabel("Products"); ax.set_ylabel("Category")
    category["sales_value"].plot.barh(ax=ax2, color="#aa6644")
    ax2.set_title("Observed sales-value index by category")
    ax2.set_xlabel("Sum of normalized amount"); ax2.set_ylabel("Category")
    save(fig, "03_bar_segment.png")

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Pearson correlation matrix — real H&M subset")
    ax.set_xlabel("Variable"); ax.set_ylabel("Variable")
    fig.colorbar(image, ax=ax, fraction=.046, label="r")
    save(fig, "04_heatmap_corr.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(feats["img_mean"], np.log10(feats["price"]), s=10, alpha=.45)
    ax.set_title(f"Real product image mean vs price (Pearson r={m['correlation']['price__img_mean']:.3f})")
    ax.set_xlabel("Mean RGB/255 across 32x32 pixels"); ax.set_ylabel("log10(normalized price)")
    save(fig, "05_scatter_price_vs_imgmean.png")

    daily = txns.groupby(txns["order_date"].dt.date)["amount"].sum()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    daily.plot(ax=ax, marker="o")
    source_window_title = "15-day mirror window" if meta["source_kind"] == "github_mirror_tail_window" else "Kaggle-original selected sample"
    ax.set_title(f"Daily observed sales-value index ({source_window_title})")
    ax.set_xlabel("Transaction date"); ax.set_ylabel("Sum of normalized amount")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "06_line_daily_revenue.png")

    rfm = ana.calculate_rfm(quantiles=5)
    rfm.to_csv(DATA / "rfm.csv", index=False)  # 고객 식별자 포함: data/는 커밋 금지
    naive = rfm.copy()
    naive["R_score"] = 6 - naive["R_score"]
    naive_combined = naive[["R_score", "F_score", "M_score"]].mean(axis=1)
    naive_segment_score = tied_quantile_scores(naive_combined, 4)
    naive["segment"] = pd.Categorical.from_codes(naive_segment_score - 1, categories=SEGMENT_LABELS, ordered=True)
    vip = rfm[rfm["segment"] == "VIP"]
    nonvip = rfm[rfm["segment"] != "VIP"]
    naive_vip = naive[naive["segment"] == "VIP"]
    if vip.empty or nonvip.empty or naive_vip.empty:
        raise ValueError("RFM segment missing; cannot run direction check")
    segments = {}
    for name, group in rfm.groupby("segment", observed=True):
        segments[str(name)] = {
            "n": len(group), "revenue_share_pct": float(group["monetary"].sum() / rfm["monetary"].sum() * 100),
            "mean_recency_days": float(group["recency"].mean()), "mean_frequency_lines": float(group["frequency"].mean()),
        }
    m["rfm"] = {
        "n_customers": len(rfm), "frequency_definition": "unique source transaction lines; no basket/order ID supplied",
        "segments": segments, "VIP_n": len(vip),
        "VIP_revenue_share_pct": float(vip["monetary"].sum() / rfm["monetary"].sum() * 100),
        "VIP_mean_recency": float(vip["recency"].mean()), "non_VIP_mean_recency": float(nonvip["recency"].mean()),
        "all_recency_median": float(rfm["recency"].median()),
        "naive_VIP_mean_recency": float(naive_vip["recency"].mean()),
        "naive_VIP_revenue_share_pct": float(naive_vip["monetary"].sum() / rfm["monetary"].sum() * 100),
        "gate_VIP_recency_lower": bool(vip["recency"].mean() < rfm["recency"].median()),
        "gate_naive_reversal_is_worse": bool(naive_vip["recency"].mean() > vip["recency"].mean()),
    }
    aggregate = rfm.groupby("segment", observed=True).agg(
        customers=("customer_id", "size"), mean_recency_days=("recency", "mean"),
        mean_frequency_lines=("frequency", "mean"), normalized_monetary_sum=("monetary", "sum"),
    ).reset_index()
    aggregate["monetary_share_pct"] = aggregate["normalized_monetary_sum"] / aggregate["normalized_monetary_sum"].sum() * 100
    aggregate.to_csv(REP / "rfm_segments.csv", index=False)

    (REP / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    render_analysis_report()
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
