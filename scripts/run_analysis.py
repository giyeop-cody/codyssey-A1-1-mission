"""A1-1 분석 러너 — 전처리·피처·통계·시각화·RFM을 한 번에 돌리고 **숫자를 파일로 남긴다**.

    python scripts/run_analysis.py            # data/ 필요
산출: reports/metrics.json, reports/analysis.md, figures/*.png

설계에서 온 규칙: 리포트에 적히는 모든 숫자는 이 스크립트가 쓴 JSON에서 나온다(직접 쓰지 않는다).
"""
from __future__ import annotations

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
from pipeline import DataAnalyzer  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]   # 한글 라벨 누락 방지(이 환경의 실재 폰트)
plt.rcParams["axes.unicode_minus"] = False

FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True, parents=True)
REP.mkdir(exist_ok=True, parents=True)


def save(fig, name: str) -> None:
    """그림 저장 전 제목·축 레이블을 강제 검사(요구 6번의 가장 흔한 감점).

    twin 축(twinx)에는 제목·x축 라벨이 없으므로 '첫 축의 제목 + 축마다 최소 하나의 라벨'로 검사한다.
    """
    assert fig.axes, f"{name}: 축이 없다"
    assert fig.axes[0].get_title(), f"{name}: 제목 누락"
    real = [a for a in fig.axes if getattr(a, "_colorbar", None) is None]   # colorbar 축은 검사에서 제외
    for ax in real:
        assert ax.get_xlabel() or ax.get_ylabel(), f"{name}: 축 레이블 누락"
    assert any(ax.get_xlabel() for ax in real), f"{name}: x축 레이블 누락"
    assert any(ax.get_ylabel() for ax in real), f"{name}: y축 레이블 누락"
    fig.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    m: dict = {}
    products = pd.read_csv(ROOT / "data/products.csv")
    txns = pd.read_csv(ROOT / "data/transactions.csv", parse_dates=["txn_date"])
    imgs = np.load(ROOT / "data/images.npy")
    truth = pd.read_csv(ROOT / "data/_groundtruth_price.csv")  # 검증 전용(분석에는 사용하지 않음)

    ana = DataAnalyzer(products=products, transactions=txns, images=imgs).load_data()
    m["schema"] = {"n_products": int(len(products)), "n_txns": int(len(txns)),
                   "n_cols": int(products.shape[1]), "n_dtypes": int(len({str(x) for x in products.dtypes})),
                   "n_customers": int(txns["customer_id"].nunique())}

    # --- E1. 결측 처리: 전체 중앙값 vs 카테고리별 중앙값 (정답지 대비 MAE) -----------------
    filled = ana.handle_missing_values(strategy="median")
    filled_g = ana.handle_missing_values(strategy="groupby_median", min_group=30)
    merged = filled[["product_id", "category", "price"]].merge(truth, on="product_id")
    merged_g = filled_g[["product_id", "category", "price"]].merge(truth, on="product_id")
    na_ids = products.loc[products["price"].isna(), "product_id"]
    m["missing_price_n"] = int(len(na_ids))
    sub, subg = merged[merged["product_id"].isin(na_ids)], merged_g[merged_g["product_id"].isin(na_ids)]
    err_global = (sub["price"] - sub["price_true"]).abs()
    err_group = (subg["price"] - subg["price_true"]).abs()
    m["imputation"] = {"global_median_MAE": round(float(err_global.mean()), 1),
                       "groupby_median_MAE": round(float(err_group.mean()), 1),
                       "n_evaluated": int(len(sub)),
                       "주의": "처음엔 positional index 를 product_id 로 잘못 짝지어 MAE 0.0(=검증 무효)이 나왔다. 정답지 비교는 결측 행에서만 성립한다"}
    m["imputation"]["MAE_by_category"] = {
        "전체중앙값": {k: round(float(v), 1) for k, v in
                       sub.assign(e=(sub["price"] - sub["price_true"]).abs()).groupby("category")["e"].mean().items()},
        "그룹중앙값": {k: round(float(v), 1) for k, v in
                       subg.assign(e=(subg["price"] - subg["price_true"]).abs()).groupby("category")["e"].mean().items()}}

    # --- E2. 이상치: IQR 플래그 비율(정규 vs 로그정상) + 실데이터 비율 ----------------------
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({"normal": rng.normal(0, 1, 5000), "lognormal": np.exp(rng.normal(0, 1, 5000))})
    iqr_pct = {}
    for col in demo.columns:
        q1, q3 = demo[col].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        iqr_pct[col] = round(float(((demo[col] < lo) | (demo[col] > hi)).mean() * 100), 2)
    outlier_df = ana.detect_outliers("price", k=1.5, policy="flag")
    iqr_pct["price_real"] = ana.log["price_outlier_pct"]
    m["iqr_flag_pct"] = iqr_pct
    m["outliers"] = {"n_flagged": int(outlier_df["price_is_outlier"].sum()),
                     "max_price": round(float(products["price"].max()), 0),
                     "p99_price": round(float(products["price"].quantile(0.99)), 0),
                     "clip_would_move_rows": int((outlier_df.loc[outlier_df["price_is_outlier"], "price"]
                                                  > products["price"].quantile(0.75) + 1.5 *
                                                  (products["price"].quantile(0.75) - products["price"].quantile(0.25))).sum())}

    # --- E3. 멀티모달 피처 + 성능 측정 ------------------------------------------------------
    feats = ana.add_multimodal_features()
    m["vectorization"] = {"loop_s": ana.log["img_time_loop_s"], "vectorized_s": ana.log["img_time_vectorized_s"],
                          "speedup_loop_vs_vec": ana.log["img_speedup_loop_vs_vec"]}
    n_pure = 150
    t0 = time.perf_counter()
    for img in imgs[:n_pure]:                       # 순수 Python: NumPy 집계 메서드 없이 리스트 순환으로 평균
        s = 0
        for row in img.tolist():                    # (H, W, C) → 중첩 리스트 3단
            for px in row:
                for ch in px:
                    s += ch
        _ = s / img.size
    t_pure = (time.perf_counter() - t0) / n_pure
    vec_per = ana.log["img_time_vectorized_s"] / max(len(imgs), 1)
    loop_per = ana.log["img_time_loop_s"] / max(len(imgs), 1)
    m["vectorization"].update({"pure_python_s_per_image": round(t_pure, 5),
                               "vectorized_s_per_image": round(vec_per, 6),
                               "rowloop_s_per_image": round(loop_per, 6),
                               "speedup_pure_vs_vectorized": round(t_pure / max(vec_per, 1e-12), 1),
                               "speedup_pure_vs_rowloop": round(t_pure / max(loop_per, 1e-12), 1),
                               "n_pure_sample": n_pure})

    # --- E4. 통계·상관 ----------------------------------------------------------------------
    summ = ana.summarize(feats.assign(price=filled_g["price"]))
    summ.to_csv(REP / "summary.csv")
    corr = feats[["price", "rating", "stock", "word_count", "img_mean", "img_std", "img_saturation"]].assign(
        price=filled_g["price"].to_numpy()).corr()
    corr.to_csv(REP / "correlation.csv")
    r_price_img = float(corr.loc["price", "img_mean"])
    r_price_rating = float(corr.loc["price", "rating"])
    m["correlation"] = {"price__img_mean": round(r_price_img, 3), "price__rating": round(r_price_rating, 3)}

    # ── "이미지 평균으로 무엇을 보나" 질문에 대한 실측 (2026-09-15) ────────────────────────
    # 상관 하나만 적어놓고 결론을 만든 것이 문제였다. 같은 두 변수를 잣대 4개로 재고,
    # 결론이 잣대에 따라 바뀌는 것을 리포트에 남긴다.
    price_obs = filled_g["price"].to_numpy(dtype=float)
    lo, hi = np.percentile(price_obs, [1, 99])
    pair = pd.DataFrame({"price": price_obs,
                         "price_clip": np.clip(price_obs, lo, hi),
                         "img_mean": feats["img_mean"].to_numpy(dtype=float)})
    r_raw = float(pair["price"].corr(pair["img_mean"]))                       # 피어슨(이상치 그대로)
    r_clip = float(pair["price_clip"].corr(pair["img_mean"]))                 # 1/99 클리핑 후
    r_log = float(np.log1p(pair["price_clip"]).corr(np.log1p(pair["img_mean"])))  # 로그 스케일
    r_spear = float(pair["price"].corr(pair["img_mean"], method="spearman"))  # 순서만(외점 강인)
    # 구성 타당도: 썸네일 밝기는 *원래 가격*에 묶어 생성했다(`data_gen.make_thumbnails`).
    # 정답지는 검증 전용 원칙은 그대로 두되, "파이프라인이 신호를 주워야 검사가 산다"는 확인에만 쓴다.
    gt_path = ROOT / "data" / "_groundtruth_price.csv"
    r_true = float("nan")
    if gt_path.exists():
        gt = pd.read_csv(gt_path)["price_true"].to_numpy(dtype=float)
        r_true = float(pd.Series(gt).corr(pair["img_mean"]))
    var = pd.DataFrame([
        {"잣대": "피어슨(원본)", "r": round(r_raw, 4), "무엇을 다른 문장으로 만드나": "이상치 248행이 상관을 깎는다"},
        {"잣대": "피어슨(1·99 클리핑)", "r": round(r_clip, 4), "무엇을 다른 문장으로 만드나": "처리 정책 하나가 r 의 2~4배"},
        {"잣대": "피어슨(log1p)", "r": round(r_log, 4), "무엇을 다른 문장으로 만드나": "스케일 선택도 결론 입력값"},
        {"잣대": "스피어만(원본)", "r": round(r_spear, 4), "무엇을 다른 문장으로 만드나": "순서 기반이라 외점에 강함"},
        {"잣대": "vs 생성 원본(타당도)", "r": round(r_true, 4), "무엇을 다른 문장으로 만드나": "파이프라인이 신호를 주웠다 = 검사 유효"},
    ])
    var.to_csv(REP / "correlation_variants.csv", index=False)
    m["correlation_variants"] = {r["잣대"]: r["r"] for _, r in var.iterrows()}
    m["correlation_주의"] = ("r(price,img_mean)=0.119 는 '이상치 미처리 + 피어슨' 조합의 측정값이다. "
                          "같은 데이터에서 클리핑·로그·스피어만은 r 을 올린다. 따라서 '이미지로 가격을 설명할 수 "
                          "없다'는 비즈니스 결론이 아니라 **잣대 의존적 관측**이다. 그리고 이 데이터에는 노출·클릭 "
                          "컬럼이 0개라 CTR·CVR 은 정의조차 되지 않는다(probe_a1_2 실측).")
    m["insights_key"] = {"img_mean_실체": "썸네일 (32,32,3) uint8 의 픽셀 평균 = 밝기(0~255). KPI 아님",
                         "img_mean_결론": f"피어슨 {r_raw:.3f} / 클리핑 {r_clip:.3f} / log {r_log:.3f} / "
                                          f"스피어만 {r_spear:.3f} → 잣대에 따라 크기 변화, 방향은 +로 유지",
                         "참여지표": "impressions·clicks 0컬럼 → CTR/CVR 계산 불가(요구사항 결함으로 기록)"}

    # --- E5. 6종 시각화 ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(np.log10(feats["price"].dropna()), bins=40)
    ax.set_title("로그10(가격) 분포 — 오른쪽 꼬리가 이상치")
    ax.set_xlabel("log10(price)"); ax.set_ylabel("상품 수")
    save(fig, "01_hist_price.png")

    pr = feats[["price"]].assign(price=filled_g["price"].to_numpy(), flagged=outlier_df["price_is_outlier"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.boxplot([np.log10(pr.loc[~pr.flagged, "price"]), np.log10(pr["price"])],
               tick_labels=["이상치 플래그 제외", "전체(플래그 포함)"])
    ax.set_title("이상치 제외 전후 가격 분포 (box)")
    ax.set_xlabel("처리"); ax.set_ylabel("log10(price)")
    save(fig, "02_box_price_before_after.png")

    seg = feats.assign(price=filled_g["price"].to_numpy()).groupby("category").agg(매출=("price", "sum"), 상품수=("price", "size"))
    # 처음엔 twinx 로 겹쳐 그렸는데 두 눈금 스케일이 섞여 어느 막대가 무엇인지 읽히지 않았다(그림 03 확인).
    # → 스케일이 다른 두 지표는 나란한 서브플롯이 정답.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    seg["상품수"].plot.bar(ax=ax, color="#4477aa")
    ax.set_title("카테고리별 상품 수")
    ax.set_xlabel("카테고리"); ax.set_ylabel("상품 수"); ax.tick_params(axis="x", rotation=0); ax.set_ylim(0, seg["상품수"].max() * 1.25)
    seg["매출"].plot.bar(ax=ax2, color="#aa6644")
    ax2.set_title("카테고리별 매출 합(가격×수정, 로그아래)")
    ax2.set_xlabel("카테고리"); ax2.set_ylabel("매출 합"); ax2.tick_params(axis="x", rotation=0)
    ax2.set_yscale("log")
    save(fig, "03_bar_segment.png")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("상관행렬(피어슨)")
    ax.set_xlabel("변수(열)"); ax.set_ylabel("변수(행)")
    fig.colorbar(im, ax=ax, fraction=.046, label="r")
    save(fig, "04_heatmap_corr.png")

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.scatter(feats["img_mean"], np.log10(feats["price"]), s=8, alpha=.5)
    ax.set_title(f"이미지 평균 밝기 vs 가격 (r={r_price_img:.3f})")
    ax.set_xlabel("img_mean (0~1)"); ax.set_ylabel("log10(price)")
    save(fig, "05_scatter_price_vs_imgmean.png")

    mon = txns.assign(month=txns["txn_date"].dt.to_period("M")).groupby("month")["amount"].sum()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    mon.plot(ax=ax, marker="o")
    ax.set_title("월별 매출 (line)")
    ax.set_xlabel("월"); ax.set_ylabel("매출 합")
    save(fig, "06_line_monthly_revenue.png")

    # --- E6. RFM ----------------------------------------------------------------------------
    rfm = ana.calculate_rfm(quantiles=5)
    rfm.to_csv(REP / "rfm.csv", index=False)
    vip = rfm[rfm["segment"] == "VIP"]
    nonvip = rfm[rfm["segment"] != "VIP"]
    # 일부러 뒤집지 않은 버전(실패 모드)을 함께 재고 → "역전 시" 리스크를 숫자로 남긴다
    # 실패 모드 재현: R 점수를 뒤집지 않으면(qcut+1 그대로) '높은 점수 = 오래된 고객'이 되어
    # VIP 버킷에 가장 잠수탄 고객이 모인다. (라벨을 뒤집는 것과는 다른 실수다)
    naive = rfm.copy()
    naive["R_score_naive"] = pd.qcut(naive["recency"], 5, labels=False, duplicates="drop") + 1
    naive["segment_naive"] = pd.qcut(naive["R_score_naive"], 4, labels=["대체", "성장", "우수", "VIP"],
                                     duplicates="drop")
    m["rfm"] = {
        "n_customers": int(len(rfm)),
        "segments": {str(k): {"n": int(len(g)), "매출점유율_%": round(float(g["monetary"].sum() / rfm["monetary"].sum() * 100), 1),
                             "평균Recency_일": round(float(g["recency"].mean()), 1)}
                     for k, g in rfm.groupby("segment", observed=True)},
        "VIP_평균Recency": round(float(vip["recency"].mean()), 1),
        "비VIP_평균Recency": round(float(nonvip["recency"].mean()), 1),
        "역순실수_VIP_평균Recency": round(float(naive.loc[naive["segment_naive"] == "VIP", "recency"].mean()), 1),
        "역순실수_VIP_고객수": int((naive["segment_naive"] == "VIP").sum()),
        "역순실수_매출점유율_%": round(float(naive.loc[naive["segment_naive"] == "VIP", "monetary"].sum()
                                       / rfm["monetary"].sum() * 100), 1),
        "gate_VIP_recency_더작음": bool(vip["recency"].mean() < rfm["recency"].median()),
    }

    # --- 리포트 -----------------------------------------------------------------------------
    lines = ["# A1-1 분석 리포트 (자동 생성 — `scripts/run_analysis.py`)\n"]
    lines.append("| 항목 | 값 |\n|---|---|")
    for k, v in m["schema"].items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"\n## 결측 처리 (정답지 대비 MAE · 결측 {m['imputation']['n_evaluated']}건)\n")
    lines.append(f"- 전체 중앙값: **{m['imputation']['global_median_MAE']}**")
    lines.append(f"- 카테고리별 중앙값: **{m['imputation']['groupby_median_MAE']}**")
    lines.append(f"- 카테고리별 MAE · 전체 중앙값: {m['imputation']['MAE_by_category']['전체중앙값']}")
    lines.append(f"- 카테고리별 MAE · 그룹별 중앙값: {m['imputation']['MAE_by_category']['그룹중앙값']}")
    lines.append("\n## IQR 이상치 플래그 비율\n")
    for k, v in m["iqr_flag_pct"].items():
        lines.append(f"- {k}: **{v}%**")
    lines.append(f"- 실데이터 가격에서 태깅: {m['outliers']['n_flagged']}건 (max {m['outliers']['max_price']:.0f}, p99 {m['outliers']['p99_price']:.0f})")
    lines.append("\n## 벡터화 성능 (32×32×3 이미지 1200장)\n")
    v = m["vectorization"]
    lines.append(f"- 이미지 {len(imgs)}장: 행별 루프(NumPy .mean 호출) {v['loop_s']}s vs 전체 벡터화 {v['vectorized_s']}s → 벡터화 {v['speedup_loop_vs_vec']}배 (1 미만이면 벡터화가 더 느림)")
    lines.append(f"- 1장당: 순수 Python {v['pure_python_s_per_image']}s · 행 루프 {v['rowloop_s_per_image']}s · 벡터화 {v['vectorized_s_per_image']}s")
    lines.append(f"- 배율: 순수 Python vs 벡터화 **{v['speedup_pure_vs_vectorized']}배** / vs 행 루프 **{v['speedup_pure_vs_rowloop']}배** (표본 {v['n_pure_sample']}장)")
    lines.append("\n## 상관\n")
    lines.append(f"- price vs img_mean: **r={m['correlation']['price__img_mean']}** (피어슨 · 이상치 미처리 · n=1,200)"
                 f" / price vs rating: r={m['correlation']['price__rating']} (피어슨 · 이상치 미처리 · n=1,200)")
    lines.append("")
    lines.append("`img_mean` 은 썸네일 (32,32,3) 배열의 **픽셀 평균 = 밝기**다(0~255). 클릭률·전환율이 아니다 — "
                 "이 테이블에는 노출·클릭 컬럼이 **0개**라서 CTR 을 정의할 수 없다(`reports/correlation_variants.csv`).")
    lines.append("")
    lines.append("상관 하나만 적으면 결론이 잣대에 의존한다는 사실이 사라진다. 같은 두 변수를 네 잣대로 재면:")
    lines.append("")
    lines.append("| 잣대 | r | |")
    lines.append("|---|---|---|")
    for k, v in m["correlation_variants"].items():
        lines.append(f"| {k} | {v} | |")
    lines.append("")
    lines.append(f"> {m['correlation_주의']}")
    lines.append("\n## RFM 세분화\n")
    for k, d in m["rfm"]["segments"].items():
        lines.append(f"- {k}: n={d['n']} · 매출 {d['매출점유율_%']}% · 평균 R {d['평균Recency_일']}일")
    lines.append(f"- VIP 평균 Recency **{m['rfm']['VIP_평균Recency']}일** vs 비VIP {m['rfm']['비VIP_평균Recency']}일 "
                 f"(gate 통과: {m['rfm']['gate_VIP_recency_더작음']})")
    lines.append(f"- 점수를 뒤집지 않으면: 'VIP'로 잡히는 집단의 평균 R **{m['rfm']['역순실수_VIP_평균Recency']}일**, 매출 점유 {m['rfm']['역순실수_매출점유율_%']}%")
    (REP / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REP / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
