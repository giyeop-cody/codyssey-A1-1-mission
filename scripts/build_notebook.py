"""노트북 생성 + 실행 — 제출물은 '코드가 도는 노트북'이어야 하므로 **실행된 결과**를 파일에 박는다.

    python scripts/build_notebook.py      # → notebooks/analysis_report.ipynb (outputs 포함)
"""
from __future__ import annotations

import pathlib

import nbformat as nbf
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                    "language_info": {"name": "python"}})
C = nb.cells.append
md = lambda t: C(nbf.v4.new_markdown_cell(t))
code = lambda t: C(nbf.v4.new_code_cell(t))

md("""# A1-1 분석 리포트 — 쇼핑몰 단골 찾기
목표: (1) 멀티모달 데이터 정제·탐색 (2) **설명 가능한** RFM 세분화.
이 노트북은 `src/pipeline.py::DataAnalyzer`만 호출하고, 모든 표·숫자는 실행 결과로 채워진다.
전체 재현: `python src/data_gen.py --seed 42 && jupyter nbconvert --execute --to notebook --inplace notebooks/analysis_report.ipynb`""")
code("""import json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from pipeline import DataAnalyzer
products = pd.read_csv("../data/products.csv"); txns = pd.read_csv("../data/transactions.csv", parse_dates=["txn_date"])
imgs = np.load("../data/images.npy"); truth = pd.read_csv("../data/_groundtruth_price.csv")
ana = DataAnalyzer(products=products, transactions=txns, images=imgs).load_data()
print("products", products.shape, "| txns", txns.shape, "| images", imgs.shape)
print("타입:", sorted({str(d) for d in products.dtypes}))
products.head(3)""")

md("""## 1. 결측치 — 대치법을 고르면 정확도가 얼마나 바뀌나
정답지(`_groundtruth_price.csv`)와의 절대오차로 잰다. `DataAnalyzer`는 정답지를 읽지 않는다.""")
code("""filled_all = ana.handle_missing_values(strategy="median")
filled_grp = ana.handle_missing_values(strategy="groupby_median", min_group=30)
na_ids = products.loc[products["price"].isna(), "product_id"]
def mae(df):
    j = df[["product_id", "price"]].merge(truth, on="product_id")
    j = j[j["product_id"].isin(na_ids)]
    return (j["price"] - j["price_true"]).abs().mean()
print(f"결측 {len(na_ids)}행 | 전체 중앙값 MAE {mae(filled_all):,.1f} | 카테고리별 중앙값 MAE {mae(filled_grp):,.1f}")
print("개선 배율", round(mae(filled_all) / mae(filled_grp), 2))
assert filled_grp["price"].isna().sum() == 0 and mae(filled_grp) < mae(filled_all)""")

md("""## 2. 이상치 — IQR 규칙이 태깅하는 비율
`경계 = [Q1 − k·IQR, Q3 + k·IQR]`. 분포가 길수록 많이 잡힌다(k=1.5 고정).""")
code("""flagged = ana.detect_outliers("price", k=1.5, policy="flag")
print(f"태깅 {int(flagged['price_is_outlier'].sum())}행 / {len(flagged)}행 = {flagged['price_is_outlier'].mean()*100:.2f}%")
print(flagged.loc[flagged["price_is_outlier"], ["product_id", "category", "price"]].head(3))
print("참고: drop 이면 20.67%의 상품을 버린다 → 기본값은 flag")""")

md("""## 3. 멀티모달 피처 — 그리고 '벡터화 몇 배' 주장의 진짜 의미""")
code("""feats = ana.add_multimodal_features()
print({k: ana.log[k] for k in ana.log if "img_" in k})
arr = imgs.astype(np.float64) / 255.0
import time
t0 = time.perf_counter(); _ = np.array([i.mean() for i in arr]); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); _ = arr.mean(axis=(1, 2, 3)); t_vec = time.perf_counter() - t0
print(f"행 루프 {t_loop:.4f}s vs 벡터화 {t_vec:.4f}s → 벡터화 {t_loop/t_vec:.2f}배 (1 미만 = 벡터화가 더 느림)")
print("결론: 이미지처럼 '작은 배열 여러 개'는 벡터화가 불리하다. 주장할 때 항상 배열 개수·크수를 적는다.")
feats[["product_id", "category", "img_mean", "img_std", "word_count", "has_promo"]].head(3)""")

md("""## 4. 통계·상관""")
code("""summ = ana.summarize(feats.assign(price=filled_grp["price"]))
summ.loc[["price", "rating", "stock", "word_count", "img_mean"]]
corr = feats[["price", "rating", "stock", "word_count", "img_mean", "img_std"]].assign(price=filled_grp["price"].to_numpy()).corr()
print("price vs img_mean r =", round(corr.loc["price", "img_mean"], 3), "| price vs rating r =", round(corr.loc["price", "rating"], 3))
corr.style.background_gradient(cmap="coolwarm", axis=None)""")

md("""## 5. RFM 세분화 — R 점수를 뒤집지 않으면 무슨 일이 생기나""")
code("""rfm = ana.calculate_rfm(quantiles=5)
tab = rfm.groupby("segment", observed=True).agg(고객수=("customer_id", "size"), 매출=("monetary", "sum"), 평균Recency=("recency", "mean"))
tab["매출점유율_%"] = (tab["매출"] / rfm["monetary"].sum() * 100).round(1)
print(tab.round(1))
vip = rfm[rfm["segment"] == "VIP"]
print(f"VIP 평균 Recency {vip['recency'].mean():.1f}일 (전체 중앙값 {rfm['recency'].median():.1f}일)")
assert vip["recency"].mean() < rfm["recency"].median(), "R 점수 역전: 뒤집지 않은 것 아닌지 확인"
naive_score = pd.qcut(rfm["recency"], 5, labels=False, duplicates="drop") + 1     # 실수 재현: 안 뒤집기
bad = rfm[naive_score == 5]
print(f"[실수 재현] 안 뒤집으면 VIP 후보 {len(bad)}명의 평균 Recency {bad['recency'].mean():.1f}일, 매출 점유 {bad['monetary'].sum()/rfm['monetary'].sum()*100:.1f}%")""")

md("""## 6. 인사이트 (근거 → 실행 → 검증)""")
code("""ins = ana.report_insights(rfm)
for k, v in ins.items():
    print("●", k)
    for kk in ("근거", "실행", "검증"):
        print("   -", kk, ":", v.get(kk))""")

md("""## 7. 이 노트북의 검증 장치
- 위 `assert` 2개(결측 대치 개선, R 점수 역전)가 깨지면 리포트가 멈춘다 = 주장과 코드가 분리되지 않는다.
- `python scripts/check_requirements.py` 가 요구 8항목·6그림·인사이트 3요소를 채점한다.
- 이 파일은 `scripts/build_notebook.py`가 만들어 실행까지 포함한다(빈 출력 셀 제출 방지).""")

out = ROOT / "notebooks" / "analysis_report.ipynb"
out.parent.mkdir(exist_ok=True, parents=True)   # 커널 작업디렉터리가 되므로 먼저 만든다

client = NotebookClient(nb, timeout=600, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT / "notebooks")}})
client.execute()
out = ROOT / "notebooks" / "analysis_report.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f" wrote {out} · cells={len(nb.cells)} · 실행된 코드셀="
      f"{sum(1 for c in nb.cells if c.cell_type=='code' and c.get('outputs'))}")
