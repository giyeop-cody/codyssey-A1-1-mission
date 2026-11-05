"""실데이터 노트북을 만들고 NBClient로 실제 실행한 출력까지 저장한다."""
from __future__ import annotations

import json
import pathlib

import nbformat
import pandas as pd
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "analysis_report.ipynb"


def main() -> None:
    dataset_meta = json.loads((ROOT / "data/dataset_meta.json").read_text(encoding="utf-8"))
    is_mirror = dataset_meta["source_kind"] == "github_mirror_tail_window"
    window_description = "15일 관찰창" if is_mirror else "선택된 Kaggle 원본 관찰 기간"
    limit_description = ("15일 부분집합의 단기 RFM이며 장기 충성도는 검증하지 않는다. " if is_mirror else
                         "전체 원본의 고객 표본 RFM이며 미래 이탈·장기 충성도는 별도 시간순 검증이 필요하다. ")
    report_metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(ROOT / "reports/summary.csv", index_col=0)
    price = summary.loc["price"]
    transaction_count = summary.loc["transaction_count"]
    rfm_metrics = report_metrics["rfm"]
    segment_characteristics = "\n".join(
        f"- **{name}**: 고객 {row['n']:,}명, 매출지표 점유 {row['revenue_share_pct']:.1f}%, "
        f"평균 Recency {row['mean_recency_days']:.2f}일, 평균 거래행 {row['mean_frequency_lines']:.2f}행"
        for name, row in rfm_metrics["segments"].items()
    )
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# A1-1 — H&M 실데이터 EDA·RFM\n\n"
            "공개 H&M 실제 상품·거래·이미지의 재현 가능한 부분집합을 사용한다. "
            "가격은 원화가 아닌 정규화 값이며, 노출·클릭·세션·평점·재고는 만들지 않는다. "
            "전체 그림과 60개 스펙은 `scripts/run_analysis.py`, `scripts/spec_curve.py`가 생성한다."
        ),
        nbformat.v4.new_code_cell(
            "import json\n"
            "import pathlib\n"
            "import sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "ROOT = pathlib.Path.cwd()\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from pipeline import DataAnalyzer\n"
            "meta = json.loads((ROOT / 'data/dataset_meta.json').read_text(encoding='utf-8'))\n"
            "products = pd.read_csv(ROOT / 'data/products.csv', dtype={'product_id': str})\n"
            "txns = pd.read_csv(ROOT / 'data/transactions.csv', dtype={'product_id': str, 'customer_id': str})\n"
            "images = np.load(ROOT / 'data/images.npy')\n"
            "print({'synthetic': meta['synthetic'], 'products': len(products), 'transactions': len(txns), "
            "'customers': txns.customer_id.nunique(), 'images': images.shape, "
            "'period': [txns.order_date.min(), txns.order_date.max()]})\n"
            "print('price unit: H&M normalized/scaled value (not KRW)')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 구조·분포·예시 탐색\n\n"
            "`info()`로 dtype과 결측을, `describe()`로 분포를 먼저 확인한다. `head()`는 고객 ID나 거래행을 "
            "노출하지 않고 상품의 비식별 컬럼 몇 개만 미리 본다. 원본·고객별 행은 저장소에 커밋하지 않는다."
        ),
        nbformat.v4.new_code_cell(
            "from IPython.display import display\n"
            "print('products.info()')\n"
            "products.info()\n"
            "display(products.describe(include='all').T)\n"
            "display(products[['name', 'category', 'price', 'transaction_count']].head(3))\n"
            "print('transactions.info() — customer rows are not displayed')\n"
            "txns.info()\n"
            "display(txns[['amount', 'sales_channel_id']].describe(include='all').T)"
        ),
        nbformat.v4.new_markdown_cell(
            "### 구조 탐색 해석\n\n"
            f"상품은 **{report_metrics['schema']['n_products']:,}행·{report_metrics['schema']['n_product_columns']}컬럼**, "
            f"거래는 **{report_metrics['schema']['n_transactions']:,}행**, 고객은 **{report_metrics['schema']['n_customers']:,}명**이다. "
            f"실제 설명 결측은 **{report_metrics['actual_missing']['description_before']}건**, 실제 가격 결측은 "
            f"**{report_metrics['actual_missing']['price_before']}건**이다. 가격 평균 **{price['mean']:.6f}**가 "
            f"중앙값 **{price['median']:.6f}**보다 크므로 오른쪽 꼬리를 고려하고, 상품별 거래 수 중앙값이 "
            f"**{transaction_count['median']:.1f}행**이므로 거래가 적은 상품의 가격·상관을 과도하게 일반화하지 않는다. "
            "시사점: 결측 처리와 `transaction_count` 확인을 후속 EDA의 품질 게이트로 둔다."
        ),
        nbformat.v4.new_markdown_cell(
            "## 실제 결측·이미지 축 연산·기술통계\n\n"
            "설명 결측만 `unknown`으로 처리한다. 가격 홀드아웃 대치 평가는 메인 데이터를 오염시키지 않도록 "
            "별도 러너의 평가용 사본에서만 수행한다."
        ),
        nbformat.v4.new_code_cell(
            "clean = DataAnalyzer(products, txns).handle_missing_values(columns=['description'])\n"
            "ana = DataAnalyzer(clean, txns, images=images).load_data()\n"
            "features = ana.add_multimodal_features()\n"
            "summary = ana.summarize(features)\n"
            "summary.loc[['price', 'transaction_count', 'word_count', 'img_mean', 'img_std', 'img_saturation']]"
        ),
        nbformat.v4.new_markdown_cell(
            "### 통계·두 변수 쌍의 상관 해석\n\n"
            f"가격 Q1/Q3는 **{price['25%']:.6f}/{price['75%']:.6f}**, 상품별 거래행 수 평균은 "
            f"**{transaction_count['mean']:.3f}행**이다. 가격–이미지 평균 Pearson 상관은 "
            f"**r={report_metrics['correlation']['price__img_mean']:.4f}**로 이 표본에서 약한 음의 관측 관계다. "
            f"두 번째 쌍인 가격–거래행 수는 **r={report_metrics['correlation']['price__transaction_count']:.4f}**로 "
            "0에 가까워 가격과 거래량의 선형 관계가 약하다. 시사점: 가격이나 밝기 하나로 판매량·고객 반응을 "
            "예측하지 않고, 카테고리·거래 수와 후속 노출 데이터를 함께 확인한다. 두 상관 모두 인과효과가 아니다."
        ),
        nbformat.v4.new_markdown_cell(
            "## RFM — 방향과 동률 검사\n\n"
            "R만 역전한다. F는 원본 주문 ID가 없으므로 바구니 수가 아니라 고유 거래행 수다. "
            "고객별 행은 출력하지 않고 세그먼트 집계만 표시한다."
        ),
        nbformat.v4.new_code_cell(
            "rfm = ana.calculate_rfm(quantiles=5)\n"
            "segments = rfm.groupby('segment', observed=True).agg(\n"
            "    customers=('customer_id', 'size'),\n"
            "    mean_recency_days=('recency', 'mean'),\n"
            "    mean_frequency_lines=('frequency', 'mean'),\n"
            "    normalized_monetary_sum=('monetary', 'sum'),\n"
            ")\n"
            "segments['monetary_share_pct'] = segments.normalized_monetary_sum / segments.normalized_monetary_sum.sum() * 100\n"
            "vip_mean = rfm.loc[rfm.segment == 'VIP', 'recency'].mean()\n"
            "assert vip_mean < rfm.recency.median()\n"
            "print(f'VIP mean recency={vip_mean:.3f} < all median={rfm.recency.median():.3f}')\n"
            "segments.round(3)"
        ),
        nbformat.v4.new_markdown_cell(
            "### 그룹별 특징과 시사점\n\n"
            f"{segment_characteristics}\n\n"
            f"VIP 평균 Recency **{rfm_metrics['VIP_mean_recency']:.2f}일**은 비VIP "
            f"**{rfm_metrics['non_VIP_mean_recency']:.2f}일**보다 짧고, R 미역전 실패 모드는 "
            f"**{rfm_metrics['naive_VIP_mean_recency']:.2f}일**로 나빠진다. 시사점: VIP는 후속 캠페인 "
            "후보로만 사용하되, 현재 관찰창만으로 장기 충성도나 캠페인 효과를 단정하지 않는다."
        ),
        nbformat.v4.new_markdown_cell(
            "## 필수 6종 시각화\n\n"
            "`scripts/run_analysis.py`가 실제 H&M 부분집합으로 저장한 히스토그램·박스플롯·막대·히트맵·산점도·라인차트를 "
            f"노트북 안에서도 순서대로 표시한다. 가격은 정규화 값이며 마지막 선 그래프는 {window_description}의 일별 매출지표다."
        ),
        nbformat.v4.new_code_cell(
            "from IPython.display import Image as DisplayImage, display\n"
            "figure_names = [\n"
            "    '01_hist_price.png', '02_box_price_before_after.png', '03_bar_segment.png',\n"
            "    '04_heatmap_corr.png', '05_scatter_price_vs_imgmean.png', '06_line_daily_revenue.png',\n"
            "]\n"
            "for name in figure_names:\n"
            "    print(name)\n"
            "    display(DisplayImage(filename=str(ROOT / 'figures' / name)))"
        ),
        nbformat.v4.new_markdown_cell(
            "## 실행 산출물과 해석 경계\n\n"
            "- `reports/metrics.json`: 데이터 출처·홀드아웃 대치·IQR·상관·RFM 측정값\n"
            "- `reports/spec_curve.csv`: 60개 분석 선택 조합\n"
            "- `figures/01..07`: 실제 데이터에서 만든 그림\n\n"
            f"{limit_description}캠페인 인과효과는 검증하지 않는다. "
            "이미지–가격 상관도 CTR·구매 전환 효과가 아니다."
        ),
        nbformat.v4.new_code_cell(
            "metrics = json.loads((ROOT / 'reports/metrics.json').read_text(encoding='utf-8'))\n"
            "spec = pd.read_csv(ROOT / 'reports/spec_curve.csv')\n"
            "result = pd.DataFrame([\n"
            "    {'metric': 'price holdout global MAE', 'value': metrics['imputation']['global_median_MAE']},\n"
            "    {'metric': 'price holdout group MAE', 'value': metrics['imputation']['groupby_median_MAE']},\n"
            "    {'metric': 'IQR flagged share', 'value': metrics['outliers']['price_outlier_pct'] / 100},\n"
            "    {'metric': 'raw Pearson price-image r', 'value': metrics['correlation']['price__img_mean']},\n"
            "    {'metric': 'raw Pearson price-transaction_count r', 'value': metrics['correlation']['price__transaction_count']},\n"
            "    {'metric': '60-spec minimum r', 'value': spec.r.min()},\n"
            "    {'metric': '60-spec maximum r', 'value': spec.r.max()},\n"
            "])\n"
            "result"
        ),
    ]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, TARGET)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, TARGET)
    print(f"executed notebook written: {TARGET}")


if __name__ == "__main__":
    main()
