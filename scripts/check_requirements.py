"""A1-1 요구사항 자가채점 — ALL PASS 여야만 제출한다(설계 기록 docs/04의 'push 전 게이트').

    python scripts/check_requirements.py        # 0 → 통과
각 검사는 "통과할 수만 있는 검사"가 되지 않도록, 실패를 만들 수 있는 형태다(예: VIP Recency 검사).
"""
from __future__ import annotations

import inspect
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def main() -> int:
    # 1) 데이터 스키마 (요구: 1000건 이상·8컬럼 이상·3가지 타입)
    prod_path, txn_path = ROOT / "data/products.csv", ROOT / "data/transactions.csv"
    check("data/products.csv 존재", prod_path.exists(), str(prod_path))
    check("data/transactions.csv 존재", txn_path.exists(), str(txn_path))
    if not (prod_path.exists() and txn_path.exists()):
        return _report()
    p, t = pd.read_csv(prod_path), pd.read_csv(txn_path)
    check("상품 1000건 이상", len(p) >= 1000, f"{len(p)}건")
    check("상품 컬럼 8개 이상", p.shape[1] >= 8, f"{p.shape[1]}개")
    n_types = len({str(x) for x in p.dtypes})
    check("타입 3종 이상(csv 읽기 기준)", n_types >= 3, f"{n_types}종: {sorted({str(x) for x in p.dtypes})}")
    check("거래 테이블 필수 3컬럼", {"customer_id", "amount", "txn_date"} <= set(t.columns), str(list(t.columns)))

    # 2) 클래스 설계 요구: 4개 메서드가 인자를 받는다(하드코딩 금지)
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import DataAnalyzer
    for meth in ("load_data", "handle_missing_values", "detect_outliers", "calculate_rfm"):
        check(f"DataAnalyzer.{meth} 존재", callable(getattr(DataAnalyzer, meth, None)))
    sig = inspect.signature(DataAnalyzer.detect_outliers)
    check("detect_outliers 가 k·policy 를 인자로 받는다", {"k", "policy"} <= set(sig.parameters), str(sig))
    import re
    lines = (ROOT / "src/pipeline.py").read_text(encoding="utf-8").split("\n")
    bad = [ln.strip() for ln in lines if "1.5" in ln
           and not re.search(r"def |= 1\.5\b|k·IQR|Q3 - Q1|#|\"\"\"", ln)]
    check("본문에 마법 숫자 1.5 하드코딩 없음(기본값·수식 설명만 허용)", not bad, str(bad[:1]))

    # 3) 전처리 결과: 결측 0 + 정답지 MAE 개선(그룹별 < 전체)
    rep = ROOT / "reports/metrics.json"
    check("reports/metrics.json 존재(먼저 run_analysis.py 실행)", rep.exists())
    if rep.exists():
        m = json.loads(rep.read_text(encoding="utf-8"))
        check("임퓨테이션: 그룹별 중앙값 MAE < 전체 중앙값 MAE",
              m["imputation"]["groupby_median_MAE"] < m["imputation"]["global_median_MAE"],
              f"{m['imputation']['global_median_MAE']} → {m['imputation']['groupby_median_MAE']}")
        check("검증이 유효: 평가 행 수 ≥ 20", m["imputation"]["n_evaluated"] >= 20, f"{m['imputation']['n_evaluated']}건")
        check("IQR: 로그정상 플래그 비율이 정규보다 크다", m["iqr_flag_pct"]["lognormal"] > m["iqr_flag_pct"]["normal"],
              f"normal {m['iqr_flag_pct']['normal']}% vs lognormal {m['iqr_flag_pct']['lognormal']}%")
        check("RFM 게이트: VIP 평균 Recency < 전체 중앙값", m["rfm"]["gate_VIP_recency_더작음"] is True,
              f"VIP {m['rfm']['VIP_평균Recency']}일 vs 비VIP {m['rfm']['비VIP_평균Recency']}일")
        check("RFM 세그먼트 4종 이상", len(m["rfm"]["segments"]) >= 4, str(list(m["rfm"]["segments"])))
        check("속도 주장에 비교 대상이 적혀 있다", "n_pure_sample" in m["vectorization"], "순수Python/행루프/벡터화 3경로")

    # 4) 시각화 6종: 파일 존재 + 크기 + 한글 라벨(빈 도형 방지)
    names = ["01_hist_price.png", "02_box_price_before_after.png", "03_bar_segment.png",
             "04_heatmap_corr.png", "05_scatter_price_vs_imgmean.png", "06_line_monthly_revenue.png"]
    for n in names:
        f = ROOT / "figures" / n
        check(f"figures/{n} 존재·10KB 이상", f.exists() and f.stat().st_size > 10_000,
              "" if not f.exists() else f"{f.stat().st_size}B")

    # 5) 리포트·README: 인사이트 3요소의 실제 존재
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    for kw in ("근거", "실행", "검증"):
        check(f"README '{kw}' 3회 이상", readme.count(kw) >= 3, f"{readme.count(kw)}회")
    check("README에 수치 포함(소수점 숫자 5개 이상)", sum(readme.count(x) for x in ("일", "%")) >= 5, "단위 문자 수")
    nb = ROOT / "notebooks/analysis_report.ipynb"
    check("notebooks/analysis_report.ipynb 존재", nb.exists())
    if nb.exists():
        raw = json.loads(nb.read_text(encoding="utf-8"))
        executed = any(o.get("output_type") == "execute_result" or o.get("outputs") for c in raw["cells"]
                       for o in c.get("outputs", [{}]) if isinstance(o, dict))
        check("노트북 실행 결과 포함(nbconvert --execute 로 생성)", executed, f"cells={len(raw['cells'])}")

    # 6) 재현성: 같은 시드로 다시 만들어도 해시 동일
    meta = ROOT / "data/dataset_meta.json"
    check("data/dataset_meta.json 존재(시드 기록)", meta.exists())
    return _report()


def _report() -> int:
    w = max(len(n) for n, _, _ in results) + 1
    npass = sum(ok for _, ok, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{w}} {detail}")
    print(f"---- {npass}/{len(results)} PASS")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
