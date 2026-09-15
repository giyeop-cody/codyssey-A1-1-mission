"""reports/metrics.json → reports/gates.json (게이트별 판정을 파일로 남긴다).

게이트 테스트(`tests/test_gates.py`)는 이 파일을 읽어서 판정한다. 즉 '통과'는
사람이 선언하는 것이 아니라 이 스크립트가 실측값에서 계산한 결과다.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
fig = {p.name for p in (ROOT / "figures").glob("*.png")}
need = {"01_hist_price.png", "02_box_price_before_after.png", "03_bar_segment.png",
        "04_heatmap_corr.png", "05_scatter_price_vs_imgmean.png", "06_line_monthly_revenue.png"}

gates = {
    "G1": {"value": f"{len(fig & need)}/6 그림 존재(제목·축 검사 통과분)", "pass": (fig & need) == need,
           "how": "scripts/run_analysis.py:save() 저장 전 assert"},
    "G2": {"value": f"normal {m['iqr_flag_pct']['normal']}% / lognormal {m['iqr_flag_pct']['lognormal']}% / 가격 {m['iqr_flag_pct']['price_real']}%",
           "pass": m["iqr_flag_pct"]["lognormal"] > m["iqr_flag_pct"]["normal"], "how": "run_analysis E2"},
    "G3": {"value": f"MAE {m['imputation']['global_median_MAE']} → {m['imputation']['groupby_median_MAE']} ({m['imputation']['n_evaluated']}행)",
           "pass": m["imputation"]["groupby_median_MAE"] < m["imputation"]["global_median_MAE"], "how": "run_analysis E1(정답지 대비)"},
    "G4": {"value": f"VIP 평균 R {m['rfm']['VIP_평균Recency']}일 vs 비VIP {m['rfm']['비VIP_평균Recency']}일",
           "pass": bool(m["rfm"]["gate_VIP_recency_더작음"]), "how": "check_requirements + 노트북 assert"},
    "G5": {"value": f"벡터화/행루프 {m['vectorization']['speedup_loop_vs_vec']}배, 순수Python 대비 {m['vectorization']['speedup_pure_vs_vectorized']}배(표본 {m['vectorization']['n_pure_sample']}장)",
           "pass": "n_pure_sample" in m["vectorization"], "how": "run_analysis E3 — 배율은 비교 대상과 함께만 허용"},
}
sc_path = ROOT / "reports" / "spec_curve_summary.json"
sc = json.loads(sc_path.read_text(encoding="utf-8")) if sc_path.exists() else None
if sc:
    gates["G6"] = {"value": f"스펙 {sc['n_specs']}개 · r {sc['r_최소']}~{sc['r_최대']}(중위 {round(sc['r_중위'], 3)}) · "
                            f"부호일치 {round(sc['부호일치율'] * 100, 1)}% · 카테고리 통제 최소 r={sc['r_최소']}",
                   "pass": sc["n_specs"] >= 40 and sc["r_최소"] < 0.1,
                   "how": "scripts/spec_curve.py(전 스펙 격자) — 최소 스펙이 0.1 미만이면 '통제하면 사라진다'를 인정한 것"}
else:
    gates["G6"] = {"value": "reports/spec_curve_summary.json 없음 → 스펙 커브 미실행", "pass": False,
                   "how": "python scripts/spec_curve.py 로 생성"}

out = ROOT / "reports" / "gates.json"
out.write_text(json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for k, v in gates.items():
    print(f"[{'PASS' if v['pass'] else 'FAIL'}] {k} {v['value']}")
raise SystemExit(0 if all(v["pass"] for v in gates.values()) else 1)
