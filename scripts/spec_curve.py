"""A1-1 보너스 분석 — 스펙 커브(specification curve)로 "이미지 평균 vs 가격"을 다시 잰다.

배경(2026-09-15): 리포트에 r=0.119 하나를 적어 "이미지로 가격을 설명할 수 없다"고 결론냈다.
질문은 옳았다. 다만 문헌이 준 더 큰 문제는 **단일 스펙의 인용** 자체다 —
Simonsohn·Simmons·Nelson(2020, Nat. Hum. Behav.)는 방어 가능한 분석 선택을 전부 돌아보고
추정값의 **분포**를 보고하라고 권한다. Rousselet·Pernet(2012)의 규칙도 함께 따른다:
"방법은 **결과가 아니라 데이터 진단**으로 고른다."(이상치가 보이면 → 강건 추정)

그래서 이 스크립트는 선택지를 내가 고르지 않고 **격자 전체**로 돌린다:
  계수 5종(pearson·spearman·kendall·20% percentage-bend·skipped)
  × 이상치 정책 3종(없음·1·99 클리핑·IQR 제거)
  × 스케일 2종(raw·log1p)
  × 카테고리 통제 2종(없음·카테고리 내 중심화)
  = 60 스펙

출력: reports/spec_curve.csv · reports/spec_curve_summary.json · figures/07_spec_curve.png
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, FIG, REP = ROOT / "data", ROOT / "figures", ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from pipeline import DataAnalyzer  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ── 강건 상관 2종 (Wilcox 1994/2004 의 간명한 구현) ─────────────────────────────────────
def _mad_center(v: np.ndarray) -> tuple[float, float]:
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, (mad if mad > 1e-12 else float(np.std(v)) or 1.0)


def percentage_bend(x: np.ndarray, y: np.ndarray, tr: float = 0.2) -> float:
    """20% percentage-bend correlation. 가장자리 관측치를 하중 감쇠 후 피어슨."""
    def bend(v: np.ndarray) -> np.ndarray:
        med, scale = _mad_center(v)
        om2 = 1 - 2 * tr
        lo, hi = 0.0, 6.0
        for _ in range(60):                              # 굽힘 상수 M 을 이분탐색
            m = (lo + hi) / 2
            psi = np.where(v <= med - m * scale, -m, np.where(v >= med + m * scale, m, (v - med) / scale))
            if float(np.mean(psi)) < om2:
                lo = m
            else:
                hi = m
        return psi
    px, py = bend(x), bend(y)
    sx, sy = px.std(), py.std()
    return float(np.mean((px - px.mean()) * (py - py.mean())) / (sx * sy)) if sx > 0 and sy > 0 else float("nan")


def skipped(x: np.ndarray, y: np.ndarray, crit: float = 13.8) -> tuple[float, int]:
    """skipped correlation 근사: 윈저화 공분산으로 마할라노비스 거리 → 이중 이상치 제외 → 피어슨.

    crit=13.8 ≈ χ²(2) 의 0.999 분위(Wilcox 가 쓰는 문턱). 원 구현은 MCD 를 쓰므로 **근사**다(갭 표에 명시).
    """
    mx, sx = _mad_center(x)
    my, sy = _mad_center(y)
    zx, zy = (x - mx) / sx, (y - my) / sy
    wins = np.clip(np.stack([zx, zy]), -0.8416, 0.8416)   # 20% 윈저화(정규기준 상수)
    cov = np.cov(wins)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return float(np.corrcoef(x, y)[0, 1]), len(x)
    d2 = np.einsum("ij,jk,ik->i", np.stack([zx, zy], axis=1), inv, np.stack([zx, zy], axis=1))
    keep = d2 < crit
    if keep.sum() < 30:
        return float("nan"), int(keep.sum())
    return float(np.corrcoef(x[keep], y[keep])[0, 1]), int(keep.sum())


def p_from_r(r: float, n: int) -> float:
    """Fisher z 근사 p(양측). r 이 0 근처·n 이 크면 충분하다."""
    if not np.isfinite(r) or n < 5 or abs(r) >= 1:
        return float("nan")
    z = math.atanh(r) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def main() -> None:
    products = pd.read_csv(DATA / "products.csv")
    txns = pd.read_csv(DATA / "transactions.csv", parse_dates=["txn_date"])
    imgs = np.load(DATA / "images.npy")
    ana = DataAnalyzer(products=products, transactions=txns, images=imgs).load_data()
    filled = ana.handle_missing_values(strategy="groupby_median", min_group=30)   # run_analysis 와 같은 순서
    feats = ana.add_multimodal_features()
    df = pd.DataFrame({"price": filled["price"].to_numpy(dtype=float),
                       "img_mean": feats["img_mean"].to_numpy(dtype=float),
                       "category": feats["category"].to_numpy()})

    rows = []
    for coef, outlier, scale, ctrl in itertools.product(
            ["pearson", "spearman", "kendall", "pb20", "skipped"],
            ["없음", "1·99클리핑", "IQR제거"], ["raw", "log1p"], ["없음", "카테고리내"]):
        d = df.dropna().copy()
        if outlier == "1·99클리핑":
            lo, hi = np.percentile(d["price"], [1, 99])
            d["price"] = d["price"].clip(lo, hi)
        elif outlier == "IQR제거":
            q1, q3 = np.percentile(d["price"], [25, 75])
            iqr = q3 - q1
            d = d[(d["price"] >= q1 - 1.5 * iqr) & (d["price"] <= q3 + 1.5 * iqr)]
        if scale == "log1p":
            d["price"], d["img_mean"] = np.log1p(d["price"] - d["price"].min() + 1), np.log1p(d["img_mean"])
        if ctrl == "카테고리내":
            d["price"] = d.groupby(d["category"])["price"].transform(lambda s: s - s.mean())
            d["img_mean"] = d.groupby(d["category"])["img_mean"].transform(lambda s: s - s.mean())
        x, y = d["price"].to_numpy(dtype=float), d["img_mean"].to_numpy(dtype=float)
        n = len(d)
        if coef == "pearson":
            r = float(np.corrcoef(x, y)[0, 1])
        elif coef == "spearman":
            r = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        elif coef == "kendall":
            r = float(pd.Series(x).corr(pd.Series(y), method="kendall"))
        elif coef == "pb20":
            r = percentage_bend(x, y)
        else:
            r, n = skipped(x, y)
        rows.append({"계수": coef, "이상치": outlier, "스케일": scale, "통제": ctrl,
                     "n": n, "r": round(r, 4), "p(근사)": f"{p_from_r(r, n):.2e}"})

    res = pd.DataFrame(rows)
    REP.mkdir(exist_ok=True)
    res.to_csv(REP / "spec_curve.csv", index=False)

    sig = res["p(근사)"].astype(float) < 0.05
    summary = {
        "n_specs": len(res),
        "r_최소": float(res["r"].min()), "r_최대": float(res["r"].max()),
        "r_중위": float(res["r"].median()), "r_1사분위": float(res["r"].quantile(0.25)),
        "r_3사분위": float(res["r"].quantile(0.75)),
        "부호일치율": float((np.sign(res["r"]) == np.sign(res["r"].median())).mean()),
        "p_0.05_미만_비율": float(sig.mean()),
        "영존재비율": float((res["r"].abs() < 0.1).mean()),
        "단일인용문제의_실측": "리포트에 적었던 r=0.119 는 60 스펙 중 하나의 값이었다",
        "선택규칙": "Rousselet·Pernet(2012): 방법은 결과가 아니라 데이터 진단으로 고른다. "
                    "이 데이터는 가격에 외점 248행(20.67%) → 강건 추정(pb20·skipped)이 1순위",
        "강건추정치": {"pb20_중위": float(res[res["계수"] == "pb20"]["r"].median()),
                      "skipped_중위": float(res[res["계수"] == "skipped"]["r"].median()),
                      "skipped_생존n_최소": int(res[res["계수"] == "skipped"]["n"].min())},
    }
    (REP / "spec_curve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 그림: 스펙 커브 ────────────────────────────────────────────────────────────────
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.1]})
    srt = res.sort_values("r").reset_index()
    cmap = {"pearson": "#c0392b", "spearman": "#2980b9", "kendall": "#27ae60",
            "pb20": "#8e44ad", "skipped": "#d35400"}
    ax.scatter(srt.index, srt["r"], c=[cmap[c] for c in srt["계수"]], s=26)
    ax.axhline(0, lw=.8, c="k", ls=":")
    ax.axhline(0.119, lw=1.0, c="k", ls="--", alpha=.7)
    ax.text(1, 0.13, "이전 리포트가 인용한 r=0.119", fontsize=7)
    ax.set_title("스펙 커브 — '이미지 평균 vs 가격'을 방어 가능한 60가지로 재기 (A1-1)")
    ax.set_ylabel("상관계수 r")
    ax.set_ylim(min(srt["r"].min(), -.05) - .05, srt["r"].max() + .1)
    for j, name in enumerate(cmap):
        m = srt["계수"] == name
        ax.scatter([], [], c=cmap[name], s=26, label=f"{name} ({m.sum()})")
    ax.legend(fontsize=6, ncol=5, loc="lower left")
    for col, lab in [("이상치", "이상치 정책"), ("스케일", "스케일"), ("통제", "카테고리 통제")]:
        vals = srt[col].unique()
        ax2.plot(srt.index, [list(vals).index(v) for v in srt[col]], lw=.8, label=lab)
    ax2.set_yticks([])
    ax2.set_xlabel("스펙 번호 (r 순으로 정렬)")
    ax2.set_ylabel("선택 축")
    ax2.legend(fontsize=6, ncol=3, loc="center left")
    real = [a for a in fig.axes if getattr(a, "_colorbar", None) is None]
    assert real[0].get_title() and any(a.get_xlabel() for a in real) and any(a.get_ylabel() for a in real)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "07_spec_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n계수별 분포:")
    print(res.groupby("계수")["r"].describe()[["count", "min", "50%", "max"]].round(4).to_string())
    print("\n최솟값·최댓값 스펙:")
    print(res.loc[res["r"].idxmin()].to_dict())
    print(res.loc[res["r"].idxmax()].to_dict())


if __name__ == "__main__":
    main()
