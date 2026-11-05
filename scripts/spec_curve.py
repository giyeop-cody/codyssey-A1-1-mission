"""실제 H&M 가격–이미지 관계를 60개 방어 가능한 분석 스펙으로 재측정한다.

계수 5종 × 이상치 3종 × 스케일 2종 × 카테고리 통제 2종 = 60.
어느 한 결과를 골라 인과나 시장 일반론을 만들지 않는다.
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
sys.path.insert(0, str(ROOT))
from scripts.run_analysis import render_analysis_report, save  # noqa: E402


def mad_center(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad if mad > 1e-12 else float(np.std(values)) or 1.0


def percentage_bend(x: np.ndarray, y: np.ndarray, trim: float = 0.2) -> float:
    def bend(values: np.ndarray) -> np.ndarray:
        median, scale = mad_center(values)
        target = 1 - 2 * trim
        lo, hi = 0.0, 6.0
        for _ in range(60):
            curvature = (lo + hi) / 2
            psi = np.where(
                values <= median - curvature * scale, -curvature,
                np.where(values >= median + curvature * scale, curvature, (values - median) / scale),
            )
            if float(np.mean(psi)) < target:
                lo = curvature
            else:
                hi = curvature
        return psi
    px, py = bend(x), bend(y)
    if px.std() == 0 or py.std() == 0:
        return float("nan")
    return float(np.mean((px - px.mean()) * (py - py.mean())) / (px.std() * py.std()))


def skipped(x: np.ndarray, y: np.ndarray, critical: float = 13.8) -> tuple[float, int]:
    """MCD 대신 윈저화 공분산을 쓰는 skipped-correlation 근사."""
    mx, sx = mad_center(x)
    my, sy = mad_center(y)
    zx, zy = (x - mx) / sx, (y - my) / sy
    winsorized = np.clip(np.stack([zx, zy]), -0.8416, 0.8416)
    try:
        inverse = np.linalg.inv(np.cov(winsorized))
    except np.linalg.LinAlgError:
        return float(np.corrcoef(x, y)[0, 1]), len(x)
    points = np.stack([zx, zy], axis=1)
    distance2 = np.einsum("ij,jk,ik->i", points, inverse, points)
    keep = distance2 < critical
    if keep.sum() < 30:
        return float("nan"), int(keep.sum())
    return float(np.corrcoef(x[keep], y[keep])[0, 1]), int(keep.sum())


def approximate_p(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 5 or abs(r) >= 1:
        return float("nan")
    z = math.atanh(r) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def main() -> None:
    products = pd.read_csv(DATA / "products.csv", dtype={"product_id": str})
    txns = pd.read_csv(DATA / "transactions.csv", dtype={"product_id": str, "customer_id": str})
    images = np.load(DATA / "images.npy")
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import DataAnalyzer
    ana = DataAnalyzer(products=products, transactions=txns, images=images).load_data()
    features = ana.add_multimodal_features()
    source = features[["price", "img_mean", "category"]].dropna().copy()

    rows = []
    for coefficient, outlier, scale, control in itertools.product(
        ["pearson", "spearman", "kendall", "pb20", "skipped"],
        ["none", "1/99clip", "IQRdrop"], ["raw", "log"], ["none", "category"],
    ):
        frame = source.copy()
        if outlier == "1/99clip":
            lo, hi = np.percentile(frame["price"], [1, 99])
            frame["price"] = frame["price"].clip(lo, hi)
        elif outlier == "IQRdrop":
            q1, q3 = np.percentile(frame["price"], [25, 75])
            iqr = q3 - q1
            frame = frame[(frame["price"] >= q1 - 1.5 * iqr) & (frame["price"] <= q3 + 1.5 * iqr)]
        if scale == "log":
            frame["price"] = np.log(frame["price"])
            frame["img_mean"] = np.log1p(frame["img_mean"])
        if control == "category":
            frame["price"] -= frame.groupby("category")["price"].transform("mean")
            frame["img_mean"] -= frame.groupby("category")["img_mean"].transform("mean")
        x, y = frame["price"].to_numpy(dtype=float), frame["img_mean"].to_numpy(dtype=float)
        n = len(frame)
        if coefficient == "pearson":
            r = float(np.corrcoef(x, y)[0, 1])
        elif coefficient == "spearman":
            r = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        elif coefficient == "kendall":
            r = float(pd.Series(x).corr(pd.Series(y), method="kendall"))
        elif coefficient == "pb20":
            r = percentage_bend(x, y)
        else:
            r, n = skipped(x, y)
        rows.append({
            "coefficient": coefficient, "outlier": outlier, "scale": scale, "control": control,
            "n": n, "r": r, "p_approx_fisher": approximate_p(r, n),
        })
    result = pd.DataFrame(rows)
    if len(result) != 60 or not np.isfinite(result["r"]).all():
        raise ValueError("60 finite specification estimates are required")
    REP.mkdir(exist_ok=True)
    result.to_csv(REP / "spec_curve.csv", index=False)
    summary = {
        "n_specs": len(result), "r_min": float(result["r"].min()), "r_max": float(result["r"].max()),
        "r_median": float(result["r"].median()), "r_q1": float(result["r"].quantile(.25)),
        "r_q3": float(result["r"].quantile(.75)),
        "sign_agreement": float((np.sign(result["r"]) == np.sign(result["r"].median())).mean()),
        "p_below_0_05_share": float((result["p_approx_fisher"] < .05).mean()),
        "near_zero_share": float((result["r"].abs() < .1).mean()),
        "robust_estimates": {
            "pb20_median": float(result.loc[result["coefficient"] == "pb20", "r"].median()),
            "skipped_median": float(result.loc[result["coefficient"] == "skipped", "r"].median()),
            "skipped_min_n": int(result.loc[result["coefficient"] == "skipped", "n"].min()),
        },
        "method_note": "Skipped uses a winsorized-covariance approximation, not MCD. Fisher p-values are approximate and descriptive across non-independent specifications.",
        "minimum_spec": result.loc[result["r"].idxmin(), ["coefficient", "outlier", "scale", "control", "n", "r"]].to_dict(),
        "maximum_spec": result.loc[result["r"].idxmax(), ["coefficient", "outlier", "scale", "control", "n", "r"]].to_dict(),
    }
    (REP / "spec_curve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ordered = result.sort_values("r").reset_index(drop=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    colors = {"pearson": "#c0392b", "spearman": "#2980b9", "kendall": "#27ae60", "pb20": "#8e44ad", "skipped": "#d35400"}
    ax.scatter(ordered.index, ordered["r"], c=[colors[value] for value in ordered["coefficient"]], s=25)
    ax.axhline(0, lw=.8, c="black", ls=":")
    raw = result.query("coefficient == 'pearson' and outlier == 'none' and scale == 'raw' and control == 'none'").iloc[0]
    ax.axhline(raw["r"], lw=1, c="black", ls="--", alpha=.7, label=f"Raw Pearson r={raw['r']:.3f}")
    ax.set_title("Specification curve — real H&M image mean vs normalized price (60 specs)")
    ax.set_ylabel("Correlation r")
    ax.set_ylim(ordered["r"].min() - .05, ordered["r"].max() + .05)
    for name, color in colors.items():
        ax.scatter([], [], c=color, s=25, label=name)
    ax.legend(fontsize=6, ncol=6, loc="best")
    for column in ["outlier", "scale", "control"]:
        values = list(ordered[column].unique())
        ax2.plot(ordered.index, [values.index(value) for value in ordered[column]], lw=.8, label=column)
    ax2.set_yticks([]); ax2.set_xlabel("Specification number (sorted by r)"); ax2.set_ylabel("Choices")
    ax2.legend(fontsize=6, ncol=3, loc="center left")
    FIG.mkdir(exist_ok=True)
    save(fig, "07_spec_curve.png")
    render_analysis_report()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
