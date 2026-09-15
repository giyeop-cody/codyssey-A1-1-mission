"""A1-1 합성 데이터 생성기 — 재현성의 핵심이므로 제출에 포함한다.

왜 합성인가: 미션은 "1000건 이상·8컬럼 이상·3가지 타입"을 요구하면서 데이터를 주지 않는다.
그래서 (1) 요구 스키마를 만족하고 (2) 전처리 실험을 **검증 가능한** 형태로 만들고 (3) RFM이
의미를 갖는 구조로 생성한다. 검증 가능성을 위해 실제 가격(price_true)과 관측 가격(price)을
분리한다 — 결측/이상치 실험의 정답지로 쓰기 위함이다(분석에서는 price_true를 쓰지 않는다).

실행:
    python src/data_gen.py --n-products 1200 --n-txns 30000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

# 카테고리별 (가격 중앙값, 가격 산포, 거래량 가중치) — GroupBy 대치가 의미 있도록 스케일을 벌린다
CATEGORIES = {
    "book":        (1.4e4,  0.35, 0.30),
    "fashion":     (4.5e4,  0.45, 0.34),
    "electronics": (4.2e5,  0.60, 0.20),
    "toys":        (2.6e4,  0.40, 0.16, ),
}
DESC_WORDS = {
    "book": ["소설", "에세이", "하드커버", "양장", "페이지", "번역", "출간"],
    "fashion": ["코튼", "오버핏", "봄버", "리넨", "슬림", "유니섹스", "계절"],
    "electronics": ["무선", "프로", "배터리", "패널", "고속", "발열", "규격"],
    "toys": ["조립", "키트", "아이", "안전", "색상", "확장", "세트"],
}
PROMO_WORDS = ["특가", "한정", "할인", "증정", "품절임박"]  # 프로모션 문구(텍스트 피처용)


def make_products(n: int, rng: np.random.Generator) -> pd.DataFrame:
    cats = list(CATEGORIES)
    weights = [CATEGORIES[c][2] for c in cats]
    cat = rng.choice(cats, size=n, p=np.array(weights) / np.sum(weights))
    price_true = np.empty(n)
    for c in cats:
        med, sd, _ = CATEGORIES[c]
        m = cat == c
        price_true[m] = np.exp(rng.normal(np.log(med), sd, m.sum()))
    price_true = np.round(price_true, -2)

    # 결측·이상치 주입(관측가) — 전처리 실험의 "원인"을 코드로 보여주기 위한 의도적 오염
    price = price_true.copy()
    miss_p = rng.random(n) < 0.03
    miss_r = rng.random(n) < 0.05
    out = rng.random(n) < 0.02
    price[out] *= np.exp(rng.normal(3.2, 0.4, out.sum()))          # 로그정상 꼬리 → IQR이 잡는 이상치
    price[miss_p] = np.nan

    names = [f"{c}-{i:04d}" for i, c in enumerate(cat)]
    desc = []
    for i in range(n):
        k = int(rng.integers(3, 12))
        words = list(rng.choice(DESC_WORDS[cat[i]], size=k)) + list(rng.choice(PROMO_WORDS, size=int(rng.poisson(0.4))))
        desc.append(" ".join(words))
    rating = np.round(np.clip(rng.normal(4.2, 0.55, n), 1, 5), 2)
    rating[miss_r] = np.nan

    created = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    return pd.DataFrame({
        "product_id": np.arange(1, n + 1),
        "name": names,
        "category": cat,
        "price": price,
        "rating": rating,
        "stock": rng.integers(0, 400, n),
        "image_path": [f"images/{i:04d}.png" for i in range(1, n + 1)],
        "description": desc,
        "created_at": created,
    }), price_true


def make_thumbnails(n: int, price_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N, 32, 32, 3) uint8. 밝기·채도·분산을 가격과 약한 상관으로 tying → 멀티모달 피처가 '무의미'하지 않게."""
    brightness = 90 + 60 * (np.log10(price_true) - np.log10(price_true.min())) / (
        np.log10(price_true.max()) - np.log10(price_true.min()))
    imgs = np.empty((n, 32, 32, 3), dtype=np.uint8)
    for i in range(n):
        base = np.clip(brightness[i] + rng.normal(0, 18, (32, 32)), 0, 255)
        tint = rng.normal(0, 22, 3)
        for ch in range(3):
            imgs[i, :, :, ch] = np.clip(base + tint[ch], 0, 255).astype(np.uint8)
    return imgs


def make_transactions(products: pd.DataFrame, n_txn: int, rng: np.random.Generator):
    # 고객 거래수 롱테일(lognormal 가중치). 처음엔 zipf(1.6) 을 썼다가 VIP 137명이 매출의 98.2%를
    # 점유하는(=세분화가 무의미한) 데이터가 나왔다 → 지수를 낮춰 'VIP 15%가 40%대 매출' 구간으로 조정한다.
    n_cust = max(400, n_txn // 25)
    pid_w = 1.0 / np.log1p(np.arange(1, len(products) + 1))
    pid = rng.choice(products["product_id"].to_numpy(), size=n_txn, p=pid_w / pid_w.sum())
    cw = rng.lognormal(0.0, 0.9, n_cust)
    cust = rng.choice(np.arange(1, n_cust + 1), size=n_txn, p=cw / cw.sum())
    dates = pd.to_datetime("2023-07-01") + pd.to_timedelta(rng.integers(0, 365, n_txn), unit="D")
    qty = rng.integers(1, 4, n_txn)
    price_map = products.set_index("product_id")["price"]
    amount = np.round(price_map.reindex(pid).to_numpy() * qty, 0)
    df = pd.DataFrame({
        "txn_id": np.arange(1, n_txn + 1),
        "product_id": pid,
        "customer_id": cust,
        "quantity": qty,
        "amount": amount,
        "txn_date": dates,
    }).sort_values("txn_date").reset_index(drop=True)
    return df, n_cust


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-products", type=int, default=1200)
    ap.add_argument("--n-txns", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    products, price_true = make_products(a.n_products, rng)
    txns, n_cust = make_transactions(products, a.n_txns, rng)
    imgs = make_thumbnails(a.n_products, price_true, rng)

    products.to_csv(out / "products.csv", index=False)
    txns.to_csv(out / "transactions.csv", index=False)
    np.save(out / "images.npy", imgs)
    # 정답지는 분석에 쓰지 않는다. 검증(MAE)에만 사용 → 별도로 저장해 "사용 안 함"을 명시
    pd.DataFrame({"product_id": products["product_id"], "price_true": price_true}).to_csv(
        out / "_groundtruth_price.csv", index=False)
    meta = {
        "seed": a.seed, "n_products": int(a.n_products), "n_txns": int(a.n_txns), "n_customers": int(n_cust),
        "missing_price_pct": round(float(products["price"].isna().mean() * 100), 2),
        "missing_rating_pct": round(float(products["rating"].isna().mean() * 100), 2),
        "columns_products": list(products.columns), "columns_txns": list(txns.columns),
        "dtypes_products": {k: str(v) for k, v in products.dtypes.items()},
        "images_shape": list(imgs.shape),
        "note": "price_true는 _groundtruth_price.csv에만 존재하며 DataAnalyzer는 읽지 않는다",
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
