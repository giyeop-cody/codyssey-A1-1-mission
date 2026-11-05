"""A1-1 OOP 분석 파이프라인 — 실제 H&M 상품·거래·이미지를 지원한다.

모든 공개 메서드는 입력 DataFrame을 바꾸지 않고 사본을 반환한다. 합성 전용
created_at/rating/stock을 요구하지 않으며, 없는 사실은 만들지 않는다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEGMENT_LABELS = ["대체고객", "성장고객", "우수고객", "VIP"]


def tied_quantile_scores(values: pd.Series, quantiles: int, *, reverse: bool = False) -> pd.Series:
    """동률을 같은 점수로 유지하는 1..q 분위 점수.

    H&M의 짧은 관찰 기간에서는 구매 빈도의 정수 동률이 많아 qcut(..., duplicates='drop')이
    분위 수를 임의로 줄일 수 있다. 평균 순위의 백분율을 사용해 동률을 억지로 나누지
    않으면서도 방향과 범위를 고정한다. reverse=True이면 작은 원값이 높은 점수다.
    """
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    if values.isna().any() or len(values) == 0:
        raise ValueError("scores require non-empty, non-missing values")
    percentile = values.rank(method="average", ascending=True) / len(values)
    scores = np.ceil(percentile * quantiles).clip(1, quantiles).astype(int)
    if reverse:
        scores = quantiles + 1 - scores
    return pd.Series(scores.to_numpy(), index=values.index, dtype=int)


@dataclass
class DataAnalyzer:
    """상품(products), 구매행(transactions), 실제 이미지 배열의 EDA·RFM 분석기."""

    products: pd.DataFrame
    transactions: pd.DataFrame
    reference_date: pd.Timestamp | None = None
    images: np.ndarray | None = None
    log: dict = field(default_factory=dict)

    def load_data(self) -> "DataAnalyzer":
        """실제 스키마를 검사하고 날짜를 캐스팅한다. shape은 유지된다."""
        p, t = self.products.copy(), self.transactions.copy()
        if "created_at" in p:  # H&M에는 없는 선택 컬럼이다. 만들지 않는다.
            p["created_at"] = pd.to_datetime(p["created_at"])
        if "order_date" not in t:
            if "txn_date" not in t:
                raise ValueError("transactions require order_date (or legacy txn_date)")
            t["order_date"] = t["txn_date"]
        t["order_date"] = pd.to_datetime(t["order_date"], errors="raise")
        if "txn_date" in t and not pd.to_datetime(t["txn_date"]).equals(t["order_date"]):
            raise ValueError("order_date and txn_date disagree")
        t["txn_date"] = t["order_date"]
        if "txn_id" not in t:
            t["txn_id"] = np.arange(len(t))
        required = {"customer_id", "amount", "txn_id"}
        if not required <= set(t.columns):
            raise ValueError(f"transactions missing columns: {required - set(t.columns)}")
        if t["txn_id"].duplicated().any():
            raise ValueError("txn_id must identify each source transaction line uniquely")
        if t["amount"].isna().any() or not pd.api.types.is_numeric_dtype(t["amount"]):
            raise ValueError("amount must be non-missing numeric data")
        if self.reference_date is None:
            self.reference_date = t["txn_date"].max() + pd.Timedelta(days=1)
        else:
            self.reference_date = pd.Timestamp(self.reference_date)
        self.products, self.transactions = p, t
        self.log["n_products"], self.log["n_txns"] = len(p), len(t)
        self.log["n_missing_cols"] = int(p.isna().any().sum())
        return self

    def handle_missing_values(self, strategy: str = "groupby_median", min_group: int = 30,
                              fallback: str = "median", columns: list[str] | None = None) -> pd.DataFrame:
        """결측을 처리한다. 수치는 중앙값, 문자/범주는 'unknown'; shape은 유지된다.

        groupby_median은 category별 중앙값을 쓰고 관측 수가 min_group보다 작은
        그룹은 전체 fallback으로 되돌린다. 모든 실제 가격이 관측된 메인 데이터와
        달리, 가격 홀드아웃 평가는 run_analysis가 만든 평가용 사본에서만 수행한다.
        """
        if strategy not in {"median", "groupby_median"}:
            raise ValueError(f"미지원 strategy: {strategy}")
        if fallback not in {"median", "mean", "zero"}:
            raise ValueError(f"미지원 fallback: {fallback}")
        if min_group < 1:
            raise ValueError("min_group must be >= 1")
        df = self.products.copy()
        cols = columns or [c for c in df.columns if df[c].isna().any()]
        text_columns = {"description", "name", "product_name", "category", "image_path"}
        for col in cols:
            if col in text_columns:
                df[col] = df[col].astype("object").fillna("unknown")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna("unknown")
                continue
            if not df[col].notna().any():
                df[col] = df[col].fillna(0.0)
                self.log[f"{col}_all_missing_fallback"] = "zero (no observed values)"
                continue
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
                continue
            if "category" not in df:
                raise ValueError("groupby_median requires category")
            grouped = df.groupby("category")[col]
            sizes = grouped.size()
            med = grouped.transform("median")
            small = sizes[sizes < min_group].index
            global_value = {
                "median": df[col].median, "mean": df[col].mean, "zero": lambda: 0.0,
            }[fallback]()
            med[df["category"].isin(small)] = global_value
            df[col] = df[col].fillna(med).fillna(global_value)
            self.log[f"{col}_small_groups"] = sorted(map(str, small))
        return df

    def detect_outliers(self, column: str, k: float = 1.5, policy: str = "flag",
                        groupby: str | None = None) -> pd.DataFrame:
        """IQR 경계 [Q1-k·IQR, Q3+k·IQR]로 flag/clip/drop한다."""
        if k <= 0:
            raise ValueError("k must be positive")
        df = self.products.copy()
        if groupby:
            grouped = df.groupby(groupby)[column]
            lo = grouped.transform(lambda s: s.quantile(0.25) - k * (s.quantile(0.75) - s.quantile(0.25)))
            hi = grouped.transform(lambda s: s.quantile(0.75) + k * (s.quantile(0.75) - s.quantile(0.25)))
        else:
            q1, q3 = df[column].quantile([0.25, 0.75])
            lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = (df[column] < lo) | (df[column] > hi)
        df[f"{column}_is_outlier"] = mask
        if policy == "clip":
            df[column] = df[column].clip(lower=lo, upper=hi)
        elif policy == "drop":
            df = df[~mask].copy()
        elif policy != "flag":
            raise ValueError(f"미지원 policy: {policy}")
        self.log[f"{column}_outlier_pct"] = round(float(mask.mean() * 100), 2)
        return df

    def add_multimodal_features(self, image_mean: np.ndarray | None = None) -> pd.DataFrame:
        """(N,H,W,3) 실제 RGB에서 평균·표준편차·채도 범위를, 텍스트에서 길이를 만든다."""
        df = self.products.copy()
        if self.images is not None:
            arr = np.asarray(self.images)
            if len(arr) != len(df) or arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError("images must have shape (n_products,H,W,3) and preserve product order")
            scaled = arr.astype(np.float64) / 255.0
            started = time.perf_counter()
            mean_vec = scaled.mean(axis=(1, 2, 3))
            std_vec = scaled.std(axis=(1, 2, 3))
            vector_seconds = time.perf_counter() - started
            started = time.perf_counter()
            mean_loop = np.array([image.mean() for image in scaled])
            loop_seconds = time.perf_counter() - started
            if not np.allclose(mean_vec, mean_loop, atol=1e-12):
                raise AssertionError("벡터화·루프 결과 불일치")
            self.log["img_time_vectorized_s"] = round(vector_seconds, 6)
            self.log["img_time_loop_s"] = round(loop_seconds, 6)
            self.log["img_speedup_loop_vs_vec"] = round(loop_seconds / max(vector_seconds, 1e-12), 2)
            df["img_mean"] = mean_vec
            df["img_std"] = std_vec
            df["img_saturation"] = (scaled.max(axis=3) - scaled.min(axis=3)).mean(axis=(1, 2))
        elif image_mean is not None:
            values = np.asarray(image_mean)
            if len(values) != len(df):
                raise ValueError("image_mean must match n_products")
            df["img_mean"] = values
        name_col = "name" if "name" in df else "product_name"
        if name_col not in df:
            raise ValueError("products require name or product_name")
        names = df[name_col].fillna("").astype(str)
        df["name_word_count"] = names.str.split().str.len().astype(int)
        df["name_length"] = names.str.len().astype(int)
        descriptions = df["description"].fillna("").astype(str) if "description" in df else names
        df["word_count"] = descriptions.str.split().str.len().fillna(0).astype(int)
        df["avg_word_len"] = descriptions.apply(
            lambda text: np.mean([len(word) for word in text.split()]) if text.strip() else 0.0
        )
        return df

    def summarize(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """숫자 컬럼의 count/mean/std/min/quartiles/median/max를 반환한다."""
        frame = self.products if df is None else df
        numeric = frame.select_dtypes(include=[np.number])
        out = numeric.agg(["count", "mean", "std", "min", "median", "max"]).T
        out["25%"] = numeric.quantile(0.25)
        out["75%"] = numeric.quantile(0.75)
        return out[["count", "mean", "std", "min", "25%", "median", "75%", "max"]]

    def correlation(self, df: pd.DataFrame | None = None, other: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = self.products if df is None else df
        left = frame.select_dtypes(include=[np.number])
        if other is not None:
            left = left.join(other.select_dtypes(include=[np.number]), rsuffix="_txn")
        return left.corr()

    def calculate_rfm(self, quantiles: int = 5, drop_duplicates: bool = True) -> pd.DataFrame:
        """고객별 RFM과 동률 안전 점수를 계산한다. (고객 수,8)을 반환한다.

        Recency=(기준일-마지막 구매일).days이며 작을수록 좋으므로 점수만 역전한다.
        frequency는 원본에 주문/바구니 ID가 없으므로 고유 거래행 수다.
        drop_duplicates는 이전 API 호환용이며 동률 안전 평균순위 방식에서는 필요 없다.
        """
        del drop_duplicates
        t = self.transactions
        agg = t.groupby("customer_id").agg(
            last_order=("txn_date", "max"), frequency=("txn_id", "nunique"), monetary=("amount", "sum")
        )
        agg["recency"] = (self.reference_date - agg["last_order"]).dt.days
        agg["R_score"] = tied_quantile_scores(agg["recency"], quantiles, reverse=True)
        agg["F_score"] = tied_quantile_scores(agg["frequency"], quantiles)
        agg["M_score"] = tied_quantile_scores(agg["monetary"], quantiles)
        combined = agg[["R_score", "F_score", "M_score"]].mean(axis=1)
        segment_score = tied_quantile_scores(combined, 4)
        agg["segment"] = pd.Categorical.from_codes(segment_score - 1, categories=SEGMENT_LABELS, ordered=True)
        return agg.reset_index()[[
            "customer_id", "recency", "frequency", "monetary",
            "R_score", "F_score", "M_score", "segment",
        ]]

    def report_insights(self, rfm: pd.DataFrame) -> dict:
        """집계값만 사용해 근거/실행/검증을 구성한다. 인과 효과는 주장하지 않는다."""
        total = float(rfm["monetary"].sum())
        vip = rfm[rfm["segment"] == "VIP"]
        others = rfm[rfm["segment"] != "VIP"]
        return {
            "VIP_매출지표점유": {
                "근거": f"VIP {len(vip)}명({len(vip)/len(rfm)*100:.1f}%)이 정규화 매출지표의 {vip['monetary'].sum()/total*100:.1f}%를 차지한다",
                "실행": f"대상은 VIP {len(vip)}명이다. 실제 발송 전 더 긴 이력으로 재계산하고 재구매 캠페인을 시뮬레이션한다. 기대 효과는 재구매율·유지율 개선 여부를 확인하는 것이다.",
                "검증": "이후 기간의 유지율·재구매율·1인당 매출을 사전 등록한 대조군과 비교한다. 차이가 없으면 가설이 반증되며 현재 관찰 기간만으로 효과를 주장하지 않는다.",
            },
            "Recency_방향": {
                "근거": f"VIP 평균 Recency {vip['recency'].mean():.2f}일, 나머지 {others['recency'].mean():.2f}일",
                "실행": "대상은 VIP와 나머지 세그먼트의 배정 규칙이다. R만 역전하고 동률은 같은 점수로 유지해 세그먼트 오배정을 줄인다. 빈도는 바구니가 아닌 거래행 수로 명시한다.",
                "검증": "VIP 평균 Recency < 전체 중앙값 및 역순 실패 사례를 자동 검사하고, 방향이 맞지 않으면 현재 세분화를 반증된 것으로 거부한다.",
            },
            "이미지_참여가설": {
                "근거": "실제 상품 이미지와 가격의 상관을 계산했지만 노출·클릭·세션은 없다.",
                "실행": "대상은 VIP·우수고객이다. 동일 상품 원본/보정안의 무작위 노출 실험을 별도 설계해 CTR·구매 전환 개선 여부를 검증한다.",
                "검증": "추가로 노출·클릭·세션·구매와 배정 로그를 수집해 CTR과 구매 전환을 나누어 평가한다. 차이가 없으면 가설이 반증되며 현재 상관으로 효과를 추론하지 않는다.",
            },
        }
