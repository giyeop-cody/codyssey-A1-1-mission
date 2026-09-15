"""A1-1 요구 2번: OOP 데이터 분석 모듈 `DataAnalyzer`.

규칙(설계 기록 docs/07에서):
- 인스턴스 밖에서 DataFrame을 변형하지 않는다. 모든 메서드는 **사본**을 반환한다(노트북 재실행 가능해야 함).
- 임계값·전략은 전부 인자. 본문에 마법 숫자를 남기지 않는다.
- docstring에 수식 1줄 + shape 변화 1줄.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataAnalyzer:
    """합성 쇼핑몰 데이터(products/transactions/images)의 전처리·피처·RFM 파이프라인.

    Parameters
    ----------
    products, transactions : pd.DataFrame
        원본 테이블. 이 생성자는 복사하지 **않고** 참조만 저장한다(읽기 전용 대우).
    reference_date : str | pd.Timestamp | None
        RFM의 기준 시점. None이면 거래 최대일 + 1일.
    """
    products: pd.DataFrame
    transactions: pd.DataFrame
    reference_date: pd.Timestamp | None = None
    images: np.ndarray | None = None
    log: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- stage 1
    def load_data(self) -> "DataAnalyzer":
        """타입·스키마를 점검하고 필요 타입을 캐스팅한다.  (self 를 그대로 반환 → 체이닝)"""
        p, t = self.products.copy(), self.transactions.copy()
        p["created_at"] = pd.to_datetime(p["created_at"])
        t["txn_date"] = pd.to_datetime(t["txn_date"])
        if self.reference_date is None:
            self.reference_date = t["txn_date"].max() + pd.Timedelta(days=1)
        else:
            self.reference_date = pd.Timestamp(self.reference_date)
        self.products, self.transactions = p, t
        self.log["n_products"], self.log["n_txns"] = len(p), len(t)
        self.log["n_missing_cols"] = int(p.isna().any().sum())
        return self

    # ---------------------------------------------------------------- stage 2
    def handle_missing_values(self, strategy: str = "groupby_median", min_group: int = 30,
                              fallback: str = "median", columns: list[str] | None = None) -> pd.DataFrame:
        """결측 처리.  `median`: 열 전체 중앙값 / `groupby_median`: (category, 열)별 중앙값.

        group이 min_group 미만이면 그 group만 fallback(전체 중앙값)으로 되돌린다 — 작은 group의
        중앙값은 noise라는 판단(설계 기록 docs/03의 결정 4).
        Returns: products 사본 (N, C) → 모양은 그대로, NaN 수만 0으로.
        """
        df = self.products.copy()
        cols = columns or [c for c in df.columns if df[c].isna().any()]
        for col in cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna("unknown")
                continue
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "groupby_median":
                g = df.groupby("category")[col]
                sizes = g.size()
                med = g.transform("median")
                small = sizes[sizes < min_group].index
                med[df["category"].isin(small)] = df[col].median()
                df[col] = df[col].fillna(med)
                self.log[f"{col}_small_groups"] = sorted(map(str, small))
            else:
                raise ValueError(f"미지원 strategy: {strategy}")
        return df

    # ---------------------------------------------------------------- stage 3
    def detect_outliers(self, column: str, k: float = 1.5, policy: str = "flag",
                        groupby: str | None = None) -> pd.DataFrame:
        """IQR 규칙: 경계 = [Q1 − k·IQR, Q3 + k·IQR], IQR = Q3 − Q1.

        policy: flag(플래그만) | clip(경계로 자름) | drop(제거). groupby를 주면 카테고리별로 계산한다.
        Returns: `{column}` + `{column}_is_outlier` 추가 (N, C) → (N, C+1) (drop이면 행 수 감소)
        """
        df = self.products.copy()
        if groupby:
            grp = df.groupby(groupby)[column]
            lo, hi = grp.transform(lambda s: s.quantile(0.25) - k * (s.quantile(0.75) - s.quantile(0.25))), \
                      grp.transform(lambda s: s.quantile(0.75) + k * (s.quantile(0.75) - s.quantile(0.25)))
        else:
            q1, q3 = df[column].quantile([0.25, 0.75])
            lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = (df[column] < lo) | (df[column] > hi)
        df[f"{column}_is_outlier"] = mask
        if policy == "flag":
            pass
        elif policy == "clip":
            df[column] = df[column].clip(lower=lo, upper=hi)
        elif policy == "drop":
            df = df[~mask].copy()
        else:
            raise ValueError(f"미지원 policy: {policy}")
        self.log[f"{column}_outlier_pct"] = round(float(mask.mean() * 100), 2)
        return df

    # ---------------------------------------------------------------- stage 4
    def add_multimodal_features(self, image_mean: np.ndarray | None = None) -> pd.DataFrame:
        """이미지·텍스트 피처.  (N,32,32,3) uint8 → (N,) 스칼라 3종 / 문자열 → 수치 3종.

        axis=(1,2,3)으로 줄이는 이유: 1장 전체의 통계량이 필요하지 픽셀별 값이 필요하지 않기 때문이다.
        for-loop와 벡터화 두 경로를 모두 재고해 시간을 남긴다(요구 3번의 '성능 측정' 대응).
        """
        df = self.products.copy()
        if self.images is not None:
            arr = self.images.astype(np.float64) / 255.0
            t0 = time.perf_counter()
            mean_vec, std_vec = arr.mean(axis=(1, 2, 3)), arr.std(axis=(1, 2, 3))
            t_vec = time.perf_counter() - t0
            t0 = time.perf_counter()
            mean_loop = np.array([img.mean() for img in arr])
            t_loop = time.perf_counter() - t0
            assert np.allclose(mean_vec, mean_loop, atol=1e-12), "벡터화·루프 결과 불일치"
            self.log["img_time_vectorized_s"] = round(t_vec, 4)
            self.log["img_time_loop_s"] = round(t_loop, 4)
            self.log["img_speedup_loop_vs_vec"] = round(t_loop / max(t_vec, 1e-9), 2)
            df["img_mean"] = mean_vec
            df["img_std"] = std_vec
            df["img_saturation"] = (arr.max(axis=3) - arr.min(axis=3)).mean(axis=(1, 2))
        elif image_mean is not None:
            df["img_mean"] = np.asarray(image_mean)[: len(df)]
        d = df["description"].fillna("").astype(str)
        df["word_count"] = d.str.split().str.len().fillna(0).astype(int)
        df["avg_word_len"] = d.apply(lambda s: np.mean([len(w) for w in s.split()]) if s.strip() else 0.0)
        df["has_promo"] = d.str.contains("|".join(["한정", "할인", "증정", "품절임박", "특가"]), regex=True).astype(int)
        return df

    # ---------------------------------------------------------------- stage 5
    def summarize(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """수치 컬럼 요약 + 가격-행동 상관.  (N,C) → (수치컬럼, 통계량) 표"""
        df = self.products if df is None else df
        num = df.select_dtypes(include=[np.number])
        out = num.agg(["count", "mean", "std", "min", "median", "max"]).T
        return out

    def correlation(self, df: pd.DataFrame | None = None, other: pd.DataFrame | None = None) -> pd.DataFrame:
        df = self.products if df is None else df
        a = df.select_dtypes(include=[np.number])
        if other is not None:
            a = a.join(other.select_dtypes(include=[np.number]), rsuffix="_txn")
        return a.corr()

    # ---------------------------------------------------------------- stage 6
    def calculate_rfm(self, quantiles: int = 5, drop_duplicates: bool = True) -> pd.DataFrame:
        """고객별 RFM.  Recency = (reference_date − 마지막 구매일).days → **작을수록 좋음**이라 점수를 뒤집는다.

        점수 = 6 − qcut(R, q) (R만) / qcut(x, q)+1 (F, M). ties 때문에 `duplicates='drop'`이 필요하다.
        Returns: (n_customers, 6) — customer_id, R, F, M, segment, revenue
        """
        t = self.transactions
        agg = t.groupby("customer_id").agg(
            last_order=("txn_date", "max"), frequency=("txn_id", "nunique"), monetary=("amount", "sum"))
        agg["recency"] = (self.reference_date - agg["last_order"]).dt.days
        dup = "drop" if drop_duplicates else "raise"
        for col, reverse in (("recency", True), ("frequency", False), ("monetary", False)):
            ranked = pd.qcut(agg[col], q=quantiles, labels=False, duplicates=dup)
            agg[f"{col[0].upper()}_score"] = (quantiles + 1 - ranked) if reverse else (ranked + 1)
        agg["segment"] = (agg[["R_score", "F_score", "M_score"]].mean(axis=1)
                           .pipe(lambda s: pd.qcut(s, 4, labels=["대체고객", "성장고객", "우수고객", "VIP"],
                                                   duplicates="drop")))
        return agg.reset_index()[["customer_id", "recency", "frequency", "monetary",
                                  "R_score", "F_score", "M_score", "segment"]]

    # ---------------------------------------------------------------- stage 7
    def report_insights(self, rfm: pd.DataFrame) -> dict:
        """인사이트 3종을 '근거/실행/검증' 형태로 반환.  숫자는 이 함수가 직접 계산한 것만 쓴다."""
        total = float(rfm["monetary"].sum())
        vip = rfm[rfm["segment"] == "VIP"]
        others = rfm[rfm["segment"] != "VIP"]
        return {
            "VIP_매출점유": {
                "근거": f"VIP {len(vip)}명({len(vip)/len(rfm)*100:.1f}%)이 매출 {vip['monetary'].sum()/total*100:.1f}%를 낸다",
                "실행": "VIP 전용 재구매 캠페인(주 1회)을 우선 편성하고 일반 세그먼트는 주 1회를 월 1회로 축소",
                "검증": "4주 후 VIP 유지율과 1인당 매출을 세그먼트별로 재측정, 유지율이 오르면 계속·아니면 중단",
            },
            "Recency_역전": {
                "근거": f"VIP 평균 Recency {vip['recency'].mean():.1f}일 vs 나머지 {others['recency'].mean():.1f}일 — R 점수를 뒤집지 않으면 VIP가 '가장 오래된 고객'이 된다",
                "실행": "점수 계산에 reverse를 적용하고, 단위 테스트로 'VIP 평균 R < 전체 중앙값'을 고정",
                "검증": "check_requirements.py의 assert가 CI에서 매번 실행",
            },
            "가격_이미지": {
                "근거": None,   # run_analysis 에서 상관계수로 채운다(문서에서 숫자 없이 쓰지 않기 위해)
                "실행": "상관 |r|<0.1 이면 '이미지 밝기는 가격을 설명하지 못한다'를 결론으로 명시하고, 상세 페이지 실험을 접는다",
                "검증": "같은 상관표를 학습 전/후 두 시점에 재계산",
            },
        }
