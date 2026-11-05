"""작은 인메모리 단위 fixture만 사용한다. 분석 산출물은 이 fixture로 만들지 않는다."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import DataAnalyzer, tied_quantile_scores


def transactions_for(product_rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "txn_id": np.arange(1, product_rows + 1),
        "product_id": ["0100000001"] * product_rows,
        "customer_id": ["customer"] * product_rows,
        "amount": np.linspace(0.01, 0.02, product_rows),
        "order_date": pd.date_range("2020-09-01", periods=product_rows).astype(str),
    })


def test_real_schema_does_not_require_created_at_rating_or_stock():
    products = pd.DataFrame({"product_id": ["0100000001"], "name": ["Strap top"], "price": [0.03]})
    analyzer = DataAnalyzer(products, transactions_for()).load_data()
    assert analyzer.reference_date == pd.Timestamp("2020-09-02")
    assert "created_at" not in analyzer.products
    assert analyzer.transactions["txn_date"].equals(analyzer.transactions["order_date"])


def test_conflicting_date_aliases_are_rejected():
    tx = transactions_for()
    tx["txn_date"] = "2020-09-09"
    with pytest.raises(ValueError, match="disagree"):
        DataAnalyzer(pd.DataFrame(), tx).load_data()


def test_duplicate_source_transaction_ids_are_rejected():
    tx = pd.concat([transactions_for(), transactions_for()], ignore_index=True)
    with pytest.raises(ValueError, match="txn_id"):
        DataAnalyzer(pd.DataFrame(), tx).load_data()


@pytest.mark.parametrize("strategy", ["median", "groupby_median"])
def test_all_missing_numeric_has_explicit_fallback(strategy):
    source = pd.DataFrame({"category": ["x", "x"], "price": [np.nan, np.nan]})
    analyzer = DataAnalyzer(source, pd.DataFrame())
    result = analyzer.handle_missing_values(strategy=strategy)
    assert result["price"].tolist() == [0.0, 0.0]
    assert source["price"].isna().all()
    assert "price_all_missing_fallback" in analyzer.log


def test_group_fallback_options_and_input_immutability():
    source = pd.DataFrame({"category": ["a", "a", "b", "b"], "price": [1.0, 3.0, 100.0, np.nan]})
    analyzer = DataAnalyzer(source, pd.DataFrame())
    assert analyzer.handle_missing_values(min_group=3, fallback="median")["price"].iloc[-1] == 3
    assert analyzer.handle_missing_values(min_group=3, fallback="mean")["price"].iloc[-1] == pytest.approx(104 / 3)
    assert analyzer.handle_missing_values(min_group=3, fallback="zero")["price"].iloc[-1] == 0
    assert pd.isna(source["price"].iloc[-1])


def test_actual_text_missing_is_filled_without_touching_numbers():
    source = pd.DataFrame({"category": ["x"], "description": [np.nan], "price": [0.03]})
    result = DataAnalyzer(source, pd.DataFrame()).handle_missing_values(columns=["description"])
    assert result.loc[0, "description"] == "unknown"
    assert result.loc[0, "price"] == 0.03


@pytest.mark.parametrize("policy,expected_rows", [("flag", 4), ("clip", 4), ("drop", 3)])
def test_outlier_policies(policy, expected_rows):
    source = pd.DataFrame({"price": [1.0, 1.1, 1.2, 20.0]})
    result = DataAnalyzer(source, pd.DataFrame()).detect_outliers("price", k=1.5, policy=policy)
    assert len(result) == expected_rows
    assert source["price"].iloc[-1] == 20.0
    if policy == "clip":
        assert result["price"].max() < 20.0
    if policy == "drop":
        assert not result["price_is_outlier"].any()


def test_invalid_outlier_arguments_fail():
    analyzer = DataAnalyzer(pd.DataFrame({"price": [1.0]}), pd.DataFrame())
    with pytest.raises(ValueError):
        analyzer.detect_outliers("price", k=0)
    with pytest.raises(ValueError):
        analyzer.detect_outliers("price", policy="unknown")


def test_real_rgb_feature_axes_and_text_features():
    products = pd.DataFrame({
        "name": ["Strap top", "Wide trousers"],
        "description": ["Jersey top with straps", np.nan],
    })
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    images[0] = 255
    images[1, :, :, 0] = 255
    result = DataAnalyzer(products, pd.DataFrame(), images=images).add_multimodal_features()
    assert result["img_mean"].tolist() == pytest.approx([1.0, 1 / 3])
    assert result["img_std"].iloc[0] == 0
    assert result["name_word_count"].tolist() == [2, 2]
    assert result["word_count"].tolist() == [4, 0]


def test_image_order_and_shape_are_validated():
    with pytest.raises(ValueError, match="shape"):
        DataAnalyzer(pd.DataFrame({"name": ["a", "b"]}), pd.DataFrame(), images=np.zeros((1, 2, 2, 3))).add_multimodal_features()


def test_summary_contains_real_quartiles():
    source = pd.DataFrame({"price": [1.0, 2.0, 3.0, 100.0]})
    summary = DataAnalyzer(source, pd.DataFrame()).summarize()
    assert summary.loc["price", "25%"] == 1.75
    assert summary.loc["price", "median"] == 2.5
    assert summary.loc["price", "75%"] == 27.25


def test_tied_quantile_scores_keep_equal_values_equal_and_reverse_direction():
    values = pd.Series([1, 1, 2, 3, 4])
    scores = tied_quantile_scores(values, 5)
    reverse = tied_quantile_scores(values, 5, reverse=True)
    assert scores.iloc[0] == scores.iloc[1]
    assert reverse.iloc[0] == reverse.iloc[1]
    assert scores.between(1, 5).all()
    assert reverse.iloc[0] > reverse.iloc[-1]


def test_rfm_recency_direction_and_four_segments_on_tied_short_window_data():
    rows = []
    for customer_number in range(40):
        frequency = customer_number % 5 + 1
        last_day = 1 + customer_number % 15
        for line in range(frequency):
            rows.append({
                "txn_id": f"{customer_number}-{line}", "customer_id": f"c{customer_number}",
                "amount": (customer_number + 1) * (line + 1) * 0.001,
                "order_date": f"2020-09-{last_day:02d}",
            })
    tx = pd.DataFrame(rows)
    analyzer = DataAnalyzer(pd.DataFrame(), tx, reference_date="2020-09-23").load_data()
    rfm = analyzer.calculate_rfm()
    assert len(rfm) == 40
    assert rfm["segment"].nunique() == 4
    assert rfm["frequency"].sum() == len(tx)
    assert rfm.loc[rfm["segment"] == "VIP", "recency"].mean() < rfm["recency"].median()
