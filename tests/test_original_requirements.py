import numpy as np
import pandas as pd
import pytest
from src.pipeline import DataAnalyzer


@pytest.mark.parametrize('strategy', ['median', 'groupby_median'])
def test_all_missing_numeric_column_has_explicit_fallback(strategy):
    p = pd.DataFrame({'category': ['x', 'x'], 'price': [np.nan, np.nan]})
    a = DataAnalyzer(p, pd.DataFrame())
    out = a.handle_missing_values(strategy=strategy)
    assert out.price.tolist() == [0., 0.]
    assert p.price.isna().all()
    assert 'price_all_missing_fallback' in a.log


def test_taskmap_order_date_schema_and_product_name_features():
    p = pd.DataFrame({'created_at': ['2024-01-01'], 'name': ['좋은 상품 이름']})
    t = pd.DataFrame({'customer_id': [1], 'order_date': ['2024-02-01'], 'amount': [100.]})
    a = DataAnalyzer(p, t).load_data()
    assert a.transactions.txn_date.equals(a.transactions.order_date)
    assert a.reference_date == pd.Timestamp('2024-02-02')
    out = a.add_multimodal_features()
    assert out.name_word_count.tolist() == [3]
    assert out.name_length.tolist() == [8]


def test_conflicting_date_aliases_are_rejected():
    p = pd.DataFrame({'created_at': ['2024-01-01']})
    t = pd.DataFrame({'order_date': ['2024-02-01'], 'txn_date': ['2024-02-02']})
    with pytest.raises(ValueError, match='disagree'):
        DataAnalyzer(p, t).load_data()
