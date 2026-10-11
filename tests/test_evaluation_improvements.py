import numpy as np
import pandas as pd
import pytest
from src.pipeline import DataAnalyzer

def test_quartiles_and_fallback_are_real_options():
    df = pd.DataFrame({'category':['a','a','b','b'], 'price':[1.,3.,100.,np.nan]})
    a = DataAnalyzer(df, pd.DataFrame())
    assert a.summarize().loc['price','25%'] == 2
    assert a.summarize().loc['price','75%'] == 51.5
    assert a.handle_missing_values(min_group=3, fallback='median').price.iloc[-1] == 3
    assert a.handle_missing_values(min_group=3, fallback='mean').price.iloc[-1] == pytest.approx(104/3)
    assert a.handle_missing_values(min_group=3, fallback='zero').price.iloc[-1] == 0
    assert pd.isna(df.price.iloc[-1])
    with pytest.raises(ValueError): a.handle_missing_values(fallback='typo')
