import pandas as pd
from statsmodels.tsa.x13 import _make_var_names

def test_make_var_names():
    if False:
        return 10
    exog = pd.Series([1, 2, 3], name='abc')
    assert _make_var_names(exog) == exog.name