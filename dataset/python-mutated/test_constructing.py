import numpy as np
import cudf

def test_construct_int_series_with_nulls_compat_mode():
    if False:
        while True:
            i = 10
    with cudf.option_context('mode.pandas_compatible', True):
        s = cudf.Series([1, 2, None])
    assert s.dtype == np.dtype('float64')