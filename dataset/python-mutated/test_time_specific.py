import pandas as pd
import cudf
from cudf.testing._utils import assert_eq

def test_tz_localize():
    if False:
        print('Hello World!')
    pidx = pd.date_range('2001-01-01', '2001-01-02', freq='1s')
    pidx = pidx.astype('<M8[ns]')
    idx = cudf.from_pandas(pidx)
    assert pidx.dtype == idx.dtype
    assert_eq(pidx.tz_localize('America/New_York'), idx.tz_localize('America/New_York'))

def test_tz_convert():
    if False:
        i = 10
        return i + 15
    pidx = pd.date_range('2023-01-01', periods=3, freq='H')
    idx = cudf.from_pandas(pidx)
    pidx = pidx.tz_localize('UTC')
    idx = idx.tz_localize('UTC')
    assert_eq(pidx.tz_convert('America/New_York'), idx.tz_convert('America/New_York'))

def test_delocalize_naive():
    if False:
        i = 10
        return i + 15
    pidx = pd.date_range('2023-01-01', periods=3, freq='H')
    idx = cudf.from_pandas(pidx)
    assert_eq(pidx.tz_localize(None), idx.tz_localize(None))