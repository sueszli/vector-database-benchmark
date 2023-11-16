import numpy as np
from pandas import DataFrame, Series, period_range
import pandas._testing as tm

def test_iat(float_frame):
    if False:
        while True:
            i = 10
    for (i, row) in enumerate(float_frame.index):
        for (j, col) in enumerate(float_frame.columns):
            result = float_frame.iat[i, j]
            expected = float_frame.at[row, col]
            assert result == expected

def test_iat_duplicate_columns():
    if False:
        print('Hello World!')
    df = DataFrame([[1, 2]], columns=['x', 'x'])
    assert df.iat[0, 0] == 1

def test_iat_getitem_series_with_period_index():
    if False:
        print('Hello World!')
    index = period_range('1/1/2001', periods=10)
    ser = Series(np.random.default_rng(2).standard_normal(10), index=index)
    expected = ser[index[0]]
    result = ser.iat[0]
    assert expected == result

def test_iat_setitem_item_cache_cleared(indexer_ial, using_copy_on_write, warn_copy_on_write):
    if False:
        i = 10
        return i + 15
    data = {'x': np.arange(8, dtype=np.int64), 'y': np.int64(0)}
    df = DataFrame(data).copy()
    ser = df['y']
    with tm.assert_cow_warning(warn_copy_on_write and indexer_ial is tm.iloc):
        indexer_ial(df)[7, 0] = 9999
    with tm.assert_cow_warning(warn_copy_on_write and indexer_ial is tm.iloc):
        indexer_ial(df)[7, 1] = 1234
    assert df.iat[7, 1] == 1234
    if not using_copy_on_write:
        assert ser.iloc[-1] == 1234
    assert df.iloc[-1, -1] == 1234