import vaex
import numpy as np
import pyarrow as pa

def test_is_missing():
    if False:
        return 10
    s = vaex.string_column(['aap', None, 'noot', 'mies'])
    o = ['aap', None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    assert df.x.ismissing().tolist() == [False, False, False, False]
    assert df.m.ismissing().tolist() == [False, True, False, True]
    assert df.s.ismissing().tolist() == [False, True, False, False]
    assert df.o.ismissing().tolist() == [False, True, False, False]
    assert df.m.notmissing().tolist() == [not k for k in [False, True, False, True]]

def test_is_nan():
    if False:
        for i in range(10):
            print('nop')
    s = vaex.string_column(['aap', None, 'noot', 'mies'])
    o = ['aap', None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    assert df.x.isnan().tolist() == [False, False, True, True]
    assert df.m.isnan().tolist() == [False, False, True, False]
    assert df.s.isnan().tolist() == [False, False, False, False]
    assert df.o.isnan().tolist() == [False, False, False, True]
    assert df.o.notnan().tolist() == [not k for k in [False, False, False, True]]

def test_is_na():
    if False:
        for i in range(10):
            print('nop')
    s = vaex.string_column(['aap', None, 'noot', 'mies'])
    o = ['aap', None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    assert df.x.isna().tolist() == [False, False, True, True]
    assert df.m.isna().tolist() == [False, True, True, True]
    assert df.s.isna().tolist() == [False, True, False, False]
    assert df.o.isna().tolist() == [False, True, False, True]

def test_isna():
    if False:
        return 10
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    y_data = np.array([np.nan, 2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_mask = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    y = np.ma.MaskedArray(data=y_data, mask=y_mask)
    df = vaex.from_arrays(x=x, y=y)
    pandas_df = df.to_pandas_df()
    assert df.x.isna().tolist() == pandas_df.x.isna().tolist()
    assert df.y.isna().tolist() == pandas_df.y.isna().tolist()

def test_notna():
    if False:
        i = 10
        return i + 15
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    y_data = np.array([np.nan, 2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_mask = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    y = np.ma.MaskedArray(data=y_data, mask=y_mask)
    df = vaex.from_arrays(x=x, y=y)
    pandas_df = df.to_pandas_df()
    assert df.x.notna().tolist() == pandas_df.x.notna().tolist()
    assert df.y.notna().tolist() == pandas_df.y.notna().tolist()

def test_notna_arrow():
    if False:
        i = 10
        return i + 15
    x = pa.array(['Apple', None, 'Orange'])
    y = pa.array([1, None, 3.5])
    z = np.array([1, np.nan, 3.5])
    df = vaex.from_arrays(x=x, y=y, z=z)
    assert df.x.notmissing().tolist() == [True, False, True]
    assert df.y.notna().tolist() == [True, False, True]
    assert df.y.notnan().tolist() == [True, True, True]
    assert df.z.notnan().tolist() == [True, False, True]
    assert df.z.notna().tolist() == [True, False, True]