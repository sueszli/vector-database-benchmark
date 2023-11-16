"""
Test IO with dask.delayed API
"""
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from dask.delayed import delayed
import cudf as gd
import dask_cudf as dgd

@delayed
def load_data(nelem, ident):
    if False:
        i = 10
        return i + 15
    df = gd.DataFrame()
    df['x'] = np.arange(nelem)
    df['ident'] = np.asarray([ident] * nelem)
    return df

@delayed
def get_combined_column(df):
    if False:
        print('Hello World!')
    return df.x * df.ident

def test_dataframe_from_delayed():
    if False:
        print('Hello World!')
    delays = [load_data(10 * i, i) for i in range(1, 3)]
    out = dgd.from_delayed(delays)
    res = out.compute()
    assert isinstance(res, gd.DataFrame)
    expected = gd.concat([d.compute() for d in delays])
    assert_frame_equal(res.to_pandas(), expected.to_pandas())

def test_series_from_delayed():
    if False:
        print('Hello World!')
    delays = [get_combined_column(load_data(10 * i, i)) for i in range(1, 3)]
    out = dgd.from_delayed(delays)
    res = out.compute()
    assert isinstance(res, gd.Series)
    expected = gd.concat([d.compute() for d in delays])
    np.testing.assert_array_equal(res.to_pandas(), expected.to_pandas())

def test_dataframe_to_delayed():
    if False:
        return 10
    nelem = 100
    df = gd.DataFrame()
    df['x'] = np.arange(nelem)
    df['y'] = np.random.randint(nelem, size=nelem)
    ddf = dgd.from_cudf(df, npartitions=5)
    delays = ddf.to_delayed()
    assert len(delays) == 5
    got = gd.concat([d.compute() for d in delays])
    assert_frame_equal(got.to_pandas(), df.to_pandas())
    divs = ddf.divisions
    assert len(divs) == len(delays) + 1
    for (i, part) in enumerate(delays):
        s = divs[i]
        e = None if i + 1 == len(delays) else divs[i + 1]
        expect = df[s:e].to_pandas()
        got = part.compute().to_pandas()
        assert_frame_equal(got, expect)

def test_series_to_delayed():
    if False:
        while True:
            i = 10
    nelem = 100
    sr = gd.Series(np.random.randint(nelem, size=nelem))
    dsr = dgd.from_cudf(sr, npartitions=5)
    delays = dsr.to_delayed()
    assert len(delays) == 5
    got = gd.concat([d.compute() for d in delays])
    assert isinstance(got, gd.Series)
    np.testing.assert_array_equal(got.to_pandas(), sr.to_pandas())
    divs = dsr.divisions
    assert len(divs) == len(delays) + 1
    for (i, part) in enumerate(delays):
        s = divs[i]
        e = None if i + 1 == len(delays) else divs[i + 1]
        expect = sr[s:e].to_pandas()
        got = part.compute().to_pandas()
        np.testing.assert_array_equal(got, expect)

def test_mixing_series_frame_error():
    if False:
        for i in range(10):
            print('nop')
    nelem = 20
    df = gd.DataFrame()
    df['x'] = np.arange(nelem)
    df['y'] = np.random.randint(nelem, size=nelem)
    ddf = dgd.from_cudf(df, npartitions=5)
    delay_frame = ddf.to_delayed()
    delay_series = ddf.x.to_delayed()
    combined = dgd.from_delayed(delay_frame + delay_series)
    with pytest.raises(ValueError) as raises:
        combined.compute()
    raises.match('^Metadata mismatch found in `from_delayed`.')

def test_frame_extra_columns_error():
    if False:
        i = 10
        return i + 15
    nelem = 20
    df = gd.DataFrame()
    df['x'] = np.arange(nelem)
    df['y'] = np.random.randint(nelem, size=nelem)
    ddf1 = dgd.from_cudf(df, npartitions=5)
    df['z'] = np.arange(nelem)
    ddf2 = dgd.from_cudf(df, npartitions=5)
    combined = dgd.from_delayed(ddf1.to_delayed() + ddf2.to_delayed())
    with pytest.raises(ValueError) as raises:
        combined.compute()
    raises.match('^Metadata mismatch found in `from_delayed`.')
    raises.match('z')

@pytest.mark.xfail(reason='')
def test_frame_dtype_error():
    if False:
        for i in range(10):
            print('nop')
    nelem = 20
    df1 = gd.DataFrame()
    df1['bad'] = np.arange(nelem)
    df1['bad'] = np.arange(nelem, dtype=np.float64)
    df2 = gd.DataFrame()
    df2['bad'] = np.arange(nelem)
    df2['bad'] = np.arange(nelem, dtype=np.float32)
    ddf1 = dgd.from_cudf(df1, npartitions=5)
    ddf2 = dgd.from_cudf(df2, npartitions=5)
    combined = dgd.from_delayed(ddf1.to_delayed() + ddf2.to_delayed())
    with pytest.raises(ValueError) as raises:
        combined.compute()
    raises.match('same type')