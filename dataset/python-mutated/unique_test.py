import datetime
from common import small_buffer
import pytest
import numpy as np
import pyarrow as pa
import vaex
from vaex.utils import dropnan

def test_unique_arrow(df_factory):
    if False:
        print('Hello World!')
    ds = df_factory(x=vaex.string_column(['a', 'b', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'a']))
    with small_buffer(ds, 2):
        assert set(ds.unique(ds.x)) == {'a', 'b'}
        (values, index) = ds.unique(ds.x, return_inverse=True)
        assert np.array(values)[index].tolist() == ds.x.tolist()

def test_unique_bool(df_factory):
    if False:
        i = 10
        return i + 15
    df = df_factory(x=[True, False, True, True, False, False])
    u = df.unique('x')
    assert len(u) == 2
    assert set(u) == {True, False}

def test_unique(df_factory):
    if False:
        while True:
            i = 10
    ds = df_factory(colors=['red', 'green', 'blue', 'green'])
    with small_buffer(ds, 2):
        assert set(ds.unique(ds.colors)) == {'red', 'green', 'blue'}
        (values, index) = ds.unique(ds.colors, return_inverse=True)
        assert np.array(values)[index].tolist() == ds.colors.tolist()
    ds = df_factory(x=['a', 'b', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'a'])
    with small_buffer(ds, 2):
        assert set(ds.unique(ds.x)) == {'a', 'b'}
        (values, index) = ds.unique(ds.x, return_inverse=True)
        assert np.array(values)[index].tolist() == ds.x.tolist()

def test_unique_f4(df_factory):
    if False:
        return 10
    x = np.array([np.nan, 0, 1, np.nan, 2, np.nan], dtype='f4')
    df = df_factory(x=x)
    assert dropnan(set(df.x.unique(dropnan=True))) == {0, 1, 2}
    assert dropnan(set(df.x.unique()), expect=1) == {0, 1, 2}

def test_unique_nan(df_factory):
    if False:
        return 10
    x = [np.nan, 0, 1, np.nan, 2, np.nan]
    df = df_factory(x=x)
    assert set(df.x.unique(dropnan=True)) == {0, 1, 2}
    assert dropnan(set(df.x.unique()), expect=1) == {0, 1, 2}
    with small_buffer(df, 2):
        (values, indices) = df.unique(df.x, return_inverse=True)
        values = np.array(values)
        values = values[indices]
        mask = np.isnan(values)
        assert values[~mask].tolist() == df.x.to_numpy()[~mask].tolist()

def test_unique_missing(df_factory):
    if False:
        for i in range(10):
            print('nop')
    x = np.array([None, 'A', 'B', -1, 0, 2, '', '', None, None, None, np.nan, np.nan, np.nan, np.nan])
    df = df_factory(x=x)
    uniques = df.x.unique(dropnan=True)
    assert set(uniques) == set(['', 'A', 'B', -1, 0, 2, None])

def test_unique_missing_numeric(array_factory):
    if False:
        i = 10
        return i + 15
    df = vaex.from_arrays(x=array_factory([1, None]))
    values = df.x.unique()
    assert set(values) == {1, None}

def test_unique_string_missing(df_factory):
    if False:
        return 10
    x = ['John', None, 'Sally', None, '0.0']
    df = df_factory(x=x)
    result = df.x.unique()
    assert len(result) == 4
    assert 'John' in result
    assert None in result
    assert 'Sally'

def test_unique_list(df_types):
    if False:
        i = 10
        return i + 15
    df = df_types
    assert set(df.string_list.unique()) == {'aap', 'noot', 'mies', None}
    assert set(df.int_list.unique()) == {1, 2, 3, 4, 5, None}

@pytest.mark.parametrize('future', [False, True])
def test_unique_categorical(df_factory, future):
    if False:
        for i in range(10):
            print('nop')
    df = df_factory(x=vaex.string_column(['a', 'c', 'b', 'a', 'a']))
    df = df.ordinal_encode('x')
    df = df._future() if future else df
    if future:
        assert df.x.dtype == str
        assert set(df.x.unique()) == {'a', 'b', 'c'}
        assert df.x.nunique() == 3
    else:
        assert df.x.dtype == int
        assert set(df.x.unique()) == {0, 1, 2}
        assert df.x.nunique() == 3
    if future:
        df = df[df.x.isin(['b', 'c'])]
        assert df.x.dtype == str
        assert set(df.x.unique()) == {'b', 'c'}
        assert df.x.nunique() == 2
    else:
        df = df[df.x.isin([1, 2])]
        assert df.x.dtype == int
        assert set(df.x.unique()) == {1, 2}
        assert df.x.nunique() == 2

def test_unique_datetime_timedelta():
    if False:
        for i in range(10):
            print('nop')
    x = [1, 2, 3, 1, 1]
    date = [np.datetime64('2020-01-01'), np.datetime64('2020-01-02'), np.datetime64('2020-01-03'), np.datetime64('2020-01-01'), np.datetime64('2020-01-01')]
    df = vaex.from_arrays(x=x, date=date)
    df['delta'] = df.date - np.datetime64('2020-01-01')
    unique_date = df.unique(expression='date')
    assert set(unique_date) == {datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)}
    unique_delta = df.unique(expression='delta')
    assert set(unique_delta) == {datetime.timedelta(0), datetime.timedelta(days=1), datetime.timedelta(days=2)}

def test_unique_limit_primitive():
    if False:
        print('Hello World!')
    x = np.arange(100)
    df = vaex.from_arrays(x=x)
    with pytest.raises(vaex.RowLimitException, match='.*Resulting hash_map_unique.*'):
        values = df.x.unique(limit=11)
    values = df.x.unique(limit=11, limit_raise=False)
    assert len(values) == 11
    with pytest.raises(vaex.RowLimitException, match='.*Resulting hash_map_unique.*'):
        df.x.nunique(limit=11)
    assert df.x.nunique(limit=11, limit_raise=False) == 11
    df.x.nunique(limit=100)

def test_unique_limit_string():
    if False:
        while True:
            i = 10
    x = np.arange(100)
    df = vaex.from_arrays(x=[str(k) for k in x])
    with pytest.raises(vaex.RowLimitException, match='.*Resulting hash_map_unique.*'):
        values = df.x.unique(limit=11)
    values = df.x.unique(limit=11, limit_raise=False)
    assert len(values) == 11
    with pytest.raises(vaex.RowLimitException, match='.*Resulting hash_map_unique.*'):
        df.x.nunique(limit=11)
    assert df.x.nunique(limit=11, limit_raise=False) == 11
    df.x.nunique(limit=100)

@pytest.mark.parametrize('selection', [True, 'default', 'custom_name'])
def test_unique_selection(df_factory, selection):
    if False:
        while True:
            i = 10
    x = ['a', 'a', 'c', 'd', 'e', 'f']
    y = np.array([10, 10, 30, 40, 50, 60])
    df = df_factory(x=x, y=y)
    if selection == 'custom_name':
        df.select(df.y == 10, name=selection)
    else:
        df.select(df.y == 10)
    assert df.x.unique(selection=selection) == ['a']