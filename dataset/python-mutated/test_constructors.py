import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, Period, PeriodIndex, Series, Timedelta, TimedeltaIndex, Timestamp
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

@pytest.mark.parametrize('dtype', [None, 'int64'])
def test_series_from_series(dtype, using_copy_on_write, warn_copy_on_write):
    if False:
        return 10
    ser = Series([1, 2, 3], name='name')
    result = Series(ser, dtype=dtype)
    assert np.shares_memory(get_array(ser), get_array(result))
    if using_copy_on_write:
        assert result._mgr.blocks[0].refs.has_reference()
    if using_copy_on_write:
        result.iloc[0] = 0
        assert ser.iloc[0] == 1
        assert not np.shares_memory(get_array(ser), get_array(result))
    else:
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[0] = 0
        assert ser.iloc[0] == 0
        assert np.shares_memory(get_array(ser), get_array(result))
    result = Series(ser, dtype=dtype)
    if using_copy_on_write:
        ser.iloc[0] = 0
        assert result.iloc[0] == 1
    else:
        with tm.assert_cow_warning(warn_copy_on_write):
            ser.iloc[0] = 0
        assert result.iloc[0] == 0

def test_series_from_series_with_reindex(using_copy_on_write, warn_copy_on_write):
    if False:
        i = 10
        return i + 15
    ser = Series([1, 2, 3], name='name')
    for index in [ser.index, ser.index.copy(), list(ser.index), ser.index.rename('idx')]:
        result = Series(ser, index=index)
        assert np.shares_memory(ser.values, result.values)
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[0] = 0
        if using_copy_on_write:
            assert ser.iloc[0] == 1
        else:
            assert ser.iloc[0] == 0
    result = Series(ser, index=[0, 1, 2, 3])
    assert not np.shares_memory(ser.values, result.values)
    if using_copy_on_write:
        assert not result._mgr.blocks[0].refs.has_reference()

@pytest.mark.parametrize('fastpath', [False, True])
@pytest.mark.parametrize('dtype', [None, 'int64'])
@pytest.mark.parametrize('idx', [None, pd.RangeIndex(start=0, stop=3, step=1)])
@pytest.mark.parametrize('arr', [np.array([1, 2, 3], dtype='int64'), pd.array([1, 2, 3], dtype='Int64')])
def test_series_from_array(using_copy_on_write, idx, dtype, fastpath, arr):
    if False:
        i = 10
        return i + 15
    if idx is None or dtype is not None:
        fastpath = False
    msg = "The 'fastpath' keyword in pd.Series is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser = Series(arr, dtype=dtype, index=idx, fastpath=fastpath)
    ser_orig = ser.copy()
    data = getattr(arr, '_data', arr)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), data)
    else:
        assert np.shares_memory(get_array(ser), data)
    arr[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series([100, 2, 3], dtype=dtype if dtype is not None else arr.dtype)
        tm.assert_series_equal(ser, expected)

@pytest.mark.parametrize('copy', [True, False, None])
def test_series_from_array_different_dtype(using_copy_on_write, copy):
    if False:
        print('Hello World!')
    arr = np.array([1, 2, 3], dtype='int64')
    ser = Series(arr, dtype='int32', copy=copy)
    assert not np.shares_memory(get_array(ser), arr)

@pytest.mark.parametrize('idx', [Index([1, 2]), DatetimeIndex([Timestamp('2019-12-31'), Timestamp('2020-12-31')]), PeriodIndex([Period('2019-12-31'), Period('2020-12-31')]), TimedeltaIndex([Timedelta('1 days'), Timedelta('2 days')])])
def test_series_from_index(using_copy_on_write, idx):
    if False:
        print('Hello World!')
    ser = Series(idx)
    expected = idx.copy(deep=True)
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(idx))
        assert not ser._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(ser), get_array(idx))
    ser.iloc[0] = ser.iloc[1]
    tm.assert_index_equal(idx, expected)

def test_series_from_index_different_dtypes(using_copy_on_write):
    if False:
        i = 10
        return i + 15
    idx = Index([1, 2, 3], dtype='int64')
    ser = Series(idx, dtype='int32')
    assert not np.shares_memory(get_array(ser), get_array(idx))
    if using_copy_on_write:
        assert ser._mgr._has_no_reference(0)

@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
@pytest.mark.parametrize('fastpath', [False, True])
@pytest.mark.parametrize('dtype', [None, 'int64'])
@pytest.mark.parametrize('idx', [None, pd.RangeIndex(start=0, stop=3, step=1)])
def test_series_from_block_manager(using_copy_on_write, idx, dtype, fastpath):
    if False:
        for i in range(10):
            print('nop')
    ser = Series([1, 2, 3], dtype='int64')
    ser_orig = ser.copy()
    msg = "The 'fastpath' keyword in pd.Series is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser2 = Series(ser._mgr, dtype=dtype, fastpath=fastpath, index=idx)
    assert np.shares_memory(get_array(ser), get_array(ser2))
    if using_copy_on_write:
        assert not ser2._mgr._has_no_reference(0)
    ser2.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series([100, 2, 3])
        tm.assert_series_equal(ser, expected)

def test_series_from_block_manager_different_dtype(using_copy_on_write):
    if False:
        return 10
    ser = Series([1, 2, 3], dtype='int64')
    msg = 'Passing a SingleBlockManager to Series'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser2 = Series(ser._mgr, dtype='int32')
    assert not np.shares_memory(get_array(ser), get_array(ser2))
    if using_copy_on_write:
        assert ser2._mgr._has_no_reference(0)

@pytest.mark.parametrize('use_mgr', [True, False])
@pytest.mark.parametrize('columns', [None, ['a']])
def test_dataframe_constructor_mgr_or_df(using_copy_on_write, warn_copy_on_write, columns, use_mgr):
    if False:
        return 10
    df = DataFrame({'a': [1, 2, 3]})
    df_orig = df.copy()
    if use_mgr:
        data = df._mgr
        warn = DeprecationWarning
    else:
        data = df
        warn = None
    msg = 'Passing a BlockManager to DataFrame'
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        new_df = DataFrame(data)
    assert np.shares_memory(get_array(df, 'a'), get_array(new_df, 'a'))
    with tm.assert_cow_warning(warn_copy_on_write and (not use_mgr)):
        new_df.iloc[0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, 'a'), get_array(new_df, 'a'))
        tm.assert_frame_equal(df, df_orig)
    else:
        assert np.shares_memory(get_array(df, 'a'), get_array(new_df, 'a'))
        tm.assert_frame_equal(df, new_df)

@pytest.mark.parametrize('dtype', [None, 'int64', 'Int64'])
@pytest.mark.parametrize('index', [None, [0, 1, 2]])
@pytest.mark.parametrize('columns', [None, ['a', 'b'], ['a', 'b', 'c']])
def test_dataframe_from_dict_of_series(request, using_copy_on_write, warn_copy_on_write, columns, index, dtype):
    if False:
        print('Hello World!')
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    s1_orig = s1.copy()
    expected = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=index, columns=columns, dtype=dtype)
    result = DataFrame({'a': s1, 'b': s2}, index=index, columns=columns, dtype=dtype, copy=False)
    assert np.shares_memory(get_array(result, 'a'), get_array(s1))
    result.iloc[0, 0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(s1))
        tm.assert_series_equal(s1, s1_orig)
    else:
        assert s1.iloc[0] == 10
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    result = DataFrame({'a': s1, 'b': s2}, index=index, columns=columns, dtype=dtype, copy=False)
    with tm.assert_cow_warning(warn_copy_on_write):
        s1.iloc[0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(s1))
        tm.assert_frame_equal(result, expected)
    else:
        assert result.iloc[0, 0] == 10

@pytest.mark.parametrize('dtype', [None, 'int64'])
def test_dataframe_from_dict_of_series_with_reindex(dtype):
    if False:
        i = 10
        return i + 15
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    df = DataFrame({'a': s1, 'b': s2}, index=[1, 2, 3], dtype=dtype, copy=False)
    arr_before = get_array(df, 'a')
    assert not np.shares_memory(arr_before, get_array(s1))
    df.iloc[0, 0] = 100
    arr_after = get_array(df, 'a')
    assert np.shares_memory(arr_before, arr_after)

@pytest.mark.parametrize('cons', [Series, Index])
@pytest.mark.parametrize('data, dtype', [([1, 2], None), ([1, 2], 'int64'), (['a', 'b'], None)])
def test_dataframe_from_series_or_index(using_copy_on_write, warn_copy_on_write, data, dtype, cons):
    if False:
        return 10
    obj = cons(data, dtype=dtype)
    obj_orig = obj.copy()
    df = DataFrame(obj, dtype=dtype)
    assert np.shares_memory(get_array(obj), get_array(df, 0))
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = data[-1]
    if using_copy_on_write:
        tm.assert_equal(obj, obj_orig)

@pytest.mark.parametrize('cons', [Series, Index])
def test_dataframe_from_series_or_index_different_dtype(using_copy_on_write, cons):
    if False:
        while True:
            i = 10
    obj = cons([1, 2], dtype='int64')
    df = DataFrame(obj, dtype='int32')
    assert not np.shares_memory(get_array(obj), get_array(df, 0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)

def test_dataframe_from_series_infer_datetime(using_copy_on_write):
    if False:
        print('Hello World!')
    ser = Series([Timestamp('2019-12-31'), Timestamp('2020-12-31')], dtype=object)
    df = DataFrame(ser)
    assert not np.shares_memory(get_array(ser), get_array(df, 0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)

@pytest.mark.parametrize('index', [None, [0, 1, 2]])
def test_dataframe_from_dict_of_series_with_dtype(index):
    if False:
        i = 10
        return i + 15
    s1 = Series([1.0, 2.0, 3.0])
    s2 = Series([4, 5, 6])
    df = DataFrame({'a': s1, 'b': s2}, index=index, dtype='int64', copy=False)
    arr_before = get_array(df, 'a')
    assert not np.shares_memory(arr_before, get_array(s1))
    df.iloc[0, 0] = 100
    arr_after = get_array(df, 'a')
    assert np.shares_memory(arr_before, arr_after)

@pytest.mark.parametrize('copy', [False, None, True])
def test_frame_from_numpy_array(using_copy_on_write, copy, using_array_manager):
    if False:
        i = 10
        return i + 15
    arr = np.array([[1, 2], [3, 4]])
    df = DataFrame(arr, copy=copy)
    if using_copy_on_write and copy is not False or copy is True or (using_array_manager and copy is None):
        assert not np.shares_memory(get_array(df, 0), arr)
    else:
        assert np.shares_memory(get_array(df, 0), arr)

def test_dataframe_from_records_with_dataframe(using_copy_on_write, warn_copy_on_write):
    if False:
        return 10
    df = DataFrame({'a': [1, 2, 3]})
    df_orig = df.copy()
    with tm.assert_produces_warning(FutureWarning):
        df2 = DataFrame.from_records(df)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
    assert np.shares_memory(get_array(df, 'a'), get_array(df2, 'a'))
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, df2)

def test_frame_from_dict_of_index(using_copy_on_write):
    if False:
        print('Hello World!')
    idx = Index([1, 2, 3])
    expected = idx.copy(deep=True)
    df = DataFrame({'a': idx}, copy=False)
    assert np.shares_memory(get_array(df, 'a'), idx._values)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
        df.iloc[0, 0] = 100
        tm.assert_index_equal(idx, expected)