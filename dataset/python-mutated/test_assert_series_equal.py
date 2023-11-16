import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, DataFrame, Series
import pandas._testing as tm

def _assert_series_equal_both(a, b, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that two Series equal.\n\n    This check is performed commutatively.\n\n    Parameters\n    ----------\n    a : Series\n        The first Series to compare.\n    b : Series\n        The second Series to compare.\n    kwargs : dict\n        The arguments passed to `tm.assert_series_equal`.\n    '
    tm.assert_series_equal(a, b, **kwargs)
    tm.assert_series_equal(b, a, **kwargs)

def _assert_not_series_equal(a, b, **kwargs):
    if False:
        print('Hello World!')
    '\n    Check that two Series are not equal.\n\n    Parameters\n    ----------\n    a : Series\n        The first Series to compare.\n    b : Series\n        The second Series to compare.\n    kwargs : dict\n        The arguments passed to `tm.assert_series_equal`.\n    '
    try:
        tm.assert_series_equal(a, b, **kwargs)
        msg = "The two Series were equal when they shouldn't have been"
        pytest.fail(msg=msg)
    except AssertionError:
        pass

def _assert_not_series_equal_both(a, b, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Check that two Series are not equal.\n\n    This check is performed commutatively.\n\n    Parameters\n    ----------\n    a : Series\n        The first Series to compare.\n    b : Series\n        The second Series to compare.\n    kwargs : dict\n        The arguments passed to `tm.assert_series_equal`.\n    '
    _assert_not_series_equal(a, b, **kwargs)
    _assert_not_series_equal(b, a, **kwargs)

@pytest.mark.parametrize('data', [range(3), list('abc'), list('áàä')])
def test_series_equal(data):
    if False:
        while True:
            i = 10
    _assert_series_equal_both(Series(data), Series(data))

@pytest.mark.parametrize('data1,data2', [(range(3), range(1, 4)), (list('abc'), list('xyz')), (list('áàä'), list('éèë')), (list('áàä'), list(b'aaa')), (range(3), range(4))])
def test_series_not_equal_value_mismatch(data1, data2):
    if False:
        i = 10
        return i + 15
    _assert_not_series_equal_both(Series(data1), Series(data2))

@pytest.mark.parametrize('kwargs', [{'dtype': 'float64'}, {'index': [1, 2, 4]}, {'name': 'foo'}])
def test_series_not_equal_metadata_mismatch(kwargs):
    if False:
        for i in range(10):
            print('nop')
    data = range(3)
    s1 = Series(data)
    s2 = Series(data, **kwargs)
    _assert_not_series_equal_both(s1, s2)

@pytest.mark.parametrize('data1,data2', [(0.12345, 0.12346), (0.1235, 0.1236)])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'Float32'])
@pytest.mark.parametrize('decimals', [0, 1, 2, 3, 5, 10])
def test_less_precise(data1, data2, dtype, decimals):
    if False:
        print('Hello World!')
    rtol = 10 ** (-decimals)
    s1 = Series([data1], dtype=dtype)
    s2 = Series([data2], dtype=dtype)
    if decimals in (5, 10) or (decimals >= 3 and abs(data1 - data2) >= 0.0005):
        msg = 'Series values are different'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, rtol=rtol)
    else:
        _assert_series_equal_both(s1, s2, rtol=rtol)

@pytest.mark.parametrize('s1,s2,msg', [(Series(['l1', 'l2'], index=[1, 2]), Series(['l1', 'l2'], index=[1.0, 2.0]), 'Series\\.index are different'), (DataFrame.from_records({'a': [1, 2], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']).c, DataFrame.from_records({'a': [1.0, 2.0], 'b': [2.1, 1.5], 'c': ['l1', 'l2']}, index=['a', 'b']).c, 'MultiIndex level \\[0\\] are different')])
def test_series_equal_index_dtype(s1, s2, msg, check_index_type):
    if False:
        while True:
            i = 10
    kwargs = {'check_index_type': check_index_type}
    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, **kwargs)
    else:
        tm.assert_series_equal(s1, s2, **kwargs)

@pytest.mark.parametrize('check_like', [True, False])
def test_series_equal_order_mismatch(check_like):
    if False:
        while True:
            i = 10
    s1 = Series([1, 2, 3], index=['a', 'b', 'c'])
    s2 = Series([3, 2, 1], index=['c', 'b', 'a'])
    if not check_like:
        with pytest.raises(AssertionError, match='Series.index are different'):
            tm.assert_series_equal(s1, s2, check_like=check_like)
    else:
        _assert_series_equal_both(s1, s2, check_like=check_like)

@pytest.mark.parametrize('check_index', [True, False])
def test_series_equal_index_mismatch(check_index):
    if False:
        for i in range(10):
            print('nop')
    s1 = Series([1, 2, 3], index=['a', 'b', 'c'])
    s2 = Series([1, 2, 3], index=['c', 'b', 'a'])
    if check_index:
        with pytest.raises(AssertionError, match='Series.index are different'):
            tm.assert_series_equal(s1, s2, check_index=check_index)
    else:
        _assert_series_equal_both(s1, s2, check_index=check_index)

def test_series_invalid_param_combination():
    if False:
        for i in range(10):
            print('nop')
    left = Series(dtype=object)
    right = Series(dtype=object)
    with pytest.raises(ValueError, match='check_like must be False if check_index is False'):
        tm.assert_series_equal(left, right, check_index=False, check_like=True)

def test_series_equal_length_mismatch(rtol):
    if False:
        i = 10
        return i + 15
    msg = 'Series are different\n\nSeries length are different\n\\[left\\]:  3, RangeIndex\\(start=0, stop=3, step=1\\)\n\\[right\\]: 4, RangeIndex\\(start=0, stop=4, step=1\\)'
    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 3, 4])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)

def test_series_equal_numeric_values_mismatch(rtol):
    if False:
        while True:
            i = 10
    msg = 'Series are different\n\nSeries values are different \\(33\\.33333 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\[1, 2, 3\\]\n\\[right\\]: \\[1, 2, 4\\]'
    s1 = Series([1, 2, 3])
    s2 = Series([1, 2, 4])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)

def test_series_equal_categorical_values_mismatch(rtol):
    if False:
        print('Hello World!')
    msg = "Series are different\n\nSeries values are different \\(66\\.66667 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\['a', 'b', 'c'\\]\nCategories \\(3, object\\): \\['a', 'b', 'c'\\]\n\\[right\\]: \\['a', 'c', 'b'\\]\nCategories \\(3, object\\): \\['a', 'b', 'c'\\]"
    s1 = Series(Categorical(['a', 'b', 'c']))
    s2 = Series(Categorical(['a', 'c', 'b']))
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)

def test_series_equal_datetime_values_mismatch(rtol):
    if False:
        return 10
    msg = 'Series are different\n\nSeries values are different \\(100.0 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\[1514764800000000000, 1514851200000000000, 1514937600000000000\\]\n\\[right\\]: \\[1549065600000000000, 1549152000000000000, 1549238400000000000\\]'
    s1 = Series(pd.date_range('2018-01-01', periods=3, freq='D'))
    s2 = Series(pd.date_range('2019-02-02', periods=3, freq='D'))
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)

def test_series_equal_categorical_mismatch(check_categorical):
    if False:
        for i in range(10):
            print('nop')
    msg = 'Attributes of Series are different\n\nAttribute "dtype" are different\n\\[left\\]:  CategoricalDtype\\(categories=\\[\'a\', \'b\'\\], ordered=False, categories_dtype=object\\)\n\\[right\\]: CategoricalDtype\\(categories=\\[\'a\', \'b\', \'c\'\\], ordered=False, categories_dtype=object\\)'
    s1 = Series(Categorical(['a', 'b']))
    s2 = Series(Categorical(['a', 'b'], categories=list('abc')))
    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, check_categorical=check_categorical)
    else:
        _assert_series_equal_both(s1, s2, check_categorical=check_categorical)

def test_assert_series_equal_extension_dtype_mismatch():
    if False:
        return 10
    left = Series(pd.array([1, 2, 3], dtype='Int64'))
    right = left.astype(int)
    msg = 'Attributes of Series are different\n\nAttribute "dtype" are different\n\\[left\\]:  Int64\n\\[right\\]: int[32|64]'
    tm.assert_series_equal(left, right, check_dtype=False)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)

def test_assert_series_equal_interval_dtype_mismatch():
    if False:
        for i in range(10):
            print('nop')
    left = Series([pd.Interval(0, 1)], dtype='interval')
    right = left.astype(object)
    msg = 'Attributes of Series are different\n\nAttribute "dtype" are different\n\\[left\\]:  interval\\[int64, right\\]\n\\[right\\]: object'
    tm.assert_series_equal(left, right, check_dtype=False)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)

def test_series_equal_series_type():
    if False:
        while True:
            i = 10

    class MySeries(Series):
        pass
    s1 = Series([1, 2])
    s2 = Series([1, 2])
    s3 = MySeries([1, 2])
    tm.assert_series_equal(s1, s2, check_series_type=False)
    tm.assert_series_equal(s1, s2, check_series_type=True)
    tm.assert_series_equal(s1, s3, check_series_type=False)
    tm.assert_series_equal(s3, s1, check_series_type=False)
    with pytest.raises(AssertionError, match='Series classes are different'):
        tm.assert_series_equal(s1, s3, check_series_type=True)
    with pytest.raises(AssertionError, match='Series classes are different'):
        tm.assert_series_equal(s3, s1, check_series_type=True)

def test_series_equal_exact_for_nonnumeric():
    if False:
        for i in range(10):
            print('nop')
    s1 = Series(['a', 'b'])
    s2 = Series(['a', 'b'])
    s3 = Series(['b', 'a'])
    tm.assert_series_equal(s1, s2, check_exact=True)
    tm.assert_series_equal(s2, s1, check_exact=True)
    msg = 'Series are different\n\nSeries values are different \\(100\\.0 %\\)\n\\[index\\]: \\[0, 1\\]\n\\[left\\]:  \\[a, b\\]\n\\[right\\]: \\[b, a\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s3, check_exact=True)
    msg = 'Series are different\n\nSeries values are different \\(100\\.0 %\\)\n\\[index\\]: \\[0, 1\\]\n\\[left\\]:  \\[b, a\\]\n\\[right\\]: \\[a, b\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s3, s1, check_exact=True)

@pytest.mark.parametrize('right_dtype', ['Int32', 'int64'])
def test_assert_series_equal_ignore_extension_dtype_mismatch(right_dtype):
    if False:
        return 10
    left = Series([1, 2, 3], dtype='Int64')
    right = Series([1, 2, 3], dtype=right_dtype)
    tm.assert_series_equal(left, right, check_dtype=False)

def test_allows_duplicate_labels():
    if False:
        i = 10
        return i + 15
    left = Series([1])
    right = Series([1]).set_flags(allows_duplicate_labels=False)
    tm.assert_series_equal(left, left)
    tm.assert_series_equal(right, right)
    tm.assert_series_equal(left, right, check_flags=False)
    tm.assert_series_equal(right, left, check_flags=False)
    with pytest.raises(AssertionError, match='<Flags'):
        tm.assert_series_equal(left, right)
    with pytest.raises(AssertionError, match='<Flags'):
        tm.assert_series_equal(left, right)

def test_assert_series_equal_identical_na(nulls_fixture):
    if False:
        print('Hello World!')
    ser = Series([nulls_fixture])
    tm.assert_series_equal(ser, ser.copy())
    idx = pd.Index(ser)
    tm.assert_index_equal(idx, idx.copy(deep=True))

def test_identical_nested_series_is_equal():
    if False:
        while True:
            i = 10
    x = Series([0, 0.0131142231938, 1.77774652865e-05, np.array([0.4722720840328748, 0.4216929783681722])])
    y = Series([0, 0.0131142231938, 1.77774652865e-05, np.array([0.4722720840328748, 0.4216929783681722])])
    tm.assert_series_equal(x, x)
    tm.assert_series_equal(x, x, check_exact=True)
    tm.assert_series_equal(x, y)
    tm.assert_series_equal(x, y, check_exact=True)

@pytest.mark.parametrize('dtype', ['datetime64', 'timedelta64'])
def test_check_dtype_false_different_reso(dtype):
    if False:
        while True:
            i = 10
    ser_s = Series([1000213, 2131232, 21312331]).astype(f'{dtype}[s]')
    ser_ms = ser_s.astype(f'{dtype}[ms]')
    with pytest.raises(AssertionError, match='Attributes of Series are different'):
        tm.assert_series_equal(ser_s, ser_ms)
    tm.assert_series_equal(ser_ms, ser_s, check_dtype=False)
    ser_ms -= Series([1, 1, 1]).astype(f'{dtype}[ms]')
    with pytest.raises(AssertionError, match='Series are different'):
        tm.assert_series_equal(ser_s, ser_ms)
    with pytest.raises(AssertionError, match='Series are different'):
        tm.assert_series_equal(ser_s, ser_ms, check_dtype=False)