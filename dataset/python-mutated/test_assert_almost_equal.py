import numpy as np
import pytest
from pandas import NA, DataFrame, Index, NaT, Series, Timestamp
import pandas._testing as tm

def _assert_almost_equal_both(a, b, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Check that two objects are approximately equal.\n\n    This check is performed commutatively.\n\n    Parameters\n    ----------\n    a : object\n        The first object to compare.\n    b : object\n        The second object to compare.\n    **kwargs\n        The arguments passed to `tm.assert_almost_equal`.\n    '
    tm.assert_almost_equal(a, b, **kwargs)
    tm.assert_almost_equal(b, a, **kwargs)

def _assert_not_almost_equal(a, b, **kwargs):
    if False:
        print('Hello World!')
    '\n    Check that two objects are not approximately equal.\n\n    Parameters\n    ----------\n    a : object\n        The first object to compare.\n    b : object\n        The second object to compare.\n    **kwargs\n        The arguments passed to `tm.assert_almost_equal`.\n    '
    try:
        tm.assert_almost_equal(a, b, **kwargs)
        msg = f"{a} and {b} were approximately equal when they shouldn't have been"
        pytest.fail(reason=msg)
    except AssertionError:
        pass

def _assert_not_almost_equal_both(a, b, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that two objects are not approximately equal.\n\n    This check is performed commutatively.\n\n    Parameters\n    ----------\n    a : object\n        The first object to compare.\n    b : object\n        The second object to compare.\n    **kwargs\n        The arguments passed to `tm.assert_almost_equal`.\n    '
    _assert_not_almost_equal(a, b, **kwargs)
    _assert_not_almost_equal(b, a, **kwargs)

@pytest.mark.parametrize('a,b', [(1.1, 1.1), (1.1, 1.100001), (np.int16(1), 1.000001), (np.float64(1.1), 1.1), (np.uint32(5), 5)])
def test_assert_almost_equal_numbers(a, b):
    if False:
        while True:
            i = 10
    _assert_almost_equal_both(a, b)

@pytest.mark.parametrize('a,b', [(1.1, 1), (1.1, True), (1, 2), (1.0001, np.int16(1)), (0.1, 0.1001), (0.0011, 0.0012)])
def test_assert_not_almost_equal_numbers(a, b):
    if False:
        i = 10
        return i + 15
    _assert_not_almost_equal_both(a, b)

@pytest.mark.parametrize('a,b', [(1.1, 1.1), (1.1, 1.100001), (1.1, 1.1001), (1e-06, 5e-06), (1000.0, 1000.0005), (1.1e-05, 1.2e-05)])
def test_assert_almost_equal_numbers_atol(a, b):
    if False:
        print('Hello World!')
    _assert_almost_equal_both(a, b, rtol=0.0005, atol=0.0005)

@pytest.mark.parametrize('a,b', [(1.1, 1.11), (0.1, 0.101), (1.1e-05, 0.001012)])
def test_assert_not_almost_equal_numbers_atol(a, b):
    if False:
        while True:
            i = 10
    _assert_not_almost_equal_both(a, b, atol=0.001)

@pytest.mark.parametrize('a,b', [(1.1, 1.1), (1.1, 1.100001), (1.1, 1.1001), (1000.0, 1000.0005), (1.1, 1.11), (0.1, 0.101)])
def test_assert_almost_equal_numbers_rtol(a, b):
    if False:
        for i in range(10):
            print('nop')
    _assert_almost_equal_both(a, b, rtol=0.05)

@pytest.mark.parametrize('a,b', [(1.1e-05, 1.2e-05), (1e-06, 5e-06)])
def test_assert_not_almost_equal_numbers_rtol(a, b):
    if False:
        while True:
            i = 10
    _assert_not_almost_equal_both(a, b, rtol=0.05)

@pytest.mark.parametrize('a,b,rtol', [(1.00001, 1.00005, 0.001), (-0.908356 + 0.2j, -0.908358 + 0.2j, 0.001), (0.1 + 1.009j, 0.1 + 1.006j, 0.1), (0.1001 + 2j, 0.1 + 2.001j, 0.01)])
def test_assert_almost_equal_complex_numbers(a, b, rtol):
    if False:
        while True:
            i = 10
    _assert_almost_equal_both(a, b, rtol=rtol)
    _assert_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)

@pytest.mark.parametrize('a,b,rtol', [(0.58310768, 0.58330768, 1e-07), (-0.908 + 0.2j, -0.978 + 0.2j, 0.001), (0.1 + 1j, 0.1 + 2j, 0.01), (-0.132 + 1.001j, -0.132 + 1.005j, 1e-05), (0.58310768j, 0.58330768j, 1e-09)])
def test_assert_not_almost_equal_complex_numbers(a, b, rtol):
    if False:
        while True:
            i = 10
    _assert_not_almost_equal_both(a, b, rtol=rtol)
    _assert_not_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_not_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)

@pytest.mark.parametrize('a,b', [(0, 0), (0, 0.0), (0, np.float64(0)), (1e-08, 0)])
def test_assert_almost_equal_numbers_with_zeros(a, b):
    if False:
        while True:
            i = 10
    _assert_almost_equal_both(a, b)

@pytest.mark.parametrize('a,b', [(0.001, 0), (1, 0)])
def test_assert_not_almost_equal_numbers_with_zeros(a, b):
    if False:
        while True:
            i = 10
    _assert_not_almost_equal_both(a, b)

@pytest.mark.parametrize('a,b', [(1, 'abc'), (1, [1]), (1, object())])
def test_assert_not_almost_equal_numbers_with_mixed(a, b):
    if False:
        for i in range(10):
            print('nop')
    _assert_not_almost_equal_both(a, b)

@pytest.mark.parametrize('left_dtype', ['M8[ns]', 'm8[ns]', 'float64', 'int64', 'object'])
@pytest.mark.parametrize('right_dtype', ['M8[ns]', 'm8[ns]', 'float64', 'int64', 'object'])
def test_assert_almost_equal_edge_case_ndarrays(left_dtype, right_dtype):
    if False:
        for i in range(10):
            print('nop')
    _assert_almost_equal_both(np.array([], dtype=left_dtype), np.array([], dtype=right_dtype), check_dtype=False)

def test_assert_almost_equal_sets():
    if False:
        return 10
    _assert_almost_equal_both({1, 2, 3}, {1, 2, 3})

def test_assert_almost_not_equal_sets():
    if False:
        return 10
    msg = '{1, 2, 3} != {1, 2, 4}'
    with pytest.raises(AssertionError, match=msg):
        _assert_almost_equal_both({1, 2, 3}, {1, 2, 4})

def test_assert_almost_equal_dicts():
    if False:
        i = 10
        return i + 15
    _assert_almost_equal_both({'a': 1, 'b': 2}, {'a': 1, 'b': 2})

@pytest.mark.parametrize('a,b', [({'a': 1, 'b': 2}, {'a': 1, 'b': 3}), ({'a': 1, 'b': 2}, {'a': 1, 'b': 2, 'c': 3}), ({'a': 1}, 1), ({'a': 1}, 'abc'), ({'a': 1}, [1])])
def test_assert_not_almost_equal_dicts(a, b):
    if False:
        while True:
            i = 10
    _assert_not_almost_equal_both(a, b)

@pytest.mark.parametrize('val', [1, 2])
def test_assert_almost_equal_dict_like_object(val):
    if False:
        print('Hello World!')
    dict_val = 1
    real_dict = {'a': val}

    class DictLikeObj:

        def keys(self):
            if False:
                while True:
                    i = 10
            return ('a',)

        def __getitem__(self, item):
            if False:
                return 10
            if item == 'a':
                return dict_val
    func = _assert_almost_equal_both if val == dict_val else _assert_not_almost_equal_both
    func(real_dict, DictLikeObj(), check_dtype=False)

def test_assert_almost_equal_strings():
    if False:
        while True:
            i = 10
    _assert_almost_equal_both('abc', 'abc')

@pytest.mark.parametrize('a,b', [('abc', 'abcd'), ('abc', 'abd'), ('abc', 1), ('abc', [1])])
def test_assert_not_almost_equal_strings(a, b):
    if False:
        while True:
            i = 10
    _assert_not_almost_equal_both(a, b)

@pytest.mark.parametrize('a,b', [([1, 2, 3], [1, 2, 3]), (np.array([1, 2, 3]), np.array([1, 2, 3]))])
def test_assert_almost_equal_iterables(a, b):
    if False:
        for i in range(10):
            print('nop')
    _assert_almost_equal_both(a, b)

@pytest.mark.parametrize('a,b', [(np.array([1, 2, 3]), [1, 2, 3]), (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])), (iter([1, 2, 3]), [1, 2, 3]), ([1, 2, 3], [1, 2, 4]), ([1, 2, 3], [1, 2, 3, 4]), ([1, 2, 3], 1)])
def test_assert_not_almost_equal_iterables(a, b):
    if False:
        for i in range(10):
            print('nop')
    _assert_not_almost_equal(a, b)

def test_assert_almost_equal_null():
    if False:
        return 10
    _assert_almost_equal_both(None, None)

@pytest.mark.parametrize('a,b', [(None, np.nan), (None, 0), (np.nan, 0)])
def test_assert_not_almost_equal_null(a, b):
    if False:
        return 10
    _assert_not_almost_equal(a, b)

@pytest.mark.parametrize('a,b', [(np.inf, np.inf), (np.inf, float('inf')), (np.array([np.inf, np.nan, -np.inf]), np.array([np.inf, np.nan, -np.inf]))])
def test_assert_almost_equal_inf(a, b):
    if False:
        return 10
    _assert_almost_equal_both(a, b)
objs = [NA, np.nan, NaT, None, np.datetime64('NaT'), np.timedelta64('NaT')]

@pytest.mark.parametrize('left', objs)
@pytest.mark.parametrize('right', objs)
def test_mismatched_na_assert_almost_equal_deprecation(left, right):
    if False:
        while True:
            i = 10
    left_arr = np.array([left], dtype=object)
    right_arr = np.array([right], dtype=object)
    msg = 'Mismatched null-like values'
    if left is right:
        _assert_almost_equal_both(left, right, check_dtype=False)
        tm.assert_numpy_array_equal(left_arr, right_arr)
        tm.assert_index_equal(Index(left_arr, dtype=object), Index(right_arr, dtype=object))
        tm.assert_series_equal(Series(left_arr, dtype=object), Series(right_arr, dtype=object))
        tm.assert_frame_equal(DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object))
    else:
        with tm.assert_produces_warning(FutureWarning, match=msg):
            _assert_almost_equal_both(left, right, check_dtype=False)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_series_equal(Series(left_arr, dtype=object), Series(right_arr, dtype=object))
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_frame_equal(DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object))

def test_assert_not_almost_equal_inf():
    if False:
        return 10
    _assert_not_almost_equal_both(np.inf, 0)

@pytest.mark.parametrize('a,b', [(Index([1.0, 1.1]), Index([1.0, 1.100001])), (Series([1.0, 1.1]), Series([1.0, 1.100001])), (np.array([1.1, 2.000001]), np.array([1.1, 2.0])), (DataFrame({'a': [1.0, 1.1]}), DataFrame({'a': [1.0, 1.100001]}))])
def test_assert_almost_equal_pandas(a, b):
    if False:
        return 10
    _assert_almost_equal_both(a, b)

def test_assert_almost_equal_object():
    if False:
        return 10
    a = [Timestamp('2011-01-01'), Timestamp('2011-01-01')]
    b = [Timestamp('2011-01-01'), Timestamp('2011-01-01')]
    _assert_almost_equal_both(a, b)

def test_assert_almost_equal_value_mismatch():
    if False:
        print('Hello World!')
    msg = 'expected 2\\.00000 but got 1\\.00000, with rtol=1e-05, atol=1e-08'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(1, 2)

@pytest.mark.parametrize('a,b,klass1,klass2', [(np.array([1]), 1, 'ndarray', 'int'), (1, np.array([1]), 'int', 'ndarray')])
def test_assert_almost_equal_class_mismatch(a, b, klass1, klass2):
    if False:
        while True:
            i = 10
    msg = f'numpy array are different\n\nnumpy array classes are different\n\\[left\\]:  {klass1}\n\\[right\\]: {klass2}'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)

def test_assert_almost_equal_value_mismatch1():
    if False:
        print('Hello World!')
    msg = 'numpy array are different\n\nnumpy array values are different \\(66\\.66667 %\\)\n\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]\n\\[right\\]: \\[1\\.0, nan, 3\\.0\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))

def test_assert_almost_equal_value_mismatch2():
    if False:
        i = 10
        return i + 15
    msg = 'numpy array are different\n\nnumpy array values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[1, 2\\]\n\\[right\\]: \\[1, 3\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([1, 3]))

def test_assert_almost_equal_value_mismatch3():
    if False:
        return 10
    msg = 'numpy array are different\n\nnumpy array values are different \\(16\\.66667 %\\)\n\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]\n\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]]))

def test_assert_almost_equal_value_mismatch4():
    if False:
        print('Hello World!')
    msg = 'numpy array are different\n\nnumpy array values are different \\(25\\.0 %\\)\n\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]\n\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]]))

def test_assert_almost_equal_shape_mismatch_override():
    if False:
        return 10
    msg = 'Index are different\n\nIndex shapes are different\n\\[left\\]:  \\(2L*,\\)\n\\[right\\]: \\(3L*,\\)'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([3, 4, 5]), obj='Index')

def test_assert_almost_equal_unicode():
    if False:
        for i in range(10):
            print('nop')
    msg = 'numpy array are different\n\nnumpy array values are different \\(33\\.33333 %\\)\n\\[left\\]:  \\[á, à, ä\\]\n\\[right\\]: \\[á, à, å\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array(['á', 'à', 'ä']), np.array(['á', 'à', 'å']))

def test_assert_almost_equal_timestamp():
    if False:
        while True:
            i = 10
    a = np.array([Timestamp('2011-01-01'), Timestamp('2011-01-01')])
    b = np.array([Timestamp('2011-01-01'), Timestamp('2011-01-02')])
    msg = 'numpy array are different\n\nnumpy array values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]\n\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)

def test_assert_almost_equal_iterable_length_mismatch():
    if False:
        return 10
    msg = 'Iterable are different\n\nIterable length are different\n\\[left\\]:  2\n\\[right\\]: 3'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [3, 4, 5])

def test_assert_almost_equal_iterable_values_mismatch():
    if False:
        return 10
    msg = 'Iterable are different\n\nIterable values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[1, 2\\]\n\\[right\\]: \\[1, 3\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [1, 3])
subarr = np.empty(2, dtype=object)
subarr[:] = [np.array([None, 'b'], dtype=object), np.array(['c', 'd'], dtype=object)]
NESTED_CASES = [(np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object), np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object)), (np.array([np.array([np.array([50, 70]), np.array([90])], dtype=object), np.array([np.array([20, 30])], dtype=object)], dtype=object), np.array([np.array([np.array([50, 70]), np.array([90])], dtype=object), np.array([np.array([20, 30])], dtype=object)], dtype=object)), (np.array([[50, 70, 90], [20, 30]], dtype=object), np.array([[50, 70, 90], [20, 30]], dtype=object)), (np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object), np.array([[1, 2, 3], [4, 5]], dtype=object)), (np.array([np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object), np.array([np.array([6]), np.array([7, 8]), np.array([9])], dtype=object)], dtype=object), np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object)), (np.array([subarr, None], dtype=object), np.array([[[None, 'b'], ['c', 'd']], None], dtype=object)), (np.array([{'f1': 1, 'f2': np.array(['a', 'b'], dtype=object)}], dtype=object), np.array([{'f1': 1, 'f2': np.array(['a', 'b'], dtype=object)}], dtype=object)), (np.array([{'f1': 1, 'f2': np.array(['a', 'b'], dtype=object)}], dtype=object), np.array([{'f1': 1, 'f2': ['a', 'b']}], dtype=object)), (np.array([np.array([{'f1': 1, 'f2': np.array(['a', 'b'], dtype=object)}], dtype=object), np.array([], dtype=object)], dtype=object), np.array([[{'f1': 1, 'f2': ['a', 'b']}], []], dtype=object))]

@pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
@pytest.mark.parametrize('a,b', NESTED_CASES)
def test_assert_almost_equal_array_nested(a, b):
    if False:
        while True:
            i = 10
    _assert_almost_equal_both(a, b)