import numpy as np
import pytest
from pandas import Timestamp, array
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray

@pytest.mark.parametrize('kwargs', [{}, {'check_exact': False}, {'check_exact': True}])
def test_assert_extension_array_equal_not_exact(kwargs):
    if False:
        while True:
            i = 10
    arr1 = SparseArray([-0.17387645482451206, 0.3414148016424936])
    arr2 = SparseArray([-0.17387645482451206, 0.3414148016424937])
    if kwargs.get('check_exact', False):
        msg = 'ExtensionArray are different\n\nExtensionArray values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[-0\\.17387645482.*, 0\\.341414801642.*\\]\n\\[right\\]: \\[-0\\.17387645482.*, 0\\.341414801642.*\\]'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)

@pytest.mark.parametrize('decimals', range(10))
def test_assert_extension_array_equal_less_precise(decimals):
    if False:
        i = 10
        return i + 15
    rtol = 0.5 * 10 ** (-decimals)
    arr1 = SparseArray([0.5, 0.123456])
    arr2 = SparseArray([0.5, 0.123457])
    if decimals >= 5:
        msg = 'ExtensionArray are different\n\nExtensionArray values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[0\\.5, 0\\.123456\\]\n\\[right\\]: \\[0\\.5, 0\\.123457\\]'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)
    else:
        tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)

def test_assert_extension_array_equal_dtype_mismatch(check_dtype):
    if False:
        print('Hello World!')
    end = 5
    kwargs = {'check_dtype': check_dtype}
    arr1 = SparseArray(np.arange(end, dtype='int64'))
    arr2 = SparseArray(np.arange(end, dtype='int32'))
    if check_dtype:
        msg = 'ExtensionArray are different\n\nAttribute "dtype" are different\n\\[left\\]:  Sparse\\[int64, 0\\]\n\\[right\\]: Sparse\\[int32, 0\\]'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)

def test_assert_extension_array_equal_missing_values():
    if False:
        for i in range(10):
            print('nop')
    arr1 = SparseArray([np.nan, 1, 2, np.nan])
    arr2 = SparseArray([np.nan, 1, 2, 3])
    msg = 'ExtensionArray NA mask are different\n\nExtensionArray NA mask values are different \\(25\\.0 %\\)\n\\[left\\]:  \\[True, False, False, True\\]\n\\[right\\]: \\[True, False, False, False\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(arr1, arr2)

@pytest.mark.parametrize('side', ['left', 'right'])
def test_assert_extension_array_equal_non_extension_array(side):
    if False:
        for i in range(10):
            print('nop')
    numpy_array = np.arange(5)
    extension_array = SparseArray(numpy_array)
    msg = f'{side} is not an ExtensionArray'
    args = (numpy_array, extension_array) if side == 'left' else (extension_array, numpy_array)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(*args)

@pytest.mark.parametrize('right_dtype', ['Int32', 'int64'])
def test_assert_extension_array_equal_ignore_dtype_mismatch(right_dtype):
    if False:
        for i in range(10):
            print('nop')
    left = array([1, 2, 3], dtype='Int64')
    right = array([1, 2, 3], dtype=right_dtype)
    tm.assert_extension_array_equal(left, right, check_dtype=False)

def test_assert_extension_array_equal_time_units():
    if False:
        i = 10
        return i + 15
    timestamp = Timestamp('2023-11-04T12')
    naive = array([timestamp], dtype='datetime64[ns]')
    utc = array([timestamp], dtype='datetime64[ns, UTC]')
    tm.assert_extension_array_equal(naive, utc, check_dtype=False)
    tm.assert_extension_array_equal(utc, naive, check_dtype=False)