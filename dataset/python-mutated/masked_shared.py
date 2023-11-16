"""
Tests shared by MaskedArray subclasses.
"""
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil

class ComparisonOps(BaseOpsUtil):

    def _compare_other(self, data, op, other):
        if False:
            i = 10
            return i + 15
        result = pd.Series(op(data, other))
        expected = pd.Series(op(data._data, other), dtype='boolean')
        expected[data._mask] = pd.NA
        tm.assert_series_equal(result, expected)
        ser = pd.Series(data)
        result = op(ser, other)
        expected = op(pd.Series(data._data), other).astype('boolean')
        expected[data._mask] = pd.NA
        tm.assert_series_equal(result, expected)

    def test_scalar(self, other, comparison_op, dtype):
        if False:
            while True:
                i = 10
        op = comparison_op
        left = pd.array([1, 0, None], dtype=dtype)
        result = op(left, other)
        if other is pd.NA:
            expected = pd.array([None, None, None], dtype='boolean')
        else:
            values = op(left._data, other)
            expected = pd.arrays.BooleanArray(values, left._mask, copy=True)
        tm.assert_extension_array_equal(result, expected)
        result[0] = pd.NA
        tm.assert_extension_array_equal(left, pd.array([1, 0, None], dtype=dtype))

class NumericOps:

    def test_searchsorted_nan(self, dtype):
        if False:
            print('Hello World!')
        arr = pd.array(range(10), dtype=dtype)
        assert arr.searchsorted(np.nan, side='left') == 10
        assert arr.searchsorted(np.nan, side='right') == 10

    def test_no_shared_mask(self, data):
        if False:
            print('Hello World!')
        result = data + 1
        assert not tm.shares_memory(result, data)

    def test_array(self, comparison_op, dtype):
        if False:
            print('Hello World!')
        op = comparison_op
        left = pd.array([0, 1, 2, None, None, None], dtype=dtype)
        right = pd.array([0, 1, None, 0, 1, None], dtype=dtype)
        result = op(left, right)
        values = op(left._data, right._data)
        mask = left._mask | right._mask
        expected = pd.arrays.BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)
        result[0] = pd.NA
        tm.assert_extension_array_equal(left, pd.array([0, 1, 2, None, None, None], dtype=dtype))
        tm.assert_extension_array_equal(right, pd.array([0, 1, None, 0, 1, None], dtype=dtype))

    def test_compare_with_booleanarray(self, comparison_op, dtype):
        if False:
            for i in range(10):
                print('nop')
        op = comparison_op
        left = pd.array([True, False, None] * 3, dtype='boolean')
        right = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype=dtype)
        other = pd.array([False] * 3 + [True] * 3 + [None] * 3, dtype='boolean')
        expected = op(left, other)
        result = op(left, right)
        tm.assert_extension_array_equal(result, expected)
        expected = op(other, left)
        result = op(right, left)
        tm.assert_extension_array_equal(result, expected)

    def test_compare_to_string(self, dtype):
        if False:
            while True:
                i = 10
        ser = pd.Series([1, None], dtype=dtype)
        result = ser == 'a'
        expected = pd.Series([False, pd.NA], dtype='boolean')
        tm.assert_series_equal(result, expected)

    def test_ufunc_with_out(self, dtype):
        if False:
            print('Hello World!')
        arr = pd.array([1, 2, 3], dtype=dtype)
        arr2 = pd.array([1, 2, pd.NA], dtype=dtype)
        mask = arr == arr
        mask2 = arr2 == arr2
        result = np.zeros(3, dtype=bool)
        result |= mask
        assert isinstance(result, np.ndarray)
        assert result.all()
        result = np.zeros(3, dtype=bool)
        msg = "Specify an appropriate 'na_value' for this dtype"
        with pytest.raises(ValueError, match=msg):
            result |= mask2
        res = np.add(arr, arr2)
        expected = pd.array([2, 4, pd.NA], dtype=dtype)
        tm.assert_extension_array_equal(res, expected)
        res = np.add(arr, arr2, out=arr)
        assert res is arr
        tm.assert_extension_array_equal(res, expected)
        tm.assert_extension_array_equal(arr, expected)

    def test_mul_td64_array(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        arr = pd.array([1, 2, pd.NA], dtype=dtype)
        other = np.arange(3, dtype=np.int64).view('m8[ns]')
        result = arr * other
        expected = pd.array([pd.Timedelta(0), pd.Timedelta(2), pd.NaT])
        tm.assert_extension_array_equal(result, expected)