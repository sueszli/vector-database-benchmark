import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray

@pytest.fixture(params=['integer', 'block'])
def kind(request):
    if False:
        for i in range(10):
            print('nop')
    'kind kwarg to pass to SparseArray'
    return request.param

@pytest.fixture(params=[True, False])
def mix(request):
    if False:
        while True:
            i = 10
    '\n    Fixture returning True or False, determining whether to operate\n    op(sparse, dense) instead of op(sparse, sparse)\n    '
    return request.param

class TestSparseArrayArithmetics:

    def _assert(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        tm.assert_numpy_array_equal(a, b)

    def _check_numeric_ops(self, a, b, a_dense, b_dense, mix: bool, op):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(a_dense, np.ndarray):
            expected = op(pd.Series(a_dense), b_dense).values
        elif isinstance(b_dense, np.ndarray):
            expected = op(a_dense, pd.Series(b_dense)).values
        else:
            raise NotImplementedError
        with np.errstate(invalid='ignore', divide='ignore'):
            if mix:
                result = op(a, b_dense).to_dense()
            else:
                result = op(a, b).to_dense()
        self._assert(result, expected)

    def _check_bool_result(self, res):
        if False:
            return 10
        assert isinstance(res, SparseArray)
        assert isinstance(res.dtype, SparseDtype)
        assert res.dtype.subtype == np.bool_
        assert isinstance(res.fill_value, bool)

    def _check_comparison_ops(self, a, b, a_dense, b_dense):
        if False:
            i = 10
            return i + 15
        with np.errstate(invalid='ignore'):
            self._check_bool_result(a == b)
            self._assert((a == b).to_dense(), a_dense == b_dense)
            self._check_bool_result(a != b)
            self._assert((a != b).to_dense(), a_dense != b_dense)
            self._check_bool_result(a >= b)
            self._assert((a >= b).to_dense(), a_dense >= b_dense)
            self._check_bool_result(a <= b)
            self._assert((a <= b).to_dense(), a_dense <= b_dense)
            self._check_bool_result(a > b)
            self._assert((a > b).to_dense(), a_dense > b_dense)
            self._check_bool_result(a < b)
            self._assert((a < b).to_dense(), a_dense < b_dense)
            self._check_bool_result(a == b_dense)
            self._assert((a == b_dense).to_dense(), a_dense == b_dense)
            self._check_bool_result(a != b_dense)
            self._assert((a != b_dense).to_dense(), a_dense != b_dense)
            self._check_bool_result(a >= b_dense)
            self._assert((a >= b_dense).to_dense(), a_dense >= b_dense)
            self._check_bool_result(a <= b_dense)
            self._assert((a <= b_dense).to_dense(), a_dense <= b_dense)
            self._check_bool_result(a > b_dense)
            self._assert((a > b_dense).to_dense(), a_dense > b_dense)
            self._check_bool_result(a < b_dense)
            self._assert((a < b_dense).to_dense(), a_dense < b_dense)

    def _check_logical_ops(self, a, b, a_dense, b_dense):
        if False:
            while True:
                i = 10
        self._check_bool_result(a & b)
        self._assert((a & b).to_dense(), a_dense & b_dense)
        self._check_bool_result(a | b)
        self._assert((a | b).to_dense(), a_dense | b_dense)
        self._check_bool_result(a & b_dense)
        self._assert((a & b_dense).to_dense(), a_dense & b_dense)
        self._check_bool_result(a | b_dense)
        self._assert((a | b_dense).to_dense(), a_dense | b_dense)

    @pytest.mark.parametrize('scalar', [0, 1, 3])
    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_scalar(self, kind, mix, all_arithmetic_functions, fill_value, scalar, request):
        if False:
            i = 10
            return i + 15
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        a = SparseArray(values, kind=kind, fill_value=fill_value)
        self._check_numeric_ops(a, scalar, values, scalar, mix, op)

    def test_float_scalar_comparison(self, kind):
        if False:
            print('Hello World!')
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        a = SparseArray(values, kind=kind)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)
        a = SparseArray(values, kind=kind, fill_value=0)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)
        a = SparseArray(values, kind=kind, fill_value=2)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

    def test_float_same_index_without_nans(self, kind, mix, all_arithmetic_functions):
        if False:
            for i in range(10):
                print('nop')
        op = all_arithmetic_functions
        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_with_nans(self, kind, mix, all_arithmetic_functions, request):
        if False:
            for i in range(10):
                print('nop')
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_comparison(self, kind):
        if False:
            i = 10
            return i + 15
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_float_array(self, kind, mix, all_arithmetic_functions):
        if False:
            while True:
                i = 10
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_different_kind(self, mix, all_arithmetic_functions):
        if False:
            for i in range(10):
                print('nop')
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])
        a = SparseArray(values, kind='integer')
        b = SparseArray(rvalues, kind='block')
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, kind='integer', fill_value=0)
        b = SparseArray(rvalues, kind='block')
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind='integer', fill_value=0)
        b = SparseArray(rvalues, kind='block', fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind='integer', fill_value=1)
        b = SparseArray(rvalues, kind='block', fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_comparison(self, kind):
        if False:
            print('Hello World!')
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_int_array(self, kind, mix, all_arithmetic_functions):
        if False:
            print('Hello World!')
        op = all_arithmetic_functions
        dtype = np.int64
        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)
        a = SparseArray(values, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, fill_value=0, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, fill_value=1, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype, fill_value=1)
        b = SparseArray(rvalues, fill_value=2, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_int_array_comparison(self, kind):
        if False:
            i = 10
            return i + 15
        dtype = 'int64'
        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)
        a = SparseArray(values, dtype=dtype, kind=kind)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=1)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_same_index(self, kind, fill_value):
        if False:
            return 10
        values = np.array([True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, True, True], dtype=np.bool_)
        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_array_logical(self, kind, fill_value):
        if False:
            return 10
        values = np.array([True, False, True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, False, True, False, True], dtype=np.bool_)
        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    def test_mixed_array_float_int(self, kind, mix, all_arithmetic_functions, request):
        if False:
            while True:
                i = 10
        op = all_arithmetic_functions
        rdtype = 'int64'
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_mixed_array_comparison(self, kind):
        if False:
            return 10
        rdtype = 'int64'
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_xor(self):
        if False:
            return 10
        s = SparseArray([True, True, False, False])
        t = SparseArray([True, False, True, False])
        result = s ^ t
        sp_index = pd.core.arrays.sparse.IntIndex(4, np.array([0, 1, 2], dtype='int32'))
        expected = SparseArray([False, True, True], sparse_index=sp_index)
        tm.assert_sp_array_equal(result, expected)

@pytest.mark.parametrize('op', [operator.eq, operator.add])
def test_with_list(op):
    if False:
        print('Hello World!')
    arr = SparseArray([0, 1], fill_value=0)
    result = op(arr, [0, 1])
    expected = op(arr, SparseArray([0, 1]))
    tm.assert_sp_array_equal(result, expected)

def test_with_dataframe():
    if False:
        for i in range(10):
            print('nop')
    arr = SparseArray([0, 1], fill_value=0)
    df = pd.DataFrame([[1, 2], [3, 4]])
    result = arr.__add__(df)
    assert result is NotImplemented

def test_with_zerodim_ndarray():
    if False:
        return 10
    arr = SparseArray([0, 1], fill_value=0)
    result = arr * np.array(2)
    expected = arr * 2
    tm.assert_sp_array_equal(result, expected)

@pytest.mark.parametrize('ufunc', [np.abs, np.exp])
@pytest.mark.parametrize('arr', [SparseArray([0, 0, -1, 1]), SparseArray([None, None, -1, 1])])
def test_ufuncs(ufunc, arr):
    if False:
        while True:
            i = 10
    result = ufunc(arr)
    fill_value = ufunc(arr.fill_value)
    expected = SparseArray(ufunc(np.asarray(arr)), fill_value=fill_value)
    tm.assert_sp_array_equal(result, expected)

@pytest.mark.parametrize('a, b', [(SparseArray([0, 0, 0]), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2]))])
@pytest.mark.parametrize('ufunc', [np.add, np.greater])
def test_binary_ufuncs(ufunc, a, b):
    if False:
        while True:
            i = 10
    result = ufunc(a, b)
    expected = ufunc(np.asarray(a), np.asarray(b))
    assert isinstance(result, SparseArray)
    tm.assert_numpy_array_equal(np.asarray(result), expected)

def test_ndarray_inplace():
    if False:
        while True:
            i = 10
    sparray = SparseArray([0, 2, 0, 0])
    ndarray = np.array([0, 1, 2, 3])
    ndarray += sparray
    expected = np.array([0, 3, 2, 3])
    tm.assert_numpy_array_equal(ndarray, expected)

def test_sparray_inplace():
    if False:
        return 10
    sparray = SparseArray([0, 2, 0, 0])
    ndarray = np.array([0, 1, 2, 3])
    sparray += ndarray
    expected = SparseArray([0, 3, 2, 3], fill_value=0)
    tm.assert_sp_array_equal(sparray, expected)

@pytest.mark.parametrize('cons', [list, np.array, SparseArray])
def test_mismatched_length_cmp_op(cons):
    if False:
        i = 10
        return i + 15
    left = SparseArray([True, True])
    right = cons([True, True, True])
    with pytest.raises(ValueError, match='operands have mismatched length'):
        left & right

@pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'])
@pytest.mark.parametrize('fill_value', [np.nan, 3])
def test_binary_operators(op, fill_value):
    if False:
        return 10
    op = getattr(operator, op)
    data1 = np.random.default_rng(2).standard_normal(20)
    data2 = np.random.default_rng(2).standard_normal(20)
    data1[::2] = fill_value
    data2[::3] = fill_value
    first = SparseArray(data1, fill_value=fill_value)
    second = SparseArray(data2, fill_value=fill_value)
    with np.errstate(all='ignore'):
        res = op(first, second)
        exp = SparseArray(op(first.to_dense(), second.to_dense()), fill_value=first.fill_value)
        assert isinstance(res, SparseArray)
        tm.assert_almost_equal(res.to_dense(), exp.to_dense())
        res2 = op(first, second.to_dense())
        assert isinstance(res2, SparseArray)
        tm.assert_sp_array_equal(res, res2)
        res3 = op(first.to_dense(), second)
        assert isinstance(res3, SparseArray)
        tm.assert_sp_array_equal(res, res3)
        res4 = op(first, 4)
        assert isinstance(res4, SparseArray)
        try:
            exp = op(first.to_dense(), 4)
            exp_fv = op(first.fill_value, 4)
        except ValueError:
            pass
        else:
            tm.assert_almost_equal(res4.fill_value, exp_fv)
            tm.assert_almost_equal(res4.to_dense(), exp)