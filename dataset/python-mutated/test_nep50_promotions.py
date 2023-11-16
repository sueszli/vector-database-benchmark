"""
This file adds basic tests to test the NEP 50 style promotion compatibility
mode.  Most of these test are likely to be simply deleted again once NEP 50
is adopted in the main test suite.  A few may be moved elsewhere.
"""
import operator
import numpy as np
import pytest
import hypothesis
from hypothesis import strategies
from numpy.testing import assert_array_equal, IS_WASM

@pytest.fixture(scope='module', autouse=True)
def _weak_promotion_enabled():
    if False:
        while True:
            i = 10
    state = np._get_promotion_state()
    np._set_promotion_state('weak_and_warn')
    yield
    np._set_promotion_state(state)

@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for fp errors")
def test_nep50_examples():
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.uint8(1) + 2
    assert res.dtype == np.uint8
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([1], np.uint8) + np.int64(1)
    assert res.dtype == np.int64
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([1], np.uint8) + np.array(1, dtype=np.int64)
    assert res.dtype == np.int64
    with pytest.warns(UserWarning, match='result dtype changed'):
        with np.errstate(over='raise'):
            res = np.uint8(100) + 200
    assert res.dtype == np.uint8
    with pytest.warns(Warning) as recwarn:
        res = np.float32(1) + 3e+100
    warning = str(recwarn.pop(UserWarning).message)
    assert warning.startswith('result dtype changed')
    warning = str(recwarn.pop(RuntimeWarning).message)
    assert warning.startswith('overflow')
    assert len(recwarn) == 0
    assert np.isinf(res)
    assert res.dtype == np.float32
    res = np.array([0.1], np.float32) == np.float64(0.1)
    assert res[0] == False
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([0.1], np.float32) + np.float64(0.1)
    assert res.dtype == np.float64
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([1.0], np.float32) + np.int64(3)
    assert res.dtype == np.float64

@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
def test_nep50_weak_integers(dtype):
    if False:
        return 10
    np._set_promotion_state('weak')
    scalar_type = np.dtype(dtype).type
    maxint = int(np.iinfo(dtype).max)
    with np.errstate(over='warn'):
        with pytest.warns(RuntimeWarning):
            res = scalar_type(100) + maxint
    assert res.dtype == dtype
    res = np.array(100, dtype=dtype) + maxint
    assert res.dtype == dtype

@pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
def test_nep50_weak_integers_with_inexact(dtype):
    if False:
        print('Hello World!')
    np._set_promotion_state('weak')
    scalar_type = np.dtype(dtype).type
    too_big_int = int(np.finfo(dtype).max) * 2
    if dtype in 'dDG':
        with pytest.raises(OverflowError):
            scalar_type(1) + too_big_int
        with pytest.raises(OverflowError):
            np.array(1, dtype=dtype) + too_big_int
    else:
        if dtype in 'gG':
            try:
                str(too_big_int)
            except ValueError:
                pytest.skip('`huge_int -> string -> longdouble` failed')
        with pytest.warns(RuntimeWarning):
            res = scalar_type(1) + too_big_int
        assert res.dtype == dtype
        assert res == np.inf
        with pytest.warns(RuntimeWarning):
            res = np.add(np.array(1, dtype=dtype), too_big_int, dtype=dtype)
        assert res.dtype == dtype
        assert res == np.inf

@pytest.mark.parametrize('op', [operator.add, operator.pow])
def test_weak_promotion_scalar_path(op):
    if False:
        i = 10
        return i + 15
    np._set_promotion_state('weak')
    res = op(np.uint8(3), 5)
    assert res == op(3, 5)
    assert res.dtype == np.uint8 or res.dtype == bool
    with pytest.raises(OverflowError):
        op(np.uint8(3), 1000)
    res = op(np.float32(3), 5.0)
    assert res == op(3.0, 5.0)
    assert res.dtype == np.float32 or res.dtype == bool

def test_nep50_complex_promotion():
    if False:
        for i in range(10):
            print('nop')
    np._set_promotion_state('weak')
    with pytest.warns(RuntimeWarning, match='.*overflow'):
        res = np.complex64(3) + complex(2 ** 300)
    assert type(res) == np.complex64

def test_nep50_integer_conversion_errors():
    if False:
        return 10
    np._set_promotion_state('weak')
    with pytest.raises(OverflowError, match='.*uint8'):
        np.array([1], np.uint8) + 300
    with pytest.raises(OverflowError, match='.*uint8'):
        np.uint8(1) + 300
    with pytest.raises(OverflowError, match='Python integer -1 out of bounds for uint8'):
        np.uint8(1) + -1

def test_nep50_integer_regression():
    if False:
        print('Hello World!')
    np._set_promotion_state('legacy')
    arr = np.array(1)
    assert (arr + 2 ** 63).dtype == np.float64
    assert (arr[()] + 2 ** 63).dtype == np.float64

def test_nep50_with_axisconcatenator():
    if False:
        print('Hello World!')
    np._set_promotion_state('weak')
    with pytest.raises(OverflowError):
        np.r_[np.arange(5, dtype=np.int8), 255]

@pytest.mark.parametrize('ufunc', [np.add, np.power])
@pytest.mark.parametrize('state', ['weak', 'weak_and_warn'])
def test_nep50_huge_integers(ufunc, state):
    if False:
        while True:
            i = 10
    np._set_promotion_state(state)
    with pytest.raises(OverflowError):
        ufunc(np.int64(0), 2 ** 63)
    if state == 'weak_and_warn':
        with pytest.warns(UserWarning, match='result dtype changed.*float64.*uint64'):
            with pytest.raises(OverflowError):
                ufunc(np.uint64(0), 2 ** 64)
    else:
        with pytest.raises(OverflowError):
            ufunc(np.uint64(0), 2 ** 64)
    if state == 'weak_and_warn':
        with pytest.warns(UserWarning, match='result dtype changed.*float64.*uint64'):
            res = ufunc(np.uint64(1), 2 ** 63)
    else:
        res = ufunc(np.uint64(1), 2 ** 63)
    assert res.dtype == np.uint64
    assert res == ufunc(1, 2 ** 63, dtype=object)
    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2 ** 63)
    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2 ** 100)
    res = ufunc(1.0, 2 ** 100)
    assert isinstance(res, np.float64)

def test_nep50_in_concat_and_choose():
    if False:
        print('Hello World!')
    np._set_promotion_state('weak_and_warn')
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.concatenate([np.float32(1), 1.0], axis=None)
    assert res.dtype == 'float32'
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.choose(1, [np.float32(1), 1.0])
    assert res.dtype == 'float32'

@pytest.mark.parametrize('expected,dtypes,optional_dtypes', [(np.float32, [np.float32], [np.float16, 0.0, np.uint16, np.int16, np.int8, 0]), (np.complex64, [np.float32, 0j], [np.float16, 0.0, np.uint16, np.int16, np.int8, 0]), (np.float32, [np.int16, np.uint16, np.float16], [np.int8, np.uint8, np.float32, 0.0, 0]), (np.int32, [np.int16, np.uint16], [np.int8, np.uint8, 0, np.bool_])])
@hypothesis.given(data=strategies.data())
def test_expected_promotion(expected, dtypes, optional_dtypes, data):
    if False:
        for i in range(10):
            print('nop')
    np._set_promotion_state('weak')
    optional = data.draw(strategies.lists(strategies.sampled_from(dtypes + optional_dtypes)))
    all_dtypes = dtypes + optional
    dtypes_sample = data.draw(strategies.permutations(all_dtypes))
    res = np.result_type(*dtypes_sample)
    assert res == expected

@pytest.mark.parametrize('sctype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('other_val', [-2 * 100, -1, 0, 9, 10, 11, 2 ** 63, 2 * 100])
@pytest.mark.parametrize('comp', [operator.eq, operator.ne, operator.le, operator.lt, operator.ge, operator.gt])
def test_integer_comparison(sctype, other_val, comp):
    if False:
        return 10
    np._set_promotion_state('weak')
    val_obj = 10
    val = sctype(val_obj)
    assert comp(10, other_val) == comp(val, other_val)
    assert comp(val, other_val) == comp(10, other_val)
    assert type(comp(val, other_val)) is np.bool_
    val_obj = np.array([10, 10], dtype=object)
    val = val_obj.astype(sctype)
    assert_array_equal(comp(val_obj, other_val), comp(val, other_val))
    assert_array_equal(comp(other_val, val_obj), comp(other_val, val))

@pytest.mark.parametrize('comp', [np.equal, np.not_equal, np.less_equal, np.less, np.greater_equal, np.greater])
def test_integer_integer_comparison(comp):
    if False:
        return 10
    np._set_promotion_state('weak')
    assert comp(2 ** 200, -2 ** 200) == comp(2 ** 200, -2 ** 200, dtype=object)