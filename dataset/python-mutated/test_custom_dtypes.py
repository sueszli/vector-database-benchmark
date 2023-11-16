import sys
from tempfile import NamedTemporaryFile
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy._core._multiarray_umath import _discover_array_parameters as discover_array_params, _get_sfloat_dtype
SF = _get_sfloat_dtype()

class TestSFloat:

    def _get_array(self, scaling, aligned=True):
        if False:
            i = 10
            return i + 15
        if not aligned:
            a = np.empty(3 * 8 + 1, dtype=np.uint8)[1:]
            a = a.view(np.float64)
            a[:] = [1.0, 2.0, 3.0]
        else:
            a = np.array([1.0, 2.0, 3.0])
        a *= 1.0 / scaling
        return a.view(SF(scaling))

    def test_sfloat_rescaled(self):
        if False:
            i = 10
            return i + 15
        sf = SF(1.0)
        sf2 = sf.scaled_by(2.0)
        assert sf2.get_scaling() == 2.0
        sf6 = sf2.scaled_by(3.0)
        assert sf6.get_scaling() == 6.0

    def test_class_discovery(self):
        if False:
            return 10
        (dt, _) = discover_array_params([1.0, 2.0, 3.0], dtype=SF)
        assert dt == SF(1.0)

    @pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
    def test_scaled_float_from_floats(self, scaling):
        if False:
            i = 10
            return i + 15
        a = np.array([1.0, 2.0, 3.0], dtype=SF(scaling))
        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1.0, 2.0, 3.0])

    def test_repr(self):
        if False:
            print('Hello World!')
        assert repr(SF(scaling=1.0)) == '_ScaledFloatTestDType(scaling=1.0)'

    def test_dtype_name(self):
        if False:
            return 10
        assert SF(1.0).name == '_ScaledFloatTestDType64'

    def test_sfloat_structured_dtype_printing(self):
        if False:
            print('Hello World!')
        dt = np.dtype([('id', int), ('value', SF(0.5))])
        assert "('value', '_ScaledFloatTestDType64')" in repr(dt)

    @pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
    def test_sfloat_from_float(self, scaling):
        if False:
            return 10
        a = np.array([1.0, 2.0, 3.0]).astype(dtype=SF(scaling))
        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1.0, 2.0, 3.0])

    @pytest.mark.parametrize('aligned', [True, False])
    @pytest.mark.parametrize('scaling', [1.0, -1.0, 2.0])
    def test_sfloat_getitem(self, aligned, scaling):
        if False:
            i = 10
            return i + 15
        a = self._get_array(1.0, aligned)
        assert a.tolist() == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize('aligned', [True, False])
    def test_sfloat_casts(self, aligned):
        if False:
            while True:
                i = 10
        a = self._get_array(1.0, aligned)
        assert np.can_cast(a, SF(-1.0), casting='equiv')
        assert not np.can_cast(a, SF(-1.0), casting='no')
        na = a.astype(SF(-1.0))
        assert_array_equal(-1 * na.view(np.float64), a.view(np.float64))
        assert np.can_cast(a, SF(2.0), casting='same_kind')
        assert not np.can_cast(a, SF(2.0), casting='safe')
        a2 = a.astype(SF(2.0))
        assert_array_equal(2 * a2.view(np.float64), a.view(np.float64))

    @pytest.mark.parametrize('aligned', [True, False])
    def test_sfloat_cast_internal_errors(self, aligned):
        if False:
            i = 10
            return i + 15
        a = self._get_array(2e+300, aligned)
        with pytest.raises(TypeError, match='error raised inside the core-loop: non-finite factor!'):
            a.astype(SF(2e-300))

    def test_sfloat_promotion(self):
        if False:
            for i in range(10):
                print('nop')
        assert np.result_type(SF(2.0), SF(3.0)) == SF(3.0)
        assert np.result_type(SF(3.0), SF(2.0)) == SF(3.0)
        assert np.result_type(SF(3.0), np.float64) == SF(3.0)
        assert np.result_type(np.float64, SF(0.5)) == SF(1.0)
        with pytest.raises(TypeError):
            np.result_type(SF(1.0), np.int64)

    def test_basic_multiply(self):
        if False:
            return 10
        a = self._get_array(2.0)
        b = self._get_array(4.0)
        res = a * b
        assert res.dtype.get_scaling() == 8.0
        expected_view = a.view(np.float64) * b.view(np.float64)
        assert_array_equal(res.view(np.float64), expected_view)

    def test_possible_and_impossible_reduce(self):
        if False:
            while True:
                i = 10
        a = self._get_array(2.0)
        res = np.add.reduce(a, initial=0.0)
        assert res == a.astype(np.float64).sum()
        with pytest.raises(TypeError, match='the resolved dtypes are not compatible'):
            np.multiply.reduce(a)

    def test_basic_ufunc_at(self):
        if False:
            i = 10
            return i + 15
        float_a = np.array([1.0, 2.0, 3.0])
        b = self._get_array(2.0)
        float_b = b.view(np.float64).copy()
        np.multiply.at(float_b, [1, 1, 1], float_a)
        np.multiply.at(b, [1, 1, 1], float_a)
        assert_array_equal(b.view(np.float64), float_b)

    def test_basic_multiply_promotion(self):
        if False:
            print('Hello World!')
        float_a = np.array([1.0, 2.0, 3.0])
        b = self._get_array(2.0)
        res1 = float_a * b
        res2 = b * float_a
        assert res1.dtype == res2.dtype == b.dtype
        expected_view = float_a * b.view(np.float64)
        assert_array_equal(res1.view(np.float64), expected_view)
        assert_array_equal(res2.view(np.float64), expected_view)
        np.multiply(b, float_a, out=res2)
        with pytest.raises(TypeError):
            np.multiply(b, float_a, out=np.arange(3))

    def test_basic_addition(self):
        if False:
            print('Hello World!')
        a = self._get_array(2.0)
        b = self._get_array(4.0)
        res = a + b
        assert res.dtype == np.result_type(a.dtype, b.dtype)
        expected_view = a.astype(res.dtype).view(np.float64) + b.astype(res.dtype).view(np.float64)
        assert_array_equal(res.view(np.float64), expected_view)

    def test_addition_cast_safety(self):
        if False:
            print('Hello World!')
        'The addition method is special for the scaled float, because it\n        includes the "cast" between different factors, thus cast-safety\n        is influenced by the implementation.\n        '
        a = self._get_array(2.0)
        b = self._get_array(-2.0)
        c = self._get_array(3.0)
        np.add(a, b, casting='equiv')
        with pytest.raises(TypeError):
            np.add(a, b, casting='no')
        with pytest.raises(TypeError):
            np.add(a, c, casting='safe')
        with pytest.raises(TypeError):
            np.add(a, a, out=c, casting='safe')

    @pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_casts_to_bool(self, ufunc):
        if False:
            for i in range(10):
                print('nop')
        a = self._get_array(2.0)
        a[0] = 0.0
        float_equiv = a.astype(float)
        expected = ufunc(float_equiv, float_equiv)
        res = ufunc(a, a)
        assert_array_equal(res, expected)
        expected = ufunc.reduce(float_equiv)
        res = ufunc.reduce(a)
        assert_array_equal(res, expected)
        with pytest.raises(TypeError):
            ufunc(a, a, out=np.empty(a.shape, dtype=int), casting='equiv')

    def test_wrapped_and_wrapped_reductions(self):
        if False:
            for i in range(10):
                print('nop')
        a = self._get_array(2.0)
        float_equiv = a.astype(float)
        expected = np.hypot(float_equiv, float_equiv)
        res = np.hypot(a, a)
        assert res.dtype == a.dtype
        res_float = res.view(np.float64) * 2
        assert_array_equal(res_float, expected)
        res = np.hypot.reduce(a, keepdims=True)
        assert res.dtype == a.dtype
        expected = np.hypot.reduce(float_equiv, keepdims=True)
        assert res.view(np.float64) * 2 == expected

    def test_astype_class(self):
        if False:
            while True:
                i = 10
        arr = np.array([1.0, 2.0, 3.0], dtype=object)
        res = arr.astype(SF)
        expected = arr.astype(SF(1.0))
        assert_array_equal(res.view(np.float64), expected.view(np.float64))

    def test_creation_class(self):
        if False:
            print('Hello World!')
        arr1 = np.array([1.0, 2.0, 3.0], dtype=SF)
        assert arr1.dtype == SF(1.0)
        arr2 = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
        assert_array_equal(arr1.view(np.float64), arr2.view(np.float64))
        assert arr1.dtype == arr2.dtype
        assert np.empty(3, dtype=SF).dtype == SF(1.0)
        assert np.empty_like(arr1, dtype=SF).dtype == SF(1.0)
        assert np.zeros(3, dtype=SF).dtype == SF(1.0)
        assert np.zeros_like(arr1, dtype=SF).dtype == SF(1.0)

    def test_np_save_load(self):
        if False:
            print('Hello World!')
        np._ScaledFloatTestDType = SF
        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
        with NamedTemporaryFile('wb', delete=False, suffix='.npz') as f:
            with pytest.warns(UserWarning) as record:
                np.savez(f.name, arr)
        assert len(record) == 1
        with np.load(f.name, allow_pickle=True) as data:
            larr = data['arr_0']
        assert_array_equal(arr.view(np.float64), larr.view(np.float64))
        assert larr.dtype == arr.dtype == SF(1.0)
        del np._ScaledFloatTestDType

    def test_flatiter(self):
        if False:
            while True:
                i = 10
        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
        for (i, val) in enumerate(arr.flat):
            assert arr[i] == val

    @pytest.mark.parametrize('index', [[1, 2], ..., slice(None, 2, None), np.array([True, True, False]), np.array([0, 1])], ids=['int_list', 'ellipsis', 'slice', 'bool_array', 'int_array'])
    def test_flatiter_index(self, index):
        if False:
            return 10
        arr = np.array([1.0, 2.0, 3.0], dtype=SF(1.0))
        np.testing.assert_array_equal(arr[index].view(np.float64), arr.flat[index].view(np.float64))
        arr2 = arr.copy()
        arr[index] = 5.0
        arr2.flat[index] = 5.0
        np.testing.assert_array_equal(arr.view(np.float64), arr2.view(np.float64))

def test_type_pickle():
    if False:
        for i in range(10):
            print('nop')
    import pickle
    np._ScaledFloatTestDType = SF
    s = pickle.dumps(SF)
    res = pickle.loads(s)
    assert res is SF
    del np._ScaledFloatTestDType

def test_is_numeric():
    if False:
        print('Hello World!')
    assert SF._is_numeric