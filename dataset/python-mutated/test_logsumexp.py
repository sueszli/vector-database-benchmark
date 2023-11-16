import pytest
import numpy
import cupy
from cupy import testing
import cupyx.scipy.special
try:
    import scipy.special
except ImportError:
    pass

@testing.with_requires('scipy')
class TestLogsumexp:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_large_inputs(self, xp, scp, dtype):
        if False:
            while True:
                i = 10
        a = xp.arange(400)
        return scp.special.logsumexp(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_more_large_inputs(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        a = xp.arange(10000)
        return scp.special.logsumexp(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_large_numbers(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([1000, 1000]).astype(dtype)
        return scp.special.logsumexp(a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_keep_dimensions(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([[100, 1000], [10000000000.0, 1e-10]])
        return scp.special.logsumexp(a, axis=-1, keepdims=True)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-06)
    def test_array_inputs(self, xp, scp, dtype):
        if False:
            print('Hello World!')
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        a = testing.shaped_random((100, 1000), xp, dtype=dtype)
        return scp.special.logsumexp(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_sign_argument(self, xp, scp, dtype):
        if False:
            while True:
                i = 10
        a = xp.array([1, 1, 1]).astype(dtype)
        b = xp.array([1, -1, -1]).astype(dtype)
        return scp.special.logsumexp(a, b=b, return_sign=True)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_sign_zero(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([1, 1]).astype(dtype)
        b = xp.array([1, -1]).astype(dtype)
        return scp.special.logsumexp(a, b=b, return_sign=True)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-06)
    def test_sign_multi_dims(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        a = testing.shaped_random((1, 1, 3, 4), xp, dtype=dtype)
        b = testing.shaped_random((1, 1, 1, 4), xp, dtype=dtype)
        return scp.special.logsumexp(a, b=b, return_sign=True)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-06)
    def test_sign_multi_dims_axis(self, xp, scp, dtype):
        if False:
            while True:
                i = 10
        a = testing.shaped_random((1, 2, 3, 4), xp, dtype=dtype)
        b = testing.shaped_random((1, 2, 3, 4), xp, dtype=dtype)
        return scp.special.logsumexp(a, axis=2, b=b, return_sign=True)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-06)
    def test_sign_multi_dims_axis_2d(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        a = testing.shaped_random((1, 2, 3, 4), xp, dtype=dtype)
        b = testing.shaped_random((1, 2, 3, 4), xp, dtype=dtype)
        return scp.special.logsumexp(a, axis=(1, 3), b=b, return_sign=True)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_b_zero(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([1, 100], dtype=dtype)
        b = xp.array([1, 0], dtype=dtype)
        return scp.special.logsumexp(a, b=b)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_b_multi_dims(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        if xp.dtype(dtype).kind in 'u':
            pytest.skip()
        a = testing.shaped_arange((4, 1, 2, 1), xp, dtype=dtype)
        b = testing.shaped_arange((3, 1, 5), xp, dtype=dtype)
        return scp.special.logsumexp(a, b=b)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_special_values(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([cupy.inf, -cupy.inf, cupy.nan, -cupy.nan])
        return scp.special.logsumexp(a)

    @testing.for_all_dtypes(no_bool=True)
    def test_empty_array_inputs(self, dtype):
        if False:
            return 10
        a = numpy.array([], dtype=dtype)
        for xp in (scipy, cupyx.scipy):
            with pytest.raises(ValueError):
                xp.special.logsumexp(a)