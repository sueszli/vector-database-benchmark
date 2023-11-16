import numpy
from cupy import testing
import cupyx.scipy.special

@testing.with_requires('scipy')
class TestBinom:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        n = testing.shaped_arange((40, 100), xp, dtype) + 20
        k = testing.shaped_arange((40, 100), xp, dtype)
        return scp.special.binom(n, k)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        n = xp.linspace(30, 60, 1000, dtype=dtype)
        k = xp.linspace(15, 60, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_linspace_largen(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        import scipy.special
        n = xp.linspace(10000000000.0, 90000000000.0, 1000, dtype=dtype)
        k = xp.linspace(0.01, 0.9, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_linspace_largeposk(self, xp, scp, dtype):
        if False:
            return 10
        import scipy.special
        n = xp.linspace(0.01, 0.9, 1000, dtype=dtype)
        k = xp.linspace(10000000000.0 + 0.5, 90000000000.0 + 0.5, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_linspace_largenegk(self, xp, scp, dtype):
        if False:
            print('Hello World!')
        import scipy.special
        n = xp.linspace(0.01, 0.9, 1000, dtype=dtype)
        k = xp.linspace(0.5 - 10000000000.0, 0.5 - 90000000000.0, 1000, dtype=dtype)
        return scp.special.binom(n, k)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_nan_inf(self, xp, scp, dtype):
        if False:
            print('Hello World!')
        import scipy.special
        a = xp.array([-numpy.inf, numpy.nan, numpy.inf, 0, -1, 100000000.0, 50000000.0], dtype=dtype)
        return scp.special.binom(a[:, xp.newaxis], a[xp.newaxis, :])