import numpy
import pytest
import cupy
from cupy import testing
import cupyx.scipy.special

def _get_logspace_max(dtype):
    if False:
        for i in range(10):
            print('nop')
    if dtype.char == 'd':
        return 200
    elif dtype.char == 'f':
        return 30
    elif dtype.char == 'e':
        return 5

@testing.with_requires('scipy')
class TestBeta:

    @pytest.mark.parametrize('function', ['beta', 'betaln'])
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_arange(self, xp, scp, dtype, function):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        func = getattr(scp.special, function)
        a = testing.shaped_arange((1, 10), xp, dtype)
        b = testing.shaped_arange((10, 1), xp, dtype)
        return func(a, b)

    @pytest.mark.skipif(cupy.cuda.runtime.is_hip and cupy.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    @pytest.mark.parametrize('function', ['beta', 'betaln'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype, function):
        if False:
            while True:
                i = 10
        import scipy.special
        func = getattr(scp.special, function)
        x = xp.linspace(-20, 21, 50, dtype=dtype)
        return func(x[:, xp.newaxis], x[xp.newaxis, :])

    @pytest.mark.parametrize('function', ['beta', 'betaln'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_logspace(self, xp, scp, dtype, function):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        func = getattr(scp.special, function)
        lmax = _get_logspace_max(xp.dtype(dtype))
        x = xp.logspace(-lmax, lmax, 32, dtype=dtype)
        return func(x[:, xp.newaxis], x[xp.newaxis, :])

    @pytest.mark.parametrize('function', ['beta', 'betaln'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype, function):
        if False:
            print('Hello World!')
        import scipy.special
        func = getattr(scp.special, function)
        a = xp.array([-numpy.inf, numpy.nan, numpy.inf, 0], dtype=dtype)
        return func(a[:, xp.newaxis], a[xp.newaxis, :])

    def test_beta_specific_vals(self):
        if False:
            i = 10
            return i + 15
        special = cupyx.scipy.special
        testing.assert_allclose(special.beta(1, 1), 1.0)
        testing.assert_allclose(special.beta(-100.3, 1e-200), special.gamma(1e-200))
        testing.assert_allclose(special.beta(0.0342, 171), 24.070498359873497, rtol=1e-13, atol=0)

    def test_betaln_specific_vals(self):
        if False:
            for i in range(10):
                print('nop')
        special = cupyx.scipy.special
        testing.assert_allclose(special.betaln(1, 1), 0.0, atol=1e-10)
        testing.assert_allclose(special.betaln(-100.3, 1e-200), special.gammaln(1e-200))
        testing.assert_allclose(special.betaln(0.0342, 170), 3.1811881124242447, rtol=1e-14, atol=0)

@testing.with_requires('scipy')
class TestBetaInc:

    @pytest.mark.parametrize('function', ['betainc', 'betaincinv'])
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_arange(self, xp, scp, dtype, function):
        if False:
            while True:
                i = 10
        import scipy.special
        func = getattr(scp.special, function)
        a = testing.shaped_arange((1, 10, 1), xp, dtype)
        b = testing.shaped_arange((10, 1, 1), xp, dtype)
        x = xp.asarray([0, 0.25, 0.5, 0.75, 1], dtype=dtype).reshape(1, 1, 5)
        return func(a, b, x)

    @pytest.mark.parametrize('function', ['betainc', 'betaincinv'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype, function):
        if False:
            print('Hello World!')
        import scipy.special
        func = getattr(scp.special, function)
        tmp = xp.linspace(-20, 21, 10, dtype=dtype)
        a = tmp[:, xp.newaxis, xp.newaxis]
        b = tmp[xp.newaxis, :, xp.newaxis]
        x = xp.linspace(0, 1, 5, dtype=dtype)[xp.newaxis, xp.newaxis, :]
        return func(a, b, x)

    @pytest.mark.parametrize('function', ['betainc', 'betaincinv'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_beta_inf_and_nan(self, xp, scp, dtype, function):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        func = getattr(scp.special, function)
        a = xp.array([-numpy.inf, numpy.nan, numpy.inf, 0], dtype=dtype)
        return func(a[:, xp.newaxis, xp.newaxis], a[xp.newaxis, :, xp.newaxis], a[xp.newaxis, xp.newaxis, :])

    def test_betainc_specific_vals(self):
        if False:
            for i in range(10):
                print('nop')
        special = cupyx.scipy.special
        assert special.betainc(1, 1, 1) == 1.0
        testing.assert_allclose(special.betainc(0.0342, 171, 1e-10), 0.5526991690180665)

    def test_betaincinv_specific_vals(self):
        if False:
            print('Hello World!')
        special = cupyx.scipy.special
        assert special.betaincinv(1, 1, 1) == 1.0
        testing.assert_allclose(special.betaincinv(0.0342, 171, 0.25), 8.423131693549896e-21, rtol=3e-12, atol=0)