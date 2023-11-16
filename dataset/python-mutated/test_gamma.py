import numpy
import pytest
import cupy
from cupy import testing
import cupyx.scipy.special

@testing.with_requires('scipy')
class TestGamma:

    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def test_arange(self, xp, scp, dtype, function):
        if False:
            return 10
        import scipy.special
        a = testing.shaped_arange((2, 3), xp, dtype)
        func = getattr(scp.special, function)
        return func(a)

    @pytest.mark.skipif(cupy.cuda.runtime.is_hip and cupy.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype, function):
        if False:
            while True:
                i = 10
        import scipy.special
        a = numpy.linspace(-30, 30, 1000, dtype=dtype)
        if a.dtype.kind == 'c':
            a -= 1j * a
        a = xp.asarray(a)
        func = getattr(scp.special, function)
        return func(a)

    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=0.01, rtol=0.001, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype, function):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        if xp.dtype(dtype).kind == 'c':
            val = dtype(1.5 + 1j)
        else:
            val = dtype(1.5)
        func = getattr(scp.special, function)
        return func(val)

    @pytest.mark.parametrize('function', ['gamma', 'loggamma', 'rgamma'])
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=0.01, rtol=0.001, scipy_name='scp')
    @testing.with_requires('scipy>=1.5.0')
    def test_inf_and_nan(self, xp, scp, dtype, function):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        a = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = xp.asarray(a)
        func = getattr(scp.special, function)
        return func(a)