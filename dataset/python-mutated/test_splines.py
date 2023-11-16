import pytest
from cupy import testing
from cupy.cuda import driver
from cupy.cuda import runtime
import cupyx.scipy.signal
try:
    import scipy.signal
except ImportError:
    pass

@testing.with_requires('scipy>=1.9')
class TestSymIIROrder:

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120])
    @pytest.mark.parametrize('precision', [-1, 2, 1.5, 1.0, 0.5, 0.25, 0.1, 0.002, 0.001])
    @testing.for_all_dtypes_combination(no_float16=True, no_bool=True, names=('dtype',))
    @testing.numpy_cupy_allclose(atol=1e-05, rtol=1e-05, scipy_name='scp', accept_error=True)
    def test_symiirorder1(self, size, precision, dtype, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        if xp.dtype(dtype).kind in {'i', 'u'}:
            pytest.skip()
        if runtime.is_hip and driver.get_build_version() < 50000000:
            pytest.skip()
        x = testing.shaped_random((size,), xp, dtype=dtype)
        (c0, z1) = (2.0, 0.5)
        return scp.signal.symiirorder1(x, c0, z1, precision)

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120])
    @pytest.mark.parametrize('precision', [-1, 2, 1.5, 1.0, 0.5, 0.25, 0.1, 0.002, 0.001])
    @pytest.mark.parametrize('omega', ['zero', 'pi', 'random'])
    @testing.for_all_dtypes_combination(no_float16=True, no_complex=True, no_bool=True, names=('dtype',))
    @testing.numpy_cupy_allclose(atol=2e-05, rtol=2e-05, scipy_name='scp', accept_error=True)
    def test_symiirorder2(self, size, precision, omega, dtype, xp, scp):
        if False:
            return 10
        if xp.dtype(dtype).kind in {'i', 'u'}:
            pytest.skip()
        if runtime.is_hip and driver.get_build_version() < 50000000:
            pytest.skip()
        if omega == 'pi':
            omega = xp.asarray(xp.pi, dtype=dtype)[0]
        elif omega == 'random':
            omega = testing.shaped_random((1,), xp, dtype=dtype, scale=2 * xp.pi)[0]
        else:
            omega = xp.zeros(1, dtype=dtype)[0]
        r = testing.shaped_random((1,), xp, scale=1)[0]
        x = testing.shaped_random((size,), xp, dtype=dtype)
        return scp.signal.symiirorder2(x, r, omega, precision)