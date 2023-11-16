import cupyx.scipy.special
import cupy
from cupy import testing
from cupy.testing import numpy_cupy_allclose
try:
    import scipy.special
except ImportError:
    pass
atol = {'default': 1e-06, cupy.float16: 0.01}
rtol = {'default': 1e-06, cupy.float16: 0.01}

@testing.with_requires('scipy')
class Testexprel:

    @testing.for_dtypes('efdFD')
    @numpy_cupy_allclose(scipy_name='scp')
    def test_exprel(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        return scp.special.exprel(-1)

    @testing.for_dtypes('efdFD')
    @numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_exprel_2(self, xp, scp, dtype):
        if False:
            return 10
        return scp.special.exprel(1)

    @testing.for_dtypes('efdFD')
    @numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_exprel_large_values(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        return scp.special.exprel(720)

    @testing.for_dtypes('efdFD')
    @numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_exprel_small_value(self, xp, scp, dtype):
        if False:
            print('Hello World!')
        return scp.special.exprel(1e-17)

    @testing.for_dtypes('efdFD')
    @numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_exprel_zero_values(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        return scp.special.exprel(0.0)

    @testing.for_dtypes('edf')
    @numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_exprel_array_inputs(self, xp, scp, dtype):
        if False:
            print('Hello World!')
        n = testing.shaped_arange((5, 1, 2), xp, dtype) * 0.001
        return scp.special.exprel(n)

    @testing.for_dtypes('edf')
    @numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_exprel_array_inputs_2(self, xp, scp, dtype):
        if False:
            return 10
        n = testing.shaped_random((5, 3), xp, dtype) * 0.001
        return scp.special.exprel(n)