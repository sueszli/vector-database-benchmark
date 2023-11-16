import numpy
from cupy import testing
import cupyx.scipy.special
try:
    import scipy.special
except ImportError:
    pass

@testing.with_requires('scipy')
class TestSphericalBessel:

    @testing.for_dtypes('i', name='order_dtype')
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-12, rtol=1e-12, scipy_name='scp')
    def test_spherical_yn_inf_and_nan(self, xp, scp, dtype, order_dtype):
        if False:
            return 10
        n = xp.arange(0, 100, dtype=order_dtype)
        a = xp.array([numpy.nan, numpy.inf, -numpy.inf], dtype=dtype)
        return scp.special.spherical_yn(n[xp.newaxis, :], a[:, xp.newaxis])

    @testing.for_dtypes('i', name='order_dtype')
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-12, rtol=1e-12, scipy_name='scp')
    def test_spherical_yn_linspace(self, xp, scp, dtype, order_dtype):
        if False:
            i = 10
            return i + 15
        n = xp.arange(0, 10, dtype=order_dtype)
        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return scp.special.spherical_yn(n[xp.newaxis, :], a[:, xp.newaxis])