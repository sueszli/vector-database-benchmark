import unittest
from cupy import testing
import cupyx.scipy.special
import numpy
import warnings

@testing.with_requires('scipy')
class TestPolygamma(unittest.TestCase):

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def test_arange(self, xp, scp, dtype):
        if False:
            for i in range(10):
                print('nop')
        import scipy.special
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_arange((2, 3), xp, dtype)
        return scp.special.polygamma(a, b)

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=0.001, rtol=0.001, scipy_name='scp')
    def test_linspace(self, xp, scp, dtype):
        if False:
            print('Hello World!')
        import scipy.special
        a = numpy.tile(numpy.arange(5), 200).astype(dtype)
        b = numpy.linspace(-30, 30, 1000, dtype=dtype)
        a = xp.asarray(a)
        b = xp.asarray(b)
        return scp.special.polygamma(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=0.01, rtol=0.001, scipy_name='scp')
    def test_scalar(self, xp, scp, dtype):
        if False:
            i = 10
            return i + 15
        import scipy.special
        return scp.special.polygamma(dtype(2.0), dtype(1.5)).astype(numpy.float32)

    @testing.with_requires('scipy>=1.1.0')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=0.01, rtol=0.001, scipy_name='scp')
    def test_inf_and_nan(self, xp, scp, dtype):
        if False:
            return 10
        import scipy.special
        x = numpy.array([-numpy.inf, numpy.nan, numpy.inf]).astype(dtype)
        a = numpy.tile(x, 3)
        b = numpy.repeat(x, 3)
        a = xp.asarray(a)
        b = xp.asarray(b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return scp.special.polygamma(a, b)