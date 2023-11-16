import unittest
import numpy
import cupy
from cupy import testing
import cupyx.scipy.special

def _boundary_inputs(boundary, rtol, atol):
    if False:
        print('Hello World!')
    left = boundary * (1 - numpy.copysign(rtol, boundary)) - atol
    right = boundary * (1 + numpy.copysign(rtol, boundary)) + atol
    return [left, boundary, right]

class _TestBase(object):

    def test_erf(self):
        if False:
            i = 10
            return i + 15
        self.check_unary('erf')

    def test_erfc(self):
        if False:
            while True:
                i = 10
        self.check_unary('erfc')

    def test_erfcx(self):
        if False:
            while True:
                i = 10
        self.check_unary('erfcx')

    @testing.with_requires('scipy>=1.4.0')
    def test_erfinv(self):
        if False:
            return 10
        self.check_unary('erfinv')
        self.check_unary_random('erfinv', scale=2, offset=-1)
        self.check_unary_boundary('erfinv', boundary=-1)
        self.check_unary_boundary('erfinv', boundary=1)

    @testing.with_requires('scipy>=1.4.0')
    def test_erfcinv(self):
        if False:
            return 10
        self.check_unary('erfcinv')
        self.check_unary_random('erfcinv', scale=2, offset=0)
        self.check_unary_boundary('erfcinv', boundary=0)
        self.check_unary_boundary('erfcinv', boundary=2)

@testing.with_requires('scipy')
class TestSpecial(unittest.TestCase, _TestBase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        if False:
            while True:
                i = 10
        import scipy.special
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(scp.special, name)(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def check_unary_random(self, name, xp, scp, dtype, scale, offset):
        if False:
            while True:
                i = 10
        import scipy.special
        a = testing.shaped_random((2, 3), xp, dtype, scale=scale) + offset
        return getattr(scp.special, name)(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def check_unary_boundary(self, name, xp, scp, dtype, boundary):
        if False:
            i = 10
            return i + 15
        import scipy.special
        a = _boundary_inputs(boundary, 1.0 / 1024, 1.0 / 1024)
        a = xp.array(a, dtype=dtype)
        return getattr(scp.special, name)(a)

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_dtypes(['f', 'd'])
    def test_erfinv_behavior(self, dtype):
        if False:
            while True:
                i = 10
        a = cupy.empty((1,), dtype=dtype)
        a[:] = 1.0 + 1e-06
        a = cupyx.scipy.special.erfinv(a)
        assert cupy.isnan(a)
        a[:] = -1.0 - 1e-06
        a = cupyx.scipy.special.erfinv(a)
        assert cupy.isnan(a)
        a[:] = 1.0
        a = cupyx.scipy.special.erfinv(a)
        assert numpy.isposinf(cupy.asnumpy(a))
        a[:] = -1.0
        a = cupyx.scipy.special.erfinv(a)
        assert numpy.isneginf(cupy.asnumpy(a))

    @testing.with_requires('scipy>=1.4.0')
    @testing.for_dtypes(['f', 'd'])
    def test_erfcinv_behavior(self, dtype):
        if False:
            return 10
        a = cupy.empty((1,), dtype=dtype)
        a[:] = 2.0 + 1e-06
        a = cupyx.scipy.special.erfcinv(a)
        assert cupy.isnan(a)
        a[:] = 0.0 - 1e-06
        a = cupyx.scipy.special.erfcinv(a)
        assert cupy.isnan(a)
        a[:] = 0.0
        a = cupyx.scipy.special.erfcinv(a)
        assert numpy.isposinf(cupy.asnumpy(a))
        a[:] = 2.0
        a = cupyx.scipy.special.erfcinv(a)
        assert numpy.isneginf(cupy.asnumpy(a))

@testing.with_requires('scipy')
class TestFusionSpecial(unittest.TestCase, _TestBase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        if False:
            return 10
        import scipy.special
        a = testing.shaped_arange((2, 3), xp, dtype)

        @cupy.fuse()
        def f(x):
            if False:
                print('Hello World!')
            return getattr(scp.special, name)(x)
        return f(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def check_unary_random(self, name, xp, scp, dtype, scale, offset):
        if False:
            print('Hello World!')
        import scipy.special
        a = testing.shaped_random((2, 3), xp, dtype, scale=scale) + offset

        @cupy.fuse()
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return getattr(scp.special, name)(x)
        return f(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-05, scipy_name='scp')
    def check_unary_boundary(self, name, xp, scp, dtype, boundary):
        if False:
            i = 10
            return i + 15
        import scipy.special
        a = _boundary_inputs(boundary, 1.0 / 1024, 1.0 / 1024)
        a = xp.array(a, dtype=dtype)

        @cupy.fuse()
        def f(x):
            if False:
                i = 10
                return i + 15
            return getattr(scp.special, name)(x)
        return f(a)