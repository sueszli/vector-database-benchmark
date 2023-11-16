import unittest
import numpy
from cupy import testing

class TestContent(unittest.TestCase):

    @testing.for_dtypes('efFdD')
    @testing.numpy_cupy_array_equal()
    def check_unary_inf(self, name, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([-3, numpy.inf, -1, -numpy.inf, 0, 1, 2], dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes('efFdD')
    @testing.numpy_cupy_array_equal()
    def check_unary_nan(self, name, xp, dtype):
        if False:
            return 10
        a = xp.array([-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, numpy.inf], dtype=dtype)
        return getattr(xp, name)(a)

    def test_isfinite(self):
        if False:
            i = 10
            return i + 15
        self.check_unary_inf('isfinite')

    def test_isinf(self):
        if False:
            return 10
        self.check_unary_inf('isinf')

    def test_isnan(self):
        if False:
            while True:
                i = 10
        self.check_unary_nan('isnan')

class TestUfuncLike(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def check_unary(self, name, xp):
        if False:
            for i in range(10):
                print('nop')
        a = xp.array([-3, xp.inf, -1, -xp.inf, 0, 1, 2, xp.nan])
        return getattr(xp, name)(a)

    def test_isneginf(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_unary('isneginf')

    def test_isposinf(self):
        if False:
            i = 10
            return i + 15
        self.check_unary('isposinf')