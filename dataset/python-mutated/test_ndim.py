import numpy
import unittest
import cupy
from cupy import testing

class TestNdim(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_ndim_ndarray1d(self, xp):
        if False:
            for i in range(10):
                print('nop')
        return xp.ndim(xp.arange(5))

    @testing.numpy_cupy_equal()
    def test_ndim_ndarray2d(self, xp):
        if False:
            for i in range(10):
                print('nop')
        return xp.ndim(xp.ones((2, 4)))

    @testing.numpy_cupy_equal()
    def test_ndim_ndarray0d(self, xp):
        if False:
            print('Hello World!')
        return xp.ndim(xp.asarray(5))

    @testing.numpy_cupy_equal()
    def test_ndim_scalar(self, xp):
        if False:
            return 10
        return xp.ndim(5)

    @testing.numpy_cupy_equal()
    def test_ndim_none(self, xp):
        if False:
            print('Hello World!')
        return xp.ndim(None)

    @testing.numpy_cupy_equal()
    def test_ndim_string(self, xp):
        if False:
            return 10
        return xp.ndim('abc')

    @testing.numpy_cupy_equal()
    def test_ndim_list1(self, xp):
        if False:
            for i in range(10):
                print('nop')
        return xp.ndim([1, 2, 3])

    @testing.numpy_cupy_equal()
    def test_ndim_list2(self, xp):
        if False:
            i = 10
            return i + 15
        return xp.ndim([[1, 2, 3], [4, 5, 6]])

    @testing.numpy_cupy_equal()
    def test_ndim_tuple(self, xp):
        if False:
            print('Hello World!')
        return xp.ndim(((1, 2, 3), (4, 5, 6)))

    @testing.numpy_cupy_equal()
    def test_ndim_set(self, xp):
        if False:
            while True:
                i = 10
        return xp.ndim({1, 2, 3})

    @testing.numpy_cupy_equal()
    def test_ndim_object(self, xp):
        if False:
            i = 10
            return i + 15
        return xp.ndim(dict(a=5, b='b'))

    def test_ndim_array_function(self):
        if False:
            return 10
        a = cupy.ones((4, 4))
        assert numpy.ndim(a) == 2
        a = cupy.asarray(5)
        assert numpy.ndim(a) == 0
        a = numpy.ones((4, 4))
        assert cupy.ndim(a) == 2
        a = numpy.asarray(5)
        assert cupy.ndim(a) == 0