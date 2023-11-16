import unittest
import numpy
from cupy import testing

class TestIsScalar(testing.NumpyAliasBasicTestBase):
    func = 'isscalar'

    @testing.with_requires('numpy>=1.18')
    def test_argspec(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_argspec()

@testing.parameterize(*testing.product({'value': [0, 0.0, True, numpy.int32(1), numpy.array([1, 2], numpy.int32), numpy.complex128(1), numpy.complex128(1j), numpy.complex128(1 + 1j), None, 'abc', '', int, numpy.int32]}))
class TestIsScalarValues(testing.NumpyAliasValuesTestBase):
    func = 'isscalar'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.args = (self.value,)

class TestIsScalarValues2(testing.NumpyAliasValuesTestBase):
    func = 'isscalar'

    def setUp(self):
        if False:
            print('Hello World!')
        value = object()
        self.args = (value,)

@testing.parameterize(*testing.product({'value': [numpy.ones(24, order='C'), numpy.ones((4, 6), order='C'), numpy.ones((4, 6), order='F'), numpy.ones((4, 6), order='C')[1:3][1:3]]}))
class TestIsFortran(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test(self, xp):
        if False:
            return 10
        return xp.isfortran(xp.asarray(self.value))

@testing.parameterize({'func': 'iscomplex'}, {'func': 'isreal'})
class TestTypeTestingFunctions(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        return getattr(xp, self.func)(xp.ones(5, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_scalar(self, xp, dtype):
        if False:
            while True:
                i = 10
        return getattr(xp, self.func)(dtype(3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_list(self, xp, dtype):
        if False:
            print('Hello World!')
        return getattr(xp, self.func)(testing.shaped_arange((2, 3), xp, dtype).tolist())

@testing.parameterize({'func': 'iscomplexobj'}, {'func': 'isrealobj'})
class TestTypeTestingObjFunctions(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test(self, xp, dtype):
        if False:
            i = 10
            return i + 15
        return getattr(xp, self.func)(xp.ones(5, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_scalar(self, xp, dtype):
        if False:
            print('Hello World!')
        return getattr(xp, self.func)(dtype(3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_list(self, xp, dtype):
        if False:
            print('Hello World!')
        return getattr(xp, self.func)(testing.shaped_arange((2, 3), xp, dtype).tolist())