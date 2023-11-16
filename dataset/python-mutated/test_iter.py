import unittest
import numpy
import pytest
import cupy
from cupy import testing

@testing.parameterize(*testing.product({'shape': [(3,), (2, 3, 4), (0,), (0, 2), (3, 0)]}))
class TestIter(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_list(self, xp, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = testing.shaped_arange(self.shape, xp, dtype)
        return list(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_len(self, xp, dtype):
        if False:
            print('Hello World!')
        x = testing.shaped_arange(self.shape, xp, dtype)
        return len(x)

class TestIterInvalid(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_iter(self, dtype):
        if False:
            return 10
        for xp in (numpy, cupy):
            x = testing.shaped_arange((), xp, dtype)
            with pytest.raises(TypeError):
                iter(x)

    @testing.for_all_dtypes()
    def test_len(self, dtype):
        if False:
            print('Hello World!')
        for xp in (numpy, cupy):
            x = testing.shaped_arange((), xp, dtype)
            with pytest.raises(TypeError):
                len(x)