import unittest
import numpy
import pytest
import cupy
from cupy import testing

@testing.parameterize({'repeats': 0, 'axis': None}, {'repeats': 2, 'axis': None}, {'repeats': 2, 'axis': 1}, {'repeats': 2, 'axis': -1}, {'repeats': [0, 0, 0], 'axis': 1}, {'repeats': [1, 2, 3], 'axis': 1}, {'repeats': [1, 2, 3], 'axis': -2})
class TestRepeat(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        if False:
            print('Hello World!')
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)

class TestRepeatRepeatsNdarray(unittest.TestCase):

    def test_func(self):
        if False:
            return 10
        a = testing.shaped_arange((2, 3, 4), cupy)
        repeats = cupy.array([2, 3], dtype=cupy.int32)
        with pytest.raises(ValueError, match='repeats'):
            cupy.repeat(a, repeats)

    def test_method(self):
        if False:
            return 10
        a = testing.shaped_arange((2, 3, 4), cupy)
        repeats = cupy.array([2, 3], dtype=cupy.int32)
        with pytest.raises(ValueError, match='repeats'):
            a.repeat(repeats)

@testing.parameterize({'repeats': [2], 'axis': None}, {'repeats': [2], 'axis': 1})
class TestRepeatListBroadcast(unittest.TestCase):
    """Test for `repeats` argument using single element list.

    This feature is only supported in NumPy 1.10 or later.
    """

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        if False:
            while True:
                i = 10
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)

@testing.parameterize({'repeats': 0, 'axis': None}, {'repeats': 2, 'axis': None}, {'repeats': 2, 'axis': 0}, {'repeats': [1, 2, 3, 4], 'axis': None}, {'repeats': [1, 2, 3, 4], 'axis': 0})
class TestRepeat1D(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        if False:
            i = 10
            return i + 15
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)

@testing.parameterize({'repeats': [2], 'axis': None}, {'repeats': [2], 'axis': 0})
class TestRepeat1DListBroadcast(unittest.TestCase):
    """See comment in TestRepeatListBroadcast class."""

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        if False:
            print('Hello World!')
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)

@testing.parameterize({'repeats': -3, 'axis': None}, {'repeats': [-3, -3], 'axis': 0}, {'repeats': [1, 2, 3], 'axis': None}, {'repeats': [1, 2], 'axis': 1}, {'repeats': 2, 'axis': -4}, {'repeats': 2, 'axis': 3})
class TestRepeatFailure(unittest.TestCase):

    def test_repeat_failure(self):
        if False:
            while True:
                i = 10
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.repeat(x, self.repeats, self.axis)

@testing.parameterize({'reps': 0}, {'reps': 1}, {'reps': 2}, {'reps': (0, 1)}, {'reps': (2, 3)}, {'reps': (2, 3, 4, 5)})
class TestTile(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_tile(self, xp):
        if False:
            return 10
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.tile(x, self.reps)

@testing.parameterize({'reps': -1}, {'reps': (-1, -2)})
class TestTileFailure(unittest.TestCase):

    def test_tile_failure(self):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.tile(x, -3)