import unittest
import numpy
import cupy
from cupy import testing
from cupy.lib import stride_tricks
import pytest

class TestAsStrided(unittest.TestCase):

    def test_as_strided(self):
        if False:
            print('Hello World!')
        a = cupy.array([1, 2, 3, 4])
        a_view = stride_tricks.as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
        expected = cupy.array([1, 3])
        testing.assert_array_equal(a_view, expected)
        a = cupy.array([1, 2, 3, 4])
        a_view = stride_tricks.as_strided(a, shape=(3, 4), strides=(0, 1 * a.itemsize))
        expected = cupy.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        testing.assert_array_equal(a_view, expected)

    @testing.numpy_cupy_array_equal()
    def test_rolling_window(self, xp):
        if False:
            for i in range(10):
                print('nop')
        a = testing.shaped_arange((3, 4), xp)
        a_rolling = rolling_window(a, 2, 0)
        return a_rolling

class TestSlidingWindowView(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_1d(self, xp):
        if False:
            print('Hello World!')
        arr = testing.shaped_arange((3, 4), xp)
        window_size = 2
        arr_view = xp.lib.stride_tricks.sliding_window_view(arr, window_size, 0)
        assert arr_view.strides == (16, 4, 16)
        return arr_view

    @testing.numpy_cupy_array_equal()
    def test_2d(self, xp):
        if False:
            return 10
        arr = testing.shaped_arange((3, 4), xp)
        window_shape = (2, 2)
        arr_view = xp.lib.stride_tricks.sliding_window_view(arr, window_shape=window_shape)
        assert arr_view.strides == (16, 4, 16, 4)
        return arr_view

    @testing.numpy_cupy_array_equal()
    def test_2d_with_axis(self, xp):
        if False:
            for i in range(10):
                print('nop')
        arr = testing.shaped_arange((3, 4), xp)
        window_shape = 3
        axis = 1
        arr_view = xp.lib.stride_tricks.sliding_window_view(arr, window_shape, axis)
        assert arr_view.strides == (16, 4, 4)
        return arr_view

    @testing.numpy_cupy_array_equal()
    def test_2d_multi_axis(self, xp):
        if False:
            return 10
        arr = testing.shaped_arange((3, 4), xp)
        window_shape = (2, 3)
        axis = (0, 1)
        arr_view = xp.lib.stride_tricks.sliding_window_view(arr, window_shape, axis)
        assert arr_view.strides == (16, 4, 16, 4)
        return arr_view

    def test_0d(self):
        if False:
            for i in range(10):
                print('nop')
        for xp in (numpy, cupy):
            arr = xp.array(42)
            window_size = 1
            with pytest.raises(ValueError, match='axis 0 is out of bounds'):
                xp.lib.stride_tricks.sliding_window_view(arr, window_size, 0)

    def test_window_shape_axis_length_mismatch(self):
        if False:
            i = 10
            return i + 15
        for xp in (numpy, cupy):
            x = xp.arange(24).reshape((2, 3, 4))
            window_shape = (2, 2)
            axis = None
            with pytest.raises(ValueError, match='Since axis is `None`'):
                xp.lib.stride_tricks.sliding_window_view(x, window_shape, axis)

    @testing.numpy_cupy_array_equal()
    def test_arraylike_input(self, xp):
        if False:
            i = 10
            return i + 15
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        arr_view = xp.lib.stride_tricks.sliding_window_view(x, 2)
        assert arr_view.strides == (8, 8)
        return arr_view

    def test_writeable_views_not_supported(self):
        if False:
            while True:
                i = 10
        x = cupy.arange(24).reshape((2, 3, 4))
        window_shape = (2, 2)
        axis = None
        writeable = True
        with self.assertRaises(NotImplementedError):
            stride_tricks.sliding_window_view(x, window_shape, axis, writeable=writeable)

def rolling_window(a, window, axis=-1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make an ndarray with a rolling window along axis.\n    This function is taken from https://github.com/numpy/numpy/pull/31\n    but slightly modified to accept axis option.\n    '
    a = numpy.swapaxes(a, axis, -1)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    if isinstance(a, numpy.ndarray):
        rolling = numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    elif isinstance(a, cupy.ndarray):
        rolling = stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolling.swapaxes(-2, axis)