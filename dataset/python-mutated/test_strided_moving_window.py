from itertools import product
import numpy as np
import pytest
from darts.utils.data.tabularization import strided_moving_window

class TestStridedMovingWindow:
    """
    Tests `strided_moving_window` function defined in `darts.utils.data.tabularization`.
    """

    def test_strided_moving_windows_extracted_windows(self):
        if False:
            while True:
                i = 10
        '\n        Tests that each of the windows extracted by `strided_moving_windows`\n        is correct over a number of input parameter combinations.\n\n        This is achieved by looping over each extracted window, and checking that the\n        `i`th window corresponds to the the next `window_len` values found after\n        `i * stride` (i.e. the index position at which the `i`th window should begin).\n        '
        window_len_combos = (1, 2, 5)
        axis_combos = (0, 1, 2)
        stride_combos = (1, 2, 3)
        x_shape = (10, 8, 12)
        x = np.arange(np.prod(x_shape)).reshape(*x_shape)
        for (axis, stride, window_len) in product(axis_combos, stride_combos, window_len_combos):
            windows = strided_moving_window(x, window_len, stride, axis)
            for i in range(windows.shape[axis]):
                window = np.moveaxis(windows, axis, -1)[:, :, :, i]
                window_start_idx = i * stride
                expected = np.moveaxis(x, axis, -1)[:, :, window_start_idx:window_start_idx + window_len]
                assert np.allclose(window, expected)

    def test_strided_moving_window_invalid_stride_error(self):
        if False:
            print('Hello World!')
        '\n        Checks that appropriate error is thrown when `stride` is set to\n        a non-positive number and/or a non-`int` value.\n        '
        x = np.arange(1)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=0)
        assert '`stride` must be a positive `int`.' == str(err.value)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=1.1)
        assert '`stride` must be a positive `int`.' == str(err.value)

    def test_strided_moving_window_negative_window_len_error(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks that appropriate error is thrown when `wendow_len`\n        is set to a non-positive number and/or a non-`int` value.\n        '
        x = np.arange(1)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=0, stride=1)
        assert '`window_len` must be a positive `int`.' == str(err.value)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1.1, stride=1)
        assert '`window_len` must be a positive `int`.' == str(err.value)

    def test_strided_moving_window_pass_invalid_axis_error(self):
        if False:
            i = 10
            return i + 15
        '\n        Checks that appropriate error is thrown when `axis`\n        is set to a non-`int` value, or a value not less than\n        `x.ndim`.\n        '
        x = np.arange(1)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=1, axis=0.1)
        assert '`axis` must be an `int` that is less than `x.ndim`.' == str(err.value)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=1, axis=1)
        assert '`axis` must be an `int` that is less than `x.ndim`.' == str(err.value)

    def test_strided_moving_window_window_len_too_large_error(self):
        if False:
            print('Hello World!')
        '\n        Checks that appropriate error is thrown when `window_len`\n        is set to a value larger than `x.shape[axis]`.\n        '
        x = np.arange(1)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=2, stride=1)
        assert '`window_len` must be less than or equal to x.shape[axis].' == str(err.value)