import numpy as np
from skimage.util import unique_rows
from skimage._shared import testing
from skimage._shared.testing import assert_equal

def test_discontiguous_array():
    if False:
        print('Hello World!')
    ar = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    ar = ar[::2]
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[1, 0, 1]], np.uint8)
    assert_equal(ar_out, desired_ar_out)

def test_uint8_array():
    if False:
        return 10
    ar = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[0, 1, 0], [1, 0, 1]], np.uint8)
    assert_equal(ar_out, desired_ar_out)

def test_float_array():
    if False:
        for i in range(10):
            print('nop')
    ar = np.array([[1.1, 0.0, 1.1], [0.0, 1.1, 0.0], [1.1, 0.0, 1.1]], float)
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[0.0, 1.1, 0.0], [1.1, 0.0, 1.1]], float)
    assert_equal(ar_out, desired_ar_out)

def test_1d_array():
    if False:
        return 10
    ar = np.array([1, 0, 1, 1], np.uint8)
    with testing.raises(ValueError):
        unique_rows(ar)

def test_3d_array():
    if False:
        while True:
            i = 10
    ar = np.arange(8).reshape((2, 2, 2))
    with testing.raises(ValueError):
        unique_rows(ar)