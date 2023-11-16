import pytest
import numpy as np
from numpy.testing import assert_array_equal

def test_matrix_transpose_raises_error_for_1d():
    if False:
        while True:
            i = 10
    msg = 'matrix transpose with ndim < 2 is undefined'
    arr = np.arange(48)
    with pytest.raises(ValueError, match=msg):
        arr.mT

def test_matrix_transpose_equals_transpose_2d():
    if False:
        while True:
            i = 10
    arr = np.arange(48).reshape((6, 8))
    assert_array_equal(arr.T, arr.mT)
ARRAY_SHAPES_TO_TEST = ((5, 2), (5, 2, 3), (5, 2, 3, 4))

@pytest.mark.parametrize('shape', ARRAY_SHAPES_TO_TEST)
def test_matrix_transpose_equals_swapaxes(shape):
    if False:
        print('Hello World!')
    num_of_axes = len(shape)
    vec = np.arange(shape[-1])
    arr = np.broadcast_to(vec, shape)
    tgt = np.swapaxes(arr, num_of_axes - 2, num_of_axes - 1)
    mT = arr.mT
    assert_array_equal(tgt, mT)