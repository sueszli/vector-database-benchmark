import numpy
import modin.numpy as np
from .utils import assert_scalar_or_array_equal

def test_zeros_like():
    if False:
        while True:
            i = 10
    modin_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(np.zeros_like(modin_arr), numpy.zeros_like(numpy_arr))
    assert_scalar_or_array_equal(np.zeros_like(modin_arr, dtype=numpy.int8), numpy.zeros_like(numpy_arr, dtype=numpy.int8))
    assert_scalar_or_array_equal(np.zeros_like(modin_arr, shape=(10, 10)), numpy.zeros_like(numpy_arr, shape=(10, 10)))
    modin_arr = np.array([[1, 2], [3, 4]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(np.zeros_like(modin_arr), numpy.zeros_like(numpy_arr))

def test_ones_like():
    if False:
        while True:
            i = 10
    modin_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(np.ones_like(modin_arr), numpy.ones_like(numpy_arr))
    assert_scalar_or_array_equal(np.ones_like(modin_arr, dtype=numpy.int8), numpy.ones_like(numpy_arr, dtype=numpy.int8))
    assert_scalar_or_array_equal(np.ones_like(modin_arr, shape=(10, 10)), numpy.ones_like(numpy_arr, shape=(10, 10)))
    modin_arr = np.array([[1, 2], [3, 4]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(np.ones_like(modin_arr), numpy.ones_like(numpy_arr))