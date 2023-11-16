import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
eps = 1e-12

def test_empty_input():
    if False:
        while True:
            i = 10
    output = flood_fill(np.empty(0), (), 2)
    assert output.size == 0
    assert flood(np.empty(0), ()).dtype == bool
    assert flood(np.empty((20, 0, 4)), ()).shape == (20, 0, 4)

def test_float16():
    if False:
        while True:
            i = 10
    image = np.array([9.0, 0.1, 42], dtype=np.float16)
    with pytest.raises(TypeError, match='dtype of `image` is float16'):
        flood_fill(image, 0, 1)

def test_overrange_tolerance_int():
    if False:
        return 10
    image = np.arange(256, dtype=np.uint8).reshape((8, 8, 4))
    expected = np.zeros_like(image)
    output = flood_fill(image, (7, 7, 3), 0, tolerance=379)
    np.testing.assert_equal(output, expected)

def test_overrange_tolerance_float():
    if False:
        return 10
    max_value = np.finfo(np.float32).max
    image = np.random.uniform(size=(64, 64), low=-1.0, high=1.0).astype(np.float32)
    image *= max_value
    expected = np.ones_like(image)
    output = flood_fill(image, (0, 1), 1.0, tolerance=max_value * 10)
    np.testing.assert_equal(output, expected)

def test_inplace_int():
    if False:
        return 10
    image = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 2, 2, 0], [0, 1, 1, 0, 2, 2, 0], [1, 0, 0, 0, 0, 0, 3], [0, 1, 1, 1, 3, 3, 4]])
    flood_fill(image, (0, 0), 5, in_place=True)
    expected = np.array([[5, 5, 5, 5, 5, 5, 5], [5, 1, 1, 5, 2, 2, 5], [5, 1, 1, 5, 2, 2, 5], [1, 5, 5, 5, 5, 5, 3], [5, 1, 1, 1, 3, 3, 4]])
    np.testing.assert_array_equal(image, expected)

def test_inplace_float():
    if False:
        i = 10
        return i + 15
    image = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 2, 2, 0], [0, 1, 1, 0, 2, 2, 0], [1, 0, 0, 0, 0, 0, 3], [0, 1, 1, 1, 3, 3, 4]], dtype=np.float32)
    flood_fill(image, (0, 0), 5, in_place=True)
    expected = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0], [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0], [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0], [5.0, 1.0, 1.0, 1.0, 3.0, 3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(image, expected)

def test_inplace_noncontiguous():
    if False:
        for i in range(10):
            print('nop')
    image = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 2, 2, 0], [0, 1, 1, 0, 2, 2, 0], [1, 0, 0, 0, 0, 0, 3], [0, 1, 1, 1, 3, 3, 4]])
    image2 = image[::2, ::2]
    flood_fill(image2, (0, 0), 5, in_place=True)
    expected2 = np.array([[5, 5, 5, 5], [5, 1, 2, 5], [5, 1, 3, 4]])
    np.testing.assert_allclose(image2, expected2)
    expected = np.array([[5, 0, 5, 0, 5, 0, 5], [0, 1, 1, 0, 2, 2, 0], [5, 1, 1, 0, 2, 2, 5], [1, 0, 0, 0, 0, 0, 3], [5, 1, 1, 1, 3, 3, 4]])
    np.testing.assert_allclose(image, expected)

def test_1d():
    if False:
        i = 10
        return i + 15
    image = np.arange(11)
    expected = np.array([0, 1, -20, -20, -20, -20, -20, -20, -20, 9, 10])
    output = flood_fill(image, 5, -20, tolerance=3)
    output2 = flood_fill(image, (5,), -20, tolerance=3)
    np.testing.assert_equal(output, expected)
    np.testing.assert_equal(output, output2)

def test_wraparound():
    if False:
        return 10
    test = np.zeros((5, 7), dtype=np.float64)
    test[:, 3] = 100
    expected = np.array([[-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0]])
    np.testing.assert_equal(flood_fill(test, (0, 0), -1), expected)

def test_neighbors():
    if False:
        return 10
    test = np.zeros((5, 7), dtype=np.float64)
    test[:, 3] = 100
    expected = np.array([[0, 0, 0, 255, 0, 0, 0], [0, 0, 0, 255, 0, 0, 0], [0, 0, 0, 255, 0, 0, 0], [0, 0, 0, 255, 0, 0, 0], [0, 0, 0, 255, 0, 0, 0]])
    output = flood_fill(test, (0, 3), 255)
    np.testing.assert_equal(output, expected)
    test[2] = 100
    expected[2] = 255
    output2 = flood_fill(test, (2, 3), 255)
    np.testing.assert_equal(output2, expected)

def test_footprint():
    if False:
        for i in range(10):
            print('nop')
    footprint = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (3, 1), 255, footprint=footprint)
    expected = np.array([[0, 255, 255, 255, 255, 255], [0, 255, 255, 255, 255, 255], [0, 255, 255, 255, 255, 255], [0, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)
    footprint = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])
    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (1, 4), 255, footprint=footprint)
    expected = np.array([[0, 0, 0, 0, 0, 0], [255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 0]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)

def test_basic_nd():
    if False:
        for i in range(10):
            print('nop')
    for dimension in (3, 4, 5):
        shape = (5,) * dimension
        hypercube = np.zeros(shape)
        slice_mid = tuple((slice(1, -1, None) for dim in range(dimension)))
        hypercube[slice_mid] = 1
        filled = flood_fill(hypercube, (2,) * dimension, 2)
        assert filled.sum() == 3 ** dimension * 2
        np.testing.assert_equal(filled, np.pad(np.ones((3,) * dimension) * 2, 1, 'constant'))

@pytest.mark.parametrize('tolerance', [None, 0])
def test_f_order(tolerance):
    if False:
        for i in range(10):
            print('nop')
    image = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], order='F')
    expected = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    mask = flood(image, seed_point=(1, 0), tolerance=tolerance)
    np.testing.assert_array_equal(expected, mask)
    mask = flood(image, seed_point=(2, 1), tolerance=tolerance)
    np.testing.assert_array_equal(expected, mask)

def test_negative_indexing_seed_point():
    if False:
        return 10
    image = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 2, 2, 0], [0, 1, 1, 0, 2, 2, 0], [1, 0, 0, 0, 0, 0, 3], [0, 1, 1, 1, 3, 3, 4]], dtype=np.float32)
    expected = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0], [5.0, 1.0, 1.0, 5.0, 2.0, 2.0, 5.0], [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0], [5.0, 1.0, 1.0, 1.0, 3.0, 3.0, 4.0]], dtype=np.float32)
    image = flood_fill(image, (0, -1), 5)
    np.testing.assert_allclose(image, expected)

def test_non_adjacent_footprint():
    if False:
        return 10
    footprint = np.array([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]])
    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (2, 3), 255, footprint=footprint)
    expected = np.array([[0, 255, 0, 0, 0, 255], [0, 0, 0, 0, 0, 0], [0, 0, 0, 255, 0, 0], [0, 0, 0, 0, 0, 0], [0, 255, 0, 0, 0, 255]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)
    footprint = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    image = np.zeros((5, 10), dtype=np.uint8)
    image[:, (3, 7, 8)] = 100
    output = flood_fill(image, (0, 0), 255, footprint=footprint)
    expected = np.array([[255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0], [255, 255, 255, 100, 255, 255, 255, 100, 100, 0]], dtype=np.uint8)
    np.testing.assert_equal(output, expected)