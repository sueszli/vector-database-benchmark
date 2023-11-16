import numpy as np
from skimage.segmentation import clear_border
from skimage._shared.testing import assert_array_equal, assert_

def test_clear_border():
    if False:
        return 10
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    result = clear_border(image.copy())
    ref = image.copy()
    ref[1:3, 0:2] = 0
    ref[0:2, -2] = 0
    assert_array_equal(result, ref)
    result = clear_border(image.copy(), 1)
    assert_array_equal(result, np.zeros(result.shape))
    result = clear_border(image.copy(), buffer_size=1, bgval=2)
    assert_array_equal(result, 2 * np.ones_like(image))
    mask = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
    result = clear_border(image.copy(), mask=mask)
    ref = image.copy()
    ref[1:3, 0:2] = 0
    assert_array_equal(result, ref)

def test_clear_border_3d():
    if False:
        print('Hello World!')
    image = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    result = clear_border(image.copy())
    ref = image.copy()
    ref[0, 3, 0] = 0
    assert_array_equal(result, ref)
    result = clear_border(image.copy(), 1)
    assert_array_equal(result, np.zeros(result.shape))
    result = clear_border(image.copy(), buffer_size=1, bgval=2)
    assert_array_equal(result, 2 * np.ones_like(image))

def test_clear_border_non_binary():
    if False:
        return 10
    image = np.array([[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]])
    result = clear_border(image)
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 5, 4, 0], [0, 4, 5, 4, 0], [0, 0, 0, 0, 0]])
    assert_array_equal(result, expected)
    assert_(not np.all(image == result))

def test_clear_border_non_binary_3d():
    if False:
        i = 10
        return i + 15
    image3d = np.array([[[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]]])
    result = clear_border(image3d)
    expected = np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 5, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    assert_array_equal(result, expected)
    assert_(not np.all(image3d == result))

def test_clear_border_non_binary_inplace():
    if False:
        for i in range(10):
            print('nop')
    image = np.array([[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]])
    result = clear_border(image, out=image)
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 5, 4, 0], [0, 4, 5, 4, 0], [0, 0, 0, 0, 0]])
    assert_array_equal(result, expected)
    assert_array_equal(image, result)

def test_clear_border_non_binary_inplace_3d():
    if False:
        while True:
            i = 10
    image3d = np.array([[[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]]])
    result = clear_border(image3d, out=image3d)
    expected = np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 5, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    assert_array_equal(result, expected)
    assert_array_equal(image3d, result)

def test_clear_border_non_binary_out():
    if False:
        return 10
    image = np.array([[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]])
    out = np.empty_like(image)
    result = clear_border(image, out=out)
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 5, 4, 0], [0, 4, 5, 4, 0], [0, 0, 0, 0, 0]])
    assert_array_equal(result, expected)
    assert_array_equal(out, result)

def test_clear_border_non_binary_out_3d():
    if False:
        for i in range(10):
            print('nop')
    image3d = np.array([[[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 5, 4, 2], [3, 4, 5, 4, 2], [3, 3, 2, 1, 2]], [[1, 2, 3, 1, 2], [3, 3, 3, 4, 2], [3, 4, 3, 4, 2], [3, 3, 2, 1, 2]]])
    out = np.empty_like(image3d)
    result = clear_border(image3d, out=out)
    expected = np.array([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 5, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    assert_array_equal(result, expected)
    assert_array_equal(out, result)