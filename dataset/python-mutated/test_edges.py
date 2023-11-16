import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_roberts_zeros(dtype):
    if False:
        i = 10
        return i + 15
    "Roberts' filter on an array of all zeros."
    result = filters.roberts(np.zeros((10, 10), dtype=dtype), np.ones((10, 10), bool))
    assert result.dtype == _supported_float_type(dtype)
    assert np.all(result == 0)

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_roberts_diagonal1(dtype):
    if False:
        while True:
            i = 10
    "Roberts' filter on a diagonal edge should be a diagonal line."
    image = np.tri(10, 10, 0, dtype=dtype)
    expected = ~(np.tri(10, 10, -1).astype(bool) | np.tri(10, 10, -2).astype(bool).transpose())
    expected[-1, -1] = 0
    result = filters.roberts(image)
    assert result.dtype == _supported_float_type(dtype)
    assert_array_almost_equal(result.astype(bool), expected)

@pytest.mark.parametrize('function_name', ['farid', 'laplace', 'prewitt', 'roberts', 'scharr', 'sobel'])
def test_int_rescaling(function_name):
    if False:
        while True:
            i = 10
    'Basic test that uint8 inputs get rescaled from [0, 255] to [0, 1.]\n\n    The output of any of these filters should be within roughly a factor of\n    two of the input range. For integer inputs, rescaling to floats in\n    [0.0, 1.0] should occur, so just verify outputs are not > 2.0.\n    '
    img = data.coins()[:128, :128]
    func = getattr(filters, function_name)
    filtered = func(img)
    assert filtered.max() <= 2.0

def test_roberts_diagonal2():
    if False:
        i = 10
        return i + 15
    "Roberts' filter on a diagonal edge should be a diagonal line."
    image = np.rot90(np.tri(10, 10, 0), 3)
    expected = ~np.rot90(np.tri(10, 10, -1).astype(bool) | np.tri(10, 10, -2).astype(bool).transpose())
    expected = _mask_filter_result(expected, None)
    result = filters.roberts(image).astype(bool)
    assert_array_almost_equal(result, expected)

def test_sobel_zeros():
    if False:
        while True:
            i = 10
    'Sobel on an array of all zeros.'
    result = filters.sobel(np.zeros((10, 10)), np.ones((10, 10), bool))
    assert np.all(result == 0)

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_sobel_mask(dtype):
    if False:
        while True:
            i = 10
    'Sobel on a masked array should be zero.'
    result = filters.sobel(np.random.uniform(size=(10, 10)).astype(dtype), np.zeros((10, 10), dtype=bool))
    assert result.dtype == _supported_float_type(dtype)
    assert np.all(result == 0)

def test_sobel_horizontal():
    if False:
        i = 10
        return i + 15
    'Sobel on a horizontal edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.sobel(image) * np.sqrt(2)
    assert_allclose(result[i == 0], 1)
    assert np.all(result[np.abs(i) > 1] == 0)

def test_sobel_vertical():
    if False:
        print('Hello World!')
    'Sobel on a vertical edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.sobel(image) * np.sqrt(2)
    assert np.all(result[j == 0] == 1)
    assert np.all(result[np.abs(j) > 1] == 0)

def test_sobel_h_zeros():
    if False:
        i = 10
        return i + 15
    'Horizontal sobel on an array of all zeros.'
    result = filters.sobel_h(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert np.all(result == 0)

def test_sobel_h_mask():
    if False:
        print('Hello World!')
    'Horizontal Sobel on a masked array should be zero.'
    result = filters.sobel_h(np.random.uniform(size=(10, 10)), np.zeros((10, 10), dtype=bool))
    assert np.all(result == 0)

def test_sobel_h_horizontal():
    if False:
        i = 10
        return i + 15
    'Horizontal Sobel on an edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.sobel_h(image)
    assert np.all(result[i == 0] == 1)
    assert np.all(result[np.abs(i) > 1] == 0)

def test_sobel_h_vertical():
    if False:
        return 10
    'Horizontal Sobel on a vertical edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float) * np.sqrt(2)
    result = filters.sobel_h(image)
    assert_allclose(result, 0, atol=1e-10)

def test_sobel_v_zeros():
    if False:
        while True:
            i = 10
    'Vertical sobel on an array of all zeros.'
    result = filters.sobel_v(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_sobel_v_mask():
    if False:
        print('Hello World!')
    'Vertical Sobel on a masked array should be zero.'
    result = filters.sobel_v(np.random.uniform(size=(10, 10)), np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_sobel_v_vertical():
    if False:
        while True:
            i = 10
    'Vertical Sobel on an edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.sobel_v(image)
    assert np.all(result[j == 0] == 1)
    assert np.all(result[np.abs(j) > 1] == 0)

def test_sobel_v_horizontal():
    if False:
        while True:
            i = 10
    'vertical Sobel on a horizontal edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.sobel_v(image)
    assert_allclose(result, 0)

def test_scharr_zeros():
    if False:
        i = 10
        return i + 15
    'Scharr on an array of all zeros.'
    result = filters.scharr(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert np.all(result < 1e-16)

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_scharr_mask(dtype):
    if False:
        print('Hello World!')
    'Scharr on a masked array should be zero.'
    result = filters.scharr(np.random.uniform(size=(10, 10)).astype(dtype), np.zeros((10, 10), dtype=bool))
    assert result.dtype == _supported_float_type(dtype)
    assert_allclose(result, 0)

def test_scharr_horizontal():
    if False:
        return 10
    'Scharr on an edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.scharr(image) * np.sqrt(2)
    assert_allclose(result[i == 0], 1)
    assert np.all(result[np.abs(i) > 1] == 0)

def test_scharr_vertical():
    if False:
        return 10
    'Scharr on a vertical edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.scharr(image) * np.sqrt(2)
    assert_allclose(result[j == 0], 1)
    assert np.all(result[np.abs(j) > 1] == 0)

def test_scharr_h_zeros():
    if False:
        while True:
            i = 10
    'Horizontal Scharr on an array of all zeros.'
    result = filters.scharr_h(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_scharr_h_mask():
    if False:
        return 10
    'Horizontal Scharr on a masked array should be zero.'
    result = filters.scharr_h(np.random.uniform(size=(10, 10)), np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_scharr_h_horizontal():
    if False:
        i = 10
        return i + 15
    'Horizontal Scharr on an edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.scharr_h(image)
    assert np.all(result[i == 0] == 1)
    assert np.all(result[np.abs(i) > 1] == 0)

def test_scharr_h_vertical():
    if False:
        print('Hello World!')
    'Horizontal Scharr on a vertical edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.scharr_h(image)
    assert_allclose(result, 0)

def test_scharr_v_zeros():
    if False:
        print('Hello World!')
    'Vertical Scharr on an array of all zeros.'
    result = filters.scharr_v(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_scharr_v_mask():
    if False:
        while True:
            i = 10
    'Vertical Scharr on a masked array should be zero.'
    result = filters.scharr_v(np.random.uniform(size=(10, 10)), np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_scharr_v_vertical():
    if False:
        for i in range(10):
            print('nop')
    'Vertical Scharr on an edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.scharr_v(image)
    assert np.all(result[j == 0] == 1)
    assert np.all(result[np.abs(j) > 1] == 0)

def test_scharr_v_horizontal():
    if False:
        for i in range(10):
            print('nop')
    'vertical Scharr on a horizontal edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.scharr_v(image)
    assert_allclose(result, 0)

def test_prewitt_zeros():
    if False:
        while True:
            i = 10
    'Prewitt on an array of all zeros.'
    result = filters.prewitt(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_prewitt_mask(dtype):
    if False:
        return 10
    'Prewitt on a masked array should be zero.'
    result = filters.prewitt(np.random.uniform(size=(10, 10)).astype(dtype), np.zeros((10, 10), dtype=bool))
    assert result.dtype == _supported_float_type(dtype)
    assert_allclose(np.abs(result), 0)

def test_prewitt_horizontal():
    if False:
        for i in range(10):
            print('nop')
    'Prewitt on an edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.prewitt(image) * np.sqrt(2)
    assert np.all(result[i == 0] == 1)
    assert_allclose(result[np.abs(i) > 1], 0, atol=1e-10)

def test_prewitt_vertical():
    if False:
        return 10
    'Prewitt on a vertical edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.prewitt(image) * np.sqrt(2)
    assert_allclose(result[j == 0], 1)
    assert_allclose(result[np.abs(j) > 1], 0, atol=1e-10)

def test_prewitt_h_zeros():
    if False:
        i = 10
        return i + 15
    'Horizontal prewitt on an array of all zeros.'
    result = filters.prewitt_h(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_prewitt_h_mask():
    if False:
        return 10
    'Horizontal prewitt on a masked array should be zero.'
    result = filters.prewitt_h(np.random.uniform(size=(10, 10)), np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_prewitt_h_horizontal():
    if False:
        for i in range(10):
            print('nop')
    'Horizontal prewitt on an edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.prewitt_h(image)
    assert np.all(result[i == 0] == 1)
    assert_allclose(result[np.abs(i) > 1], 0, atol=1e-10)

def test_prewitt_h_vertical():
    if False:
        print('Hello World!')
    'Horizontal prewitt on a vertical edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.prewitt_h(image)
    assert_allclose(result, 0, atol=1e-10)

def test_prewitt_v_zeros():
    if False:
        while True:
            i = 10
    'Vertical prewitt on an array of all zeros.'
    result = filters.prewitt_v(np.zeros((10, 10)), np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_prewitt_v_mask():
    if False:
        for i in range(10):
            print('nop')
    'Vertical prewitt on a masked array should be zero.'
    result = filters.prewitt_v(np.random.uniform(size=(10, 10)), np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_prewitt_v_vertical():
    if False:
        while True:
            i = 10
    'Vertical prewitt on an edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.prewitt_v(image)
    assert np.all(result[j == 0] == 1)
    assert_allclose(result[np.abs(j) > 1], 0, atol=1e-10)

def test_prewitt_v_horizontal():
    if False:
        return 10
    'Vertical prewitt on a horizontal edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.prewitt_v(image)
    assert_allclose(result, 0)

def test_laplace_zeros():
    if False:
        i = 10
        return i + 15
    'Laplace on a square image.'
    image = np.zeros((9, 9))
    image[3:-3, 3:-3] = 1
    result = filters.laplace(image)
    check_result = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.0, 1.0, 2.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.0, 1.0, 2.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert_allclose(result, check_result)

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_laplace_mask(dtype):
    if False:
        return 10
    'Laplace on a masked array should be zero.'
    image = np.zeros((9, 9), dtype=dtype)
    image[3:-3, 3:-3] = 1
    result = filters.laplace(image, ksize=3, mask=np.zeros((9, 9), dtype=bool))
    assert result.dtype == _supported_float_type(dtype)
    assert np.all(result == 0)

def test_farid_zeros():
    if False:
        print('Hello World!')
    'Farid on an array of all zeros.'
    result = filters.farid(np.zeros((10, 10)), mask=np.ones((10, 10), dtype=bool))
    assert np.all(result == 0)

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_farid_mask(dtype):
    if False:
        return 10
    'Farid on a masked array should be zero.'
    result = filters.farid(np.random.uniform(size=(10, 10)).astype(dtype), mask=np.zeros((10, 10), dtype=bool))
    assert result.dtype == _supported_float_type(dtype)
    assert np.all(result == 0)

def test_farid_horizontal():
    if False:
        print('Hello World!')
    'Farid on a horizontal edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.farid(image) * np.sqrt(2)
    assert np.all(result[i == 0] == result[i == 0][0])
    assert_allclose(result[np.abs(i) > 2], 0, atol=1e-10)

def test_farid_vertical():
    if False:
        print('Hello World!')
    'Farid on a vertical edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.farid(image) * np.sqrt(2)
    assert np.all(result[j == 0] == result[j == 0][0])
    assert_allclose(result[np.abs(j) > 2], 0, atol=1e-10)

def test_farid_h_zeros():
    if False:
        while True:
            i = 10
    'Horizontal Farid on an array of all zeros.'
    result = filters.farid_h(np.zeros((10, 10)), mask=np.ones((10, 10), dtype=bool))
    assert np.all(result == 0)

def test_farid_h_mask():
    if False:
        i = 10
        return i + 15
    'Horizontal Farid on a masked array should be zero.'
    result = filters.farid_h(np.random.uniform(size=(10, 10)), mask=np.zeros((10, 10), dtype=bool))
    assert np.all(result == 0)

def test_farid_h_horizontal():
    if False:
        i = 10
        return i + 15
    'Horizontal Farid on an edge should be a horizontal line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.farid_h(image)
    assert np.all(result[i == 0] == result[i == 0][0])
    assert_allclose(result[np.abs(i) > 2], 0, atol=1e-10)

def test_farid_h_vertical():
    if False:
        while True:
            i = 10
    'Horizontal Farid on a vertical edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float) * np.sqrt(2)
    result = filters.farid_h(image)
    assert_allclose(result, 0, atol=1e-10)

def test_farid_v_zeros():
    if False:
        for i in range(10):
            print('nop')
    'Vertical Farid on an array of all zeros.'
    result = filters.farid_v(np.zeros((10, 10)), mask=np.ones((10, 10), dtype=bool))
    assert_allclose(result, 0, atol=1e-10)

def test_farid_v_mask():
    if False:
        for i in range(10):
            print('nop')
    'Vertical Farid on a masked array should be zero.'
    result = filters.farid_v(np.random.uniform(size=(10, 10)), mask=np.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)

def test_farid_v_vertical():
    if False:
        return 10
    'Vertical Farid on an edge should be a vertical line.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.farid_v(image)
    assert np.all(result[j == 0] == result[j == 0][0])
    assert_allclose(result[np.abs(j) > 2], 0, atol=1e-10)

def test_farid_v_horizontal():
    if False:
        return 10
    'vertical Farid on a horizontal edge should be zero.'
    (i, j) = np.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.farid_v(image)
    assert_allclose(result, 0, atol=1e-10)

@pytest.mark.parametrize('grad_func', (filters.prewitt_h, filters.sobel_h, filters.scharr_h))
def test_horizontal_mask_line(grad_func):
    if False:
        for i in range(10):
            print('nop')
    'Horizontal edge filters mask pixels surrounding input mask.'
    (vgrad, _) = np.mgrid[:1:11j, :1:11j]
    vgrad[5, :] = 1
    mask = np.ones_like(vgrad)
    mask[5, :] = 0
    expected = np.zeros_like(vgrad)
    expected[1:-1, 1:-1] = 0.2
    expected[4:7, 1:-1] = 0
    result = grad_func(vgrad, mask)
    assert_allclose(result, expected)

@pytest.mark.parametrize('grad_func', (filters.prewitt_v, filters.sobel_v, filters.scharr_v))
def test_vertical_mask_line(grad_func):
    if False:
        for i in range(10):
            print('nop')
    'Vertical edge filters mask pixels surrounding input mask.'
    (_, hgrad) = np.mgrid[:1:11j, :1:11j]
    hgrad[:, 5] = 1
    mask = np.ones_like(hgrad)
    mask[:, 5] = 0
    expected = np.zeros_like(hgrad)
    expected[1:-1, 1:-1] = 0.2
    expected[1:-1, 4:7] = 0
    result = grad_func(hgrad, mask)
    assert_allclose(result, expected)
MAX_SOBEL_0 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=float)
MAX_SOBEL_ND = np.array([[[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 1, 0], [1, 1, 0]], [[1, 1, 0], [1, 1, 0], [1, 1, 0]]], dtype=float)
MAX_SCHARR_ND = np.array([[[0, 0, 0], [0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1], [1, 1, 1]]], dtype=float)
MAX_FARID_0 = np.zeros((5, 5, 5), dtype=float)
MAX_FARID_0[2:, :, :] = 1
MAX_FARID_ND = np.array([[[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]], dtype=float)

@pytest.mark.parametrize(('func', 'max_edge'), [(filters.prewitt, MAX_SOBEL_ND), (filters.sobel, MAX_SOBEL_ND), (filters.scharr, MAX_SCHARR_ND), (filters.farid, MAX_FARID_ND)])
def test_3d_edge_filters(func, max_edge):
    if False:
        while True:
            i = 10
    blobs = data.binary_blobs(length=128, n_dim=3, rng=5)
    edges = func(blobs)
    center = max_edge.shape[0] // 2
    if center == 2:
        rtol = 0.001
    else:
        rtol = 1e-07
    assert_allclose(np.max(edges), func(max_edge)[center, center, center], rtol=rtol)

@pytest.mark.parametrize(('func', 'max_edge'), [(filters.prewitt, MAX_SOBEL_0), (filters.sobel, MAX_SOBEL_0), (filters.scharr, MAX_SOBEL_0), (filters.farid, MAX_FARID_0)])
def test_3d_edge_filters_single_axis(func, max_edge):
    if False:
        i = 10
        return i + 15
    blobs = data.binary_blobs(length=128, n_dim=3, rng=5)
    edges0 = func(blobs, axis=0)
    center = max_edge.shape[0] // 2
    if center == 2:
        rtol = 0.001
    else:
        rtol = 1e-07
    assert_allclose(np.max(edges0), func(max_edge, axis=0)[center, center, center], rtol=rtol)

@pytest.mark.parametrize('detector', [filters.sobel, filters.scharr, filters.prewitt, filters.roberts, filters.farid])
def test_range(detector):
    if False:
        i = 10
        return i + 15
    'Output of edge detection should be in [0, 1]'
    image = np.random.random((100, 100))
    out = detector(image)
    assert_(out.min() >= 0, f'Minimum of `{detector.__name__}` is smaller than 0.')
    assert_(out.max() <= 1, f'Maximum of `{detector.__name__}` is larger than 1.')