import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float

class TestCanny(unittest.TestCase):

    def test_00_00_zeros(self):
        if False:
            return 10
        'Test that the Canny filter finds no points for a blank field'
        result = feature.canny(np.zeros((20, 20)), 4, 0, 0, np.ones((20, 20), bool))
        self.assertFalse(np.any(result))

    def test_00_01_zeros_mask(self):
        if False:
            print('Hello World!')
        'Test that the Canny filter finds no points in a masked image'
        result = feature.canny(np.random.uniform(size=(20, 20)), 4, 0, 0, np.zeros((20, 20), bool))
        self.assertFalse(np.any(result))

    def test_01_01_circle(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the Canny filter finds the outlines of a circle'
        (i, j) = np.mgrid[-200:200, -200:200].astype(float) / 200
        c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
        result = feature.canny(c.astype(float), 4, 0, 0, np.ones(c.shape, bool))
        cd = binary_dilation(c, iterations=3)
        ce = binary_erosion(c, iterations=3)
        cde = np.logical_and(cd, np.logical_not(ce))
        self.assertTrue(np.all(cde[result]))
        point_count = np.sum(result)
        self.assertTrue(point_count > 1200)
        self.assertTrue(point_count < 1600)

    def test_01_02_circle_with_noise(self):
        if False:
            i = 10
            return i + 15
        'Test that the Canny filter finds the circle outlines\n        in a noisy image'
        np.random.seed(0)
        (i, j) = np.mgrid[-200:200, -200:200].astype(float) / 200
        c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
        cf = c.astype(float) * 0.5 + np.random.uniform(size=c.shape) * 0.5
        result = feature.canny(cf, 4, 0.1, 0.2, np.ones(c.shape, bool))
        cd = binary_dilation(c, iterations=4)
        ce = binary_erosion(c, iterations=4)
        cde = np.logical_and(cd, np.logical_not(ce))
        self.assertTrue(np.all(cde[result]))
        point_count = np.sum(result)
        self.assertTrue(point_count > 1200)
        self.assertTrue(point_count < 1600)

    def test_image_shape(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, feature.canny, np.zeros((20, 20, 20)), 4, 0, 0)

    def test_mask_none(self):
        if False:
            return 10
        result1 = feature.canny(np.zeros((20, 20)), 4, 0, 0, np.ones((20, 20), bool))
        result2 = feature.canny(np.zeros((20, 20)), 4, 0, 0)
        self.assertTrue(np.all(result1 == result2))

    def test_use_quantiles(self):
        if False:
            for i in range(10):
                print('nop')
        image = img_as_float(data.camera()[::100, ::100])
        correct_output = np.array([[False, False, False, False, False, False], [False, True, True, True, False, False], [False, False, False, True, False, False], [False, False, False, True, False, False], [False, False, True, True, False, False], [False, False, False, False, False, False]])
        result = feature.canny(image, low_threshold=0.6, high_threshold=0.8, use_quantiles=True)
        assert_equal(result, correct_output)

    def test_img_all_ones(self):
        if False:
            i = 10
            return i + 15
        image = np.ones((10, 10))
        assert np.all(feature.canny(image) == 0)

    def test_invalid_use_quantiles(self):
        if False:
            for i in range(10):
                print('nop')
        image = img_as_float(data.camera()[::50, ::50])
        self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=0.5, high_threshold=3.6)
        self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=-5, high_threshold=0.5)
        self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=99, high_threshold=0.9)
        self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=0.5, high_threshold=-100)
        image = data.camera()
        self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=50, high_threshold=150)

    def test_dtype(self):
        if False:
            print('Hello World!')
        'Check that the same output is produced regardless of image dtype.'
        image_uint8 = data.camera()
        image_float = img_as_float(image_uint8)
        result_uint8 = feature.canny(image_uint8)
        result_float = feature.canny(image_float)
        assert_equal(result_uint8, result_float)
        low = 0.1
        high = 0.2
        assert_equal(feature.canny(image_float, 1.0, low, high), feature.canny(image_uint8, 1.0, 255 * low, 255 * high))

    def test_full_mask_matches_no_mask(self):
        if False:
            while True:
                i = 10
        'The masked and unmasked algorithms should return the same result.'
        image = data.camera()
        for mode in ('constant', 'nearest', 'reflect'):
            assert_equal(feature.canny(image, mode=mode), feature.canny(image, mode=mode, mask=np.ones_like(image, dtype=bool)))

    def test_unsupported_int64(self):
        if False:
            print('Hello World!')
        for dtype in (np.int64, np.uint64):
            image = np.zeros((10, 10), dtype=dtype)
            image[3, 3] = np.iinfo(dtype).max
            with pytest.raises(ValueError, match='64-bit integer images are not supported'):
                feature.canny(image)