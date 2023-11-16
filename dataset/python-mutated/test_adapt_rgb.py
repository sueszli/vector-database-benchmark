from functools import partial
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import color, data, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
COLOR_IMAGE = data.astronaut()[::5, ::6]
GRAY_IMAGE = data.camera()[::5, ::5]
SIGMA = 3
smooth = partial(filters.gaussian, sigma=SIGMA)
assert_allclose = partial(np.testing.assert_allclose, atol=1e-08)

@adapt_rgb(each_channel)
def edges_each(image):
    if False:
        i = 10
        return i + 15
    return filters.sobel(image)

@adapt_rgb(each_channel)
def smooth_each(image, sigma):
    if False:
        return 10
    return filters.gaussian(image, sigma)

@adapt_rgb(each_channel)
def mask_each(image, mask):
    if False:
        i = 10
        return i + 15
    result = image.copy()
    result[mask] = 0
    return result

@adapt_rgb(hsv_value)
def edges_hsv(image):
    if False:
        print('Hello World!')
    return filters.sobel(image)

@adapt_rgb(hsv_value)
def smooth_hsv(image, sigma):
    if False:
        for i in range(10):
            print('nop')
    return filters.gaussian(image, sigma)

@adapt_rgb(hsv_value)
def edges_hsv_uint(image):
    if False:
        while True:
            i = 10
    return img_as_uint(filters.sobel(image))

def test_gray_scale_image():
    if False:
        return 10
    assert_allclose(edges_each(GRAY_IMAGE), filters.sobel(GRAY_IMAGE))

def test_each_channel():
    if False:
        for i in range(10):
            print('nop')
    filtered = edges_each(COLOR_IMAGE)
    for (i, channel) in enumerate(np.rollaxis(filtered, axis=-1)):
        expected = img_as_float(filters.sobel(COLOR_IMAGE[:, :, i]))
        assert_allclose(channel, expected)

def test_each_channel_with_filter_argument():
    if False:
        i = 10
        return i + 15
    filtered = smooth_each(COLOR_IMAGE, SIGMA)
    for (i, channel) in enumerate(np.rollaxis(filtered, axis=-1)):
        assert_allclose(channel, smooth(COLOR_IMAGE[:, :, i]))

def test_each_channel_with_asymmetric_kernel():
    if False:
        while True:
            i = 10
    mask = np.triu(np.ones(COLOR_IMAGE.shape[:2], dtype=bool))
    mask_each(COLOR_IMAGE, mask)

def test_hsv_value():
    if False:
        for i in range(10):
            print('nop')
    filtered = edges_hsv(COLOR_IMAGE)
    value = color.rgb2hsv(COLOR_IMAGE)[:, :, 2]
    assert_allclose(color.rgb2hsv(filtered)[:, :, 2], filters.sobel(value))

def test_hsv_value_with_filter_argument():
    if False:
        return 10
    filtered = smooth_hsv(COLOR_IMAGE, SIGMA)
    value = color.rgb2hsv(COLOR_IMAGE)[:, :, 2]
    assert_allclose(color.rgb2hsv(filtered)[:, :, 2], smooth(value))

def test_hsv_value_with_non_float_output():
    if False:
        return 10
    filtered = edges_hsv_uint(COLOR_IMAGE)
    filtered_value = color.rgb2hsv(filtered)[:, :, 2]
    value = color.rgb2hsv(COLOR_IMAGE)[:, :, 2]
    assert_allclose(filtered_value, filters.sobel(value), rtol=1e-05, atol=1e-05)