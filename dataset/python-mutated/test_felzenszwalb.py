import numpy as np
from skimage import data
from skimage.segmentation import felzenszwalb
from skimage._shared import testing
from skimage._shared.testing import assert_greater, run_in_parallel, assert_equal, assert_array_equal, assert_warns, assert_no_warnings

@run_in_parallel()
def test_grey():
    if False:
        i = 10
        return i + 15
    img = np.zeros((20, 21))
    img[:10, 10:] = 0.2
    img[10:, :10] = 0.4
    img[10:, 10:] = 0.6
    seg = felzenszwalb(img, sigma=0)
    assert_equal(len(np.unique(seg)), 4)
    for i in range(4):
        hist = np.histogram(img[seg == i], bins=[0, 0.1, 0.3, 0.5, 1])[0]
        assert_greater(hist[i], 40)

def test_minsize():
    if False:
        return 10
    img = data.coins()[20:168, 0:128]
    for min_size in np.arange(10, 100, 10):
        segments = felzenszwalb(img, min_size=min_size, sigma=3)
        counts = np.bincount(segments.ravel())
        assert_greater(counts.min() + 1, min_size)
    coffee = data.coffee()[::4, ::4]
    for min_size in np.arange(10, 100, 10):
        segments = felzenszwalb(coffee, min_size=min_size, sigma=3)
        counts = np.bincount(segments.ravel())
        assert_greater(counts.min() + 1, min_size)

@testing.parametrize('channel_axis', [0, -1])
def test_3D(channel_axis):
    if False:
        return 10
    grey_img = np.zeros((10, 10))
    rgb_img = np.zeros((10, 10, 3))
    three_d_img = np.zeros((10, 10, 10))
    rgb_img = np.moveaxis(rgb_img, -1, channel_axis)
    with assert_no_warnings():
        felzenszwalb(grey_img, channel_axis=-1)
        felzenszwalb(grey_img, channel_axis=None)
        felzenszwalb(rgb_img, channel_axis=channel_axis)
    with assert_warns(RuntimeWarning):
        felzenszwalb(three_d_img, channel_axis=channel_axis)
    with testing.raises(ValueError):
        felzenszwalb(rgb_img, channel_axis=None)
        felzenszwalb(three_d_img, channel_axis=None)

def test_color():
    if False:
        return 10
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    seg = felzenszwalb(img, sigma=0)
    assert_equal(len(np.unique(seg)), 4)
    assert_array_equal(seg[:10, :10], 0)
    assert_array_equal(seg[10:, :10], 2)
    assert_array_equal(seg[:10, 10:], 1)
    assert_array_equal(seg[10:, 10:], 3)

def test_merging():
    if False:
        return 10
    img = np.array([[0, 0.3], [0.7, 1]])
    seg = felzenszwalb(img, scale=0, sigma=0, min_size=2)
    assert_equal(len(np.unique(seg)), 2)
    assert_array_equal(seg[0, :], 0)
    assert_array_equal(seg[1, :], 1)