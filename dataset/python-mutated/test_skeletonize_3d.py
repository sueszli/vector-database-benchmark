import numpy as np
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch

def test_skeletonize_wrong_dim():
    if False:
        print('Hello World!')
    im = np.zeros(5, dtype=np.uint8)
    with testing.raises(ValueError):
        skeletonize(im, method='lee')
    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with testing.raises(ValueError):
        skeletonize(im, method='lee')

def test_skeletonize_1D_old_api():
    if False:
        return 10
    im = np.ones((5, 1), dtype=np.uint8)
    res = skeletonize_3d(im)
    assert_equal(res, im)

def test_skeletonize_1D():
    if False:
        for i in range(10):
            print('nop')
    im = np.ones((5, 1), dtype=np.uint8)
    res = skeletonize(im, method='lee')
    assert_equal(res, im)

def test_skeletonize_no_foreground():
    if False:
        for i in range(10):
            print('nop')
    im = np.zeros((5, 5), dtype=np.uint8)
    result = skeletonize(im, method='lee')
    assert_equal(result, im)

def test_skeletonize_all_foreground():
    if False:
        for i in range(10):
            print('nop')
    im = np.ones((3, 4), dtype=np.uint8)
    assert_equal(skeletonize(im, method='lee'), np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.uint8))

def test_skeletonize_single_point():
    if False:
        while True:
            i = 10
    im = np.zeros((5, 5), dtype=np.uint8)
    im[3, 3] = 1
    result = skeletonize(im, method='lee')
    assert_equal(result, im)

def test_skeletonize_already_thinned():
    if False:
        print('Hello World!')
    im = np.zeros((5, 5), dtype=np.uint8)
    im[3, 1:-1] = 1
    im[2, -1] = 1
    im[4, 0] = 1
    result = skeletonize(im, method='lee')
    assert_equal(result, im)

def test_dtype_conv():
    if False:
        for i in range(10):
            print('nop')
    img = np.random.random((16, 16))[::2, ::2]
    img[img < 0.5] = 0
    orig = img.copy()
    res = skeletonize(img, method='lee')
    img_max = img_as_ubyte(img).max()
    assert_equal(res.dtype, np.uint8)
    assert_equal(img, orig)
    assert_equal(res.max(), img_max)

@parametrize('img', [np.ones((8, 8), dtype=float), np.ones((4, 8, 8), dtype=float)])
def test_input_with_warning(img):
    if False:
        for i in range(10):
            print('nop')
    check_input(img)

@parametrize('img', [np.ones((8, 8), dtype=np.uint8), np.ones((4, 8, 8), dtype=np.uint8), np.ones((8, 8), dtype=bool), np.ones((4, 8, 8), dtype=bool)])
def test_input_without_warning(img):
    if False:
        i = 10
        return i + 15
    check_input(img)

def check_input(img):
    if False:
        return 10
    orig = img.copy()
    skeletonize(img, method='lee')
    assert_equal(img, orig)

def test_skeletonize_num_neighbors():
    if False:
        i = 10
        return i + 15
    image = np.zeros((300, 300))
    image[10:-10, 10:100] = 1
    image[-100:-10, 10:-10] = 1
    image[10:-10, -100:-10] = 1
    (rs, cs) = draw.line(250, 150, 10, 280)
    for i in range(10):
        image[rs + i, cs] = 1
    (rs, cs) = draw.line(10, 150, 250, 280)
    for i in range(20):
        image[rs + i, cs] = 1
    (ir, ic) = np.indices(image.shape)
    circle1 = (ic - 135) ** 2 + (ir - 150) ** 2 < 30 ** 2
    circle2 = (ic - 135) ** 2 + (ir - 150) ** 2 < 20 ** 2
    image[circle1] = 1
    image[circle2] = 0
    result = skeletonize(image, method='lee')
    mask = np.array([[1, 1], [1, 1]], np.uint8)
    blocks = ndi.correlate(result, mask, mode='constant')
    assert_(not np.any(blocks == 4))

def test_two_hole_image():
    if False:
        i = 10
        return i + 15
    img_o = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    img_f = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    res = skeletonize(img_o, method='lee')
    assert_equal(res, img_f)

def test_3d_vs_fiji():
    if False:
        for i in range(10):
            print('nop')
    img = binary_blobs(32, 0.05, n_dim=3, rng=1234)
    img = img[:-2, ...]
    img = img.astype(np.uint8) * 255
    img_s = skeletonize(img)
    img_f = io.imread(fetch('data/_blobs_3d_fiji_skeleton.tif'))
    assert_equal(img_s, img_f)