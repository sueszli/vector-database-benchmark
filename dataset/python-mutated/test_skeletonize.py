import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import expected_warnings, fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts

class TestSkeletonize:

    def test_skeletonize_no_foreground(self):
        if False:
            while True:
                i = 10
        im = np.zeros((5, 5))
        result = skeletonize(im)
        assert_array_equal(result, np.zeros((5, 5)))

    def test_skeletonize_wrong_dim1(self):
        if False:
            i = 10
            return i + 15
        im = np.zeros(5)
        with pytest.raises(ValueError):
            skeletonize(im)

    def test_skeletonize_wrong_dim2(self):
        if False:
            print('Hello World!')
        im = np.zeros((5, 5, 5))
        with pytest.raises(ValueError):
            skeletonize(im, method='zhang')

    def test_skeletonize_wrong_method(self):
        if False:
            return 10
        im = np.ones((5, 5))
        with pytest.raises(ValueError):
            skeletonize(im, method='foo')

    def test_skeletonize_all_foreground(self):
        if False:
            for i in range(10):
                print('nop')
        im = np.ones((3, 4))
        skeletonize(im)

    def test_skeletonize_single_point(self):
        if False:
            i = 10
            return i + 15
        im = np.zeros((5, 5), np.uint8)
        im[3, 3] = 1
        result = skeletonize(im)
        assert_array_equal(result, im)

    def test_skeletonize_already_thinned(self):
        if False:
            i = 10
            return i + 15
        im = np.zeros((5, 5), np.uint8)
        im[3, 1:-1] = 1
        im[2, -1] = 1
        im[4, 0] = 1
        result = skeletonize(im)
        assert_array_equal(result, im)

    def test_skeletonize_output(self):
        if False:
            for i in range(10):
                print('nop')
        im = imread(fetch('data/bw_text.png'), as_gray=True)
        im = im == 0
        result = skeletonize(im)
        expected = np.load(fetch('data/bw_text_skeleton.npy'))
        assert_array_equal(result, expected)

    def test_skeletonize_num_neighbors(self):
        if False:
            for i in range(10):
                print('nop')
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
        result = skeletonize(image)
        mask = np.array([[1, 1], [1, 1]], np.uint8)
        blocks = correlate(result, mask, mode='constant')
        assert not np.any(blocks == 4)

    def test_lut_fix(self):
        if False:
            i = 10
            return i + 15
        im = np.zeros((6, 6), np.uint8)
        im[1, 2] = 1
        im[2, 2] = 1
        im[2, 3] = 1
        im[3, 3] = 1
        im[3, 4] = 1
        im[4, 4] = 1
        im[4, 5] = 1
        result = skeletonize(im)
        expected = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        assert np.all(result == expected)

class TestThin:

    @property
    def input_image(self):
        if False:
            for i in range(10):
                print('nop')
        'image to test thinning with'
        ii = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        return ii

    def test_zeros(self):
        if False:
            while True:
                i = 10
        assert np.all(thin(np.zeros((10, 10))) == False)

    def test_iter_1(self):
        if False:
            while True:
                i = 10
        result = thin(self.input_image, 1).astype(np.uint8)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_noiter(self):
        if False:
            return 10
        result = thin(self.input_image).astype(np.uint8)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_baddim(self):
        if False:
            return 10
        for ii in [np.zeros(3), np.zeros((3, 3, 3))]:
            with pytest.raises(ValueError):
                thin(ii)

    def test_lut_generation(self):
        if False:
            return 10
        (g123, g123p) = _generate_thin_luts()
        assert_array_equal(g123, G123_LUT)
        assert_array_equal(g123p, G123P_LUT)

class TestMedialAxis:

    def test_00_00_zeros(self):
        if False:
            i = 10
            return i + 15
        'Test skeletonize on an array of all zeros'
        result = medial_axis(np.zeros((10, 10), bool))
        assert np.all(result == False)

    def test_00_01_zeros_masked(self):
        if False:
            print('Hello World!')
        'Test skeletonize on an array that is completely masked'
        result = medial_axis(np.zeros((10, 10), bool), np.zeros((10, 10), bool))
        assert np.all(result == False)

    def test_vertical_line(self):
        if False:
            while True:
                i = 10
        'Test a thick vertical line, issue #3861'
        img = np.zeros((9, 9))
        img[:, 2] = 1
        img[:, 3] = 1
        img[:, 4] = 1
        expected = np.full(img.shape, False)
        expected[:, 3] = True
        result = medial_axis(img)
        assert_array_equal(result, expected)

    def test_01_01_rectangle(self):
        if False:
            for i in range(10):
                print('nop')
        'Test skeletonize on a rectangle'
        image = np.zeros((9, 15), bool)
        image[1:-1, 1:-1] = True
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        result = medial_axis(image)
        assert np.all(result == expected)
        (result, distance) = medial_axis(image, return_distance=True)
        assert distance.max() == 4

    def test_01_02_hole(self):
        if False:
            print('Hello World!')
        'Test skeletonize on a rectangle with a hole in the middle'
        image = np.zeros((9, 15), bool)
        image[1:-1, 1:-1] = True
        image[4, 4:-4] = False
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        result = medial_axis(image)
        assert np.all(result == expected)

    def test_narrow_image(self):
        if False:
            return 10
        'Test skeletonize on a 1-pixel thin strip'
        image = np.zeros((1, 5), bool)
        image[:, 1:-1] = True
        result = medial_axis(image)
        assert np.all(result == image)

    def test_deprecated_random_state(self):
        if False:
            print('Hello World!')
        'Test medial_axis on an array of all zeros.'
        with expected_warnings(['`random_state` is a deprecated argument']):
            medial_axis(np.zeros((10, 10), bool), random_state=None)