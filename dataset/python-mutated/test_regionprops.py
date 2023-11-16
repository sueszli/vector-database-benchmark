import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
import numpydoc
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import COL_DTYPES, OBJECT_COLUMNS, PROPS, _inertia_eigvals_to_axes_lengths_3D, _parse_docs, _props_to_dict, _require_intensity_image, euler_number, perimeter, perimeter_crofton, regionprops, regionprops_table
from skimage.segmentation import slic
SAMPLE = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
INTENSITY_SAMPLE = SAMPLE.copy()
INTENSITY_SAMPLE[1, 9:11] = 2
INTENSITY_FLOAT_SAMPLE = INTENSITY_SAMPLE.copy().astype(np.float64) / 10.0
INTENSITY_FLOAT_SAMPLE_MULTICHANNEL = INTENSITY_FLOAT_SAMPLE[..., np.newaxis] * [1, 2, 3]
SAMPLE_MULTIPLE = np.eye(10, dtype=np.int32)
SAMPLE_MULTIPLE[3:5, 7:8] = 2
INTENSITY_SAMPLE_MULTIPLE = SAMPLE_MULTIPLE.copy() * 2.0
SAMPLE_3D = np.zeros((6, 6, 6), dtype=np.uint8)
SAMPLE_3D[1:3, 1:3, 1:3] = 1
SAMPLE_3D[3, 2, 2] = 1
INTENSITY_SAMPLE_3D = SAMPLE_3D.copy()

def get_moment_function(img, spacing=(1, 1)):
    if False:
        while True:
            i = 10
    (rows, cols) = img.shape
    (Y, X) = np.meshgrid(np.linspace(0, rows * spacing[0], rows, endpoint=False), np.linspace(0, cols * spacing[1], cols, endpoint=False), indexing='ij')
    return lambda p, q: np.sum(Y ** p * X ** q * img)

def get_moment3D_function(img, spacing=(1, 1, 1)):
    if False:
        while True:
            i = 10
    (slices, rows, cols) = img.shape
    (Z, Y, X) = np.meshgrid(np.linspace(0, slices * spacing[0], slices, endpoint=False), np.linspace(0, rows * spacing[1], rows, endpoint=False), np.linspace(0, cols * spacing[2], cols, endpoint=False), indexing='ij')
    return lambda p, q, r: np.sum(Z ** p * Y ** q * X ** r * img)

def get_central_moment_function(img, spacing=(1, 1)):
    if False:
        return 10
    (rows, cols) = img.shape
    (Y, X) = np.meshgrid(np.linspace(0, rows * spacing[0], rows, endpoint=False), np.linspace(0, cols * spacing[1], cols, endpoint=False), indexing='ij')
    Mpq = get_moment_function(img, spacing=spacing)
    cY = Mpq(1, 0) / Mpq(0, 0)
    cX = Mpq(0, 1) / Mpq(0, 0)
    return lambda p, q: np.sum((Y - cY) ** p * (X - cX) ** q * img)

def test_all_props():
    if False:
        print('Hello World!')
    region = regionprops(SAMPLE, INTENSITY_SAMPLE)[0]
    for prop in PROPS:
        try:
            assert_almost_equal(region[prop], getattr(region, PROPS[prop]))
            if prop.lower() == prop:
                assert_almost_equal(getattr(region, prop), getattr(region, PROPS[prop]))
        except TypeError:
            pass

def test_all_props_3d():
    if False:
        print('Hello World!')
    region = regionprops(SAMPLE_3D, INTENSITY_SAMPLE_3D)[0]
    for prop in PROPS:
        try:
            assert_almost_equal(region[prop], getattr(region, PROPS[prop]))
            if prop.lower() == prop:
                assert_almost_equal(getattr(region, prop), getattr(region, PROPS[prop]))
        except (NotImplementedError, TypeError):
            pass

def test_num_pixels():
    if False:
        return 10
    num_pixels = regionprops(SAMPLE)[0].num_pixels
    assert num_pixels == 72
    num_pixels = regionprops(SAMPLE, spacing=(2, 1))[0].num_pixels
    assert num_pixels == 72

def test_dtype():
    if False:
        return 10
    regionprops(np.zeros((10, 10), dtype=int))
    regionprops(np.zeros((10, 10), dtype=np.uint))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=float))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=np.float64))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=bool))

def test_ndim():
    if False:
        return 10
    regionprops(np.zeros((10, 10), dtype=int))
    regionprops(np.zeros((10, 10, 1), dtype=int))
    regionprops(np.zeros((10, 10, 10), dtype=int))
    regionprops(np.zeros((1, 1), dtype=int))
    regionprops(np.zeros((1, 1, 1), dtype=int))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10, 10, 2), dtype=int))

def test_feret_diameter_max():
    if False:
        while True:
            i = 10
    comparator_result = 18
    test_result = regionprops(SAMPLE)[0].feret_diameter_max
    assert np.abs(test_result - comparator_result) < 1
    comparator_result_spacing = 10
    test_result_spacing = regionprops(SAMPLE, spacing=[1, 0.1])[0].feret_diameter_max
    assert np.abs(test_result_spacing - comparator_result_spacing) < 1
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:-2, 2:-2] = 1
    feret_diameter_max = regionprops(img)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - 16 * np.sqrt(2)) < 1
    assert np.abs(feret_diameter_max - np.sqrt(16 ** 2 + (16 - 1) ** 2)) < 1e-06

def test_feret_diameter_max_spacing():
    if False:
        print('Hello World!')
    comparator_result = 18
    test_result = regionprops(SAMPLE)[0].feret_diameter_max
    assert np.abs(test_result - comparator_result) < 1
    spacing = (2, 1)
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:-2, 2:-2] = 1
    feret_diameter_max = regionprops(img, spacing=spacing)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - np.sqrt((spacing[0] * 16 - (spacing[0] <= spacing[1])) ** 2 + (spacing[1] * 16 - (spacing[1] < spacing[0])) ** 2)) < 1e-06

def test_feret_diameter_max_3d():
    if False:
        while True:
            i = 10
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:-2, 2:-2] = 1
    img_3d = np.dstack((img,) * 3)
    feret_diameter_max = regionprops(img_3d)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - np.sqrt((16 - 1) ** 2 + 16 ** 2 + (3 - 1) ** 2)) < 1e-06
    spacing = (1, 2, 3)
    feret_diameter_max = regionprops(img_3d, spacing=spacing)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - np.sqrt((spacing[0] * (16 - 1)) ** 2 + (spacing[1] * (16 - 0)) ** 2 + (spacing[2] * (3 - 1)) ** 2)) < 1e-06
    assert np.abs(feret_diameter_max - np.sqrt((spacing[0] * (16 - 1)) ** 2 + (spacing[1] * (16 - 1)) ** 2 + (spacing[2] * (3 - 0)) ** 2)) > 1e-06
    assert np.abs(feret_diameter_max - np.sqrt((spacing[0] * (16 - 0)) ** 2 + (spacing[1] * (16 - 1)) ** 2 + (spacing[2] * (3 - 1)) ** 2)) > 1e-06

@pytest.mark.parametrize('sample,spacing', [(SAMPLE, None), (SAMPLE, 1), (SAMPLE, (1, 1)), (SAMPLE, (1, 2)), (SAMPLE_3D, None), (SAMPLE_3D, 1), (SAMPLE_3D, (2, 1, 3))])
def test_area(sample, spacing):
    if False:
        print('Hello World!')
    area = regionprops(sample, spacing=spacing)[0].area
    desired = np.sum(sample * (np.prod(spacing) if spacing else 1))
    assert area == desired

def test_bbox():
    if False:
        while True:
            i = 10
    bbox = regionprops(SAMPLE)[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]))
    bbox = regionprops(SAMPLE, spacing=(1, 2))[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]))
    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[:, -1] = 0
    bbox = regionprops(SAMPLE_mod)[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1] - 1))
    bbox = regionprops(SAMPLE_mod, spacing=(3, 2))[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1] - 1))
    bbox = regionprops(SAMPLE_3D)[0].bbox
    assert_array_almost_equal(bbox, (1, 1, 1, 4, 3, 3))
    bbox = regionprops(SAMPLE_3D, spacing=(0.5, 2, 7))[0].bbox
    assert_array_almost_equal(bbox, (1, 1, 1, 4, 3, 3))

def test_area_bbox():
    if False:
        print('Hello World!')
    padded = np.pad(SAMPLE, 5, mode='constant')
    bbox_area = regionprops(padded)[0].area_bbox
    assert_array_almost_equal(bbox_area, SAMPLE.size)

def test_area_bbox_spacing():
    if False:
        print('Hello World!')
    spacing = (0.5, 3)
    padded = np.pad(SAMPLE, 5, mode='constant')
    bbox_area = regionprops(padded, spacing=spacing)[0].area_bbox
    assert_array_almost_equal(bbox_area, SAMPLE.size * np.prod(spacing))

def test_moments_central():
    if False:
        print('Hello World!')
    mu = regionprops(SAMPLE)[0].moments_central
    assert_almost_equal(mu[2, 0], 436.00000000000045)
    assert_almost_equal(mu[3, 0], -737.333333333333)
    assert_almost_equal(mu[1, 1], -87.33333333333303)
    assert_almost_equal(mu[2, 1], -127.5555555555593)
    assert_almost_equal(mu[0, 2], 1259.7777777777774)
    assert_almost_equal(mu[1, 2], 2000.296296296291)
    assert_almost_equal(mu[0, 3], -760.0246913580195)
    centralMpq = get_central_moment_function(SAMPLE, spacing=(1, 1))
    assert_almost_equal(centralMpq(2, 0), mu[2, 0])
    assert_almost_equal(centralMpq(3, 0), mu[3, 0])
    assert_almost_equal(centralMpq(1, 1), mu[1, 1])
    assert_almost_equal(centralMpq(2, 1), mu[2, 1])
    assert_almost_equal(centralMpq(0, 2), mu[0, 2])
    assert_almost_equal(centralMpq(1, 2), mu[1, 2])
    assert_almost_equal(centralMpq(0, 3), mu[0, 3])

def test_moments_central_spacing():
    if False:
        print('Hello World!')
    spacing = (1.8, 0.8)
    centralMpq = get_central_moment_function(SAMPLE, spacing=spacing)
    mu = regionprops(SAMPLE, spacing=spacing)[0].moments_central
    assert_almost_equal(mu[2, 0], centralMpq(2, 0))
    assert_almost_equal(mu[3, 0], centralMpq(3, 0))
    assert_almost_equal(mu[1, 1], centralMpq(1, 1))
    assert_almost_equal(mu[2, 1], centralMpq(2, 1))
    assert_almost_equal(mu[0, 2], centralMpq(0, 2))
    assert_almost_equal(mu[1, 2], centralMpq(1, 2))
    assert_almost_equal(mu[0, 3], centralMpq(0, 3))

def test_centroid():
    if False:
        while True:
            i = 10
    centroid = regionprops(SAMPLE)[0].centroid
    assert_array_almost_equal(centroid, (5.66666666666666, 9.444444444444445))
    Mpq = get_moment_function(SAMPLE, spacing=(1, 1))
    cY = Mpq(1, 0) / Mpq(0, 0)
    cX = Mpq(0, 1) / Mpq(0, 0)
    assert_array_almost_equal((cY, cX), centroid)

def test_centroid_spacing():
    if False:
        while True:
            i = 10
    spacing = (1.8, 0.8)
    Mpq = get_moment_function(SAMPLE, spacing=spacing)
    cY = Mpq(1, 0) / Mpq(0, 0)
    cX = Mpq(0, 1) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, spacing=spacing)[0].centroid
    assert_array_almost_equal(centroid, (cY, cX))

def test_centroid_3d():
    if False:
        return 10
    centroid = regionprops(SAMPLE_3D)[0].centroid
    assert_array_almost_equal(centroid, (1.66666667, 1.55555556, 1.55555556))
    Mpqr = get_moment3D_function(SAMPLE_3D, spacing=(1, 1, 1))
    cZ = Mpqr(1, 0, 0) / Mpqr(0, 0, 0)
    cY = Mpqr(0, 1, 0) / Mpqr(0, 0, 0)
    cX = Mpqr(0, 0, 1) / Mpqr(0, 0, 0)
    assert_array_almost_equal((cZ, cY, cX), centroid)

@pytest.mark.parametrize('spacing', [[2.1, 2.2, 2.3], [2.0, 2.0, 2.0], [2, 2, 2]])
def test_spacing_parameter_3d(spacing):
    if False:
        while True:
            i = 10
    'Test the _normalize_spacing code.'
    Mpqr = get_moment3D_function(SAMPLE_3D, spacing=spacing)
    cZ = Mpqr(1, 0, 0) / Mpqr(0, 0, 0)
    cY = Mpqr(0, 1, 0) / Mpqr(0, 0, 0)
    cX = Mpqr(0, 0, 1) / Mpqr(0, 0, 0)
    centroid = regionprops(SAMPLE_3D, spacing=spacing)[0].centroid
    assert_array_almost_equal(centroid, (cZ, cY, cX))

@pytest.mark.parametrize('spacing', [(1, 1j), 1 + 0j])
def test_spacing_parameter_complex_input(spacing):
    if False:
        for i in range(10):
            print('nop')
    'Test the _normalize_spacing code.'
    with pytest.raises(TypeError, match="Element of spacing isn't float or integer type, got"):
        regionprops(SAMPLE, spacing=spacing)[0].centroid

@pytest.mark.parametrize('spacing', [np.nan, np.inf, -np.inf])
def test_spacing_parameter_nan_inf(spacing):
    if False:
        i = 10
        return i + 15
    'Test the _normalize_spacing code.'
    with pytest.raises(ValueError):
        regionprops(SAMPLE, spacing=spacing)[0].centroid

@pytest.mark.parametrize('spacing', ([1], [[1, 1]], (1, 1, 1)))
def test_spacing_mismtaching_shape(spacing):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match="spacing isn't a scalar nor a sequence"):
        regionprops(SAMPLE, spacing=spacing)[0].centroid

@pytest.mark.parametrize('spacing', [[2.1, 2.2], [2.0, 2.0], [2, 2]])
def test_spacing_parameter_2d(spacing):
    if False:
        while True:
            i = 10
    'Test the _normalize_spacing code.'
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, (cX, cY))

@pytest.mark.parametrize('spacing', [['bad input'], ['bad input', 1, 2.1]])
def test_spacing_parameter_2d_bad_input(spacing):
    if False:
        print('Hello World!')
    'Test the _normalize_spacing code.'
    with pytest.raises(ValueError):
        regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].centroid_weighted

def test_area_convex():
    if False:
        while True:
            i = 10
    area = regionprops(SAMPLE)[0].area_convex
    assert area == 125

def test_area_convex_spacing():
    if False:
        while True:
            i = 10
    spacing = (1, 4)
    area = regionprops(SAMPLE, spacing=spacing)[0].area_convex
    assert area == 125 * np.prod(spacing)

def test_image_convex():
    if False:
        while True:
            i = 10
    img = regionprops(SAMPLE)[0].image_convex
    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    assert_array_equal(img, ref)

def test_coordinates():
    if False:
        i = 10
        return i + 15
    sample = np.zeros((10, 10), dtype=np.int8)
    coords = np.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1
    prop_coords = regionprops(sample)[0].coords
    assert_array_equal(prop_coords, coords)
    prop_coords = regionprops(sample, spacing=(0.5, 1.2))[0].coords
    assert_array_equal(prop_coords, coords)

@pytest.mark.parametrize('spacing', [None, 1, 2, (1, 1), (1, 0.5)])
def test_coordinates_scaled(spacing):
    if False:
        return 10
    sample = np.zeros((10, 10), dtype=np.int8)
    coords = np.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1
    prop_coords = regionprops(sample, spacing=spacing)[0].coords_scaled
    if spacing is None:
        desired_coords = coords
    else:
        desired_coords = coords * np.array(spacing)
    assert_array_equal(prop_coords, desired_coords)

@pytest.mark.parametrize('spacing', [None, 1, 2, (0.2, 3, 2.3)])
def test_coordinates_scaled_3d(spacing):
    if False:
        while True:
            i = 10
    sample = np.zeros((6, 6, 6), dtype=np.int8)
    coords = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1]])
    sample[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    prop_coords = regionprops(sample, spacing=spacing)[0].coords_scaled
    if spacing is None:
        desired_coords = coords
    else:
        desired_coords = coords * np.array(spacing)
    assert_array_equal(prop_coords, desired_coords)

def test_slice():
    if False:
        print('Hello World!')
    padded = np.pad(SAMPLE, ((2, 4), (5, 2)), mode='constant')
    (nrow, ncol) = SAMPLE.shape
    result = regionprops(padded)[0].slice
    expected = (slice(2, 2 + nrow), slice(5, 5 + ncol))
    assert_equal(result, expected)

def test_slice_spacing():
    if False:
        print('Hello World!')
    padded = np.pad(SAMPLE, ((2, 4), (5, 2)), mode='constant')
    (nrow, ncol) = SAMPLE.shape
    result = regionprops(padded)[0].slice
    expected = (slice(2, 2 + nrow), slice(5, 5 + ncol))
    spacing = (2, 0.2)
    result = regionprops(padded, spacing=spacing)[0].slice
    assert_equal(result, expected)

def test_eccentricity():
    if False:
        print('Hello World!')
    eps = regionprops(SAMPLE)[0].eccentricity
    assert_almost_equal(eps, 0.814629313427)
    eps = regionprops(SAMPLE, spacing=(1.5, 1.5))[0].eccentricity
    assert_almost_equal(eps, 0.814629313427)
    img = np.zeros((5, 5), dtype=int)
    img[2, 2] = 1
    eps = regionprops(img)[0].eccentricity
    assert_almost_equal(eps, 0)
    eps = regionprops(img, spacing=(3, 3))[0].eccentricity
    assert_almost_equal(eps, 0)

def test_equivalent_diameter_area():
    if False:
        return 10
    diameter = regionprops(SAMPLE)[0].equivalent_diameter_area
    assert_almost_equal(diameter, 9.57461472963)
    spacing = (1, 3)
    diameter = regionprops(SAMPLE, spacing=spacing)[0].equivalent_diameter_area
    equivalent_area = np.pi * (diameter / 2.0) ** 2
    assert_almost_equal(equivalent_area, SAMPLE.sum() * np.prod(spacing))

def test_euler_number():
    if False:
        while True:
            i = 10
    for spacing in [(1, 1), (2.1, 0.9)]:
        en = regionprops(SAMPLE, spacing=spacing)[0].euler_number
        assert en == 0
        SAMPLE_mod = SAMPLE.copy()
        SAMPLE_mod[7, -3] = 0
        en = regionprops(SAMPLE_mod, spacing=spacing)[0].euler_number
        assert en == -1
        en = euler_number(SAMPLE, 1)
        assert en == 2
        en = euler_number(SAMPLE_mod, 1)
        assert en == 1
    en = euler_number(SAMPLE_3D, 1)
    assert en == 1
    en = euler_number(SAMPLE_3D, 3)
    assert en == 1
    SAMPLE_3D_2 = np.zeros((100, 100, 100))
    SAMPLE_3D_2[40:60, 40:60, 40:60] = 1
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 1
    SAMPLE_3D_2[45:55, 45:55, 45:55] = 0
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 2

def test_extent():
    if False:
        i = 10
        return i + 15
    extent = regionprops(SAMPLE)[0].extent
    assert_almost_equal(extent, 0.4)
    extent = regionprops(SAMPLE, spacing=(5, 0.2))[0].extent
    assert_almost_equal(extent, 0.4)

def test_moments_hu():
    if False:
        while True:
            i = 10
    hu = regionprops(SAMPLE)[0].moments_hu
    ref = np.array([0.327117627, 0.0263869194, 0.023539006, 0.00123151193, 1.3888233e-06, -2.72586158e-05, -6.48350653e-06])
    assert_array_almost_equal(hu, ref)
    with testing.raises(NotImplementedError):
        regionprops(SAMPLE, spacing=(2, 1))[0].moments_hu

def test_image():
    if False:
        print('Hello World!')
    img = regionprops(SAMPLE)[0].image
    assert_array_equal(img, SAMPLE)
    img = regionprops(SAMPLE_3D)[0].image
    assert_array_equal(img, SAMPLE_3D[1:4, 1:3, 1:3])

def test_label():
    if False:
        print('Hello World!')
    label = regionprops(SAMPLE)[0].label
    assert_array_equal(label, 1)
    label = regionprops(SAMPLE_3D)[0].label
    assert_array_equal(label, 1)

def test_area_filled():
    if False:
        for i in range(10):
            print('nop')
    area = regionprops(SAMPLE)[0].area_filled
    assert area == np.sum(SAMPLE)

def test_area_filled_zero():
    if False:
        for i in range(10):
            print('nop')
    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    area = regionprops(SAMPLE_mod)[0].area_filled
    assert area == np.sum(SAMPLE)

def test_area_filled_spacing():
    if False:
        for i in range(10):
            print('nop')
    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    spacing = (2, 1.2)
    area = regionprops(SAMPLE, spacing=spacing)[0].area_filled
    assert area == np.sum(SAMPLE) * np.prod(spacing)
    area = regionprops(SAMPLE_mod, spacing=spacing)[0].area_filled
    assert area == np.sum(SAMPLE) * np.prod(spacing)

def test_image_filled():
    if False:
        for i in range(10):
            print('nop')
    img = regionprops(SAMPLE)[0].image_filled
    assert_array_equal(img, SAMPLE)
    img = regionprops(SAMPLE, spacing=(1, 4))[0].image_filled
    assert_array_equal(img, SAMPLE)

def test_axis_major_length():
    if False:
        for i in range(10):
            print('nop')
    length = regionprops(SAMPLE)[0].axis_major_length
    target_length = 16.7924234999
    assert_almost_equal(length, target_length)
    length = regionprops(SAMPLE, spacing=(2, 2))[0].axis_major_length
    assert_almost_equal(length, 2 * target_length)
    from skimage.draw import ellipse
    img = np.zeros((20, 24), dtype=np.uint8)
    (rr, cc) = ellipse(11, 11, 7, 9, rotation=np.deg2rad(45))
    img[rr, cc] = 1
    target_length = regionprops(img, spacing=(1, 1))[0].axis_major_length
    length_wo_spacing = regionprops(img[::2], spacing=(1, 1))[0].axis_minor_length
    assert abs(length_wo_spacing - target_length) > 0.1
    length = regionprops(img[:, ::2], spacing=(1, 2))[0].axis_major_length
    assert_almost_equal(length, target_length, decimal=0)

def test_intensity_max():
    if False:
        return 10
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].intensity_max
    assert_almost_equal(intensity, 2)

def test_intensity_mean():
    if False:
        print('Hello World!')
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].intensity_mean
    assert_almost_equal(intensity, 1.02777777777777)

def test_intensity_min():
    if False:
        print('Hello World!')
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].intensity_min
    assert_almost_equal(intensity, 1)

def test_intensity_std():
    if False:
        print('Hello World!')
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].intensity_std
    assert_almost_equal(intensity, 0.16433554953054486)

def test_axis_minor_length():
    if False:
        print('Hello World!')
    length = regionprops(SAMPLE)[0].axis_minor_length
    target_length = 9.739302807263
    assert_almost_equal(length, target_length)
    length = regionprops(SAMPLE, spacing=(1.5, 1.5))[0].axis_minor_length
    assert_almost_equal(length, 1.5 * target_length)
    from skimage.draw import ellipse
    img = np.zeros((10, 12), dtype=np.uint8)
    (rr, cc) = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    img[rr, cc] = 1
    target_length = regionprops(img, spacing=(1, 1))[0].axis_minor_length
    length_wo_spacing = regionprops(img[::2], spacing=(1, 1))[0].axis_minor_length
    assert abs(length_wo_spacing - target_length) > 0.1
    length = regionprops(img[::2], spacing=(2, 1))[0].axis_minor_length
    assert_almost_equal(length, target_length, decimal=1)

def test_moments():
    if False:
        for i in range(10):
            print('nop')
    m = regionprops(SAMPLE)[0].moments
    assert_almost_equal(m[0, 0], 72.0)
    assert_almost_equal(m[0, 1], 680.0)
    assert_almost_equal(m[0, 2], 7682.0)
    assert_almost_equal(m[0, 3], 95588.0)
    assert_almost_equal(m[1, 0], 408.0)
    assert_almost_equal(m[1, 1], 3766.0)
    assert_almost_equal(m[1, 2], 43882.0)
    assert_almost_equal(m[2, 0], 2748.0)
    assert_almost_equal(m[2, 1], 24836.0)
    assert_almost_equal(m[3, 0], 19776.0)
    Mpq = get_moment_function(SAMPLE, spacing=(1, 1))
    assert_almost_equal(Mpq(0, 0), m[0, 0])
    assert_almost_equal(Mpq(0, 1), m[0, 1])
    assert_almost_equal(Mpq(0, 2), m[0, 2])
    assert_almost_equal(Mpq(0, 3), m[0, 3])
    assert_almost_equal(Mpq(1, 0), m[1, 0])
    assert_almost_equal(Mpq(1, 1), m[1, 1])
    assert_almost_equal(Mpq(1, 2), m[1, 2])
    assert_almost_equal(Mpq(2, 0), m[2, 0])
    assert_almost_equal(Mpq(2, 1), m[2, 1])
    assert_almost_equal(Mpq(3, 0), m[3, 0])

def test_moments_spacing():
    if False:
        for i in range(10):
            print('nop')
    spacing = (2, 0.3)
    m = regionprops(SAMPLE, spacing=spacing)[0].moments
    Mpq = get_moment_function(SAMPLE, spacing=spacing)
    assert_almost_equal(m[0, 0], Mpq(0, 0))
    assert_almost_equal(m[0, 1], Mpq(0, 1))
    assert_almost_equal(m[0, 2], Mpq(0, 2))
    assert_almost_equal(m[0, 3], Mpq(0, 3))
    assert_almost_equal(m[1, 0], Mpq(1, 0))
    assert_almost_equal(m[1, 1], Mpq(1, 1))
    assert_almost_equal(m[1, 2], Mpq(1, 2))
    assert_almost_equal(m[2, 0], Mpq(2, 0))
    assert_almost_equal(m[2, 1], Mpq(2, 1))
    assert_almost_equal(m[3, 0], Mpq(3, 0))

def test_moments_normalized():
    if False:
        print('Hello World!')
    nu = regionprops(SAMPLE)[0].moments_normalized
    assert_almost_equal(nu[0, 2], 0.24301268861454037)
    assert_almost_equal(nu[0, 3], -0.017278118992041805)
    assert_almost_equal(nu[1, 1], -0.016846707818929982)
    assert_almost_equal(nu[1, 2], 0.045473992910668816)
    assert_almost_equal(nu[2, 0], 0.08410493827160502)
    assert_almost_equal(nu[2, 1], -0.002899800614433943)

def test_moments_normalized_spacing():
    if False:
        print('Hello World!')
    spacing = (3, 3)
    nu = regionprops(SAMPLE, spacing=spacing)[0].moments_normalized
    assert_almost_equal(nu[0, 2], 0.24301268861454037)
    assert_almost_equal(nu[0, 3], -0.017278118992041805)
    assert_almost_equal(nu[1, 1], -0.016846707818929982)
    assert_almost_equal(nu[1, 2], 0.045473992910668816)
    assert_almost_equal(nu[2, 0], 0.08410493827160502)
    assert_almost_equal(nu[2, 1], -0.002899800614433943)

def test_orientation():
    if False:
        i = 10
        return i + 15
    orient = regionprops(SAMPLE)[0].orientation
    target_orient = -1.4663278802756865
    assert_almost_equal(orient, target_orient)
    orient = regionprops(SAMPLE, spacing=(2, 2))[0].orientation
    assert_almost_equal(orient, target_orient)
    diag = np.eye(10, dtype=int)
    orient_diag = regionprops(diag)[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(diag, spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))
    orient_diag = regionprops(np.flipud(diag))[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(np.flipud(diag), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, -np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))
    orient_diag = regionprops(np.fliplr(diag))[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(np.fliplr(diag), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, -np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))
    orient_diag = regionprops(np.fliplr(np.flipud(diag)))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(np.fliplr(np.flipud(diag)), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, np.arccos(0.5 / np.sqrt(1 + 0.5 ** 2)))

def test_orientation_continuity():
    if False:
        for i in range(10):
            print('nop')
    arr1 = np.array([[0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    arr2 = np.array([[0, 0, 0, 2], [0, 0, 2, 0], [0, 2, 0, 0], [2, 0, 0, 0]])
    arr3 = np.array([[0, 0, 0, 3], [0, 0, 3, 3], [0, 3, 0, 0], [3, 0, 0, 0]])
    image = np.hstack((arr1, arr2, arr3))
    props = regionprops(image)
    orientations = [prop.orientation for prop in props]
    np.testing.assert_allclose(orientations, orientations[1], rtol=0, atol=0.08)
    assert_almost_equal(orientations[0], -0.7144496360953664)
    assert_almost_equal(orientations[1], -0.7853981633974483)
    assert_almost_equal(orientations[2], -0.8563466906995303)
    spacing = (3.2, 1.2)
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].moments_weighted_central
    centralMpq = get_central_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    assert_almost_equal(wmu[0, 0], centralMpq(0, 0))
    assert_almost_equal(wmu[0, 1], centralMpq(0, 1))
    assert_almost_equal(wmu[0, 2], centralMpq(0, 2))
    assert_almost_equal(wmu[0, 3], centralMpq(0, 3))
    assert_almost_equal(wmu[1, 0], centralMpq(1, 0))
    assert_almost_equal(wmu[1, 1], centralMpq(1, 1))
    assert_almost_equal(wmu[1, 2], centralMpq(1, 2))
    assert_almost_equal(wmu[1, 3], centralMpq(1, 3))
    assert_almost_equal(wmu[2, 0], centralMpq(2, 0))
    assert_almost_equal(wmu[2, 1], centralMpq(2, 1))
    assert_almost_equal(wmu[2, 2], centralMpq(2, 2))
    assert_almost_equal(wmu[2, 3], centralMpq(2, 3))
    assert_almost_equal(wmu[3, 0], centralMpq(3, 0))
    assert_almost_equal(wmu[3, 1], centralMpq(3, 1))
    assert_almost_equal(wmu[3, 2], centralMpq(3, 2))
    assert_almost_equal(wmu[3, 3], centralMpq(3, 3))

def test_perimeter():
    if False:
        i = 10
        return i + 15
    per = regionprops(SAMPLE)[0].perimeter
    target_per = 55.2487373415
    assert_almost_equal(per, target_per)
    per = regionprops(SAMPLE, spacing=(2, 2))[0].perimeter
    assert_almost_equal(per, 2 * target_per)
    per = perimeter(SAMPLE.astype('double'), neighborhood=8)
    assert_almost_equal(per, 46.8284271247)
    with testing.raises(NotImplementedError):
        per = regionprops(SAMPLE, spacing=(2, 1))[0].perimeter

def test_perimeter_crofton():
    if False:
        return 10
    per = regionprops(SAMPLE)[0].perimeter_crofton
    target_per_crof = 61.0800637973
    assert_almost_equal(per, target_per_crof)
    per = regionprops(SAMPLE, spacing=(2, 2))[0].perimeter_crofton
    assert_almost_equal(per, 2 * target_per_crof)
    per = perimeter_crofton(SAMPLE.astype('double'), directions=2)
    assert_almost_equal(per, 64.4026493985)
    with testing.raises(NotImplementedError):
        per = regionprops(SAMPLE, spacing=(2, 1))[0].perimeter_crofton

def test_solidity():
    if False:
        print('Hello World!')
    solidity = regionprops(SAMPLE)[0].solidity
    target_solidity = 0.576
    assert_almost_equal(solidity, target_solidity)
    solidity = regionprops(SAMPLE, spacing=(3, 9))[0].solidity
    assert_almost_equal(solidity, target_solidity)

def test_multichannel_centroid_weighted_table():
    if False:
        return 10
    'Test for https://github.com/scikit-image/scikit-image/issues/6860.'
    intensity_image = INTENSITY_FLOAT_SAMPLE_MULTICHANNEL
    rp0 = regionprops(SAMPLE, intensity_image=intensity_image[..., 0])[0]
    rp1 = regionprops(SAMPLE, intensity_image=intensity_image[..., 0:1])[0]
    rpm = regionprops(SAMPLE, intensity_image=intensity_image)[0]
    np.testing.assert_almost_equal(rp0.centroid_weighted, np.squeeze(rp1.centroid_weighted))
    np.testing.assert_almost_equal(rp0.centroid_weighted, np.array(rpm.centroid_weighted)[:, 0])
    assert np.shape(rp0.centroid_weighted) == (SAMPLE.ndim,)
    assert np.shape(rp1.centroid_weighted) == (SAMPLE.ndim, 1)
    assert np.shape(rpm.centroid_weighted) == (SAMPLE.ndim, intensity_image.shape[-1])
    table = regionprops_table(SAMPLE, intensity_image=intensity_image, properties=('centroid_weighted',))
    assert len(table) == np.size(rpm.centroid_weighted)

def test_moments_weighted_central():
    if False:
        print('Hello World!')
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted_central
    ref = np.array([[74.0, 3.7303493627e-14, 1260.2837838, -765.61796932], [-2.1316282073e-13, -87.837837838, 2157.1526662, -4238.5971907], [478.37837838, -148.01314828, 6698.979942, -9950.1164076], [-759.43608473, -1271.4707125, 15304.076361, -33156.729271]])
    np.set_printoptions(precision=10)
    assert_array_almost_equal(wmu, ref)
    centralMpq = get_central_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    assert_almost_equal(centralMpq(0, 0), ref[0, 0])
    assert_almost_equal(centralMpq(0, 1), ref[0, 1])
    assert_almost_equal(centralMpq(0, 2), ref[0, 2])
    assert_almost_equal(centralMpq(0, 3), ref[0, 3])
    assert_almost_equal(centralMpq(1, 0), ref[1, 0])
    assert_almost_equal(centralMpq(1, 1), ref[1, 1])
    assert_almost_equal(centralMpq(1, 2), ref[1, 2])
    assert_almost_equal(centralMpq(1, 3), ref[1, 3])
    assert_almost_equal(centralMpq(2, 0), ref[2, 0])
    assert_almost_equal(centralMpq(2, 1), ref[2, 1])
    assert_almost_equal(centralMpq(2, 2), ref[2, 2])
    assert_almost_equal(centralMpq(2, 3), ref[2, 3])
    assert_almost_equal(centralMpq(3, 0), ref[3, 0])
    assert_almost_equal(centralMpq(3, 1), ref[3, 1])
    assert_almost_equal(centralMpq(3, 2), ref[3, 2])
    assert_almost_equal(centralMpq(3, 3), ref[3, 3])

def test_centroid_weighted():
    if False:
        for i in range(10):
            print('nop')
    sample_for_spacing = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    target_centroid_wspacing = (4.0, 4.0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].centroid_weighted
    target_centroid = (5.54054054054, 9.445945945945)
    assert_array_almost_equal(centroid, target_centroid)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    assert_almost_equal((cX, cY), centroid)
    spacing = (2, 2)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, (cX, cY))
    assert_almost_equal(centroid, 2 * np.array(target_centroid))
    centroid = regionprops(sample_for_spacing, intensity_image=sample_for_spacing, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, 2 * np.array(target_centroid_wspacing))
    spacing = (1.3, 0.7)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = Mpq(0, 1) / Mpq(0, 0)
    cX = Mpq(1, 0) / Mpq(0, 0)
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, (cX, cY))
    centroid = regionprops(sample_for_spacing, intensity_image=sample_for_spacing, spacing=spacing)[0].centroid_weighted
    assert_almost_equal(centroid, spacing * np.array(target_centroid_wspacing))

def test_moments_weighted_hu():
    if False:
        for i in range(10):
            print('nop')
    whu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted_hu
    ref = np.array([0.31750587329, 0.021417517159, 0.023609322038, 0.001256568336, 8.3014209421e-07, -3.5073773473e-05, -6.7936409056e-06])
    assert_array_almost_equal(whu, ref)
    with testing.raises(NotImplementedError):
        regionprops(SAMPLE, spacing=(2, 1))[0].moments_weighted_hu

def test_moments_weighted():
    if False:
        for i in range(10):
            print('nop')
    wm = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted
    ref = np.array([[74.0, 699.0, 7863.0, 97317.0], [410.0, 3785.0, 44063.0, 572567.0], [2750.0, 24855.0, 293477.0, 3900717.0], [19778.0, 175001.0, 2081051.0, 28078871.0]])
    assert_array_almost_equal(wm, ref)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    assert_almost_equal(Mpq(0, 0), ref[0, 0])
    assert_almost_equal(Mpq(0, 1), ref[0, 1])
    assert_almost_equal(Mpq(0, 2), ref[0, 2])
    assert_almost_equal(Mpq(0, 3), ref[0, 3])
    assert_almost_equal(Mpq(1, 0), ref[1, 0])
    assert_almost_equal(Mpq(1, 1), ref[1, 1])
    assert_almost_equal(Mpq(1, 2), ref[1, 2])
    assert_almost_equal(Mpq(1, 3), ref[1, 3])
    assert_almost_equal(Mpq(2, 0), ref[2, 0])
    assert_almost_equal(Mpq(2, 1), ref[2, 1])
    assert_almost_equal(Mpq(2, 2), ref[2, 2])
    assert_almost_equal(Mpq(2, 3), ref[2, 3])
    assert_almost_equal(Mpq(3, 0), ref[3, 0])
    assert_almost_equal(Mpq(3, 1), ref[3, 1])
    assert_almost_equal(Mpq(3, 2), ref[3, 2])
    assert_almost_equal(Mpq(3, 3), ref[3, 3])

def test_moments_weighted_spacing():
    if False:
        i = 10
        return i + 15
    wm = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted
    ref = np.array([[74.0, 699.0, 7863.0, 97317.0], [410.0, 3785.0, 44063.0, 572567.0], [2750.0, 24855.0, 293477.0, 3900717.0], [19778.0, 175001.0, 2081051.0, 28078871.0]])
    assert_array_almost_equal(wm, ref)
    spacing = (3.2, 1.2)
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].moments_weighted
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    assert_almost_equal(wmu[0, 0], Mpq(0, 0))
    assert_almost_equal(wmu[0, 1], Mpq(0, 1))
    assert_almost_equal(wmu[0, 2], Mpq(0, 2))
    assert_almost_equal(wmu[0, 3], Mpq(0, 3))
    assert_almost_equal(wmu[1, 0], Mpq(1, 0))
    assert_almost_equal(wmu[1, 1], Mpq(1, 1))
    assert_almost_equal(wmu[1, 2], Mpq(1, 2))
    assert_almost_equal(wmu[1, 3], Mpq(1, 3))
    assert_almost_equal(wmu[2, 0], Mpq(2, 0))
    assert_almost_equal(wmu[2, 1], Mpq(2, 1))
    assert_almost_equal(wmu[2, 2], Mpq(2, 2))
    assert_almost_equal(wmu[2, 3], Mpq(2, 3))
    assert_almost_equal(wmu[3, 0], Mpq(3, 0))
    assert_almost_equal(wmu[3, 1], Mpq(3, 1))
    assert_almost_equal(wmu[3, 2], Mpq(3, 2))
    assert_almost_equal(wmu[3, 3], Mpq(3, 3), decimal=6)

def test_moments_weighted_normalized():
    if False:
        i = 10
        return i + 15
    wnu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0].moments_weighted_normalized
    ref = np.array([[np.nan, np.nan, 0.230146783, -0.0162529732], [np.nan, -0.0160405109, 0.0457932622, -0.0104598869], [0.0873590903, -0.0031421072, 0.0165315478, -0.0028544152], [-0.0161217406, -0.0031376984, 0.0043903193, -0.0011057191]])
    assert_array_almost_equal(wnu, ref)

def test_moments_weighted_normalized_spacing():
    if False:
        return 10
    spacing = (3, 3)
    wnu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing)[0].moments_weighted_normalized
    np.array([[np.nan, np.nan, 0.230146783, -0.0162529732], [np.nan, -0.0160405109, 0.0457932622, -0.0104598869], [0.0873590903, -0.0031421072, 0.0165315478, -0.0028544152], [-0.0161217406, -0.0031376984, 0.0043903193, -0.0011057191]])
    assert_almost_equal(wnu[0, 2], 0.230146783)
    assert_almost_equal(wnu[0, 3], -0.0162529732)
    assert_almost_equal(wnu[1, 1], -0.0160405109)
    assert_almost_equal(wnu[1, 2], 0.0457932622)
    assert_almost_equal(wnu[1, 3], -0.0104598869)
    assert_almost_equal(wnu[2, 0], 0.0873590903)
    assert_almost_equal(wnu[2, 1], -0.0031421072)
    assert_almost_equal(wnu[2, 2], 0.0165315478)
    assert_almost_equal(wnu[2, 3], -0.0028544152)
    assert_almost_equal(wnu[3, 0], -0.0161217406)
    assert_almost_equal(wnu[3, 1], -0.0031376984)
    assert_almost_equal(wnu[3, 2], 0.0043903193)
    assert_almost_equal(wnu[3, 3], -0.0011057191)

def test_offset_features():
    if False:
        return 10
    props = regionprops(SAMPLE)[0]
    offset = np.array([1024, 2048])
    props_offset = regionprops(SAMPLE, offset=offset)[0]
    assert_allclose(props.centroid, props_offset.centroid - offset)

def test_label_sequence():
    if False:
        for i in range(10):
            print('nop')
    a = np.empty((2, 2), dtype=int)
    a[:, :] = 2
    ps = regionprops(a)
    assert len(ps) == 1
    assert ps[0].label == 2

def test_pure_background():
    if False:
        for i in range(10):
            print('nop')
    a = np.zeros((2, 2), dtype=int)
    ps = regionprops(a)
    assert len(ps) == 0

def test_invalid():
    if False:
        i = 10
        return i + 15
    ps = regionprops(SAMPLE)

    def get_intensity_image():
        if False:
            for i in range(10):
                print('nop')
        ps[0].image_intensity
    with pytest.raises(AttributeError):
        get_intensity_image()

def test_invalid_size():
    if False:
        print('Hello World!')
    wrong_intensity_sample = np.array([[1], [1]])
    with pytest.raises(ValueError):
        regionprops(SAMPLE, wrong_intensity_sample)

def test_equals():
    if False:
        i = 10
        return i + 15
    arr = np.zeros((100, 100), dtype=int)
    arr[0:25, 0:25] = 1
    arr[50:99, 50:99] = 2
    regions = regionprops(arr)
    r1 = regions[0]
    regions = regionprops(arr)
    r2 = regions[0]
    r3 = regions[1]
    assert_equal(r1 == r2, True, 'Same regionprops are not equal')
    assert_equal(r1 != r3, True, 'Different regionprops are equal')

def test_iterate_all_props():
    if False:
        while True:
            i = 10
    region = regionprops(SAMPLE)[0]
    p0 = {p: region[p] for p in region}
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    p1 = {p: region[p] for p in region}
    assert len(p0) < len(p1)

def test_cache():
    if False:
        for i in range(10):
            print('nop')
    SAMPLE_mod = SAMPLE.copy()
    region = regionprops(SAMPLE_mod)[0]
    f0 = region.image_filled
    region._label_image[:10] = 1
    f1 = region.image_filled
    assert_array_equal(f0, f1)
    region._cache_active = False
    f1 = region.image_filled
    assert np.any(f0 != f1)

def test_docstrings_and_props():
    if False:
        i = 10
        return i + 15

    def foo():
        if False:
            i = 10
            return i + 15
        'foo'
    has_docstrings = bool(foo.__doc__)
    region = regionprops(SAMPLE)[0]
    docs = _parse_docs()
    props = [m for m in dir(region) if not m.startswith('_')]
    nr_docs_parsed = len(docs)
    nr_props = len(props)
    if has_docstrings:
        assert_equal(nr_docs_parsed, nr_props)
        ds = docs['moments_weighted_normalized']
        assert 'iteration' not in ds
        assert len(ds.split('\n')) > 3
    else:
        assert_equal(nr_docs_parsed, 0)

def test_props_to_dict():
    if False:
        while True:
            i = 10
    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions)
    assert out == {'label': np.array([1]), 'bbox-0': np.array([0]), 'bbox-1': np.array([0]), 'bbox-2': np.array([10]), 'bbox-3': np.array([18])}
    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions, properties=('label', 'area', 'bbox'), separator='+')
    assert out == {'label': np.array([1]), 'area': np.array([72]), 'bbox+0': np.array([0]), 'bbox+1': np.array([0]), 'bbox+2': np.array([10]), 'bbox+3': np.array([18])}

def test_regionprops_table():
    if False:
        print('Hello World!')
    out = regionprops_table(SAMPLE)
    assert out == {'label': np.array([1]), 'bbox-0': np.array([0]), 'bbox-1': np.array([0]), 'bbox-2': np.array([10]), 'bbox-3': np.array([18])}
    out = regionprops_table(SAMPLE, properties=('label', 'area', 'bbox'), separator='+')
    assert out == {'label': np.array([1]), 'area': np.array([72]), 'bbox+0': np.array([0]), 'bbox+1': np.array([0]), 'bbox+2': np.array([10]), 'bbox+3': np.array([18])}

def test_regionprops_table_deprecated_vector_property():
    if False:
        return 10
    out = regionprops_table(SAMPLE, properties=('local_centroid',))
    for key in out.keys():
        assert key.startswith('local_centroid')

def test_regionprops_table_deprecated_scalar_property():
    if False:
        print('Hello World!')
    out = regionprops_table(SAMPLE, properties=('bbox_area',))
    assert list(out.keys()) == ['bbox_area']

def test_regionprops_table_equal_to_original():
    if False:
        i = 10
        return i + 15
    regions = regionprops(SAMPLE, INTENSITY_FLOAT_SAMPLE)
    out_table = regionprops_table(SAMPLE, INTENSITY_FLOAT_SAMPLE, properties=COL_DTYPES.keys())
    for (prop, dtype) in COL_DTYPES.items():
        for (i, reg) in enumerate(regions):
            rp = reg[prop]
            if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
                assert_array_equal(rp, out_table[prop][i])
            else:
                shape = rp.shape if isinstance(rp, np.ndarray) else (len(rp),)
                for ind in np.ndindex(shape):
                    modified_prop = '-'.join(map(str, (prop,) + ind))
                    loc = ind if len(ind) > 1 else ind[0]
                    assert_equal(rp[loc], out_table[modified_prop][i])

def test_regionprops_table_no_regions():
    if False:
        i = 10
        return i + 15
    out = regionprops_table(np.zeros((2, 2), dtype=int), properties=('label', 'area', 'bbox'), separator='+')
    assert len(out) == 6
    assert len(out['label']) == 0
    assert len(out['area']) == 0
    assert len(out['bbox+0']) == 0
    assert len(out['bbox+1']) == 0
    assert len(out['bbox+2']) == 0
    assert len(out['bbox+3']) == 0

def test_column_dtypes_correct():
    if False:
        print('Hello World!')
    msg = 'mismatch with expected type,'
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    for col in COL_DTYPES:
        r = region[col]
        if col in OBJECT_COLUMNS:
            assert COL_DTYPES[col] == object
            continue
        t = type(np.ravel(r)[0])
        if np.issubdtype(t, np.floating):
            assert COL_DTYPES[col] == float, f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
        elif np.issubdtype(t, np.integer):
            assert COL_DTYPES[col] == int, f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
        else:
            assert False, f'{col} dtype {t} {msg} {COL_DTYPES[col]}'

def test_all_documented_items_in_col_dtypes():
    if False:
        while True:
            i = 10
    docstring = numpydoc.docscrape.FunctionDoc(regionprops)
    notes_lines = docstring['Notes']
    property_lines = filter(lambda line: line.startswith('**'), notes_lines)
    pattern = '\\*\\*(?P<property_name>[a-z_]+)\\*\\*.*'
    property_names = {re.search(pattern, property_line).group('property_name') for property_line in property_lines}
    column_keys = set(COL_DTYPES.keys())
    assert column_keys == property_names

def pixelcount(regionmask):
    if False:
        while True:
            i = 10
    'a short test for an extra property'
    return np.sum(regionmask)

def intensity_median(regionmask, image_intensity):
    if False:
        i = 10
        return i + 15
    return np.median(image_intensity[regionmask])

def bbox_list(regionmask):
    if False:
        print('Hello World!')
    'Extra property whose output shape is dependent on mask shape.'
    return [1] * regionmask.shape[1]

def too_many_args(regionmask, image_intensity, superfluous):
    if False:
        for i in range(10):
            print('nop')
    return 1

def too_few_args():
    if False:
        print('Hello World!')
    return 1

def test_extra_properties():
    if False:
        return 10
    region = regionprops(SAMPLE, extra_properties=(pixelcount,))[0]
    assert region.pixelcount == np.sum(SAMPLE == 1)

def test_extra_properties_intensity():
    if False:
        return 10
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, extra_properties=(intensity_median,))[0]
    assert region.intensity_median == np.median(INTENSITY_SAMPLE[SAMPLE == 1])

@pytest.mark.parametrize('intensity_prop', _require_intensity_image)
def test_intensity_image_required(intensity_prop):
    if False:
        for i in range(10):
            print('nop')
    region = regionprops(SAMPLE)[0]
    with pytest.raises(AttributeError) as e:
        getattr(region, intensity_prop)
    expected_error = f"Attribute '{intensity_prop}' unavailable when `intensity_image` has not been specified."
    assert expected_error == str(e.value)

def test_extra_properties_no_intensity_provided():
    if False:
        return 10
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(intensity_median,))[0]
        _ = region.intensity_median

def test_extra_properties_nr_args():
    if False:
        return 10
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_few_args,))[0]
        _ = region.too_few_args
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_many_args,))[0]
        _ = region.too_many_args

def test_extra_properties_mixed():
    if False:
        print('Hello World!')
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE, extra_properties=(intensity_median, pixelcount))[0]
    assert region.intensity_median == np.median(INTENSITY_SAMPLE[SAMPLE == 1])
    assert region.pixelcount == np.sum(SAMPLE == 1)

def test_extra_properties_table():
    if False:
        for i in range(10):
            print('nop')
    out = regionprops_table(SAMPLE_MULTIPLE, intensity_image=INTENSITY_SAMPLE_MULTIPLE, properties=('label',), extra_properties=(intensity_median, pixelcount, bbox_list))
    assert_array_almost_equal(out['intensity_median'], np.array([2.0, 4.0]))
    assert_array_equal(out['pixelcount'], np.array([10, 2]))
    assert out['bbox_list'].dtype == np.object_
    assert out['bbox_list'][0] == [1] * 10
    assert out['bbox_list'][1] == [1] * 1

def test_multichannel():
    if False:
        print('Hello World!')
    'Test that computing multichannel properties works.'
    astro = data.astronaut()[::4, ::4]
    astro_green = astro[..., 1]
    labels = slic(astro.astype(float), start_label=1)
    segment_idx = np.max(labels) // 2
    region = regionprops(labels, astro_green, extra_properties=[intensity_median])[segment_idx]
    region_multi = regionprops(labels, astro, extra_properties=[intensity_median])[segment_idx]
    for prop in list(PROPS.keys()) + ['intensity_median']:
        p = region[prop]
        p_multi = region_multi[prop]
        if np.shape(p) == np.shape(p_multi):
            assert_array_equal(p, p_multi)
        else:
            assert_allclose(p, np.asarray(p_multi)[..., 1], rtol=1e-12, atol=1e-12)

def test_3d_ellipsoid_axis_lengths():
    if False:
        print('Hello World!')
    'Verify that estimated axis lengths are correct.\n\n    Uses an ellipsoid at an arbitrary position and orientation.\n    '
    half_lengths = (20, 10, 50)
    e = draw.ellipsoid(*half_lengths).astype(int)
    e = np.pad(e, pad_width=[(30, 18), (30, 12), (40, 20)], mode='constant')
    R = transform.EuclideanTransform(rotation=[0.2, 0.3, 0.4], dimensionality=3)
    e = ndi.affine_transform(e, R.params)
    rp = regionprops(e)[0]
    evs = rp.inertia_tensor_eigvals
    axis_lengths = _inertia_eigvals_to_axes_lengths_3D(evs)
    expected_lengths = sorted([2 * h for h in half_lengths], reverse=True)
    for (ax_len_expected, ax_len) in zip(expected_lengths, axis_lengths):
        assert abs(ax_len - ax_len_expected) < 0.01 * ax_len_expected
    assert abs(rp.axis_major_length - axis_lengths[0]) < 1e-07
    assert abs(rp.axis_minor_length - axis_lengths[-1]) < 1e-07