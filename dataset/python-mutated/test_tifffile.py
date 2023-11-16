import pathlib
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from skimage._shared.testing import fetch
from skimage.io import imread, imsave, reset_plugins, use_plugin

def setup():
    if False:
        return 10
    use_plugin('tifffile')
    np.random.seed(0)

def teardown():
    if False:
        for i in range(10):
            print('nop')
    reset_plugins()

def test_imread_uint16():
    if False:
        for i in range(10):
            print('nop')
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)

def test_imread_uint16_big_endian():
    if False:
        return 10
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)

def test_imread_multipage_rgb_tif():
    if False:
        return 10
    img = imread(fetch('data/multipage_rgb.tif'))
    assert img.shape == (2, 10, 10, 3), img.shape

def test_tifffile_kwarg_passthrough():
    if False:
        for i in range(10):
            print('nop')
    img = imread(fetch('data/multipage.tif'), key=[1], is_ome=True)
    assert img.shape == (15, 10), img.shape

def test_imread_handle():
    if False:
        return 10
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    with open(fetch('data/chessboard_GRAY_U16.tif'), 'rb') as fh:
        img = imread(fh)
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)

class TestSave:

    def roundtrip(self, dtype, x, use_pathlib=False, **kwargs):
        if False:
            print('Hello World!')
        with NamedTemporaryFile(suffix='.tif') as f:
            fname = f.name
        if use_pathlib:
            fname = pathlib.Path(fname)
        imsave(fname, x, check_contrast=False, **kwargs)
        y = imread(fname)
        assert_array_equal(x, y)
    shapes = ((10, 10), (10, 10, 3), (10, 10, 4))
    dtypes = (np.uint8, np.uint16, np.float32, np.int16, np.float64)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('use_pathlib', [False, True])
    @pytest.mark.parametrize('explicit_photometric_kwarg', [False, True])
    def test_imsave_roundtrip(self, shape, dtype, use_pathlib, explicit_photometric_kwarg):
        if False:
            return 10
        x = np.random.rand(*shape)
        if not np.issubdtype(dtype, np.floating):
            x = (x * np.iinfo(dtype).max).astype(dtype)
        else:
            x = x.astype(dtype)
        if explicit_photometric_kwarg and x.shape[-1] in [3, 4]:
            kwargs = {'photometric': 'rgb'}
        else:
            kwargs = {}
        self.roundtrip(dtype, x, use_pathlib, **kwargs)