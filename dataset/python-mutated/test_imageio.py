from tempfile import NamedTemporaryFile
import numpy as np
from skimage.io import imread, imsave, plugin_order
from skimage._shared import testing
from skimage._shared.testing import fetch
import pytest

def test_prefered_plugin():
    if False:
        i = 10
        return i + 15
    order = plugin_order()
    assert order['imread'][0] == 'imageio'
    assert order['imsave'][0] == 'imageio'
    assert order['imread_collection'][0] == 'imageio'

def test_imageio_as_gray():
    if False:
        i = 10
        return i + 15
    img = imread(fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(fetch('data/camera.png'), as_gray=True)
    assert np.core.numerictypes.sctype2char(img.dtype) in np.typecodes['AllInteger']

def test_imageio_palette():
    if False:
        i = 10
        return i + 15
    img = imread(fetch('data/palette_color.png'))
    assert img.ndim == 3

def test_imageio_truncated_jpg():
    if False:
        while True:
            i = 10
    with testing.raises((OSError, SyntaxError)):
        imread(fetch('data/truncated.jpg'))

class TestSave:

    @pytest.mark.parametrize('shape,dtype', [((10, 10), np.uint8), ((10, 10), np.uint16), ((10, 10, 2), np.uint8), ((10, 10, 3), np.uint8), ((10, 10, 4), np.uint8)])
    def test_imsave_roundtrip(self, shape, dtype, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        if np.issubdtype(dtype, np.floating):
            min_ = 0
            max_ = 1
        else:
            min_ = 0
            max_ = np.iinfo(dtype).max
        expected = np.linspace(min_, max_, endpoint=True, num=np.prod(shape), dtype=dtype)
        expected = expected.reshape(shape)
        file_path = tmp_path / 'roundtrip.png'
        imsave(file_path, expected)
        actual = imread(file_path)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_bool_array_save(self):
        if False:
            return 10
        with NamedTemporaryFile(suffix='.png') as f:
            fname = f.name
        with pytest.warns(UserWarning, match='.* is a boolean image'):
            a = np.zeros((5, 5), bool)
            a[2, 2] = True
            imsave(fname, a)

def test_return_class():
    if False:
        for i in range(10):
            print('nop')
    testing.assert_equal(type(imread(fetch('data/color.png'))), np.ndarray)