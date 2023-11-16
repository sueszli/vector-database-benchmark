from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
from skimage.io import imread, imsave, use_plugin, reset_plugins
from skimage._shared import testing
from skimage._shared.testing import TestCase, assert_array_equal, assert_array_almost_equal, fetch
from pytest import importorskip
importorskip('imread')

def setup():
    if False:
        for i in range(10):
            print('nop')
    use_plugin('imread')

def teardown():
    if False:
        i = 10
        return i + 15
    reset_plugins()

def test_imread_as_gray():
    if False:
        while True:
            i = 10
    img = imread(fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(fetch('data/camera.png'), as_gray=True)
    assert np.core.numerictypes.sctype2char(img.dtype) in np.typecodes['AllInteger']

def test_imread_palette():
    if False:
        while True:
            i = 10
    img = imread(fetch('data/palette_color.png'))
    assert img.ndim == 3

def test_imread_truncated_jpg():
    if False:
        print('Hello World!')
    with testing.raises(RuntimeError):
        io.imread(fetch('data/truncated.jpg'))

def test_bilevel():
    if False:
        return 10
    expected = np.zeros((10, 10), bool)
    expected[::2] = 1
    img = imread(fetch('data/checker_bilevel.png'))
    assert_array_equal(img.astype(bool), expected)

class TestSave(TestCase):

    def roundtrip(self, x, scaling=1):
        if False:
            while True:
                i = 10
        with NamedTemporaryFile(suffix='.png') as f:
            fname = f.name
        imsave(fname, x)
        y = imread(fname)
        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    def test_imsave_roundtrip(self):
        if False:
            print('Hello World!')
        dtype = np.uint8
        np.random.seed(0)
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            x = np.ones(shape, dtype=dtype) * np.random.rand(*shape)
            if np.issubdtype(dtype, np.floating):
                yield (self.roundtrip, x, 255)
            else:
                x = (x * 255).astype(dtype)
                yield (self.roundtrip, x)