import warnings
import pytest
from PIL import ImageQt
from .helper import assert_image_similar, hopper
pytestmark = pytest.mark.skipif(not ImageQt.qt_is_installed, reason='Qt bindings are not installed')
if ImageQt.qt_is_installed:
    from PIL.ImageQt import qRgba

def test_rgb():
    if False:
        while True:
            i = 10
    if ImageQt.qt_version == '6':
        from PyQt6.QtGui import qRgb
    elif ImageQt.qt_version == 'side6':
        from PySide6.QtGui import qRgb
    assert qRgb(0, 0, 0) == qRgba(0, 0, 0, 255)

    def checkrgb(r, g, b):
        if False:
            for i in range(10):
                print('nop')
        val = ImageQt.rgb(r, g, b)
        val = val % 2 ** 24
        assert val >> 16 == r
        assert (val >> 8) % 2 ** 8 == g
        assert val % 2 ** 8 == b
    checkrgb(0, 0, 0)
    checkrgb(255, 0, 0)
    checkrgb(0, 255, 0)
    checkrgb(0, 0, 255)

def test_image():
    if False:
        print('Hello World!')
    modes = ['1', 'RGB', 'RGBA', 'L', 'P']
    qt_format = ImageQt.QImage.Format if ImageQt.qt_version == '6' else ImageQt.QImage
    if hasattr(qt_format, 'Format_Grayscale16'):
        modes.append('I;16')
    for mode in modes:
        im = hopper(mode)
        roundtripped_im = ImageQt.fromqimage(ImageQt.ImageQt(im))
        if mode not in ('RGB', 'RGBA'):
            im = im.convert('RGB')
        assert_image_similar(roundtripped_im, im, 1)

def test_closed_file():
    if False:
        return 10
    with warnings.catch_warnings():
        ImageQt.ImageQt('Tests/images/hopper.gif')