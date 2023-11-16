import warnings
import pytest
from PIL import Image
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=DeprecationWarning)
    from PIL import ImageQt
from .helper import assert_image_equal, hopper
pytestmark = pytest.mark.skipif(not ImageQt.qt_is_installed, reason='Qt bindings are not installed')

@pytest.fixture
def test_images():
    if False:
        print('Hello World!')
    ims = [hopper(), Image.open('Tests/images/transparent.png'), Image.open('Tests/images/7x13.png')]
    try:
        yield ims
    finally:
        for im in ims:
            im.close()

def roundtrip(expected):
    if False:
        return 10
    intermediate = expected.toqimage()
    result = ImageQt.fromqimage(intermediate)
    if intermediate.hasAlphaChannel():
        assert_image_equal(result, expected.convert('RGBA'))
    else:
        assert_image_equal(result, expected.convert('RGB'))

def test_sanity_1(test_images):
    if False:
        return 10
    for im in test_images:
        roundtrip(im.convert('1'))

def test_sanity_rgb(test_images):
    if False:
        for i in range(10):
            print('nop')
    for im in test_images:
        roundtrip(im.convert('RGB'))

def test_sanity_rgba(test_images):
    if False:
        i = 10
        return i + 15
    for im in test_images:
        roundtrip(im.convert('RGBA'))

def test_sanity_l(test_images):
    if False:
        return 10
    for im in test_images:
        roundtrip(im.convert('L'))

def test_sanity_p(test_images):
    if False:
        for i in range(10):
            print('nop')
    for im in test_images:
        roundtrip(im.convert('P'))