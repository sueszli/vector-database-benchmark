from io import BytesIO
import pytest
from PIL import FitsImagePlugin, Image
from .helper import assert_image_equal, hopper
TEST_FILE = 'Tests/images/hopper.fits'

def test_open():
    if False:
        return 10
    with Image.open(TEST_FILE) as im:
        assert im.format == 'FITS'
        assert im.size == (128, 128)
        assert im.mode == 'L'
        assert_image_equal(im, hopper('L'))

def test_invalid_file():
    if False:
        for i in range(10):
            print('nop')
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        FitsImagePlugin.FitsImageFile(invalid_file)

def test_truncated_fits():
    if False:
        return 10
    image_data = b'SIMPLE  =                    T' + b' ' * 50 + b'TRUNCATE'
    with pytest.raises(OSError):
        FitsImagePlugin.FitsImageFile(BytesIO(image_data))

def test_naxis_zero():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        with Image.open('Tests/images/hopper_naxis_zero.fits'):
            pass

def test_comment():
    if False:
        print('Hello World!')
    image_data = b'SIMPLE  =                    T / comment string'
    with pytest.raises(OSError):
        FitsImagePlugin.FitsImageFile(BytesIO(image_data))