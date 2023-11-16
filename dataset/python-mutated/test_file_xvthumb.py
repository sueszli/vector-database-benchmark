import pytest
from PIL import Image, XVThumbImagePlugin
from .helper import assert_image_similar, hopper
TEST_FILE = 'Tests/images/hopper.p7'

def test_open():
    if False:
        return 10
    with Image.open(TEST_FILE) as im:
        assert im.format == 'XVThumb'
        im_hopper = hopper().quantize(palette=im)
        assert_image_similar(im, im_hopper, 9)

def test_unexpected_eof():
    if False:
        i = 10
        return i + 15
    bad_file = 'Tests/images/hopper_bad.p7'
    with pytest.raises(SyntaxError):
        XVThumbImagePlugin.XVThumbImageFile(bad_file)

def test_invalid_file():
    if False:
        print('Hello World!')
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        XVThumbImagePlugin.XVThumbImageFile(invalid_file)