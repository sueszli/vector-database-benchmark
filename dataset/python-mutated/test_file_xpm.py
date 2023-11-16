import pytest
from PIL import Image, XpmImagePlugin
from .helper import assert_image_similar, hopper
TEST_FILE = 'Tests/images/hopper.xpm'

def test_sanity():
    if False:
        i = 10
        return i + 15
    with Image.open(TEST_FILE) as im:
        im.load()
        assert im.mode == 'P'
        assert im.size == (128, 128)
        assert im.format == 'XPM'
        assert_image_similar(im.convert('RGB'), hopper('RGB'), 60)

def test_invalid_file():
    if False:
        return 10
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        XpmImagePlugin.XpmImageFile(invalid_file)

def test_load_read():
    if False:
        return 10
    with Image.open(TEST_FILE) as im:
        dummy_bytes = 1
        data = im.load_read(dummy_bytes)
    assert len(data) == 16384