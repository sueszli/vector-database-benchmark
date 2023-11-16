import pytest
from PIL import GbrImagePlugin, Image
from .helper import assert_image_equal_tofile

def test_gbr_file():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/gbr.gbr') as im:
        assert_image_equal_tofile(im, 'Tests/images/gbr.png')

def test_load():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/gbr.gbr') as im:
        assert im.load()[0, 0] == (0, 0, 0, 0)
        assert im.load()[0, 0] == (0, 0, 0, 0)

def test_multiple_load_operations():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/gbr.gbr') as im:
        im.load()
        im.load()
        assert_image_equal_tofile(im, 'Tests/images/gbr.png')

def test_invalid_file():
    if False:
        i = 10
        return i + 15
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        GbrImagePlugin.GbrImageFile(invalid_file)