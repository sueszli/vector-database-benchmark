import pytest
from PIL import Image, McIdasImagePlugin
from .helper import assert_image_equal_tofile

def test_invalid_file():
    if False:
        while True:
            i = 10
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        McIdasImagePlugin.McIdasImageFile(invalid_file)

def test_valid_file():
    if False:
        i = 10
        return i + 15
    test_file = 'Tests/images/cmx3g8_wv_1998.260_0745_mcidas.ara'
    saved_file = 'Tests/images/cmx3g8_wv_1998.260_0745_mcidas.png'
    with Image.open(test_file) as im:
        im.load()
        assert im.format == 'MCIDAS'
        assert im.mode == 'I'
        assert im.size == (1800, 400)
        assert_image_equal_tofile(im, saved_file)