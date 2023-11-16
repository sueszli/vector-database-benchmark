import os
import pytest
from PIL import Image, SunImagePlugin
from .helper import assert_image_equal_tofile, assert_image_similar, hopper
EXTRA_DIR = 'Tests/images/sunraster'

def test_sanity():
    if False:
        i = 10
        return i + 15
    test_file = 'Tests/images/hopper.ras'
    with Image.open(test_file) as im:
        assert im.size == (128, 128)
        assert_image_similar(im, hopper(), 5)
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        SunImagePlugin.SunImageFile(invalid_file)

def test_im1():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/sunraster.im1') as im:
        assert_image_equal_tofile(im, 'Tests/images/sunraster.im1.png')

@pytest.mark.skipif(not os.path.exists(EXTRA_DIR), reason='Extra image files not installed')
def test_others():
    if False:
        i = 10
        return i + 15
    files = (os.path.join(EXTRA_DIR, f) for f in os.listdir(EXTRA_DIR) if os.path.splitext(f)[1] in ('.sun', '.SUN', '.ras'))
    for path in files:
        with Image.open(path) as im:
            im.load()
            assert isinstance(im, SunImagePlugin.SunImageFile)
            assert_image_equal_tofile(im, f'{os.path.splitext(path)[0]}.png')