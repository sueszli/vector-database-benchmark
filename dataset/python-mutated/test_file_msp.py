import os
import pytest
from PIL import Image, MspImagePlugin
from .helper import assert_image_equal, assert_image_equal_tofile, hopper
TEST_FILE = 'Tests/images/hopper.msp'
EXTRA_DIR = 'Tests/images/picins'
YA_EXTRA_DIR = 'Tests/images/msp'

def test_sanity(tmp_path):
    if False:
        i = 10
        return i + 15
    test_file = str(tmp_path / 'temp.msp')
    hopper('1').save(test_file)
    with Image.open(test_file) as im:
        im.load()
        assert im.mode == '1'
        assert im.size == (128, 128)
        assert im.format == 'MSP'

def test_invalid_file():
    if False:
        for i in range(10):
            print('nop')
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        MspImagePlugin.MspImageFile(invalid_file)

def test_bad_checksum():
    if False:
        while True:
            i = 10
    bad_checksum = 'Tests/images/hopper_bad_checksum.msp'
    with pytest.raises(SyntaxError):
        MspImagePlugin.MspImageFile(bad_checksum)

def test_open_windows_v1():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(TEST_FILE) as im:
        assert_image_equal(im, hopper('1'))
        assert isinstance(im, MspImagePlugin.MspImageFile)

def _assert_file_image_equal(source_path, target_path):
    if False:
        i = 10
        return i + 15
    with Image.open(source_path) as im:
        assert_image_equal_tofile(im, target_path)

@pytest.mark.skipif(not os.path.exists(EXTRA_DIR), reason='Extra image files not installed')
def test_open_windows_v2():
    if False:
        return 10
    files = (os.path.join(EXTRA_DIR, f) for f in os.listdir(EXTRA_DIR) if os.path.splitext(f)[1] == '.msp')
    for path in files:
        _assert_file_image_equal(path, path.replace('.msp', '.png'))

@pytest.mark.skipif(not os.path.exists(YA_EXTRA_DIR), reason='Even More Extra image files not installed')
def test_msp_v2():
    if False:
        while True:
            i = 10
    for f in os.listdir(YA_EXTRA_DIR):
        if '.MSP' not in f:
            continue
        path = os.path.join(YA_EXTRA_DIR, f)
        _assert_file_image_equal(path, path.replace('.MSP', '.png'))

def test_cannot_save_wrong_mode(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    im = hopper()
    filename = str(tmp_path / 'temp.msp')
    with pytest.raises(OSError):
        im.save(filename)