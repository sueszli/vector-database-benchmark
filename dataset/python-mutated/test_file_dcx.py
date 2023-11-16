import warnings
import pytest
from PIL import DcxImagePlugin, Image
from .helper import assert_image_equal, hopper, is_pypy
TEST_FILE = 'Tests/images/hopper.dcx'

def test_sanity():
    if False:
        return 10
    with Image.open(TEST_FILE) as im:
        assert im.size == (128, 128)
        assert isinstance(im, DcxImagePlugin.DcxImageFile)
        orig = hopper()
        assert_image_equal(im, orig)

@pytest.mark.skipif(is_pypy(), reason='Requires CPython')
def test_unclosed_file():
    if False:
        i = 10
        return i + 15

    def open():
        if False:
            while True:
                i = 10
        im = Image.open(TEST_FILE)
        im.load()
    with pytest.warns(ResourceWarning):
        open()

def test_closed_file():
    if False:
        i = 10
        return i + 15
    with warnings.catch_warnings():
        im = Image.open(TEST_FILE)
        im.load()
        im.close()

def test_context_manager():
    if False:
        while True:
            i = 10
    with warnings.catch_warnings():
        with Image.open(TEST_FILE) as im:
            im.load()

def test_invalid_file():
    if False:
        for i in range(10):
            print('nop')
    with open('Tests/images/flower.jpg', 'rb') as fp:
        with pytest.raises(SyntaxError):
            DcxImagePlugin.DcxImageFile(fp)

def test_tell():
    if False:
        print('Hello World!')
    with Image.open(TEST_FILE) as im:
        frame = im.tell()
        assert frame == 0

def test_n_frames():
    if False:
        i = 10
        return i + 15
    with Image.open(TEST_FILE) as im:
        assert im.n_frames == 1
        assert not im.is_animated

def test_eoferror():
    if False:
        while True:
            i = 10
    with Image.open(TEST_FILE) as im:
        n_frames = im.n_frames
        with pytest.raises(EOFError):
            im.seek(n_frames)
        assert im.tell() < n_frames
        im.seek(n_frames - 1)

def test_seek_too_far():
    if False:
        return 10
    with Image.open(TEST_FILE) as im:
        frame = 999
    with pytest.raises(EOFError):
        im.seek(frame)