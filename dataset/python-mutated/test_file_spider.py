import tempfile
import warnings
from io import BytesIO
import pytest
from PIL import Image, ImageSequence, SpiderImagePlugin
from .helper import assert_image_equal_tofile, hopper, is_pypy
TEST_FILE = 'Tests/images/hopper.spider'

def test_sanity():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(TEST_FILE) as im:
        im.load()
        assert im.mode == 'F'
        assert im.size == (128, 128)
        assert im.format == 'SPIDER'

@pytest.mark.skipif(is_pypy(), reason='Requires CPython')
def test_unclosed_file():
    if False:
        while True:
            i = 10

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
        return 10
    with warnings.catch_warnings():
        with Image.open(TEST_FILE) as im:
            im.load()

def test_save(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    temp = str(tmp_path / 'temp.spider')
    im = hopper()
    im.save(temp, 'SPIDER')
    with Image.open(temp) as im2:
        assert im2.mode == 'F'
        assert im2.size == (128, 128)
        assert im2.format == 'SPIDER'

def test_tempfile():
    if False:
        for i in range(10):
            print('nop')
    im = hopper()
    with tempfile.TemporaryFile() as fp:
        im.save(fp, 'SPIDER')
        fp.seek(0)
        with Image.open(fp) as reloaded:
            assert reloaded.mode == 'F'
            assert reloaded.size == (128, 128)
            assert reloaded.format == 'SPIDER'

def test_is_spider_image():
    if False:
        while True:
            i = 10
    assert SpiderImagePlugin.isSpiderImage(TEST_FILE)

def test_tell():
    if False:
        while True:
            i = 10
    with Image.open(TEST_FILE) as im:
        index = im.tell()
        assert index == 0

def test_n_frames():
    if False:
        i = 10
        return i + 15
    with Image.open(TEST_FILE) as im:
        assert im.n_frames == 1
        assert not im.is_animated

def test_load_image_series():
    if False:
        return 10
    not_spider_file = 'Tests/images/hopper.ppm'
    file_list = [TEST_FILE, not_spider_file, 'path/not_found.ext']
    img_list = SpiderImagePlugin.loadImageSeries(file_list)
    assert len(img_list) == 1
    assert isinstance(img_list[0], Image.Image)
    assert img_list[0].size == (128, 128)

def test_load_image_series_no_input():
    if False:
        print('Hello World!')
    file_list = None
    img_list = SpiderImagePlugin.loadImageSeries(file_list)
    assert img_list is None

def test_is_int_not_a_number():
    if False:
        print('Hello World!')
    not_a_number = 'a'
    ret = SpiderImagePlugin.isInt(not_a_number)
    assert ret == 0

def test_invalid_file():
    if False:
        for i in range(10):
            print('nop')
    invalid_file = 'Tests/images/invalid.spider'
    with pytest.raises(OSError):
        with Image.open(invalid_file):
            pass

def test_nonstack_file():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(TEST_FILE) as im:
        with pytest.raises(EOFError):
            im.seek(0)

def test_nonstack_dos():
    if False:
        print('Hello World!')
    with Image.open(TEST_FILE) as im:
        for (i, frame) in enumerate(ImageSequence.Iterator(im)):
            assert i <= 1, 'Non-stack DOS file test failed'

def test_odd_size():
    if False:
        print('Hello World!')
    data = BytesIO()
    width = 100
    im = Image.new('F', (width, 64))
    im.save(data, format='SPIDER')
    data.seek(0)
    assert_image_equal_tofile(im, data)