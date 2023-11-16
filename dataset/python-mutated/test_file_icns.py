import io
import os
import warnings
import pytest
from PIL import IcnsImagePlugin, Image, _binary
from .helper import assert_image_equal, assert_image_similar_tofile, skip_unless_feature
TEST_FILE = 'Tests/images/pillow.icns'

def test_sanity():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(TEST_FILE) as im:
        with warnings.catch_warnings():
            im.load()
        assert im.mode == 'RGBA'
        assert im.size == (1024, 1024)
        assert im.format == 'ICNS'

def test_load():
    if False:
        print('Hello World!')
    with Image.open(TEST_FILE) as im:
        assert im.load()[0, 0] == (0, 0, 0, 0)
        assert im.load()[0, 0] == (0, 0, 0, 0)

def test_save(tmp_path):
    if False:
        return 10
    temp_file = str(tmp_path / 'temp.icns')
    with Image.open(TEST_FILE) as im:
        im.save(temp_file)
    with Image.open(temp_file) as reread:
        assert reread.mode == 'RGBA'
        assert reread.size == (1024, 1024)
        assert reread.format == 'ICNS'
    file_length = os.path.getsize(temp_file)
    with open(temp_file, 'rb') as fp:
        fp.seek(4)
        assert _binary.i32be(fp.read(4)) == file_length

def test_save_append_images(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    temp_file = str(tmp_path / 'temp.icns')
    provided_im = Image.new('RGBA', (32, 32), (255, 0, 0, 128))
    with Image.open(TEST_FILE) as im:
        im.save(temp_file, append_images=[provided_im])
        assert_image_similar_tofile(im, temp_file, 1)
        with Image.open(temp_file) as reread:
            reread.size = (16, 16, 2)
            reread.load()
            assert_image_equal(reread, provided_im)

def test_save_fp():
    if False:
        i = 10
        return i + 15
    fp = io.BytesIO()
    with Image.open(TEST_FILE) as im:
        im.save(fp, format='ICNS')
    with Image.open(fp) as reread:
        assert reread.mode == 'RGBA'
        assert reread.size == (1024, 1024)
        assert reread.format == 'ICNS'

def test_sizes():
    if False:
        i = 10
        return i + 15
    with Image.open(TEST_FILE) as im:
        for (w, h, r) in im.info['sizes']:
            wr = w * r
            hr = h * r
            im.size = (w, h, r)
            im.load()
            assert im.mode == 'RGBA'
            assert im.size == (wr, hr)
        with pytest.raises(ValueError):
            im.size = (1, 1)

def test_older_icon():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/pillow2.icns') as im:
        for (w, h, r) in im.info['sizes']:
            wr = w * r
            hr = h * r
            with Image.open('Tests/images/pillow2.icns') as im2:
                im2.size = (w, h, r)
                im2.load()
                assert im2.mode == 'RGBA'
                assert im2.size == (wr, hr)

@skip_unless_feature('jpg_2000')
def test_jp2_icon():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/pillow3.icns') as im:
        for (w, h, r) in im.info['sizes']:
            wr = w * r
            hr = h * r
            with Image.open('Tests/images/pillow3.icns') as im2:
                im2.size = (w, h, r)
                im2.load()
                assert im2.mode == 'RGBA'
                assert im2.size == (wr, hr)

def test_getimage():
    if False:
        for i in range(10):
            print('nop')
    with open(TEST_FILE, 'rb') as fp:
        icns_file = IcnsImagePlugin.IcnsFile(fp)
        im = icns_file.getimage()
        assert im.mode == 'RGBA'
        assert im.size == (1024, 1024)
        im = icns_file.getimage((512, 512))
        assert im.mode == 'RGBA'
        assert im.size == (512, 512)

def test_not_an_icns_file():
    if False:
        print('Hello World!')
    with io.BytesIO(b'invalid\n') as fp:
        with pytest.raises(SyntaxError):
            IcnsImagePlugin.IcnsFile(fp)

@skip_unless_feature('jpg_2000')
def test_icns_decompression_bomb():
    if False:
        return 10
    with Image.open('Tests/images/oom-8ed3316a4109213ca96fb8a256a0bfefdece1461.icns') as im:
        with pytest.raises(Image.DecompressionBombError):
            im.load()