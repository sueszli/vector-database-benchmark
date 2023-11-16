import io
import os
import pytest
from PIL import IcoImagePlugin, Image, ImageDraw
from .helper import assert_image_equal, assert_image_equal_tofile, hopper
TEST_ICO_FILE = 'Tests/images/hopper.ico'

def test_sanity():
    if False:
        print('Hello World!')
    with Image.open(TEST_ICO_FILE) as im:
        im.load()
    assert im.mode == 'RGBA'
    assert im.size == (16, 16)
    assert im.format == 'ICO'
    assert im.get_format_mimetype() == 'image/x-icon'

def test_load():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(TEST_ICO_FILE) as im:
        assert im.load()[0, 0] == (1, 1, 9, 255)

def test_mask():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/hopper_mask.ico') as im:
        assert_image_equal_tofile(im, 'Tests/images/hopper_mask.png')

def test_black_and_white():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/black_and_white.ico') as im:
        assert im.mode == 'RGBA'
        assert im.size == (16, 16)

def test_invalid_file():
    if False:
        for i in range(10):
            print('nop')
    with open('Tests/images/flower.jpg', 'rb') as fp:
        with pytest.raises(SyntaxError):
            IcoImagePlugin.IcoImageFile(fp)

def test_save_to_bytes():
    if False:
        return 10
    output = io.BytesIO()
    im = hopper()
    im.save(output, 'ico', sizes=[(32, 32), (64, 64)])
    output.seek(0)
    with Image.open(output) as reloaded:
        assert reloaded.info['sizes'] == {(32, 32), (64, 64)}
        assert im.mode == reloaded.mode
        assert (64, 64) == reloaded.size
        assert reloaded.format == 'ICO'
        assert_image_equal(reloaded, hopper().resize((64, 64), Image.Resampling.LANCZOS))
    output.seek(0)
    with Image.open(output) as reloaded:
        reloaded.size = (32, 32)
        assert im.mode == reloaded.mode
        assert (32, 32) == reloaded.size
        assert reloaded.format == 'ICO'
        assert_image_equal(reloaded, hopper().resize((32, 32), Image.Resampling.LANCZOS))

def test_getpixel(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    temp_file = str(tmp_path / 'temp.ico')
    im = hopper()
    im.save(temp_file, 'ico', sizes=[(32, 32), (64, 64)])
    with Image.open(temp_file) as reloaded:
        reloaded.load()
        reloaded.size = (32, 32)
        assert reloaded.getpixel((0, 0)) == (18, 20, 62)

def test_no_duplicates(tmp_path):
    if False:
        print('Hello World!')
    temp_file = str(tmp_path / 'temp.ico')
    temp_file2 = str(tmp_path / 'temp2.ico')
    im = hopper()
    sizes = [(32, 32), (64, 64)]
    im.save(temp_file, 'ico', sizes=sizes)
    sizes.append(sizes[-1])
    im.save(temp_file2, 'ico', sizes=sizes)
    assert os.path.getsize(temp_file) == os.path.getsize(temp_file2)

def test_different_bit_depths(tmp_path):
    if False:
        return 10
    temp_file = str(tmp_path / 'temp.ico')
    temp_file2 = str(tmp_path / 'temp2.ico')
    im = hopper()
    im.save(temp_file, 'ico', bitmap_format='bmp', sizes=[(128, 128)])
    hopper('1').save(temp_file2, 'ico', bitmap_format='bmp', sizes=[(128, 128)], append_images=[im])
    assert os.path.getsize(temp_file) != os.path.getsize(temp_file2)
    temp_file3 = str(tmp_path / 'temp3.ico')
    temp_file4 = str(tmp_path / 'temp4.ico')
    im.save(temp_file3, 'ico', bitmap_format='bmp', sizes=[(128, 128)])
    im.save(temp_file4, 'ico', bitmap_format='bmp', sizes=[(128, 128)], append_images=[Image.new('P', (64, 64))])
    assert os.path.getsize(temp_file3) == os.path.getsize(temp_file4)

@pytest.mark.parametrize('mode', ('1', 'L', 'P', 'RGB', 'RGBA'))
def test_save_to_bytes_bmp(mode):
    if False:
        print('Hello World!')
    output = io.BytesIO()
    im = hopper(mode)
    im.save(output, 'ico', bitmap_format='bmp', sizes=[(32, 32), (64, 64)])
    output.seek(0)
    with Image.open(output) as reloaded:
        assert reloaded.info['sizes'] == {(32, 32), (64, 64)}
        assert 'RGBA' == reloaded.mode
        assert (64, 64) == reloaded.size
        assert reloaded.format == 'ICO'
        im = hopper(mode).resize((64, 64), Image.Resampling.LANCZOS).convert('RGBA')
        assert_image_equal(reloaded, im)
    output.seek(0)
    with Image.open(output) as reloaded:
        reloaded.size = (32, 32)
        assert 'RGBA' == reloaded.mode
        assert (32, 32) == reloaded.size
        assert reloaded.format == 'ICO'
        im = hopper(mode).resize((32, 32), Image.Resampling.LANCZOS).convert('RGBA')
        assert_image_equal(reloaded, im)

def test_incorrect_size():
    if False:
        return 10
    with Image.open(TEST_ICO_FILE) as im:
        with pytest.raises(ValueError):
            im.size = (1, 1)

def test_save_256x256(tmp_path):
    if False:
        i = 10
        return i + 15
    'Issue #2264 https://github.com/python-pillow/Pillow/issues/2264'
    with Image.open('Tests/images/hopper_256x256.ico') as im:
        outfile = str(tmp_path / 'temp_saved_hopper_256x256.ico')
        im.save(outfile)
    with Image.open(outfile) as im_saved:
        assert im_saved.size == (256, 256)

def test_only_save_relevant_sizes(tmp_path):
    if False:
        while True:
            i = 10
    'Issue #2266 https://github.com/python-pillow/Pillow/issues/2266\n    Should save in 16x16, 24x24, 32x32, 48x48 sizes\n    and not in 16x16, 24x24, 32x32, 48x48, 48x48, 48x48, 48x48 sizes\n    '
    with Image.open('Tests/images/python.ico') as im:
        outfile = str(tmp_path / 'temp_saved_python.ico')
        im.save(outfile)
    with Image.open(outfile) as im_saved:
        assert im_saved.info['sizes'] == {(16, 16), (24, 24), (32, 32), (48, 48)}

def test_save_append_images(tmp_path):
    if False:
        print('Hello World!')
    im = hopper('RGBA')
    provided_im = Image.new('RGBA', (32, 32), (255, 0, 0))
    outfile = str(tmp_path / 'temp_saved_multi_icon.ico')
    im.save(outfile, sizes=[(32, 32), (128, 128)], append_images=[provided_im])
    with Image.open(outfile) as reread:
        assert_image_equal(reread, hopper('RGBA'))
        reread.size = (32, 32)
        assert_image_equal(reread, provided_im)

def test_unexpected_size():
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(UserWarning):
        with Image.open('Tests/images/hopper_unexpected.ico') as im:
            assert im.size == (16, 16)

def test_draw_reloaded(tmp_path):
    if False:
        i = 10
        return i + 15
    with Image.open(TEST_ICO_FILE) as im:
        outfile = str(tmp_path / 'temp_saved_hopper_draw.ico')
        draw = ImageDraw.Draw(im)
        draw.line((0, 0) + im.size, '#f00')
        im.save(outfile)
    with Image.open(outfile) as im:
        assert_image_equal_tofile(im, 'Tests/images/hopper_draw.ico')