import os
from glob import glob
from itertools import product
import pytest
from PIL import Image
from .helper import assert_image_equal, assert_image_equal_tofile, hopper
_TGA_DIR = os.path.join('Tests', 'images', 'tga')
_TGA_DIR_COMMON = os.path.join(_TGA_DIR, 'common')
_MODES = ('L', 'LA', 'P', 'RGB', 'RGBA')
_ORIGINS = ('tl', 'bl')
_ORIGIN_TO_ORIENTATION = {'tl': 1, 'bl': -1}

@pytest.mark.parametrize('mode', _MODES)
def test_sanity(mode, tmp_path):
    if False:
        print('Hello World!')

    def roundtrip(original_im):
        if False:
            for i in range(10):
                print('nop')
        out = str(tmp_path / 'temp.tga')
        original_im.save(out, rle=rle)
        with Image.open(out) as saved_im:
            if rle:
                assert saved_im.info['compression'] == original_im.info['compression']
            assert saved_im.info['orientation'] == original_im.info['orientation']
            if mode == 'P':
                assert saved_im.getpalette() == original_im.getpalette()
            assert_image_equal(saved_im, original_im)
    png_paths = glob(os.path.join(_TGA_DIR_COMMON, f'*x*_{mode.lower()}.png'))
    for png_path in png_paths:
        with Image.open(png_path) as reference_im:
            assert reference_im.mode == mode
            path_no_ext = os.path.splitext(png_path)[0]
            for (origin, rle) in product(_ORIGINS, (True, False)):
                tga_path = '{}_{}_{}.tga'.format(path_no_ext, origin, 'rle' if rle else 'raw')
                with Image.open(tga_path) as original_im:
                    assert original_im.format == 'TGA'
                    assert original_im.get_format_mimetype() == 'image/x-tga'
                    if rle:
                        assert original_im.info['compression'] == 'tga_rle'
                    assert original_im.info['orientation'] == _ORIGIN_TO_ORIENTATION[origin]
                    if mode == 'P':
                        assert original_im.getpalette() == reference_im.getpalette()
                    assert_image_equal(original_im, reference_im)
                    roundtrip(original_im)

def test_palette_depth_16(tmp_path):
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/p_16.tga') as im:
        assert_image_equal_tofile(im.convert('RGB'), 'Tests/images/p_16.png')
        out = str(tmp_path / 'temp.png')
        im.save(out)
        with Image.open(out) as reloaded:
            assert_image_equal_tofile(reloaded.convert('RGB'), 'Tests/images/p_16.png')

def test_id_field():
    if False:
        while True:
            i = 10
    test_file = 'Tests/images/tga_id_field.tga'
    with Image.open(test_file) as im:
        assert im.size == (100, 100)

def test_id_field_rle():
    if False:
        while True:
            i = 10
    test_file = 'Tests/images/rgb32rle.tga'
    with Image.open(test_file) as im:
        assert im.size == (199, 199)

def test_cross_scan_line():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/cross_scan_line.tga') as im:
        assert_image_equal_tofile(im, 'Tests/images/cross_scan_line.png')
    with Image.open('Tests/images/cross_scan_line_truncated.tga') as im:
        with pytest.raises(OSError):
            im.load()

def test_save(tmp_path):
    if False:
        return 10
    test_file = 'Tests/images/tga_id_field.tga'
    with Image.open(test_file) as im:
        out = str(tmp_path / 'temp.tga')
        im.save(out)
        with Image.open(out) as test_im:
            assert test_im.size == (100, 100)
            assert test_im.info['id_section'] == im.info['id_section']
        im.convert('RGBA').save(out)
    with Image.open(out) as test_im:
        assert test_im.size == (100, 100)

def test_small_palette(tmp_path):
    if False:
        i = 10
        return i + 15
    im = Image.new('P', (1, 1))
    colors = [0, 0, 0]
    im.putpalette(colors)
    out = str(tmp_path / 'temp.tga')
    im.save(out)
    with Image.open(out) as reloaded:
        assert reloaded.getpalette() == colors

def test_save_wrong_mode(tmp_path):
    if False:
        while True:
            i = 10
    im = hopper('PA')
    out = str(tmp_path / 'temp.tga')
    with pytest.raises(OSError):
        im.save(out)

def test_save_mapdepth():
    if False:
        return 10
    test_file = 'Tests/images/200x32_p_bl_raw_origin.tga'
    with Image.open(test_file) as im:
        assert_image_equal_tofile(im, 'Tests/images/tga/common/200x32_p.png')

def test_save_id_section(tmp_path):
    if False:
        return 10
    test_file = 'Tests/images/rgb32rle.tga'
    with Image.open(test_file) as im:
        out = str(tmp_path / 'temp.tga')
        im.save(out)
    with Image.open(out) as test_im:
        assert 'id_section' not in test_im.info
    im.save(out, id_section=b'Test content')
    with Image.open(out) as test_im:
        assert test_im.info['id_section'] == b'Test content'
    id_section = b'Test content' * 25
    with pytest.warns(UserWarning):
        im.save(out, id_section=id_section)
    with Image.open(out) as test_im:
        assert test_im.info['id_section'] == id_section[:255]
    test_file = 'Tests/images/tga_id_field.tga'
    with Image.open(test_file) as im:
        im.save(out, id_section='')
    with Image.open(out) as test_im:
        assert 'id_section' not in test_im.info

def test_save_orientation(tmp_path):
    if False:
        print('Hello World!')
    test_file = 'Tests/images/rgb32rle.tga'
    out = str(tmp_path / 'temp.tga')
    with Image.open(test_file) as im:
        assert im.info['orientation'] == -1
        im.save(out, orientation=1)
    with Image.open(out) as test_im:
        assert test_im.info['orientation'] == 1

def test_horizontal_orientations():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/rgb32rle_top_right.tga') as im:
        assert im.load()[90, 90][:3] == (0, 0, 0)
    with Image.open('Tests/images/rgb32rle_bottom_right.tga') as im:
        assert im.load()[90, 90][:3] == (0, 255, 0)

def test_save_rle(tmp_path):
    if False:
        while True:
            i = 10
    test_file = 'Tests/images/rgb32rle.tga'
    with Image.open(test_file) as im:
        assert im.info['compression'] == 'tga_rle'
        out = str(tmp_path / 'temp.tga')
        im.save(out)
    with Image.open(out) as test_im:
        assert test_im.size == (199, 199)
        assert test_im.info['compression'] == 'tga_rle'
    im.save(out, compression=None)
    with Image.open(out) as test_im:
        assert 'compression' not in test_im.info
    im.convert('RGBA').save(out)
    with Image.open(out) as test_im:
        assert test_im.size == (199, 199)
    test_file = 'Tests/images/tga_id_field.tga'
    with Image.open(test_file) as im:
        assert 'compression' not in im.info
        im.save(out, compression='tga_rle')
    with Image.open(out) as test_im:
        assert test_im.info['compression'] == 'tga_rle'

def test_save_l_transparency(tmp_path):
    if False:
        return 10
    num_transparent = 559
    in_file = 'Tests/images/la.tga'
    with Image.open(in_file) as im:
        assert im.mode == 'LA'
        assert im.getchannel('A').getcolors()[0][0] == num_transparent
        out = str(tmp_path / 'temp.tga')
        im.save(out)
    with Image.open(out) as test_im:
        assert test_im.mode == 'LA'
        assert test_im.getchannel('A').getcolors()[0][0] == num_transparent
        assert_image_equal(im, test_im)