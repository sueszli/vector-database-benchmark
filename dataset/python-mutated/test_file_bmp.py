import io
import pytest
from PIL import BmpImagePlugin, Image
from .helper import assert_image_equal, assert_image_equal_tofile, assert_image_similar_tofile, hopper

def test_sanity(tmp_path):
    if False:
        for i in range(10):
            print('nop')

    def roundtrip(im):
        if False:
            return 10
        outfile = str(tmp_path / 'temp.bmp')
        im.save(outfile, 'BMP')
        with Image.open(outfile) as reloaded:
            reloaded.load()
            assert im.mode == reloaded.mode
            assert im.size == reloaded.size
            assert reloaded.format == 'BMP'
            assert reloaded.get_format_mimetype() == 'image/bmp'
    roundtrip(hopper())
    roundtrip(hopper('1'))
    roundtrip(hopper('L'))
    roundtrip(hopper('P'))
    roundtrip(hopper('RGB'))

def test_invalid_file():
    if False:
        for i in range(10):
            print('nop')
    with open('Tests/images/flower.jpg', 'rb') as fp:
        with pytest.raises(SyntaxError):
            BmpImagePlugin.BmpImageFile(fp)

def test_fallback_if_mmap_errors():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/mmap_error.bmp') as im:
        assert_image_equal_tofile(im, 'Tests/images/pal8_offset.bmp')

def test_save_to_bytes():
    if False:
        for i in range(10):
            print('nop')
    output = io.BytesIO()
    im = hopper()
    im.save(output, 'BMP')
    output.seek(0)
    with Image.open(output) as reloaded:
        assert im.mode == reloaded.mode
        assert im.size == reloaded.size
        assert reloaded.format == 'BMP'

def test_small_palette(tmp_path):
    if False:
        print('Hello World!')
    im = Image.new('P', (1, 1))
    colors = [0, 0, 0, 125, 125, 125, 255, 255, 255]
    im.putpalette(colors)
    out = str(tmp_path / 'temp.bmp')
    im.save(out)
    with Image.open(out) as reloaded:
        assert reloaded.getpalette() == colors

def test_save_too_large(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    outfile = str(tmp_path / 'temp.bmp')
    with Image.new('RGB', (1, 1)) as im:
        im._size = (37838, 37838)
        with pytest.raises(ValueError):
            im.save(outfile)

def test_dpi():
    if False:
        i = 10
        return i + 15
    dpi = (72, 72)
    output = io.BytesIO()
    with hopper() as im:
        im.save(output, 'BMP', dpi=dpi)
    output.seek(0)
    with Image.open(output) as reloaded:
        assert reloaded.info['dpi'] == (72.008961115161, 72.008961115161)

def test_save_bmp_with_dpi(tmp_path):
    if False:
        return 10
    outfile = str(tmp_path / 'temp.jpg')
    with Image.open('Tests/images/hopper.bmp') as im:
        assert im.info['dpi'] == (95.98654816726399, 95.98654816726399)
        im.save(outfile, 'JPEG', dpi=im.info['dpi'])
        with Image.open(outfile) as reloaded:
            reloaded.load()
            assert reloaded.info['dpi'] == (96, 96)
            assert reloaded.size == im.size
            assert reloaded.format == 'JPEG'

def test_save_float_dpi(tmp_path):
    if False:
        i = 10
        return i + 15
    outfile = str(tmp_path / 'temp.bmp')
    with Image.open('Tests/images/hopper.bmp') as im:
        im.save(outfile, dpi=(72.21216100543306, 72.21216100543306))
        with Image.open(outfile) as reloaded:
            assert reloaded.info['dpi'] == (72.21216100543306, 72.21216100543306)

def test_load_dib():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/clipboard.dib') as im:
        assert im.format == 'DIB'
        assert im.get_format_mimetype() == 'image/bmp'
        assert_image_equal_tofile(im, 'Tests/images/clipboard_target.png')

def test_save_dib(tmp_path):
    if False:
        while True:
            i = 10
    outfile = str(tmp_path / 'temp.dib')
    with Image.open('Tests/images/clipboard.dib') as im:
        im.save(outfile)
        with Image.open(outfile) as reloaded:
            assert reloaded.format == 'DIB'
            assert reloaded.get_format_mimetype() == 'image/bmp'
            assert_image_equal(im, reloaded)

def test_rgba_bitfields():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/rgb32bf-rgba.bmp') as im:
        (b, g, r) = im.split()[1:]
        im = Image.merge('RGB', (r, g, b))
    assert_image_equal_tofile(im, 'Tests/images/bmp/q/rgb32bf-xbgr.bmp')
    with Image.open('Tests/images/rgb32bf-abgr.bmp') as im:
        assert_image_equal_tofile(im.convert('RGB'), 'Tests/images/bmp/q/rgb32bf-xbgr.bmp')

def test_rle8():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/hopper_rle8.bmp') as im:
        assert_image_similar_tofile(im.convert('RGB'), 'Tests/images/hopper.bmp', 12)
    with Image.open('Tests/images/hopper_rle8_grayscale.bmp') as im:
        assert_image_equal_tofile(im, 'Tests/images/bw_gradient.png')
    with Image.open('Tests/images/hopper_rle8_row_overflow.bmp') as im:
        assert_image_similar_tofile(im.convert('RGB'), 'Tests/images/hopper.bmp', 12)
    with open('Tests/images/bmp/g/pal8rle.bmp', 'rb') as fp:
        data = fp.read(1063) + b'\x01'
        with Image.open(io.BytesIO(data)) as im:
            with pytest.raises(ValueError):
                im.load()

def test_rle4():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/bmp/g/pal4rle.bmp') as im:
        assert_image_similar_tofile(im, 'Tests/images/bmp/g/pal4.bmp', 12)

@pytest.mark.parametrize('file_name,length', (('Tests/images/hopper_rle8.bmp', 1078), ('Tests/images/bmp/q/pal8rletrns.bmp', 3670), ('Tests/images/bmp/g/pal8rle.bmp', 1064)))
def test_rle8_eof(file_name, length):
    if False:
        return 10
    with open(file_name, 'rb') as fp:
        data = fp.read(length)
        with Image.open(io.BytesIO(data)) as im:
            with pytest.raises(ValueError):
                im.load()

def test_offset():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/pal8_offset.bmp') as im:
        assert_image_equal_tofile(im, 'Tests/images/bmp/g/pal8.bmp')