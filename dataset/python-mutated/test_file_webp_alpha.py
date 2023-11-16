import pytest
from PIL import Image
from .helper import assert_image_equal, assert_image_similar, assert_image_similar_tofile, hopper
_webp = pytest.importorskip('PIL._webp', reason='WebP support not installed')

def setup_module():
    if False:
        i = 10
        return i + 15
    if _webp.WebPDecoderBuggyAlpha():
        pytest.skip('Buggy early version of WebP installed, not testing transparency')

def test_read_rgba():
    if False:
        for i in range(10):
            print('nop')
    '\n    Can we read an RGBA mode file without error?\n    Does it have the bits we expect?\n    '
    file_path = 'Tests/images/transparent.webp'
    with Image.open(file_path) as image:
        assert image.mode == 'RGBA'
        assert image.size == (200, 150)
        assert image.format == 'WEBP'
        image.load()
        image.getdata()
        image.tobytes()
        assert_image_similar_tofile(image, 'Tests/images/transparent.png', 20.0)

def test_write_lossless_rgb(tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Can we write an RGBA mode file with lossless compression without error?\n    Does it have the bits we expect?\n    '
    temp_file = str(tmp_path / 'temp.webp')
    pil_image = hopper('RGBA')
    mask = Image.new('RGBA', (64, 64), (128, 128, 128, 128))
    pil_image.paste(mask, (0, 0), mask)
    pil_image.save(temp_file, lossless=True)
    with Image.open(temp_file) as image:
        image.load()
        assert image.mode == 'RGBA'
        assert image.size == pil_image.size
        assert image.format == 'WEBP'
        image.load()
        image.getdata()
        assert_image_equal(image, pil_image)

def test_write_rgba(tmp_path):
    if False:
        print('Hello World!')
    '\n    Can we write a RGBA mode file to WebP without error.\n    Does it have the bits we expect?\n    '
    temp_file = str(tmp_path / 'temp.webp')
    pil_image = Image.new('RGBA', (10, 10), (255, 0, 0, 20))
    pil_image.save(temp_file)
    if _webp.WebPDecoderBuggyAlpha():
        return
    with Image.open(temp_file) as image:
        image.load()
        assert image.mode == 'RGBA'
        assert image.size == (10, 10)
        assert image.format == 'WEBP'
        image.load()
        image.getdata()
        if _webp.WebPDecoderVersion() <= 513:
            assert_image_similar(image, pil_image, 3.0)
        else:
            assert_image_similar(image, pil_image, 1.0)

def test_keep_rgb_values_when_transparent(tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Saving transparent pixels should retain their original RGB values\n    when using the "exact" parameter.\n    '
    image = hopper('RGB')
    half_transparent_image = image.copy()
    new_alpha = Image.new('L', (128, 128), 255)
    new_alpha.paste(0, (0, 0, 64, 128))
    half_transparent_image.putalpha(new_alpha)
    temp_file = str(tmp_path / 'temp.webp')
    half_transparent_image.save(temp_file, exact=True, lossless=True)
    with Image.open(temp_file) as reloaded:
        assert reloaded.mode == 'RGBA'
        assert reloaded.format == 'WEBP'
        assert_image_equal(reloaded.convert('RGB'), image)

def test_write_unsupported_mode_PA(tmp_path):
    if False:
        return 10
    '\n    Saving a palette-based file with transparency to WebP format\n    should work, and be similar to the original file.\n    '
    temp_file = str(tmp_path / 'temp.webp')
    file_path = 'Tests/images/transparent.gif'
    with Image.open(file_path) as im:
        im.save(temp_file)
    with Image.open(temp_file) as image:
        assert image.mode == 'RGBA'
        assert image.size == (200, 150)
        assert image.format == 'WEBP'
        image.load()
        image.getdata()
        with Image.open(file_path) as im:
            target = im.convert('RGBA')
        assert_image_similar(image, target, 25.0)