import os
import pytest
from PIL import FontFile, Image, ImageDraw, ImageFont, PcfFontFile
from .helper import assert_image_equal_tofile, assert_image_similar_tofile, skip_unless_feature
fontname = 'Tests/fonts/10x20-ISO8859-1.pcf'
message = 'hello, world'
pytestmark = skip_unless_feature('zlib')

def save_font(request, tmp_path):
    if False:
        i = 10
        return i + 15
    with open(fontname, 'rb') as test_file:
        font = PcfFontFile.PcfFontFile(test_file)
    assert isinstance(font, FontFile.FontFile)
    assert len([_f for _f in font.glyph if _f]) == 223
    tempname = str(tmp_path / 'temp.pil')

    def delete_tempfile():
        if False:
            i = 10
            return i + 15
        try:
            os.remove(tempname[:-4] + '.pbm')
        except OSError:
            pass
    request.addfinalizer(delete_tempfile)
    font.save(tempname)
    with Image.open(tempname.replace('.pil', '.pbm')) as loaded:
        assert_image_equal_tofile(loaded, 'Tests/fonts/10x20.pbm')
    with open(tempname, 'rb') as f_loaded:
        with open('Tests/fonts/10x20.pil', 'rb') as f_target:
            assert f_loaded.read() == f_target.read()
    return tempname

def test_sanity(request, tmp_path):
    if False:
        return 10
    save_font(request, tmp_path)

def test_less_than_256_characters():
    if False:
        for i in range(10):
            print('nop')
    with open('Tests/fonts/10x20-ISO8859-1-fewer-characters.pcf', 'rb') as test_file:
        font = PcfFontFile.PcfFontFile(test_file)
    assert isinstance(font, FontFile.FontFile)
    assert len([_f for _f in font.glyph if _f]) == 127

def test_invalid_file():
    if False:
        print('Hello World!')
    with open('Tests/images/flower.jpg', 'rb') as fp:
        with pytest.raises(SyntaxError):
            PcfFontFile.PcfFontFile(fp)

def test_draw(request, tmp_path):
    if False:
        while True:
            i = 10
    tempname = save_font(request, tmp_path)
    font = ImageFont.load(tempname)
    im = Image.new('L', (130, 30), 'white')
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), message, 'black', font=font)
    assert_image_similar_tofile(im, 'Tests/images/test_draw_pbm_target.png', 0)

def test_textsize(request, tmp_path):
    if False:
        while True:
            i = 10
    tempname = save_font(request, tmp_path)
    font = ImageFont.load(tempname)
    for i in range(255):
        (ox, oy, dx, dy) = font.getbbox(chr(i))
        assert ox == 0
        assert oy == 0
        assert dy == 20
        assert dx in (0, 10)
        assert font.getlength(chr(i)) == dx
    for i in range(len(message)):
        msg = message[:i + 1]
        assert font.getlength(msg) == len(msg) * 10
        assert font.getbbox(msg) == (0, 0, len(msg) * 10, 20)

def _test_high_characters(request, tmp_path, message):
    if False:
        return 10
    tempname = save_font(request, tmp_path)
    font = ImageFont.load(tempname)
    im = Image.new('L', (750, 30), 'white')
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), message, 'black', font=font)
    assert_image_similar_tofile(im, 'Tests/images/high_ascii_chars.png', 0)

def test_high_characters(request, tmp_path):
    if False:
        return 10
    message = ''.join((chr(i + 1) for i in range(140, 232)))
    _test_high_characters(request, tmp_path, message)
    _test_high_characters(request, tmp_path, message.encode('latin1'))