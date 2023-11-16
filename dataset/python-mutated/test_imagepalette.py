import pytest
from PIL import Image, ImagePalette
from .helper import assert_image_equal, assert_image_equal_tofile

def test_sanity():
    if False:
        while True:
            i = 10
    palette = ImagePalette.ImagePalette('RGB', list(range(256)) * 3)
    assert len(palette.colors) == 256

def test_reload():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/hopper.gif') as im:
        original = im.copy()
        im.palette.dirty = 1
        assert_image_equal(im.convert('RGB'), original.convert('RGB'))

def test_getcolor():
    if False:
        return 10
    palette = ImagePalette.ImagePalette()
    assert len(palette.palette) == 0
    assert len(palette.colors) == 0
    test_map = {}
    for i in range(256):
        test_map[palette.getcolor((i, i, i))] = i
    assert len(test_map) == 256
    rgba_palette = ImagePalette.ImagePalette('RGBA')
    assert rgba_palette.getcolor((0, 0, 0)) == rgba_palette.getcolor((0, 0, 0, 255))
    assert palette.getcolor((0, 0, 0)) == palette.getcolor((0, 0, 0, 255))
    with pytest.raises(ValueError):
        palette.getcolor((1, 2, 3))
    palette.getcolor((1, 2, 3), image=Image.new('P', (1, 1)))
    with pytest.raises(ValueError):
        palette.getcolor('unknown')

def test_getcolor_rgba_color_rgb_palette():
    if False:
        i = 10
        return i + 15
    palette = ImagePalette.ImagePalette('RGB')
    assert palette.getcolor((0, 0, 0, 255)) == palette.getcolor((0, 0, 0))
    with pytest.raises(ValueError):
        palette.getcolor((0, 0, 0, 128))

@pytest.mark.parametrize('index, palette', [(0, ImagePalette.ImagePalette()), (255, ImagePalette.ImagePalette('RGB', list(range(256)) * 3))])
def test_getcolor_not_special(index, palette):
    if False:
        i = 10
        return i + 15
    im = Image.new('P', (1, 1))
    im.info['transparency'] = index
    index1 = palette.getcolor((0, 0, 0), im)
    assert index1 != index
    im.info['background'] = index1
    index2 = palette.getcolor((0, 0, 1), im)
    assert index2 not in (index, index1)

def test_file(tmp_path):
    if False:
        while True:
            i = 10
    palette = ImagePalette.ImagePalette('RGB', list(range(256)) * 3)
    f = str(tmp_path / 'temp.lut')
    palette.save(f)
    p = ImagePalette.load(f)
    assert len(p[0]) == 768
    assert p[1] == 'RGB'
    p = ImagePalette.raw(p[1], p[0])
    assert isinstance(p, ImagePalette.ImagePalette)
    assert p.palette == palette.tobytes()

def test_make_linear_lut():
    if False:
        i = 10
        return i + 15
    black = 0
    white = 255
    lut = ImagePalette.make_linear_lut(black, white)
    assert isinstance(lut, list)
    assert len(lut) == 256
    for i in range(0, len(lut)):
        assert lut[i] == i

def test_make_linear_lut_not_yet_implemented():
    if False:
        while True:
            i = 10
    black = 1
    white = 255
    with pytest.raises(NotImplementedError):
        ImagePalette.make_linear_lut(black, white)

def test_make_gamma_lut():
    if False:
        for i in range(10):
            print('nop')
    exp = 5
    lut = ImagePalette.make_gamma_lut(exp)
    assert isinstance(lut, list)
    assert len(lut) == 256
    assert lut[0] == 0
    assert lut[63] == 0
    assert lut[127] == 8
    assert lut[191] == 60
    assert lut[255] == 255

def test_rawmode_valueerrors(tmp_path):
    if False:
        return 10
    palette = ImagePalette.raw('RGB', list(range(256)) * 3)
    with pytest.raises(ValueError):
        palette.tobytes()
    with pytest.raises(ValueError):
        palette.getcolor((1, 2, 3))
    f = str(tmp_path / 'temp.lut')
    with pytest.raises(ValueError):
        palette.save(f)

def test_getdata():
    if False:
        print('Hello World!')
    data_in = list(range(256)) * 3
    palette = ImagePalette.ImagePalette('RGB', data_in)
    (mode, data_out) = palette.getdata()
    assert mode == 'RGB'

def test_rawmode_getdata():
    if False:
        i = 10
        return i + 15
    data_in = list(range(256)) * 3
    palette = ImagePalette.raw('RGB', data_in)
    (rawmode, data_out) = palette.getdata()
    assert rawmode == 'RGB'
    assert data_in == data_out

def test_2bit_palette(tmp_path):
    if False:
        while True:
            i = 10
    outfile = str(tmp_path / 'temp.png')
    rgb = b'\x00' * 2 + b'\x01' * 2 + b'\x02' * 2
    img = Image.frombytes('P', (6, 1), rgb)
    img.putpalette(b'\xff\x00\x00\x00\xff\x00\x00\x00\xff')
    img.save(outfile, format='PNG')
    assert_image_equal_tofile(img, outfile)

def test_invalid_palette():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(OSError):
        ImagePalette.load('Tests/images/hopper.jpg')