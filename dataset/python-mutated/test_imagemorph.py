import pytest
from PIL import Image, ImageMorph, _imagingmorph
from .helper import assert_image_equal_tofile, hopper

def string_to_img(image_string):
    if False:
        i = 10
        return i + 15
    'Turn a string image representation into a binary image'
    rows = [s for s in image_string.replace(' ', '').split('\n') if len(s)]
    height = len(rows)
    width = len(rows[0])
    im = Image.new('L', (width, height))
    for i in range(width):
        for j in range(height):
            c = rows[j][i]
            v = c in 'X1'
            im.putpixel((i, j), v)
    return im
A = string_to_img('\n    .......\n    .......\n    ..111..\n    ..111..\n    ..111..\n    .......\n    .......\n    ')

def img_to_string(im):
    if False:
        i = 10
        return i + 15
    'Turn a (small) binary image into a string representation'
    chars = '.1'
    (width, height) = im.size
    return '\n'.join((''.join((chars[im.getpixel((c, r)) > 0] for c in range(width))) for r in range(height)))

def img_string_normalize(im):
    if False:
        i = 10
        return i + 15
    return img_to_string(string_to_img(im))

def assert_img_equal_img_string(a, b_string):
    if False:
        return 10
    assert img_to_string(a) == img_string_normalize(b_string)

def test_str_to_img():
    if False:
        return 10
    assert_image_equal_tofile(A, 'Tests/images/morph_a.png')

def create_lut():
    if False:
        print('Hello World!')
    for op in ('corner', 'dilation4', 'dilation8', 'erosion4', 'erosion8', 'edge'):
        lb = ImageMorph.LutBuilder(op_name=op)
        lut = lb.build_lut()
        with open(f'Tests/images/{op}.lut', 'wb') as f:
            f.write(lut)

@pytest.mark.parametrize('op', ('corner', 'dilation4', 'dilation8', 'erosion4', 'erosion8', 'edge'))
def test_lut(op):
    if False:
        i = 10
        return i + 15
    lb = ImageMorph.LutBuilder(op_name=op)
    assert lb.get_lut() is None
    lut = lb.build_lut()
    with open(f'Tests/images/{op}.lut', 'rb') as f:
        assert lut == bytearray(f.read())

def test_no_operator_loaded():
    if False:
        print('Hello World!')
    mop = ImageMorph.MorphOp()
    with pytest.raises(Exception) as e:
        mop.apply(None)
    assert str(e.value) == 'No operator loaded'
    with pytest.raises(Exception) as e:
        mop.match(None)
    assert str(e.value) == 'No operator loaded'
    with pytest.raises(Exception) as e:
        mop.save_lut(None)
    assert str(e.value) == 'No operator loaded'

def test_erosion8():
    if False:
        print('Hello World!')
    mop = ImageMorph.MorphOp(op_name='erosion8')
    (count, Aout) = mop.apply(A)
    assert count == 8
    assert_img_equal_img_string(Aout, '\n                                     .......\n                                     .......\n                                     .......\n                                     ...1...\n                                     .......\n                                     .......\n                                     .......\n                                     ')

def test_dialation8():
    if False:
        while True:
            i = 10
    mop = ImageMorph.MorphOp(op_name='dilation8')
    (count, Aout) = mop.apply(A)
    assert count == 16
    assert_img_equal_img_string(Aout, '\n                                     .......\n                                     .11111.\n                                     .11111.\n                                     .11111.\n                                     .11111.\n                                     .11111.\n                                     .......\n                                     ')

def test_erosion4():
    if False:
        while True:
            i = 10
    mop = ImageMorph.MorphOp(op_name='dilation4')
    (count, Aout) = mop.apply(A)
    assert count == 12
    assert_img_equal_img_string(Aout, '\n                                     .......\n                                     ..111..\n                                     .11111.\n                                     .11111.\n                                     .11111.\n                                     ..111..\n                                     .......\n                                     ')

def test_edge():
    if False:
        for i in range(10):
            print('nop')
    mop = ImageMorph.MorphOp(op_name='edge')
    (count, Aout) = mop.apply(A)
    assert count == 1
    assert_img_equal_img_string(Aout, '\n                                     .......\n                                     .......\n                                     ..111..\n                                     ..1.1..\n                                     ..111..\n                                     .......\n                                     .......\n                                     ')

def test_corner():
    if False:
        while True:
            i = 10
    mop = ImageMorph.MorphOp(patterns=['1:(... ... ...)->0', '4:(00. 01. ...)->1'])
    (count, Aout) = mop.apply(A)
    assert count == 5
    assert_img_equal_img_string(Aout, '\n                                     .......\n                                     .......\n                                     ..1.1..\n                                     .......\n                                     ..1.1..\n                                     .......\n                                     .......\n                                     ')
    coords = mop.match(A)
    assert len(coords) == 4
    assert tuple(coords) == ((2, 2), (4, 2), (2, 4), (4, 4))
    coords = mop.get_on_pixels(Aout)
    assert len(coords) == 4
    assert tuple(coords) == ((2, 2), (4, 2), (2, 4), (4, 4))

def test_mirroring():
    if False:
        i = 10
        return i + 15
    mop = ImageMorph.MorphOp(patterns=['1:(... ... ...)->0', 'M:(00. 01. ...)->1'])
    (count, Aout) = mop.apply(A)
    assert count == 7
    assert_img_equal_img_string(Aout, '\n                                     .......\n                                     .......\n                                     ..1.1..\n                                     .......\n                                     .......\n                                     .......\n                                     .......\n                                     ')

def test_negate():
    if False:
        return 10
    mop = ImageMorph.MorphOp(patterns=['1:(... ... ...)->0', 'N:(00. 01. ...)->1'])
    (count, Aout) = mop.apply(A)
    assert count == 8
    assert_img_equal_img_string(Aout, '\n                                     .......\n                                     .......\n                                     ..1....\n                                     .......\n                                     .......\n                                     .......\n                                     .......\n                                     ')

def test_incorrect_mode():
    if False:
        for i in range(10):
            print('nop')
    im = hopper('RGB')
    mop = ImageMorph.MorphOp(op_name='erosion8')
    with pytest.raises(ValueError) as e:
        mop.apply(im)
    assert str(e.value) == 'Image mode must be L'
    with pytest.raises(ValueError) as e:
        mop.match(im)
    assert str(e.value) == 'Image mode must be L'
    with pytest.raises(ValueError) as e:
        mop.get_on_pixels(im)
    assert str(e.value) == 'Image mode must be L'

def test_add_patterns():
    if False:
        print('Hello World!')
    lb = ImageMorph.LutBuilder(op_name='corner')
    assert lb.patterns == ['1:(... ... ...)->0', '4:(00. 01. ...)->1']
    new_patterns = ['M:(00. 01. ...)->1', 'N:(00. 01. ...)->1']
    lb.add_patterns(new_patterns)
    assert lb.patterns == ['1:(... ... ...)->0', '4:(00. 01. ...)->1', 'M:(00. 01. ...)->1', 'N:(00. 01. ...)->1']

def test_unknown_pattern():
    if False:
        i = 10
        return i + 15
    with pytest.raises(Exception):
        ImageMorph.LutBuilder(op_name='unknown')

def test_pattern_syntax_error():
    if False:
        for i in range(10):
            print('nop')
    lb = ImageMorph.LutBuilder(op_name='corner')
    new_patterns = ['a pattern with a syntax error']
    lb.add_patterns(new_patterns)
    with pytest.raises(Exception) as e:
        lb.build_lut()
    assert str(e.value) == 'Syntax error in pattern "a pattern with a syntax error"'

def test_load_invalid_mrl():
    if False:
        i = 10
        return i + 15
    invalid_mrl = 'Tests/images/hopper.png'
    mop = ImageMorph.MorphOp()
    with pytest.raises(Exception) as e:
        mop.load_lut(invalid_mrl)
    assert str(e.value) == 'Wrong size operator file!'

def test_roundtrip_mrl(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    tempfile = str(tmp_path / 'temp.mrl')
    mop = ImageMorph.MorphOp(op_name='corner')
    initial_lut = mop.lut
    mop.save_lut(tempfile)
    mop.load_lut(tempfile)
    assert mop.lut == initial_lut

def test_set_lut():
    if False:
        while True:
            i = 10
    lb = ImageMorph.LutBuilder(op_name='corner')
    lut = lb.build_lut()
    mop = ImageMorph.MorphOp()
    mop.set_lut(lut)
    assert mop.lut == lut

def test_wrong_mode():
    if False:
        return 10
    lut = ImageMorph.LutBuilder(op_name='corner').build_lut()
    imrgb = Image.new('RGB', (10, 10))
    iml = Image.new('L', (10, 10))
    with pytest.raises(RuntimeError):
        _imagingmorph.apply(bytes(lut), imrgb.im.id, iml.im.id)
    with pytest.raises(RuntimeError):
        _imagingmorph.apply(bytes(lut), iml.im.id, imrgb.im.id)
    with pytest.raises(RuntimeError):
        _imagingmorph.match(bytes(lut), imrgb.im.id)
    _imagingmorph.match(bytes(lut), iml.im.id)