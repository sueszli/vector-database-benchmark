from panda3d.core import Texture, LColor
from array import array

def peeker_from_pixel(component_type, format, data):
    if False:
        return 10
    ' Creates a 1-pixel texture with the given settings and pixel data,\n    then returns a TexturePeeker as result of calling texture.peek(). '
    tex = Texture('')
    tex.setup_1d_texture(1, component_type, format)
    tex.set_ram_image(data)
    peeker = tex.peek()
    assert peeker.has_pixel(0, 0)
    return peeker

def test_texture_peek_ubyte():
    if False:
        print('Hello World!')
    maxval = 255
    data = array('B', (2, 1, 0, maxval))
    peeker = peeker_from_pixel(Texture.T_unsigned_byte, Texture.F_rgba, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    col *= maxval
    assert col == (0, 1, 2, maxval)

def test_texture_peek_ushort():
    if False:
        print('Hello World!')
    maxval = 65535
    data = array('H', (2, 1, 0, maxval))
    peeker = peeker_from_pixel(Texture.T_unsigned_short, Texture.F_rgba, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    col *= maxval
    assert col == (0, 1, 2, maxval)

def test_texture_peek_uint():
    if False:
        for i in range(10):
            print('nop')
    maxval = 4294967295
    data = array('I', (2, 1, 0, maxval))
    peeker = peeker_from_pixel(Texture.T_unsigned_int, Texture.F_rgba, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    col *= maxval
    assert col == (0, 1, 2, maxval)

def test_texture_peek_float():
    if False:
        print('Hello World!')
    data = array('f', (1.0, 0.0, -2.0, 10000.0))
    peeker = peeker_from_pixel(Texture.T_float, Texture.F_rgba, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (-2.0, 0.0, 1.0, 10000.0)

def test_texture_peek_half():
    if False:
        print('Hello World!')
    data = array('H', (15360, 49152, 31743, 13653))
    peeker = peeker_from_pixel(Texture.T_half_float, Texture.F_rgba, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (65504.0, -2.0, 1.0, 0.333251953125)

def test_texture_peek_srgb():
    if False:
        print('Hello World!')
    data = array('B', [188, 188, 188])
    peeker = peeker_from_pixel(Texture.T_unsigned_byte, Texture.F_srgb, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col.almost_equal((0.5, 0.5, 0.5, 1.0), 1 / 255.0)

def test_texture_peek_srgba():
    if False:
        i = 10
        return i + 15
    data = array('B', [188, 188, 188, 188])
    peeker = peeker_from_pixel(Texture.T_unsigned_byte, Texture.F_srgb_alpha, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col.almost_equal((0.5, 0.5, 0.5, 188 / 255.0), 1 / 255.0)

def test_texture_peek_ubyte_i():
    if False:
        print('Hello World!')
    maxval = 255
    data = array('B', (2, 1, 0, maxval))
    peeker = peeker_from_pixel(Texture.T_unsigned_byte, Texture.F_rgba8i, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (0, 1, 2, maxval)

def test_texture_peek_byte_i():
    if False:
        for i in range(10):
            print('nop')
    minval = -128
    maxval = 127
    data = array('b', (0, -1, minval, maxval))
    peeker = peeker_from_pixel(Texture.T_byte, Texture.F_rgba8i, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (minval, -1, 0, maxval)

def test_texture_peek_ushort_i():
    if False:
        print('Hello World!')
    maxval = 65535
    data = array('H', (2, 1, 0, maxval))
    peeker = peeker_from_pixel(Texture.T_unsigned_short, Texture.F_rgba16i, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (0, 1, 2, maxval)

def test_texture_peek_short_i():
    if False:
        for i in range(10):
            print('nop')
    minval = -32768
    maxval = 32767
    data = array('h', (0, -1, minval, maxval))
    peeker = peeker_from_pixel(Texture.T_short, Texture.F_rgba16i, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (minval, -1, 0, maxval)

def test_texture_peek_uint_i():
    if False:
        print('Hello World!')
    maxval = 2147483648
    data = array('I', (2, 1, 0, maxval))
    peeker = peeker_from_pixel(Texture.T_unsigned_int, Texture.F_rgba32i, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (0, 1, 2, maxval)

def test_texture_peek_int_i():
    if False:
        i = 10
        return i + 15
    minval = -2147483648
    maxval = 2147483647
    data = array('i', (0, -1, minval, maxval))
    peeker = peeker_from_pixel(Texture.T_int, Texture.F_rgba32i, data)
    col = LColor()
    peeker.fetch_pixel(col, 0, 0)
    assert col == (minval, -1, 0, maxval)

def test_texture_peek_cube():
    if False:
        i = 10
        return i + 15
    maxval = 255
    data_list = []
    for z in range(6):
        for y in range(3):
            for x in range(3):
                data_list += [z, y, x, maxval]
    data = array('B', data_list)
    tex = Texture('')
    tex.setup_cube_map(3, Texture.T_unsigned_byte, Texture.F_rgba8i)
    tex.set_ram_image(data)
    peeker = tex.peek()
    assert peeker.has_pixel(0, 0)
    assert peeker.has_pixel(0, 0, 0)
    col = LColor()
    peeker.fetch_pixel(col, 1, 2)
    assert col == (1, 2, 0, maxval)
    for faceidx in range(6):
        col = LColor()
        peeker.fetch_pixel(col, 0, 0, faceidx)
        assert col == (0, 0, faceidx, maxval)

    def lookup(*vec):
        if False:
            return 10
        col = LColor()
        peeker.lookup(col, *vec)
        return col
    assert lookup(1, 0, 0) == (1, 1, 0, maxval)
    assert lookup(-1, 0, 0) == (1, 1, 1, maxval)
    assert lookup(0, 1, 0) == (1, 1, 2, maxval)
    assert lookup(0, -1, 0) == (1, 1, 3, maxval)
    assert lookup(0, 0, 1) == (1, 1, 4, maxval)
    assert lookup(0, 0, -1) == (1, 1, 5, maxval)
    assert lookup(0, 2, 0) == (1, 1, 2, maxval)
    assert lookup(0, 0, -0.5) == (1, 1, 5, maxval)
    assert lookup(1.00001, 1, 1) == (0, 0, 0, maxval)
    assert lookup(1.00001, 1, 0) == (1, 0, 0, maxval)
    assert lookup(1, 1.00001, 0) == (2, 1, 2, maxval)