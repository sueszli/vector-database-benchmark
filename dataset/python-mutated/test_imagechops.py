from PIL import Image, ImageChops
from .helper import assert_image_equal, hopper
BLACK = (0, 0, 0)
BROWN = (127, 64, 0)
CYAN = (0, 255, 255)
DARK_GREEN = (0, 128, 0)
GREEN = (0, 255, 0)
ORANGE = (255, 128, 0)
WHITE = (255, 255, 255)
GRAY = 128

def test_sanity():
    if False:
        print('Hello World!')
    im = hopper('L')
    ImageChops.constant(im, 128)
    ImageChops.duplicate(im)
    ImageChops.invert(im)
    ImageChops.lighter(im, im)
    ImageChops.darker(im, im)
    ImageChops.difference(im, im)
    ImageChops.multiply(im, im)
    ImageChops.screen(im, im)
    ImageChops.add(im, im)
    ImageChops.add(im, im, 2.0)
    ImageChops.add(im, im, 2.0, 128)
    ImageChops.subtract(im, im)
    ImageChops.subtract(im, im, 2.0)
    ImageChops.subtract(im, im, 2.0, 128)
    ImageChops.add_modulo(im, im)
    ImageChops.subtract_modulo(im, im)
    ImageChops.blend(im, im, 0.5)
    ImageChops.composite(im, im, im)
    ImageChops.soft_light(im, im)
    ImageChops.hard_light(im, im)
    ImageChops.overlay(im, im)
    ImageChops.offset(im, 10)
    ImageChops.offset(im, 10, 20)

def test_add():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/imagedraw_ellipse_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_floodfill_RGB.png') as im2:
            new = ImageChops.add(im1, im2)
    assert new.getbbox() == (25, 25, 76, 76)
    assert new.getpixel((50, 50)) == ORANGE

def test_add_scale_offset():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/imagedraw_ellipse_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_floodfill_RGB.png') as im2:
            new = ImageChops.add(im1, im2, scale=2.5, offset=100)
    assert new.getbbox() == (0, 0, 100, 100)
    assert new.getpixel((50, 50)) == (202, 151, 100)

def test_add_clip():
    if False:
        i = 10
        return i + 15
    im = hopper()
    new = ImageChops.add(im, im)
    assert new.getpixel((50, 50)) == (255, 255, 254)

def test_add_modulo():
    if False:
        return 10
    with Image.open('Tests/images/imagedraw_ellipse_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_floodfill_RGB.png') as im2:
            new = ImageChops.add_modulo(im1, im2)
    assert new.getbbox() == (25, 25, 76, 76)
    assert new.getpixel((50, 50)) == ORANGE

def test_add_modulo_no_clip():
    if False:
        return 10
    im = hopper()
    new = ImageChops.add_modulo(im, im)
    assert new.getpixel((50, 50)) == (224, 76, 254)

def test_blend():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/imagedraw_ellipse_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_floodfill_RGB.png') as im2:
            new = ImageChops.blend(im1, im2, 0.5)
    assert new.getbbox() == (25, 25, 76, 76)
    assert new.getpixel((50, 50)) == BROWN

def test_constant():
    if False:
        for i in range(10):
            print('nop')
    im = Image.new('RGB', (20, 10))
    new = ImageChops.constant(im, GRAY)
    assert new.size == im.size
    assert new.getpixel((0, 0)) == GRAY
    assert new.getpixel((19, 9)) == GRAY

def test_darker_image():
    if False:
        return 10
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_outline_chord_RGB.png') as im2:
            new = ImageChops.darker(im1, im2)
            assert_image_equal(new, im2)

def test_darker_pixel():
    if False:
        return 10
    im1 = hopper()
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im2:
        new = ImageChops.darker(im1, im2)
    assert new.getpixel((50, 50)) == (240, 166, 0)

def test_difference():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/imagedraw_arc_end_le_start.png') as im1:
        with Image.open('Tests/images/imagedraw_arc_no_loops.png') as im2:
            new = ImageChops.difference(im1, im2)
    assert new.getbbox() == (25, 25, 76, 76)

def test_difference_pixel():
    if False:
        return 10
    im1 = hopper()
    with Image.open('Tests/images/imagedraw_polygon_kite_RGB.png') as im2:
        new = ImageChops.difference(im1, im2)
    assert new.getpixel((50, 50)) == (240, 166, 128)

def test_duplicate():
    if False:
        print('Hello World!')
    im = hopper()
    new = ImageChops.duplicate(im)
    assert_image_equal(new, im)

def test_invert():
    if False:
        return 10
    with Image.open('Tests/images/imagedraw_floodfill_RGB.png') as im:
        new = ImageChops.invert(im)
    assert new.getbbox() == (0, 0, 100, 100)
    assert new.getpixel((0, 0)) == WHITE
    assert new.getpixel((50, 50)) == CYAN

def test_lighter_image():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_outline_chord_RGB.png') as im2:
            new = ImageChops.lighter(im1, im2)
        assert_image_equal(new, im1)

def test_lighter_pixel():
    if False:
        for i in range(10):
            print('nop')
    im1 = hopper()
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im2:
        new = ImageChops.lighter(im1, im2)
    assert new.getpixel((50, 50)) == (255, 255, 127)

def test_multiply_black():
    if False:
        print('Hello World!')
    'If you multiply an image with a solid black image,\n    the result is black.'
    im1 = hopper()
    black = Image.new('RGB', im1.size, 'black')
    new = ImageChops.multiply(im1, black)
    assert_image_equal(new, black)

def test_multiply_green():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/imagedraw_floodfill_RGB.png') as im:
        green = Image.new('RGB', im.size, 'green')
        new = ImageChops.multiply(im, green)
    assert new.getbbox() == (25, 25, 76, 76)
    assert new.getpixel((25, 25)) == DARK_GREEN
    assert new.getpixel((50, 50)) == BLACK

def test_multiply_white():
    if False:
        for i in range(10):
            print('nop')
    'If you multiply with a solid white image, the image is unaffected.'
    im1 = hopper()
    white = Image.new('RGB', im1.size, 'white')
    new = ImageChops.multiply(im1, white)
    assert_image_equal(new, im1)

def test_offset():
    if False:
        for i in range(10):
            print('nop')
    xoffset = 45
    yoffset = 20
    with Image.open('Tests/images/imagedraw_ellipse_RGB.png') as im:
        new = ImageChops.offset(im, xoffset, yoffset)
        assert new.getbbox() == (0, 45, 100, 96)
        assert new.getpixel((50, 50)) == BLACK
        assert new.getpixel((50 + xoffset, 50 + yoffset)) == DARK_GREEN
        assert ImageChops.offset(im, xoffset) == ImageChops.offset(im, xoffset, xoffset)

def test_screen():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/imagedraw_ellipse_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_floodfill_RGB.png') as im2:
            new = ImageChops.screen(im1, im2)
    assert new.getbbox() == (25, 25, 76, 76)
    assert new.getpixel((50, 50)) == ORANGE

def test_subtract():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_outline_chord_RGB.png') as im2:
            new = ImageChops.subtract(im1, im2)
    assert new.getbbox() == (25, 50, 76, 76)
    assert new.getpixel((50, 51)) == GREEN
    assert new.getpixel((50, 52)) == BLACK

def test_subtract_scale_offset():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_outline_chord_RGB.png') as im2:
            new = ImageChops.subtract(im1, im2, scale=2.5, offset=100)
    assert new.getbbox() == (0, 0, 100, 100)
    assert new.getpixel((50, 50)) == (100, 202, 100)

def test_subtract_clip():
    if False:
        i = 10
        return i + 15
    im1 = hopper()
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im2:
        new = ImageChops.subtract(im1, im2)
    assert new.getpixel((50, 50)) == (0, 0, 127)

def test_subtract_modulo():
    if False:
        return 10
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im1:
        with Image.open('Tests/images/imagedraw_outline_chord_RGB.png') as im2:
            new = ImageChops.subtract_modulo(im1, im2)
    assert new.getbbox() == (25, 50, 76, 76)
    assert new.getpixel((50, 51)) == GREEN
    assert new.getpixel((50, 52)) == BLACK

def test_subtract_modulo_no_clip():
    if False:
        while True:
            i = 10
    im1 = hopper()
    with Image.open('Tests/images/imagedraw_chord_RGB.png') as im2:
        new = ImageChops.subtract_modulo(im1, im2)
    assert new.getpixel((50, 50)) == (241, 167, 127)

def test_soft_light():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/hopper.png') as im1:
        with Image.open('Tests/images/hopper-XYZ.png') as im2:
            new = ImageChops.soft_light(im1, im2)
    assert new.getpixel((64, 64)) == (163, 54, 32)
    assert new.getpixel((15, 100)) == (1, 1, 3)

def test_hard_light():
    if False:
        return 10
    with Image.open('Tests/images/hopper.png') as im1:
        with Image.open('Tests/images/hopper-XYZ.png') as im2:
            new = ImageChops.hard_light(im1, im2)
    assert new.getpixel((64, 64)) == (144, 50, 27)
    assert new.getpixel((15, 100)) == (1, 1, 2)

def test_overlay():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/hopper.png') as im1:
        with Image.open('Tests/images/hopper-XYZ.png') as im2:
            new = ImageChops.overlay(im1, im2)
    assert new.getpixel((64, 64)) == (159, 50, 27)
    assert new.getpixel((15, 100)) == (1, 1, 2)

def test_logical():
    if False:
        for i in range(10):
            print('nop')

    def table(op, a, b):
        if False:
            return 10
        out = []
        for x in (a, b):
            imx = Image.new('1', (1, 1), x)
            for y in (a, b):
                imy = Image.new('1', (1, 1), y)
                out.append(op(imx, imy).getpixel((0, 0)))
        return tuple(out)
    assert table(ImageChops.logical_and, 0, 1) == (0, 0, 0, 255)
    assert table(ImageChops.logical_or, 0, 1) == (0, 255, 255, 255)
    assert table(ImageChops.logical_xor, 0, 1) == (0, 255, 255, 0)
    assert table(ImageChops.logical_and, 0, 128) == (0, 0, 0, 255)
    assert table(ImageChops.logical_or, 0, 128) == (0, 255, 255, 255)
    assert table(ImageChops.logical_xor, 0, 128) == (0, 255, 255, 0)
    assert table(ImageChops.logical_and, 0, 255) == (0, 0, 0, 255)
    assert table(ImageChops.logical_or, 0, 255) == (0, 255, 255, 255)
    assert table(ImageChops.logical_xor, 0, 255) == (0, 255, 255, 0)