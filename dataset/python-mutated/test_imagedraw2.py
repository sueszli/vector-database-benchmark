import os.path
import pytest
from PIL import Image, ImageDraw, ImageDraw2, features
from .helper import assert_image_equal, assert_image_equal_tofile, assert_image_similar_tofile, hopper, skip_unless_feature
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (190, 190, 190)
DEFAULT_MODE = 'RGB'
IMAGES_PATH = os.path.join('Tests', 'images', 'imagedraw')
(W, H) = (100, 100)
X0 = int(W / 4)
X1 = int(X0 * 3)
Y0 = int(H / 4)
Y1 = int(X0 * 3)
BBOX = (((X0, Y0), (X1, Y1)), [(X0, Y0), (X1, Y1)], (X0, Y0, X1, Y1), [X0, Y0, X1, Y1])
POINTS = (((10, 10), (20, 40), (30, 30)), [(10, 10), (20, 40), (30, 30)], (10, 10, 20, 40, 30, 30), [10, 10, 20, 40, 30, 30])
FONT_PATH = 'Tests/fonts/FreeMono.ttf'

def test_sanity():
    if False:
        for i in range(10):
            print('nop')
    im = hopper('RGB').copy()
    draw = ImageDraw2.Draw(im)
    pen = ImageDraw2.Pen('blue', width=7)
    draw.line(list(range(10)), pen)
    (draw, handler) = ImageDraw.getdraw(im)
    pen = ImageDraw2.Pen('blue', width=7)
    draw.line(list(range(10)), pen)

@pytest.mark.parametrize('bbox', BBOX)
def test_ellipse(bbox):
    if False:
        print('Hello World!')
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    pen = ImageDraw2.Pen('blue', width=2)
    brush = ImageDraw2.Brush('green')
    draw.ellipse(bbox, pen, brush)
    assert_image_similar_tofile(im, 'Tests/images/imagedraw_ellipse_RGB.png', 1)

def test_ellipse_edge():
    if False:
        return 10
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    brush = ImageDraw2.Brush('white')
    draw.ellipse(((0, 0), (W - 1, H - 1)), brush)
    assert_image_similar_tofile(im, 'Tests/images/imagedraw_ellipse_edge.png', 1)

@pytest.mark.parametrize('points', POINTS)
def test_line(points):
    if False:
        print('Hello World!')
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    pen = ImageDraw2.Pen('yellow', width=2)
    draw.line(points, pen)
    assert_image_equal_tofile(im, 'Tests/images/imagedraw_line.png')

@pytest.mark.parametrize('points', POINTS)
def test_line_pen_as_brush(points):
    if False:
        return 10
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    pen = None
    brush = ImageDraw2.Pen('yellow', width=2)
    draw.line(points, pen, brush)
    assert_image_equal_tofile(im, 'Tests/images/imagedraw_line.png')

@pytest.mark.parametrize('points', POINTS)
def test_polygon(points):
    if False:
        i = 10
        return i + 15
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    pen = ImageDraw2.Pen('blue', width=2)
    brush = ImageDraw2.Brush('red')
    draw.polygon(points, pen, brush)
    assert_image_equal_tofile(im, 'Tests/images/imagedraw_polygon.png')

@pytest.mark.parametrize('bbox', BBOX)
def test_rectangle(bbox):
    if False:
        i = 10
        return i + 15
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    pen = ImageDraw2.Pen('green', width=2)
    brush = ImageDraw2.Brush('black')
    draw.rectangle(bbox, pen, brush)
    assert_image_equal_tofile(im, 'Tests/images/imagedraw_rectangle.png')

def test_big_rectangle():
    if False:
        print('Hello World!')
    im = Image.new('RGB', (W, H))
    bbox = [(-1, -1), (W + 1, H + 1)]
    brush = ImageDraw2.Brush('orange')
    draw = ImageDraw2.Draw(im)
    expected = 'Tests/images/imagedraw_big_rectangle.png'
    draw.rectangle(bbox, brush)
    assert_image_similar_tofile(im, expected, 1)

@skip_unless_feature('freetype2')
def test_text():
    if False:
        print('Hello World!')
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    font = ImageDraw2.Font('white', FONT_PATH)
    expected = 'Tests/images/imagedraw2_text.png'
    draw.text((5, 5), 'ImageDraw2', font)
    assert_image_similar_tofile(im, expected, 13)

@skip_unless_feature('freetype2')
def test_textbbox():
    if False:
        i = 10
        return i + 15
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    font = ImageDraw2.Font('white', FONT_PATH)
    bbox = draw.textbbox((0, 0), 'ImageDraw2', font)
    right = 72 if features.check_feature('raqm') else 70
    assert bbox == (0, 2, right, 12)

@skip_unless_feature('freetype2')
def test_textsize_empty_string():
    if False:
        while True:
            i = 10
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    font = ImageDraw2.Font('white', FONT_PATH)
    draw.textbbox((0, 0), '', font)
    draw.textbbox((0, 0), '\n', font)
    draw.textbbox((0, 0), 'test\n', font)
    draw.textlength('', font)

@skip_unless_feature('freetype2')
def test_flush():
    if False:
        return 10
    im = Image.new('RGB', (W, H))
    draw = ImageDraw2.Draw(im)
    font = ImageDraw2.Font('white', FONT_PATH)
    draw.text((5, 5), 'ImageDraw2', font)
    im2 = draw.flush()
    assert_image_equal(im, im2)