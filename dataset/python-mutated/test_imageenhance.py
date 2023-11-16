import pytest
from PIL import Image, ImageEnhance
from .helper import assert_image_equal, hopper

def test_sanity():
    if False:
        return 10
    ImageEnhance.Color(hopper()).enhance(0.5)
    ImageEnhance.Contrast(hopper()).enhance(0.5)
    ImageEnhance.Brightness(hopper()).enhance(0.5)
    ImageEnhance.Sharpness(hopper()).enhance(0.5)

def test_crash():
    if False:
        i = 10
        return i + 15
    im = Image.new('RGB', (1, 1))
    ImageEnhance.Sharpness(im).enhance(0.5)

def _half_transparent_image():
    if False:
        print('Hello World!')
    im = hopper('RGB')
    transparent = Image.new('L', im.size, 0)
    solid = Image.new('L', (im.size[0] // 2, im.size[1]), 255)
    transparent.paste(solid, (0, 0))
    im.putalpha(transparent)
    return im

def _check_alpha(im, original, op, amount):
    if False:
        return 10
    assert im.getbands() == original.getbands()
    assert_image_equal(im.getchannel('A'), original.getchannel('A'), f'Diff on {op}: {amount}')

@pytest.mark.parametrize('op', ('Color', 'Brightness', 'Contrast', 'Sharpness'))
def test_alpha(op):
    if False:
        i = 10
        return i + 15
    original = _half_transparent_image()
    for amount in [0, 0.5, 1.0]:
        _check_alpha(getattr(ImageEnhance, op)(original).enhance(amount), original, op, amount)