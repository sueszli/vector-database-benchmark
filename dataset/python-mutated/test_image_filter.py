import pytest
from PIL import Image, ImageFilter
from .helper import assert_image_equal, hopper

@pytest.mark.parametrize('filter_to_apply', (ImageFilter.BLUR, ImageFilter.CONTOUR, ImageFilter.DETAIL, ImageFilter.EDGE_ENHANCE, ImageFilter.EDGE_ENHANCE_MORE, ImageFilter.EMBOSS, ImageFilter.FIND_EDGES, ImageFilter.SMOOTH, ImageFilter.SMOOTH_MORE, ImageFilter.SHARPEN, ImageFilter.MaxFilter, ImageFilter.MedianFilter, ImageFilter.MinFilter, ImageFilter.ModeFilter, ImageFilter.GaussianBlur, ImageFilter.GaussianBlur(0), ImageFilter.GaussianBlur(5), ImageFilter.GaussianBlur((2, 5)), ImageFilter.BoxBlur(0), ImageFilter.BoxBlur(5), ImageFilter.BoxBlur((2, 5)), ImageFilter.UnsharpMask, ImageFilter.UnsharpMask(10)))
@pytest.mark.parametrize('mode', ('L', 'I', 'RGB', 'CMYK'))
def test_sanity(filter_to_apply, mode):
    if False:
        return 10
    im = hopper(mode)
    if mode != 'I' or isinstance(filter_to_apply, ImageFilter.BuiltinFilter):
        out = im.filter(filter_to_apply)
        assert out.mode == im.mode
        assert out.size == im.size

@pytest.mark.parametrize('mode', ('L', 'I', 'RGB', 'CMYK'))
def test_sanity_error(mode):
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        im = hopper(mode)
        im.filter('hello')

@pytest.mark.parametrize('size', ((1, 1), (2, 2), (3, 3)))
def test_crash(size):
    if False:
        print('Hello World!')
    im = Image.new('RGB', size)
    im.filter(ImageFilter.SMOOTH)

@pytest.mark.parametrize('mode, expected', (('1', (4, 0)), ('L', (4, 0)), ('P', (4, 0)), ('RGB', ((4, 0, 0), (0, 0, 0)))))
def test_modefilter(mode, expected):
    if False:
        print('Hello World!')
    im = Image.new(mode, (3, 3), None)
    im.putdata(list(range(9)))
    mod = im.filter(ImageFilter.ModeFilter).getpixel((1, 1))
    im.putdata([0, 0, 1, 2, 5, 1, 5, 2, 0])
    mod2 = im.filter(ImageFilter.ModeFilter).getpixel((1, 1))
    assert (mod, mod2) == expected

@pytest.mark.parametrize('mode, expected', (('1', (0, 4, 8)), ('L', (0, 4, 8)), ('RGB', ((0, 0, 0), (4, 0, 0), (8, 0, 0))), ('I', (0, 4, 8)), ('F', (0.0, 4.0, 8.0))))
def test_rankfilter(mode, expected):
    if False:
        return 10
    im = Image.new(mode, (3, 3), None)
    im.putdata(list(range(9)))
    minimum = im.filter(ImageFilter.MinFilter).getpixel((1, 1))
    med = im.filter(ImageFilter.MedianFilter).getpixel((1, 1))
    maximum = im.filter(ImageFilter.MaxFilter).getpixel((1, 1))
    assert (minimum, med, maximum) == expected

@pytest.mark.parametrize('filter', (ImageFilter.MinFilter, ImageFilter.MedianFilter, ImageFilter.MaxFilter))
def test_rankfilter_error(filter):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        im = Image.new('P', (3, 3), None)
        im.putdata(list(range(9)))
        im.filter(filter).getpixel((1, 1))

def test_rankfilter_properties():
    if False:
        while True:
            i = 10
    rankfilter = ImageFilter.RankFilter(1, 2)
    assert rankfilter.size == 1
    assert rankfilter.rank == 2

def test_builtinfilter_p():
    if False:
        i = 10
        return i + 15
    builtin_filter = ImageFilter.BuiltinFilter()
    with pytest.raises(ValueError):
        builtin_filter.filter(hopper('P'))

def test_kernel_not_enough_coefficients():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        ImageFilter.Kernel((3, 3), (0, 0))

@pytest.mark.parametrize('mode', ('L', 'LA', 'I', 'RGB', 'CMYK'))
def test_consistency_3x3(mode):
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/hopper.bmp') as source:
        reference_name = 'hopper_emboss'
        reference_name += '_I.png' if mode == 'I' else '.bmp'
        with Image.open('Tests/images/' + reference_name) as reference:
            kernel = ImageFilter.Kernel((3, 3), (-1, -1, 0, -1, 0, 1, 0, 1, 1), 0.3)
            source = source.split() * 2
            reference = reference.split() * 2
            if mode == 'I':
                source = source[0].convert(mode)
            else:
                source = Image.merge(mode, source[:len(mode)])
            reference = Image.merge(mode, reference[:len(mode)])
            assert_image_equal(source.filter(kernel), reference)

@pytest.mark.parametrize('mode', ('L', 'LA', 'I', 'RGB', 'CMYK'))
def test_consistency_5x5(mode):
    if False:
        print('Hello World!')
    with Image.open('Tests/images/hopper.bmp') as source:
        reference_name = 'hopper_emboss_more'
        reference_name += '_I.png' if mode == 'I' else '.bmp'
        with Image.open('Tests/images/' + reference_name) as reference:
            kernel = ImageFilter.Kernel((5, 5), (-1, -1, -1, -1, 0, -1, -1, -1, 0, 1, -1, -1, 0, 1, 1, -1, 0, 1, 1, 1, 0, 1, 1, 1, 1), 0.3)
            source = source.split() * 2
            reference = reference.split() * 2
            if mode == 'I':
                source = source[0].convert(mode)
            else:
                source = Image.merge(mode, source[:len(mode)])
            reference = Image.merge(mode, reference[:len(mode)])
            assert_image_equal(source.filter(kernel), reference)

@pytest.mark.parametrize('radius', (-2, (-2, -2), (-2, 2), (2, -2)))
def test_invalid_box_blur_filter(radius):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        ImageFilter.BoxBlur(radius)
    im = hopper()
    box_blur_filter = ImageFilter.BoxBlur(2)
    box_blur_filter.radius = radius
    with pytest.raises(ValueError):
        im.filter(box_blur_filter)