import pytest
from PIL import Image
from .helper import assert_image_equal, assert_image_similar, fromstring, hopper, skip_unless_feature, tostring

def test_sanity():
    if False:
        for i in range(10):
            print('nop')
    im = hopper()
    assert im.thumbnail((100, 100)) is None
    assert im.size == (100, 100)

def test_aspect():
    if False:
        return 10
    im = Image.new('L', (128, 128))
    im.thumbnail((100, 100))
    assert im.size == (100, 100)
    im = Image.new('L', (128, 256))
    im.thumbnail((100, 100))
    assert im.size == (50, 100)
    im = Image.new('L', (128, 256))
    im.thumbnail((50, 100))
    assert im.size == (50, 100)
    im = Image.new('L', (256, 128))
    im.thumbnail((100, 100))
    assert im.size == (100, 50)
    im = Image.new('L', (256, 128))
    im.thumbnail((100, 50))
    assert im.size == (100, 50)
    im = Image.new('L', (64, 64))
    im.thumbnail((100, 100))
    assert im.size == (64, 64)
    im = Image.new('L', (256, 162))
    im.thumbnail((33, 33))
    assert im.size == (33, 21)
    im = Image.new('L', (162, 256))
    im.thumbnail((33, 33))
    assert im.size == (21, 33)
    im = Image.new('L', (145, 100))
    im.thumbnail((50, 50))
    assert im.size == (50, 34)
    im = Image.new('L', (100, 145))
    im.thumbnail((50, 50))
    assert im.size == (34, 50)
    im = Image.new('L', (100, 30))
    im.thumbnail((75, 75))
    assert im.size == (75, 23)

def test_division_by_zero():
    if False:
        while True:
            i = 10
    im = Image.new('L', (200, 2))
    im.thumbnail((75, 75))
    assert im.size == (75, 1)

def test_float():
    if False:
        print('Hello World!')
    im = Image.new('L', (128, 128))
    im.thumbnail((99.9, 99.9))
    assert im.size == (99, 99)

def test_no_resize():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/hopper.jpg') as im:
        im.draft(None, (64, 64))
        assert im.size == (64, 64)
    with Image.open('Tests/images/hopper.jpg') as im:
        im.thumbnail((64, 64))
        assert im.size == (64, 64)

@skip_unless_feature('libtiff')
def test_load_first():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/g4_orientation_5.tif') as im:
        im.thumbnail((64, 64))
        assert im.size == (64, 10)
    with Image.open('Tests/images/g4_orientation_5.tif') as im:
        im.thumbnail((590, 88), reducing_gap=None)
        assert im.size == (590, 88)

def test_load_first_unless_jpeg():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/hopper.jpg') as im:
        draft = im.draft

        def im_draft(mode, size):
            if False:
                while True:
                    i = 10
            result = draft(mode, size)
            assert result is not None
            return result
        im.draft = im_draft
        im.thumbnail((64, 64))

@pytest.mark.valgrind_known_error(reason='Known Failing')
def test_DCT_scaling_edges():
    if False:
        return 10
    im = Image.new('RGB', (257, 257), 'red')
    im.paste(Image.new('RGB', (235, 235)), (11, 11))
    thumb = fromstring(tostring(im, 'JPEG', quality=99, subsampling=0))
    thumb.thumbnail((32, 32), Image.Resampling.BICUBIC, reducing_gap=1.0)
    ref = im.resize((32, 32), Image.Resampling.BICUBIC)
    assert_image_similar(thumb, ref, 1.5)

def test_reducing_gap_values():
    if False:
        for i in range(10):
            print('nop')
    im = hopper()
    im.thumbnail((18, 18), Image.Resampling.BICUBIC)
    ref = hopper()
    ref.thumbnail((18, 18), Image.Resampling.BICUBIC, reducing_gap=2.0)
    assert_image_equal(ref, im)
    ref = hopper()
    ref.thumbnail((18, 18), Image.Resampling.BICUBIC, reducing_gap=None)
    with pytest.raises(pytest.fail.Exception):
        assert_image_equal(ref, im)
    assert_image_similar(ref, im, 3.5)

def test_reducing_gap_for_DCT_scaling():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/hopper.jpg') as ref:
        ref.draft(None, (18 * 3, 18 * 3))
        ref = ref.resize((18, 18), Image.Resampling.BICUBIC)
        with Image.open('Tests/images/hopper.jpg') as im:
            im.thumbnail((18, 18), Image.Resampling.BICUBIC, reducing_gap=3.0)
            assert_image_similar(ref, im, 1.4)