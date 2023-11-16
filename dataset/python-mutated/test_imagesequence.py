import pytest
from PIL import Image, ImageSequence, TiffImagePlugin
from .helper import assert_image_equal, hopper, skip_unless_feature

def test_sanity(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    test_file = str(tmp_path / 'temp.im')
    im = hopper('RGB')
    im.save(test_file)
    seq = ImageSequence.Iterator(im)
    index = 0
    for frame in seq:
        assert_image_equal(im, frame)
        assert im.tell() == index
        index += 1
    assert index == 1
    with pytest.raises(AttributeError):
        ImageSequence.Iterator(0)

def test_iterator():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/multipage.tiff') as im:
        i = ImageSequence.Iterator(im)
        for index in range(0, im.n_frames):
            assert i[index] == next(i)
        with pytest.raises(IndexError):
            i[index + 1]
        with pytest.raises(StopIteration):
            next(i)

def test_iterator_min_frame():
    if False:
        i = 10
        return i + 15
    with Image.open('Tests/images/hopper.psd') as im:
        i = ImageSequence.Iterator(im)
        for index in range(1, im.n_frames):
            assert i[index] == next(i)

def _test_multipage_tiff():
    if False:
        return 10
    with Image.open('Tests/images/multipage.tiff') as im:
        for (index, frame) in enumerate(ImageSequence.Iterator(im)):
            frame.load()
            assert index == im.tell()
            frame.convert('RGB')

def test_tiff():
    if False:
        while True:
            i = 10
    _test_multipage_tiff()

@skip_unless_feature('libtiff')
def test_libtiff():
    if False:
        print('Hello World!')
    TiffImagePlugin.READ_LIBTIFF = True
    _test_multipage_tiff()
    TiffImagePlugin.READ_LIBTIFF = False

def test_consecutive():
    if False:
        return 10
    with Image.open('Tests/images/multipage.tiff') as im:
        first_frame = None
        for frame in ImageSequence.Iterator(im):
            if first_frame is None:
                first_frame = frame.copy()
        for frame in ImageSequence.Iterator(im):
            assert_image_equal(frame, first_frame)
            break

def test_palette_mmap():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/multipage-mmap.tiff') as im:
        color1 = im.getpalette()[:3]
        im.seek(0)
        color2 = im.getpalette()[:3]
        assert color1 == color2

def test_all_frames():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/iss634.gif') as im:
        ims = ImageSequence.all_frames(im)
        assert len(ims) == 42
        for (i, im_frame) in enumerate(ims):
            assert im_frame is not im
            im.seek(i)
            assert_image_equal(im, im_frame)
        ims = ImageSequence.all_frames([im, hopper(), im])
        assert len(ims) == 85
        ims = ImageSequence.all_frames(im, lambda im_frame: im_frame.rotate(90))
        for (i, im_frame) in enumerate(ims):
            im.seek(i)
            assert_image_equal(im.rotate(90), im_frame)