import warnings
from io import BytesIO
import pytest
from PIL import Image
from .helper import assert_image_equal, assert_image_similar, is_pypy, skip_unless_feature
test_files = ['Tests/images/sugarshack.mpo', 'Tests/images/frozenpond.mpo']
pytestmark = skip_unless_feature('jpg')

def roundtrip(im, **options):
    if False:
        return 10
    out = BytesIO()
    im.save(out, 'MPO', **options)
    test_bytes = out.tell()
    out.seek(0)
    im = Image.open(out)
    im.bytes = test_bytes
    return im

@pytest.mark.parametrize('test_file', test_files)
def test_sanity(test_file):
    if False:
        i = 10
        return i + 15
    with Image.open(test_file) as im:
        im.load()
        assert im.mode == 'RGB'
        assert im.size == (640, 480)
        assert im.format == 'MPO'

@pytest.mark.skipif(is_pypy(), reason='Requires CPython')
def test_unclosed_file():
    if False:
        print('Hello World!')

    def open():
        if False:
            return 10
        im = Image.open(test_files[0])
        im.load()
    with pytest.warns(ResourceWarning):
        open()

def test_closed_file():
    if False:
        print('Hello World!')
    with warnings.catch_warnings():
        im = Image.open(test_files[0])
        im.load()
        im.close()

def test_seek_after_close():
    if False:
        for i in range(10):
            print('nop')
    im = Image.open(test_files[0])
    im.close()
    with pytest.raises(ValueError):
        im.seek(1)

def test_context_manager():
    if False:
        while True:
            i = 10
    with warnings.catch_warnings():
        with Image.open(test_files[0]) as im:
            im.load()

@pytest.mark.parametrize('test_file', test_files)
def test_app(test_file):
    if False:
        i = 10
        return i + 15
    with Image.open(test_file) as im:
        assert im.applist[0][0] == 'APP1'
        assert im.applist[1][0] == 'APP2'
        assert im.applist[1][1][:16] == b'MPF\x00MM\x00*\x00\x00\x00\x08\x00\x03\xb0\x00'
        assert len(im.applist) == 2

@pytest.mark.parametrize('test_file', test_files)
def test_exif(test_file):
    if False:
        while True:
            i = 10
    with Image.open(test_file) as im_original:
        im_reloaded = roundtrip(im_original, save_all=True, exif=im_original.getexif())
    for im in (im_original, im_reloaded):
        info = im._getexif()
        assert info[272] == 'Nintendo 3DS'
        assert info[296] == 2
        assert info[34665] == 188

def test_frame_size():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/sugarshack_frame_size.mpo') as im:
        assert im.size == (640, 480)
        im.seek(1)
        assert im.size == (680, 480)
        im.seek(0)
        assert im.size == (640, 480)

def test_ignore_frame_size():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/ignore_frame_size.mpo') as im:
        assert im.size == (64, 64)
        im.seek(1)
        assert im.mpinfo[45058][1]['Attribute']['MPType'] == 'Multi-Frame Image: (Disparity)'
        assert im.size == (64, 64)

def test_parallax():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/sugarshack.mpo') as im:
        exif = im.getexif()
        assert exif.get_ifd(37500)[4353]['Parallax'] == -44.798187255859375
    with Image.open('Tests/images/fujifilm.mpo') as im:
        im.seek(1)
        exif = im.getexif()
        assert exif.get_ifd(37500)[45585] == -3.125

def test_reload_exif_after_seek():
    if False:
        return 10
    with Image.open('Tests/images/sugarshack.mpo') as im:
        exif = im.getexif()
        del exif[296]
        im.seek(1)
        assert 296 in exif

@pytest.mark.parametrize('test_file', test_files)
def test_mp(test_file):
    if False:
        print('Hello World!')
    with Image.open(test_file) as im:
        mpinfo = im._getmp()
        assert mpinfo[45056] == b'0100'
        assert mpinfo[45057] == 2

def test_mp_offset():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/sugarshack_ifd_offset.mpo') as im:
        mpinfo = im._getmp()
        assert mpinfo[45056] == b'0100'
        assert mpinfo[45057] == 2

def test_mp_no_data():
    if False:
        print('Hello World!')
    with Image.open('Tests/images/sugarshack_no_data.mpo') as im:
        with pytest.raises(ValueError):
            im.seek(1)

@pytest.mark.parametrize('test_file', test_files)
def test_mp_attribute(test_file):
    if False:
        for i in range(10):
            print('nop')
    with Image.open(test_file) as im:
        mpinfo = im._getmp()
    for (frame_number, mpentry) in enumerate(mpinfo[45058]):
        mpattr = mpentry['Attribute']
        if frame_number:
            assert not mpattr['RepresentativeImageFlag']
        else:
            assert mpattr['RepresentativeImageFlag']
        assert not mpattr['DependentParentImageFlag']
        assert not mpattr['DependentChildImageFlag']
        assert mpattr['ImageDataFormat'] == 'JPEG'
        assert mpattr['MPType'] == 'Multi-Frame Image: (Disparity)'
        assert mpattr['Reserved'] == 0

@pytest.mark.parametrize('test_file', test_files)
def test_seek(test_file):
    if False:
        print('Hello World!')
    with Image.open(test_file) as im:
        assert im.tell() == 0
        with pytest.raises(EOFError):
            im.seek(-1)
        with pytest.raises(EOFError):
            im.seek(-523)
        with pytest.raises(EOFError):
            im.seek(2)
        with pytest.raises(EOFError):
            im.seek(523)
        assert im.tell() == 0
        im.seek(1)
        assert im.tell() == 1
        im.seek(0)
        assert im.tell() == 0

def test_n_frames():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/sugarshack.mpo') as im:
        assert im.n_frames == 2
        assert im.is_animated

def test_eoferror():
    if False:
        for i in range(10):
            print('nop')
    with Image.open('Tests/images/sugarshack.mpo') as im:
        n_frames = im.n_frames
        with pytest.raises(EOFError):
            im.seek(n_frames)
        assert im.tell() < n_frames
        im.seek(n_frames - 1)

@pytest.mark.parametrize('test_file', test_files)
def test_image_grab(test_file):
    if False:
        while True:
            i = 10
    with Image.open(test_file) as im:
        assert im.tell() == 0
        im0 = im.tobytes()
        im.seek(1)
        assert im.tell() == 1
        im1 = im.tobytes()
        im.seek(0)
        assert im.tell() == 0
        im02 = im.tobytes()
        assert im0 == im02
        assert im0 != im1

@pytest.mark.parametrize('test_file', test_files)
def test_save(test_file):
    if False:
        i = 10
        return i + 15
    with Image.open(test_file) as im:
        assert im.tell() == 0
        jpg0 = roundtrip(im)
        assert_image_similar(im, jpg0, 30)
        im.seek(1)
        assert im.tell() == 1
        jpg1 = roundtrip(im)
        assert_image_similar(im, jpg1, 30)

def test_save_all():
    if False:
        return 10
    for test_file in test_files:
        with Image.open(test_file) as im:
            im_reloaded = roundtrip(im, save_all=True)
            im.seek(0)
            assert_image_similar(im, im_reloaded, 30)
            im.seek(1)
            im_reloaded.seek(1)
            assert_image_similar(im, im_reloaded, 30)
    im = Image.new('RGB', (1, 1))
    im2 = Image.new('RGB', (1, 1), '#f00')
    im_reloaded = roundtrip(im, save_all=True, append_images=[im2])
    assert_image_equal(im, im_reloaded)
    assert im_reloaded.mpinfo[45056] == b'0100'
    im_reloaded.seek(1)
    assert_image_similar(im2, im_reloaded, 1)
    jpg = roundtrip(im, save_all=True)
    assert 'mp' not in jpg.info