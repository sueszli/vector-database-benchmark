import warnings
import pytest
from PIL import FliImagePlugin, Image
from .helper import assert_image_equal, assert_image_equal_tofile, is_pypy
static_test_file = 'Tests/images/hopper.fli'
animated_test_file = 'Tests/images/a.fli'

def test_sanity():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(static_test_file) as im:
        im.load()
        assert im.mode == 'P'
        assert im.size == (128, 128)
        assert im.format == 'FLI'
        assert not im.is_animated
    with Image.open(animated_test_file) as im:
        assert im.mode == 'P'
        assert im.size == (320, 200)
        assert im.format == 'FLI'
        assert im.info['duration'] == 71
        assert im.is_animated

@pytest.mark.skipif(is_pypy(), reason='Requires CPython')
def test_unclosed_file():
    if False:
        while True:
            i = 10

    def open():
        if False:
            for i in range(10):
                print('nop')
        im = Image.open(static_test_file)
        im.load()
    with pytest.warns(ResourceWarning):
        open()

def test_closed_file():
    if False:
        while True:
            i = 10
    with warnings.catch_warnings():
        im = Image.open(static_test_file)
        im.load()
        im.close()

def test_seek_after_close():
    if False:
        return 10
    im = Image.open(animated_test_file)
    im.seek(1)
    im.close()
    with pytest.raises(ValueError):
        im.seek(0)

def test_context_manager():
    if False:
        i = 10
        return i + 15
    with warnings.catch_warnings():
        with Image.open(static_test_file) as im:
            im.load()

def test_tell():
    if False:
        i = 10
        return i + 15
    with Image.open(static_test_file) as im:
        frame = im.tell()
        assert frame == 0

def test_invalid_file():
    if False:
        while True:
            i = 10
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        FliImagePlugin.FliImageFile(invalid_file)

def test_palette_chunk_second():
    if False:
        while True:
            i = 10
    with Image.open('Tests/images/hopper_palette_chunk_second.fli') as im:
        with Image.open(static_test_file) as expected:
            assert_image_equal(im.convert('RGB'), expected.convert('RGB'))

def test_n_frames():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(static_test_file) as im:
        assert im.n_frames == 1
        assert not im.is_animated
    with Image.open(animated_test_file) as im:
        assert im.n_frames == 384
        assert im.is_animated

def test_eoferror():
    if False:
        print('Hello World!')
    with Image.open(animated_test_file) as im:
        n_frames = im.n_frames
        with pytest.raises(EOFError):
            im.seek(n_frames)
        assert im.tell() < n_frames
        im.seek(n_frames - 1)

def test_seek_tell():
    if False:
        while True:
            i = 10
    with Image.open(animated_test_file) as im:
        layer_number = im.tell()
        assert layer_number == 0
        im.seek(0)
        layer_number = im.tell()
        assert layer_number == 0
        im.seek(1)
        layer_number = im.tell()
        assert layer_number == 1
        im.seek(2)
        layer_number = im.tell()
        assert layer_number == 2
        im.seek(1)
        layer_number = im.tell()
        assert layer_number == 1

def test_seek():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(animated_test_file) as im:
        im.seek(50)
        assert_image_equal_tofile(im, 'Tests/images/a_fli.png')

@pytest.mark.parametrize('test_file', ['Tests/images/timeout-9139147ce93e20eb14088fe238e541443ffd64b3.fli', 'Tests/images/timeout-bff0a9dc7243a8e6ede2408d2ffa6a9964698b87.fli'])
@pytest.mark.timeout(timeout=3)
def test_timeouts(test_file):
    if False:
        print('Hello World!')
    with open(test_file, 'rb') as f:
        with Image.open(f) as im:
            with pytest.raises(OSError):
                im.load()

@pytest.mark.parametrize('test_file', ['Tests/images/crash-5762152299364352.fli'])
def test_crash(test_file):
    if False:
        i = 10
        return i + 15
    with open(test_file, 'rb') as f:
        with Image.open(f) as im:
            with pytest.raises(OSError):
                im.load()