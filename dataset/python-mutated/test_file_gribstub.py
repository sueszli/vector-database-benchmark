import pytest
from PIL import GribStubImagePlugin, Image
from .helper import hopper
TEST_FILE = 'Tests/images/WAlaska.wind.7days.grb'

def test_open():
    if False:
        print('Hello World!')
    with Image.open(TEST_FILE) as im:
        assert im.format == 'GRIB'
        assert im.mode == 'F'
        assert im.size == (1, 1)

def test_invalid_file():
    if False:
        return 10
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        GribStubImagePlugin.GribStubImageFile(invalid_file)

def test_load():
    if False:
        while True:
            i = 10
    with Image.open(TEST_FILE) as im:
        with pytest.raises(OSError):
            im.load()

def test_save(tmp_path):
    if False:
        i = 10
        return i + 15
    im = hopper()
    tmpfile = str(tmp_path / 'temp.grib')
    with pytest.raises(OSError):
        im.save(tmpfile)

def test_handler(tmp_path):
    if False:
        print('Hello World!')

    class TestHandler:
        opened = False
        loaded = False
        saved = False

        def open(self, im):
            if False:
                return 10
            self.opened = True

        def load(self, im):
            if False:
                while True:
                    i = 10
            self.loaded = True
            im.fp.close()
            return Image.new('RGB', (1, 1))

        def save(self, im, fp, filename):
            if False:
                for i in range(10):
                    print('nop')
            self.saved = True
    handler = TestHandler()
    GribStubImagePlugin.register_handler(handler)
    with Image.open(TEST_FILE) as im:
        assert handler.opened
        assert not handler.loaded
        im.load()
        assert handler.loaded
        temp_file = str(tmp_path / 'temp.grib')
        im.save(temp_file)
        assert handler.saved
    GribStubImagePlugin._handler = None