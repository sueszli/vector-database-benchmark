import pytest
from PIL import BufrStubImagePlugin, Image
from .helper import hopper
TEST_FILE = 'Tests/images/gfs.t06z.rassda.tm00.bufr_d'

def test_open():
    if False:
        while True:
            i = 10
    with Image.open(TEST_FILE) as im:
        assert im.format == 'BUFR'
        assert im.mode == 'F'
        assert im.size == (1, 1)

def test_invalid_file():
    if False:
        for i in range(10):
            print('nop')
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        BufrStubImagePlugin.BufrStubImageFile(invalid_file)

def test_load():
    if False:
        while True:
            i = 10
    with Image.open(TEST_FILE) as im:
        with pytest.raises(OSError):
            im.load()

def test_save(tmp_path):
    if False:
        return 10
    im = hopper()
    tmpfile = str(tmp_path / 'temp.bufr')
    with pytest.raises(OSError):
        im.save(tmpfile)

def test_handler(tmp_path):
    if False:
        i = 10
        return i + 15

    class TestHandler:
        opened = False
        loaded = False
        saved = False

        def open(self, im):
            if False:
                for i in range(10):
                    print('nop')
            self.opened = True

        def load(self, im):
            if False:
                print('Hello World!')
            self.loaded = True
            im.fp.close()
            return Image.new('RGB', (1, 1))

        def save(self, im, fp, filename):
            if False:
                print('Hello World!')
            self.saved = True
    handler = TestHandler()
    BufrStubImagePlugin.register_handler(handler)
    with Image.open(TEST_FILE) as im:
        assert handler.opened
        assert not handler.loaded
        im.load()
        assert handler.loaded
        temp_file = str(tmp_path / 'temp.bufr')
        im.save(temp_file)
        assert handler.saved
    BufrStubImagePlugin._handler = None