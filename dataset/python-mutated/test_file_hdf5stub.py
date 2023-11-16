import pytest
from PIL import Hdf5StubImagePlugin, Image
TEST_FILE = 'Tests/images/hdf5.h5'

def test_open():
    if False:
        while True:
            i = 10
    with Image.open(TEST_FILE) as im:
        assert im.format == 'HDF5'
        assert im.mode == 'F'
        assert im.size == (1, 1)

def test_invalid_file():
    if False:
        i = 10
        return i + 15
    invalid_file = 'Tests/images/flower.jpg'
    with pytest.raises(SyntaxError):
        Hdf5StubImagePlugin.HDF5StubImageFile(invalid_file)

def test_load():
    if False:
        for i in range(10):
            print('nop')
    with Image.open(TEST_FILE) as im:
        with pytest.raises(OSError):
            im.load()

def test_save():
    if False:
        print('Hello World!')
    with Image.open(TEST_FILE) as im:
        dummy_fp = None
        dummy_filename = 'dummy.filename'
        with pytest.raises(OSError):
            im.save(dummy_filename)
        with pytest.raises(OSError):
            Hdf5StubImagePlugin._save(im, dummy_fp, dummy_filename)

def test_handler(tmp_path):
    if False:
        print('Hello World!')

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
                return 10
            self.loaded = True
            im.fp.close()
            return Image.new('RGB', (1, 1))

        def save(self, im, fp, filename):
            if False:
                i = 10
                return i + 15
            self.saved = True
    handler = TestHandler()
    Hdf5StubImagePlugin.register_handler(handler)
    with Image.open(TEST_FILE) as im:
        assert handler.opened
        assert not handler.loaded
        im.load()
        assert handler.loaded
        temp_file = str(tmp_path / 'temp.h5')
        im.save(temp_file)
        assert handler.saved
    Hdf5StubImagePlugin._handler = None