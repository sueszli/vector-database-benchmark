import pytest
from PIL import ImageWin
from .helper import hopper, is_win32

class TestImageWin:

    def test_sanity(self):
        if False:
            print('Hello World!')
        dir(ImageWin)

    def test_hdc(self):
        if False:
            while True:
                i = 10
        dc = 50
        hdc = ImageWin.HDC(dc)
        dc2 = int(hdc)
        assert dc2 == 50

    def test_hwnd(self):
        if False:
            i = 10
            return i + 15
        wnd = 50
        hwnd = ImageWin.HWND(wnd)
        wnd2 = int(hwnd)
        assert wnd2 == 50

@pytest.mark.skipif(not is_win32(), reason='Windows only')
class TestImageWinDib:

    def test_dib_image(self):
        if False:
            for i in range(10):
                print('nop')
        im = hopper()
        dib = ImageWin.Dib(im)
        assert dib.size == im.size

    def test_dib_mode_string(self):
        if False:
            for i in range(10):
                print('nop')
        mode = 'RGBA'
        size = (128, 128)
        dib = ImageWin.Dib(mode, size)
        assert dib.size == (128, 128)

    def test_dib_paste(self):
        if False:
            return 10
        im = hopper()
        mode = 'RGBA'
        size = (128, 128)
        dib = ImageWin.Dib(mode, size)
        dib.paste(im)
        assert dib.size == (128, 128)

    def test_dib_paste_bbox(self):
        if False:
            while True:
                i = 10
        im = hopper()
        bbox = (0, 0, 10, 10)
        mode = 'RGBA'
        size = (128, 128)
        dib = ImageWin.Dib(mode, size)
        dib.paste(im, bbox)
        assert dib.size == (128, 128)

    def test_dib_frombytes_tobytes_roundtrip(self):
        if False:
            while True:
                i = 10
        im = hopper()
        dib1 = ImageWin.Dib(im)
        mode = 'RGB'
        size = (128, 128)
        dib2 = ImageWin.Dib(mode, size)
        assert dib1.tobytes() != dib2.tobytes()
        test_buffer = dib1.tobytes()
        for datatype in ('bytes', 'memoryview'):
            if datatype == 'memoryview':
                test_buffer = memoryview(test_buffer)
            dib2.frombytes(test_buffer)
            assert dib1.tobytes() == dib2.tobytes()