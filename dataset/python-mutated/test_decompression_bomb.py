import pytest
from PIL import Image
from .helper import hopper
TEST_FILE = 'Tests/images/hopper.ppm'
ORIGINAL_LIMIT = Image.MAX_IMAGE_PIXELS

class TestDecompressionBomb:

    def teardown_method(self, method):
        if False:
            while True:
                i = 10
        Image.MAX_IMAGE_PIXELS = ORIGINAL_LIMIT

    def test_no_warning_small_file(self):
        if False:
            print('Hello World!')
        with Image.open(TEST_FILE):
            pass

    def test_no_warning_no_limit(self):
        if False:
            i = 10
            return i + 15
        Image.MAX_IMAGE_PIXELS = None
        assert Image.MAX_IMAGE_PIXELS is None
        with Image.open(TEST_FILE):
            pass

    def test_warning(self):
        if False:
            i = 10
            return i + 15
        Image.MAX_IMAGE_PIXELS = 128 * 128 - 1
        assert Image.MAX_IMAGE_PIXELS == 128 * 128 - 1
        with pytest.warns(Image.DecompressionBombWarning):
            with Image.open(TEST_FILE):
                pass

    def test_exception(self):
        if False:
            while True:
                i = 10
        Image.MAX_IMAGE_PIXELS = 64 * 128 - 1
        assert Image.MAX_IMAGE_PIXELS == 64 * 128 - 1
        with pytest.raises(Image.DecompressionBombError):
            with Image.open(TEST_FILE):
                pass

    def test_exception_ico(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(Image.DecompressionBombError):
            with Image.open('Tests/images/decompression_bomb.ico'):
                pass

    def test_exception_gif(self):
        if False:
            print('Hello World!')
        with pytest.raises(Image.DecompressionBombError):
            with Image.open('Tests/images/decompression_bomb.gif'):
                pass

    def test_exception_gif_extents(self):
        if False:
            while True:
                i = 10
        with Image.open('Tests/images/decompression_bomb_extents.gif') as im:
            with pytest.raises(Image.DecompressionBombError):
                im.seek(1)

    def test_exception_gif_zero_width(self):
        if False:
            while True:
                i = 10
        Image.MAX_IMAGE_PIXELS = 4 * 64 * 128
        assert Image.MAX_IMAGE_PIXELS == 4 * 64 * 128
        with pytest.raises(Image.DecompressionBombError):
            with Image.open('Tests/images/zero_width.gif'):
                pass

    def test_exception_bmp(self):
        if False:
            while True:
                i = 10
        with pytest.raises(Image.DecompressionBombError):
            with Image.open('Tests/images/bmp/b/reallybig.bmp'):
                pass

class TestDecompressionCrop:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        (width, height) = (128, 128)
        Image.MAX_IMAGE_PIXELS = height * width * 4 - 1

    @classmethod
    def teardown_class(cls):
        if False:
            for i in range(10):
                print('nop')
        Image.MAX_IMAGE_PIXELS = ORIGINAL_LIMIT

    def test_enlarge_crop(self):
        if False:
            print('Hello World!')
        with hopper() as src:
            box = (0, 0, src.width * 2, src.height * 2)
            with pytest.warns(Image.DecompressionBombWarning):
                src.crop(box)

    def test_crop_decompression_checks(self):
        if False:
            print('Hello World!')
        im = Image.new('RGB', (100, 100))
        for value in ((-9999, -9999, -9990, -9990), (-999, -999, -990, -990)):
            assert im.crop(value).size == (9, 9)
        with pytest.warns(Image.DecompressionBombWarning):
            im.crop((-160, -160, 99, 99))
        with pytest.raises(Image.DecompressionBombError):
            im.crop((-99909, -99990, 99999, 99999))