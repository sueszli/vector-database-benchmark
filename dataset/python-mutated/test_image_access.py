import os
import subprocess
import sys
import sysconfig
import pytest
from PIL import Image
from .helper import assert_image_equal, hopper, is_win32
if os.environ.get('PYTHONOPTIMIZE') == '2':
    cffi = None
else:
    try:
        import cffi
        from PIL import PyAccess
    except ImportError:
        cffi = None
try:
    import numpy
except ImportError:
    numpy = None

class AccessTest:
    _init_cffi_access = Image.USE_CFFI_ACCESS
    _need_cffi_access = False

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        Image.USE_CFFI_ACCESS = cls._need_cffi_access

    @classmethod
    def teardown_class(cls):
        if False:
            print('Hello World!')
        Image.USE_CFFI_ACCESS = cls._init_cffi_access

class TestImagePutPixel(AccessTest):

    def test_sanity(self):
        if False:
            i = 10
            return i + 15
        im1 = hopper()
        im2 = Image.new(im1.mode, im1.size, 0)
        for y in range(im1.size[1]):
            for x in range(im1.size[0]):
                pos = (x, y)
                im2.putpixel(pos, im1.getpixel(pos))
        assert_image_equal(im1, im2)
        im2 = Image.new(im1.mode, im1.size, 0)
        im2.readonly = 1
        for y in range(im1.size[1]):
            for x in range(im1.size[0]):
                pos = (x, y)
                im2.putpixel(pos, im1.getpixel(pos))
        assert not im2.readonly
        assert_image_equal(im1, im2)
        im2 = Image.new(im1.mode, im1.size, 0)
        pix1 = im1.load()
        pix2 = im2.load()
        for (x, y) in ((0, '0'), ('0', 0)):
            with pytest.raises(TypeError):
                pix1[x, y]
        for y in range(im1.size[1]):
            for x in range(im1.size[0]):
                pix2[x, y] = pix1[x, y]
        assert_image_equal(im1, im2)

    def test_sanity_negative_index(self):
        if False:
            i = 10
            return i + 15
        im1 = hopper()
        im2 = Image.new(im1.mode, im1.size, 0)
        (width, height) = im1.size
        assert im1.getpixel((0, 0)) == im1.getpixel((-width, -height))
        assert im1.getpixel((-1, -1)) == im1.getpixel((width - 1, height - 1))
        for y in range(-1, -im1.size[1] - 1, -1):
            for x in range(-1, -im1.size[0] - 1, -1):
                pos = (x, y)
                im2.putpixel(pos, im1.getpixel(pos))
        assert_image_equal(im1, im2)
        im2 = Image.new(im1.mode, im1.size, 0)
        im2.readonly = 1
        for y in range(-1, -im1.size[1] - 1, -1):
            for x in range(-1, -im1.size[0] - 1, -1):
                pos = (x, y)
                im2.putpixel(pos, im1.getpixel(pos))
        assert not im2.readonly
        assert_image_equal(im1, im2)
        im2 = Image.new(im1.mode, im1.size, 0)
        pix1 = im1.load()
        pix2 = im2.load()
        for y in range(-1, -im1.size[1] - 1, -1):
            for x in range(-1, -im1.size[0] - 1, -1):
                pix2[x, y] = pix1[x, y]
        assert_image_equal(im1, im2)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_numpy(self):
        if False:
            i = 10
            return i + 15
        im = hopper()
        pix = im.load()
        assert pix[numpy.int32(1), numpy.int32(2)] == (18, 20, 59)

class TestImageGetPixel(AccessTest):

    @staticmethod
    def color(mode):
        if False:
            return 10
        bands = Image.getmodebands(mode)
        if bands == 1:
            return 1
        if mode in ('BGR;15', 'BGR;16'):
            return (16, 32, 49)
        return tuple(range(1, bands + 1))

    def check(self, mode, expected_color=None):
        if False:
            return 10
        if self._need_cffi_access and mode.startswith('BGR;'):
            pytest.skip('Support not added to deprecated module for BGR;* modes')
        if not expected_color:
            expected_color = self.color(mode)
        im = Image.new(mode, (1, 1), None)
        im.putpixel((0, 0), expected_color)
        actual_color = im.getpixel((0, 0))
        assert actual_color == expected_color, f'put/getpixel roundtrip failed for mode {mode}, expected {expected_color} got {actual_color}'
        im.putpixel((-1, -1), expected_color)
        actual_color = im.getpixel((-1, -1))
        assert actual_color == expected_color, f'put/getpixel roundtrip negative index failed for mode {mode}, expected {expected_color} got {actual_color}'
        im = Image.new(mode, (0, 0), None)
        assert im.load() is not None
        error = ValueError if self._need_cffi_access else IndexError
        with pytest.raises(error):
            im.putpixel((0, 0), expected_color)
        with pytest.raises(error):
            im.getpixel((0, 0))
        with pytest.raises(error):
            im.putpixel((-1, -1), expected_color)
        with pytest.raises(error):
            im.getpixel((-1, -1))
        im = Image.new(mode, (1, 1), expected_color)
        actual_color = im.getpixel((0, 0))
        assert actual_color == expected_color, f'initial color failed for mode {mode}, expected {expected_color} got {actual_color}'
        actual_color = im.getpixel((-1, -1))
        assert actual_color == expected_color, f'initial color failed with negative index for mode {mode}, expected {expected_color} got {actual_color}'
        im = Image.new(mode, (0, 0), expected_color)
        with pytest.raises(error):
            im.getpixel((0, 0))
        with pytest.raises(error):
            im.getpixel((-1, -1))

    @pytest.mark.parametrize('mode', ('1', 'L', 'LA', 'I', 'I;16', 'I;16B', 'F', 'P', 'PA', 'BGR;15', 'BGR;16', 'BGR;24', 'RGB', 'RGBA', 'RGBX', 'CMYK', 'YCbCr'))
    def test_basic(self, mode):
        if False:
            print('Hello World!')
        self.check(mode)

    def test_list(self):
        if False:
            while True:
                i = 10
        im = hopper()
        assert im.getpixel([0, 0]) == (20, 20, 70)

    @pytest.mark.parametrize('mode', ('I;16', 'I;16B'))
    @pytest.mark.parametrize('expected_color', (2 ** 15 - 1, 2 ** 15, 2 ** 15 + 1, 2 ** 16 - 1))
    def test_signedness(self, mode, expected_color):
        if False:
            while True:
                i = 10
        self.check(mode, expected_color)

    @pytest.mark.parametrize('mode', ('P', 'PA'))
    @pytest.mark.parametrize('color', ((255, 0, 0), (255, 0, 0, 255)))
    def test_p_putpixel_rgb_rgba(self, mode, color):
        if False:
            for i in range(10):
                print('nop')
        im = Image.new(mode, (1, 1))
        im.putpixel((0, 0), color)
        alpha = color[3] if len(color) == 4 and mode == 'PA' else 255
        assert im.convert('RGBA').getpixel((0, 0)) == (255, 0, 0, alpha)

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.skipif(cffi is None, reason='No CFFI')
class TestCffiPutPixel(TestImagePutPixel):
    _need_cffi_access = True

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.skipif(cffi is None, reason='No CFFI')
class TestCffiGetPixel(TestImageGetPixel):
    _need_cffi_access = True

@pytest.mark.skipif(cffi is None, reason='No CFFI')
class TestCffi(AccessTest):
    _need_cffi_access = True

    def _test_get_access(self, im):
        if False:
            i = 10
            return i + 15
        'Do we get the same thing as the old pixel access\n\n        Using private interfaces, forcing a capi access and\n        a pyaccess for the same image'
        caccess = im.im.pixel_access(False)
        with pytest.warns(DeprecationWarning):
            access = PyAccess.new(im, False)
        (w, h) = im.size
        for x in range(0, w, 10):
            for y in range(0, h, 10):
                assert access[x, y] == caccess[x, y]
        with pytest.raises(ValueError):
            access[access.xsize + 1, access.ysize + 1]

    def test_get_vs_c(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(DeprecationWarning):
            rgb = hopper('RGB')
            rgb.load()
            self._test_get_access(rgb)
            for mode in ('RGBA', 'L', 'LA', '1', 'P', 'F'):
                self._test_get_access(hopper(mode))
            for mode in ('I;16', 'I;16L', 'I;16B', 'I;16N', 'I'):
                im = Image.new(mode, (10, 10), 40000)
                self._test_get_access(im)

    def _test_set_access(self, im, color):
        if False:
            print('Hello World!')
        'Are we writing the correct bits into the image?\n\n        Using private interfaces, forcing a capi access and\n        a pyaccess for the same image'
        caccess = im.im.pixel_access(False)
        with pytest.warns(DeprecationWarning):
            access = PyAccess.new(im, False)
        (w, h) = im.size
        for x in range(0, w, 10):
            for y in range(0, h, 10):
                access[x, y] = color
                assert color == caccess[x, y]
        with pytest.warns(DeprecationWarning):
            access = PyAccess.new(im, True)
        with pytest.raises(ValueError):
            access[0, 0] = color

    def test_set_vs_c(self):
        if False:
            print('Hello World!')
        rgb = hopper('RGB')
        with pytest.warns(DeprecationWarning):
            rgb.load()
        self._test_set_access(rgb, (255, 128, 0))
        self._test_set_access(hopper('RGBA'), (255, 192, 128, 0))
        self._test_set_access(hopper('L'), 128)
        self._test_set_access(hopper('LA'), (128, 128))
        self._test_set_access(hopper('1'), 255)
        self._test_set_access(hopper('P'), 128)
        self._test_set_access(hopper('F'), 1024.0)
        for mode in ('I;16', 'I;16L', 'I;16B', 'I;16N', 'I'):
            im = Image.new(mode, (10, 10), 40000)
            self._test_set_access(im, 45000)

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_not_implemented(self):
        if False:
            print('Hello World!')
        assert PyAccess.new(hopper('BGR;15')) is None

    def test_reference_counting(self):
        if False:
            return 10
        size = 10
        for _ in range(10):
            with pytest.warns(DeprecationWarning):
                px = Image.new('L', (size, 1), 0).load()
            for i in range(size):
                assert px[i, 0] == 0

    @pytest.mark.parametrize('mode', ('P', 'PA'))
    def test_p_putpixel_rgb_rgba(self, mode):
        if False:
            for i in range(10):
                print('nop')
        for color in ((255, 0, 0), (255, 0, 0, 127 if mode == 'PA' else 255)):
            im = Image.new(mode, (1, 1))
            with pytest.warns(DeprecationWarning):
                access = PyAccess.new(im, False)
                access.putpixel((0, 0), color)
                if len(color) == 3:
                    color += (255,)
                assert im.convert('RGBA').getpixel((0, 0)) == color

class TestImagePutPixelError(AccessTest):
    IMAGE_MODES1 = ['LA', 'RGB', 'RGBA', 'BGR;15']
    IMAGE_MODES2 = ['L', 'I', 'I;16']
    INVALID_TYPES = ['foo', 1.0, None]

    @pytest.mark.parametrize('mode', IMAGE_MODES1)
    def test_putpixel_type_error1(self, mode):
        if False:
            return 10
        im = hopper(mode)
        for v in self.INVALID_TYPES:
            with pytest.raises(TypeError, match='color must be int or tuple'):
                im.putpixel((0, 0), v)

    @pytest.mark.parametrize(('mode', 'band_numbers', 'match'), (('L', (0, 2), 'color must be int or single-element tuple'), ('LA', (0, 3), 'color must be int, or tuple of one or two elements'), ('BGR;15', (0, 2), 'color must be int, or tuple of one or three elements'), ('RGB', (0, 2, 5), 'color must be int, or tuple of one, three or four elements')))
    def test_putpixel_invalid_number_of_bands(self, mode, band_numbers, match):
        if False:
            while True:
                i = 10
        im = hopper(mode)
        for band_number in band_numbers:
            with pytest.raises(TypeError, match=match):
                im.putpixel((0, 0), (0,) * band_number)

    @pytest.mark.parametrize('mode', IMAGE_MODES2)
    def test_putpixel_type_error2(self, mode):
        if False:
            i = 10
            return i + 15
        im = hopper(mode)
        for v in self.INVALID_TYPES:
            with pytest.raises(TypeError, match='color must be int or single-element tuple'):
                im.putpixel((0, 0), v)

    @pytest.mark.parametrize('mode', IMAGE_MODES1 + IMAGE_MODES2)
    def test_putpixel_overflow_error(self, mode):
        if False:
            for i in range(10):
                print('nop')
        im = hopper(mode)
        with pytest.raises(OverflowError):
            im.putpixel((0, 0), 2 ** 80)

class TestEmbeddable:

    @pytest.mark.xfail(reason='failing test')
    @pytest.mark.skipif(not is_win32(), reason='requires Windows')
    def test_embeddable(self):
        if False:
            for i in range(10):
                print('nop')
        import ctypes
        from setuptools.command.build_ext import new_compiler
        with open('embed_pil.c', 'w', encoding='utf-8') as fh:
            fh.write('\n#include "Python.h"\n\nint main(int argc, char* argv[])\n{\n    char *home = "%s";\n    wchar_t *whome = Py_DecodeLocale(home, NULL);\n    Py_SetPythonHome(whome);\n\n    Py_InitializeEx(0);\n    Py_DECREF(PyImport_ImportModule("PIL.Image"));\n    Py_Finalize();\n\n    Py_InitializeEx(0);\n    Py_DECREF(PyImport_ImportModule("PIL.Image"));\n    Py_Finalize();\n\n    PyMem_RawFree(whome);\n\n    return 0;\n}\n        ' % sys.prefix.replace('\\', '\\\\'))
        compiler = new_compiler()
        compiler.add_include_dir(sysconfig.get_config_var('INCLUDEPY'))
        libdir = sysconfig.get_config_var('LIBDIR') or sysconfig.get_config_var('INCLUDEPY').replace('include', 'libs')
        compiler.add_library_dir(libdir)
        objects = compiler.compile(['embed_pil.c'])
        compiler.link_executable(objects, 'embed_pil')
        env = os.environ.copy()
        env['PATH'] = sys.prefix + ';' + env['PATH']
        ctypes.windll.kernel32.SetErrorMode(2)
        process = subprocess.Popen(['embed_pil.exe'], env=env)
        process.communicate()
        assert process.returncode == 0