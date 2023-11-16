import logging
import sys
from ._deprecate import deprecate
try:
    from cffi import FFI
    defs = '\n    struct Pixel_RGBA {\n        unsigned char r,g,b,a;\n    };\n    struct Pixel_I16 {\n        unsigned char l,r;\n    };\n    '
    ffi = FFI()
    ffi.cdef(defs)
except ImportError as ex:
    from ._util import DeferredError
    FFI = ffi = DeferredError(ex)
logger = logging.getLogger(__name__)

class PyAccess:

    def __init__(self, img, readonly=False):
        if False:
            return 10
        deprecate('PyAccess', 11)
        vals = dict(img.im.unsafe_ptrs)
        self.readonly = readonly
        self.image8 = ffi.cast('unsigned char **', vals['image8'])
        self.image32 = ffi.cast('int **', vals['image32'])
        self.image = ffi.cast('unsigned char **', vals['image'])
        (self.xsize, self.ysize) = img.im.size
        self._img = img
        self._im = img.im
        if self._im.mode in ('P', 'PA'):
            self._palette = img.palette
        self._post_init()

    def _post_init(self):
        if False:
            return 10
        pass

    def __setitem__(self, xy, color):
        if False:
            print('Hello World!')
        '\n        Modifies the pixel at x,y. The color is given as a single\n        numerical value for single band images, and a tuple for\n        multi-band images\n\n        :param xy: The pixel coordinate, given as (x, y). See\n           :ref:`coordinate-system`.\n        :param color: The pixel value.\n        '
        if self.readonly:
            msg = 'Attempt to putpixel a read only image'
            raise ValueError(msg)
        (x, y) = xy
        if x < 0:
            x = self.xsize + x
        if y < 0:
            y = self.ysize + y
        (x, y) = self.check_xy((x, y))
        if self._im.mode in ('P', 'PA') and isinstance(color, (list, tuple)) and (len(color) in [3, 4]):
            if self._im.mode == 'PA':
                alpha = color[3] if len(color) == 4 else 255
                color = color[:3]
            color = self._palette.getcolor(color, self._img)
            if self._im.mode == 'PA':
                color = (color, alpha)
        return self.set_pixel(x, y, color)

    def __getitem__(self, xy):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the pixel at x,y. The pixel is returned as a single\n        value for single band images or a tuple for multiple band\n        images\n\n        :param xy: The pixel coordinate, given as (x, y). See\n          :ref:`coordinate-system`.\n        :returns: a pixel value for single band images, a tuple of\n          pixel values for multiband images.\n        '
        (x, y) = xy
        if x < 0:
            x = self.xsize + x
        if y < 0:
            y = self.ysize + y
        (x, y) = self.check_xy((x, y))
        return self.get_pixel(x, y)
    putpixel = __setitem__
    getpixel = __getitem__

    def check_xy(self, xy):
        if False:
            print('Hello World!')
        (x, y) = xy
        if not (0 <= x < self.xsize and 0 <= y < self.ysize):
            msg = 'pixel location out of range'
            raise ValueError(msg)
        return xy

class _PyAccess32_2(PyAccess):
    """PA, LA, stored in first and last bytes of a 32 bit word"""

    def _post_init(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.pixels = ffi.cast('struct Pixel_RGBA **', self.image32)

    def get_pixel(self, x, y):
        if False:
            i = 10
            return i + 15
        pixel = self.pixels[y][x]
        return (pixel.r, pixel.a)

    def set_pixel(self, x, y, color):
        if False:
            return 10
        pixel = self.pixels[y][x]
        pixel.r = min(color[0], 255)
        pixel.a = min(color[1], 255)

class _PyAccess32_3(PyAccess):
    """RGB and friends, stored in the first three bytes of a 32 bit word"""

    def _post_init(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.pixels = ffi.cast('struct Pixel_RGBA **', self.image32)

    def get_pixel(self, x, y):
        if False:
            while True:
                i = 10
        pixel = self.pixels[y][x]
        return (pixel.r, pixel.g, pixel.b)

    def set_pixel(self, x, y, color):
        if False:
            i = 10
            return i + 15
        pixel = self.pixels[y][x]
        pixel.r = min(color[0], 255)
        pixel.g = min(color[1], 255)
        pixel.b = min(color[2], 255)
        pixel.a = 255

class _PyAccess32_4(PyAccess):
    """RGBA etc, all 4 bytes of a 32 bit word"""

    def _post_init(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.pixels = ffi.cast('struct Pixel_RGBA **', self.image32)

    def get_pixel(self, x, y):
        if False:
            i = 10
            return i + 15
        pixel = self.pixels[y][x]
        return (pixel.r, pixel.g, pixel.b, pixel.a)

    def set_pixel(self, x, y, color):
        if False:
            return 10
        pixel = self.pixels[y][x]
        pixel.r = min(color[0], 255)
        pixel.g = min(color[1], 255)
        pixel.b = min(color[2], 255)
        pixel.a = min(color[3], 255)

class _PyAccess8(PyAccess):
    """1, L, P, 8 bit images stored as uint8"""

    def _post_init(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.pixels = self.image8

    def get_pixel(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.pixels[y][x] = min(color, 255)
        except TypeError:
            self.pixels[y][x] = min(color[0], 255)

class _PyAccessI16_N(PyAccess):
    """I;16 access, native bitendian without conversion"""

    def _post_init(self, *args, **kwargs):
        if False:
            return 10
        self.pixels = ffi.cast('unsigned short **', self.image)

    def get_pixel(self, x, y):
        if False:
            while True:
                i = 10
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        if False:
            while True:
                i = 10
        try:
            self.pixels[y][x] = min(color, 65535)
        except TypeError:
            self.pixels[y][x] = min(color[0], 65535)

class _PyAccessI16_L(PyAccess):
    """I;16L access, with conversion"""

    def _post_init(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.pixels = ffi.cast('struct Pixel_I16 **', self.image)

    def get_pixel(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        pixel = self.pixels[y][x]
        return pixel.l + pixel.r * 256

    def set_pixel(self, x, y, color):
        if False:
            for i in range(10):
                print('nop')
        pixel = self.pixels[y][x]
        try:
            color = min(color, 65535)
        except TypeError:
            color = min(color[0], 65535)
        pixel.l = color & 255
        pixel.r = color >> 8

class _PyAccessI16_B(PyAccess):
    """I;16B access, with conversion"""

    def _post_init(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.pixels = ffi.cast('struct Pixel_I16 **', self.image)

    def get_pixel(self, x, y):
        if False:
            i = 10
            return i + 15
        pixel = self.pixels[y][x]
        return pixel.l * 256 + pixel.r

    def set_pixel(self, x, y, color):
        if False:
            return 10
        pixel = self.pixels[y][x]
        try:
            color = min(color, 65535)
        except Exception:
            color = min(color[0], 65535)
        pixel.l = color >> 8
        pixel.r = color & 255

class _PyAccessI32_N(PyAccess):
    """Signed Int32 access, native endian"""

    def _post_init(self, *args, **kwargs):
        if False:
            return 10
        self.pixels = self.image32

    def get_pixel(self, x, y):
        if False:
            print('Hello World!')
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        if False:
            return 10
        self.pixels[y][x] = color

class _PyAccessI32_Swap(PyAccess):
    """I;32L/B access, with byteswapping conversion"""

    def _post_init(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.pixels = self.image32

    def reverse(self, i):
        if False:
            for i in range(10):
                print('nop')
        orig = ffi.new('int *', i)
        chars = ffi.cast('unsigned char *', orig)
        (chars[0], chars[1], chars[2], chars[3]) = (chars[3], chars[2], chars[1], chars[0])
        return ffi.cast('int *', chars)[0]

    def get_pixel(self, x, y):
        if False:
            i = 10
            return i + 15
        return self.reverse(self.pixels[y][x])

    def set_pixel(self, x, y, color):
        if False:
            i = 10
            return i + 15
        self.pixels[y][x] = self.reverse(color)

class _PyAccessF(PyAccess):
    """32 bit float access"""

    def _post_init(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.pixels = ffi.cast('float **', self.image32)

    def get_pixel(self, x, y):
        if False:
            i = 10
            return i + 15
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        if False:
            while True:
                i = 10
        try:
            self.pixels[y][x] = color
        except TypeError:
            self.pixels[y][x] = color[0]
mode_map = {'1': _PyAccess8, 'L': _PyAccess8, 'P': _PyAccess8, 'I;16N': _PyAccessI16_N, 'LA': _PyAccess32_2, 'La': _PyAccess32_2, 'PA': _PyAccess32_2, 'RGB': _PyAccess32_3, 'LAB': _PyAccess32_3, 'HSV': _PyAccess32_3, 'YCbCr': _PyAccess32_3, 'RGBA': _PyAccess32_4, 'RGBa': _PyAccess32_4, 'RGBX': _PyAccess32_4, 'CMYK': _PyAccess32_4, 'F': _PyAccessF, 'I': _PyAccessI32_N}
if sys.byteorder == 'little':
    mode_map['I;16'] = _PyAccessI16_N
    mode_map['I;16L'] = _PyAccessI16_N
    mode_map['I;16B'] = _PyAccessI16_B
    mode_map['I;32L'] = _PyAccessI32_N
    mode_map['I;32B'] = _PyAccessI32_Swap
else:
    mode_map['I;16'] = _PyAccessI16_L
    mode_map['I;16L'] = _PyAccessI16_L
    mode_map['I;16B'] = _PyAccessI16_N
    mode_map['I;32L'] = _PyAccessI32_Swap
    mode_map['I;32B'] = _PyAccessI32_N

def new(img, readonly=False):
    if False:
        return 10
    access_type = mode_map.get(img.mode, None)
    if not access_type:
        logger.debug('PyAccess Not Implemented: %s', img.mode)
        return None
    return access_type(img, readonly)