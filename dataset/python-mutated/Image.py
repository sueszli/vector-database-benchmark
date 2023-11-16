import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
try:
    import defusedxml.ElementTree as ElementTree
except ImportError:
    ElementTree = None
from . import ExifTags, ImageMode, TiffTags, UnidentifiedImageError, __version__, _plugins
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
logger = logging.getLogger(__name__)

class DecompressionBombWarning(RuntimeWarning):
    pass

class DecompressionBombError(Exception):
    pass
MAX_IMAGE_PIXELS = int(1024 * 1024 * 1024 // 4 // 3)
try:
    from . import _imaging as core
    if __version__ != getattr(core, 'PILLOW_VERSION', None):
        msg = f"The _imaging extension was built for another version of Pillow or PIL:\nCore version: {getattr(core, 'PILLOW_VERSION', None)}\nPillow version: {__version__}"
        raise ImportError(msg)
except ImportError as v:
    core = DeferredError(ImportError('The _imaging C module is not installed.'))
    if str(v).startswith('Module use of python'):
        warnings.warn('The _imaging extension was built for another version of Python.', RuntimeWarning)
    elif str(v).startswith('The _imaging extension'):
        warnings.warn(str(v), RuntimeWarning)
    raise
USE_CFFI_ACCESS = False
try:
    import cffi
except ImportError:
    cffi = None

def isImageType(t):
    if False:
        i = 10
        return i + 15
    "\n    Checks if an object is an image object.\n\n    .. warning::\n\n       This function is for internal use only.\n\n    :param t: object to check if it's an image\n    :returns: True if the object is an image\n    "
    return hasattr(t, 'im')

class Transpose(IntEnum):
    FLIP_LEFT_RIGHT = 0
    FLIP_TOP_BOTTOM = 1
    ROTATE_90 = 2
    ROTATE_180 = 3
    ROTATE_270 = 4
    TRANSPOSE = 5
    TRANSVERSE = 6

class Transform(IntEnum):
    AFFINE = 0
    EXTENT = 1
    PERSPECTIVE = 2
    QUAD = 3
    MESH = 4

class Resampling(IntEnum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1
_filters_support = {Resampling.BOX: 0.5, Resampling.BILINEAR: 1.0, Resampling.HAMMING: 1.0, Resampling.BICUBIC: 2.0, Resampling.LANCZOS: 3.0}

class Dither(IntEnum):
    NONE = 0
    ORDERED = 1
    RASTERIZE = 2
    FLOYDSTEINBERG = 3

class Palette(IntEnum):
    WEB = 0
    ADAPTIVE = 1

class Quantize(IntEnum):
    MEDIANCUT = 0
    MAXCOVERAGE = 1
    FASTOCTREE = 2
    LIBIMAGEQUANT = 3
module = sys.modules[__name__]
for enum in (Transpose, Transform, Resampling, Dither, Palette, Quantize):
    for item in enum:
        setattr(module, item.name, item.value)
if hasattr(core, 'DEFAULT_STRATEGY'):
    DEFAULT_STRATEGY = core.DEFAULT_STRATEGY
    FILTERED = core.FILTERED
    HUFFMAN_ONLY = core.HUFFMAN_ONLY
    RLE = core.RLE
    FIXED = core.FIXED
ID = []
OPEN = {}
MIME = {}
SAVE = {}
SAVE_ALL = {}
EXTENSION = {}
DECODERS = {}
ENCODERS = {}
_ENDIAN = '<' if sys.byteorder == 'little' else '>'

def _conv_type_shape(im):
    if False:
        for i in range(10):
            print('nop')
    m = ImageMode.getmode(im.mode)
    shape = (im.height, im.width)
    extra = len(m.bands)
    if extra != 1:
        shape += (extra,)
    return (shape, m.typestr)
MODES = ['1', 'CMYK', 'F', 'HSV', 'I', 'L', 'LAB', 'P', 'RGB', 'RGBA', 'RGBX', 'YCbCr']
_MAPMODES = ('L', 'P', 'RGBX', 'RGBA', 'CMYK', 'I;16', 'I;16L', 'I;16B')

def getmodebase(mode):
    if False:
        i = 10
        return i + 15
    '\n    Gets the "base" mode for given mode.  This function returns "L" for\n    images that contain grayscale data, and "RGB" for images that\n    contain color data.\n\n    :param mode: Input mode.\n    :returns: "L" or "RGB".\n    :exception KeyError: If the input mode was not a standard mode.\n    '
    return ImageMode.getmode(mode).basemode

def getmodetype(mode):
    if False:
        i = 10
        return i + 15
    '\n    Gets the storage type mode.  Given a mode, this function returns a\n    single-layer mode suitable for storing individual bands.\n\n    :param mode: Input mode.\n    :returns: "L", "I", or "F".\n    :exception KeyError: If the input mode was not a standard mode.\n    '
    return ImageMode.getmode(mode).basetype

def getmodebandnames(mode):
    if False:
        return 10
    '\n    Gets a list of individual band names.  Given a mode, this function returns\n    a tuple containing the names of individual bands (use\n    :py:method:`~PIL.Image.getmodetype` to get the mode used to store each\n    individual band.\n\n    :param mode: Input mode.\n    :returns: A tuple containing band names.  The length of the tuple\n        gives the number of bands in an image of the given mode.\n    :exception KeyError: If the input mode was not a standard mode.\n    '
    return ImageMode.getmode(mode).bands

def getmodebands(mode):
    if False:
        return 10
    '\n    Gets the number of individual bands for this mode.\n\n    :param mode: Input mode.\n    :returns: The number of bands in this mode.\n    :exception KeyError: If the input mode was not a standard mode.\n    '
    return len(ImageMode.getmode(mode).bands)
_initialized = 0

def preinit():
    if False:
        print('Hello World!')
    '\n    Explicitly loads BMP, GIF, JPEG, PPM and PPM file format drivers.\n\n    It is called when opening or saving images.\n    '
    global _initialized
    if _initialized >= 1:
        return
    try:
        from . import BmpImagePlugin
        assert BmpImagePlugin
    except ImportError:
        pass
    try:
        from . import GifImagePlugin
        assert GifImagePlugin
    except ImportError:
        pass
    try:
        from . import JpegImagePlugin
        assert JpegImagePlugin
    except ImportError:
        pass
    try:
        from . import PpmImagePlugin
        assert PpmImagePlugin
    except ImportError:
        pass
    try:
        from . import PngImagePlugin
        assert PngImagePlugin
    except ImportError:
        pass
    _initialized = 1

def init():
    if False:
        return 10
    '\n    Explicitly initializes the Python Imaging Library. This function\n    loads all available file format drivers.\n\n    It is called when opening or saving images if :py:meth:`~preinit()` is\n    insufficient, and by :py:meth:`~PIL.features.pilinfo`.\n    '
    global _initialized
    if _initialized >= 2:
        return 0
    for plugin in _plugins:
        try:
            logger.debug('Importing %s', plugin)
            __import__(f'PIL.{plugin}', globals(), locals(), [])
        except ImportError as e:
            logger.debug('Image: failed to import %s: %s', plugin, e)
    if OPEN or SAVE:
        _initialized = 2
        return 1

def _getdecoder(mode, decoder_name, args, extra=()):
    if False:
        for i in range(10):
            print('nop')
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)
    try:
        decoder = DECODERS[decoder_name]
    except KeyError:
        pass
    else:
        return decoder(mode, *args + extra)
    try:
        decoder = getattr(core, decoder_name + '_decoder')
    except AttributeError as e:
        msg = f'decoder {decoder_name} not available'
        raise OSError(msg) from e
    return decoder(mode, *args + extra)

def _getencoder(mode, encoder_name, args, extra=()):
    if False:
        i = 10
        return i + 15
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)
    try:
        encoder = ENCODERS[encoder_name]
    except KeyError:
        pass
    else:
        return encoder(mode, *args + extra)
    try:
        encoder = getattr(core, encoder_name + '_encoder')
    except AttributeError as e:
        msg = f'encoder {encoder_name} not available'
        raise OSError(msg) from e
    return encoder(mode, *args + extra)

class _E:

    def __init__(self, scale, offset):
        if False:
            for i in range(10):
                print('nop')
        self.scale = scale
        self.offset = offset

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return _E(-self.scale, -self.offset)

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, _E):
            return _E(self.scale + other.scale, self.offset + other.offset)
        return _E(self.scale, self.offset + other)
    __radd__ = __add__

    def __sub__(self, other):
        if False:
            return 10
        return self + -other

    def __rsub__(self, other):
        if False:
            print('Hello World!')
        return other + -self

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, _E):
            return NotImplemented
        return _E(self.scale * other, self.offset * other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, _E):
            return NotImplemented
        return _E(self.scale / other, self.offset / other)

def _getscaleoffset(expr):
    if False:
        while True:
            i = 10
    a = expr(_E(1, 0))
    return (a.scale, a.offset) if isinstance(a, _E) else (0, a)

class Image:
    """
    This class represents an image object.  To create
    :py:class:`~PIL.Image.Image` objects, use the appropriate factory
    functions.  There's hardly ever any reason to call the Image constructor
    directly.

    * :py:func:`~PIL.Image.open`
    * :py:func:`~PIL.Image.new`
    * :py:func:`~PIL.Image.frombytes`
    """
    format = None
    format_description = None
    _close_exclusive_fp_after_loading = True

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.im = None
        self._mode = ''
        self._size = (0, 0)
        self.palette = None
        self.info = {}
        self.readonly = 0
        self.pyaccess = None
        self._exif = None

    @property
    def width(self):
        if False:
            while True:
                i = 10
        return self.size[0]

    @property
    def height(self):
        if False:
            print('Hello World!')
        return self.size[1]

    @property
    def size(self):
        if False:
            return 10
        return self._size

    @property
    def mode(self):
        if False:
            i = 10
            return i + 15
        return self._mode

    def _new(self, im):
        if False:
            for i in range(10):
                print('nop')
        new = Image()
        new.im = im
        new._mode = im.mode
        new._size = im.size
        if im.mode in ('P', 'PA'):
            if self.palette:
                new.palette = self.palette.copy()
            else:
                from . import ImagePalette
                new.palette = ImagePalette.ImagePalette()
        new.info = self.info.copy()
        return new

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        if hasattr(self, 'fp') and getattr(self, '_exclusive_fp', False):
            if getattr(self, '_fp', False):
                if self._fp != self.fp:
                    self._fp.close()
                self._fp = DeferredError(ValueError('Operation on closed image'))
            if self.fp:
                self.fp.close()
        self.fp = None

    def close(self):
        if False:
            while True:
                i = 10
        '\n        Closes the file pointer, if possible.\n\n        This operation will destroy the image core and release its memory.\n        The image data will be unusable afterward.\n\n        This function is required to close images that have multiple frames or\n        have not had their file read and closed by the\n        :py:meth:`~PIL.Image.Image.load` method. See :ref:`file-handling` for\n        more information.\n        '
        try:
            if getattr(self, '_fp', False):
                if self._fp != self.fp:
                    self._fp.close()
                self._fp = DeferredError(ValueError('Operation on closed image'))
            if self.fp:
                self.fp.close()
            self.fp = None
        except Exception as msg:
            logger.debug('Error closing: %s', msg)
        if getattr(self, 'map', None):
            self.map = None
        self.im = DeferredError(ValueError('Operation on closed image'))

    def _copy(self):
        if False:
            for i in range(10):
                print('nop')
        self.load()
        self.im = self.im.copy()
        self.pyaccess = None
        self.readonly = 0

    def _ensure_mutable(self):
        if False:
            for i in range(10):
                print('nop')
        if self.readonly:
            self._copy()
        else:
            self.load()

    def _dump(self, file=None, format=None, **options):
        if False:
            i = 10
            return i + 15
        suffix = ''
        if format:
            suffix = '.' + format
        if not file:
            (f, filename) = tempfile.mkstemp(suffix)
            os.close(f)
        else:
            filename = file
            if not filename.endswith(suffix):
                filename = filename + suffix
        self.load()
        if not format or format == 'PPM':
            self.im.save_ppm(filename)
        else:
            self.save(filename, format, **options)
        return filename

    def __eq__(self, other):
        if False:
            return 10
        return self.__class__ is other.__class__ and self.mode == other.mode and (self.size == other.size) and (self.info == other.info) and (self.getpalette() == other.getpalette()) and (self.tobytes() == other.tobytes())

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s.%s image mode=%s size=%dx%d at 0x%X>' % (self.__class__.__module__, self.__class__.__name__, self.mode, self.size[0], self.size[1], id(self))

    def _repr_pretty_(self, p, cycle):
        if False:
            while True:
                i = 10
        'IPython plain text display support'
        p.text('<%s.%s image mode=%s size=%dx%d>' % (self.__class__.__module__, self.__class__.__name__, self.mode, self.size[0], self.size[1]))

    def _repr_image(self, image_format, **kwargs):
        if False:
            print('Hello World!')
        'Helper function for iPython display hook.\n\n        :param image_format: Image format.\n        :returns: image as bytes, saved into the given format.\n        '
        b = io.BytesIO()
        try:
            self.save(b, image_format, **kwargs)
        except Exception:
            return None
        return b.getvalue()

    def _repr_png_(self):
        if False:
            while True:
                i = 10
        'iPython display hook support for PNG format.\n\n        :returns: PNG version of the image as bytes\n        '
        return self._repr_image('PNG', compress_level=1)

    def _repr_jpeg_(self):
        if False:
            return 10
        'iPython display hook support for JPEG format.\n\n        :returns: JPEG version of the image as bytes\n        '
        return self._repr_image('JPEG')

    @property
    def __array_interface__(self):
        if False:
            print('Hello World!')
        new = {'version': 3}
        try:
            if self.mode == '1':
                new['data'] = self.tobytes('raw', 'L')
            else:
                new['data'] = self.tobytes()
        except Exception as e:
            if not isinstance(e, (MemoryError, RecursionError)):
                try:
                    import numpy
                    from packaging.version import parse as parse_version
                except ImportError:
                    pass
                else:
                    if parse_version(numpy.__version__) < parse_version('1.23'):
                        warnings.warn(e)
            raise
        (new['shape'], new['typestr']) = _conv_type_shape(self)
        return new

    def __getstate__(self):
        if False:
            while True:
                i = 10
        im_data = self.tobytes()
        return [self.info, self.mode, self.size, self.getpalette(), im_data]

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        Image.__init__(self)
        (info, mode, size, palette, data) = state
        self.info = info
        self._mode = mode
        self._size = size
        self.im = core.new(mode, size)
        if mode in ('L', 'LA', 'P', 'PA') and palette:
            self.putpalette(palette)
        self.frombytes(data)

    def tobytes(self, encoder_name='raw', *args):
        if False:
            print('Hello World!')
        '\n        Return image as a bytes object.\n\n        .. warning::\n\n            This method returns the raw image data from the internal\n            storage.  For compressed image data (e.g. PNG, JPEG) use\n            :meth:`~.save`, with a BytesIO parameter for in-memory\n            data.\n\n        :param encoder_name: What encoder to use.  The default is to\n                             use the standard "raw" encoder.\n\n                             A list of C encoders can be seen under\n                             codecs section of the function array in\n                             :file:`_imaging.c`. Python encoders are\n                             registered within the relevant plugins.\n        :param args: Extra arguments to the encoder.\n        :returns: A :py:class:`bytes` object.\n        '
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        if encoder_name == 'raw' and args == ():
            args = self.mode
        self.load()
        if self.width == 0 or self.height == 0:
            return b''
        e = _getencoder(self.mode, encoder_name, args)
        e.setimage(self.im)
        bufsize = max(65536, self.size[0] * 4)
        output = []
        while True:
            (bytes_consumed, errcode, data) = e.encode(bufsize)
            output.append(data)
            if errcode:
                break
        if errcode < 0:
            msg = f'encoder error {errcode} in tobytes'
            raise RuntimeError(msg)
        return b''.join(output)

    def tobitmap(self, name='image'):
        if False:
            i = 10
            return i + 15
        '\n        Returns the image converted to an X11 bitmap.\n\n        .. note:: This method only works for mode "1" images.\n\n        :param name: The name prefix to use for the bitmap variables.\n        :returns: A string containing an X11 bitmap.\n        :raises ValueError: If the mode is not "1"\n        '
        self.load()
        if self.mode != '1':
            msg = 'not a bitmap'
            raise ValueError(msg)
        data = self.tobytes('xbm')
        return b''.join([f'#define {name}_width {self.size[0]}\n'.encode('ascii'), f'#define {name}_height {self.size[1]}\n'.encode('ascii'), f'static char {name}_bits[] = {{\n'.encode('ascii'), data, b'};'])

    def frombytes(self, data, decoder_name='raw', *args):
        if False:
            i = 10
            return i + 15
        '\n        Loads this image with pixel data from a bytes object.\n\n        This method is similar to the :py:func:`~PIL.Image.frombytes` function,\n        but loads data into this image instead of creating a new image object.\n        '
        if self.width == 0 or self.height == 0:
            return
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        if decoder_name == 'raw' and args == ():
            args = self.mode
        d = _getdecoder(self.mode, decoder_name, args)
        d.setimage(self.im)
        s = d.decode(data)
        if s[0] >= 0:
            msg = 'not enough image data'
            raise ValueError(msg)
        if s[1] != 0:
            msg = 'cannot decode image data'
            raise ValueError(msg)

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Allocates storage for the image and loads the pixel data.  In\n        normal cases, you don't need to call this method, since the\n        Image class automatically loads an opened image when it is\n        accessed for the first time.\n\n        If the file associated with the image was opened by Pillow, then this\n        method will close it. The exception to this is if the image has\n        multiple frames, in which case the file will be left open for seek\n        operations. See :ref:`file-handling` for more information.\n\n        :returns: An image access object.\n        :rtype: :ref:`PixelAccess` or :py:class:`PIL.PyAccess`\n        "
        if self.im is not None and self.palette and self.palette.dirty:
            (mode, arr) = self.palette.getdata()
            self.im.putpalette(mode, arr)
            self.palette.dirty = 0
            self.palette.rawmode = None
            if 'transparency' in self.info and mode in ('LA', 'PA'):
                if isinstance(self.info['transparency'], int):
                    self.im.putpalettealpha(self.info['transparency'], 0)
                else:
                    self.im.putpalettealphas(self.info['transparency'])
                self.palette.mode = 'RGBA'
            else:
                palette_mode = 'RGBA' if mode.startswith('RGBA') else 'RGB'
                self.palette.mode = palette_mode
                self.palette.palette = self.im.getpalette(palette_mode, palette_mode)
        if self.im is not None:
            if cffi and USE_CFFI_ACCESS:
                if self.pyaccess:
                    return self.pyaccess
                from . import PyAccess
                self.pyaccess = PyAccess.new(self, self.readonly)
                if self.pyaccess:
                    return self.pyaccess
            return self.im.pixel_access(self.readonly)

    def verify(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verifies the contents of a file. For data read from a file, this\n        method attempts to determine if the file is broken, without\n        actually decoding the image data.  If this method finds any\n        problems, it raises suitable exceptions.  If you need to load\n        the image after using this method, you must reopen the image\n        file.\n        '
        pass

    def convert(self, mode=None, matrix=None, dither=None, palette=Palette.WEB, colors=256):
        if False:
            i = 10
            return i + 15
        '\n        Returns a converted copy of this image. For the "P" mode, this\n        method translates pixels through the palette.  If mode is\n        omitted, a mode is chosen so that all information in the image\n        and the palette can be represented without a palette.\n\n        The current version supports all possible conversions between\n        "L", "RGB" and "CMYK". The ``matrix`` argument only supports "L"\n        and "RGB".\n\n        When translating a color image to grayscale (mode "L"),\n        the library uses the ITU-R 601-2 luma transform::\n\n            L = R * 299/1000 + G * 587/1000 + B * 114/1000\n\n        The default method of converting a grayscale ("L") or "RGB"\n        image into a bilevel (mode "1") image uses Floyd-Steinberg\n        dither to approximate the original image luminosity levels. If\n        dither is ``None``, all values larger than 127 are set to 255 (white),\n        all other values to 0 (black). To use other thresholds, use the\n        :py:meth:`~PIL.Image.Image.point` method.\n\n        When converting from "RGBA" to "P" without a ``matrix`` argument,\n        this passes the operation to :py:meth:`~PIL.Image.Image.quantize`,\n        and ``dither`` and ``palette`` are ignored.\n\n        When converting from "PA", if an "RGBA" palette is present, the alpha\n        channel from the image will be used instead of the values from the palette.\n\n        :param mode: The requested mode. See: :ref:`concept-modes`.\n        :param matrix: An optional conversion matrix.  If given, this\n           should be 4- or 12-tuple containing floating point values.\n        :param dither: Dithering method, used when converting from\n           mode "RGB" to "P" or from "RGB" or "L" to "1".\n           Available methods are :data:`Dither.NONE` or :data:`Dither.FLOYDSTEINBERG`\n           (default). Note that this is not used when ``matrix`` is supplied.\n        :param palette: Palette to use when converting from mode "RGB"\n           to "P".  Available palettes are :data:`Palette.WEB` or\n           :data:`Palette.ADAPTIVE`.\n        :param colors: Number of colors to use for the :data:`Palette.ADAPTIVE`\n           palette. Defaults to 256.\n        :rtype: :py:class:`~PIL.Image.Image`\n        :returns: An :py:class:`~PIL.Image.Image` object.\n        '
        self.load()
        has_transparency = 'transparency' in self.info
        if not mode and self.mode == 'P':
            if self.palette:
                mode = self.palette.mode
            else:
                mode = 'RGB'
            if mode == 'RGB' and has_transparency:
                mode = 'RGBA'
        if not mode or (mode == self.mode and (not matrix)):
            return self.copy()
        if matrix:
            if mode not in ('L', 'RGB'):
                msg = 'illegal conversion'
                raise ValueError(msg)
            im = self.im.convert_matrix(mode, matrix)
            new_im = self._new(im)
            if has_transparency and self.im.bands == 3:
                transparency = new_im.info['transparency']

                def convert_transparency(m, v):
                    if False:
                        print('Hello World!')
                    v = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * 0.5
                    return max(0, min(255, int(v)))
                if mode == 'L':
                    transparency = convert_transparency(matrix, transparency)
                elif len(mode) == 3:
                    transparency = tuple((convert_transparency(matrix[i * 4:i * 4 + 4], transparency) for i in range(0, len(transparency))))
                new_im.info['transparency'] = transparency
            return new_im
        if mode == 'P' and self.mode == 'RGBA':
            return self.quantize(colors)
        trns = None
        delete_trns = False
        if has_transparency:
            if self.mode in ('1', 'L', 'I') and mode in ('LA', 'RGBA') or (self.mode == 'RGB' and mode == 'RGBA'):
                new_im = self._new(self.im.convert_transparent(mode, self.info['transparency']))
                del new_im.info['transparency']
                return new_im
            elif self.mode in ('L', 'RGB', 'P') and mode in ('L', 'RGB', 'P'):
                t = self.info['transparency']
                if isinstance(t, bytes):
                    warnings.warn('Palette images with Transparency expressed in bytes should be converted to RGBA images')
                    delete_trns = True
                else:
                    trns_im = new(self.mode, (1, 1))
                    if self.mode == 'P':
                        trns_im.putpalette(self.palette)
                        if isinstance(t, tuple):
                            err = "Couldn't allocate a palette color for transparency"
                            try:
                                t = trns_im.palette.getcolor(t, self)
                            except ValueError as e:
                                if str(e) == 'cannot allocate more than 256 colors':
                                    t = None
                                else:
                                    raise ValueError(err) from e
                    if t is None:
                        trns = None
                    else:
                        trns_im.putpixel((0, 0), t)
                        if mode in ('L', 'RGB'):
                            trns_im = trns_im.convert(mode)
                        else:
                            trns_im = trns_im.convert('RGB')
                        trns = trns_im.getpixel((0, 0))
            elif self.mode == 'P' and mode in ('LA', 'PA', 'RGBA'):
                t = self.info['transparency']
                delete_trns = True
                if isinstance(t, bytes):
                    self.im.putpalettealphas(t)
                elif isinstance(t, int):
                    self.im.putpalettealpha(t, 0)
                else:
                    msg = 'Transparency for P mode should be bytes or int'
                    raise ValueError(msg)
        if mode == 'P' and palette == Palette.ADAPTIVE:
            im = self.im.quantize(colors)
            new_im = self._new(im)
            from . import ImagePalette
            new_im.palette = ImagePalette.ImagePalette('RGB', new_im.im.getpalette('RGB'))
            if delete_trns:
                del new_im.info['transparency']
            if trns is not None:
                try:
                    new_im.info['transparency'] = new_im.palette.getcolor(trns, new_im)
                except Exception:
                    del new_im.info['transparency']
                    warnings.warn("Couldn't allocate palette entry for transparency")
            return new_im
        if 'LAB' in (self.mode, mode):
            other_mode = mode if self.mode == 'LAB' else self.mode
            if other_mode in ('RGB', 'RGBA', 'RGBX'):
                from . import ImageCms
                srgb = ImageCms.createProfile('sRGB')
                lab = ImageCms.createProfile('LAB')
                profiles = [lab, srgb] if self.mode == 'LAB' else [srgb, lab]
                transform = ImageCms.buildTransform(profiles[0], profiles[1], self.mode, mode)
                return transform.apply(self)
        if dither is None:
            dither = Dither.FLOYDSTEINBERG
        try:
            im = self.im.convert(mode, dither)
        except ValueError:
            try:
                modebase = getmodebase(self.mode)
                if modebase == self.mode:
                    raise
                im = self.im.convert(modebase)
                im = im.convert(mode, dither)
            except KeyError as e:
                msg = 'illegal conversion'
                raise ValueError(msg) from e
        new_im = self._new(im)
        if mode == 'P' and palette != Palette.ADAPTIVE:
            from . import ImagePalette
            new_im.palette = ImagePalette.ImagePalette('RGB', im.getpalette('RGB'))
        if delete_trns:
            del new_im.info['transparency']
        if trns is not None:
            if new_im.mode == 'P':
                try:
                    new_im.info['transparency'] = new_im.palette.getcolor(trns, new_im)
                except ValueError as e:
                    del new_im.info['transparency']
                    if str(e) != 'cannot allocate more than 256 colors':
                        warnings.warn("Couldn't allocate palette entry for transparency")
            else:
                new_im.info['transparency'] = trns
        return new_im

    def quantize(self, colors=256, method=None, kmeans=0, palette=None, dither=Dither.FLOYDSTEINBERG):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert the image to \'P\' mode with the specified number\n        of colors.\n\n        :param colors: The desired number of colors, <= 256\n        :param method: :data:`Quantize.MEDIANCUT` (median cut),\n                       :data:`Quantize.MAXCOVERAGE` (maximum coverage),\n                       :data:`Quantize.FASTOCTREE` (fast octree),\n                       :data:`Quantize.LIBIMAGEQUANT` (libimagequant; check support\n                       using :py:func:`PIL.features.check_feature` with\n                       ``feature="libimagequant"``).\n\n                       By default, :data:`Quantize.MEDIANCUT` will be used.\n\n                       The exception to this is RGBA images. :data:`Quantize.MEDIANCUT`\n                       and :data:`Quantize.MAXCOVERAGE` do not support RGBA images, so\n                       :data:`Quantize.FASTOCTREE` is used by default instead.\n        :param kmeans: Integer\n        :param palette: Quantize to the palette of given\n                        :py:class:`PIL.Image.Image`.\n        :param dither: Dithering method, used when converting from\n           mode "RGB" to "P" or from "RGB" or "L" to "1".\n           Available methods are :data:`Dither.NONE` or :data:`Dither.FLOYDSTEINBERG`\n           (default).\n        :returns: A new image\n        '
        self.load()
        if method is None:
            method = Quantize.MEDIANCUT
            if self.mode == 'RGBA':
                method = Quantize.FASTOCTREE
        if self.mode == 'RGBA' and method not in (Quantize.FASTOCTREE, Quantize.LIBIMAGEQUANT):
            msg = 'Fast Octree (method == 2) and libimagequant (method == 3) are the only valid methods for quantizing RGBA images'
            raise ValueError(msg)
        if palette:
            palette.load()
            if palette.mode != 'P':
                msg = 'bad mode for palette image'
                raise ValueError(msg)
            if self.mode != 'RGB' and self.mode != 'L':
                msg = 'only RGB or L mode images can be quantized to a palette'
                raise ValueError(msg)
            im = self.im.convert('P', dither, palette.im)
            new_im = self._new(im)
            new_im.palette = palette.palette.copy()
            return new_im
        im = self._new(self.im.quantize(colors, method, kmeans))
        from . import ImagePalette
        mode = im.im.getpalettemode()
        palette = im.im.getpalette(mode, mode)[:colors * len(mode)]
        im.palette = ImagePalette.ImagePalette(mode, palette)
        return im

    def copy(self):
        if False:
            print('Hello World!')
        '\n        Copies this image. Use this method if you wish to paste things\n        into an image, but still retain the original.\n\n        :rtype: :py:class:`~PIL.Image.Image`\n        :returns: An :py:class:`~PIL.Image.Image` object.\n        '
        self.load()
        return self._new(self.im.copy())
    __copy__ = copy

    def crop(self, box=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a rectangular region from this image. The box is a\n        4-tuple defining the left, upper, right, and lower pixel\n        coordinate. See :ref:`coordinate-system`.\n\n        Note: Prior to Pillow 3.4.0, this was a lazy operation.\n\n        :param box: The crop rectangle, as a (left, upper, right, lower)-tuple.\n        :rtype: :py:class:`~PIL.Image.Image`\n        :returns: An :py:class:`~PIL.Image.Image` object.\n        '
        if box is None:
            return self.copy()
        if box[2] < box[0]:
            msg = "Coordinate 'right' is less than 'left'"
            raise ValueError(msg)
        elif box[3] < box[1]:
            msg = "Coordinate 'lower' is less than 'upper'"
            raise ValueError(msg)
        self.load()
        return self._new(self._crop(self.im, box))

    def _crop(self, im, box):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a rectangular region from the core image object im.\n\n        This is equivalent to calling im.crop((x0, y0, x1, y1)), but\n        includes additional sanity checks.\n\n        :param im: a core image object\n        :param box: The crop rectangle, as a (left, upper, right, lower)-tuple.\n        :returns: A core image object.\n        '
        (x0, y0, x1, y1) = map(int, map(round, box))
        absolute_values = (abs(x1 - x0), abs(y1 - y0))
        _decompression_bomb_check(absolute_values)
        return im.crop((x0, y0, x1, y1))

    def draft(self, mode, size):
        if False:
            print('Hello World!')
        '\n        Configures the image file loader so it returns a version of the\n        image that as closely as possible matches the given mode and\n        size. For example, you can use this method to convert a color\n        JPEG to grayscale while loading it.\n\n        If any changes are made, returns a tuple with the chosen ``mode`` and\n        ``box`` with coordinates of the original image within the altered one.\n\n        Note that this method modifies the :py:class:`~PIL.Image.Image` object\n        in place. If the image has already been loaded, this method has no\n        effect.\n\n        Note: This method is not implemented for most images. It is\n        currently implemented only for JPEG and MPO images.\n\n        :param mode: The requested mode.\n        :param size: The requested size in pixels, as a 2-tuple:\n           (width, height).\n        '
        pass

    def _expand(self, xmargin, ymargin=None):
        if False:
            print('Hello World!')
        if ymargin is None:
            ymargin = xmargin
        self.load()
        return self._new(self.im.expand(xmargin, ymargin))

    def filter(self, filter):
        if False:
            while True:
                i = 10
        '\n        Filters this image using the given filter.  For a list of\n        available filters, see the :py:mod:`~PIL.ImageFilter` module.\n\n        :param filter: Filter kernel.\n        :returns: An :py:class:`~PIL.Image.Image` object.'
        from . import ImageFilter
        self.load()
        if isinstance(filter, Callable):
            filter = filter()
        if not hasattr(filter, 'filter'):
            msg = 'filter argument should be ImageFilter.Filter instance or class'
            raise TypeError(msg)
        multiband = isinstance(filter, ImageFilter.MultibandFilter)
        if self.im.bands == 1 or multiband:
            return self._new(filter.filter(self.im))
        ims = []
        for c in range(self.im.bands):
            ims.append(self._new(filter.filter(self.im.getband(c))))
        return merge(self.mode, ims)

    def getbands(self):
        if False:
            return 10
        '\n        Returns a tuple containing the name of each band in this image.\n        For example, ``getbands`` on an RGB image returns ("R", "G", "B").\n\n        :returns: A tuple containing band names.\n        :rtype: tuple\n        '
        return ImageMode.getmode(self.mode).bands

    def getbbox(self, *, alpha_only=True):
        if False:
            print('Hello World!')
        '\n        Calculates the bounding box of the non-zero regions in the\n        image.\n\n        :param alpha_only: Optional flag, defaulting to ``True``.\n           If ``True`` and the image has an alpha channel, trim transparent pixels.\n           Otherwise, trim pixels when all channels are zero.\n           Keyword-only argument.\n        :returns: The bounding box is returned as a 4-tuple defining the\n           left, upper, right, and lower pixel coordinate. See\n           :ref:`coordinate-system`. If the image is completely empty, this\n           method returns None.\n\n        '
        self.load()
        return self.im.getbbox(alpha_only)

    def getcolors(self, maxcolors=256):
        if False:
            while True:
                i = 10
        "\n        Returns a list of colors used in this image.\n\n        The colors will be in the image's mode. For example, an RGB image will\n        return a tuple of (red, green, blue) color values, and a P image will\n        return the index of the color in the palette.\n\n        :param maxcolors: Maximum number of colors.  If this number is\n           exceeded, this method returns None.  The default limit is\n           256 colors.\n        :returns: An unsorted list of (count, pixel) values.\n        "
        self.load()
        if self.mode in ('1', 'L', 'P'):
            h = self.im.histogram()
            out = []
            for i in range(256):
                if h[i]:
                    out.append((h[i], i))
            if len(out) > maxcolors:
                return None
            return out
        return self.im.getcolors(maxcolors)

    def getdata(self, band=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the contents of this image as a sequence object\n        containing pixel values.  The sequence object is flattened, so\n        that values for line one follow directly after the values of\n        line zero, and so on.\n\n        Note that the sequence object returned by this method is an\n        internal PIL data type, which only supports certain sequence\n        operations.  To convert it to an ordinary sequence (e.g. for\n        printing), use ``list(im.getdata())``.\n\n        :param band: What band to return.  The default is to return\n           all bands.  To return a single band, pass in the index\n           value (e.g. 0 to get the "R" band from an "RGB" image).\n        :returns: A sequence-like object.\n        '
        self.load()
        if band is not None:
            return self.im.getband(band)
        return self.im

    def getextrema(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the minimum and maximum pixel values for each band in\n        the image.\n\n        :returns: For a single-band image, a 2-tuple containing the\n           minimum and maximum pixel value.  For a multi-band image,\n           a tuple containing one 2-tuple for each band.\n        '
        self.load()
        if self.im.bands > 1:
            extrema = []
            for i in range(self.im.bands):
                extrema.append(self.im.getband(i).getextrema())
            return tuple(extrema)
        return self.im.getextrema()

    def _getxmp(self, xmp_tags):
        if False:
            for i in range(10):
                print('nop')

        def get_name(tag):
            if False:
                for i in range(10):
                    print('nop')
            return re.sub('^{[^}]+}', '', tag)

        def get_value(element):
            if False:
                print('Hello World!')
            value = {get_name(k): v for (k, v) in element.attrib.items()}
            children = list(element)
            if children:
                for child in children:
                    name = get_name(child.tag)
                    child_value = get_value(child)
                    if name in value:
                        if not isinstance(value[name], list):
                            value[name] = [value[name]]
                        value[name].append(child_value)
                    else:
                        value[name] = child_value
            elif value:
                if element.text:
                    value['text'] = element.text
            else:
                return element.text
            return value
        if ElementTree is None:
            warnings.warn('XMP data cannot be read without defusedxml dependency')
            return {}
        else:
            root = ElementTree.fromstring(xmp_tags)
            return {get_name(root.tag): get_value(root)}

    def getexif(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets EXIF data from the image.\n\n        :returns: an :py:class:`~PIL.Image.Exif` object.\n        '
        if self._exif is None:
            self._exif = Exif()
            self._exif._loaded = False
        elif self._exif._loaded:
            return self._exif
        self._exif._loaded = True
        exif_info = self.info.get('exif')
        if exif_info is None:
            if 'Raw profile type exif' in self.info:
                exif_info = bytes.fromhex(''.join(self.info['Raw profile type exif'].split('\n')[3:]))
            elif hasattr(self, 'tag_v2'):
                self._exif.bigtiff = self.tag_v2._bigtiff
                self._exif.endian = self.tag_v2._endian
                self._exif.load_from_fp(self.fp, self.tag_v2._offset)
        if exif_info is not None:
            self._exif.load(exif_info)
        if ExifTags.Base.Orientation not in self._exif:
            xmp_tags = self.info.get('XML:com.adobe.xmp')
            if xmp_tags:
                match = re.search('tiff:Orientation(="|>)([0-9])', xmp_tags)
                if match:
                    self._exif[ExifTags.Base.Orientation] = int(match[2])
        return self._exif

    def _reload_exif(self):
        if False:
            return 10
        if self._exif is None or not self._exif._loaded:
            return
        self._exif._loaded = False
        self.getexif()

    def get_child_images(self):
        if False:
            return 10
        child_images = []
        exif = self.getexif()
        ifds = []
        if ExifTags.Base.SubIFDs in exif:
            subifd_offsets = exif[ExifTags.Base.SubIFDs]
            if subifd_offsets:
                if not isinstance(subifd_offsets, tuple):
                    subifd_offsets = (subifd_offsets,)
                for subifd_offset in subifd_offsets:
                    ifds.append((exif._get_ifd_dict(subifd_offset), subifd_offset))
        ifd1 = exif.get_ifd(ExifTags.IFD.IFD1)
        if ifd1 and ifd1.get(513):
            ifds.append((ifd1, exif._info.next))
        offset = None
        for (ifd, ifd_offset) in ifds:
            current_offset = self.fp.tell()
            if offset is None:
                offset = current_offset
            fp = self.fp
            thumbnail_offset = ifd.get(513)
            if thumbnail_offset is not None:
                try:
                    thumbnail_offset += self._exif_offset
                except AttributeError:
                    pass
                self.fp.seek(thumbnail_offset)
                data = self.fp.read(ifd.get(514))
                fp = io.BytesIO(data)
            with open(fp) as im:
                if thumbnail_offset is None:
                    im._frame_pos = [ifd_offset]
                    im._seek(0)
                im.load()
                child_images.append(im)
        if offset is not None:
            self.fp.seek(offset)
        return child_images

    def getim(self):
        if False:
            print('Hello World!')
        '\n        Returns a capsule that points to the internal image memory.\n\n        :returns: A capsule object.\n        '
        self.load()
        return self.im.ptr

    def getpalette(self, rawmode='RGB'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the image palette as a list.\n\n        :param rawmode: The mode in which to return the palette. ``None`` will\n           return the palette in its current mode.\n\n           .. versionadded:: 9.1.0\n\n        :returns: A list of color values [r, g, b, ...], or None if the\n           image has no palette.\n        '
        self.load()
        try:
            mode = self.im.getpalettemode()
        except ValueError:
            return None
        if rawmode is None:
            rawmode = mode
        return list(self.im.getpalette(mode, rawmode))

    @property
    def has_transparency_data(self) -> bool:
        if False:
            return 10
        '\n        Determine if an image has transparency data, whether in the form of an\n        alpha channel, a palette with an alpha channel, or a "transparency" key\n        in the info dictionary.\n\n        Note the image might still appear solid, if all of the values shown\n        within are opaque.\n\n        :returns: A boolean.\n        '
        return self.mode in ('LA', 'La', 'PA', 'RGBA', 'RGBa') or (self.mode == 'P' and self.palette.mode.endswith('A')) or 'transparency' in self.info

    def apply_transparency(self):
        if False:
            i = 10
            return i + 15
        '\n        If a P mode image has a "transparency" key in the info dictionary,\n        remove the key and instead apply the transparency to the palette.\n        Otherwise, the image is unchanged.\n        '
        if self.mode != 'P' or 'transparency' not in self.info:
            return
        from . import ImagePalette
        palette = self.getpalette('RGBA')
        transparency = self.info['transparency']
        if isinstance(transparency, bytes):
            for (i, alpha) in enumerate(transparency):
                palette[i * 4 + 3] = alpha
        else:
            palette[transparency * 4 + 3] = 0
        self.palette = ImagePalette.ImagePalette('RGBA', bytes(palette))
        self.palette.dirty = 1
        del self.info['transparency']

    def getpixel(self, xy):
        if False:
            while True:
                i = 10
        '\n        Returns the pixel value at a given position.\n\n        :param xy: The coordinate, given as (x, y). See\n           :ref:`coordinate-system`.\n        :returns: The pixel value.  If the image is a multi-layer image,\n           this method returns a tuple.\n        '
        self.load()
        if self.pyaccess:
            return self.pyaccess.getpixel(xy)
        return self.im.getpixel(tuple(xy))

    def getprojection(self):
        if False:
            return 10
        '\n        Get projection to x and y axes\n\n        :returns: Two sequences, indicating where there are non-zero\n            pixels along the X-axis and the Y-axis, respectively.\n        '
        self.load()
        (x, y) = self.im.getprojection()
        return (list(x), list(y))

    def histogram(self, mask=None, extrema=None):
        if False:
            while True:
                i = 10
        '\n        Returns a histogram for the image. The histogram is returned as a\n        list of pixel counts, one for each pixel value in the source\n        image. Counts are grouped into 256 bins for each band, even if\n        the image has more than 8 bits per band. If the image has more\n        than one band, the histograms for all bands are concatenated (for\n        example, the histogram for an "RGB" image contains 768 values).\n\n        A bilevel image (mode "1") is treated as a grayscale ("L") image\n        by this method.\n\n        If a mask is provided, the method returns a histogram for those\n        parts of the image where the mask image is non-zero. The mask\n        image must have the same size as the image, and be either a\n        bi-level image (mode "1") or a grayscale image ("L").\n\n        :param mask: An optional mask.\n        :param extrema: An optional tuple of manually-specified extrema.\n        :returns: A list containing pixel counts.\n        '
        self.load()
        if mask:
            mask.load()
            return self.im.histogram((0, 0), mask.im)
        if self.mode in ('I', 'F'):
            if extrema is None:
                extrema = self.getextrema()
            return self.im.histogram(extrema)
        return self.im.histogram()

    def entropy(self, mask=None, extrema=None):
        if False:
            print('Hello World!')
        '\n        Calculates and returns the entropy for the image.\n\n        A bilevel image (mode "1") is treated as a grayscale ("L")\n        image by this method.\n\n        If a mask is provided, the method employs the histogram for\n        those parts of the image where the mask image is non-zero.\n        The mask image must have the same size as the image, and be\n        either a bi-level image (mode "1") or a grayscale image ("L").\n\n        :param mask: An optional mask.\n        :param extrema: An optional tuple of manually-specified extrema.\n        :returns: A float value representing the image entropy\n        '
        self.load()
        if mask:
            mask.load()
            return self.im.entropy((0, 0), mask.im)
        if self.mode in ('I', 'F'):
            if extrema is None:
                extrema = self.getextrema()
            return self.im.entropy(extrema)
        return self.im.entropy()

    def paste(self, im, box=None, mask=None):
        if False:
            i = 10
            return i + 15
        '\n        Pastes another image into this image. The box argument is either\n        a 2-tuple giving the upper left corner, a 4-tuple defining the\n        left, upper, right, and lower pixel coordinate, or None (same as\n        (0, 0)). See :ref:`coordinate-system`. If a 4-tuple is given, the size\n        of the pasted image must match the size of the region.\n\n        If the modes don\'t match, the pasted image is converted to the mode of\n        this image (see the :py:meth:`~PIL.Image.Image.convert` method for\n        details).\n\n        Instead of an image, the source can be a integer or tuple\n        containing pixel values.  The method then fills the region\n        with the given color.  When creating RGB images, you can\n        also use color strings as supported by the ImageColor module.\n\n        If a mask is given, this method updates only the regions\n        indicated by the mask. You can use either "1", "L", "LA", "RGBA"\n        or "RGBa" images (if present, the alpha band is used as mask).\n        Where the mask is 255, the given image is copied as is.  Where\n        the mask is 0, the current value is preserved.  Intermediate\n        values will mix the two images together, including their alpha\n        channels if they have them.\n\n        See :py:meth:`~PIL.Image.Image.alpha_composite` if you want to\n        combine images with respect to their alpha channels.\n\n        :param im: Source image or pixel value (integer or tuple).\n        :param box: An optional 4-tuple giving the region to paste into.\n           If a 2-tuple is used instead, it\'s treated as the upper left\n           corner.  If omitted or None, the source is pasted into the\n           upper left corner.\n\n           If an image is given as the second argument and there is no\n           third, the box defaults to (0, 0), and the second argument\n           is interpreted as a mask image.\n        :param mask: An optional mask image.\n        '
        if isImageType(box) and mask is None:
            mask = box
            box = None
        if box is None:
            box = (0, 0)
        if len(box) == 2:
            if isImageType(im):
                size = im.size
            elif isImageType(mask):
                size = mask.size
            else:
                msg = 'cannot determine region size; use 4-item box'
                raise ValueError(msg)
            box += (box[0] + size[0], box[1] + size[1])
        if isinstance(im, str):
            from . import ImageColor
            im = ImageColor.getcolor(im, self.mode)
        elif isImageType(im):
            im.load()
            if self.mode != im.mode:
                if self.mode != 'RGB' or im.mode not in ('LA', 'RGBA', 'RGBa'):
                    im = im.convert(self.mode)
            im = im.im
        self._ensure_mutable()
        if mask:
            mask.load()
            self.im.paste(im, box, mask.im)
        else:
            self.im.paste(im, box)

    def alpha_composite(self, im, dest=(0, 0), source=(0, 0)):
        if False:
            for i in range(10):
                print('nop')
        "'In-place' analog of Image.alpha_composite. Composites an image\n        onto this image.\n\n        :param im: image to composite over this one\n        :param dest: Optional 2 tuple (left, top) specifying the upper\n          left corner in this (destination) image.\n        :param source: Optional 2 (left, top) tuple for the upper left\n          corner in the overlay source image, or 4 tuple (left, top, right,\n          bottom) for the bounds of the source rectangle\n\n        Performance Note: Not currently implemented in-place in the core layer.\n        "
        if not isinstance(source, (list, tuple)):
            msg = 'Source must be a tuple'
            raise ValueError(msg)
        if not isinstance(dest, (list, tuple)):
            msg = 'Destination must be a tuple'
            raise ValueError(msg)
        if len(source) not in (2, 4):
            msg = 'Source must be a 2 or 4-tuple'
            raise ValueError(msg)
        if not len(dest) == 2:
            msg = 'Destination must be a 2-tuple'
            raise ValueError(msg)
        if min(source) < 0:
            msg = 'Source must be non-negative'
            raise ValueError(msg)
        if len(source) == 2:
            source = source + im.size
        if source == (0, 0) + im.size:
            overlay = im
        else:
            overlay = im.crop(source)
        box = dest + (dest[0] + overlay.width, dest[1] + overlay.height)
        if box == (0, 0) + self.size:
            background = self
        else:
            background = self.crop(box)
        result = alpha_composite(background, overlay)
        self.paste(result, box)

    def point(self, lut, mode=None):
        if False:
            return 10
        '\n        Maps this image through a lookup table or function.\n\n        :param lut: A lookup table, containing 256 (or 65536 if\n           self.mode=="I" and mode == "L") values per band in the\n           image.  A function can be used instead, it should take a\n           single argument. The function is called once for each\n           possible pixel value, and the resulting table is applied to\n           all bands of the image.\n\n           It may also be an :py:class:`~PIL.Image.ImagePointHandler`\n           object::\n\n               class Example(Image.ImagePointHandler):\n                 def point(self, data):\n                   # Return result\n        :param mode: Output mode (default is same as input).  In the\n           current version, this can only be used if the source image\n           has mode "L" or "P", and the output has mode "1" or the\n           source image mode is "I" and the output mode is "L".\n        :returns: An :py:class:`~PIL.Image.Image` object.\n        '
        self.load()
        if isinstance(lut, ImagePointHandler):
            return lut.point(self)
        if callable(lut):
            if self.mode in ('I', 'I;16', 'F'):
                (scale, offset) = _getscaleoffset(lut)
                return self._new(self.im.point_transform(scale, offset))
            lut = [lut(i) for i in range(256)] * self.im.bands
        if self.mode == 'F':
            msg = 'point operation not supported for this mode'
            raise ValueError(msg)
        if mode != 'F':
            lut = [round(i) for i in lut]
        return self._new(self.im.point(lut, mode))

    def putalpha(self, alpha):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds or replaces the alpha layer in this image.  If the image\n        does not have an alpha layer, it\'s converted to "LA" or "RGBA".\n        The new layer must be either "L" or "1".\n\n        :param alpha: The new alpha layer.  This can either be an "L" or "1"\n           image having the same size as this image, or an integer or\n           other color value.\n        '
        self._ensure_mutable()
        if self.mode not in ('LA', 'PA', 'RGBA'):
            try:
                mode = getmodebase(self.mode) + 'A'
                try:
                    self.im.setmode(mode)
                except (AttributeError, ValueError) as e:
                    im = self.im.convert(mode)
                    if im.mode not in ('LA', 'PA', 'RGBA'):
                        msg = 'alpha channel could not be added'
                        raise ValueError(msg) from e
                    self.im = im
                self.pyaccess = None
                self._mode = self.im.mode
            except KeyError as e:
                msg = 'illegal image mode'
                raise ValueError(msg) from e
        if self.mode in ('LA', 'PA'):
            band = 1
        else:
            band = 3
        if isImageType(alpha):
            if alpha.mode not in ('1', 'L'):
                msg = 'illegal image mode'
                raise ValueError(msg)
            alpha.load()
            if alpha.mode == '1':
                alpha = alpha.convert('L')
        else:
            try:
                self.im.fillband(band, alpha)
            except (AttributeError, ValueError):
                alpha = new('L', self.size, alpha)
            else:
                return
        self.im.putband(alpha.im, band)

    def putdata(self, data, scale=1.0, offset=0.0):
        if False:
            while True:
                i = 10
        '\n        Copies pixel data from a flattened sequence object into the image. The\n        values should start at the upper left corner (0, 0), continue to the\n        end of the line, followed directly by the first value of the second\n        line, and so on. Data will be read until either the image or the\n        sequence ends. The scale and offset values are used to adjust the\n        sequence values: **pixel = value*scale + offset**.\n\n        :param data: A flattened sequence object.\n        :param scale: An optional scale value.  The default is 1.0.\n        :param offset: An optional offset value.  The default is 0.0.\n        '
        self._ensure_mutable()
        self.im.putdata(data, scale, offset)

    def putpalette(self, data, rawmode='RGB'):
        if False:
            while True:
                i = 10
        '\n        Attaches a palette to this image.  The image must be a "P", "PA", "L"\n        or "LA" image.\n\n        The palette sequence must contain at most 256 colors, made up of one\n        integer value for each channel in the raw mode.\n        For example, if the raw mode is "RGB", then it can contain at most 768\n        values, made up of red, green and blue values for the corresponding pixel\n        index in the 256 colors.\n        If the raw mode is "RGBA", then it can contain at most 1024 values,\n        containing red, green, blue and alpha values.\n\n        Alternatively, an 8-bit string may be used instead of an integer sequence.\n\n        :param data: A palette sequence (either a list or a string).\n        :param rawmode: The raw mode of the palette. Either "RGB", "RGBA", or a mode\n           that can be transformed to "RGB" or "RGBA" (e.g. "R", "BGR;15", "RGBA;L").\n        '
        from . import ImagePalette
        if self.mode not in ('L', 'LA', 'P', 'PA'):
            msg = 'illegal image mode'
            raise ValueError(msg)
        if isinstance(data, ImagePalette.ImagePalette):
            palette = ImagePalette.raw(data.rawmode, data.palette)
        else:
            if not isinstance(data, bytes):
                data = bytes(data)
            palette = ImagePalette.raw(rawmode, data)
        self._mode = 'PA' if 'A' in self.mode else 'P'
        self.palette = palette
        self.palette.mode = 'RGB'
        self.load()

    def putpixel(self, xy, value):
        if False:
            i = 10
            return i + 15
        '\n        Modifies the pixel at the given position. The color is given as\n        a single numerical value for single-band images, and a tuple for\n        multi-band images. In addition to this, RGB and RGBA tuples are\n        accepted for P and PA images.\n\n        Note that this method is relatively slow.  For more extensive changes,\n        use :py:meth:`~PIL.Image.Image.paste` or the :py:mod:`~PIL.ImageDraw`\n        module instead.\n\n        See:\n\n        * :py:meth:`~PIL.Image.Image.paste`\n        * :py:meth:`~PIL.Image.Image.putdata`\n        * :py:mod:`~PIL.ImageDraw`\n\n        :param xy: The pixel coordinate, given as (x, y). See\n           :ref:`coordinate-system`.\n        :param value: The pixel value.\n        '
        if self.readonly:
            self._copy()
        self.load()
        if self.pyaccess:
            return self.pyaccess.putpixel(xy, value)
        if self.mode in ('P', 'PA') and isinstance(value, (list, tuple)) and (len(value) in [3, 4]):
            if self.mode == 'PA':
                alpha = value[3] if len(value) == 4 else 255
                value = value[:3]
            value = self.palette.getcolor(value, self)
            if self.mode == 'PA':
                value = (value, alpha)
        return self.im.putpixel(xy, value)

    def remap_palette(self, dest_map, source_palette=None):
        if False:
            return 10
        '\n        Rewrites the image to reorder the palette.\n\n        :param dest_map: A list of indexes into the original palette.\n           e.g. ``[1,0]`` would swap a two item palette, and ``list(range(256))``\n           is the identity transform.\n        :param source_palette: Bytes or None.\n        :returns:  An :py:class:`~PIL.Image.Image` object.\n\n        '
        from . import ImagePalette
        if self.mode not in ('L', 'P'):
            msg = 'illegal image mode'
            raise ValueError(msg)
        bands = 3
        palette_mode = 'RGB'
        if source_palette is None:
            if self.mode == 'P':
                self.load()
                palette_mode = self.im.getpalettemode()
                if palette_mode == 'RGBA':
                    bands = 4
                source_palette = self.im.getpalette(palette_mode, palette_mode)
            else:
                source_palette = bytearray((i // 3 for i in range(768)))
        palette_bytes = b''
        new_positions = [0] * 256
        for (i, oldPosition) in enumerate(dest_map):
            palette_bytes += source_palette[oldPosition * bands:oldPosition * bands + bands]
            new_positions[oldPosition] = i
        mapping_palette = bytearray(new_positions)
        m_im = self.copy()
        m_im._mode = 'P'
        m_im.palette = ImagePalette.ImagePalette(palette_mode, palette=mapping_palette * bands)
        m_im.im.putpalette(palette_mode + ';L', m_im.palette.tobytes())
        m_im = m_im.convert('L')
        m_im.putpalette(palette_bytes, palette_mode)
        m_im.palette = ImagePalette.ImagePalette(palette_mode, palette=palette_bytes)
        if 'transparency' in self.info:
            try:
                m_im.info['transparency'] = dest_map.index(self.info['transparency'])
            except ValueError:
                if 'transparency' in m_im.info:
                    del m_im.info['transparency']
        return m_im

    def _get_safe_box(self, size, resample, box):
        if False:
            print('Hello World!')
        'Expands the box so it includes adjacent pixels\n        that may be used by resampling with the given resampling filter.\n        '
        filter_support = _filters_support[resample] - 0.5
        scale_x = (box[2] - box[0]) / size[0]
        scale_y = (box[3] - box[1]) / size[1]
        support_x = filter_support * scale_x
        support_y = filter_support * scale_y
        return (max(0, int(box[0] - support_x)), max(0, int(box[1] - support_y)), min(self.size[0], math.ceil(box[2] + support_x)), min(self.size[1], math.ceil(box[3] + support_y)))

    def resize(self, size, resample=None, box=None, reducing_gap=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns a resized copy of this image.\n\n        :param size: The requested size in pixels, as a 2-tuple:\n           (width, height).\n        :param resample: An optional resampling filter.  This can be\n           one of :py:data:`Resampling.NEAREST`, :py:data:`Resampling.BOX`,\n           :py:data:`Resampling.BILINEAR`, :py:data:`Resampling.HAMMING`,\n           :py:data:`Resampling.BICUBIC` or :py:data:`Resampling.LANCZOS`.\n           If the image has mode "1" or "P", it is always set to\n           :py:data:`Resampling.NEAREST`. If the image mode specifies a number\n           of bits, such as "I;16", then the default filter is\n           :py:data:`Resampling.NEAREST`. Otherwise, the default filter is\n           :py:data:`Resampling.BICUBIC`. See: :ref:`concept-filters`.\n        :param box: An optional 4-tuple of floats providing\n           the source image region to be scaled.\n           The values must be within (0, 0, width, height) rectangle.\n           If omitted or None, the entire source is used.\n        :param reducing_gap: Apply optimization by resizing the image\n           in two steps. First, reducing the image by integer times\n           using :py:meth:`~PIL.Image.Image.reduce`.\n           Second, resizing using regular resampling. The last step\n           changes size no less than by ``reducing_gap`` times.\n           ``reducing_gap`` may be None (no first step is performed)\n           or should be greater than 1.0. The bigger ``reducing_gap``,\n           the closer the result to the fair resampling.\n           The smaller ``reducing_gap``, the faster resizing.\n           With ``reducing_gap`` greater or equal to 3.0, the result is\n           indistinguishable from fair resampling in most cases.\n           The default value is None (no optimization).\n        :returns: An :py:class:`~PIL.Image.Image` object.\n        '
        if resample is None:
            type_special = ';' in self.mode
            resample = Resampling.NEAREST if type_special else Resampling.BICUBIC
        elif resample not in (Resampling.NEAREST, Resampling.BILINEAR, Resampling.BICUBIC, Resampling.LANCZOS, Resampling.BOX, Resampling.HAMMING):
            msg = f'Unknown resampling filter ({resample}).'
            filters = [f'{filter[1]} ({filter[0]})' for filter in ((Resampling.NEAREST, 'Image.Resampling.NEAREST'), (Resampling.LANCZOS, 'Image.Resampling.LANCZOS'), (Resampling.BILINEAR, 'Image.Resampling.BILINEAR'), (Resampling.BICUBIC, 'Image.Resampling.BICUBIC'), (Resampling.BOX, 'Image.Resampling.BOX'), (Resampling.HAMMING, 'Image.Resampling.HAMMING'))]
            msg += ' Use ' + ', '.join(filters[:-1]) + ' or ' + filters[-1]
            raise ValueError(msg)
        if reducing_gap is not None and reducing_gap < 1.0:
            msg = 'reducing_gap must be 1.0 or greater'
            raise ValueError(msg)
        size = tuple(size)
        self.load()
        if box is None:
            box = (0, 0) + self.size
        else:
            box = tuple(box)
        if self.size == size and box == (0, 0) + self.size:
            return self.copy()
        if self.mode in ('1', 'P'):
            resample = Resampling.NEAREST
        if self.mode in ['LA', 'RGBA'] and resample != Resampling.NEAREST:
            im = self.convert({'LA': 'La', 'RGBA': 'RGBa'}[self.mode])
            im = im.resize(size, resample, box)
            return im.convert(self.mode)
        self.load()
        if reducing_gap is not None and resample != Resampling.NEAREST:
            factor_x = int((box[2] - box[0]) / size[0] / reducing_gap) or 1
            factor_y = int((box[3] - box[1]) / size[1] / reducing_gap) or 1
            if factor_x > 1 or factor_y > 1:
                reduce_box = self._get_safe_box(size, resample, box)
                factor = (factor_x, factor_y)
                if callable(self.reduce):
                    self = self.reduce(factor, box=reduce_box)
                else:
                    self = Image.reduce(self, factor, box=reduce_box)
                box = ((box[0] - reduce_box[0]) / factor_x, (box[1] - reduce_box[1]) / factor_y, (box[2] - reduce_box[0]) / factor_x, (box[3] - reduce_box[1]) / factor_y)
        return self._new(self.im.resize(size, resample, box))

    def reduce(self, factor, box=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a copy of the image reduced ``factor`` times.\n        If the size of the image is not dividable by ``factor``,\n        the resulting size will be rounded up.\n\n        :param factor: A greater than 0 integer or tuple of two integers\n           for width and height separately.\n        :param box: An optional 4-tuple of ints providing\n           the source image region to be reduced.\n           The values must be within ``(0, 0, width, height)`` rectangle.\n           If omitted or ``None``, the entire source is used.\n        '
        if not isinstance(factor, (list, tuple)):
            factor = (factor, factor)
        if box is None:
            box = (0, 0) + self.size
        else:
            box = tuple(box)
        if factor == (1, 1) and box == (0, 0) + self.size:
            return self.copy()
        if self.mode in ['LA', 'RGBA']:
            im = self.convert({'LA': 'La', 'RGBA': 'RGBa'}[self.mode])
            im = im.reduce(factor, box)
            return im.convert(self.mode)
        self.load()
        return self._new(self.im.reduce(factor, box))

    def rotate(self, angle, resample=Resampling.NEAREST, expand=0, center=None, translate=None, fillcolor=None):
        if False:
            print('Hello World!')
        '\n        Returns a rotated copy of this image.  This method returns a\n        copy of this image, rotated the given number of degrees counter\n        clockwise around its centre.\n\n        :param angle: In degrees counter clockwise.\n        :param resample: An optional resampling filter.  This can be\n           one of :py:data:`Resampling.NEAREST` (use nearest neighbour),\n           :py:data:`Resampling.BILINEAR` (linear interpolation in a 2x2\n           environment), or :py:data:`Resampling.BICUBIC` (cubic spline\n           interpolation in a 4x4 environment). If omitted, or if the image has\n           mode "1" or "P", it is set to :py:data:`Resampling.NEAREST`.\n           See :ref:`concept-filters`.\n        :param expand: Optional expansion flag.  If true, expands the output\n           image to make it large enough to hold the entire rotated image.\n           If false or omitted, make the output image the same size as the\n           input image.  Note that the expand flag assumes rotation around\n           the center and no translation.\n        :param center: Optional center of rotation (a 2-tuple).  Origin is\n           the upper left corner.  Default is the center of the image.\n        :param translate: An optional post-rotate translation (a 2-tuple).\n        :param fillcolor: An optional color for area outside the rotated image.\n        :returns: An :py:class:`~PIL.Image.Image` object.\n        '
        angle = angle % 360.0
        if not (center or translate):
            if angle == 0:
                return self.copy()
            if angle == 180:
                return self.transpose(Transpose.ROTATE_180)
            if angle in (90, 270) and (expand or self.width == self.height):
                return self.transpose(Transpose.ROTATE_90 if angle == 90 else Transpose.ROTATE_270)
        (w, h) = self.size
        if translate is None:
            post_trans = (0, 0)
        else:
            post_trans = translate
        if center is None:
            rotn_center = (w / 2.0, h / 2.0)
        else:
            rotn_center = center
        angle = -math.radians(angle)
        matrix = [round(math.cos(angle), 15), round(math.sin(angle), 15), 0.0, round(-math.sin(angle), 15), round(math.cos(angle), 15), 0.0]

        def transform(x, y, matrix):
            if False:
                i = 10
                return i + 15
            (a, b, c, d, e, f) = matrix
            return (a * x + b * y + c, d * x + e * y + f)
        (matrix[2], matrix[5]) = transform(-rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix)
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        if expand:
            xx = []
            yy = []
            for (x, y) in ((0, 0), (w, 0), (w, h), (0, h)):
                (x, y) = transform(x, y, matrix)
                xx.append(x)
                yy.append(y)
            nw = math.ceil(max(xx)) - math.floor(min(xx))
            nh = math.ceil(max(yy)) - math.floor(min(yy))
            (matrix[2], matrix[5]) = transform(-(nw - w) / 2.0, -(nh - h) / 2.0, matrix)
            (w, h) = (nw, nh)
        return self.transform((w, h), Transform.AFFINE, matrix, resample, fillcolor=fillcolor)

    def save(self, fp, format=None, **params):
        if False:
            while True:
                i = 10
        "\n        Saves this image under the given filename.  If no format is\n        specified, the format to use is determined from the filename\n        extension, if possible.\n\n        Keyword options can be used to provide additional instructions\n        to the writer. If a writer doesn't recognise an option, it is\n        silently ignored. The available options are described in the\n        :doc:`image format documentation\n        <../handbook/image-file-formats>` for each writer.\n\n        You can use a file object instead of a filename. In this case,\n        you must always specify the format. The file object must\n        implement the ``seek``, ``tell``, and ``write``\n        methods, and be opened in binary mode.\n\n        :param fp: A filename (string), pathlib.Path object or file object.\n        :param format: Optional format override.  If omitted, the\n           format to use is determined from the filename extension.\n           If a file object was used instead of a filename, this\n           parameter should always be used.\n        :param params: Extra parameters to the image writer.\n        :returns: None\n        :exception ValueError: If the output format could not be determined\n           from the file name.  Use the format option to solve this.\n        :exception OSError: If the file could not be written.  The file\n           may have been created, and may contain partial data.\n        "
        filename = ''
        open_fp = False
        if isinstance(fp, Path):
            filename = str(fp)
            open_fp = True
        elif is_path(fp):
            filename = fp
            open_fp = True
        elif fp == sys.stdout:
            try:
                fp = sys.stdout.buffer
            except AttributeError:
                pass
        if not filename and hasattr(fp, 'name') and is_path(fp.name):
            filename = fp.name
        self._ensure_mutable()
        save_all = params.pop('save_all', False)
        self.encoderinfo = params
        self.encoderconfig = ()
        preinit()
        ext = os.path.splitext(filename)[1].lower()
        if not format:
            if ext not in EXTENSION:
                init()
            try:
                format = EXTENSION[ext]
            except KeyError as e:
                msg = f'unknown file extension: {ext}'
                raise ValueError(msg) from e
        if format.upper() not in SAVE:
            init()
        if save_all:
            save_handler = SAVE_ALL[format.upper()]
        else:
            save_handler = SAVE[format.upper()]
        created = False
        if open_fp:
            created = not os.path.exists(filename)
            if params.get('append', False):
                fp = builtins.open(filename, 'r+b')
            else:
                fp = builtins.open(filename, 'w+b')
        try:
            save_handler(self, fp, filename)
        except Exception:
            if open_fp:
                fp.close()
            if created:
                try:
                    os.remove(filename)
                except PermissionError:
                    pass
            raise
        if open_fp:
            fp.close()

    def seek(self, frame):
        if False:
            return 10
        '\n        Seeks to the given frame in this sequence file. If you seek\n        beyond the end of the sequence, the method raises an\n        ``EOFError`` exception. When a sequence file is opened, the\n        library automatically seeks to frame 0.\n\n        See :py:meth:`~PIL.Image.Image.tell`.\n\n        If defined, :attr:`~PIL.Image.Image.n_frames` refers to the\n        number of available frames.\n\n        :param frame: Frame number, starting at 0.\n        :exception EOFError: If the call attempts to seek beyond the end\n            of the sequence.\n        '
        if frame != 0:
            msg = 'no more images in file'
            raise EOFError(msg)

    def show(self, title=None):
        if False:
            while True:
                i = 10
        '\n        Displays this image. This method is mainly intended for debugging purposes.\n\n        This method calls :py:func:`PIL.ImageShow.show` internally. You can use\n        :py:func:`PIL.ImageShow.register` to override its default behaviour.\n\n        The image is first saved to a temporary file. By default, it will be in\n        PNG format.\n\n        On Unix, the image is then opened using the **xdg-open**, **display**,\n        **gm**, **eog** or **xv** utility, depending on which one can be found.\n\n        On macOS, the image is opened with the native Preview application.\n\n        On Windows, the image is opened with the standard PNG display utility.\n\n        :param title: Optional title to use for the image window, where possible.\n        '
        _show(self, title=title)

    def split(self):
        if False:
            while True:
                i = 10
        '\n        Split this image into individual bands. This method returns a\n        tuple of individual image bands from an image. For example,\n        splitting an "RGB" image creates three new images each\n        containing a copy of one of the original bands (red, green,\n        blue).\n\n        If you need only one band, :py:meth:`~PIL.Image.Image.getchannel`\n        method can be more convenient and faster.\n\n        :returns: A tuple containing bands.\n        '
        self.load()
        if self.im.bands == 1:
            ims = [self.copy()]
        else:
            ims = map(self._new, self.im.split())
        return tuple(ims)

    def getchannel(self, channel):
        if False:
            print('Hello World!')
        '\n        Returns an image containing a single channel of the source image.\n\n        :param channel: What channel to return. Could be index\n          (0 for "R" channel of "RGB") or channel name\n          ("A" for alpha channel of "RGBA").\n        :returns: An image in "L" mode.\n\n        .. versionadded:: 4.3.0\n        '
        self.load()
        if isinstance(channel, str):
            try:
                channel = self.getbands().index(channel)
            except ValueError as e:
                msg = f'The image has no channel "{channel}"'
                raise ValueError(msg) from e
        return self._new(self.im.getband(channel))

    def tell(self):
        if False:
            return 10
        '\n        Returns the current frame number. See :py:meth:`~PIL.Image.Image.seek`.\n\n        If defined, :attr:`~PIL.Image.Image.n_frames` refers to the\n        number of available frames.\n\n        :returns: Frame number, starting with 0.\n        '
        return 0

    def thumbnail(self, size, resample=Resampling.BICUBIC, reducing_gap=2.0):
        if False:
            while True:
                i = 10
        '\n        Make this image into a thumbnail.  This method modifies the\n        image to contain a thumbnail version of itself, no larger than\n        the given size.  This method calculates an appropriate thumbnail\n        size to preserve the aspect of the image, calls the\n        :py:meth:`~PIL.Image.Image.draft` method to configure the file reader\n        (where applicable), and finally resizes the image.\n\n        Note that this function modifies the :py:class:`~PIL.Image.Image`\n        object in place.  If you need to use the full resolution image as well,\n        apply this method to a :py:meth:`~PIL.Image.Image.copy` of the original\n        image.\n\n        :param size: The requested size in pixels, as a 2-tuple:\n           (width, height).\n        :param resample: Optional resampling filter.  This can be one\n           of :py:data:`Resampling.NEAREST`, :py:data:`Resampling.BOX`,\n           :py:data:`Resampling.BILINEAR`, :py:data:`Resampling.HAMMING`,\n           :py:data:`Resampling.BICUBIC` or :py:data:`Resampling.LANCZOS`.\n           If omitted, it defaults to :py:data:`Resampling.BICUBIC`.\n           (was :py:data:`Resampling.NEAREST` prior to version 2.5.0).\n           See: :ref:`concept-filters`.\n        :param reducing_gap: Apply optimization by resizing the image\n           in two steps. First, reducing the image by integer times\n           using :py:meth:`~PIL.Image.Image.reduce` or\n           :py:meth:`~PIL.Image.Image.draft` for JPEG images.\n           Second, resizing using regular resampling. The last step\n           changes size no less than by ``reducing_gap`` times.\n           ``reducing_gap`` may be None (no first step is performed)\n           or should be greater than 1.0. The bigger ``reducing_gap``,\n           the closer the result to the fair resampling.\n           The smaller ``reducing_gap``, the faster resizing.\n           With ``reducing_gap`` greater or equal to 3.0, the result is\n           indistinguishable from fair resampling in most cases.\n           The default value is 2.0 (very close to fair resampling\n           while still being faster in many cases).\n        :returns: None\n        '
        provided_size = tuple(map(math.floor, size))

        def preserve_aspect_ratio():
            if False:
                i = 10
                return i + 15

            def round_aspect(number, key):
                if False:
                    i = 10
                    return i + 15
                return max(min(math.floor(number), math.ceil(number), key=key), 1)
            (x, y) = provided_size
            if x >= self.width and y >= self.height:
                return
            aspect = self.width / self.height
            if x / y >= aspect:
                x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
            else:
                y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
            return (x, y)
        box = None
        if reducing_gap is not None:
            size = preserve_aspect_ratio()
            if size is None:
                return
            res = self.draft(None, (size[0] * reducing_gap, size[1] * reducing_gap))
            if res is not None:
                box = res[1]
        if box is None:
            self.load()
            size = preserve_aspect_ratio()
            if size is None:
                return
        if self.size != size:
            im = self.resize(size, resample, box=box, reducing_gap=reducing_gap)
            self.im = im.im
            self._size = size
            self._mode = self.im.mode
        self.readonly = 0
        self.pyaccess = None

    def transform(self, size, method, data=None, resample=Resampling.NEAREST, fill=1, fillcolor=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transforms this image.  This method creates a new image with the\n        given size, and the same mode as the original, and copies data\n        to the new image using the given transform.\n\n        :param size: The output size in pixels, as a 2-tuple:\n           (width, height).\n        :param method: The transformation method.  This is one of\n          :py:data:`Transform.EXTENT` (cut out a rectangular subregion),\n          :py:data:`Transform.AFFINE` (affine transform),\n          :py:data:`Transform.PERSPECTIVE` (perspective transform),\n          :py:data:`Transform.QUAD` (map a quadrilateral to a rectangle), or\n          :py:data:`Transform.MESH` (map a number of source quadrilaterals\n          in one operation).\n\n          It may also be an :py:class:`~PIL.Image.ImageTransformHandler`\n          object::\n\n            class Example(Image.ImageTransformHandler):\n                def transform(self, size, data, resample, fill=1):\n                    # Return result\n\n          It may also be an object with a ``method.getdata`` method\n          that returns a tuple supplying new ``method`` and ``data`` values::\n\n            class Example:\n                def getdata(self):\n                    method = Image.Transform.EXTENT\n                    data = (0, 0, 100, 100)\n                    return method, data\n        :param data: Extra data to the transformation method.\n        :param resample: Optional resampling filter.  It can be one of\n           :py:data:`Resampling.NEAREST` (use nearest neighbour),\n           :py:data:`Resampling.BILINEAR` (linear interpolation in a 2x2\n           environment), or :py:data:`Resampling.BICUBIC` (cubic spline\n           interpolation in a 4x4 environment). If omitted, or if the image\n           has mode "1" or "P", it is set to :py:data:`Resampling.NEAREST`.\n           See: :ref:`concept-filters`.\n        :param fill: If ``method`` is an\n          :py:class:`~PIL.Image.ImageTransformHandler` object, this is one of\n          the arguments passed to it. Otherwise, it is unused.\n        :param fillcolor: Optional fill color for the area outside the\n           transform in the output image.\n        :returns: An :py:class:`~PIL.Image.Image` object.\n        '
        if self.mode in ('LA', 'RGBA') and resample != Resampling.NEAREST:
            return self.convert({'LA': 'La', 'RGBA': 'RGBa'}[self.mode]).transform(size, method, data, resample, fill, fillcolor).convert(self.mode)
        if isinstance(method, ImageTransformHandler):
            return method.transform(size, self, resample=resample, fill=fill)
        if hasattr(method, 'getdata'):
            (method, data) = method.getdata()
        if data is None:
            msg = 'missing method data'
            raise ValueError(msg)
        im = new(self.mode, size, fillcolor)
        if self.mode == 'P' and self.palette:
            im.palette = self.palette.copy()
        im.info = self.info.copy()
        if method == Transform.MESH:
            for (box, quad) in data:
                im.__transformer(box, self, Transform.QUAD, quad, resample, fillcolor is None)
        else:
            im.__transformer((0, 0) + size, self, method, data, resample, fillcolor is None)
        return im

    def __transformer(self, box, image, method, data, resample=Resampling.NEAREST, fill=1):
        if False:
            return 10
        w = box[2] - box[0]
        h = box[3] - box[1]
        if method == Transform.AFFINE:
            data = data[:6]
        elif method == Transform.EXTENT:
            (x0, y0, x1, y1) = data
            xs = (x1 - x0) / w
            ys = (y1 - y0) / h
            method = Transform.AFFINE
            data = (xs, 0, x0, 0, ys, y0)
        elif method == Transform.PERSPECTIVE:
            data = data[:8]
        elif method == Transform.QUAD:
            nw = data[:2]
            sw = data[2:4]
            se = data[4:6]
            ne = data[6:8]
            (x0, y0) = nw
            As = 1.0 / w
            At = 1.0 / h
            data = (x0, (ne[0] - x0) * As, (sw[0] - x0) * At, (se[0] - sw[0] - ne[0] + x0) * As * At, y0, (ne[1] - y0) * As, (sw[1] - y0) * At, (se[1] - sw[1] - ne[1] + y0) * As * At)
        else:
            msg = 'unknown transformation method'
            raise ValueError(msg)
        if resample not in (Resampling.NEAREST, Resampling.BILINEAR, Resampling.BICUBIC):
            if resample in (Resampling.BOX, Resampling.HAMMING, Resampling.LANCZOS):
                msg = {Resampling.BOX: 'Image.Resampling.BOX', Resampling.HAMMING: 'Image.Resampling.HAMMING', Resampling.LANCZOS: 'Image.Resampling.LANCZOS'}[resample] + f' ({resample}) cannot be used.'
            else:
                msg = f'Unknown resampling filter ({resample}).'
            filters = [f'{filter[1]} ({filter[0]})' for filter in ((Resampling.NEAREST, 'Image.Resampling.NEAREST'), (Resampling.BILINEAR, 'Image.Resampling.BILINEAR'), (Resampling.BICUBIC, 'Image.Resampling.BICUBIC'))]
            msg += ' Use ' + ', '.join(filters[:-1]) + ' or ' + filters[-1]
            raise ValueError(msg)
        image.load()
        self.load()
        if image.mode in ('1', 'P'):
            resample = Resampling.NEAREST
        self.im.transform2(box, image.im, method, data, resample, fill)

    def transpose(self, method):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transpose image (flip or rotate in 90 degree steps)\n\n        :param method: One of :py:data:`Transpose.FLIP_LEFT_RIGHT`,\n          :py:data:`Transpose.FLIP_TOP_BOTTOM`, :py:data:`Transpose.ROTATE_90`,\n          :py:data:`Transpose.ROTATE_180`, :py:data:`Transpose.ROTATE_270`,\n          :py:data:`Transpose.TRANSPOSE` or :py:data:`Transpose.TRANSVERSE`.\n        :returns: Returns a flipped or rotated copy of this image.\n        '
        self.load()
        return self._new(self.im.transpose(method))

    def effect_spread(self, distance):
        if False:
            i = 10
            return i + 15
        '\n        Randomly spread pixels in an image.\n\n        :param distance: Distance to spread pixels.\n        '
        self.load()
        return self._new(self.im.effect_spread(distance))

    def toqimage(self):
        if False:
            print('Hello World!')
        'Returns a QImage copy of this image'
        from . import ImageQt
        if not ImageQt.qt_is_installed:
            msg = 'Qt bindings are not installed'
            raise ImportError(msg)
        return ImageQt.toqimage(self)

    def toqpixmap(self):
        if False:
            while True:
                i = 10
        'Returns a QPixmap copy of this image'
        from . import ImageQt
        if not ImageQt.qt_is_installed:
            msg = 'Qt bindings are not installed'
            raise ImportError(msg)
        return ImageQt.toqpixmap(self)

class ImagePointHandler:
    """
    Used as a mixin by point transforms
    (for use with :py:meth:`~PIL.Image.Image.point`)
    """
    pass

class ImageTransformHandler:
    """
    Used as a mixin by geometry transforms
    (for use with :py:meth:`~PIL.Image.Image.transform`)
    """
    pass

def _wedge():
    if False:
        while True:
            i = 10
    'Create grayscale wedge (for debugging only)'
    return Image()._new(core.wedge('L'))

def _check_size(size):
    if False:
        i = 10
        return i + 15
    '\n    Common check to enforce type and sanity check on size tuples\n\n    :param size: Should be a 2 tuple of (width, height)\n    :returns: True, or raises a ValueError\n    '
    if not isinstance(size, (list, tuple)):
        msg = 'Size must be a tuple'
        raise ValueError(msg)
    if len(size) != 2:
        msg = 'Size must be a tuple of length 2'
        raise ValueError(msg)
    if size[0] < 0 or size[1] < 0:
        msg = 'Width and height must be >= 0'
        raise ValueError(msg)
    return True

def new(mode, size, color=0):
    if False:
        return 10
    '\n    Creates a new image with the given mode and size.\n\n    :param mode: The mode to use for the new image. See:\n       :ref:`concept-modes`.\n    :param size: A 2-tuple, containing (width, height) in pixels.\n    :param color: What color to use for the image.  Default is black.\n       If given, this should be a single integer or floating point value\n       for single-band modes, and a tuple for multi-band modes (one value\n       per band).  When creating RGB or HSV images, you can also use color\n       strings as supported by the ImageColor module.  If the color is\n       None, the image is not initialised.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    '
    _check_size(size)
    if color is None:
        return Image()._new(core.new(mode, size))
    if isinstance(color, str):
        from . import ImageColor
        color = ImageColor.getcolor(color, mode)
    im = Image()
    if mode == 'P' and isinstance(color, (list, tuple)) and (len(color) in [3, 4]):
        from . import ImagePalette
        im.palette = ImagePalette.ImagePalette()
        color = im.palette.getcolor(color)
    return im._new(core.fill(mode, size, color))

def frombytes(mode, size, data, decoder_name='raw', *args):
    if False:
        return 10
    '\n    Creates a copy of an image memory from pixel data in a buffer.\n\n    In its simplest form, this function takes three arguments\n    (mode, size, and unpacked pixel data).\n\n    You can also use any pixel decoder supported by PIL. For more\n    information on available decoders, see the section\n    :ref:`Writing Your Own File Codec <file-codecs>`.\n\n    Note that this function decodes pixel data only, not entire images.\n    If you have an entire image in a string, wrap it in a\n    :py:class:`~io.BytesIO` object, and use :py:func:`~PIL.Image.open` to load\n    it.\n\n    :param mode: The image mode. See: :ref:`concept-modes`.\n    :param size: The image size.\n    :param data: A byte buffer containing raw data for the given mode.\n    :param decoder_name: What decoder to use.\n    :param args: Additional parameters for the given decoder.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    '
    _check_size(size)
    im = new(mode, size)
    if im.width != 0 and im.height != 0:
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        if decoder_name == 'raw' and args == ():
            args = mode
        im.frombytes(data, decoder_name, args)
    return im

def frombuffer(mode, size, data, decoder_name='raw', *args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates an image memory referencing pixel data in a byte buffer.\n\n    This function is similar to :py:func:`~PIL.Image.frombytes`, but uses data\n    in the byte buffer, where possible.  This means that changes to the\n    original buffer object are reflected in this image).  Not all modes can\n    share memory; supported modes include "L", "RGBX", "RGBA", and "CMYK".\n\n    Note that this function decodes pixel data only, not entire images.\n    If you have an entire image file in a string, wrap it in a\n    :py:class:`~io.BytesIO` object, and use :py:func:`~PIL.Image.open` to load it.\n\n    In the current version, the default parameters used for the "raw" decoder\n    differs from that used for :py:func:`~PIL.Image.frombytes`.  This is a\n    bug, and will probably be fixed in a future release.  The current release\n    issues a warning if you do this; to disable the warning, you should provide\n    the full set of parameters.  See below for details.\n\n    :param mode: The image mode. See: :ref:`concept-modes`.\n    :param size: The image size.\n    :param data: A bytes or other buffer object containing raw\n        data for the given mode.\n    :param decoder_name: What decoder to use.\n    :param args: Additional parameters for the given decoder.  For the\n        default encoder ("raw"), it\'s recommended that you provide the\n        full set of parameters::\n\n            frombuffer(mode, size, data, "raw", mode, 0, 1)\n\n    :returns: An :py:class:`~PIL.Image.Image` object.\n\n    .. versionadded:: 1.1.4\n    '
    _check_size(size)
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]
    if decoder_name == 'raw':
        if args == ():
            args = (mode, 0, 1)
        if args[0] in _MAPMODES:
            im = new(mode, (0, 0))
            im = im._new(core.map_buffer(data, size, decoder_name, 0, args))
            if mode == 'P':
                from . import ImagePalette
                im.palette = ImagePalette.ImagePalette('RGB', im.im.getpalette('RGB'))
            im.readonly = 1
            return im
    return frombytes(mode, size, data, decoder_name, args)

def fromarray(obj, mode=None):
    if False:
        print('Hello World!')
    '\n    Creates an image memory from an object exporting the array interface\n    (using the buffer protocol)::\n\n      from PIL import Image\n      import numpy as np\n      a = np.zeros((5, 5))\n      im = Image.fromarray(a)\n\n    If ``obj`` is not contiguous, then the ``tobytes`` method is called\n    and :py:func:`~PIL.Image.frombuffer` is used.\n\n    In the case of NumPy, be aware that Pillow modes do not always correspond\n    to NumPy dtypes. Pillow modes only offer 1-bit pixels, 8-bit pixels,\n    32-bit signed integer pixels, and 32-bit floating point pixels.\n\n    Pillow images can also be converted to arrays::\n\n      from PIL import Image\n      import numpy as np\n      im = Image.open("hopper.jpg")\n      a = np.asarray(im)\n\n    When converting Pillow images to arrays however, only pixel values are\n    transferred. This means that P and PA mode images will lose their palette.\n\n    :param obj: Object with array interface\n    :param mode: Optional mode to use when reading ``obj``. Will be determined from\n      type if ``None``.\n\n      This will not be used to convert the data after reading, but will be used to\n      change how the data is read::\n\n        from PIL import Image\n        import numpy as np\n        a = np.full((1, 1), 300)\n        im = Image.fromarray(a, mode="L")\n        im.getpixel((0, 0))  # 44\n        im = Image.fromarray(a, mode="RGB")\n        im.getpixel((0, 0))  # (44, 1, 0)\n\n      See: :ref:`concept-modes` for general information about modes.\n    :returns: An image object.\n\n    .. versionadded:: 1.1.6\n    '
    arr = obj.__array_interface__
    shape = arr['shape']
    ndim = len(shape)
    strides = arr.get('strides', None)
    if mode is None:
        try:
            typekey = ((1, 1) + shape[2:], arr['typestr'])
        except KeyError as e:
            msg = 'Cannot handle this data type'
            raise TypeError(msg) from e
        try:
            (mode, rawmode) = _fromarray_typemap[typekey]
        except KeyError as e:
            (typekey_shape, typestr) = typekey
            msg = f'Cannot handle this data type: {typekey_shape}, {typestr}'
            raise TypeError(msg) from e
    else:
        rawmode = mode
    if mode in ['1', 'L', 'I', 'P', 'F']:
        ndmax = 2
    elif mode == 'RGB':
        ndmax = 3
    else:
        ndmax = 4
    if ndim > ndmax:
        msg = f'Too many dimensions: {ndim} > {ndmax}.'
        raise ValueError(msg)
    size = (1 if ndim == 1 else shape[1], shape[0])
    if strides is not None:
        if hasattr(obj, 'tobytes'):
            obj = obj.tobytes()
        else:
            obj = obj.tostring()
    return frombuffer(mode, size, obj, 'raw', rawmode, 0, 1)

def fromqimage(im):
    if False:
        return 10
    'Creates an image instance from a QImage image'
    from . import ImageQt
    if not ImageQt.qt_is_installed:
        msg = 'Qt bindings are not installed'
        raise ImportError(msg)
    return ImageQt.fromqimage(im)

def fromqpixmap(im):
    if False:
        for i in range(10):
            print('nop')
    'Creates an image instance from a QPixmap image'
    from . import ImageQt
    if not ImageQt.qt_is_installed:
        msg = 'Qt bindings are not installed'
        raise ImportError(msg)
    return ImageQt.fromqpixmap(im)
_fromarray_typemap = {((1, 1), '|b1'): ('1', '1;8'), ((1, 1), '|u1'): ('L', 'L'), ((1, 1), '|i1'): ('I', 'I;8'), ((1, 1), '<u2'): ('I', 'I;16'), ((1, 1), '>u2'): ('I', 'I;16B'), ((1, 1), '<i2'): ('I', 'I;16S'), ((1, 1), '>i2'): ('I', 'I;16BS'), ((1, 1), '<u4'): ('I', 'I;32'), ((1, 1), '>u4'): ('I', 'I;32B'), ((1, 1), '<i4'): ('I', 'I;32S'), ((1, 1), '>i4'): ('I', 'I;32BS'), ((1, 1), '<f4'): ('F', 'F;32F'), ((1, 1), '>f4'): ('F', 'F;32BF'), ((1, 1), '<f8'): ('F', 'F;64F'), ((1, 1), '>f8'): ('F', 'F;64BF'), ((1, 1, 2), '|u1'): ('LA', 'LA'), ((1, 1, 3), '|u1'): ('RGB', 'RGB'), ((1, 1, 4), '|u1'): ('RGBA', 'RGBA'), ((1, 1), _ENDIAN + 'i4'): ('I', 'I'), ((1, 1), _ENDIAN + 'f4'): ('F', 'F')}

def _decompression_bomb_check(size):
    if False:
        print('Hello World!')
    if MAX_IMAGE_PIXELS is None:
        return
    pixels = max(1, size[0]) * max(1, size[1])
    if pixels > 2 * MAX_IMAGE_PIXELS:
        msg = f'Image size ({pixels} pixels) exceeds limit of {2 * MAX_IMAGE_PIXELS} pixels, could be decompression bomb DOS attack.'
        raise DecompressionBombError(msg)
    if pixels > MAX_IMAGE_PIXELS:
        warnings.warn(f'Image size ({pixels} pixels) exceeds limit of {MAX_IMAGE_PIXELS} pixels, could be decompression bomb DOS attack.', DecompressionBombWarning)

def open(fp, mode='r', formats=None):
    if False:
        return 10
    '\n    Opens and identifies the given image file.\n\n    This is a lazy operation; this function identifies the file, but\n    the file remains open and the actual image data is not read from\n    the file until you try to process the data (or call the\n    :py:meth:`~PIL.Image.Image.load` method).  See\n    :py:func:`~PIL.Image.new`. See :ref:`file-handling`.\n\n    :param fp: A filename (string), pathlib.Path object or a file object.\n       The file object must implement ``file.read``,\n       ``file.seek``, and ``file.tell`` methods,\n       and be opened in binary mode. The file object will also seek to zero\n       before reading.\n    :param mode: The mode.  If given, this argument must be "r".\n    :param formats: A list or tuple of formats to attempt to load the file in.\n       This can be used to restrict the set of formats checked.\n       Pass ``None`` to try all supported formats. You can print the set of\n       available formats by running ``python3 -m PIL`` or using\n       the :py:func:`PIL.features.pilinfo` function.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    :exception FileNotFoundError: If the file cannot be found.\n    :exception PIL.UnidentifiedImageError: If the image cannot be opened and\n       identified.\n    :exception ValueError: If the ``mode`` is not "r", or if a ``StringIO``\n       instance is used for ``fp``.\n    :exception TypeError: If ``formats`` is not ``None``, a list or a tuple.\n    '
    if mode != 'r':
        msg = f'bad mode {repr(mode)}'
        raise ValueError(msg)
    elif isinstance(fp, io.StringIO):
        msg = 'StringIO cannot be used to open an image. Binary data must be used instead.'
        raise ValueError(msg)
    if formats is None:
        formats = ID
    elif not isinstance(formats, (list, tuple)):
        msg = 'formats must be a list or tuple'
        raise TypeError(msg)
    exclusive_fp = False
    filename = ''
    if isinstance(fp, Path):
        filename = str(fp.resolve())
    elif is_path(fp):
        filename = fp
    if filename:
        fp = builtins.open(filename, 'rb')
        exclusive_fp = True
    try:
        fp.seek(0)
    except (AttributeError, io.UnsupportedOperation):
        fp = io.BytesIO(fp.read())
        exclusive_fp = True
    prefix = fp.read(16)
    preinit()
    accept_warnings = []

    def _open_core(fp, filename, prefix, formats):
        if False:
            i = 10
            return i + 15
        for i in formats:
            i = i.upper()
            if i not in OPEN:
                init()
            try:
                (factory, accept) = OPEN[i]
                result = not accept or accept(prefix)
                if type(result) in [str, bytes]:
                    accept_warnings.append(result)
                elif result:
                    fp.seek(0)
                    im = factory(fp, filename)
                    _decompression_bomb_check(im.size)
                    return im
            except (SyntaxError, IndexError, TypeError, struct.error):
                continue
            except BaseException:
                if exclusive_fp:
                    fp.close()
                raise
        return None
    im = _open_core(fp, filename, prefix, formats)
    if im is None and formats is ID:
        checked_formats = formats.copy()
        if init():
            im = _open_core(fp, filename, prefix, tuple((format for format in formats if format not in checked_formats)))
    if im:
        im._exclusive_fp = exclusive_fp
        return im
    if exclusive_fp:
        fp.close()
    for message in accept_warnings:
        warnings.warn(message)
    msg = 'cannot identify image file %r' % (filename if filename else fp)
    raise UnidentifiedImageError(msg)

def alpha_composite(im1, im2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Alpha composite im2 over im1.\n\n    :param im1: The first image. Must have mode RGBA.\n    :param im2: The second image.  Must have mode RGBA, and the same size as\n       the first image.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    '
    im1.load()
    im2.load()
    return im1._new(core.alpha_composite(im1.im, im2.im))

def blend(im1, im2, alpha):
    if False:
        i = 10
        return i + 15
    '\n    Creates a new image by interpolating between two input images, using\n    a constant alpha::\n\n        out = image1 * (1.0 - alpha) + image2 * alpha\n\n    :param im1: The first image.\n    :param im2: The second image.  Must have the same mode and size as\n       the first image.\n    :param alpha: The interpolation alpha factor.  If alpha is 0.0, a\n       copy of the first image is returned. If alpha is 1.0, a copy of\n       the second image is returned. There are no restrictions on the\n       alpha value. If necessary, the result is clipped to fit into\n       the allowed output range.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    '
    im1.load()
    im2.load()
    return im1._new(core.blend(im1.im, im2.im, alpha))

def composite(image1, image2, mask):
    if False:
        return 10
    '\n    Create composite image by blending images using a transparency mask.\n\n    :param image1: The first image.\n    :param image2: The second image.  Must have the same mode and\n       size as the first image.\n    :param mask: A mask image.  This image can have mode\n       "1", "L", or "RGBA", and must have the same size as the\n       other two images.\n    '
    image = image2.copy()
    image.paste(image1, None, mask)
    return image

def eval(image, *args):
    if False:
        i = 10
        return i + 15
    '\n    Applies the function (which should take one argument) to each pixel\n    in the given image. If the image has more than one band, the same\n    function is applied to each band. Note that the function is\n    evaluated once for each possible pixel value, so you cannot use\n    random components or other generators.\n\n    :param image: The input image.\n    :param function: A function object, taking one integer argument.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    '
    return image.point(args[0])

def merge(mode, bands):
    if False:
        print('Hello World!')
    '\n    Merge a set of single band images into a new multiband image.\n\n    :param mode: The mode to use for the output image. See:\n        :ref:`concept-modes`.\n    :param bands: A sequence containing one single-band image for\n        each band in the output image.  All bands must have the\n        same size.\n    :returns: An :py:class:`~PIL.Image.Image` object.\n    '
    if getmodebands(mode) != len(bands) or '*' in mode:
        msg = 'wrong number of bands'
        raise ValueError(msg)
    for band in bands[1:]:
        if band.mode != getmodetype(mode):
            msg = 'mode mismatch'
            raise ValueError(msg)
        if band.size != bands[0].size:
            msg = 'size mismatch'
            raise ValueError(msg)
    for band in bands:
        band.load()
    return bands[0]._new(core.merge(mode, *[b.im for b in bands]))

def register_open(id, factory, accept=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Register an image file plugin.  This function should not be used\n    in application code.\n\n    :param id: An image format identifier.\n    :param factory: An image file factory method.\n    :param accept: An optional function that can be used to quickly\n       reject images having another format.\n    '
    id = id.upper()
    if id not in ID:
        ID.append(id)
    OPEN[id] = (factory, accept)

def register_mime(id, mimetype):
    if False:
        print('Hello World!')
    '\n    Registers an image MIME type by populating ``Image.MIME``. This function\n    should not be used in application code.\n\n    ``Image.MIME`` provides a mapping from image format identifiers to mime\n    formats, but :py:meth:`~PIL.ImageFile.ImageFile.get_format_mimetype` can\n    provide a different result for specific images.\n\n    :param id: An image format identifier.\n    :param mimetype: The image MIME type for this format.\n    '
    MIME[id.upper()] = mimetype

def register_save(id, driver):
    if False:
        print('Hello World!')
    '\n    Registers an image save function.  This function should not be\n    used in application code.\n\n    :param id: An image format identifier.\n    :param driver: A function to save images in this format.\n    '
    SAVE[id.upper()] = driver

def register_save_all(id, driver):
    if False:
        for i in range(10):
            print('nop')
    '\n    Registers an image function to save all the frames\n    of a multiframe format.  This function should not be\n    used in application code.\n\n    :param id: An image format identifier.\n    :param driver: A function to save images in this format.\n    '
    SAVE_ALL[id.upper()] = driver

def register_extension(id, extension):
    if False:
        for i in range(10):
            print('nop')
    '\n    Registers an image extension.  This function should not be\n    used in application code.\n\n    :param id: An image format identifier.\n    :param extension: An extension used for this format.\n    '
    EXTENSION[extension.lower()] = id.upper()

def register_extensions(id, extensions):
    if False:
        return 10
    '\n    Registers image extensions.  This function should not be\n    used in application code.\n\n    :param id: An image format identifier.\n    :param extensions: A list of extensions used for this format.\n    '
    for extension in extensions:
        register_extension(id, extension)

def registered_extensions():
    if False:
        print('Hello World!')
    '\n    Returns a dictionary containing all file extensions belonging\n    to registered plugins\n    '
    init()
    return EXTENSION

def register_decoder(name, decoder):
    if False:
        for i in range(10):
            print('nop')
    '\n    Registers an image decoder.  This function should not be\n    used in application code.\n\n    :param name: The name of the decoder\n    :param decoder: A callable(mode, args) that returns an\n                    ImageFile.PyDecoder object\n\n    .. versionadded:: 4.1.0\n    '
    DECODERS[name] = decoder

def register_encoder(name, encoder):
    if False:
        i = 10
        return i + 15
    '\n    Registers an image encoder.  This function should not be\n    used in application code.\n\n    :param name: The name of the encoder\n    :param encoder: A callable(mode, args) that returns an\n                    ImageFile.PyEncoder object\n\n    .. versionadded:: 4.1.0\n    '
    ENCODERS[name] = encoder

def _show(image, **options):
    if False:
        print('Hello World!')
    from . import ImageShow
    ImageShow.show(image, **options)

def effect_mandelbrot(size, extent, quality):
    if False:
        return 10
    '\n    Generate a Mandelbrot set covering the given extent.\n\n    :param size: The requested size in pixels, as a 2-tuple:\n       (width, height).\n    :param extent: The extent to cover, as a 4-tuple:\n       (x0, y0, x1, y1).\n    :param quality: Quality.\n    '
    return Image()._new(core.effect_mandelbrot(size, extent, quality))

def effect_noise(size, sigma):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate Gaussian noise centered around 128.\n\n    :param size: The requested size in pixels, as a 2-tuple:\n       (width, height).\n    :param sigma: Standard deviation of noise.\n    '
    return Image()._new(core.effect_noise(size, sigma))

def linear_gradient(mode):
    if False:
        print('Hello World!')
    '\n    Generate 256x256 linear gradient from black to white, top to bottom.\n\n    :param mode: Input mode.\n    '
    return Image()._new(core.linear_gradient(mode))

def radial_gradient(mode):
    if False:
        return 10
    '\n    Generate 256x256 radial gradient from black to white, centre to edge.\n\n    :param mode: Input mode.\n    '
    return Image()._new(core.radial_gradient(mode))

def _apply_env_variables(env=None):
    if False:
        i = 10
        return i + 15
    if env is None:
        env = os.environ
    for (var_name, setter) in [('PILLOW_ALIGNMENT', core.set_alignment), ('PILLOW_BLOCK_SIZE', core.set_block_size), ('PILLOW_BLOCKS_MAX', core.set_blocks_max)]:
        if var_name not in env:
            continue
        var = env[var_name].lower()
        units = 1
        for (postfix, mul) in [('k', 1024), ('m', 1024 * 1024)]:
            if var.endswith(postfix):
                units = mul
                var = var[:-len(postfix)]
        try:
            var = int(var) * units
        except ValueError:
            warnings.warn(f'{var_name} is not int')
            continue
        try:
            setter(var)
        except ValueError as e:
            warnings.warn(f'{var_name}: {e}')
_apply_env_variables()
atexit.register(core.clear_cache)

class Exif(MutableMapping):
    """
    This class provides read and write access to EXIF image data::

      from PIL import Image
      im = Image.open("exif.png")
      exif = im.getexif()  # Returns an instance of this class

    Information can be read and written, iterated over or deleted::

      print(exif[274])  # 1
      exif[274] = 2
      for k, v in exif.items():
        print("Tag", k, "Value", v)  # Tag 274 Value 2
      del exif[274]

    To access information beyond IFD0, :py:meth:`~PIL.Image.Exif.get_ifd`
    returns a dictionary::

      from PIL import ExifTags
      im = Image.open("exif_gps.jpg")
      exif = im.getexif()
      gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
      print(gps_ifd)

    Other IFDs include ``ExifTags.IFD.Exif``, ``ExifTags.IFD.Makernote``,
    ``ExifTags.IFD.Interop`` and ``ExifTags.IFD.IFD1``.

    :py:mod:`~PIL.ExifTags` also has enum classes to provide names for data::

      print(exif[ExifTags.Base.Software])  # PIL
      print(gps_ifd[ExifTags.GPS.GPSDateStamp])  # 1999:99:99 99:99:99
    """
    endian = None
    bigtiff = False

    def __init__(self):
        if False:
            print('Hello World!')
        self._data = {}
        self._hidden_data = {}
        self._ifds = {}
        self._info = None
        self._loaded_exif = None

    def _fixup(self, value):
        if False:
            print('Hello World!')
        try:
            if len(value) == 1 and isinstance(value, tuple):
                return value[0]
        except Exception:
            pass
        return value

    def _fixup_dict(self, src_dict):
        if False:
            i = 10
            return i + 15
        return {k: self._fixup(v) for (k, v) in src_dict.items()}

    def _get_ifd_dict(self, offset):
        if False:
            i = 10
            return i + 15
        try:
            self.fp.seek(offset)
        except (KeyError, TypeError):
            pass
        else:
            from . import TiffImagePlugin
            info = TiffImagePlugin.ImageFileDirectory_v2(self.head)
            info.load(self.fp)
            return self._fixup_dict(info)

    def _get_head(self):
        if False:
            i = 10
            return i + 15
        version = b'+' if self.bigtiff else b'*'
        if self.endian == '<':
            head = b'II' + version + b'\x00' + o32le(8)
        else:
            head = b'MM\x00' + version + o32be(8)
        if self.bigtiff:
            head += o32le(8) if self.endian == '<' else o32be(8)
            head += b'\x00\x00\x00\x00'
        return head

    def load(self, data):
        if False:
            for i in range(10):
                print('nop')
        if data == self._loaded_exif:
            return
        self._loaded_exif = data
        self._data.clear()
        self._hidden_data.clear()
        self._ifds.clear()
        if data and data.startswith(b'Exif\x00\x00'):
            data = data[6:]
        if not data:
            self._info = None
            return
        self.fp = io.BytesIO(data)
        self.head = self.fp.read(8)
        from . import TiffImagePlugin
        self._info = TiffImagePlugin.ImageFileDirectory_v2(self.head)
        self.endian = self._info._endian
        self.fp.seek(self._info.next)
        self._info.load(self.fp)

    def load_from_fp(self, fp, offset=None):
        if False:
            print('Hello World!')
        self._loaded_exif = None
        self._data.clear()
        self._hidden_data.clear()
        self._ifds.clear()
        from . import TiffImagePlugin
        self.fp = fp
        if offset is not None:
            self.head = self._get_head()
        else:
            self.head = self.fp.read(8)
        self._info = TiffImagePlugin.ImageFileDirectory_v2(self.head)
        if self.endian is None:
            self.endian = self._info._endian
        if offset is None:
            offset = self._info.next
        self.fp.tell()
        self.fp.seek(offset)
        self._info.load(self.fp)

    def _get_merged_dict(self):
        if False:
            for i in range(10):
                print('nop')
        merged_dict = dict(self)
        if ExifTags.IFD.Exif in self:
            ifd = self._get_ifd_dict(self[ExifTags.IFD.Exif])
            if ifd:
                merged_dict.update(ifd)
        if ExifTags.IFD.GPSInfo in self:
            merged_dict[ExifTags.IFD.GPSInfo] = self._get_ifd_dict(self[ExifTags.IFD.GPSInfo])
        return merged_dict

    def tobytes(self, offset=8):
        if False:
            while True:
                i = 10
        from . import TiffImagePlugin
        head = self._get_head()
        ifd = TiffImagePlugin.ImageFileDirectory_v2(ifh=head)
        for (tag, value) in self.items():
            if tag in [ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo] and (not isinstance(value, dict)):
                value = self.get_ifd(tag)
                if tag == ExifTags.IFD.Exif and ExifTags.IFD.Interop in value and (not isinstance(value[ExifTags.IFD.Interop], dict)):
                    value = value.copy()
                    value[ExifTags.IFD.Interop] = self.get_ifd(ExifTags.IFD.Interop)
            ifd[tag] = value
        return b'Exif\x00\x00' + head + ifd.tobytes(offset)

    def get_ifd(self, tag):
        if False:
            return 10
        if tag not in self._ifds:
            if tag == ExifTags.IFD.IFD1:
                if self._info is not None and self._info.next != 0:
                    self._ifds[tag] = self._get_ifd_dict(self._info.next)
            elif tag in [ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo]:
                offset = self._hidden_data.get(tag, self.get(tag))
                if offset is not None:
                    self._ifds[tag] = self._get_ifd_dict(offset)
            elif tag in [ExifTags.IFD.Interop, ExifTags.IFD.Makernote]:
                if ExifTags.IFD.Exif not in self._ifds:
                    self.get_ifd(ExifTags.IFD.Exif)
                tag_data = self._ifds[ExifTags.IFD.Exif][tag]
                if tag == ExifTags.IFD.Makernote:
                    from .TiffImagePlugin import ImageFileDirectory_v2
                    if tag_data[:8] == b'FUJIFILM':
                        ifd_offset = i32le(tag_data, 8)
                        ifd_data = tag_data[ifd_offset:]
                        makernote = {}
                        for i in range(0, struct.unpack('<H', ifd_data[:2])[0]):
                            (ifd_tag, typ, count, data) = struct.unpack('<HHL4s', ifd_data[i * 12 + 2:(i + 1) * 12 + 2])
                            try:
                                (unit_size, handler) = ImageFileDirectory_v2._load_dispatch[typ]
                            except KeyError:
                                continue
                            size = count * unit_size
                            if size > 4:
                                (offset,) = struct.unpack('<L', data)
                                data = ifd_data[offset - 12:offset + size - 12]
                            else:
                                data = data[:size]
                            if len(data) != size:
                                warnings.warn(f'Possibly corrupt EXIF MakerNote data.  Expecting to read {size} bytes but only got {len(data)}. Skipping tag {ifd_tag}')
                                continue
                            if not data:
                                continue
                            makernote[ifd_tag] = handler(ImageFileDirectory_v2(), data, False)
                        self._ifds[tag] = dict(self._fixup_dict(makernote))
                    elif self.get(271) == 'Nintendo':
                        makernote = {}
                        for i in range(0, struct.unpack('>H', tag_data[:2])[0]):
                            (ifd_tag, typ, count, data) = struct.unpack('>HHL4s', tag_data[i * 12 + 2:(i + 1) * 12 + 2])
                            if ifd_tag == 4353:
                                (offset,) = struct.unpack('>L', data)
                                self.fp.seek(offset)
                                camerainfo = {'ModelID': self.fp.read(4)}
                                self.fp.read(4)
                                camerainfo['TimeStamp'] = i32le(self.fp.read(12))
                                self.fp.read(4)
                                camerainfo['InternalSerialNumber'] = self.fp.read(4)
                                self.fp.read(12)
                                parallax = self.fp.read(4)
                                handler = ImageFileDirectory_v2._load_dispatch[TiffTags.FLOAT][1]
                                camerainfo['Parallax'] = handler(ImageFileDirectory_v2(), parallax, False)
                                self.fp.read(4)
                                camerainfo['Category'] = self.fp.read(2)
                                makernote = {4353: dict(self._fixup_dict(camerainfo))}
                        self._ifds[tag] = makernote
                else:
                    self._ifds[tag] = self._get_ifd_dict(tag_data)
        ifd = self._ifds.get(tag, {})
        if tag == ExifTags.IFD.Exif and self._hidden_data:
            ifd = {k: v for (k, v) in ifd.items() if k not in (ExifTags.IFD.Interop, ExifTags.IFD.Makernote)}
        return ifd

    def hide_offsets(self):
        if False:
            return 10
        for tag in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
            if tag in self:
                self._hidden_data[tag] = self[tag]
                del self[tag]

    def __str__(self):
        if False:
            print('Hello World!')
        if self._info is not None:
            for tag in self._info:
                self[tag]
        return str(self._data)

    def __len__(self):
        if False:
            while True:
                i = 10
        keys = set(self._data)
        if self._info is not None:
            keys.update(self._info)
        return len(keys)

    def __getitem__(self, tag):
        if False:
            while True:
                i = 10
        if self._info is not None and tag not in self._data and (tag in self._info):
            self._data[tag] = self._fixup(self._info[tag])
            del self._info[tag]
        return self._data[tag]

    def __contains__(self, tag):
        if False:
            return 10
        return tag in self._data or (self._info is not None and tag in self._info)

    def __setitem__(self, tag, value):
        if False:
            return 10
        if self._info is not None and tag in self._info:
            del self._info[tag]
        self._data[tag] = value

    def __delitem__(self, tag):
        if False:
            i = 10
            return i + 15
        if self._info is not None and tag in self._info:
            del self._info[tag]
        else:
            del self._data[tag]

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        keys = set(self._data)
        if self._info is not None:
            keys.update(self._info)
        return iter(keys)