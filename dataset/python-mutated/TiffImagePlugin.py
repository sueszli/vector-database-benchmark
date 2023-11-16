import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
logger = logging.getLogger(__name__)
READ_LIBTIFF = False
WRITE_LIBTIFF = False
IFD_LEGACY_API = True
STRIP_SIZE = 65536
II = b'II'
MM = b'MM'
IMAGEWIDTH = 256
IMAGELENGTH = 257
BITSPERSAMPLE = 258
COMPRESSION = 259
PHOTOMETRIC_INTERPRETATION = 262
FILLORDER = 266
IMAGEDESCRIPTION = 270
STRIPOFFSETS = 273
SAMPLESPERPIXEL = 277
ROWSPERSTRIP = 278
STRIPBYTECOUNTS = 279
X_RESOLUTION = 282
Y_RESOLUTION = 283
PLANAR_CONFIGURATION = 284
RESOLUTION_UNIT = 296
TRANSFERFUNCTION = 301
SOFTWARE = 305
DATE_TIME = 306
ARTIST = 315
PREDICTOR = 317
COLORMAP = 320
TILEWIDTH = 322
TILELENGTH = 323
TILEOFFSETS = 324
TILEBYTECOUNTS = 325
SUBIFD = 330
EXTRASAMPLES = 338
SAMPLEFORMAT = 339
JPEGTABLES = 347
YCBCRSUBSAMPLING = 530
REFERENCEBLACKWHITE = 532
COPYRIGHT = 33432
IPTC_NAA_CHUNK = 33723
PHOTOSHOP_CHUNK = 34377
ICCPROFILE = 34675
EXIFIFD = 34665
XMP = 700
JPEGQUALITY = 65537
IMAGEJ_META_DATA_BYTE_COUNTS = 50838
IMAGEJ_META_DATA = 50839
COMPRESSION_INFO = {1: 'raw', 2: 'tiff_ccitt', 3: 'group3', 4: 'group4', 5: 'tiff_lzw', 6: 'tiff_jpeg', 7: 'jpeg', 8: 'tiff_adobe_deflate', 32771: 'tiff_raw_16', 32773: 'packbits', 32809: 'tiff_thunderscan', 32946: 'tiff_deflate', 34676: 'tiff_sgilog', 34677: 'tiff_sgilog24', 34925: 'lzma', 50000: 'zstd', 50001: 'webp'}
COMPRESSION_INFO_REV = {v: k for (k, v) in COMPRESSION_INFO.items()}
OPEN_INFO = {(II, 0, (1,), 1, (1,), ()): ('1', '1;I'), (MM, 0, (1,), 1, (1,), ()): ('1', '1;I'), (II, 0, (1,), 2, (1,), ()): ('1', '1;IR'), (MM, 0, (1,), 2, (1,), ()): ('1', '1;IR'), (II, 1, (1,), 1, (1,), ()): ('1', '1'), (MM, 1, (1,), 1, (1,), ()): ('1', '1'), (II, 1, (1,), 2, (1,), ()): ('1', '1;R'), (MM, 1, (1,), 2, (1,), ()): ('1', '1;R'), (II, 0, (1,), 1, (2,), ()): ('L', 'L;2I'), (MM, 0, (1,), 1, (2,), ()): ('L', 'L;2I'), (II, 0, (1,), 2, (2,), ()): ('L', 'L;2IR'), (MM, 0, (1,), 2, (2,), ()): ('L', 'L;2IR'), (II, 1, (1,), 1, (2,), ()): ('L', 'L;2'), (MM, 1, (1,), 1, (2,), ()): ('L', 'L;2'), (II, 1, (1,), 2, (2,), ()): ('L', 'L;2R'), (MM, 1, (1,), 2, (2,), ()): ('L', 'L;2R'), (II, 0, (1,), 1, (4,), ()): ('L', 'L;4I'), (MM, 0, (1,), 1, (4,), ()): ('L', 'L;4I'), (II, 0, (1,), 2, (4,), ()): ('L', 'L;4IR'), (MM, 0, (1,), 2, (4,), ()): ('L', 'L;4IR'), (II, 1, (1,), 1, (4,), ()): ('L', 'L;4'), (MM, 1, (1,), 1, (4,), ()): ('L', 'L;4'), (II, 1, (1,), 2, (4,), ()): ('L', 'L;4R'), (MM, 1, (1,), 2, (4,), ()): ('L', 'L;4R'), (II, 0, (1,), 1, (8,), ()): ('L', 'L;I'), (MM, 0, (1,), 1, (8,), ()): ('L', 'L;I'), (II, 0, (1,), 2, (8,), ()): ('L', 'L;IR'), (MM, 0, (1,), 2, (8,), ()): ('L', 'L;IR'), (II, 1, (1,), 1, (8,), ()): ('L', 'L'), (MM, 1, (1,), 1, (8,), ()): ('L', 'L'), (II, 1, (2,), 1, (8,), ()): ('L', 'L'), (MM, 1, (2,), 1, (8,), ()): ('L', 'L'), (II, 1, (1,), 2, (8,), ()): ('L', 'L;R'), (MM, 1, (1,), 2, (8,), ()): ('L', 'L;R'), (II, 1, (1,), 1, (12,), ()): ('I;16', 'I;12'), (II, 0, (1,), 1, (16,), ()): ('I;16', 'I;16'), (II, 1, (1,), 1, (16,), ()): ('I;16', 'I;16'), (MM, 1, (1,), 1, (16,), ()): ('I;16B', 'I;16B'), (II, 1, (1,), 2, (16,), ()): ('I;16', 'I;16R'), (II, 1, (2,), 1, (16,), ()): ('I', 'I;16S'), (MM, 1, (2,), 1, (16,), ()): ('I', 'I;16BS'), (II, 0, (3,), 1, (32,), ()): ('F', 'F;32F'), (MM, 0, (3,), 1, (32,), ()): ('F', 'F;32BF'), (II, 1, (1,), 1, (32,), ()): ('I', 'I;32N'), (II, 1, (2,), 1, (32,), ()): ('I', 'I;32S'), (MM, 1, (2,), 1, (32,), ()): ('I', 'I;32BS'), (II, 1, (3,), 1, (32,), ()): ('F', 'F;32F'), (MM, 1, (3,), 1, (32,), ()): ('F', 'F;32BF'), (II, 1, (1,), 1, (8, 8), (2,)): ('LA', 'LA'), (MM, 1, (1,), 1, (8, 8), (2,)): ('LA', 'LA'), (II, 2, (1,), 1, (8, 8, 8), ()): ('RGB', 'RGB'), (MM, 2, (1,), 1, (8, 8, 8), ()): ('RGB', 'RGB'), (II, 2, (1,), 2, (8, 8, 8), ()): ('RGB', 'RGB;R'), (MM, 2, (1,), 2, (8, 8, 8), ()): ('RGB', 'RGB;R'), (II, 2, (1,), 1, (8, 8, 8, 8), ()): ('RGBA', 'RGBA'), (MM, 2, (1,), 1, (8, 8, 8, 8), ()): ('RGBA', 'RGBA'), (II, 2, (1,), 1, (8, 8, 8, 8), (0,)): ('RGBX', 'RGBX'), (MM, 2, (1,), 1, (8, 8, 8, 8), (0,)): ('RGBX', 'RGBX'), (II, 2, (1,), 1, (8, 8, 8, 8, 8), (0, 0)): ('RGBX', 'RGBXX'), (MM, 2, (1,), 1, (8, 8, 8, 8, 8), (0, 0)): ('RGBX', 'RGBXX'), (II, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0, 0)): ('RGBX', 'RGBXXX'), (MM, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0, 0)): ('RGBX', 'RGBXXX'), (II, 2, (1,), 1, (8, 8, 8, 8), (1,)): ('RGBA', 'RGBa'), (MM, 2, (1,), 1, (8, 8, 8, 8), (1,)): ('RGBA', 'RGBa'), (II, 2, (1,), 1, (8, 8, 8, 8, 8), (1, 0)): ('RGBA', 'RGBaX'), (MM, 2, (1,), 1, (8, 8, 8, 8, 8), (1, 0)): ('RGBA', 'RGBaX'), (II, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (1, 0, 0)): ('RGBA', 'RGBaXX'), (MM, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (1, 0, 0)): ('RGBA', 'RGBaXX'), (II, 2, (1,), 1, (8, 8, 8, 8), (2,)): ('RGBA', 'RGBA'), (MM, 2, (1,), 1, (8, 8, 8, 8), (2,)): ('RGBA', 'RGBA'), (II, 2, (1,), 1, (8, 8, 8, 8, 8), (2, 0)): ('RGBA', 'RGBAX'), (MM, 2, (1,), 1, (8, 8, 8, 8, 8), (2, 0)): ('RGBA', 'RGBAX'), (II, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (2, 0, 0)): ('RGBA', 'RGBAXX'), (MM, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (2, 0, 0)): ('RGBA', 'RGBAXX'), (II, 2, (1,), 1, (8, 8, 8, 8), (999,)): ('RGBA', 'RGBA'), (MM, 2, (1,), 1, (8, 8, 8, 8), (999,)): ('RGBA', 'RGBA'), (II, 2, (1,), 1, (16, 16, 16), ()): ('RGB', 'RGB;16L'), (MM, 2, (1,), 1, (16, 16, 16), ()): ('RGB', 'RGB;16B'), (II, 2, (1,), 1, (16, 16, 16, 16), ()): ('RGBA', 'RGBA;16L'), (MM, 2, (1,), 1, (16, 16, 16, 16), ()): ('RGBA', 'RGBA;16B'), (II, 2, (1,), 1, (16, 16, 16, 16), (0,)): ('RGBX', 'RGBX;16L'), (MM, 2, (1,), 1, (16, 16, 16, 16), (0,)): ('RGBX', 'RGBX;16B'), (II, 2, (1,), 1, (16, 16, 16, 16), (1,)): ('RGBA', 'RGBa;16L'), (MM, 2, (1,), 1, (16, 16, 16, 16), (1,)): ('RGBA', 'RGBa;16B'), (II, 2, (1,), 1, (16, 16, 16, 16), (2,)): ('RGBA', 'RGBA;16L'), (MM, 2, (1,), 1, (16, 16, 16, 16), (2,)): ('RGBA', 'RGBA;16B'), (II, 3, (1,), 1, (1,), ()): ('P', 'P;1'), (MM, 3, (1,), 1, (1,), ()): ('P', 'P;1'), (II, 3, (1,), 2, (1,), ()): ('P', 'P;1R'), (MM, 3, (1,), 2, (1,), ()): ('P', 'P;1R'), (II, 3, (1,), 1, (2,), ()): ('P', 'P;2'), (MM, 3, (1,), 1, (2,), ()): ('P', 'P;2'), (II, 3, (1,), 2, (2,), ()): ('P', 'P;2R'), (MM, 3, (1,), 2, (2,), ()): ('P', 'P;2R'), (II, 3, (1,), 1, (4,), ()): ('P', 'P;4'), (MM, 3, (1,), 1, (4,), ()): ('P', 'P;4'), (II, 3, (1,), 2, (4,), ()): ('P', 'P;4R'), (MM, 3, (1,), 2, (4,), ()): ('P', 'P;4R'), (II, 3, (1,), 1, (8,), ()): ('P', 'P'), (MM, 3, (1,), 1, (8,), ()): ('P', 'P'), (II, 3, (1,), 1, (8, 8), (2,)): ('PA', 'PA'), (MM, 3, (1,), 1, (8, 8), (2,)): ('PA', 'PA'), (II, 3, (1,), 2, (8,), ()): ('P', 'P;R'), (MM, 3, (1,), 2, (8,), ()): ('P', 'P;R'), (II, 5, (1,), 1, (8, 8, 8, 8), ()): ('CMYK', 'CMYK'), (MM, 5, (1,), 1, (8, 8, 8, 8), ()): ('CMYK', 'CMYK'), (II, 5, (1,), 1, (8, 8, 8, 8, 8), (0,)): ('CMYK', 'CMYKX'), (MM, 5, (1,), 1, (8, 8, 8, 8, 8), (0,)): ('CMYK', 'CMYKX'), (II, 5, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0)): ('CMYK', 'CMYKXX'), (MM, 5, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0)): ('CMYK', 'CMYKXX'), (II, 5, (1,), 1, (16, 16, 16, 16), ()): ('CMYK', 'CMYK;16L'), (II, 6, (1,), 1, (8,), ()): ('L', 'L'), (MM, 6, (1,), 1, (8,), ()): ('L', 'L'), (II, 6, (1,), 1, (8, 8, 8), ()): ('RGB', 'RGBX'), (MM, 6, (1,), 1, (8, 8, 8), ()): ('RGB', 'RGBX'), (II, 8, (1,), 1, (8, 8, 8), ()): ('LAB', 'LAB'), (MM, 8, (1,), 1, (8, 8, 8), ()): ('LAB', 'LAB')}
MAX_SAMPLESPERPIXEL = max((len(key_tp[4]) for key_tp in OPEN_INFO))
PREFIXES = [b'MM\x00*', b'II*\x00', b'MM*\x00', b'II\x00*', b'MM\x00+', b'II+\x00']

def _accept(prefix):
    if False:
        while True:
            i = 10
    return prefix[:4] in PREFIXES

def _limit_rational(val, max_val):
    if False:
        while True:
            i = 10
    inv = abs(val) > 1
    n_d = IFDRational(1 / val if inv else val).limit_rational(max_val)
    return n_d[::-1] if inv else n_d

def _limit_signed_rational(val, max_val, min_val):
    if False:
        i = 10
        return i + 15
    frac = Fraction(val)
    n_d = (frac.numerator, frac.denominator)
    if min(n_d) < min_val:
        n_d = _limit_rational(val, abs(min_val))
    if max(n_d) > max_val:
        val = Fraction(*n_d)
        n_d = _limit_rational(val, max_val)
    return n_d
_load_dispatch = {}
_write_dispatch = {}

class IFDRational(Rational):
    """Implements a rational class where 0/0 is a legal value to match
    the in the wild use of exif rationals.

    e.g., DigitalZoomRatio - 0.00/0.00  indicates that no digital zoom was used
    """
    " If the denominator is 0, store this as a float('nan'), otherwise store\n    as a fractions.Fraction(). Delegate as appropriate\n\n    "
    __slots__ = ('_numerator', '_denominator', '_val')

    def __init__(self, value, denominator=1):
        if False:
            print('Hello World!')
        '\n        :param value: either an integer numerator, a\n        float/rational/other number, or an IFDRational\n        :param denominator: Optional integer denominator\n        '
        if isinstance(value, IFDRational):
            self._numerator = value.numerator
            self._denominator = value.denominator
            self._val = value._val
            return
        if isinstance(value, Fraction):
            self._numerator = value.numerator
            self._denominator = value.denominator
        else:
            self._numerator = value
            self._denominator = denominator
        if denominator == 0:
            self._val = float('nan')
        elif denominator == 1:
            self._val = Fraction(value)
        else:
            self._val = Fraction(value, denominator)

    @property
    def numerator(self):
        if False:
            print('Hello World!')
        return self._numerator

    @property
    def denominator(self):
        if False:
            i = 10
            return i + 15
        return self._denominator

    def limit_rational(self, max_denominator):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :param max_denominator: Integer, the maximum denominator value\n        :returns: Tuple of (numerator, denominator)\n        '
        if self.denominator == 0:
            return (self.numerator, self.denominator)
        f = self._val.limit_denominator(max_denominator)
        return (f.numerator, f.denominator)

    def __repr__(self):
        if False:
            return 10
        return str(float(self._val))

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._val.__hash__()

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        val = self._val
        if isinstance(other, IFDRational):
            other = other._val
        if isinstance(other, float):
            val = float(val)
        return val == other

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return [self._val, self._numerator, self._denominator]

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        IFDRational.__init__(self, 0)
        (_val, _numerator, _denominator) = state
        self._val = _val
        self._numerator = _numerator
        self._denominator = _denominator

    def _delegate(op):
        if False:
            return 10

        def delegate(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            return getattr(self._val, op)(*args)
        return delegate
    ' a = [\'add\',\'radd\', \'sub\', \'rsub\', \'mul\', \'rmul\',\n             \'truediv\', \'rtruediv\', \'floordiv\', \'rfloordiv\',\n             \'mod\',\'rmod\', \'pow\',\'rpow\', \'pos\', \'neg\',\n             \'abs\', \'trunc\', \'lt\', \'gt\', \'le\', \'ge\', \'bool\',\n             \'ceil\', \'floor\', \'round\']\n        print("\n".join("__%s__ = _delegate(\'__%s__\')" % (s,s) for s in a))\n        '
    __add__ = _delegate('__add__')
    __radd__ = _delegate('__radd__')
    __sub__ = _delegate('__sub__')
    __rsub__ = _delegate('__rsub__')
    __mul__ = _delegate('__mul__')
    __rmul__ = _delegate('__rmul__')
    __truediv__ = _delegate('__truediv__')
    __rtruediv__ = _delegate('__rtruediv__')
    __floordiv__ = _delegate('__floordiv__')
    __rfloordiv__ = _delegate('__rfloordiv__')
    __mod__ = _delegate('__mod__')
    __rmod__ = _delegate('__rmod__')
    __pow__ = _delegate('__pow__')
    __rpow__ = _delegate('__rpow__')
    __pos__ = _delegate('__pos__')
    __neg__ = _delegate('__neg__')
    __abs__ = _delegate('__abs__')
    __trunc__ = _delegate('__trunc__')
    __lt__ = _delegate('__lt__')
    __gt__ = _delegate('__gt__')
    __le__ = _delegate('__le__')
    __ge__ = _delegate('__ge__')
    __bool__ = _delegate('__bool__')
    __ceil__ = _delegate('__ceil__')
    __floor__ = _delegate('__floor__')
    __round__ = _delegate('__round__')
    if hasattr(Fraction, '__int__'):
        __int__ = _delegate('__int__')

class ImageFileDirectory_v2(MutableMapping):
    """This class represents a TIFF tag directory.  To speed things up, we
    don't decode tags unless they're asked for.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v2()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        'Some Data'

    Individual values are returned as the strings or numbers, sequences are
    returned as tuples of the values.

    The tiff metadata type of each item is stored in a dictionary of
    tag types in
    :attr:`~PIL.TiffImagePlugin.ImageFileDirectory_v2.tagtype`. The types
    are read from a tiff file, guessed from the type added, or added
    manually.

    Data Structures:

        * ``self.tagtype = {}``

          * Key: numerical TIFF tag number
          * Value: integer corresponding to the data type from
            :py:data:`.TiffTags.TYPES`

          .. versionadded:: 3.0.0

    'Internal' data structures:

        * ``self._tags_v2 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data, as tuple for multiple values

        * ``self._tagdata = {}``

          * Key: numerical TIFF tag number
          * Value: undecoded byte string from file

        * ``self._tags_v1 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data in the v1 format

    Tags will be found in the private attributes ``self._tagdata``, and in
    ``self._tags_v2`` once decoded.

    ``self.legacy_api`` is a value for internal use, and shouldn't be changed
    from outside code. In cooperation with
    :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`, if ``legacy_api``
    is true, then decoded tags will be populated into both ``_tags_v1`` and
    ``_tags_v2``. ``_tags_v2`` will be used if this IFD is used in the TIFF
    save routine. Tags should be read from ``_tags_v1`` if
    ``legacy_api == true``.

    """

    def __init__(self, ifh=b'II*\x00\x00\x00\x00\x00', prefix=None, group=None):
        if False:
            i = 10
            return i + 15
        "Initialize an ImageFileDirectory.\n\n        To construct an ImageFileDirectory from a real file, pass the 8-byte\n        magic header to the constructor.  To only set the endianness, pass it\n        as the 'prefix' keyword argument.\n\n        :param ifh: One of the accepted magic headers (cf. PREFIXES); also sets\n              endianness.\n        :param prefix: Override the endianness of the file.\n        "
        if not _accept(ifh):
            msg = f'not a TIFF file (header {repr(ifh)} not valid)'
            raise SyntaxError(msg)
        self._prefix = prefix if prefix is not None else ifh[:2]
        if self._prefix == MM:
            self._endian = '>'
        elif self._prefix == II:
            self._endian = '<'
        else:
            msg = 'not a TIFF IFD'
            raise SyntaxError(msg)
        self._bigtiff = ifh[2] == 43
        self.group = group
        self.tagtype = {}
        ' Dictionary of tag types '
        self.reset()
        (self.next,) = self._unpack('Q', ifh[8:]) if self._bigtiff else self._unpack('L', ifh[4:])
        self._legacy_api = False
    prefix = property(lambda self: self._prefix)
    offset = property(lambda self: self._offset)
    legacy_api = property(lambda self: self._legacy_api)

    @legacy_api.setter
    def legacy_api(self, value):
        if False:
            while True:
                i = 10
        msg = 'Not allowing setting of legacy api'
        raise Exception(msg)

    def reset(self):
        if False:
            while True:
                i = 10
        self._tags_v1 = {}
        self._tags_v2 = {}
        self._tagdata = {}
        self.tagtype = {}
        self._next = None
        self._offset = None

    def __str__(self):
        if False:
            while True:
                i = 10
        return str(dict(self))

    def named(self):
        if False:
            while True:
                i = 10
        '\n        :returns: dict of name|key: value\n\n        Returns the complete tag dictionary, with named tags where possible.\n        '
        return {TiffTags.lookup(code, self.group).name: value for (code, value) in self.items()}

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(set(self._tagdata) | set(self._tags_v2))

    def __getitem__(self, tag):
        if False:
            return 10
        if tag not in self._tags_v2:
            data = self._tagdata[tag]
            typ = self.tagtype[tag]
            (size, handler) = self._load_dispatch[typ]
            self[tag] = handler(self, data, self.legacy_api)
        val = self._tags_v2[tag]
        if self.legacy_api and (not isinstance(val, (tuple, bytes))):
            val = (val,)
        return val

    def __contains__(self, tag):
        if False:
            i = 10
            return i + 15
        return tag in self._tags_v2 or tag in self._tagdata

    def __setitem__(self, tag, value):
        if False:
            i = 10
            return i + 15
        self._setitem(tag, value, self.legacy_api)

    def _setitem(self, tag, value, legacy_api):
        if False:
            for i in range(10):
                print('nop')
        basetypes = (Number, bytes, str)
        info = TiffTags.lookup(tag, self.group)
        values = [value] if isinstance(value, basetypes) else value
        if tag not in self.tagtype:
            if info.type:
                self.tagtype[tag] = info.type
            else:
                self.tagtype[tag] = TiffTags.UNDEFINED
                if all((isinstance(v, IFDRational) for v in values)):
                    self.tagtype[tag] = TiffTags.RATIONAL if all((v >= 0 for v in values)) else TiffTags.SIGNED_RATIONAL
                elif all((isinstance(v, int) for v in values)):
                    if all((0 <= v < 2 ** 16 for v in values)):
                        self.tagtype[tag] = TiffTags.SHORT
                    elif all((-2 ** 15 < v < 2 ** 15 for v in values)):
                        self.tagtype[tag] = TiffTags.SIGNED_SHORT
                    else:
                        self.tagtype[tag] = TiffTags.LONG if all((v >= 0 for v in values)) else TiffTags.SIGNED_LONG
                elif all((isinstance(v, float) for v in values)):
                    self.tagtype[tag] = TiffTags.DOUBLE
                elif all((isinstance(v, str) for v in values)):
                    self.tagtype[tag] = TiffTags.ASCII
                elif all((isinstance(v, bytes) for v in values)):
                    self.tagtype[tag] = TiffTags.BYTE
        if self.tagtype[tag] == TiffTags.UNDEFINED:
            values = [v.encode('ascii', 'replace') if isinstance(v, str) else v for v in values]
        elif self.tagtype[tag] == TiffTags.RATIONAL:
            values = [float(v) if isinstance(v, int) else v for v in values]
        is_ifd = self.tagtype[tag] == TiffTags.LONG and isinstance(values, dict)
        if not is_ifd:
            values = tuple((info.cvt_enum(value) for value in values))
        dest = self._tags_v1 if legacy_api else self._tags_v2
        if not is_ifd and (info.length == 1 or self.tagtype[tag] == TiffTags.BYTE or (info.length is None and len(values) == 1 and (not legacy_api))):
            if legacy_api and self.tagtype[tag] in [TiffTags.RATIONAL, TiffTags.SIGNED_RATIONAL]:
                values = (values,)
            try:
                (dest[tag],) = values
            except ValueError:
                warnings.warn(f'Metadata Warning, tag {tag} had too many entries: {len(values)}, expected 1')
                dest[tag] = values[0]
        else:
            dest[tag] = values

    def __delitem__(self, tag):
        if False:
            i = 10
            return i + 15
        self._tags_v2.pop(tag, None)
        self._tags_v1.pop(tag, None)
        self._tagdata.pop(tag, None)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(set(self._tagdata) | set(self._tags_v2))

    def _unpack(self, fmt, data):
        if False:
            i = 10
            return i + 15
        return struct.unpack(self._endian + fmt, data)

    def _pack(self, fmt, *values):
        if False:
            while True:
                i = 10
        return struct.pack(self._endian + fmt, *values)

    def _register_loader(idx, size):
        if False:
            for i in range(10):
                print('nop')

        def decorator(func):
            if False:
                while True:
                    i = 10
            from .TiffTags import TYPES
            if func.__name__.startswith('load_'):
                TYPES[idx] = func.__name__[5:].replace('_', ' ')
            _load_dispatch[idx] = (size, func)
            return func
        return decorator

    def _register_writer(idx):
        if False:
            i = 10
            return i + 15

        def decorator(func):
            if False:
                while True:
                    i = 10
            _write_dispatch[idx] = func
            return func
        return decorator

    def _register_basic(idx_fmt_name):
        if False:
            while True:
                i = 10
        from .TiffTags import TYPES
        (idx, fmt, name) = idx_fmt_name
        TYPES[idx] = name
        size = struct.calcsize('=' + fmt)
        _load_dispatch[idx] = (size, lambda self, data, legacy_api=True: self._unpack(f'{len(data) // size}{fmt}', data))
        _write_dispatch[idx] = lambda self, *values: b''.join((self._pack(fmt, value) for value in values))
    list(map(_register_basic, [(TiffTags.SHORT, 'H', 'short'), (TiffTags.LONG, 'L', 'long'), (TiffTags.SIGNED_BYTE, 'b', 'signed byte'), (TiffTags.SIGNED_SHORT, 'h', 'signed short'), (TiffTags.SIGNED_LONG, 'l', 'signed long'), (TiffTags.FLOAT, 'f', 'float'), (TiffTags.DOUBLE, 'd', 'double'), (TiffTags.IFD, 'L', 'long'), (TiffTags.LONG8, 'Q', 'long8')]))

    @_register_loader(1, 1)
    def load_byte(self, data, legacy_api=True):
        if False:
            return 10
        return data

    @_register_writer(1)
    def write_byte(self, data):
        if False:
            print('Hello World!')
        if isinstance(data, IFDRational):
            data = int(data)
        if isinstance(data, int):
            data = bytes((data,))
        return data

    @_register_loader(2, 1)
    def load_string(self, data, legacy_api=True):
        if False:
            while True:
                i = 10
        if data.endswith(b'\x00'):
            data = data[:-1]
        return data.decode('latin-1', 'replace')

    @_register_writer(2)
    def write_string(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, bytes):
            value = value.encode('ascii', 'replace')
        return value + b'\x00'

    @_register_loader(5, 8)
    def load_rational(self, data, legacy_api=True):
        if False:
            return 10
        vals = self._unpack(f'{len(data) // 4}L', data)

        def combine(a, b):
            if False:
                return 10
            return (a, b) if legacy_api else IFDRational(a, b)
        return tuple((combine(num, denom) for (num, denom) in zip(vals[::2], vals[1::2])))

    @_register_writer(5)
    def write_rational(self, *values):
        if False:
            i = 10
            return i + 15
        return b''.join((self._pack('2L', *_limit_rational(frac, 2 ** 32 - 1)) for frac in values))

    @_register_loader(7, 1)
    def load_undefined(self, data, legacy_api=True):
        if False:
            print('Hello World!')
        return data

    @_register_writer(7)
    def write_undefined(self, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, int):
            value = str(value).encode('ascii', 'replace')
        return value

    @_register_loader(10, 8)
    def load_signed_rational(self, data, legacy_api=True):
        if False:
            return 10
        vals = self._unpack(f'{len(data) // 4}l', data)

        def combine(a, b):
            if False:
                return 10
            return (a, b) if legacy_api else IFDRational(a, b)
        return tuple((combine(num, denom) for (num, denom) in zip(vals[::2], vals[1::2])))

    @_register_writer(10)
    def write_signed_rational(self, *values):
        if False:
            for i in range(10):
                print('nop')
        return b''.join((self._pack('2l', *_limit_signed_rational(frac, 2 ** 31 - 1, -2 ** 31)) for frac in values))

    def _ensure_read(self, fp, size):
        if False:
            i = 10
            return i + 15
        ret = fp.read(size)
        if len(ret) != size:
            msg = f'Corrupt EXIF data.  Expecting to read {size} bytes but only got {len(ret)}. '
            raise OSError(msg)
        return ret

    def load(self, fp):
        if False:
            print('Hello World!')
        self.reset()
        self._offset = fp.tell()
        try:
            tag_count = (self._unpack('Q', self._ensure_read(fp, 8)) if self._bigtiff else self._unpack('H', self._ensure_read(fp, 2)))[0]
            for i in range(tag_count):
                (tag, typ, count, data) = self._unpack('HHQ8s', self._ensure_read(fp, 20)) if self._bigtiff else self._unpack('HHL4s', self._ensure_read(fp, 12))
                tagname = TiffTags.lookup(tag, self.group).name
                typname = TYPES.get(typ, 'unknown')
                msg = f'tag: {tagname} ({tag}) - type: {typname} ({typ})'
                try:
                    (unit_size, handler) = self._load_dispatch[typ]
                except KeyError:
                    logger.debug('%s - unsupported type %s', msg, typ)
                    continue
                size = count * unit_size
                if size > (8 if self._bigtiff else 4):
                    here = fp.tell()
                    (offset,) = self._unpack('Q' if self._bigtiff else 'L', data)
                    msg += f' Tag Location: {here} - Data Location: {offset}'
                    fp.seek(offset)
                    data = ImageFile._safe_read(fp, size)
                    fp.seek(here)
                else:
                    data = data[:size]
                if len(data) != size:
                    warnings.warn(f'Possibly corrupt EXIF data.  Expecting to read {size} bytes but only got {len(data)}. Skipping tag {tag}')
                    logger.debug(msg)
                    continue
                if not data:
                    logger.debug(msg)
                    continue
                self._tagdata[tag] = data
                self.tagtype[tag] = typ
                msg += ' - value: ' + ('<table: %d bytes>' % size if size > 32 else repr(data))
                logger.debug(msg)
            (self.next,) = self._unpack('Q', self._ensure_read(fp, 8)) if self._bigtiff else self._unpack('L', self._ensure_read(fp, 4))
        except OSError as msg:
            warnings.warn(str(msg))
            return

    def tobytes(self, offset=0):
        if False:
            while True:
                i = 10
        result = self._pack('H', len(self._tags_v2))
        entries = []
        offset = offset + len(result) + len(self._tags_v2) * 12 + 4
        stripoffsets = None
        for (tag, value) in sorted(self._tags_v2.items()):
            if tag == STRIPOFFSETS:
                stripoffsets = len(entries)
            typ = self.tagtype.get(tag)
            logger.debug('Tag %s, Type: %s, Value: %s', tag, typ, repr(value))
            is_ifd = typ == TiffTags.LONG and isinstance(value, dict)
            if is_ifd:
                if self._endian == '<':
                    ifh = b'II*\x00\x08\x00\x00\x00'
                else:
                    ifh = b'MM\x00*\x00\x00\x00\x08'
                ifd = ImageFileDirectory_v2(ifh, group=tag)
                values = self._tags_v2[tag]
                for (ifd_tag, ifd_value) in values.items():
                    ifd[ifd_tag] = ifd_value
                data = ifd.tobytes(offset)
            else:
                values = value if isinstance(value, tuple) else (value,)
                data = self._write_dispatch[typ](self, *values)
            tagname = TiffTags.lookup(tag, self.group).name
            typname = 'ifd' if is_ifd else TYPES.get(typ, 'unknown')
            msg = f'save: {tagname} ({tag}) - type: {typname} ({typ})'
            msg += ' - value: ' + ('<table: %d bytes>' % len(data) if len(data) >= 16 else str(values))
            logger.debug(msg)
            if is_ifd:
                count = 1
            elif typ in [TiffTags.BYTE, TiffTags.ASCII, TiffTags.UNDEFINED]:
                count = len(data)
            else:
                count = len(values)
            if len(data) <= 4:
                entries.append((tag, typ, count, data.ljust(4, b'\x00'), b''))
            else:
                entries.append((tag, typ, count, self._pack('L', offset), data))
                offset += (len(data) + 1) // 2 * 2
        if stripoffsets is not None:
            (tag, typ, count, value, data) = entries[stripoffsets]
            if data:
                msg = 'multistrip support not yet implemented'
                raise NotImplementedError(msg)
            value = self._pack('L', self._unpack('L', value)[0] + offset)
            entries[stripoffsets] = (tag, typ, count, value, data)
        for (tag, typ, count, value, data) in entries:
            logger.debug('%s %s %s %s %s', tag, typ, count, repr(value), repr(data))
            result += self._pack('HHL4s', tag, typ, count, value)
        result += b'\x00\x00\x00\x00'
        for (tag, typ, count, value, data) in entries:
            result += data
            if len(data) & 1:
                result += b'\x00'
        return result

    def save(self, fp):
        if False:
            for i in range(10):
                print('nop')
        if fp.tell() == 0:
            fp.write(self._prefix + self._pack('HL', 42, 8))
        offset = fp.tell()
        result = self.tobytes(offset)
        fp.write(result)
        return offset + len(result)
ImageFileDirectory_v2._load_dispatch = _load_dispatch
ImageFileDirectory_v2._write_dispatch = _write_dispatch
for (idx, name) in TYPES.items():
    name = name.replace(' ', '_')
    setattr(ImageFileDirectory_v2, 'load_' + name, _load_dispatch[idx][1])
    setattr(ImageFileDirectory_v2, 'write_' + name, _write_dispatch[idx])
del _load_dispatch, _write_dispatch, idx, name

class ImageFileDirectory_v1(ImageFileDirectory_v2):
    """This class represents the **legacy** interface to a TIFF tag directory.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v1()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        ('Some Data',)

    Also contains a dictionary of tag types as read from the tiff image file,
    :attr:`~PIL.TiffImagePlugin.ImageFileDirectory_v1.tagtype`.

    Values are returned as a tuple.

    ..  deprecated:: 3.0.0
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._legacy_api = True
    tags = property(lambda self: self._tags_v1)
    tagdata = property(lambda self: self._tagdata)
    tagtype: dict
    'Dictionary of tag types'

    @classmethod
    def from_v2(cls, original):
        if False:
            while True:
                i = 10
        'Returns an\n        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`\n        instance with the same data as is contained in the original\n        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`\n        instance.\n\n        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`\n\n        '
        ifd = cls(prefix=original.prefix)
        ifd._tagdata = original._tagdata
        ifd.tagtype = original.tagtype
        ifd.next = original.next
        return ifd

    def to_v2(self):
        if False:
            i = 10
            return i + 15
        'Returns an\n        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`\n        instance with the same data as is contained in the original\n        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`\n        instance.\n\n        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`\n\n        '
        ifd = ImageFileDirectory_v2(prefix=self.prefix)
        ifd._tagdata = dict(self._tagdata)
        ifd.tagtype = dict(self.tagtype)
        ifd._tags_v2 = dict(self._tags_v2)
        return ifd

    def __contains__(self, tag):
        if False:
            print('Hello World!')
        return tag in self._tags_v1 or tag in self._tagdata

    def __len__(self):
        if False:
            return 10
        return len(set(self._tagdata) | set(self._tags_v1))

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(set(self._tagdata) | set(self._tags_v1))

    def __setitem__(self, tag, value):
        if False:
            i = 10
            return i + 15
        for legacy_api in (False, True):
            self._setitem(tag, value, legacy_api)

    def __getitem__(self, tag):
        if False:
            print('Hello World!')
        if tag not in self._tags_v1:
            data = self._tagdata[tag]
            typ = self.tagtype[tag]
            (size, handler) = self._load_dispatch[typ]
            for legacy in (False, True):
                self._setitem(tag, handler(self, data, legacy), legacy)
        val = self._tags_v1[tag]
        if not isinstance(val, (tuple, bytes)):
            val = (val,)
        return val
ImageFileDirectory = ImageFileDirectory_v1

class TiffImageFile(ImageFile.ImageFile):
    format = 'TIFF'
    format_description = 'Adobe TIFF'
    _close_exclusive_fp_after_loading = False

    def __init__(self, fp=None, filename=None):
        if False:
            print('Hello World!')
        self.tag_v2 = None
        ' Image file directory (tag dictionary) '
        self.tag = None
        ' Legacy tag entries '
        super().__init__(fp, filename)

    def _open(self):
        if False:
            while True:
                i = 10
        'Open the first image in a TIFF file'
        ifh = self.fp.read(8)
        if ifh[2] == 43:
            ifh += self.fp.read(8)
        self.tag_v2 = ImageFileDirectory_v2(ifh)
        self.ifd = None
        self.__first = self.__next = self.tag_v2.next
        self.__frame = -1
        self._fp = self.fp
        self._frame_pos = []
        self._n_frames = None
        logger.debug('*** TiffImageFile._open ***')
        logger.debug('- __first: %s', self.__first)
        logger.debug('- ifh: %s', repr(ifh))
        self._seek(0)

    @property
    def n_frames(self):
        if False:
            return 10
        if self._n_frames is None:
            current = self.tell()
            self._seek(len(self._frame_pos))
            while self._n_frames is None:
                self._seek(self.tell() + 1)
            self.seek(current)
        return self._n_frames

    def seek(self, frame):
        if False:
            print('Hello World!')
        'Select a given frame as current image'
        if not self._seek_check(frame):
            return
        self._seek(frame)
        Image._decompression_bomb_check(self.size)
        self.im = Image.core.new(self.mode, self.size)

    def _seek(self, frame):
        if False:
            return 10
        self.fp = self._fp
        self.fp.tell()
        while len(self._frame_pos) <= frame:
            if not self.__next:
                msg = 'no more images in TIFF file'
                raise EOFError(msg)
            logger.debug('Seeking to frame %s, on frame %s, __next %s, location: %s', frame, self.__frame, self.__next, self.fp.tell())
            self.fp.seek(self.__next)
            self._frame_pos.append(self.__next)
            logger.debug('Loading tags, location: %s', self.fp.tell())
            self.tag_v2.load(self.fp)
            if self.tag_v2.next in self._frame_pos:
                self.__next = 0
            else:
                self.__next = self.tag_v2.next
            if self.__next == 0:
                self._n_frames = frame + 1
            if len(self._frame_pos) == 1:
                self.is_animated = self.__next != 0
            self.__frame += 1
        self.fp.seek(self._frame_pos[frame])
        self.tag_v2.load(self.fp)
        self._reload_exif()
        self.tag = self.ifd = ImageFileDirectory_v1.from_v2(self.tag_v2)
        self.__frame = frame
        self._setup()

    def tell(self):
        if False:
            return 10
        'Return the current frame number'
        return self.__frame

    def getxmp(self):
        if False:
            print('Hello World!')
        '\n        Returns a dictionary containing the XMP tags.\n        Requires defusedxml to be installed.\n\n        :returns: XMP tags in a dictionary.\n        '
        return self._getxmp(self.tag_v2[XMP]) if XMP in self.tag_v2 else {}

    def get_photoshop_blocks(self):
        if False:
            print('Hello World!')
        '\n        Returns a dictionary of Photoshop "Image Resource Blocks".\n        The keys are the image resource ID. For more information, see\n        https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/#50577409_pgfId-1037727\n\n        :returns: Photoshop "Image Resource Blocks" in a dictionary.\n        '
        blocks = {}
        val = self.tag_v2.get(ExifTags.Base.ImageResources)
        if val:
            while val[:4] == b'8BIM':
                id = i16(val[4:6])
                n = math.ceil((val[6] + 1) / 2) * 2
                size = i32(val[6 + n:10 + n])
                data = val[10 + n:10 + n + size]
                blocks[id] = {'data': data}
                val = val[math.ceil((10 + n + size) / 2) * 2:]
        return blocks

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        if self.tile and self.use_load_libtiff:
            return self._load_libtiff()
        return super().load()

    def load_end(self):
        if False:
            print('Hello World!')
        if not self.is_animated:
            self._close_exclusive_fp_after_loading = True
            self.fp.tell()
            exif = self.getexif()
            for key in TiffTags.TAGS_V2_GROUPS:
                if key not in exif:
                    continue
                exif.get_ifd(key)
        ImageOps.exif_transpose(self, in_place=True)
        if ExifTags.Base.Orientation in self.tag_v2:
            del self.tag_v2[ExifTags.Base.Orientation]

    def _load_libtiff(self):
        if False:
            print('Hello World!')
        'Overload method triggered when we detect a compressed tiff\n        Calls out to libtiff'
        Image.Image.load(self)
        self.load_prepare()
        if not len(self.tile) == 1:
            msg = 'Not exactly one tile'
            raise OSError(msg)
        extents = self.tile[0][1]
        args = list(self.tile[0][3])
        try:
            fp = hasattr(self.fp, 'fileno') and self.fp.fileno()
            if hasattr(self.fp, 'flush'):
                self.fp.flush()
        except OSError:
            fp = False
        if fp:
            args[2] = fp
        decoder = Image._getdecoder(self.mode, 'libtiff', tuple(args), self.decoderconfig)
        try:
            decoder.setimage(self.im, extents)
        except ValueError as e:
            msg = "Couldn't set the image"
            raise OSError(msg) from e
        close_self_fp = self._exclusive_fp and (not self.is_animated)
        if hasattr(self.fp, 'getvalue'):
            logger.debug('have getvalue. just sending in a string from getvalue')
            (n, err) = decoder.decode(self.fp.getvalue())
        elif fp:
            logger.debug('have fileno, calling fileno version of the decoder.')
            if not close_self_fp:
                self.fp.seek(0)
            (n, err) = decoder.decode(b'fpfp')
        else:
            logger.debug("don't have fileno or getvalue. just reading")
            self.fp.seek(0)
            (n, err) = decoder.decode(self.fp.read())
        self.tile = []
        self.readonly = 0
        self.load_end()
        if close_self_fp:
            self.fp.close()
            self.fp = None
        if err < 0:
            raise OSError(err)
        return Image.Image.load(self)

    def _setup(self):
        if False:
            print('Hello World!')
        'Setup this image object based on current tags'
        if 48129 in self.tag_v2:
            msg = 'Windows Media Photo files not yet supported'
            raise OSError(msg)
        self._compression = COMPRESSION_INFO[self.tag_v2.get(COMPRESSION, 1)]
        self._planar_configuration = self.tag_v2.get(PLANAR_CONFIGURATION, 1)
        photo = self.tag_v2.get(PHOTOMETRIC_INTERPRETATION, 0)
        if self._compression == 'tiff_jpeg':
            photo = 6
        fillorder = self.tag_v2.get(FILLORDER, 1)
        logger.debug('*** Summary ***')
        logger.debug('- compression: %s', self._compression)
        logger.debug('- photometric_interpretation: %s', photo)
        logger.debug('- planar_configuration: %s', self._planar_configuration)
        logger.debug('- fill_order: %s', fillorder)
        logger.debug('- YCbCr subsampling: %s', self.tag.get(YCBCRSUBSAMPLING))
        xsize = int(self.tag_v2.get(IMAGEWIDTH))
        ysize = int(self.tag_v2.get(IMAGELENGTH))
        self._size = (xsize, ysize)
        logger.debug('- size: %s', self.size)
        sample_format = self.tag_v2.get(SAMPLEFORMAT, (1,))
        if len(sample_format) > 1 and max(sample_format) == min(sample_format) == 1:
            sample_format = (1,)
        bps_tuple = self.tag_v2.get(BITSPERSAMPLE, (1,))
        extra_tuple = self.tag_v2.get(EXTRASAMPLES, ())
        if photo in (2, 6, 8):
            bps_count = 3
        elif photo == 5:
            bps_count = 4
        else:
            bps_count = 1
        bps_count += len(extra_tuple)
        bps_actual_count = len(bps_tuple)
        samples_per_pixel = self.tag_v2.get(SAMPLESPERPIXEL, 3 if self._compression == 'tiff_jpeg' and photo in (2, 6) else 1)
        if samples_per_pixel > MAX_SAMPLESPERPIXEL:
            logger.error('More samples per pixel than can be decoded: %s', samples_per_pixel)
            msg = 'Invalid value for samples per pixel'
            raise SyntaxError(msg)
        if samples_per_pixel < bps_actual_count:
            bps_tuple = bps_tuple[:samples_per_pixel]
        elif samples_per_pixel > bps_actual_count and bps_actual_count == 1:
            bps_tuple = bps_tuple * samples_per_pixel
        if len(bps_tuple) != samples_per_pixel:
            msg = 'unknown data organization'
            raise SyntaxError(msg)
        key = (self.tag_v2.prefix, photo, sample_format, fillorder, bps_tuple, extra_tuple)
        logger.debug('format key: %s', key)
        try:
            (self._mode, rawmode) = OPEN_INFO[key]
        except KeyError as e:
            logger.debug('- unsupported format')
            msg = 'unknown pixel mode'
            raise SyntaxError(msg) from e
        logger.debug('- raw mode: %s', rawmode)
        logger.debug('- pil mode: %s', self.mode)
        self.info['compression'] = self._compression
        xres = self.tag_v2.get(X_RESOLUTION, 1)
        yres = self.tag_v2.get(Y_RESOLUTION, 1)
        if xres and yres:
            resunit = self.tag_v2.get(RESOLUTION_UNIT)
            if resunit == 2:
                self.info['dpi'] = (xres, yres)
            elif resunit == 3:
                self.info['dpi'] = (xres * 2.54, yres * 2.54)
            elif resunit is None:
                self.info['dpi'] = (xres, yres)
                self.info['resolution'] = (xres, yres)
            else:
                self.info['resolution'] = (xres, yres)
        x = y = layer = 0
        self.tile = []
        self.use_load_libtiff = READ_LIBTIFF or self._compression != 'raw'
        if self.use_load_libtiff:
            if fillorder == 2:
                key = key[:3] + (1,) + key[4:]
                logger.debug('format key: %s', key)
                (self._mode, rawmode) = OPEN_INFO[key]
            if rawmode == 'I;16':
                rawmode = 'I;16N'
            if ';16B' in rawmode:
                rawmode = rawmode.replace(';16B', ';16N')
            if ';16L' in rawmode:
                rawmode = rawmode.replace(';16L', ';16N')
            if photo == 6 and self._compression == 'jpeg' and (self._planar_configuration == 1):
                rawmode = 'RGB'
            a = (rawmode, self._compression, False, self.tag_v2.offset)
            self.tile.append(('libtiff', (0, 0, xsize, ysize), 0, a))
        elif STRIPOFFSETS in self.tag_v2 or TILEOFFSETS in self.tag_v2:
            if STRIPOFFSETS in self.tag_v2:
                offsets = self.tag_v2[STRIPOFFSETS]
                h = self.tag_v2.get(ROWSPERSTRIP, ysize)
                w = self.size[0]
            else:
                offsets = self.tag_v2[TILEOFFSETS]
                w = self.tag_v2.get(TILEWIDTH)
                h = self.tag_v2.get(TILELENGTH)
            for offset in offsets:
                if x + w > xsize:
                    stride = w * sum(bps_tuple) / 8
                else:
                    stride = 0
                tile_rawmode = rawmode
                if self._planar_configuration == 2:
                    tile_rawmode = rawmode[layer]
                    stride /= bps_count
                a = (tile_rawmode, int(stride), 1)
                self.tile.append((self._compression, (x, y, min(x + w, xsize), min(y + h, ysize)), offset, a))
                x = x + w
                if x >= self.size[0]:
                    (x, y) = (0, y + h)
                    if y >= self.size[1]:
                        x = y = 0
                        layer += 1
        else:
            logger.debug('- unsupported data organization')
            msg = 'unknown data organization'
            raise SyntaxError(msg)
        if ICCPROFILE in self.tag_v2:
            self.info['icc_profile'] = self.tag_v2[ICCPROFILE]
        if self.mode in ['P', 'PA']:
            palette = [o8(b // 256) for b in self.tag_v2[COLORMAP]]
            self.palette = ImagePalette.raw('RGB;L', b''.join(palette))
SAVE_INFO = {'1': ('1', II, 1, 1, (1,), None), 'L': ('L', II, 1, 1, (8,), None), 'LA': ('LA', II, 1, 1, (8, 8), 2), 'P': ('P', II, 3, 1, (8,), None), 'PA': ('PA', II, 3, 1, (8, 8), 2), 'I': ('I;32S', II, 1, 2, (32,), None), 'I;16': ('I;16', II, 1, 1, (16,), None), 'I;16S': ('I;16S', II, 1, 2, (16,), None), 'F': ('F;32F', II, 1, 3, (32,), None), 'RGB': ('RGB', II, 2, 1, (8, 8, 8), None), 'RGBX': ('RGBX', II, 2, 1, (8, 8, 8, 8), 0), 'RGBA': ('RGBA', II, 2, 1, (8, 8, 8, 8), 2), 'CMYK': ('CMYK', II, 5, 1, (8, 8, 8, 8), None), 'YCbCr': ('YCbCr', II, 6, 1, (8, 8, 8), None), 'LAB': ('LAB', II, 8, 1, (8, 8, 8), None), 'I;32BS': ('I;32BS', MM, 1, 2, (32,), None), 'I;16B': ('I;16B', MM, 1, 1, (16,), None), 'I;16BS': ('I;16BS', MM, 1, 2, (16,), None), 'F;32BF': ('F;32BF', MM, 1, 3, (32,), None)}

def _save(im, fp, filename):
    if False:
        for i in range(10):
            print('nop')
    try:
        (rawmode, prefix, photo, format, bits, extra) = SAVE_INFO[im.mode]
    except KeyError as e:
        msg = f'cannot write mode {im.mode} as TIFF'
        raise OSError(msg) from e
    ifd = ImageFileDirectory_v2(prefix=prefix)
    encoderinfo = im.encoderinfo
    encoderconfig = im.encoderconfig
    try:
        compression = encoderinfo['compression']
    except KeyError:
        compression = im.info.get('compression')
        if isinstance(compression, int):
            compression = None
    if compression is None:
        compression = 'raw'
    elif compression == 'tiff_jpeg':
        compression = 'jpeg'
    elif compression == 'tiff_deflate':
        compression = 'tiff_adobe_deflate'
    libtiff = WRITE_LIBTIFF or compression != 'raw'
    ifd[PLANAR_CONFIGURATION] = 1
    ifd[IMAGEWIDTH] = im.size[0]
    ifd[IMAGELENGTH] = im.size[1]
    if 'tiffinfo' in encoderinfo:
        info = encoderinfo['tiffinfo']
    elif 'exif' in encoderinfo:
        info = encoderinfo['exif']
        if isinstance(info, bytes):
            exif = Image.Exif()
            exif.load(info)
            info = exif
    else:
        info = {}
    logger.debug('Tiffinfo Keys: %s', list(info))
    if isinstance(info, ImageFileDirectory_v1):
        info = info.to_v2()
    for key in info:
        if isinstance(info, Image.Exif) and key in TiffTags.TAGS_V2_GROUPS:
            ifd[key] = info.get_ifd(key)
        else:
            ifd[key] = info.get(key)
        try:
            ifd.tagtype[key] = info.tagtype[key]
        except Exception:
            pass
    if hasattr(im, 'tag_v2'):
        for key in (RESOLUTION_UNIT, X_RESOLUTION, Y_RESOLUTION, IPTC_NAA_CHUNK, PHOTOSHOP_CHUNK, XMP):
            if key in im.tag_v2:
                ifd[key] = im.tag_v2[key]
                ifd.tagtype[key] = im.tag_v2.tagtype[key]
    icc = encoderinfo.get('icc_profile', im.info.get('icc_profile'))
    if icc:
        ifd[ICCPROFILE] = icc
    for (key, name) in [(IMAGEDESCRIPTION, 'description'), (X_RESOLUTION, 'resolution'), (Y_RESOLUTION, 'resolution'), (X_RESOLUTION, 'x_resolution'), (Y_RESOLUTION, 'y_resolution'), (RESOLUTION_UNIT, 'resolution_unit'), (SOFTWARE, 'software'), (DATE_TIME, 'date_time'), (ARTIST, 'artist'), (COPYRIGHT, 'copyright')]:
        if name in encoderinfo:
            ifd[key] = encoderinfo[name]
    dpi = encoderinfo.get('dpi')
    if dpi:
        ifd[RESOLUTION_UNIT] = 2
        ifd[X_RESOLUTION] = dpi[0]
        ifd[Y_RESOLUTION] = dpi[1]
    if bits != (1,):
        ifd[BITSPERSAMPLE] = bits
        if len(bits) != 1:
            ifd[SAMPLESPERPIXEL] = len(bits)
    if extra is not None:
        ifd[EXTRASAMPLES] = extra
    if format != 1:
        ifd[SAMPLEFORMAT] = format
    if PHOTOMETRIC_INTERPRETATION not in ifd:
        ifd[PHOTOMETRIC_INTERPRETATION] = photo
    elif im.mode in ('1', 'L') and ifd[PHOTOMETRIC_INTERPRETATION] == 0:
        if im.mode == '1':
            inverted_im = im.copy()
            px = inverted_im.load()
            for y in range(inverted_im.height):
                for x in range(inverted_im.width):
                    px[x, y] = 0 if px[x, y] == 255 else 255
            im = inverted_im
        else:
            im = ImageOps.invert(im)
    if im.mode in ['P', 'PA']:
        lut = im.im.getpalette('RGB', 'RGB;L')
        colormap = []
        colors = len(lut) // 3
        for i in range(3):
            colormap += [v * 256 for v in lut[colors * i:colors * (i + 1)]]
            colormap += [0] * (256 - colors)
        ifd[COLORMAP] = colormap
    stride = len(bits) * ((im.size[0] * bits[0] + 7) // 8)
    if libtiff:
        im_strip_size = encoderinfo.get('strip_size', STRIP_SIZE)
        rows_per_strip = 1 if stride == 0 else min(im_strip_size // stride, im.size[1])
        if compression == 'jpeg':
            rows_per_strip = min((rows_per_strip + 7) // 8 * 8, im.size[1])
    else:
        rows_per_strip = im.size[1]
    if rows_per_strip == 0:
        rows_per_strip = 1
    strip_byte_counts = 1 if stride == 0 else stride * rows_per_strip
    strips_per_image = (im.size[1] + rows_per_strip - 1) // rows_per_strip
    ifd[ROWSPERSTRIP] = rows_per_strip
    if strip_byte_counts >= 2 ** 16:
        ifd.tagtype[STRIPBYTECOUNTS] = TiffTags.LONG
    ifd[STRIPBYTECOUNTS] = (strip_byte_counts,) * (strips_per_image - 1) + (stride * im.size[1] - strip_byte_counts * (strips_per_image - 1),)
    ifd[STRIPOFFSETS] = tuple(range(0, strip_byte_counts * strips_per_image, strip_byte_counts))
    ifd[COMPRESSION] = COMPRESSION_INFO_REV.get(compression, 1)
    if im.mode == 'YCbCr':
        for (tag, value) in {YCBCRSUBSAMPLING: (1, 1), REFERENCEBLACKWHITE: (0, 255, 128, 255, 128, 255)}.items():
            ifd.setdefault(tag, value)
    blocklist = [TILEWIDTH, TILELENGTH, TILEOFFSETS, TILEBYTECOUNTS]
    if libtiff:
        if 'quality' in encoderinfo:
            quality = encoderinfo['quality']
            if not isinstance(quality, int) or quality < 0 or quality > 100:
                msg = 'Invalid quality setting'
                raise ValueError(msg)
            if compression != 'jpeg':
                msg = "quality setting only supported for 'jpeg' compression"
                raise ValueError(msg)
            ifd[JPEGQUALITY] = quality
        logger.debug('Saving using libtiff encoder')
        logger.debug('Items: %s', sorted(ifd.items()))
        _fp = 0
        if hasattr(fp, 'fileno'):
            try:
                fp.seek(0)
                _fp = os.dup(fp.fileno())
            except io.UnsupportedOperation:
                pass
        types = {}
        blocklist += [REFERENCEBLACKWHITE, STRIPBYTECOUNTS, STRIPOFFSETS, TRANSFERFUNCTION, SUBIFD]
        atts = {BITSPERSAMPLE: bits[0]}
        legacy_ifd = {}
        if hasattr(im, 'tag'):
            legacy_ifd = im.tag.to_v2()
        supplied_tags = {**getattr(im, 'tag_v2', {}), **legacy_ifd}
        if SAMPLEFORMAT in supplied_tags:
            del supplied_tags[SAMPLEFORMAT]
        for (tag, value) in itertools.chain(ifd.items(), supplied_tags.items()):
            if tag not in TiffTags.LIBTIFF_CORE:
                if not getattr(Image.core, 'libtiff_support_custom_tags', False):
                    continue
                if tag in ifd.tagtype:
                    types[tag] = ifd.tagtype[tag]
                elif not isinstance(value, (int, float, str, bytes)):
                    continue
                else:
                    type = TiffTags.lookup(tag).type
                    if type:
                        types[tag] = type
            if tag not in atts and tag not in blocklist:
                if isinstance(value, str):
                    atts[tag] = value.encode('ascii', 'replace') + b'\x00'
                elif isinstance(value, IFDRational):
                    atts[tag] = float(value)
                else:
                    atts[tag] = value
        if SAMPLEFORMAT in atts and len(atts[SAMPLEFORMAT]) == 1:
            atts[SAMPLEFORMAT] = atts[SAMPLEFORMAT][0]
        logger.debug('Converted items: %s', sorted(atts.items()))
        if im.mode in ('I;16B', 'I;16'):
            rawmode = 'I;16N'
        tags = list(atts.items())
        tags.sort()
        a = (rawmode, compression, _fp, filename, tags, types)
        e = Image._getencoder(im.mode, 'libtiff', a, encoderconfig)
        e.setimage(im.im, (0, 0) + im.size)
        while True:
            (errcode, data) = e.encode(16 * 1024)[1:]
            if not _fp:
                fp.write(data)
            if errcode:
                break
        if _fp:
            try:
                os.close(_fp)
            except OSError:
                pass
        if errcode < 0:
            msg = f'encoder error {errcode} when writing image file'
            raise OSError(msg)
    else:
        for tag in blocklist:
            del ifd[tag]
        offset = ifd.save(fp)
        ImageFile._save(im, fp, [('raw', (0, 0) + im.size, offset, (rawmode, stride, 1))])
    if '_debug_multipage' in encoderinfo:
        im._debug_multipage = ifd

class AppendingTiffWriter:
    fieldSizes = [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 2, 4, 8]
    Tags = {273, 288, 324, 519, 520, 521}

    def __init__(self, fn, new=False):
        if False:
            return 10
        if hasattr(fn, 'read'):
            self.f = fn
            self.close_fp = False
        else:
            self.name = fn
            self.close_fp = True
            try:
                self.f = open(fn, 'w+b' if new else 'r+b')
            except OSError:
                self.f = open(fn, 'w+b')
        self.beginning = self.f.tell()
        self.setup()

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.f.seek(self.beginning, os.SEEK_SET)
        self.whereToWriteNewIFDOffset = None
        self.offsetOfNewPage = 0
        self.IIMM = iimm = self.f.read(4)
        if not iimm:
            self.isFirst = True
            return
        self.isFirst = False
        if iimm == b'II*\x00':
            self.setEndian('<')
        elif iimm == b'MM\x00*':
            self.setEndian('>')
        else:
            msg = 'Invalid TIFF file header'
            raise RuntimeError(msg)
        self.skipIFDs()
        self.goToEnd()

    def finalize(self):
        if False:
            while True:
                i = 10
        if self.isFirst:
            return
        self.f.seek(self.offsetOfNewPage)
        iimm = self.f.read(4)
        if not iimm:
            return
        if iimm != self.IIMM:
            msg = "IIMM of new page doesn't match IIMM of first page"
            raise RuntimeError(msg)
        ifd_offset = self.readLong()
        ifd_offset += self.offsetOfNewPage
        self.f.seek(self.whereToWriteNewIFDOffset)
        self.writeLong(ifd_offset)
        self.f.seek(ifd_offset)
        self.fixIFD()

    def newFrame(self):
        if False:
            i = 10
            return i + 15
        self.finalize()
        self.setup()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if self.close_fp:
            self.close()
        return False

    def tell(self):
        if False:
            i = 10
            return i + 15
        return self.f.tell() - self.offsetOfNewPage

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            print('Hello World!')
        if whence == os.SEEK_SET:
            offset += self.offsetOfNewPage
        self.f.seek(offset, whence)
        return self.tell()

    def goToEnd(self):
        if False:
            return 10
        self.f.seek(0, os.SEEK_END)
        pos = self.f.tell()
        pad_bytes = 16 - pos % 16
        if 0 < pad_bytes < 16:
            self.f.write(bytes(pad_bytes))
        self.offsetOfNewPage = self.f.tell()

    def setEndian(self, endian):
        if False:
            i = 10
            return i + 15
        self.endian = endian
        self.longFmt = self.endian + 'L'
        self.shortFmt = self.endian + 'H'
        self.tagFormat = self.endian + 'HHL'

    def skipIFDs(self):
        if False:
            return 10
        while True:
            ifd_offset = self.readLong()
            if ifd_offset == 0:
                self.whereToWriteNewIFDOffset = self.f.tell() - 4
                break
            self.f.seek(ifd_offset)
            num_tags = self.readShort()
            self.f.seek(num_tags * 12, os.SEEK_CUR)

    def write(self, data):
        if False:
            i = 10
            return i + 15
        return self.f.write(data)

    def readShort(self):
        if False:
            print('Hello World!')
        (value,) = struct.unpack(self.shortFmt, self.f.read(2))
        return value

    def readLong(self):
        if False:
            return 10
        (value,) = struct.unpack(self.longFmt, self.f.read(4))
        return value

    def rewriteLastShortToLong(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.f.seek(-2, os.SEEK_CUR)
        bytes_written = self.f.write(struct.pack(self.longFmt, value))
        if bytes_written is not None and bytes_written != 4:
            msg = f'wrote only {bytes_written} bytes but wanted 4'
            raise RuntimeError(msg)

    def rewriteLastShort(self, value):
        if False:
            i = 10
            return i + 15
        self.f.seek(-2, os.SEEK_CUR)
        bytes_written = self.f.write(struct.pack(self.shortFmt, value))
        if bytes_written is not None and bytes_written != 2:
            msg = f'wrote only {bytes_written} bytes but wanted 2'
            raise RuntimeError(msg)

    def rewriteLastLong(self, value):
        if False:
            return 10
        self.f.seek(-4, os.SEEK_CUR)
        bytes_written = self.f.write(struct.pack(self.longFmt, value))
        if bytes_written is not None and bytes_written != 4:
            msg = f'wrote only {bytes_written} bytes but wanted 4'
            raise RuntimeError(msg)

    def writeShort(self, value):
        if False:
            while True:
                i = 10
        bytes_written = self.f.write(struct.pack(self.shortFmt, value))
        if bytes_written is not None and bytes_written != 2:
            msg = f'wrote only {bytes_written} bytes but wanted 2'
            raise RuntimeError(msg)

    def writeLong(self, value):
        if False:
            i = 10
            return i + 15
        bytes_written = self.f.write(struct.pack(self.longFmt, value))
        if bytes_written is not None and bytes_written != 4:
            msg = f'wrote only {bytes_written} bytes but wanted 4'
            raise RuntimeError(msg)

    def close(self):
        if False:
            return 10
        self.finalize()
        self.f.close()

    def fixIFD(self):
        if False:
            return 10
        num_tags = self.readShort()
        for i in range(num_tags):
            (tag, field_type, count) = struct.unpack(self.tagFormat, self.f.read(8))
            field_size = self.fieldSizes[field_type]
            total_size = field_size * count
            is_local = total_size <= 4
            if not is_local:
                offset = self.readLong()
                offset += self.offsetOfNewPage
                self.rewriteLastLong(offset)
            if tag in self.Tags:
                cur_pos = self.f.tell()
                if is_local:
                    self.fixOffsets(count, isShort=field_size == 2, isLong=field_size == 4)
                    self.f.seek(cur_pos + 4)
                else:
                    self.f.seek(offset)
                    self.fixOffsets(count, isShort=field_size == 2, isLong=field_size == 4)
                    self.f.seek(cur_pos)
                offset = cur_pos = None
            elif is_local:
                self.f.seek(4, os.SEEK_CUR)

    def fixOffsets(self, count, isShort=False, isLong=False):
        if False:
            return 10
        if not isShort and (not isLong):
            msg = 'offset is neither short nor long'
            raise RuntimeError(msg)
        for i in range(count):
            offset = self.readShort() if isShort else self.readLong()
            offset += self.offsetOfNewPage
            if isShort and offset >= 65536:
                if count != 1:
                    msg = 'not implemented'
                    raise RuntimeError(msg)
                self.rewriteLastShortToLong(offset)
                self.f.seek(-10, os.SEEK_CUR)
                self.writeShort(TiffTags.LONG)
                self.f.seek(8, os.SEEK_CUR)
            elif isShort:
                self.rewriteLastShort(offset)
            else:
                self.rewriteLastLong(offset)

def _save_all(im, fp, filename):
    if False:
        i = 10
        return i + 15
    encoderinfo = im.encoderinfo.copy()
    encoderconfig = im.encoderconfig
    append_images = list(encoderinfo.get('append_images', []))
    if not hasattr(im, 'n_frames') and (not append_images):
        return _save(im, fp, filename)
    cur_idx = im.tell()
    try:
        with AppendingTiffWriter(fp) as tf:
            for ims in [im] + append_images:
                ims.encoderinfo = encoderinfo
                ims.encoderconfig = encoderconfig
                if not hasattr(ims, 'n_frames'):
                    nfr = 1
                else:
                    nfr = ims.n_frames
                for idx in range(nfr):
                    ims.seek(idx)
                    ims.load()
                    _save(ims, tf, filename)
                    tf.newFrame()
    finally:
        im.seek(cur_idx)
Image.register_open(TiffImageFile.format, TiffImageFile, _accept)
Image.register_save(TiffImageFile.format, _save)
Image.register_save_all(TiffImageFile.format, _save_all)
Image.register_extensions(TiffImageFile.format, ['.tif', '.tiff'])
Image.register_mime(TiffImageFile.format, 'image/tiff')