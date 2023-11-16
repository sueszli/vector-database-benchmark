import ctypes
import math
import time
from contextlib import suppress
import numpy as np
from astropy.io.fits import conf
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.hdu.base import BITPIX2DTYPE, DELAYED, DTYPE2BITPIX, ExtensionHDU
from astropy.io.fits.hdu.compressed._tiled_compression import compress_image_data
from astropy.io.fits.hdu.image import ImageHDU
from astropy.io.fits.hdu.table import BinTableHDU
from astropy.io.fits.util import _get_array_mmap, _is_int, _is_pseudo_integer, _pseudo_zero
from astropy.utils import lazyproperty
from astropy.utils.decorators import deprecated_renamed_argument
from .header import CompImageHeader, _bintable_header_to_image_header, _image_header_to_bintable_header_and_coldefs
from .section import CompImageSection
from .settings import CMTYPE_ALIASES, DEFAULT_COMPRESSION_TYPE, DEFAULT_DITHER_SEED, DEFAULT_HCOMP_SCALE, DEFAULT_HCOMP_SMOOTH, DEFAULT_QUANTIZE_LEVEL, DEFAULT_QUANTIZE_METHOD, DITHER_SEED_CHECKSUM, DITHER_SEED_CLOCK
COMPRESSION_ENABLED = True

class CompImageHDU(BinTableHDU):
    """
    Compressed Image HDU class.
    """
    _manages_own_heap = True
    "\n    The calls to CFITSIO lay out the heap data in memory, and we write it out\n    the same way CFITSIO organizes it.  In principle this would break if a user\n    manually changes the underlying compressed data by hand, but there is no\n    reason they would want to do that (and if they do that's their\n    responsibility).\n    "
    _load_variable_length_data = False
    "\n    We don't want to always load all the tiles so by setting this option\n    we can then access the tiles as needed.\n    "
    _default_name = 'COMPRESSED_IMAGE'

    @deprecated_renamed_argument('tile_size', None, since='5.3', message='The tile_size argument has been deprecated. Use tile_shape instead, but note that this should be given in the reverse order to tile_size (tile_shape should be in Numpy C order).')
    def __init__(self, data=None, header=None, name=None, compression_type=DEFAULT_COMPRESSION_TYPE, tile_shape=None, hcomp_scale=DEFAULT_HCOMP_SCALE, hcomp_smooth=DEFAULT_HCOMP_SMOOTH, quantize_level=DEFAULT_QUANTIZE_LEVEL, quantize_method=DEFAULT_QUANTIZE_METHOD, dither_seed=DEFAULT_DITHER_SEED, do_not_scale_image_data=False, uint=False, scale_back=False, tile_size=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        data : array, optional\n            Uncompressed image data\n\n        header : `~astropy.io.fits.Header`, optional\n            Header to be associated with the image; when reading the HDU from a\n            file (data=DELAYED), the header read from the file\n\n        name : str, optional\n            The ``EXTNAME`` value; if this value is `None`, then the name from\n            the input image header will be used; if there is no name in the\n            input image header then the default name ``COMPRESSED_IMAGE`` is\n            used.\n\n        compression_type : str, optional\n            Compression algorithm: one of\n            ``\'RICE_1\'``, ``\'RICE_ONE\'``, ``\'PLIO_1\'``, ``\'GZIP_1\'``,\n            ``\'GZIP_2\'``, ``\'HCOMPRESS_1\'``, ``\'NOCOMPRESS\'``\n\n        tile_shape : tuple, optional\n            Compression tile shape, which should be specified using the default\n            Numpy convention for array shapes (C order). The default is to\n            treat each row of image as a tile.\n\n        hcomp_scale : float, optional\n            HCOMPRESS scale parameter\n\n        hcomp_smooth : float, optional\n            HCOMPRESS smooth parameter\n\n        quantize_level : float, optional\n            Floating point quantization level; see note below\n\n        quantize_method : int, optional\n            Floating point quantization dithering method; can be either\n            ``NO_DITHER`` (-1; default), ``SUBTRACTIVE_DITHER_1`` (1), or\n            ``SUBTRACTIVE_DITHER_2`` (2); see note below\n\n        dither_seed : int, optional\n            Random seed to use for dithering; can be either an integer in the\n            range 1 to 1000 (inclusive), ``DITHER_SEED_CLOCK`` (0; default), or\n            ``DITHER_SEED_CHECKSUM`` (-1); see note below\n\n        Notes\n        -----\n        The astropy.io.fits package supports 2 methods of image compression:\n\n            1) The entire FITS file may be externally compressed with the gzip\n               or pkzip utility programs, producing a ``*.gz`` or ``*.zip``\n               file, respectively.  When reading compressed files of this type,\n               Astropy first uncompresses the entire file into a temporary file\n               before performing the requested read operations.  The\n               astropy.io.fits package does not support writing to these types\n               of compressed files.  This type of compression is supported in\n               the ``_File`` class, not in the `CompImageHDU` class.  The file\n               compression type is recognized by the ``.gz`` or ``.zip`` file\n               name extension.\n\n            2) The `CompImageHDU` class supports the FITS tiled image\n               compression convention in which the image is subdivided into a\n               grid of rectangular tiles, and each tile of pixels is\n               individually compressed.  The details of this FITS compression\n               convention are described at the `FITS Support Office web site\n               <https://fits.gsfc.nasa.gov/registry/tilecompression.html>`_.\n               Basically, the compressed image tiles are stored in rows of a\n               variable length array column in a FITS binary table.  The\n               astropy.io.fits recognizes that this binary table extension\n               contains an image and treats it as if it were an image\n               extension.  Under this tile-compression format, FITS header\n               keywords remain uncompressed.  At this time, Astropy does not\n               support the ability to extract and uncompress sections of the\n               image without having to uncompress the entire image.\n\n        The astropy.io.fits package supports 3 general-purpose compression\n        algorithms plus one other special-purpose compression technique that is\n        designed for data masks with positive integer pixel values.  The 3\n        general purpose algorithms are GZIP, Rice, and HCOMPRESS, and the\n        special-purpose technique is the IRAF pixel list compression technique\n        (PLIO).  The ``compression_type`` parameter defines the compression\n        algorithm to be used.\n\n        The FITS image can be subdivided into any desired rectangular grid of\n        compression tiles.  With the GZIP, Rice, and PLIO algorithms, the\n        default is to take each row of the image as a tile.  The HCOMPRESS\n        algorithm is inherently 2-dimensional in nature, so the default in this\n        case is to take 16 rows of the image per tile.  In most cases, it makes\n        little difference what tiling pattern is used, so the default tiles are\n        usually adequate.  In the case of very small images, it could be more\n        efficient to compress the whole image as a single tile.  Note that the\n        image dimensions are not required to be an integer multiple of the tile\n        dimensions; if not, then the tiles at the edges of the image will be\n        smaller than the other tiles.  The ``tile_shape`` parameter may be\n        provided as a list of tile sizes, one for each dimension in the image.\n        For example a ``tile_shape`` value of ``(100,100)`` would divide a 300 X\n        300 image into 9 100 X 100 tiles.\n\n        The 4 supported image compression algorithms are all \'lossless\' when\n        applied to integer FITS images; the pixel values are preserved exactly\n        with no loss of information during the compression and uncompression\n        process.  In addition, the HCOMPRESS algorithm supports a \'lossy\'\n        compression mode that will produce larger amount of image compression.\n        This is achieved by specifying a non-zero value for the ``hcomp_scale``\n        parameter.  Since the amount of compression that is achieved depends\n        directly on the RMS noise in the image, it is usually more convenient\n        to specify the ``hcomp_scale`` factor relative to the RMS noise.\n        Setting ``hcomp_scale = 2.5`` means use a scale factor that is 2.5\n        times the calculated RMS noise in the image tile.  In some cases it may\n        be desirable to specify the exact scaling to be used, instead of\n        specifying it relative to the calculated noise value.  This may be done\n        by specifying the negative of the desired scale value (typically in the\n        range -2 to -100).\n\n        Very high compression factors (of 100 or more) can be achieved by using\n        large ``hcomp_scale`` values, however, this can produce undesirable\n        \'blocky\' artifacts in the compressed image.  A variation of the\n        HCOMPRESS algorithm (called HSCOMPRESS) can be used in this case to\n        apply a small amount of smoothing of the image when it is uncompressed\n        to help cover up these artifacts.  This smoothing is purely cosmetic\n        and does not cause any significant change to the image pixel values.\n        Setting the ``hcomp_smooth`` parameter to 1 will engage the smoothing\n        algorithm.\n\n        Floating point FITS images (which have ``BITPIX`` = -32 or -64) usually\n        contain too much \'noise\' in the least significant bits of the mantissa\n        of the pixel values to be effectively compressed with any lossless\n        algorithm.  Consequently, floating point images are first quantized\n        into scaled integer pixel values (and thus throwing away much of the\n        noise) before being compressed with the specified algorithm (either\n        GZIP, RICE, or HCOMPRESS).  This technique produces much higher\n        compression factors than simply using the GZIP utility to externally\n        compress the whole FITS file, but it also means that the original\n        floating point value pixel values are not exactly preserved.  When done\n        properly, this integer scaling technique will only discard the\n        insignificant noise while still preserving all the real information in\n        the image.  The amount of precision that is retained in the pixel\n        values is controlled by the ``quantize_level`` parameter.  Larger\n        values will result in compressed images whose pixels more closely match\n        the floating point pixel values, but at the same time the amount of\n        compression that is achieved will be reduced.  Users should experiment\n        with different values for this parameter to determine the optimal value\n        that preserves all the useful information in the image, without\n        needlessly preserving all the \'noise\' which will hurt the compression\n        efficiency.\n\n        The default value for the ``quantize_level`` scale factor is 16, which\n        means that scaled integer pixel values will be quantized such that the\n        difference between adjacent integer values will be 1/16th of the noise\n        level in the image background.  An optimized algorithm is used to\n        accurately estimate the noise in the image.  As an example, if the RMS\n        noise in the background pixels of an image = 32.0, then the spacing\n        between adjacent scaled integer pixel values will equal 2.0 by default.\n        Note that the RMS noise is independently calculated for each tile of\n        the image, so the resulting integer scaling factor may fluctuate\n        slightly for each tile.  In some cases, it may be desirable to specify\n        the exact quantization level to be used, instead of specifying it\n        relative to the calculated noise value.  This may be done by specifying\n        the negative of desired quantization level for the value of\n        ``quantize_level``.  In the previous example, one could specify\n        ``quantize_level = -2.0`` so that the quantized integer levels differ\n        by 2.0.  Larger negative values for ``quantize_level`` means that the\n        levels are more coarsely-spaced, and will produce higher compression\n        factors.\n\n        The quantization algorithm can also apply one of two random dithering\n        methods in order to reduce bias in the measured intensity of background\n        regions.  The default method, specified with the constant\n        ``SUBTRACTIVE_DITHER_1`` adds dithering to the zero-point of the\n        quantization array itself rather than adding noise to the actual image.\n        The random noise is added on a pixel-by-pixel basis, so in order\n        restore each pixel from its integer value to its floating point value\n        it is necessary to replay the same sequence of random numbers for each\n        pixel (see below).  The other method, ``SUBTRACTIVE_DITHER_2``, is\n        exactly like the first except that before dithering any pixel with a\n        floating point value of ``0.0`` is replaced with the special integer\n        value ``-2147483647``.  When the image is uncompressed, pixels with\n        this value are restored back to ``0.0`` exactly.  Finally, a value of\n        ``NO_DITHER`` disables dithering entirely.\n\n        As mentioned above, when using the subtractive dithering algorithm it\n        is necessary to be able to generate a (pseudo-)random sequence of noise\n        for each pixel, and replay that same sequence upon decompressing.  To\n        facilitate this, a random seed between 1 and 10000 (inclusive) is used\n        to seed a random number generator, and that seed is stored in the\n        ``ZDITHER0`` keyword in the header of the compressed HDU.  In order to\n        use that seed to generate the same sequence of random numbers the same\n        random number generator must be used at compression and decompression\n        time; for that reason the tiled image convention provides an\n        implementation of a very simple pseudo-random number generator.  The\n        seed itself can be provided in one of three ways, controllable by the\n        ``dither_seed`` argument:  It may be specified manually, or it may be\n        generated arbitrarily based on the system\'s clock\n        (``DITHER_SEED_CLOCK``) or based on a checksum of the pixels in the\n        image\'s first tile (``DITHER_SEED_CHECKSUM``).  The clock-based method\n        is the default, and is sufficient to ensure that the value is\n        reasonably "arbitrary" and that the same seed is unlikely to be\n        generated sequentially.  The checksum method, on the other hand,\n        ensures that the same seed is used every time for a specific image.\n        This is particularly useful for software testing as it ensures that the\n        same image will always use the same seed.\n        '
        compression_type = CMTYPE_ALIASES.get(compression_type, compression_type)
        if tile_shape is None and tile_size is not None:
            tile_shape = tuple(tile_size[::-1])
        elif tile_shape is not None and tile_size is not None:
            raise ValueError('Cannot specify both tile_size and tile_shape. Note that tile_size is deprecated and tile_shape alone should be used.')
        if data is DELAYED:
            super().__init__(data=data, header=header)
        else:
            super().__init__(data=None, header=header)
            self.data = data
            self._update_header_data(header, name, compression_type=compression_type, tile_shape=tile_shape, hcomp_scale=hcomp_scale, hcomp_smooth=hcomp_smooth, quantize_level=quantize_level, quantize_method=quantize_method, dither_seed=dither_seed)
        self._do_not_scale_image_data = do_not_scale_image_data
        self._uint = uint
        self._scale_back = scale_back
        self._axes = [self._header.get('ZNAXIS' + str(axis + 1), 0) for axis in range(self._header.get('ZNAXIS', 0))]
        if do_not_scale_image_data:
            self._bzero = 0
            self._bscale = 1
        else:
            self._bzero = self._header.get('BZERO', 0)
            self._bscale = self._header.get('BSCALE', 1)
        self._bitpix = self._header['ZBITPIX']
        self._orig_bzero = self._bzero
        self._orig_bscale = self._bscale
        self._orig_bitpix = self._bitpix

    def _remove_unnecessary_default_extnames(self, header):
        if False:
            while True:
                i = 10
        'Remove default EXTNAME values if they are unnecessary.\n\n        Some data files (eg from CFHT) can have the default EXTNAME and\n        an explicit value.  This method removes the default if a more\n        specific header exists. It also removes any duplicate default\n        values.\n        '
        if 'EXTNAME' in header:
            indices = header._keyword_indices['EXTNAME']
            n_extname = len(indices)
            if n_extname > 1:
                extnames_to_remove = [index for index in indices if header[index] == self._default_name]
                if len(extnames_to_remove) == n_extname:
                    extnames_to_remove.pop(0)
                for index in sorted(extnames_to_remove, reverse=True):
                    del header[index]

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return str(self.header.get('EXTNAME', self._default_name))

    @name.setter
    def name(self, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, str):
            raise TypeError("'name' attribute must be a string")
        if not conf.extension_name_case_sensitive:
            value = value.upper()
        if 'EXTNAME' in self.header:
            self.header['EXTNAME'] = value
        else:
            self.header['EXTNAME'] = (value, 'extension name')

    @classmethod
    def match_header(cls, header):
        if False:
            return 10
        card = header.cards[0]
        if card.keyword != 'XTENSION':
            return False
        xtension = card.value
        if isinstance(xtension, str):
            xtension = xtension.rstrip()
        if xtension not in ('BINTABLE', 'A3DTABLE'):
            return False
        if 'ZIMAGE' not in header or not header['ZIMAGE']:
            return False
        return COMPRESSION_ENABLED

    def _update_header_data(self, image_header, name=None, compression_type=None, tile_shape=None, hcomp_scale=None, hcomp_smooth=None, quantize_level=None, quantize_method=None, dither_seed=None):
        if False:
            i = 10
            return i + 15
        "\n        Update the table header (`_header`) to the compressed\n        image format and to match the input data (if any).  Create\n        the image header (`_image_header`) from the input image\n        header (if any) and ensure it matches the input\n        data. Create the initially-empty table data array to hold\n        the compressed data.\n\n        This method is mainly called internally, but a user may wish to\n        call this method after assigning new data to the `CompImageHDU`\n        object that is of a different type.\n\n        Parameters\n        ----------\n        image_header : `~astropy.io.fits.Header`\n            header to be associated with the image\n\n        name : str, optional\n            the ``EXTNAME`` value; if this value is `None`, then the name from\n            the input image header will be used; if there is no name in the\n            input image header then the default name 'COMPRESSED_IMAGE' is used\n\n        compression_type : str, optional\n            compression algorithm 'RICE_1', 'PLIO_1', 'GZIP_1', 'GZIP_2',\n            'HCOMPRESS_1', 'NOCOMPRESS'; if this value is `None`, use value\n            already in the header; if no value already in the header, use\n            'RICE_1'\n\n        tile_shape : tuple of int, optional\n            compression tile shape (in C order); if this value is `None`, use\n            value already in the header; if no value already in the header,\n            treat each row of image as a tile\n\n        hcomp_scale : float, optional\n            HCOMPRESS scale parameter; if this value is `None`, use the value\n            already in the header; if no value already in the header, use 1\n\n        hcomp_smooth : float, optional\n            HCOMPRESS smooth parameter; if this value is `None`, use the value\n            already in the header; if no value already in the header, use 0\n\n        quantize_level : float, optional\n            floating point quantization level; if this value is `None`, use the\n            value already in the header; if no value already in header, use 16\n\n        quantize_method : int, optional\n            floating point quantization dithering method; can be either\n            NO_DITHER (-1), SUBTRACTIVE_DITHER_1 (1; default), or\n            SUBTRACTIVE_DITHER_2 (2)\n\n        dither_seed : int, optional\n            random seed to use for dithering; can be either an integer in the\n            range 1 to 1000 (inclusive), DITHER_SEED_CLOCK (0; default), or\n            DITHER_SEED_CHECKSUM (-1)\n        "
        self._remove_unnecessary_default_extnames(self._header)
        image_hdu = ImageHDU(data=self.data, header=self._header)
        self._image_header = CompImageHeader(self._header, image_hdu.header)
        self._axes = image_hdu._axes
        del image_hdu
        if self._has_data:
            huge_hdu = self.data.nbytes > 2 ** 32
        else:
            huge_hdu = False
        (self._header, self.columns) = _image_header_to_bintable_header_and_coldefs(image_header, self._image_header, self._header, name=name, huge_hdu=huge_hdu, compression_type=compression_type, tile_shape=tile_shape, hcomp_scale=hcomp_scale, hcomp_smooth=hcomp_smooth, quantize_level=quantize_level, quantize_method=quantize_method, dither_seed=dither_seed, axes=self._axes, generate_dither_seed=self._generate_dither_seed)
        if name:
            self.name = name

    def _scale_data(self, data):
        if False:
            i = 10
            return i + 15
        if self._orig_bzero != 0 or self._orig_bscale != 1:
            new_dtype = self._dtype_for_bitpix()
            data = np.array(data, dtype=new_dtype)
            if 'BLANK' in self._header:
                blanks = data == np.array(self._header['BLANK'], dtype='int32')
            else:
                blanks = None
            if self._orig_bscale != 1:
                np.multiply(data, self._orig_bscale, data)
            if self._orig_bzero != 0:
                np.add(data, self._orig_bzero, out=data, casting='unsafe')
            if blanks is not None:
                data = np.where(blanks, np.nan, data)
        return data

    @lazyproperty
    def data(self):
        if False:
            i = 10
            return i + 15
        '\n        The decompressed data array.\n\n        Note that accessing this will cause all the tiles to be loaded,\n        decompressed, and combined into a single data array. If you do\n        not need to access the whole array, consider instead using the\n        :attr:`~astropy.io.fits.CompImageHDU.section` property.\n        '
        if len(self.compressed_data) == 0:
            return None
        data = self.section[...]
        self._update_header_scale_info(data.dtype)
        return data

    @data.setter
    def data(self, data):
        if False:
            print('Hello World!')
        if data is not None and (not isinstance(data, np.ndarray) or data.dtype.fields is not None):
            raise TypeError('CompImageHDU data has incorrect type:{}; dtype.fields = {}'.format(type(data), data.dtype.fields))

    @lazyproperty
    def compressed_data(self):
        if False:
            for i in range(10):
                print('nop')
        compressed_data = super().data
        if isinstance(compressed_data, np.rec.recarray):
            del self.__dict__['data']
            return compressed_data
        else:
            self._update_compressed_data()
        return self.compressed_data

    @compressed_data.deleter
    def compressed_data(self):
        if False:
            for i in range(10):
                print('nop')
        if 'compressed_data' in self.__dict__:
            del self.__dict__['compressed_data']._coldefs
            del self.__dict__['compressed_data']

    @property
    def shape(self):
        if False:
            print('Hello World!')
        '\n        Shape of the image array--should be equivalent to ``self.data.shape``.\n        '
        return tuple(reversed(self._axes))

    @lazyproperty
    def header(self):
        if False:
            while True:
                i = 10
        if hasattr(self, '_image_header'):
            return self._image_header
        self._remove_unnecessary_default_extnames(self._header)
        self._image_header = _bintable_header_to_image_header(self._header)
        return self._image_header

    def _summary(self):
        if False:
            while True:
                i = 10
        '\n        Summarize the HDU: name, dimensions, and formats.\n        '
        class_name = self.__class__.__name__
        if self._data_loaded:
            if self.data is None:
                (_shape, _format) = ((), '')
            else:
                _shape = list(self.data.shape)
                _format = self.data.dtype.name
                _shape.reverse()
                _shape = tuple(_shape)
                _format = _format[_format.rfind('.') + 1:]
        else:
            _shape = ()
            for idx in range(self.header['NAXIS']):
                _shape += (self.header['NAXIS' + str(idx + 1)],)
            _format = BITPIX2DTYPE[self.header['BITPIX']]
        return (self.name, self.ver, class_name, len(self.header), _shape, _format)

    def _update_compressed_data(self):
        if False:
            print('Hello World!')
        '\n        Compress the image data so that it may be written to a file.\n        '
        image_bitpix = DTYPE2BITPIX[self.data.dtype.name]
        if image_bitpix != self._orig_bitpix or self.data.shape != self.shape:
            self._update_header_data(self.header)
        old_data = self.data
        if _is_pseudo_integer(self.data.dtype):
            self.data = np.array(self.data - _pseudo_zero(self.data.dtype), dtype=f'=i{self.data.dtype.itemsize}')
        try:
            nrows = self._header['NAXIS2']
            tbsize = self._header['NAXIS1'] * nrows
            self._header['PCOUNT'] = 0
            if 'THEAP' in self._header:
                del self._header['THEAP']
            self._theap = tbsize
            del self.compressed_data
            (heapsize, self.compressed_data) = compress_image_data(self.data, self.compression_type, self._header, self.columns)
        finally:
            self.data = old_data
        table_len = len(self.compressed_data) - heapsize
        if table_len != self._theap:
            raise Exception(f'Unexpected compressed table size (expected {self._theap}, got {table_len})')
        dtype = self.columns.dtype.newbyteorder('>')
        buf = self.compressed_data
        compressed_data = buf[:self._theap].view(dtype=dtype, type=np.rec.recarray)
        self.compressed_data = compressed_data.view(FITS_rec)
        self.compressed_data._coldefs = self.columns
        self.compressed_data._heapoffset = self._theap
        self.compressed_data._heapsize = heapsize

    def scale(self, type=None, option='old', bscale=1, bzero=0):
        if False:
            return 10
        '\n        Scale image data by using ``BSCALE`` and ``BZERO``.\n\n        Calling this method will scale ``self.data`` and update the keywords of\n        ``BSCALE`` and ``BZERO`` in ``self._header`` and ``self._image_header``.\n        This method should only be used right before writing to the output\n        file, as the data will be scaled and is therefore not very usable after\n        the call.\n\n        Parameters\n        ----------\n        type : str, optional\n            destination data type, use a string representing a numpy dtype\n            name, (e.g. ``\'uint8\'``, ``\'int16\'``, ``\'float32\'`` etc.).  If is\n            `None`, use the current data type.\n\n        option : str, optional\n            how to scale the data: if ``"old"``, use the original ``BSCALE``\n            and ``BZERO`` values when the data was read/created. If\n            ``"minmax"``, use the minimum and maximum of the data to scale.\n            The option will be overwritten by any user-specified bscale/bzero\n            values.\n\n        bscale, bzero : int, optional\n            user specified ``BSCALE`` and ``BZERO`` values.\n        '
        if self.data is None:
            return
        if type is None:
            type = BITPIX2DTYPE[self._bitpix]
        _type = getattr(np, type)
        if bscale != 1 or bzero != 0:
            _scale = bscale
            _zero = bzero
        elif option == 'old':
            _scale = self._orig_bscale
            _zero = self._orig_bzero
        elif option == 'minmax':
            if isinstance(_type, np.floating):
                _scale = 1
                _zero = 0
            else:
                _min = np.minimum.reduce(self.data.flat)
                _max = np.maximum.reduce(self.data.flat)
                if _type == np.uint8:
                    _zero = _min
                    _scale = (_max - _min) / (2.0 ** 8 - 1)
                else:
                    _zero = (_max + _min) / 2.0
                    _scale = (_max - _min) / (2.0 ** (8 * _type.bytes) - 2)
        if _zero != 0:
            np.subtract(self.data, _zero, out=self.data, casting='unsafe')
            self.header['BZERO'] = _zero
        else:
            for header in (self.header, self._header):
                with suppress(KeyError):
                    del header['BZERO']
        if _scale != 1:
            self.data /= _scale
            self.header['BSCALE'] = _scale
        else:
            for header in (self.header, self._header):
                with suppress(KeyError):
                    del header['BSCALE']
        if self.data.dtype.type != _type:
            self.data = np.array(np.around(self.data), dtype=_type)
        self._bitpix = DTYPE2BITPIX[self.data.dtype.name]
        self._bzero = self.header.get('BZERO', 0)
        self._bscale = self.header.get('BSCALE', 1)
        self.header['BITPIX'] = self._bitpix
        self._update_header_data(self.header)
        self._orig_bitpix = self._bitpix
        self._orig_bzero = self._bzero
        self._orig_bscale = self._bscale

    def _prewriteto(self, checksum=False, inplace=False):
        if False:
            return 10
        if self._scale_back:
            self.scale(BITPIX2DTYPE[self._orig_bitpix])
        if self._has_data:
            self._update_compressed_data()
            self._update_pseudo_int_scale_keywords()
            image_hdu = ImageHDU(data=self.data, header=self.header)
            image_hdu._update_checksum(checksum)
            if 'CHECKSUM' in image_hdu.header:
                self._image_header.set('CHECKSUM', image_hdu.header['CHECKSUM'], image_hdu.header.comments['CHECKSUM'])
            if 'DATASUM' in image_hdu.header:
                self._image_header.set('DATASUM', image_hdu.header['DATASUM'], image_hdu.header.comments['DATASUM'])
            self._imagedata = self.data
            self.__dict__['data'] = self.compressed_data
        return super()._prewriteto(checksum=checksum, inplace=inplace)

    def _writeheader(self, fileobj):
        if False:
            i = 10
            return i + 15
        "\n        Bypasses `BinTableHDU._writeheader()` which updates the header with\n        metadata about the data that is meaningless here; another reason\n        why this class maybe shouldn't inherit directly from BinTableHDU...\n        "
        return ExtensionHDU._writeheader(self, fileobj)

    def _writedata(self, fileobj):
        if False:
            return 10
        '\n        Wrap the basic ``_writedata`` method to restore the ``.data``\n        attribute to the uncompressed image data in the case of an exception.\n        '
        try:
            return super()._writedata(fileobj)
        finally:
            if hasattr(self, '_imagedata'):
                self.__dict__['data'] = self._imagedata
                del self._imagedata
            else:
                del self.data

    def _close(self, closed=True):
        if False:
            print('Hello World!')
        super()._close(closed=closed)
        if closed and self._data_loaded and (_get_array_mmap(self.compressed_data) is not None):
            del self.compressed_data

    def _dtype_for_bitpix(self):
        if False:
            while True:
                i = 10
        '\n        Determine the dtype that the data should be converted to depending on\n        the BITPIX value in the header, and possibly on the BSCALE value as\n        well.  Returns None if there should not be any change.\n        '
        bitpix = self._orig_bitpix
        if self._uint and self._orig_bscale == 1:
            for (bits, dtype) in ((16, np.dtype('uint16')), (32, np.dtype('uint32')), (64, np.dtype('uint64'))):
                if bitpix == bits and self._orig_bzero == 1 << bits - 1:
                    return dtype
        if bitpix > 16:
            return np.dtype('float64')
        elif bitpix > 0:
            return np.dtype('float32')

    def _update_header_scale_info(self, dtype=None):
        if False:
            return 10
        if not self._do_not_scale_image_data and (not (self._orig_bzero == 0 and self._orig_bscale == 1)):
            for keyword in ['BSCALE', 'BZERO']:
                for header in (self.header, self._header):
                    with suppress(KeyError):
                        del header[keyword]
                        header.append()
            if dtype is None:
                dtype = self._dtype_for_bitpix()
            if dtype is not None:
                self.header['BITPIX'] = DTYPE2BITPIX[dtype.name]
            self._bzero = 0
            self._bscale = 1
            self._bitpix = self.header['BITPIX']

    def _generate_dither_seed(self, seed):
        if False:
            return 10
        if not _is_int(seed):
            raise TypeError('Seed must be an integer')
        if not -1 <= seed <= 10000:
            raise ValueError(f'Seed for random dithering must be either between 1 and 10000 inclusive, 0 for autogeneration from the system clock, or -1 for autogeneration from a checksum of the first image tile (got {seed})')
        if seed == DITHER_SEED_CHECKSUM:
            tile_dims = self.tile_shape
            first_tile = self.data[tuple((slice(d) for d in tile_dims))]
            csum = first_tile.view(dtype='uint8').sum()
            return ctypes.c_ulong(csum).value % 10000 + 1
        elif seed == DITHER_SEED_CLOCK:
            return (sum((int(x) for x in math.modf(time.time()))) + id(self)) % 10000 + 1
        else:
            return seed

    @property
    def section(self):
        if False:
            print('Hello World!')
        '\n        Efficiently access a section of the image array\n\n        This property can be used to access a section of the data without\n        loading and decompressing the entire array into memory.\n\n        The :class:`~astropy.io.fits.CompImageSection` object returned by this\n        attribute is not meant to be used directly by itself. Rather, slices of\n        the section return the appropriate slice of the data, and loads *only*\n        that section into memory. Any valid basic Numpy index can be used to\n        slice :class:`~astropy.io.fits.CompImageSection`.\n\n        Note that accessing data using :attr:`CompImageHDU.section` will always\n        load tiles one at a time from disk, and therefore when accessing a large\n        fraction of the data (or slicing it in a way that would cause most tiles\n        to be loaded) you may obtain better performance by using\n        :attr:`CompImageHDU.data`.\n        '
        return CompImageSection(self)

    @property
    def tile_shape(self):
        if False:
            while True:
                i = 10
        '\n        The tile shape used for the tiled compression.\n\n        This shape is given in Numpy/C order\n        '
        return tuple([self._header[f'ZTILE{idx + 1}'] for idx in range(self._header['ZNAXIS'] - 1, -1, -1)])

    @property
    def compression_type(self):
        if False:
            while True:
                i = 10
        '\n        The name of the compression algorithm.\n        '
        return self._header.get('ZCMPTYPE', DEFAULT_COMPRESSION_TYPE)