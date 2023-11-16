import io
import itertools
import struct
import sys
from . import Image
from ._util import is_path
MAXBLOCK = 65536
SAFEBLOCK = 1024 * 1024
LOAD_TRUNCATED_IMAGES = False
'Whether or not to load truncated image files. User code may change this.'
ERRORS = {-1: 'image buffer overrun error', -2: 'decoding error', -3: 'unknown error', -8: 'bad configuration', -9: 'out of memory error'}
'\nDict of known error codes returned from :meth:`.PyDecoder.decode`,\n:meth:`.PyEncoder.encode` :meth:`.PyEncoder.encode_to_pyfd` and\n:meth:`.PyEncoder.encode_to_file`.\n'

def raise_oserror(error):
    if False:
        i = 10
        return i + 15
    try:
        msg = Image.core.getcodecstatus(error)
    except AttributeError:
        msg = ERRORS.get(error)
    if not msg:
        msg = f'decoder error {error}'
    msg += ' when reading image file'
    raise OSError(msg)

def _tilesort(t):
    if False:
        i = 10
        return i + 15
    return t[2]

class ImageFile(Image.Image):
    """Base class for image file format handlers."""

    def __init__(self, fp=None, filename=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._min_frame = 0
        self.custom_mimetype = None
        self.tile = None
        ' A list of tile descriptors, or ``None`` '
        self.readonly = 1
        self.decoderconfig = ()
        self.decodermaxblock = MAXBLOCK
        if is_path(fp):
            self.fp = open(fp, 'rb')
            self.filename = fp
            self._exclusive_fp = True
        else:
            self.fp = fp
            self.filename = filename
            self._exclusive_fp = None
        try:
            try:
                self._open()
            except (IndexError, TypeError, KeyError, EOFError, struct.error) as v:
                raise SyntaxError(v) from v
            if not self.mode or self.size[0] <= 0 or self.size[1] <= 0:
                msg = 'not identified by this driver'
                raise SyntaxError(msg)
        except BaseException:
            if self._exclusive_fp:
                self.fp.close()
            raise

    def get_format_mimetype(self):
        if False:
            i = 10
            return i + 15
        if self.custom_mimetype:
            return self.custom_mimetype
        if self.format is not None:
            return Image.MIME.get(self.format.upper())

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        self.tile = []
        super().__setstate__(state)

    def verify(self):
        if False:
            print('Hello World!')
        'Check file integrity'
        if self._exclusive_fp:
            self.fp.close()
        self.fp = None

    def load(self):
        if False:
            i = 10
            return i + 15
        'Load image data based on tile list'
        if self.tile is None:
            msg = 'cannot load this image'
            raise OSError(msg)
        pixel = Image.Image.load(self)
        if not self.tile:
            return pixel
        self.map = None
        use_mmap = self.filename and len(self.tile) == 1
        use_mmap = use_mmap and (not hasattr(sys, 'pypy_version_info'))
        readonly = 0
        try:
            read = self.load_read
            use_mmap = False
        except AttributeError:
            read = self.fp.read
        try:
            seek = self.load_seek
            use_mmap = False
        except AttributeError:
            seek = self.fp.seek
        if use_mmap:
            (decoder_name, extents, offset, args) = self.tile[0]
            if decoder_name == 'raw' and len(args) >= 3 and (args[0] == self.mode) and (args[0] in Image._MAPMODES):
                try:
                    import mmap
                    with open(self.filename) as fp:
                        self.map = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                    if offset + self.size[1] * args[1] > self.map.size():
                        msg = 'buffer is not large enough'
                        raise OSError(msg)
                    self.im = Image.core.map_buffer(self.map, self.size, decoder_name, offset, args)
                    readonly = 1
                    if self.palette:
                        self.palette.dirty = 1
                except (AttributeError, OSError, ImportError):
                    self.map = None
        self.load_prepare()
        err_code = -3
        if not self.map:
            self.tile.sort(key=_tilesort)
            try:
                prefix = self.tile_prefix
            except AttributeError:
                prefix = b''
            self.tile = [list(tiles)[-1] for (_, tiles) in itertools.groupby(self.tile, lambda tile: (tile[0], tile[1], tile[3]))]
            for (decoder_name, extents, offset, args) in self.tile:
                seek(offset)
                decoder = Image._getdecoder(self.mode, decoder_name, args, self.decoderconfig)
                try:
                    decoder.setimage(self.im, extents)
                    if decoder.pulls_fd:
                        decoder.setfd(self.fp)
                        err_code = decoder.decode(b'')[1]
                    else:
                        b = prefix
                        while True:
                            try:
                                s = read(self.decodermaxblock)
                            except (IndexError, struct.error) as e:
                                if LOAD_TRUNCATED_IMAGES:
                                    break
                                else:
                                    msg = 'image file is truncated'
                                    raise OSError(msg) from e
                            if not s:
                                if LOAD_TRUNCATED_IMAGES:
                                    break
                                else:
                                    msg = f'image file is truncated ({len(b)} bytes not processed)'
                                    raise OSError(msg)
                            b = b + s
                            (n, err_code) = decoder.decode(b)
                            if n < 0:
                                break
                            b = b[n:]
                finally:
                    decoder.cleanup()
        self.tile = []
        self.readonly = readonly
        self.load_end()
        if self._exclusive_fp and self._close_exclusive_fp_after_loading:
            self.fp.close()
        self.fp = None
        if not self.map and (not LOAD_TRUNCATED_IMAGES) and (err_code < 0):
            raise_oserror(err_code)
        return Image.Image.load(self)

    def load_prepare(self):
        if False:
            print('Hello World!')
        if not self.im or self.im.mode != self.mode or self.im.size != self.size:
            self.im = Image.core.new(self.mode, self.size)
        if self.mode == 'P':
            Image.Image.load(self)

    def load_end(self):
        if False:
            return 10
        pass

    def _seek_check(self, frame):
        if False:
            return 10
        if frame < self._min_frame or (not (hasattr(self, '_n_frames') and self._n_frames is None) and frame >= self.n_frames + self._min_frame):
            msg = 'attempt to seek outside sequence'
            raise EOFError(msg)
        return self.tell() != frame

class StubImageFile(ImageFile):
    """
    Base class for stub image loaders.

    A stub loader is an image loader that can identify files of a
    certain format, but relies on external code to load the file.
    """

    def _open(self):
        if False:
            return 10
        msg = 'StubImageFile subclass must implement _open'
        raise NotImplementedError(msg)

    def load(self):
        if False:
            return 10
        loader = self._load()
        if loader is None:
            msg = f'cannot find loader for this {self.format} file'
            raise OSError(msg)
        image = loader.load(self)
        assert image is not None
        self.__class__ = image.__class__
        self.__dict__ = image.__dict__
        return image.load()

    def _load(self):
        if False:
            i = 10
            return i + 15
        '(Hook) Find actual image loader.'
        msg = 'StubImageFile subclass must implement _load'
        raise NotImplementedError(msg)

class Parser:
    """
    Incremental image parser.  This class implements the standard
    feed/close consumer interface.
    """
    incremental = None
    image = None
    data = None
    decoder = None
    offset = 0
    finished = 0

    def reset(self):
        if False:
            while True:
                i = 10
        "\n        (Consumer) Reset the parser.  Note that you can only call this\n        method immediately after you've created a parser; parser\n        instances cannot be reused.\n        "
        assert self.data is None, 'cannot reuse parsers'

    def feed(self, data):
        if False:
            print('Hello World!')
        '\n        (Consumer) Feed data to the parser.\n\n        :param data: A string buffer.\n        :exception OSError: If the parser failed to parse the image file.\n        '
        if self.finished:
            return
        if self.data is None:
            self.data = data
        else:
            self.data = self.data + data
        if self.decoder:
            if self.offset > 0:
                skip = min(len(self.data), self.offset)
                self.data = self.data[skip:]
                self.offset = self.offset - skip
                if self.offset > 0 or not self.data:
                    return
            (n, e) = self.decoder.decode(self.data)
            if n < 0:
                self.data = None
                self.finished = 1
                if e < 0:
                    self.image = None
                    raise_oserror(e)
                else:
                    return
            self.data = self.data[n:]
        elif self.image:
            pass
        else:
            try:
                with io.BytesIO(self.data) as fp:
                    im = Image.open(fp)
            except OSError:
                pass
            else:
                flag = hasattr(im, 'load_seek') or hasattr(im, 'load_read')
                if flag or len(im.tile) != 1:
                    self.decode = None
                else:
                    im.load_prepare()
                    (d, e, o, a) = im.tile[0]
                    im.tile = []
                    self.decoder = Image._getdecoder(im.mode, d, a, im.decoderconfig)
                    self.decoder.setimage(im.im, e)
                    self.offset = o
                    if self.offset <= len(self.data):
                        self.data = self.data[self.offset:]
                        self.offset = 0
                self.image = im

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        self.close()

    def close(self):
        if False:
            print('Hello World!')
        '\n        (Consumer) Close the stream.\n\n        :returns: An image object.\n        :exception OSError: If the parser failed to parse the image file either\n                            because it cannot be identified or cannot be\n                            decoded.\n        '
        if self.decoder:
            self.feed(b'')
            self.data = self.decoder = None
            if not self.finished:
                msg = 'image was incomplete'
                raise OSError(msg)
        if not self.image:
            msg = 'cannot parse this image'
            raise OSError(msg)
        if self.data:
            with io.BytesIO(self.data) as fp:
                try:
                    self.image = Image.open(fp)
                finally:
                    self.image.load()
        return self.image

def _save(im, fp, tile, bufsize=0):
    if False:
        for i in range(10):
            print('nop')
    'Helper to save image based on tile list\n\n    :param im: Image object.\n    :param fp: File object.\n    :param tile: Tile list.\n    :param bufsize: Optional buffer size\n    '
    im.load()
    if not hasattr(im, 'encoderconfig'):
        im.encoderconfig = ()
    tile.sort(key=_tilesort)
    bufsize = max(MAXBLOCK, bufsize, im.size[0] * 4)
    try:
        fh = fp.fileno()
        fp.flush()
        _encode_tile(im, fp, tile, bufsize, fh)
    except (AttributeError, io.UnsupportedOperation) as exc:
        _encode_tile(im, fp, tile, bufsize, None, exc)
    if hasattr(fp, 'flush'):
        fp.flush()

def _encode_tile(im, fp, tile, bufsize, fh, exc=None):
    if False:
        i = 10
        return i + 15
    for (e, b, o, a) in tile:
        if o > 0:
            fp.seek(o)
        encoder = Image._getencoder(im.mode, e, a, im.encoderconfig)
        try:
            encoder.setimage(im.im, b)
            if encoder.pushes_fd:
                encoder.setfd(fp)
                errcode = encoder.encode_to_pyfd()[1]
            elif exc:
                while True:
                    (errcode, data) = encoder.encode(bufsize)[1:]
                    fp.write(data)
                    if errcode:
                        break
            else:
                errcode = encoder.encode_to_file(fh, bufsize)
            if errcode < 0:
                msg = f'encoder error {errcode} when writing image file'
                raise OSError(msg) from exc
        finally:
            encoder.cleanup()

def _safe_read(fp, size):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reads large blocks in a safe way.  Unlike fp.read(n), this function\n    doesn't trust the user.  If the requested size is larger than\n    SAFEBLOCK, the file is read block by block.\n\n    :param fp: File handle.  Must implement a <b>read</b> method.\n    :param size: Number of bytes to read.\n    :returns: A string containing <i>size</i> bytes of data.\n\n    Raises an OSError if the file is truncated and the read cannot be completed\n\n    "
    if size <= 0:
        return b''
    if size <= SAFEBLOCK:
        data = fp.read(size)
        if len(data) < size:
            msg = 'Truncated File Read'
            raise OSError(msg)
        return data
    data = []
    remaining_size = size
    while remaining_size > 0:
        block = fp.read(min(remaining_size, SAFEBLOCK))
        if not block:
            break
        data.append(block)
        remaining_size -= len(block)
    if sum((len(d) for d in data)) < size:
        msg = 'Truncated File Read'
        raise OSError(msg)
    return b''.join(data)

class PyCodecState:

    def __init__(self):
        if False:
            print('Hello World!')
        self.xsize = 0
        self.ysize = 0
        self.xoff = 0
        self.yoff = 0

    def extents(self):
        if False:
            while True:
                i = 10
        return (self.xoff, self.yoff, self.xoff + self.xsize, self.yoff + self.ysize)

class PyCodec:

    def __init__(self, mode, *args):
        if False:
            return 10
        self.im = None
        self.state = PyCodecState()
        self.fd = None
        self.mode = mode
        self.init(args)

    def init(self, args):
        if False:
            i = 10
            return i + 15
        '\n        Override to perform codec specific initialization\n\n        :param args: Array of args items from the tile entry\n        :returns: None\n        '
        self.args = args

    def cleanup(self):
        if False:
            while True:
                i = 10
        '\n        Override to perform codec specific cleanup\n\n        :returns: None\n        '
        pass

    def setfd(self, fd):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called from ImageFile to set the Python file-like object\n\n        :param fd: A Python file-like object\n        :returns: None\n        '
        self.fd = fd

    def setimage(self, im, extents=None):
        if False:
            i = 10
            return i + 15
        '\n        Called from ImageFile to set the core output image for the codec\n\n        :param im: A core image object\n        :param extents: a 4 tuple of (x0, y0, x1, y1) defining the rectangle\n            for this tile\n        :returns: None\n        '
        self.im = im
        if extents:
            (x0, y0, x1, y1) = extents
        else:
            (x0, y0, x1, y1) = (0, 0, 0, 0)
        if x0 == 0 and x1 == 0:
            (self.state.xsize, self.state.ysize) = self.im.size
        else:
            self.state.xoff = x0
            self.state.yoff = y0
            self.state.xsize = x1 - x0
            self.state.ysize = y1 - y0
        if self.state.xsize <= 0 or self.state.ysize <= 0:
            msg = 'Size cannot be negative'
            raise ValueError(msg)
        if self.state.xsize + self.state.xoff > self.im.size[0] or self.state.ysize + self.state.yoff > self.im.size[1]:
            msg = 'Tile cannot extend outside image'
            raise ValueError(msg)

class PyDecoder(PyCodec):
    """
    Python implementation of a format decoder. Override this class and
    add the decoding logic in the :meth:`decode` method.

    See :ref:`Writing Your Own File Codec in Python<file-codecs-py>`
    """
    _pulls_fd = False

    @property
    def pulls_fd(self):
        if False:
            return 10
        return self._pulls_fd

    def decode(self, buffer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override to perform the decoding process.\n\n        :param buffer: A bytes object with the data to be decoded.\n        :returns: A tuple of ``(bytes consumed, errcode)``.\n            If finished with decoding return -1 for the bytes consumed.\n            Err codes are from :data:`.ImageFile.ERRORS`.\n        '
        msg = 'unavailable in base decoder'
        raise NotImplementedError(msg)

    def set_as_raw(self, data, rawmode=None):
        if False:
            print('Hello World!')
        '\n        Convenience method to set the internal image from a stream of raw data\n\n        :param data: Bytes to be set\n        :param rawmode: The rawmode to be used for the decoder.\n            If not specified, it will default to the mode of the image\n        :returns: None\n        '
        if not rawmode:
            rawmode = self.mode
        d = Image._getdecoder(self.mode, 'raw', rawmode)
        d.setimage(self.im, self.state.extents())
        s = d.decode(data)
        if s[0] >= 0:
            msg = 'not enough image data'
            raise ValueError(msg)
        if s[1] != 0:
            msg = 'cannot decode image data'
            raise ValueError(msg)

class PyEncoder(PyCodec):
    """
    Python implementation of a format encoder. Override this class and
    add the decoding logic in the :meth:`encode` method.

    See :ref:`Writing Your Own File Codec in Python<file-codecs-py>`
    """
    _pushes_fd = False

    @property
    def pushes_fd(self):
        if False:
            return 10
        return self._pushes_fd

    def encode(self, bufsize):
        if False:
            while True:
                i = 10
        '\n        Override to perform the encoding process.\n\n        :param bufsize: Buffer size.\n        :returns: A tuple of ``(bytes encoded, errcode, bytes)``.\n            If finished with encoding return 1 for the error code.\n            Err codes are from :data:`.ImageFile.ERRORS`.\n        '
        msg = 'unavailable in base encoder'
        raise NotImplementedError(msg)

    def encode_to_pyfd(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If ``pushes_fd`` is ``True``, then this method will be used,\n        and ``encode()`` will only be called once.\n\n        :returns: A tuple of ``(bytes consumed, errcode)``.\n            Err codes are from :data:`.ImageFile.ERRORS`.\n        '
        if not self.pushes_fd:
            return (0, -8)
        (bytes_consumed, errcode, data) = self.encode(0)
        if data:
            self.fd.write(data)
        return (bytes_consumed, errcode)

    def encode_to_file(self, fh, bufsize):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param fh: File handle.\n        :param bufsize: Buffer size.\n\n        :returns: If finished successfully, return 0.\n            Otherwise, return an error code. Err codes are from\n            :data:`.ImageFile.ERRORS`.\n        '
        errcode = 0
        while errcode == 0:
            (status, errcode, buf) = self.encode(bufsize)
            if status > 0:
                fh.write(buf[status:])
        return errcode