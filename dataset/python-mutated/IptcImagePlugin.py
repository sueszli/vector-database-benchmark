import os
import tempfile
from . import Image, ImageFile
from ._binary import i8, o8
from ._binary import i16be as i16
from ._binary import i32be as i32
COMPRESSION = {1: 'raw', 5: 'jpeg'}
PAD = o8(0) * 4

def i(c):
    if False:
        i = 10
        return i + 15
    return i32((PAD + c)[-4:])

def dump(c):
    if False:
        for i in range(10):
            print('nop')
    for i in c:
        print('%02x' % i8(i), end=' ')
    print()

class IptcImageFile(ImageFile.ImageFile):
    format = 'IPTC'
    format_description = 'IPTC/NAA'

    def getint(self, key):
        if False:
            for i in range(10):
                print('nop')
        return i(self.info[key])

    def field(self):
        if False:
            print('Hello World!')
        s = self.fp.read(5)
        if not s.strip(b'\x00'):
            return (None, 0)
        tag = (s[1], s[2])
        if s[0] != 28 or tag[0] not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 240]:
            msg = 'invalid IPTC/NAA file'
            raise SyntaxError(msg)
        size = s[3]
        if size > 132:
            msg = 'illegal field length in IPTC/NAA file'
            raise OSError(msg)
        elif size == 128:
            size = 0
        elif size > 128:
            size = i(self.fp.read(size - 128))
        else:
            size = i16(s, 3)
        return (tag, size)

    def _open(self):
        if False:
            print('Hello World!')
        while True:
            offset = self.fp.tell()
            (tag, size) = self.field()
            if not tag or tag == (8, 10):
                break
            if size:
                tagdata = self.fp.read(size)
            else:
                tagdata = None
            if tag in self.info:
                if isinstance(self.info[tag], list):
                    self.info[tag].append(tagdata)
                else:
                    self.info[tag] = [self.info[tag], tagdata]
            else:
                self.info[tag] = tagdata
        layers = i8(self.info[3, 60][0])
        component = i8(self.info[3, 60][1])
        if (3, 65) in self.info:
            id = i8(self.info[3, 65][0]) - 1
        else:
            id = 0
        if layers == 1 and (not component):
            self._mode = 'L'
        elif layers == 3 and component:
            self._mode = 'RGB'[id]
        elif layers == 4 and component:
            self._mode = 'CMYK'[id]
        self._size = (self.getint((3, 20)), self.getint((3, 30)))
        try:
            compression = COMPRESSION[self.getint((3, 120))]
        except KeyError as e:
            msg = 'Unknown IPTC image compression'
            raise OSError(msg) from e
        if tag == (8, 10):
            self.tile = [('iptc', (compression, offset), (0, 0, self.size[0], self.size[1]))]

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.tile) != 1 or self.tile[0][0] != 'iptc':
            return ImageFile.ImageFile.load(self)
        (type, tile, box) = self.tile[0]
        (encoding, offset) = tile
        self.fp.seek(offset)
        (o_fd, outfile) = tempfile.mkstemp(text=False)
        o = os.fdopen(o_fd)
        if encoding == 'raw':
            o.write('P5\n%d %d\n255\n' % self.size)
        while True:
            (type, size) = self.field()
            if type != (8, 10):
                break
            while size > 0:
                s = self.fp.read(min(size, 8192))
                if not s:
                    break
                o.write(s)
                size -= len(s)
        o.close()
        try:
            with Image.open(outfile) as _im:
                _im.load()
                self.im = _im.im
        finally:
            try:
                os.unlink(outfile)
            except OSError:
                pass
Image.register_open(IptcImageFile.format, IptcImageFile)
Image.register_extension(IptcImageFile.format, '.iim')

def getiptcinfo(im):
    if False:
        return 10
    '\n    Get IPTC information from TIFF, JPEG, or IPTC file.\n\n    :param im: An image containing IPTC data.\n    :returns: A dictionary containing IPTC information, or None if\n        no IPTC information block was found.\n    '
    import io
    from . import JpegImagePlugin, TiffImagePlugin
    data = None
    if isinstance(im, IptcImageFile):
        return im.info
    elif isinstance(im, JpegImagePlugin.JpegImageFile):
        photoshop = im.info.get('photoshop')
        if photoshop:
            data = photoshop.get(1028)
    elif isinstance(im, TiffImagePlugin.TiffImageFile):
        try:
            data = im.tag.tagdata[TiffImagePlugin.IPTC_NAA_CHUNK]
        except (AttributeError, KeyError):
            pass
    if data is None:
        return None

    class FakeImage:
        pass
    im = FakeImage()
    im.__class__ = IptcImageFile
    im.info = {}
    im.fp = io.BytesIO(data)
    try:
        im._open()
    except (IndexError, KeyError):
        pass
    return im.info