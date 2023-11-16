import os
import struct
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8

def _accept(prefix):
    if False:
        while True:
            i = 10
    return len(prefix) >= 2 and i16(prefix) == 474
MODES = {(1, 1, 1): 'L', (1, 2, 1): 'L', (2, 1, 1): 'L;16B', (2, 2, 1): 'L;16B', (1, 3, 3): 'RGB', (2, 3, 3): 'RGB;16B', (1, 3, 4): 'RGBA', (2, 3, 4): 'RGBA;16B'}

class SgiImageFile(ImageFile.ImageFile):
    format = 'SGI'
    format_description = 'SGI Image File Format'

    def _open(self):
        if False:
            print('Hello World!')
        headlen = 512
        s = self.fp.read(headlen)
        if not _accept(s):
            msg = 'Not an SGI image file'
            raise ValueError(msg)
        compression = s[2]
        bpc = s[3]
        dimension = i16(s, 4)
        xsize = i16(s, 6)
        ysize = i16(s, 8)
        zsize = i16(s, 10)
        layout = (bpc, dimension, zsize)
        rawmode = ''
        try:
            rawmode = MODES[layout]
        except KeyError:
            pass
        if rawmode == '':
            msg = 'Unsupported SGI image mode'
            raise ValueError(msg)
        self._size = (xsize, ysize)
        self._mode = rawmode.split(';')[0]
        if self.mode == 'RGB':
            self.custom_mimetype = 'image/rgb'
        orientation = -1
        if compression == 0:
            pagesize = xsize * ysize * bpc
            if bpc == 2:
                self.tile = [('SGI16', (0, 0) + self.size, headlen, (self.mode, 0, orientation))]
            else:
                self.tile = []
                offset = headlen
                for layer in self.mode:
                    self.tile.append(('raw', (0, 0) + self.size, offset, (layer, 0, orientation)))
                    offset += pagesize
        elif compression == 1:
            self.tile = [('sgi_rle', (0, 0) + self.size, headlen, (rawmode, orientation, bpc))]

def _save(im, fp, filename):
    if False:
        for i in range(10):
            print('nop')
    if im.mode != 'RGB' and im.mode != 'RGBA' and (im.mode != 'L'):
        msg = 'Unsupported SGI image mode'
        raise ValueError(msg)
    info = im.encoderinfo
    bpc = info.get('bpc', 1)
    if bpc not in (1, 2):
        msg = 'Unsupported number of bytes per pixel'
        raise ValueError(msg)
    orientation = -1
    magic_number = 474
    rle = 0
    dim = 3
    (x, y) = im.size
    if im.mode == 'L' and y == 1:
        dim = 1
    elif im.mode == 'L':
        dim = 2
    z = len(im.mode)
    if dim == 1 or dim == 2:
        z = 1
    if len(im.getbands()) != z:
        msg = f'incorrect number of bands in SGI write: {z} vs {len(im.getbands())}'
        raise ValueError(msg)
    pinmin = 0
    pinmax = 255
    img_name = os.path.splitext(os.path.basename(filename))[0]
    img_name = img_name.encode('ascii', 'ignore')
    colormap = 0
    fp.write(struct.pack('>h', magic_number))
    fp.write(o8(rle))
    fp.write(o8(bpc))
    fp.write(struct.pack('>H', dim))
    fp.write(struct.pack('>H', x))
    fp.write(struct.pack('>H', y))
    fp.write(struct.pack('>H', z))
    fp.write(struct.pack('>l', pinmin))
    fp.write(struct.pack('>l', pinmax))
    fp.write(struct.pack('4s', b''))
    fp.write(struct.pack('79s', img_name))
    fp.write(struct.pack('s', b''))
    fp.write(struct.pack('>l', colormap))
    fp.write(struct.pack('404s', b''))
    rawmode = 'L'
    if bpc == 2:
        rawmode = 'L;16B'
    for channel in im.split():
        fp.write(channel.tobytes('raw', rawmode, 0, orientation))
    if hasattr(fp, 'flush'):
        fp.flush()

class SGI16Decoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        if False:
            return 10
        (rawmode, stride, orientation) = self.args
        pagesize = self.state.xsize * self.state.ysize
        zsize = len(self.mode)
        self.fd.seek(512)
        for band in range(zsize):
            channel = Image.new('L', (self.state.xsize, self.state.ysize))
            channel.frombytes(self.fd.read(2 * pagesize), 'raw', 'L;16B', stride, orientation)
            self.im.putband(channel.im, band)
        return (-1, 0)
Image.register_decoder('SGI16', SGI16Decoder)
Image.register_open(SgiImageFile.format, SgiImageFile, _accept)
Image.register_save(SgiImageFile.format, _save)
Image.register_mime(SgiImageFile.format, 'image/sgi')
Image.register_extensions(SgiImageFile.format, ['.bw', '.rgb', '.rgba', '.sgi'])