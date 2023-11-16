from . import Image, ImageFile, ImagePalette
from ._binary import i32be as i32

def _accept(prefix):
    if False:
        print('Hello World!')
    return len(prefix) >= 4 and i32(prefix) == 1504078485

class SunImageFile(ImageFile.ImageFile):
    format = 'SUN'
    format_description = 'Sun Raster File'

    def _open(self):
        if False:
            print('Hello World!')
        s = self.fp.read(32)
        if not _accept(s):
            msg = 'not an SUN raster file'
            raise SyntaxError(msg)
        offset = 32
        self._size = (i32(s, 4), i32(s, 8))
        depth = i32(s, 12)
        file_type = i32(s, 20)
        palette_type = i32(s, 24)
        palette_length = i32(s, 28)
        if depth == 1:
            (self._mode, rawmode) = ('1', '1;I')
        elif depth == 4:
            (self._mode, rawmode) = ('L', 'L;4')
        elif depth == 8:
            self._mode = rawmode = 'L'
        elif depth == 24:
            if file_type == 3:
                (self._mode, rawmode) = ('RGB', 'RGB')
            else:
                (self._mode, rawmode) = ('RGB', 'BGR')
        elif depth == 32:
            if file_type == 3:
                (self._mode, rawmode) = ('RGB', 'RGBX')
            else:
                (self._mode, rawmode) = ('RGB', 'BGRX')
        else:
            msg = 'Unsupported Mode/Bit Depth'
            raise SyntaxError(msg)
        if palette_length:
            if palette_length > 1024:
                msg = 'Unsupported Color Palette Length'
                raise SyntaxError(msg)
            if palette_type != 1:
                msg = 'Unsupported Palette Type'
                raise SyntaxError(msg)
            offset = offset + palette_length
            self.palette = ImagePalette.raw('RGB;L', self.fp.read(palette_length))
            if self.mode == 'L':
                self._mode = 'P'
                rawmode = rawmode.replace('L', 'P')
        stride = (self.size[0] * depth + 15) // 16 * 2
        if file_type in (0, 1, 3, 4, 5):
            self.tile = [('raw', (0, 0) + self.size, offset, (rawmode, stride))]
        elif file_type == 2:
            self.tile = [('sun_rle', (0, 0) + self.size, offset, rawmode)]
        else:
            msg = 'Unsupported Sun Raster file type'
            raise SyntaxError(msg)
Image.register_open(SunImageFile.format, SunImageFile, _accept)
Image.register_extension(SunImageFile.format, '.ras')