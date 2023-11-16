from . import Image, ImageFile
from ._binary import i32be as i32

def _accept(prefix):
    if False:
        print('Hello World!')
    return len(prefix) >= 8 and i32(prefix, 0) >= 20 and (i32(prefix, 4) in (1, 2))

class GbrImageFile(ImageFile.ImageFile):
    format = 'GBR'
    format_description = 'GIMP brush file'

    def _open(self):
        if False:
            for i in range(10):
                print('nop')
        header_size = i32(self.fp.read(4))
        if header_size < 20:
            msg = 'not a GIMP brush'
            raise SyntaxError(msg)
        version = i32(self.fp.read(4))
        if version not in (1, 2):
            msg = f'Unsupported GIMP brush version: {version}'
            raise SyntaxError(msg)
        width = i32(self.fp.read(4))
        height = i32(self.fp.read(4))
        color_depth = i32(self.fp.read(4))
        if width <= 0 or height <= 0:
            msg = 'not a GIMP brush'
            raise SyntaxError(msg)
        if color_depth not in (1, 4):
            msg = f'Unsupported GIMP brush color depth: {color_depth}'
            raise SyntaxError(msg)
        if version == 1:
            comment_length = header_size - 20
        else:
            comment_length = header_size - 28
            magic_number = self.fp.read(4)
            if magic_number != b'GIMP':
                msg = 'not a GIMP brush, bad magic number'
                raise SyntaxError(msg)
            self.info['spacing'] = i32(self.fp.read(4))
        comment = self.fp.read(comment_length)[:-1]
        if color_depth == 1:
            self._mode = 'L'
        else:
            self._mode = 'RGBA'
        self._size = (width, height)
        self.info['comment'] = comment
        Image._decompression_bomb_check(self.size)
        self._data_size = width * height * color_depth

    def load(self):
        if False:
            i = 10
            return i + 15
        if not self.im:
            self.im = Image.core.new(self.mode, self.size)
            self.frombytes(self.fp.read(self._data_size))
        return Image.Image.load(self)
Image.register_open(GbrImageFile.format, GbrImageFile, _accept)
Image.register_extension(GbrImageFile.format, '.gbr')