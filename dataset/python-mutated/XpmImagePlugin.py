import re
from . import Image, ImageFile, ImagePalette
from ._binary import o8
xpm_head = re.compile(b'"([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)')

def _accept(prefix):
    if False:
        i = 10
        return i + 15
    return prefix[:9] == b'/* XPM */'

class XpmImageFile(ImageFile.ImageFile):
    format = 'XPM'
    format_description = 'X11 Pixel Map'

    def _open(self):
        if False:
            i = 10
            return i + 15
        if not _accept(self.fp.read(9)):
            msg = 'not an XPM file'
            raise SyntaxError(msg)
        while True:
            s = self.fp.readline()
            if not s:
                msg = 'broken XPM file'
                raise SyntaxError(msg)
            m = xpm_head.match(s)
            if m:
                break
        self._size = (int(m.group(1)), int(m.group(2)))
        pal = int(m.group(3))
        bpp = int(m.group(4))
        if pal > 256 or bpp != 1:
            msg = 'cannot read this XPM file'
            raise ValueError(msg)
        palette = [b'\x00\x00\x00'] * 256
        for _ in range(pal):
            s = self.fp.readline()
            if s[-2:] == b'\r\n':
                s = s[:-2]
            elif s[-1:] in b'\r\n':
                s = s[:-1]
            c = s[1]
            s = s[2:-2].split()
            for i in range(0, len(s), 2):
                if s[i] == b'c':
                    rgb = s[i + 1]
                    if rgb == b'None':
                        self.info['transparency'] = c
                    elif rgb[:1] == b'#':
                        rgb = int(rgb[1:], 16)
                        palette[c] = o8(rgb >> 16 & 255) + o8(rgb >> 8 & 255) + o8(rgb & 255)
                    else:
                        msg = 'cannot read this XPM file'
                        raise ValueError(msg)
                    break
            else:
                msg = 'cannot read this XPM file'
                raise ValueError(msg)
        self._mode = 'P'
        self.palette = ImagePalette.raw('RGB', b''.join(palette))
        self.tile = [('raw', (0, 0) + self.size, self.fp.tell(), ('P', 0, 1))]

    def load_read(self, bytes):
        if False:
            print('Hello World!')
        (xsize, ysize) = self.size
        s = [None] * ysize
        for i in range(ysize):
            s[i] = self.fp.readline()[1:xsize + 1].ljust(xsize)
        return b''.join(s)
Image.register_open(XpmImageFile.format, XpmImageFile, _accept)
Image.register_extension(XpmImageFile.format, '.xpm')
Image.register_mime(XpmImageFile.format, 'image/xpm')