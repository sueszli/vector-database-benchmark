import olefile
from . import Image, ImageFile
from ._binary import i32le as i32
MODES = {(32766,): ('A', 'L'), (65536,): ('L', 'L'), (98304, 98302): ('RGBA', 'LA'), (131072, 131073, 131074): ('RGB', 'YCC;P'), (163840, 163841, 163842, 163838): ('RGBA', 'YCCA;P'), (196608, 196609, 196610): ('RGB', 'RGB'), (229376, 229377, 229378, 229374): ('RGBA', 'RGBA')}

def _accept(prefix):
    if False:
        i = 10
        return i + 15
    return prefix[:8] == olefile.MAGIC

class FpxImageFile(ImageFile.ImageFile):
    format = 'FPX'
    format_description = 'FlashPix'

    def _open(self):
        if False:
            print('Hello World!')
        try:
            self.ole = olefile.OleFileIO(self.fp)
        except OSError as e:
            msg = 'not an FPX file; invalid OLE file'
            raise SyntaxError(msg) from e
        if self.ole.root.clsid != '56616700-C154-11CE-8553-00AA00A1F95B':
            msg = 'not an FPX file; bad root CLSID'
            raise SyntaxError(msg)
        self._open_index(1)

    def _open_index(self, index=1):
        if False:
            while True:
                i = 10
        prop = self.ole.getproperties([f'Data Object Store {index:06d}', '\x05Image Contents'])
        self._size = (prop[16777218], prop[16777219])
        size = max(self.size)
        i = 1
        while size > 64:
            size = size / 2
            i += 1
        self.maxid = i - 1
        id = self.maxid << 16
        s = prop[33554434 | id]
        colors = []
        bands = i32(s, 4)
        if bands > 4:
            msg = 'Invalid number of bands'
            raise OSError(msg)
        for i in range(bands):
            colors.append(i32(s, 8 + i * 4) & 2147483647)
        (self._mode, self.rawmode) = MODES[tuple(colors)]
        self.jpeg = {}
        for i in range(256):
            id = 50331649 | i << 16
            if id in prop:
                self.jpeg[i] = prop[id]
        self._open_subimage(1, self.maxid)

    def _open_subimage(self, index=1, subimage=0):
        if False:
            while True:
                i = 10
        stream = [f'Data Object Store {index:06d}', f'Resolution {subimage:04d}', 'Subimage 0000 Header']
        fp = self.ole.openstream(stream)
        fp.read(28)
        s = fp.read(36)
        size = (i32(s, 4), i32(s, 8))
        tilesize = (i32(s, 16), i32(s, 20))
        offset = i32(s, 28)
        length = i32(s, 32)
        if size != self.size:
            msg = 'subimage mismatch'
            raise OSError(msg)
        fp.seek(28 + offset)
        s = fp.read(i32(s, 12) * length)
        x = y = 0
        (xsize, ysize) = size
        (xtile, ytile) = tilesize
        self.tile = []
        for i in range(0, len(s), length):
            x1 = min(xsize, x + xtile)
            y1 = min(ysize, y + ytile)
            compression = i32(s, i + 8)
            if compression == 0:
                self.tile.append(('raw', (x, y, x1, y1), i32(s, i) + 28, (self.rawmode,)))
            elif compression == 1:
                self.tile.append(('fill', (x, y, x1, y1), i32(s, i) + 28, (self.rawmode, s[12:16])))
            elif compression == 2:
                internal_color_conversion = s[14]
                jpeg_tables = s[15]
                rawmode = self.rawmode
                if internal_color_conversion:
                    if rawmode == 'RGBA':
                        (jpegmode, rawmode) = ('YCbCrK', 'CMYK')
                    else:
                        jpegmode = None
                else:
                    jpegmode = rawmode
                self.tile.append(('jpeg', (x, y, x1, y1), i32(s, i) + 28, (rawmode, jpegmode)))
                if jpeg_tables:
                    self.tile_prefix = self.jpeg[jpeg_tables]
            else:
                msg = 'unknown/invalid compression'
                raise OSError(msg)
            x = x + xtile
            if x >= xsize:
                (x, y) = (0, y + ytile)
                if y >= ysize:
                    break
        self.stream = stream
        self.fp = None

    def load(self):
        if False:
            print('Hello World!')
        if not self.fp:
            self.fp = self.ole.openstream(self.stream[:2] + ['Subimage 0000 Data'])
        return ImageFile.ImageFile.load(self)

    def close(self):
        if False:
            return 10
        self.ole.close()
        super().close()

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        self.ole.close()
        super().__exit__()
Image.register_open(FpxImageFile.format, FpxImageFile, _accept)
Image.register_extension(FpxImageFile.format, '.fpx')