from . import BmpImagePlugin, Image
from ._binary import i16le as i16
from ._binary import i32le as i32

def _accept(prefix):
    if False:
        print('Hello World!')
    return prefix[:4] == b'\x00\x00\x02\x00'

class CurImageFile(BmpImagePlugin.BmpImageFile):
    format = 'CUR'
    format_description = 'Windows Cursor'

    def _open(self):
        if False:
            return 10
        offset = self.fp.tell()
        s = self.fp.read(6)
        if not _accept(s):
            msg = 'not a CUR file'
            raise SyntaxError(msg)
        m = b''
        for i in range(i16(s, 4)):
            s = self.fp.read(16)
            if not m:
                m = s
            elif s[0] > m[0] and s[1] > m[1]:
                m = s
        if not m:
            msg = 'No cursors were found'
            raise TypeError(msg)
        self._bitmap(i32(m, 12) + offset)
        self._size = (self.size[0], self.size[1] // 2)
        (d, e, o, a) = self.tile[0]
        self.tile[0] = (d, (0, 0) + self.size, o, a)
        return
Image.register_open(CurImageFile.format, CurImageFile, _accept)
Image.register_extension(CurImageFile.format, '.cur')