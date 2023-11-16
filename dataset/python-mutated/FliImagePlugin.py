import os
from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8

def _accept(prefix):
    if False:
        while True:
            i = 10
    return len(prefix) >= 6 and i16(prefix, 4) in [44817, 44818] and (i16(prefix, 14) in [0, 3])

class FliImageFile(ImageFile.ImageFile):
    format = 'FLI'
    format_description = 'Autodesk FLI/FLC Animation'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        if False:
            return 10
        s = self.fp.read(128)
        if not (_accept(s) and s[20:22] == b'\x00\x00'):
            msg = 'not an FLI/FLC file'
            raise SyntaxError(msg)
        self.n_frames = i16(s, 6)
        self.is_animated = self.n_frames > 1
        self._mode = 'P'
        self._size = (i16(s, 8), i16(s, 10))
        duration = i32(s, 16)
        magic = i16(s, 4)
        if magic == 44817:
            duration = duration * 1000 // 70
        self.info['duration'] = duration
        palette = [(a, a, a) for a in range(256)]
        s = self.fp.read(16)
        self.__offset = 128
        if i16(s, 4) == 61696:
            self.__offset = self.__offset + i32(s)
            s = self.fp.read(16)
        if i16(s, 4) == 61946:
            number_of_subchunks = i16(s, 6)
            chunk_size = None
            for _ in range(number_of_subchunks):
                if chunk_size is not None:
                    self.fp.seek(chunk_size - 6, os.SEEK_CUR)
                s = self.fp.read(6)
                chunk_type = i16(s, 4)
                if chunk_type in (4, 11):
                    self._palette(palette, 2 if chunk_type == 11 else 0)
                    break
                chunk_size = i32(s)
                if not chunk_size:
                    break
        palette = [o8(r) + o8(g) + o8(b) for (r, g, b) in palette]
        self.palette = ImagePalette.raw('RGB', b''.join(palette))
        self.__frame = -1
        self._fp = self.fp
        self.__rewind = self.fp.tell()
        self.seek(0)

    def _palette(self, palette, shift):
        if False:
            return 10
        i = 0
        for e in range(i16(self.fp.read(2))):
            s = self.fp.read(2)
            i = i + s[0]
            n = s[1]
            if n == 0:
                n = 256
            s = self.fp.read(n * 3)
            for n in range(0, len(s), 3):
                r = s[n] << shift
                g = s[n + 1] << shift
                b = s[n + 2] << shift
                palette[i] = (r, g, b)
                i += 1

    def seek(self, frame):
        if False:
            return 10
        if not self._seek_check(frame):
            return
        if frame < self.__frame:
            self._seek(0)
        for f in range(self.__frame + 1, frame + 1):
            self._seek(f)

    def _seek(self, frame):
        if False:
            for i in range(10):
                print('nop')
        if frame == 0:
            self.__frame = -1
            self._fp.seek(self.__rewind)
            self.__offset = 128
        else:
            self.load()
        if frame != self.__frame + 1:
            msg = f'cannot seek to frame {frame}'
            raise ValueError(msg)
        self.__frame = frame
        self.fp = self._fp
        self.fp.seek(self.__offset)
        s = self.fp.read(4)
        if not s:
            msg = 'missing frame size'
            raise EOFError(msg)
        framesize = i32(s)
        self.decodermaxblock = framesize
        self.tile = [('fli', (0, 0) + self.size, self.__offset, None)]
        self.__offset += framesize

    def tell(self):
        if False:
            return 10
        return self.__frame
Image.register_open(FliImageFile.format, FliImageFile, _accept)
Image.register_extensions(FliImageFile.format, ['.fli', '.flc'])