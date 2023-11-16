from . import Image, ImageFile
from ._binary import i8

class BitStream:

    def __init__(self, fp):
        if False:
            for i in range(10):
                print('nop')
        self.fp = fp
        self.bits = 0
        self.bitbuffer = 0

    def next(self):
        if False:
            print('Hello World!')
        return i8(self.fp.read(1))

    def peek(self, bits):
        if False:
            while True:
                i = 10
        while self.bits < bits:
            c = self.next()
            if c < 0:
                self.bits = 0
                continue
            self.bitbuffer = (self.bitbuffer << 8) + c
            self.bits += 8
        return self.bitbuffer >> self.bits - bits & (1 << bits) - 1

    def skip(self, bits):
        if False:
            i = 10
            return i + 15
        while self.bits < bits:
            self.bitbuffer = (self.bitbuffer << 8) + i8(self.fp.read(1))
            self.bits += 8
        self.bits = self.bits - bits

    def read(self, bits):
        if False:
            while True:
                i = 10
        v = self.peek(bits)
        self.bits = self.bits - bits
        return v

class MpegImageFile(ImageFile.ImageFile):
    format = 'MPEG'
    format_description = 'MPEG'

    def _open(self):
        if False:
            i = 10
            return i + 15
        s = BitStream(self.fp)
        if s.read(32) != 435:
            msg = 'not an MPEG file'
            raise SyntaxError(msg)
        self._mode = 'RGB'
        self._size = (s.read(12), s.read(12))
Image.register_open(MpegImageFile.format, MpegImageFile)
Image.register_extensions(MpegImageFile.format, ['.mpg', '.mpeg'])
Image.register_mime(MpegImageFile.format, 'video/mpeg')