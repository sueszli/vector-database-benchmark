import io
import struct
from . import Image, ImageFile
from ._binary import i16le as i16
from ._binary import o16le as o16

def _accept(prefix):
    if False:
        while True:
            i = 10
    return prefix[:4] in [b'DanM', b'LinS']

class MspImageFile(ImageFile.ImageFile):
    format = 'MSP'
    format_description = 'Windows Paint'

    def _open(self):
        if False:
            print('Hello World!')
        s = self.fp.read(32)
        if not _accept(s):
            msg = 'not an MSP file'
            raise SyntaxError(msg)
        checksum = 0
        for i in range(0, 32, 2):
            checksum = checksum ^ i16(s, i)
        if checksum != 0:
            msg = 'bad MSP checksum'
            raise SyntaxError(msg)
        self._mode = '1'
        self._size = (i16(s, 4), i16(s, 6))
        if s[:4] == b'DanM':
            self.tile = [('raw', (0, 0) + self.size, 32, ('1', 0, 1))]
        else:
            self.tile = [('MSP', (0, 0) + self.size, 32, None)]

class MspDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        if False:
            return 10
        img = io.BytesIO()
        blank_line = bytearray((255,) * ((self.state.xsize + 7) // 8))
        try:
            self.fd.seek(32)
            rowmap = struct.unpack_from(f'<{self.state.ysize}H', self.fd.read(self.state.ysize * 2))
        except struct.error as e:
            msg = 'Truncated MSP file in row map'
            raise OSError(msg) from e
        for (x, rowlen) in enumerate(rowmap):
            try:
                if rowlen == 0:
                    img.write(blank_line)
                    continue
                row = self.fd.read(rowlen)
                if len(row) != rowlen:
                    msg = f'Truncated MSP file, expected {rowlen} bytes on row {x}'
                    raise OSError(msg)
                idx = 0
                while idx < rowlen:
                    runtype = row[idx]
                    idx += 1
                    if runtype == 0:
                        (runcount, runval) = struct.unpack_from('Bc', row, idx)
                        img.write(runval * runcount)
                        idx += 2
                    else:
                        runcount = runtype
                        img.write(row[idx:idx + runcount])
                        idx += runcount
            except struct.error as e:
                msg = f'Corrupted MSP file in row {x}'
                raise OSError(msg) from e
        self.set_as_raw(img.getvalue(), ('1', 0, 1))
        return (-1, 0)
Image.register_decoder('MSP', MspDecoder)

def _save(im, fp, filename):
    if False:
        return 10
    if im.mode != '1':
        msg = f'cannot write mode {im.mode} as MSP'
        raise OSError(msg)
    header = [0] * 16
    (header[0], header[1]) = (i16(b'Da'), i16(b'nM'))
    (header[2], header[3]) = im.size
    (header[4], header[5]) = (1, 1)
    (header[6], header[7]) = (1, 1)
    (header[8], header[9]) = im.size
    checksum = 0
    for h in header:
        checksum = checksum ^ h
    header[12] = checksum
    for h in header:
        fp.write(o16(h))
    ImageFile._save(im, fp, [('raw', (0, 0) + im.size, 32, ('1', 0, 1))])
Image.register_open(MspImageFile.format, MspImageFile, _accept)
Image.register_save(MspImageFile.format, _save)
Image.register_extension(MspImageFile.format, '.msp')