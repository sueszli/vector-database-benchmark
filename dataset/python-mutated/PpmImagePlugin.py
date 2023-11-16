from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
b_whitespace = b' \t\n\x0b\x0c\r'
MODES = {b'P1': '1', b'P2': 'L', b'P3': 'RGB', b'P4': '1', b'P5': 'L', b'P6': 'RGB', b'P0CMYK': 'CMYK', b'PyP': 'P', b'PyRGBA': 'RGBA', b'PyCMYK': 'CMYK'}

def _accept(prefix):
    if False:
        i = 10
        return i + 15
    return prefix[0:1] == b'P' and prefix[1] in b'0123456y'

class PpmImageFile(ImageFile.ImageFile):
    format = 'PPM'
    format_description = 'Pbmplus image'

    def _read_magic(self):
        if False:
            i = 10
            return i + 15
        magic = b''
        for _ in range(6):
            c = self.fp.read(1)
            if not c or c in b_whitespace:
                break
            magic += c
        return magic

    def _read_token(self):
        if False:
            for i in range(10):
                print('nop')
        token = b''
        while len(token) <= 10:
            c = self.fp.read(1)
            if not c:
                break
            elif c in b_whitespace:
                if not token:
                    continue
                break
            elif c == b'#':
                while self.fp.read(1) not in b'\r\n':
                    pass
                continue
            token += c
        if not token:
            msg = 'Reached EOF while reading header'
            raise ValueError(msg)
        elif len(token) > 10:
            msg = f'Token too long in file header: {token.decode()}'
            raise ValueError(msg)
        return token

    def _open(self):
        if False:
            print('Hello World!')
        magic_number = self._read_magic()
        try:
            mode = MODES[magic_number]
        except KeyError:
            msg = 'not a PPM file'
            raise SyntaxError(msg)
        if magic_number in (b'P1', b'P4'):
            self.custom_mimetype = 'image/x-portable-bitmap'
        elif magic_number in (b'P2', b'P5'):
            self.custom_mimetype = 'image/x-portable-graymap'
        elif magic_number in (b'P3', b'P6'):
            self.custom_mimetype = 'image/x-portable-pixmap'
        maxval = None
        decoder_name = 'raw'
        if magic_number in (b'P1', b'P2', b'P3'):
            decoder_name = 'ppm_plain'
        for ix in range(3):
            token = int(self._read_token())
            if ix == 0:
                xsize = token
            elif ix == 1:
                ysize = token
                if mode == '1':
                    self._mode = '1'
                    rawmode = '1;I'
                    break
                else:
                    self._mode = rawmode = mode
            elif ix == 2:
                maxval = token
                if not 0 < maxval < 65536:
                    msg = 'maxval must be greater than 0 and less than 65536'
                    raise ValueError(msg)
                if maxval > 255 and mode == 'L':
                    self._mode = 'I'
                if decoder_name != 'ppm_plain':
                    if maxval == 65535 and mode == 'L':
                        rawmode = 'I;16B'
                    elif maxval != 255:
                        decoder_name = 'ppm'
        args = (rawmode, 0, 1) if decoder_name == 'raw' else (rawmode, maxval)
        self._size = (xsize, ysize)
        self.tile = [(decoder_name, (0, 0, xsize, ysize), self.fp.tell(), args)]

class PpmPlainDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def _read_block(self):
        if False:
            while True:
                i = 10
        return self.fd.read(ImageFile.SAFEBLOCK)

    def _find_comment_end(self, block, start=0):
        if False:
            return 10
        a = block.find(b'\n', start)
        b = block.find(b'\r', start)
        return min(a, b) if a * b > 0 else max(a, b)

    def _ignore_comments(self, block):
        if False:
            for i in range(10):
                print('nop')
        if self._comment_spans:
            while block:
                comment_end = self._find_comment_end(block)
                if comment_end != -1:
                    block = block[comment_end + 1:]
                    break
                else:
                    block = self._read_block()
        self._comment_spans = False
        while True:
            comment_start = block.find(b'#')
            if comment_start == -1:
                break
            comment_end = self._find_comment_end(block, comment_start)
            if comment_end != -1:
                block = block[:comment_start] + block[comment_end + 1:]
            else:
                block = block[:comment_start]
                self._comment_spans = True
                break
        return block

    def _decode_bitonal(self):
        if False:
            print('Hello World!')
        '\n        This is a separate method because in the plain PBM format, all data tokens are\n        exactly one byte, so the inter-token whitespace is optional.\n        '
        data = bytearray()
        total_bytes = self.state.xsize * self.state.ysize
        while len(data) != total_bytes:
            block = self._read_block()
            if not block:
                break
            block = self._ignore_comments(block)
            tokens = b''.join(block.split())
            for token in tokens:
                if token not in (48, 49):
                    msg = b'Invalid token for this mode: %s' % bytes([token])
                    raise ValueError(msg)
            data = (data + tokens)[:total_bytes]
        invert = bytes.maketrans(b'01', b'\xff\x00')
        return data.translate(invert)

    def _decode_blocks(self, maxval):
        if False:
            i = 10
            return i + 15
        data = bytearray()
        max_len = 10
        out_byte_count = 4 if self.mode == 'I' else 1
        out_max = 65535 if self.mode == 'I' else 255
        bands = Image.getmodebands(self.mode)
        total_bytes = self.state.xsize * self.state.ysize * bands * out_byte_count
        half_token = False
        while len(data) != total_bytes:
            block = self._read_block()
            if not block:
                if half_token:
                    block = bytearray(b' ')
                else:
                    break
            block = self._ignore_comments(block)
            if half_token:
                block = half_token + block
                half_token = False
            tokens = block.split()
            if block and (not block[-1:].isspace()):
                half_token = tokens.pop()
                if len(half_token) > max_len:
                    msg = b'Token too long found in data: %s' % half_token[:max_len + 1]
                    raise ValueError(msg)
            for token in tokens:
                if len(token) > max_len:
                    msg = b'Token too long found in data: %s' % token[:max_len + 1]
                    raise ValueError(msg)
                value = int(token)
                if value > maxval:
                    msg = f'Channel value too large for this mode: {value}'
                    raise ValueError(msg)
                value = round(value / maxval * out_max)
                data += o32(value) if self.mode == 'I' else o8(value)
                if len(data) == total_bytes:
                    break
        return data

    def decode(self, buffer):
        if False:
            i = 10
            return i + 15
        self._comment_spans = False
        if self.mode == '1':
            data = self._decode_bitonal()
            rawmode = '1;8'
        else:
            maxval = self.args[-1]
            data = self._decode_blocks(maxval)
            rawmode = 'I;32' if self.mode == 'I' else self.mode
        self.set_as_raw(bytes(data), rawmode)
        return (-1, 0)

class PpmDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        if False:
            for i in range(10):
                print('nop')
        data = bytearray()
        maxval = self.args[-1]
        in_byte_count = 1 if maxval < 256 else 2
        out_byte_count = 4 if self.mode == 'I' else 1
        out_max = 65535 if self.mode == 'I' else 255
        bands = Image.getmodebands(self.mode)
        while len(data) < self.state.xsize * self.state.ysize * bands * out_byte_count:
            pixels = self.fd.read(in_byte_count * bands)
            if len(pixels) < in_byte_count * bands:
                break
            for b in range(bands):
                value = pixels[b] if in_byte_count == 1 else i16(pixels, b * in_byte_count)
                value = min(out_max, round(value / maxval * out_max))
                data += o32(value) if self.mode == 'I' else o8(value)
        rawmode = 'I;32' if self.mode == 'I' else self.mode
        self.set_as_raw(bytes(data), rawmode)
        return (-1, 0)

def _save(im, fp, filename):
    if False:
        print('Hello World!')
    if im.mode == '1':
        (rawmode, head) = ('1;I', b'P4')
    elif im.mode == 'L':
        (rawmode, head) = ('L', b'P5')
    elif im.mode == 'I':
        (rawmode, head) = ('I;16B', b'P5')
    elif im.mode in ('RGB', 'RGBA'):
        (rawmode, head) = ('RGB', b'P6')
    else:
        msg = f'cannot write mode {im.mode} as PPM'
        raise OSError(msg)
    fp.write(head + b'\n%d %d\n' % im.size)
    if head == b'P6':
        fp.write(b'255\n')
    elif head == b'P5':
        if rawmode == 'L':
            fp.write(b'255\n')
        else:
            fp.write(b'65535\n')
    ImageFile._save(im, fp, [('raw', (0, 0) + im.size, 0, (rawmode, 0, 1))])
Image.register_open(PpmImageFile.format, PpmImageFile, _accept)
Image.register_save(PpmImageFile.format, _save)
Image.register_decoder('ppm', PpmDecoder)
Image.register_decoder('ppm_plain', PpmPlainDecoder)
Image.register_extensions(PpmImageFile.format, ['.pbm', '.pgm', '.ppm', '.pnm'])
Image.register_mime(PpmImageFile.format, 'image/x-portable-anymap')