import os
from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32
BIT2MODE = {1: ('P', 'P;1'), 4: ('P', 'P;4'), 8: ('P', 'P'), 16: ('RGB', 'BGR;15'), 24: ('RGB', 'BGR'), 32: ('RGB', 'BGRX')}

def _accept(prefix):
    if False:
        while True:
            i = 10
    return prefix[:2] == b'BM'

def _dib_accept(prefix):
    if False:
        while True:
            i = 10
    return i32(prefix) in [12, 40, 64, 108, 124]

class BmpImageFile(ImageFile.ImageFile):
    """Image plugin for the Windows Bitmap format (BMP)"""
    format_description = 'Windows Bitmap'
    format = 'BMP'
    COMPRESSIONS = {'RAW': 0, 'RLE8': 1, 'RLE4': 2, 'BITFIELDS': 3, 'JPEG': 4, 'PNG': 5}
    for (k, v) in COMPRESSIONS.items():
        vars()[k] = v

    def _bitmap(self, header=0, offset=0):
        if False:
            i = 10
            return i + 15
        'Read relevant info about the BMP'
        (read, seek) = (self.fp.read, self.fp.seek)
        if header:
            seek(header)
        file_info = {'header_size': i32(read(4)), 'direction': -1}
        header_data = ImageFile._safe_read(self.fp, file_info['header_size'] - 4)
        if file_info['header_size'] == 12:
            file_info['width'] = i16(header_data, 0)
            file_info['height'] = i16(header_data, 2)
            file_info['planes'] = i16(header_data, 4)
            file_info['bits'] = i16(header_data, 6)
            file_info['compression'] = self.RAW
            file_info['palette_padding'] = 3
        elif file_info['header_size'] in (40, 64, 108, 124):
            file_info['y_flip'] = header_data[7] == 255
            file_info['direction'] = 1 if file_info['y_flip'] else -1
            file_info['width'] = i32(header_data, 0)
            file_info['height'] = i32(header_data, 4) if not file_info['y_flip'] else 2 ** 32 - i32(header_data, 4)
            file_info['planes'] = i16(header_data, 8)
            file_info['bits'] = i16(header_data, 10)
            file_info['compression'] = i32(header_data, 12)
            file_info['data_size'] = i32(header_data, 16)
            file_info['pixels_per_meter'] = (i32(header_data, 20), i32(header_data, 24))
            file_info['colors'] = i32(header_data, 28)
            file_info['palette_padding'] = 4
            self.info['dpi'] = tuple((x / 39.3701 for x in file_info['pixels_per_meter']))
            if file_info['compression'] == self.BITFIELDS:
                if len(header_data) >= 52:
                    for (idx, mask) in enumerate(['r_mask', 'g_mask', 'b_mask', 'a_mask']):
                        file_info[mask] = i32(header_data, 36 + idx * 4)
                else:
                    file_info['a_mask'] = 0
                    for mask in ['r_mask', 'g_mask', 'b_mask']:
                        file_info[mask] = i32(read(4))
                file_info['rgb_mask'] = (file_info['r_mask'], file_info['g_mask'], file_info['b_mask'])
                file_info['rgba_mask'] = (file_info['r_mask'], file_info['g_mask'], file_info['b_mask'], file_info['a_mask'])
        else:
            msg = f"Unsupported BMP header type ({file_info['header_size']})"
            raise OSError(msg)
        self._size = (file_info['width'], file_info['height'])
        file_info['colors'] = file_info['colors'] if file_info.get('colors', 0) else 1 << file_info['bits']
        if offset == 14 + file_info['header_size'] and file_info['bits'] <= 8:
            offset += 4 * file_info['colors']
        (self._mode, raw_mode) = BIT2MODE.get(file_info['bits'], (None, None))
        if self.mode is None:
            msg = f"Unsupported BMP pixel depth ({file_info['bits']})"
            raise OSError(msg)
        decoder_name = 'raw'
        if file_info['compression'] == self.BITFIELDS:
            SUPPORTED = {32: [(16711680, 65280, 255, 0), (4278190080, 16711680, 65280, 0), (4278190080, 16711680, 65280, 255), (255, 65280, 16711680, 4278190080), (16711680, 65280, 255, 4278190080), (0, 0, 0, 0)], 24: [(16711680, 65280, 255)], 16: [(63488, 2016, 31), (31744, 992, 31)]}
            MASK_MODES = {(32, (16711680, 65280, 255, 0)): 'BGRX', (32, (4278190080, 16711680, 65280, 0)): 'XBGR', (32, (4278190080, 16711680, 65280, 255)): 'ABGR', (32, (255, 65280, 16711680, 4278190080)): 'RGBA', (32, (16711680, 65280, 255, 4278190080)): 'BGRA', (32, (0, 0, 0, 0)): 'BGRA', (24, (16711680, 65280, 255)): 'BGR', (16, (63488, 2016, 31)): 'BGR;16', (16, (31744, 992, 31)): 'BGR;15'}
            if file_info['bits'] in SUPPORTED:
                if file_info['bits'] == 32 and file_info['rgba_mask'] in SUPPORTED[file_info['bits']]:
                    raw_mode = MASK_MODES[file_info['bits'], file_info['rgba_mask']]
                    self._mode = 'RGBA' if 'A' in raw_mode else self.mode
                elif file_info['bits'] in (24, 16) and file_info['rgb_mask'] in SUPPORTED[file_info['bits']]:
                    raw_mode = MASK_MODES[file_info['bits'], file_info['rgb_mask']]
                else:
                    msg = 'Unsupported BMP bitfields layout'
                    raise OSError(msg)
            else:
                msg = 'Unsupported BMP bitfields layout'
                raise OSError(msg)
        elif file_info['compression'] == self.RAW:
            if file_info['bits'] == 32 and header == 22:
                (raw_mode, self._mode) = ('BGRA', 'RGBA')
        elif file_info['compression'] in (self.RLE8, self.RLE4):
            decoder_name = 'bmp_rle'
        else:
            msg = f"Unsupported BMP compression ({file_info['compression']})"
            raise OSError(msg)
        if self.mode == 'P':
            if not 0 < file_info['colors'] <= 65536:
                msg = f"Unsupported BMP Palette size ({file_info['colors']})"
                raise OSError(msg)
            else:
                padding = file_info['palette_padding']
                palette = read(padding * file_info['colors'])
                grayscale = True
                indices = (0, 255) if file_info['colors'] == 2 else list(range(file_info['colors']))
                for (ind, val) in enumerate(indices):
                    rgb = palette[ind * padding:ind * padding + 3]
                    if rgb != o8(val) * 3:
                        grayscale = False
                if grayscale:
                    self._mode = '1' if file_info['colors'] == 2 else 'L'
                    raw_mode = self.mode
                else:
                    self._mode = 'P'
                    self.palette = ImagePalette.raw('BGRX' if padding == 4 else 'BGR', palette)
        self.info['compression'] = file_info['compression']
        args = [raw_mode]
        if decoder_name == 'bmp_rle':
            args.append(file_info['compression'] == self.RLE4)
        else:
            args.append(file_info['width'] * file_info['bits'] + 31 >> 3 & ~3)
        args.append(file_info['direction'])
        self.tile = [(decoder_name, (0, 0, file_info['width'], file_info['height']), offset or self.fp.tell(), tuple(args))]

    def _open(self):
        if False:
            i = 10
            return i + 15
        'Open file, check magic number and read header'
        head_data = self.fp.read(14)
        if not _accept(head_data):
            msg = 'Not a BMP file'
            raise SyntaxError(msg)
        offset = i32(head_data, 10)
        self._bitmap(offset=offset)

class BmpRleDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        if False:
            return 10
        rle4 = self.args[1]
        data = bytearray()
        x = 0
        while len(data) < self.state.xsize * self.state.ysize:
            pixels = self.fd.read(1)
            byte = self.fd.read(1)
            if not pixels or not byte:
                break
            num_pixels = pixels[0]
            if num_pixels:
                if x + num_pixels > self.state.xsize:
                    num_pixels = max(0, self.state.xsize - x)
                if rle4:
                    first_pixel = o8(byte[0] >> 4)
                    second_pixel = o8(byte[0] & 15)
                    for index in range(num_pixels):
                        if index % 2 == 0:
                            data += first_pixel
                        else:
                            data += second_pixel
                else:
                    data += byte * num_pixels
                x += num_pixels
            elif byte[0] == 0:
                while len(data) % self.state.xsize != 0:
                    data += b'\x00'
                x = 0
            elif byte[0] == 1:
                break
            elif byte[0] == 2:
                bytes_read = self.fd.read(2)
                if len(bytes_read) < 2:
                    break
                (right, up) = self.fd.read(2)
                data += b'\x00' * (right + up * self.state.xsize)
                x = len(data) % self.state.xsize
            else:
                if rle4:
                    byte_count = byte[0] // 2
                    bytes_read = self.fd.read(byte_count)
                    for byte_read in bytes_read:
                        data += o8(byte_read >> 4)
                        data += o8(byte_read & 15)
                else:
                    byte_count = byte[0]
                    bytes_read = self.fd.read(byte_count)
                    data += bytes_read
                if len(bytes_read) < byte_count:
                    break
                x += byte[0]
                if self.fd.tell() % 2 != 0:
                    self.fd.seek(1, os.SEEK_CUR)
        rawmode = 'L' if self.mode == 'L' else 'P'
        self.set_as_raw(bytes(data), (rawmode, 0, self.args[-1]))
        return (-1, 0)

class DibImageFile(BmpImageFile):
    format = 'DIB'
    format_description = 'Windows Bitmap'

    def _open(self):
        if False:
            for i in range(10):
                print('nop')
        self._bitmap()
SAVE = {'1': ('1', 1, 2), 'L': ('L', 8, 256), 'P': ('P', 8, 256), 'RGB': ('BGR', 24, 0), 'RGBA': ('BGRA', 32, 0)}

def _dib_save(im, fp, filename):
    if False:
        print('Hello World!')
    _save(im, fp, filename, False)

def _save(im, fp, filename, bitmap_header=True):
    if False:
        return 10
    try:
        (rawmode, bits, colors) = SAVE[im.mode]
    except KeyError as e:
        msg = f'cannot write mode {im.mode} as BMP'
        raise OSError(msg) from e
    info = im.encoderinfo
    dpi = info.get('dpi', (96, 96))
    ppm = tuple(map(lambda x: int(x * 39.3701 + 0.5), dpi))
    stride = (im.size[0] * bits + 7) // 8 + 3 & ~3
    header = 40
    image = stride * im.size[1]
    if im.mode == '1':
        palette = b''.join((o8(i) * 4 for i in (0, 255)))
    elif im.mode == 'L':
        palette = b''.join((o8(i) * 4 for i in range(256)))
    elif im.mode == 'P':
        palette = im.im.getpalette('RGB', 'BGRX')
        colors = len(palette) // 4
    else:
        palette = None
    if bitmap_header:
        offset = 14 + header + colors * 4
        file_size = offset + image
        if file_size > 2 ** 32 - 1:
            msg = 'File size is too large for the BMP format'
            raise ValueError(msg)
        fp.write(b'BM' + o32(file_size) + o32(0) + o32(offset))
    fp.write(o32(header) + o32(im.size[0]) + o32(im.size[1]) + o16(1) + o16(bits) + o32(0) + o32(image) + o32(ppm[0]) + o32(ppm[1]) + o32(colors) + o32(colors))
    fp.write(b'\x00' * (header - 40))
    if palette:
        fp.write(palette)
    ImageFile._save(im, fp, [('raw', (0, 0) + im.size, 0, (rawmode, stride, -1))])
Image.register_open(BmpImageFile.format, BmpImageFile, _accept)
Image.register_save(BmpImageFile.format, _save)
Image.register_extension(BmpImageFile.format, '.bmp')
Image.register_mime(BmpImageFile.format, 'image/bmp')
Image.register_decoder('bmp_rle', BmpRleDecoder)
Image.register_open(DibImageFile.format, DibImageFile, _dib_accept)
Image.register_save(DibImageFile.format, _dib_save)
Image.register_extension(DibImageFile.format, '.dib')
Image.register_mime(DibImageFile.format, 'image/bmp')