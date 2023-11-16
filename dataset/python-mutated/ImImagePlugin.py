import os
import re
from . import Image, ImageFile, ImagePalette
COMMENT = 'Comment'
DATE = 'Date'
EQUIPMENT = 'Digitalization equipment'
FRAMES = 'File size (no of images)'
LUT = 'Lut'
NAME = 'Name'
SCALE = 'Scale (x,y)'
SIZE = 'Image size (x*y)'
MODE = 'Image type'
TAGS = {COMMENT: 0, DATE: 0, EQUIPMENT: 0, FRAMES: 0, LUT: 0, NAME: 0, SCALE: 0, SIZE: 0, MODE: 0}
OPEN = {'0 1 image': ('1', '1'), 'L 1 image': ('1', '1'), 'Greyscale image': ('L', 'L'), 'Grayscale image': ('L', 'L'), 'RGB image': ('RGB', 'RGB;L'), 'RLB image': ('RGB', 'RLB'), 'RYB image': ('RGB', 'RLB'), 'B1 image': ('1', '1'), 'B2 image': ('P', 'P;2'), 'B4 image': ('P', 'P;4'), 'X 24 image': ('RGB', 'RGB'), 'L 32 S image': ('I', 'I;32'), 'L 32 F image': ('F', 'F;32'), 'RGB3 image': ('RGB', 'RGB;T'), 'RYB3 image': ('RGB', 'RYB;T'), 'LA image': ('LA', 'LA;L'), 'PA image': ('LA', 'PA;L'), 'RGBA image': ('RGBA', 'RGBA;L'), 'RGBX image': ('RGBX', 'RGBX;L'), 'CMYK image': ('CMYK', 'CMYK;L'), 'YCC image': ('YCbCr', 'YCbCr;L')}
for i in ['8', '8S', '16', '16S', '32', '32F']:
    OPEN[f'L {i} image'] = ('F', f'F;{i}')
    OPEN[f'L*{i} image'] = ('F', f'F;{i}')
for i in ['16', '16L', '16B']:
    OPEN[f'L {i} image'] = (f'I;{i}', f'I;{i}')
    OPEN[f'L*{i} image'] = (f'I;{i}', f'I;{i}')
for i in ['32S']:
    OPEN[f'L {i} image'] = ('I', f'I;{i}')
    OPEN[f'L*{i} image'] = ('I', f'I;{i}')
for i in range(2, 33):
    OPEN[f'L*{i} image'] = ('F', f'F;{i}')
split = re.compile(b'^([A-Za-z][^:]*):[ \\t]*(.*)[ \\t]*$')

def number(s):
    if False:
        i = 10
        return i + 15
    try:
        return int(s)
    except ValueError:
        return float(s)

class ImImageFile(ImageFile.ImageFile):
    format = 'IM'
    format_description = 'IFUNC Image Memory'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        if False:
            return 10
        if b'\n' not in self.fp.read(100):
            msg = 'not an IM file'
            raise SyntaxError(msg)
        self.fp.seek(0)
        n = 0
        self.info[MODE] = 'L'
        self.info[SIZE] = (512, 512)
        self.info[FRAMES] = 1
        self.rawmode = 'L'
        while True:
            s = self.fp.read(1)
            if s == b'\r':
                continue
            if not s or s == b'\x00' or s == b'\x1a':
                break
            s = s + self.fp.readline()
            if len(s) > 100:
                msg = 'not an IM file'
                raise SyntaxError(msg)
            if s[-2:] == b'\r\n':
                s = s[:-2]
            elif s[-1:] == b'\n':
                s = s[:-1]
            try:
                m = split.match(s)
            except re.error as e:
                msg = 'not an IM file'
                raise SyntaxError(msg) from e
            if m:
                (k, v) = m.group(1, 2)
                k = k.decode('latin-1', 'replace')
                v = v.decode('latin-1', 'replace')
                if k in [FRAMES, SCALE, SIZE]:
                    v = v.replace('*', ',')
                    v = tuple(map(number, v.split(',')))
                    if len(v) == 1:
                        v = v[0]
                elif k == MODE and v in OPEN:
                    (v, self.rawmode) = OPEN[v]
                if k == COMMENT:
                    if k in self.info:
                        self.info[k].append(v)
                    else:
                        self.info[k] = [v]
                else:
                    self.info[k] = v
                if k in TAGS:
                    n += 1
            else:
                msg = 'Syntax error in IM header: ' + s.decode('ascii', 'replace')
                raise SyntaxError(msg)
        if not n:
            msg = 'Not an IM file'
            raise SyntaxError(msg)
        self._size = self.info[SIZE]
        self._mode = self.info[MODE]
        while s and s[:1] != b'\x1a':
            s = self.fp.read(1)
        if not s:
            msg = 'File truncated'
            raise SyntaxError(msg)
        if LUT in self.info:
            palette = self.fp.read(768)
            greyscale = 1
            linear = 1
            for i in range(256):
                if palette[i] == palette[i + 256] == palette[i + 512]:
                    if palette[i] != i:
                        linear = 0
                else:
                    greyscale = 0
            if self.mode in ['L', 'LA', 'P', 'PA']:
                if greyscale:
                    if not linear:
                        self.lut = list(palette[:256])
                else:
                    if self.mode in ['L', 'P']:
                        self._mode = self.rawmode = 'P'
                    elif self.mode in ['LA', 'PA']:
                        self._mode = 'PA'
                        self.rawmode = 'PA;L'
                    self.palette = ImagePalette.raw('RGB;L', palette)
            elif self.mode == 'RGB':
                if not greyscale or not linear:
                    self.lut = list(palette)
        self.frame = 0
        self.__offset = offs = self.fp.tell()
        self._fp = self.fp
        if self.rawmode[:2] == 'F;':
            try:
                bits = int(self.rawmode[2:])
                if bits not in [8, 16, 32]:
                    self.tile = [('bit', (0, 0) + self.size, offs, (bits, 8, 3, 0, -1))]
                    return
            except ValueError:
                pass
        if self.rawmode in ['RGB;T', 'RYB;T']:
            size = self.size[0] * self.size[1]
            self.tile = [('raw', (0, 0) + self.size, offs, ('G', 0, -1)), ('raw', (0, 0) + self.size, offs + size, ('R', 0, -1)), ('raw', (0, 0) + self.size, offs + 2 * size, ('B', 0, -1))]
        else:
            self.tile = [('raw', (0, 0) + self.size, offs, (self.rawmode, 0, -1))]

    @property
    def n_frames(self):
        if False:
            return 10
        return self.info[FRAMES]

    @property
    def is_animated(self):
        if False:
            while True:
                i = 10
        return self.info[FRAMES] > 1

    def seek(self, frame):
        if False:
            i = 10
            return i + 15
        if not self._seek_check(frame):
            return
        self.frame = frame
        if self.mode == '1':
            bits = 1
        else:
            bits = 8 * len(self.mode)
        size = (self.size[0] * bits + 7) // 8 * self.size[1]
        offs = self.__offset + frame * size
        self.fp = self._fp
        self.tile = [('raw', (0, 0) + self.size, offs, (self.rawmode, 0, -1))]

    def tell(self):
        if False:
            return 10
        return self.frame
SAVE = {'1': ('0 1', '1'), 'L': ('Greyscale', 'L'), 'LA': ('LA', 'LA;L'), 'P': ('Greyscale', 'P'), 'PA': ('LA', 'PA;L'), 'I': ('L 32S', 'I;32S'), 'I;16': ('L 16', 'I;16'), 'I;16L': ('L 16L', 'I;16L'), 'I;16B': ('L 16B', 'I;16B'), 'F': ('L 32F', 'F;32F'), 'RGB': ('RGB', 'RGB;L'), 'RGBA': ('RGBA', 'RGBA;L'), 'RGBX': ('RGBX', 'RGBX;L'), 'CMYK': ('CMYK', 'CMYK;L'), 'YCbCr': ('YCC', 'YCbCr;L')}

def _save(im, fp, filename):
    if False:
        return 10
    try:
        (image_type, rawmode) = SAVE[im.mode]
    except KeyError as e:
        msg = f'Cannot save {im.mode} images as IM'
        raise ValueError(msg) from e
    frames = im.encoderinfo.get('frames', 1)
    fp.write(f'Image type: {image_type} image\r\n'.encode('ascii'))
    if filename:
        (name, ext) = os.path.splitext(os.path.basename(filename))
        name = ''.join([name[:92 - len(ext)], ext])
        fp.write(f'Name: {name}\r\n'.encode('ascii'))
    fp.write(('Image size (x*y): %d*%d\r\n' % im.size).encode('ascii'))
    fp.write(f'File size (no of images): {frames}\r\n'.encode('ascii'))
    if im.mode in ['P', 'PA']:
        fp.write(b'Lut: 1\r\n')
    fp.write(b'\x00' * (511 - fp.tell()) + b'\x1a')
    if im.mode in ['P', 'PA']:
        im_palette = im.im.getpalette('RGB', 'RGB;L')
        colors = len(im_palette) // 3
        palette = b''
        for i in range(3):
            palette += im_palette[colors * i:colors * (i + 1)]
            palette += b'\x00' * (256 - colors)
        fp.write(palette)
    ImageFile._save(im, fp, [('raw', (0, 0) + im.size, 0, (rawmode, 0, -1))])
Image.register_open(ImImageFile.format, ImImageFile)
Image.register_save(ImImageFile.format, _save)
Image.register_extension(ImImageFile.format, '.im')