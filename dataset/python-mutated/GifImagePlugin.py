import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16

class LoadingStrategy(IntEnum):
    """.. versionadded:: 9.1.0"""
    RGB_AFTER_FIRST = 0
    RGB_AFTER_DIFFERENT_PALETTE_ONLY = 1
    RGB_ALWAYS = 2
LOADING_STRATEGY = LoadingStrategy.RGB_AFTER_FIRST

def _accept(prefix):
    if False:
        while True:
            i = 10
    return prefix[:6] in [b'GIF87a', b'GIF89a']

class GifImageFile(ImageFile.ImageFile):
    format = 'GIF'
    format_description = 'Compuserve GIF'
    _close_exclusive_fp_after_loading = False
    global_palette = None

    def data(self):
        if False:
            return 10
        s = self.fp.read(1)
        if s and s[0]:
            return self.fp.read(s[0])
        return None

    def _is_palette_needed(self, p):
        if False:
            while True:
                i = 10
        for i in range(0, len(p), 3):
            if not i // 3 == p[i] == p[i + 1] == p[i + 2]:
                return True
        return False

    def _open(self):
        if False:
            i = 10
            return i + 15
        s = self.fp.read(13)
        if not _accept(s):
            msg = 'not a GIF file'
            raise SyntaxError(msg)
        self.info['version'] = s[:6]
        self._size = (i16(s, 6), i16(s, 8))
        self.tile = []
        flags = s[10]
        bits = (flags & 7) + 1
        if flags & 128:
            self.info['background'] = s[11]
            p = self.fp.read(3 << bits)
            if self._is_palette_needed(p):
                p = ImagePalette.raw('RGB', p)
                self.global_palette = self.palette = p
        self._fp = self.fp
        self.__rewind = self.fp.tell()
        self._n_frames = None
        self._is_animated = None
        self._seek(0)

    @property
    def n_frames(self):
        if False:
            return 10
        if self._n_frames is None:
            current = self.tell()
            try:
                while True:
                    self._seek(self.tell() + 1, False)
            except EOFError:
                self._n_frames = self.tell() + 1
            self.seek(current)
        return self._n_frames

    @property
    def is_animated(self):
        if False:
            return 10
        if self._is_animated is None:
            if self._n_frames is not None:
                self._is_animated = self._n_frames != 1
            else:
                current = self.tell()
                if current:
                    self._is_animated = True
                else:
                    try:
                        self._seek(1, False)
                        self._is_animated = True
                    except EOFError:
                        self._is_animated = False
                    self.seek(current)
        return self._is_animated

    def seek(self, frame):
        if False:
            print('Hello World!')
        if not self._seek_check(frame):
            return
        if frame < self.__frame:
            self.im = None
            self._seek(0)
        last_frame = self.__frame
        for f in range(self.__frame + 1, frame + 1):
            try:
                self._seek(f)
            except EOFError as e:
                self.seek(last_frame)
                msg = 'no more images in GIF file'
                raise EOFError(msg) from e

    def _seek(self, frame, update_image=True):
        if False:
            i = 10
            return i + 15
        if frame == 0:
            self.__offset = 0
            self.dispose = None
            self.__frame = -1
            self._fp.seek(self.__rewind)
            self.disposal_method = 0
            if 'comment' in self.info:
                del self.info['comment']
        elif self.tile and update_image:
            self.load()
        if frame != self.__frame + 1:
            msg = f'cannot seek to frame {frame}'
            raise ValueError(msg)
        self.fp = self._fp
        if self.__offset:
            self.fp.seek(self.__offset)
            while self.data():
                pass
            self.__offset = 0
        s = self.fp.read(1)
        if not s or s == b';':
            msg = 'no more images in GIF file'
            raise EOFError(msg)
        palette = None
        info = {}
        frame_transparency = None
        interlace = None
        frame_dispose_extent = None
        while True:
            if not s:
                s = self.fp.read(1)
            if not s or s == b';':
                break
            elif s == b'!':
                s = self.fp.read(1)
                block = self.data()
                if s[0] == 249:
                    flags = block[0]
                    if flags & 1:
                        frame_transparency = block[3]
                    info['duration'] = i16(block, 1) * 10
                    dispose_bits = 28 & flags
                    dispose_bits = dispose_bits >> 2
                    if dispose_bits:
                        self.disposal_method = dispose_bits
                elif s[0] == 254:
                    comment = b''
                    while block:
                        comment += block
                        block = self.data()
                    if 'comment' in info:
                        info['comment'] += b'\n' + comment
                    else:
                        info['comment'] = comment
                    s = None
                    continue
                elif s[0] == 255 and frame == 0:
                    info['extension'] = (block, self.fp.tell())
                    if block[:11] == b'NETSCAPE2.0':
                        block = self.data()
                        if len(block) >= 3 and block[0] == 1:
                            self.info['loop'] = i16(block, 1)
                while self.data():
                    pass
            elif s == b',':
                s = self.fp.read(9)
                (x0, y0) = (i16(s, 0), i16(s, 2))
                (x1, y1) = (x0 + i16(s, 4), y0 + i16(s, 6))
                if (x1 > self.size[0] or y1 > self.size[1]) and update_image:
                    self._size = (max(x1, self.size[0]), max(y1, self.size[1]))
                    Image._decompression_bomb_check(self._size)
                frame_dispose_extent = (x0, y0, x1, y1)
                flags = s[8]
                interlace = flags & 64 != 0
                if flags & 128:
                    bits = (flags & 7) + 1
                    p = self.fp.read(3 << bits)
                    if self._is_palette_needed(p):
                        palette = ImagePalette.raw('RGB', p)
                    else:
                        palette = False
                bits = self.fp.read(1)[0]
                self.__offset = self.fp.tell()
                break
            s = None
        if interlace is None:
            msg = 'image not found in GIF frame'
            raise EOFError(msg)
        self.__frame = frame
        if not update_image:
            return
        self.tile = []
        if self.dispose:
            self.im.paste(self.dispose, self.dispose_extent)
        self._frame_palette = palette if palette is not None else self.global_palette
        self._frame_transparency = frame_transparency
        if frame == 0:
            if self._frame_palette:
                if LOADING_STRATEGY == LoadingStrategy.RGB_ALWAYS:
                    self._mode = 'RGBA' if frame_transparency is not None else 'RGB'
                else:
                    self._mode = 'P'
            else:
                self._mode = 'L'
            if not palette and self.global_palette:
                from copy import copy
                palette = copy(self.global_palette)
            self.palette = palette
        elif self.mode == 'P':
            if LOADING_STRATEGY != LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY or palette:
                self.pyaccess = None
                if 'transparency' in self.info:
                    self.im.putpalettealpha(self.info['transparency'], 0)
                    self.im = self.im.convert('RGBA', Image.Dither.FLOYDSTEINBERG)
                    self._mode = 'RGBA'
                    del self.info['transparency']
                else:
                    self._mode = 'RGB'
                    self.im = self.im.convert('RGB', Image.Dither.FLOYDSTEINBERG)

        def _rgb(color):
            if False:
                while True:
                    i = 10
            if self._frame_palette:
                color = tuple(self._frame_palette.palette[color * 3:color * 3 + 3])
            else:
                color = (color, color, color)
            return color
        self.dispose_extent = frame_dispose_extent
        try:
            if self.disposal_method < 2:
                self.dispose = None
            elif self.disposal_method == 2:
                (x0, y0, x1, y1) = self.dispose_extent
                dispose_size = (x1 - x0, y1 - y0)
                Image._decompression_bomb_check(dispose_size)
                dispose_mode = 'P'
                color = self.info.get('transparency', frame_transparency)
                if color is not None:
                    if self.mode in ('RGB', 'RGBA'):
                        dispose_mode = 'RGBA'
                        color = _rgb(color) + (0,)
                else:
                    color = self.info.get('background', 0)
                    if self.mode in ('RGB', 'RGBA'):
                        dispose_mode = 'RGB'
                        color = _rgb(color)
                self.dispose = Image.core.fill(dispose_mode, dispose_size, color)
            elif self.im is not None:
                self.dispose = self._crop(self.im, self.dispose_extent)
            elif frame_transparency is not None:
                (x0, y0, x1, y1) = self.dispose_extent
                dispose_size = (x1 - x0, y1 - y0)
                Image._decompression_bomb_check(dispose_size)
                dispose_mode = 'P'
                color = frame_transparency
                if self.mode in ('RGB', 'RGBA'):
                    dispose_mode = 'RGBA'
                    color = _rgb(frame_transparency) + (0,)
                self.dispose = Image.core.fill(dispose_mode, dispose_size, color)
        except AttributeError:
            pass
        if interlace is not None:
            transparency = -1
            if frame_transparency is not None:
                if frame == 0:
                    if LOADING_STRATEGY != LoadingStrategy.RGB_ALWAYS:
                        self.info['transparency'] = frame_transparency
                elif self.mode not in ('RGB', 'RGBA'):
                    transparency = frame_transparency
            self.tile = [('gif', (x0, y0, x1, y1), self.__offset, (bits, interlace, transparency))]
        if info.get('comment'):
            self.info['comment'] = info['comment']
        for k in ['duration', 'extension']:
            if k in info:
                self.info[k] = info[k]
            elif k in self.info:
                del self.info[k]

    def load_prepare(self):
        if False:
            while True:
                i = 10
        temp_mode = 'P' if self._frame_palette else 'L'
        self._prev_im = None
        if self.__frame == 0:
            if self._frame_transparency is not None:
                self.im = Image.core.fill(temp_mode, self.size, self._frame_transparency)
        elif self.mode in ('RGB', 'RGBA'):
            self._prev_im = self.im
            if self._frame_palette:
                self.im = Image.core.fill('P', self.size, self._frame_transparency or 0)
                self.im.putpalette(*self._frame_palette.getdata())
            else:
                self.im = None
        self._mode = temp_mode
        self._frame_palette = None
        super().load_prepare()

    def load_end(self):
        if False:
            while True:
                i = 10
        if self.__frame == 0:
            if self.mode == 'P' and LOADING_STRATEGY == LoadingStrategy.RGB_ALWAYS:
                if self._frame_transparency is not None:
                    self.im.putpalettealpha(self._frame_transparency, 0)
                    self._mode = 'RGBA'
                else:
                    self._mode = 'RGB'
                self.im = self.im.convert(self.mode, Image.Dither.FLOYDSTEINBERG)
            return
        if not self._prev_im:
            return
        if self._frame_transparency is not None:
            self.im.putpalettealpha(self._frame_transparency, 0)
            frame_im = self.im.convert('RGBA')
        else:
            frame_im = self.im.convert('RGB')
        frame_im = self._crop(frame_im, self.dispose_extent)
        self.im = self._prev_im
        self._mode = self.im.mode
        if frame_im.mode == 'RGBA':
            self.im.paste(frame_im, self.dispose_extent, frame_im)
        else:
            self.im.paste(frame_im, self.dispose_extent)

    def tell(self):
        if False:
            while True:
                i = 10
        return self.__frame
RAWMODE = {'1': 'L', 'L': 'L', 'P': 'P'}

def _normalize_mode(im):
    if False:
        i = 10
        return i + 15
    "\n    Takes an image (or frame), returns an image in a mode that is appropriate\n    for saving in a Gif.\n\n    It may return the original image, or it may return an image converted to\n    palette or 'L' mode.\n\n    :param im: Image object\n    :returns: Image object\n    "
    if im.mode in RAWMODE:
        im.load()
        return im
    if Image.getmodebase(im.mode) == 'RGB':
        im = im.convert('P', palette=Image.Palette.ADAPTIVE)
        if im.palette.mode == 'RGBA':
            for rgba in im.palette.colors:
                if rgba[3] == 0:
                    im.info['transparency'] = im.palette.colors[rgba]
                    break
        return im
    return im.convert('L')

def _normalize_palette(im, palette, info):
    if False:
        print('Hello World!')
    "\n    Normalizes the palette for image.\n      - Sets the palette to the incoming palette, if provided.\n      - Ensures that there's a palette for L mode images\n      - Optimizes the palette if necessary/desired.\n\n    :param im: Image object\n    :param palette: bytes object containing the source palette, or ....\n    :param info: encoderinfo\n    :returns: Image object\n    "
    source_palette = None
    if palette:
        if isinstance(palette, (bytes, bytearray, list)):
            source_palette = bytearray(palette[:768])
        if isinstance(palette, ImagePalette.ImagePalette):
            source_palette = bytearray(palette.palette)
    if im.mode == 'P':
        if not source_palette:
            source_palette = im.im.getpalette('RGB')[:768]
    else:
        if not source_palette:
            source_palette = bytearray((i // 3 for i in range(768)))
        im.palette = ImagePalette.ImagePalette('RGB', palette=source_palette)
    if palette:
        used_palette_colors = []
        for i in range(0, len(source_palette), 3):
            source_color = tuple(source_palette[i:i + 3])
            index = im.palette.colors.get(source_color)
            if index in used_palette_colors:
                index = None
            used_palette_colors.append(index)
        for (i, index) in enumerate(used_palette_colors):
            if index is None:
                for j in range(len(used_palette_colors)):
                    if j not in used_palette_colors:
                        used_palette_colors[i] = j
                        break
        im = im.remap_palette(used_palette_colors)
    else:
        used_palette_colors = _get_optimize(im, info)
        if used_palette_colors is not None:
            return im.remap_palette(used_palette_colors, source_palette)
    im.palette.palette = source_palette
    return im

def _write_single_frame(im, fp, palette):
    if False:
        print('Hello World!')
    im_out = _normalize_mode(im)
    for (k, v) in im_out.info.items():
        im.encoderinfo.setdefault(k, v)
    im_out = _normalize_palette(im_out, palette, im.encoderinfo)
    for s in _get_global_header(im_out, im.encoderinfo):
        fp.write(s)
    flags = 0
    if get_interlace(im):
        flags = flags | 64
    _write_local_header(fp, im, (0, 0), flags)
    im_out.encoderconfig = (8, get_interlace(im))
    ImageFile._save(im_out, fp, [('gif', (0, 0) + im.size, 0, RAWMODE[im_out.mode])])
    fp.write(b'\x00')

def _getbbox(base_im, im_frame):
    if False:
        for i in range(10):
            print('nop')
    if _get_palette_bytes(im_frame) == _get_palette_bytes(base_im):
        delta = ImageChops.subtract_modulo(im_frame, base_im)
    else:
        delta = ImageChops.subtract_modulo(im_frame.convert('RGBA'), base_im.convert('RGBA'))
    return delta.getbbox(alpha_only=False)

def _write_multiple_frames(im, fp, palette):
    if False:
        return 10
    duration = im.encoderinfo.get('duration')
    disposal = im.encoderinfo.get('disposal', im.info.get('disposal'))
    im_frames = []
    frame_count = 0
    background_im = None
    for imSequence in itertools.chain([im], im.encoderinfo.get('append_images', [])):
        for im_frame in ImageSequence.Iterator(imSequence):
            im_frame = _normalize_mode(im_frame.copy())
            if frame_count == 0:
                for (k, v) in im_frame.info.items():
                    if k == 'transparency':
                        continue
                    im.encoderinfo.setdefault(k, v)
            encoderinfo = im.encoderinfo.copy()
            im_frame = _normalize_palette(im_frame, palette, encoderinfo)
            if 'transparency' in im_frame.info:
                encoderinfo.setdefault('transparency', im_frame.info['transparency'])
            if isinstance(duration, (list, tuple)):
                encoderinfo['duration'] = duration[frame_count]
            elif duration is None and 'duration' in im_frame.info:
                encoderinfo['duration'] = im_frame.info['duration']
            if isinstance(disposal, (list, tuple)):
                encoderinfo['disposal'] = disposal[frame_count]
            frame_count += 1
            if im_frames:
                previous = im_frames[-1]
                bbox = _getbbox(previous['im'], im_frame)
                if not bbox:
                    if encoderinfo.get('duration'):
                        previous['encoderinfo']['duration'] += encoderinfo['duration']
                    continue
                if encoderinfo.get('disposal') == 2:
                    if background_im is None:
                        color = im.encoderinfo.get('transparency', im.info.get('transparency', (0, 0, 0)))
                        background = _get_background(im_frame, color)
                        background_im = Image.new('P', im_frame.size, background)
                        background_im.putpalette(im_frames[0]['im'].palette)
                    bbox = _getbbox(background_im, im_frame)
            else:
                bbox = None
            im_frames.append({'im': im_frame, 'bbox': bbox, 'encoderinfo': encoderinfo})
    if len(im_frames) > 1:
        for frame_data in im_frames:
            im_frame = frame_data['im']
            if not frame_data['bbox']:
                for s in _get_global_header(im_frame, frame_data['encoderinfo']):
                    fp.write(s)
                offset = (0, 0)
            else:
                if not palette:
                    frame_data['encoderinfo']['include_color_table'] = True
                im_frame = im_frame.crop(frame_data['bbox'])
                offset = frame_data['bbox'][:2]
            _write_frame_data(fp, im_frame, offset, frame_data['encoderinfo'])
        return True
    elif 'duration' in im.encoderinfo and isinstance(im.encoderinfo['duration'], (list, tuple)):
        im.encoderinfo['duration'] = sum(im.encoderinfo['duration'])

def _save_all(im, fp, filename):
    if False:
        print('Hello World!')
    _save(im, fp, filename, save_all=True)

def _save(im, fp, filename, save_all=False):
    if False:
        i = 10
        return i + 15
    if 'palette' in im.encoderinfo or 'palette' in im.info:
        palette = im.encoderinfo.get('palette', im.info.get('palette'))
    else:
        palette = None
        im.encoderinfo['optimize'] = im.encoderinfo.get('optimize', True)
    if not save_all or not _write_multiple_frames(im, fp, palette):
        _write_single_frame(im, fp, palette)
    fp.write(b';')
    if hasattr(fp, 'flush'):
        fp.flush()

def get_interlace(im):
    if False:
        return 10
    interlace = im.encoderinfo.get('interlace', 1)
    if min(im.size) < 16:
        interlace = 0
    return interlace

def _write_local_header(fp, im, offset, flags):
    if False:
        while True:
            i = 10
    transparent_color_exists = False
    try:
        transparency = int(im.encoderinfo['transparency'])
    except (KeyError, ValueError):
        pass
    else:
        transparent_color_exists = True
        used_palette_colors = _get_optimize(im, im.encoderinfo)
        if used_palette_colors is not None:
            try:
                transparency = used_palette_colors.index(transparency)
            except ValueError:
                transparent_color_exists = False
    if 'duration' in im.encoderinfo:
        duration = int(im.encoderinfo['duration'] / 10)
    else:
        duration = 0
    disposal = int(im.encoderinfo.get('disposal', 0))
    if transparent_color_exists or duration != 0 or disposal:
        packed_flag = 1 if transparent_color_exists else 0
        packed_flag |= disposal << 2
        if not transparent_color_exists:
            transparency = 0
        fp.write(b'!' + o8(249) + o8(4) + o8(packed_flag) + o16(duration) + o8(transparency) + o8(0))
    include_color_table = im.encoderinfo.get('include_color_table')
    if include_color_table:
        palette_bytes = _get_palette_bytes(im)
        color_table_size = _get_color_table_size(palette_bytes)
        if color_table_size:
            flags = flags | 128
            flags = flags | color_table_size
    fp.write(b',' + o16(offset[0]) + o16(offset[1]) + o16(im.size[0]) + o16(im.size[1]) + o8(flags))
    if include_color_table and color_table_size:
        fp.write(_get_header_palette(palette_bytes))
    fp.write(o8(8))

def _save_netpbm(im, fp, filename):
    if False:
        while True:
            i = 10
    tempfile = im._dump()
    try:
        with open(filename, 'wb') as f:
            if im.mode != 'RGB':
                subprocess.check_call(['ppmtogif', tempfile], stdout=f, stderr=subprocess.DEVNULL)
            else:
                quant_cmd = ['ppmquant', '256', tempfile]
                togif_cmd = ['ppmtogif']
                quant_proc = subprocess.Popen(quant_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                togif_proc = subprocess.Popen(togif_cmd, stdin=quant_proc.stdout, stdout=f, stderr=subprocess.DEVNULL)
                quant_proc.stdout.close()
                retcode = quant_proc.wait()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, quant_cmd)
                retcode = togif_proc.wait()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, togif_cmd)
    finally:
        try:
            os.unlink(tempfile)
        except OSError:
            pass
_FORCE_OPTIMIZE = False

def _get_optimize(im, info):
    if False:
        for i in range(10):
            print('nop')
    '\n    Palette optimization is a potentially expensive operation.\n\n    This function determines if the palette should be optimized using\n    some heuristics, then returns the list of palette entries in use.\n\n    :param im: Image object\n    :param info: encoderinfo\n    :returns: list of indexes of palette entries in use, or None\n    '
    if im.mode in ('P', 'L') and info and info.get('optimize', 0):
        optimise = _FORCE_OPTIMIZE or im.mode == 'L'
        if optimise or im.width * im.height < 512 * 512:
            used_palette_colors = []
            for (i, count) in enumerate(im.histogram()):
                if count:
                    used_palette_colors.append(i)
            if optimise or max(used_palette_colors) >= len(used_palette_colors):
                return used_palette_colors
            num_palette_colors = len(im.palette.palette) // Image.getmodebands(im.palette.mode)
            current_palette_size = 1 << (num_palette_colors - 1).bit_length()
            if len(used_palette_colors) <= current_palette_size // 2 and current_palette_size > 2:
                return used_palette_colors

def _get_color_table_size(palette_bytes):
    if False:
        print('Hello World!')
    if not palette_bytes:
        return 0
    elif len(palette_bytes) < 9:
        return 1
    else:
        return math.ceil(math.log(len(palette_bytes) // 3, 2)) - 1

def _get_header_palette(palette_bytes):
    if False:
        i = 10
        return i + 15
    '\n    Returns the palette, null padded to the next power of 2 (*3) bytes\n    suitable for direct inclusion in the GIF header\n\n    :param palette_bytes: Unpadded palette bytes, in RGBRGB form\n    :returns: Null padded palette\n    '
    color_table_size = _get_color_table_size(palette_bytes)
    actual_target_size_diff = (2 << color_table_size) - len(palette_bytes) // 3
    if actual_target_size_diff > 0:
        palette_bytes += o8(0) * 3 * actual_target_size_diff
    return palette_bytes

def _get_palette_bytes(im):
    if False:
        while True:
            i = 10
    '\n    Gets the palette for inclusion in the gif header\n\n    :param im: Image object\n    :returns: Bytes, len<=768 suitable for inclusion in gif header\n    '
    return im.palette.palette if im.palette else b''

def _get_background(im, info_background):
    if False:
        while True:
            i = 10
    background = 0
    if info_background:
        if isinstance(info_background, tuple):
            try:
                background = im.palette.getcolor(info_background, im)
            except ValueError as e:
                if str(e) not in ('cannot allocate more than 256 colors', 'cannot add non-opaque RGBA color to RGB palette'):
                    raise
        else:
            background = info_background
    return background

def _get_global_header(im, info):
    if False:
        print('Hello World!')
    'Return a list of strings representing a GIF header'
    version = b'87a'
    if im.info.get('version') == b'89a' or (info and ('transparency' in info or info.get('loop') is not None or info.get('duration') or info.get('comment'))):
        version = b'89a'
    background = _get_background(im, info.get('background'))
    palette_bytes = _get_palette_bytes(im)
    color_table_size = _get_color_table_size(palette_bytes)
    header = [b'GIF' + version + o16(im.size[0]) + o16(im.size[1]), o8(color_table_size + 128), o8(background) + o8(0), _get_header_palette(palette_bytes)]
    if info.get('loop') is not None:
        header.append(b'!' + o8(255) + o8(11) + b'NETSCAPE2.0' + o8(3) + o8(1) + o16(info['loop']) + o8(0))
    if info.get('comment'):
        comment_block = b'!' + o8(254)
        comment = info['comment']
        if isinstance(comment, str):
            comment = comment.encode()
        for i in range(0, len(comment), 255):
            subblock = comment[i:i + 255]
            comment_block += o8(len(subblock)) + subblock
        comment_block += o8(0)
        header.append(comment_block)
    return header

def _write_frame_data(fp, im_frame, offset, params):
    if False:
        while True:
            i = 10
    try:
        im_frame.encoderinfo = params
        _write_local_header(fp, im_frame, offset, 0)
        ImageFile._save(im_frame, fp, [('gif', (0, 0) + im_frame.size, 0, RAWMODE[im_frame.mode])])
        fp.write(b'\x00')
    finally:
        del im_frame.encoderinfo

def getheader(im, palette=None, info=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Legacy Method to get Gif data from image.\n\n    Warning:: May modify image data.\n\n    :param im: Image object\n    :param palette: bytes object containing the source palette, or ....\n    :param info: encoderinfo\n    :returns: tuple of(list of header items, optimized palette)\n\n    '
    used_palette_colors = _get_optimize(im, info)
    if info is None:
        info = {}
    if 'background' not in info and 'background' in im.info:
        info['background'] = im.info['background']
    im_mod = _normalize_palette(im, palette, info)
    im.palette = im_mod.palette
    im.im = im_mod.im
    header = _get_global_header(im, info)
    return (header, used_palette_colors)

def getdata(im, offset=(0, 0), **params):
    if False:
        i = 10
        return i + 15
    '\n    Legacy Method\n\n    Return a list of strings representing this image.\n    The first string is a local image header, the rest contains\n    encoded image data.\n\n    To specify duration, add the time in milliseconds,\n    e.g. ``getdata(im_frame, duration=1000)``\n\n    :param im: Image object\n    :param offset: Tuple of (x, y) pixels. Defaults to (0, 0)\n    :param \\**params: e.g. duration or other encoder info parameters\n    :returns: List of bytes containing GIF encoded frame data\n\n    '

    class Collector:
        data = []

        def write(self, data):
            if False:
                print('Hello World!')
            self.data.append(data)
    im.load()
    fp = Collector()
    _write_frame_data(fp, im, offset, params)
    return fp.data
Image.register_open(GifImageFile.format, GifImageFile, _accept)
Image.register_save(GifImageFile.format, _save)
Image.register_save_all(GifImageFile.format, _save_all)
Image.register_extension(GifImageFile.format, '.gif')
Image.register_mime(GifImageFile.format, 'image/gif')