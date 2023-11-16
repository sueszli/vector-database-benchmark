import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets

def Skip(self, marker):
    if False:
        print('Hello World!')
    n = i16(self.fp.read(2)) - 2
    ImageFile._safe_read(self.fp, n)

def APP(self, marker):
    if False:
        return 10
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    app = 'APP%d' % (marker & 15)
    self.app[app] = s
    self.applist.append((app, s))
    if marker == 65504 and s[:4] == b'JFIF':
        self.info['jfif'] = version = i16(s, 5)
        self.info['jfif_version'] = divmod(version, 256)
        try:
            jfif_unit = s[7]
            jfif_density = (i16(s, 8), i16(s, 10))
        except Exception:
            pass
        else:
            if jfif_unit == 1:
                self.info['dpi'] = jfif_density
            self.info['jfif_unit'] = jfif_unit
            self.info['jfif_density'] = jfif_density
    elif marker == 65505 and s[:5] == b'Exif\x00':
        if 'exif' not in self.info:
            self.info['exif'] = s
            self._exif_offset = self.fp.tell() - n + 6
    elif marker == 65506 and s[:5] == b'FPXR\x00':
        self.info['flashpix'] = s
    elif marker == 65506 and s[:12] == b'ICC_PROFILE\x00':
        self.icclist.append(s)
    elif marker == 65517 and s[:14] == b'Photoshop 3.0\x00':
        offset = 14
        photoshop = self.info.setdefault('photoshop', {})
        while s[offset:offset + 4] == b'8BIM':
            try:
                offset += 4
                code = i16(s, offset)
                offset += 2
                name_len = s[offset]
                offset += 1 + name_len
                offset += offset & 1
                size = i32(s, offset)
                offset += 4
                data = s[offset:offset + size]
                if code == 1005:
                    data = {'XResolution': i32(data, 0) / 65536, 'DisplayedUnitsX': i16(data, 4), 'YResolution': i32(data, 8) / 65536, 'DisplayedUnitsY': i16(data, 12)}
                photoshop[code] = data
                offset += size
                offset += offset & 1
            except struct.error:
                break
    elif marker == 65518 and s[:5] == b'Adobe':
        self.info['adobe'] = i16(s, 5)
        try:
            adobe_transform = s[11]
        except IndexError:
            pass
        else:
            self.info['adobe_transform'] = adobe_transform
    elif marker == 65506 and s[:4] == b'MPF\x00':
        self.info['mp'] = s[4:]
        self.info['mpoffset'] = self.fp.tell() - n + 4
    if 'dpi' not in self.info and 'exif' in self.info:
        try:
            exif = self.getexif()
            resolution_unit = exif[296]
            x_resolution = exif[282]
            try:
                dpi = float(x_resolution[0]) / x_resolution[1]
            except TypeError:
                dpi = x_resolution
            if math.isnan(dpi):
                msg = 'DPI is not a number'
                raise ValueError(msg)
            if resolution_unit == 3:
                dpi *= 2.54
            self.info['dpi'] = (dpi, dpi)
        except (struct.error, KeyError, SyntaxError, TypeError, ValueError, ZeroDivisionError):
            self.info['dpi'] = (72, 72)

def COM(self, marker):
    if False:
        print('Hello World!')
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    self.info['comment'] = s
    self.app['COM'] = s
    self.applist.append(('COM', s))

def SOF(self, marker):
    if False:
        for i in range(10):
            print('nop')
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    self._size = (i16(s, 3), i16(s, 1))
    self.bits = s[0]
    if self.bits != 8:
        msg = f'cannot handle {self.bits}-bit layers'
        raise SyntaxError(msg)
    self.layers = s[5]
    if self.layers == 1:
        self._mode = 'L'
    elif self.layers == 3:
        self._mode = 'RGB'
    elif self.layers == 4:
        self._mode = 'CMYK'
    else:
        msg = f'cannot handle {self.layers}-layer images'
        raise SyntaxError(msg)
    if marker in [65474, 65478, 65482, 65486]:
        self.info['progressive'] = self.info['progression'] = 1
    if self.icclist:
        self.icclist.sort()
        if self.icclist[0][13] == len(self.icclist):
            profile = []
            for p in self.icclist:
                profile.append(p[14:])
            icc_profile = b''.join(profile)
        else:
            icc_profile = None
        self.info['icc_profile'] = icc_profile
        self.icclist = []
    for i in range(6, len(s), 3):
        t = s[i:i + 3]
        self.layer.append((t[0], t[1] // 16, t[1] & 15, t[2]))

def DQT(self, marker):
    if False:
        for i in range(10):
            print('nop')
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    while len(s):
        v = s[0]
        precision = 1 if v // 16 == 0 else 2
        qt_length = 1 + precision * 64
        if len(s) < qt_length:
            msg = 'bad quantization table marker'
            raise SyntaxError(msg)
        data = array.array('B' if precision == 1 else 'H', s[1:qt_length])
        if sys.byteorder == 'little' and precision > 1:
            data.byteswap()
        self.quantization[v & 15] = [data[i] for i in zigzag_index]
        s = s[qt_length:]
MARKER = {65472: ('SOF0', 'Baseline DCT', SOF), 65473: ('SOF1', 'Extended Sequential DCT', SOF), 65474: ('SOF2', 'Progressive DCT', SOF), 65475: ('SOF3', 'Spatial lossless', SOF), 65476: ('DHT', 'Define Huffman table', Skip), 65477: ('SOF5', 'Differential sequential DCT', SOF), 65478: ('SOF6', 'Differential progressive DCT', SOF), 65479: ('SOF7', 'Differential spatial', SOF), 65480: ('JPG', 'Extension', None), 65481: ('SOF9', 'Extended sequential DCT (AC)', SOF), 65482: ('SOF10', 'Progressive DCT (AC)', SOF), 65483: ('SOF11', 'Spatial lossless DCT (AC)', SOF), 65484: ('DAC', 'Define arithmetic coding conditioning', Skip), 65485: ('SOF13', 'Differential sequential DCT (AC)', SOF), 65486: ('SOF14', 'Differential progressive DCT (AC)', SOF), 65487: ('SOF15', 'Differential spatial (AC)', SOF), 65488: ('RST0', 'Restart 0', None), 65489: ('RST1', 'Restart 1', None), 65490: ('RST2', 'Restart 2', None), 65491: ('RST3', 'Restart 3', None), 65492: ('RST4', 'Restart 4', None), 65493: ('RST5', 'Restart 5', None), 65494: ('RST6', 'Restart 6', None), 65495: ('RST7', 'Restart 7', None), 65496: ('SOI', 'Start of image', None), 65497: ('EOI', 'End of image', None), 65498: ('SOS', 'Start of scan', Skip), 65499: ('DQT', 'Define quantization table', DQT), 65500: ('DNL', 'Define number of lines', Skip), 65501: ('DRI', 'Define restart interval', Skip), 65502: ('DHP', 'Define hierarchical progression', SOF), 65503: ('EXP', 'Expand reference component', Skip), 65504: ('APP0', 'Application segment 0', APP), 65505: ('APP1', 'Application segment 1', APP), 65506: ('APP2', 'Application segment 2', APP), 65507: ('APP3', 'Application segment 3', APP), 65508: ('APP4', 'Application segment 4', APP), 65509: ('APP5', 'Application segment 5', APP), 65510: ('APP6', 'Application segment 6', APP), 65511: ('APP7', 'Application segment 7', APP), 65512: ('APP8', 'Application segment 8', APP), 65513: ('APP9', 'Application segment 9', APP), 65514: ('APP10', 'Application segment 10', APP), 65515: ('APP11', 'Application segment 11', APP), 65516: ('APP12', 'Application segment 12', APP), 65517: ('APP13', 'Application segment 13', APP), 65518: ('APP14', 'Application segment 14', APP), 65519: ('APP15', 'Application segment 15', APP), 65520: ('JPG0', 'Extension 0', None), 65521: ('JPG1', 'Extension 1', None), 65522: ('JPG2', 'Extension 2', None), 65523: ('JPG3', 'Extension 3', None), 65524: ('JPG4', 'Extension 4', None), 65525: ('JPG5', 'Extension 5', None), 65526: ('JPG6', 'Extension 6', None), 65527: ('JPG7', 'Extension 7', None), 65528: ('JPG8', 'Extension 8', None), 65529: ('JPG9', 'Extension 9', None), 65530: ('JPG10', 'Extension 10', None), 65531: ('JPG11', 'Extension 11', None), 65532: ('JPG12', 'Extension 12', None), 65533: ('JPG13', 'Extension 13', None), 65534: ('COM', 'Comment', COM)}

def _accept(prefix):
    if False:
        i = 10
        return i + 15
    return prefix[:3] == b'\xff\xd8\xff'

class JpegImageFile(ImageFile.ImageFile):
    format = 'JPEG'
    format_description = 'JPEG (ISO 10918)'

    def _open(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.fp.read(3)
        if not _accept(s):
            msg = 'not a JPEG file'
            raise SyntaxError(msg)
        s = b'\xff'
        self.bits = self.layers = 0
        self.layer = []
        self.huffman_dc = {}
        self.huffman_ac = {}
        self.quantization = {}
        self.app = {}
        self.applist = []
        self.icclist = []
        while True:
            i = s[0]
            if i == 255:
                s = s + self.fp.read(1)
                i = i16(s)
            else:
                s = self.fp.read(1)
                continue
            if i in MARKER:
                (name, description, handler) = MARKER[i]
                if handler is not None:
                    handler(self, i)
                if i == 65498:
                    rawmode = self.mode
                    if self.mode == 'CMYK':
                        rawmode = 'CMYK;I'
                    self.tile = [('jpeg', (0, 0) + self.size, 0, (rawmode, ''))]
                    break
                s = self.fp.read(1)
            elif i == 0 or i == 65535:
                s = b'\xff'
            elif i == 65280:
                s = self.fp.read(1)
            else:
                msg = 'no marker found'
                raise SyntaxError(msg)

    def load_read(self, read_bytes):
        if False:
            for i in range(10):
                print('nop')
        '\n        internal: read more image data\n        For premature EOF and LOAD_TRUNCATED_IMAGES adds EOI marker\n        so libjpeg can finish decoding\n        '
        s = self.fp.read(read_bytes)
        if not s and ImageFile.LOAD_TRUNCATED_IMAGES and (not hasattr(self, '_ended')):
            self._ended = True
            return b'\xff\xd9'
        return s

    def draft(self, mode, size):
        if False:
            for i in range(10):
                print('nop')
        if len(self.tile) != 1:
            return
        if self.decoderconfig:
            return
        (d, e, o, a) = self.tile[0]
        scale = 1
        original_size = self.size
        if a[0] == 'RGB' and mode in ['L', 'YCbCr']:
            self._mode = mode
            a = (mode, '')
        if size:
            scale = min(self.size[0] // size[0], self.size[1] // size[1])
            for s in [8, 4, 2, 1]:
                if scale >= s:
                    break
            e = (e[0], e[1], (e[2] - e[0] + s - 1) // s + e[0], (e[3] - e[1] + s - 1) // s + e[1])
            self._size = ((self.size[0] + s - 1) // s, (self.size[1] + s - 1) // s)
            scale = s
        self.tile = [(d, e, o, a)]
        self.decoderconfig = (scale, 0)
        box = (0, 0, original_size[0] / scale, original_size[1] / scale)
        return (self.mode, box)

    def load_djpeg(self):
        if False:
            for i in range(10):
                print('nop')
        (f, path) = tempfile.mkstemp()
        os.close(f)
        if os.path.exists(self.filename):
            subprocess.check_call(['djpeg', '-outfile', path, self.filename])
        else:
            try:
                os.unlink(path)
            except OSError:
                pass
            msg = 'Invalid Filename'
            raise ValueError(msg)
        try:
            with Image.open(path) as _im:
                _im.load()
                self.im = _im.im
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._mode = self.im.mode
        self._size = self.im.size
        self.tile = []

    def _getexif(self):
        if False:
            for i in range(10):
                print('nop')
        return _getexif(self)

    def _getmp(self):
        if False:
            print('Hello World!')
        return _getmp(self)

    def getxmp(self):
        if False:
            return 10
        '\n        Returns a dictionary containing the XMP tags.\n        Requires defusedxml to be installed.\n\n        :returns: XMP tags in a dictionary.\n        '
        for (segment, content) in self.applist:
            if segment == 'APP1':
                (marker, xmp_tags) = content.split(b'\x00')[:2]
                if marker == b'http://ns.adobe.com/xap/1.0/':
                    return self._getxmp(xmp_tags)
        return {}

def _getexif(self):
    if False:
        return 10
    if 'exif' not in self.info:
        return None
    return self.getexif()._get_merged_dict()

def _getmp(self):
    if False:
        return 10
    try:
        data = self.info['mp']
    except KeyError:
        return None
    file_contents = io.BytesIO(data)
    head = file_contents.read(8)
    endianness = '>' if head[:4] == b'MM\x00*' else '<'
    from . import TiffImagePlugin
    try:
        info = TiffImagePlugin.ImageFileDirectory_v2(head)
        file_contents.seek(info.next)
        info.load(file_contents)
        mp = dict(info)
    except Exception as e:
        msg = 'malformed MP Index (unreadable directory)'
        raise SyntaxError(msg) from e
    try:
        quant = mp[45057]
    except KeyError as e:
        msg = 'malformed MP Index (no number of images)'
        raise SyntaxError(msg) from e
    mpentries = []
    try:
        rawmpentries = mp[45058]
        for entrynum in range(0, quant):
            unpackedentry = struct.unpack_from(f'{endianness}LLLHH', rawmpentries, entrynum * 16)
            labels = ('Attribute', 'Size', 'DataOffset', 'EntryNo1', 'EntryNo2')
            mpentry = dict(zip(labels, unpackedentry))
            mpentryattr = {'DependentParentImageFlag': bool(mpentry['Attribute'] & 1 << 31), 'DependentChildImageFlag': bool(mpentry['Attribute'] & 1 << 30), 'RepresentativeImageFlag': bool(mpentry['Attribute'] & 1 << 29), 'Reserved': (mpentry['Attribute'] & 3 << 27) >> 27, 'ImageDataFormat': (mpentry['Attribute'] & 7 << 24) >> 24, 'MPType': mpentry['Attribute'] & 16777215}
            if mpentryattr['ImageDataFormat'] == 0:
                mpentryattr['ImageDataFormat'] = 'JPEG'
            else:
                msg = 'unsupported picture format in MPO'
                raise SyntaxError(msg)
            mptypemap = {0: 'Undefined', 65537: 'Large Thumbnail (VGA Equivalent)', 65538: 'Large Thumbnail (Full HD Equivalent)', 131073: 'Multi-Frame Image (Panorama)', 131074: 'Multi-Frame Image: (Disparity)', 131075: 'Multi-Frame Image: (Multi-Angle)', 196608: 'Baseline MP Primary Image'}
            mpentryattr['MPType'] = mptypemap.get(mpentryattr['MPType'], 'Unknown')
            mpentry['Attribute'] = mpentryattr
            mpentries.append(mpentry)
        mp[45058] = mpentries
    except KeyError as e:
        msg = 'malformed MP Index (bad MP Entry)'
        raise SyntaxError(msg) from e
    return mp
RAWMODE = {'1': 'L', 'L': 'L', 'RGB': 'RGB', 'RGBX': 'RGB', 'CMYK': 'CMYK;I', 'YCbCr': 'YCbCr'}
zigzag_index = (0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63)
samplings = {(1, 1, 1, 1, 1, 1): 0, (2, 1, 1, 1, 1, 1): 1, (2, 2, 1, 1, 1, 1): 2}

def get_sampling(im):
    if False:
        print('Hello World!')
    if not hasattr(im, 'layers') or im.layers in (1, 4):
        return -1
    sampling = im.layer[0][1:3] + im.layer[1][1:3] + im.layer[2][1:3]
    return samplings.get(sampling, -1)

def _save(im, fp, filename):
    if False:
        while True:
            i = 10
    if im.width == 0 or im.height == 0:
        msg = 'cannot write empty image as JPEG'
        raise ValueError(msg)
    try:
        rawmode = RAWMODE[im.mode]
    except KeyError as e:
        msg = f'cannot write mode {im.mode} as JPEG'
        raise OSError(msg) from e
    info = im.encoderinfo
    dpi = [round(x) for x in info.get('dpi', (0, 0))]
    quality = info.get('quality', -1)
    subsampling = info.get('subsampling', -1)
    qtables = info.get('qtables')
    if quality == 'keep':
        quality = -1
        subsampling = 'keep'
        qtables = 'keep'
    elif quality in presets:
        preset = presets[quality]
        quality = -1
        subsampling = preset.get('subsampling', -1)
        qtables = preset.get('quantization')
    elif not isinstance(quality, int):
        msg = 'Invalid quality setting'
        raise ValueError(msg)
    else:
        if subsampling in presets:
            subsampling = presets[subsampling].get('subsampling', -1)
        if isinstance(qtables, str) and qtables in presets:
            qtables = presets[qtables].get('quantization')
    if subsampling == '4:4:4':
        subsampling = 0
    elif subsampling == '4:2:2':
        subsampling = 1
    elif subsampling == '4:2:0':
        subsampling = 2
    elif subsampling == '4:1:1':
        subsampling = 2
    elif subsampling == 'keep':
        if im.format != 'JPEG':
            msg = "Cannot use 'keep' when original image is not a JPEG"
            raise ValueError(msg)
        subsampling = get_sampling(im)

    def validate_qtables(qtables):
        if False:
            print('Hello World!')
        if qtables is None:
            return qtables
        if isinstance(qtables, str):
            try:
                lines = [int(num) for line in qtables.splitlines() for num in line.split('#', 1)[0].split()]
            except ValueError as e:
                msg = 'Invalid quantization table'
                raise ValueError(msg) from e
            else:
                qtables = [lines[s:s + 64] for s in range(0, len(lines), 64)]
        if isinstance(qtables, (tuple, list, dict)):
            if isinstance(qtables, dict):
                qtables = [qtables[key] for key in range(len(qtables)) if key in qtables]
            elif isinstance(qtables, tuple):
                qtables = list(qtables)
            if not 0 < len(qtables) < 5:
                msg = 'None or too many quantization tables'
                raise ValueError(msg)
            for (idx, table) in enumerate(qtables):
                try:
                    if len(table) != 64:
                        msg = 'Invalid quantization table'
                        raise TypeError(msg)
                    table = array.array('H', table)
                except TypeError as e:
                    msg = 'Invalid quantization table'
                    raise ValueError(msg) from e
                else:
                    qtables[idx] = list(table)
            return qtables
    if qtables == 'keep':
        if im.format != 'JPEG':
            msg = "Cannot use 'keep' when original image is not a JPEG"
            raise ValueError(msg)
        qtables = getattr(im, 'quantization', None)
    qtables = validate_qtables(qtables)
    extra = info.get('extra', b'')
    MAX_BYTES_IN_MARKER = 65533
    icc_profile = info.get('icc_profile')
    if icc_profile:
        ICC_OVERHEAD_LEN = 14
        MAX_DATA_BYTES_IN_MARKER = MAX_BYTES_IN_MARKER - ICC_OVERHEAD_LEN
        markers = []
        while icc_profile:
            markers.append(icc_profile[:MAX_DATA_BYTES_IN_MARKER])
            icc_profile = icc_profile[MAX_DATA_BYTES_IN_MARKER:]
        i = 1
        for marker in markers:
            size = o16(2 + ICC_OVERHEAD_LEN + len(marker))
            extra += b'\xff\xe2' + size + b'ICC_PROFILE\x00' + o8(i) + o8(len(markers)) + marker
            i += 1
    comment = info.get('comment', im.info.get('comment'))
    progressive = info.get('progressive', False) or info.get('progression', False)
    optimize = info.get('optimize', False)
    exif = info.get('exif', b'')
    if isinstance(exif, Image.Exif):
        exif = exif.tobytes()
    if len(exif) > MAX_BYTES_IN_MARKER:
        msg = 'EXIF data is too long'
        raise ValueError(msg)
    im.encoderconfig = (quality, progressive, info.get('smooth', 0), optimize, info.get('streamtype', 0), dpi[0], dpi[1], subsampling, info.get('restart_marker_blocks', 0), info.get('restart_marker_rows', 0), qtables, comment, extra, exif)
    bufsize = 0
    if optimize or progressive:
        if im.mode == 'CMYK':
            bufsize = 4 * im.size[0] * im.size[1]
        elif quality >= 95 or quality == -1:
            bufsize = 2 * im.size[0] * im.size[1]
        else:
            bufsize = im.size[0] * im.size[1]
        if exif:
            bufsize += len(exif) + 5
        if extra:
            bufsize += len(extra) + 1
    else:
        bufsize = max(bufsize, len(exif) + 5, len(extra) + 1)
    ImageFile._save(im, fp, [('jpeg', (0, 0) + im.size, 0, rawmode)], bufsize)

def _save_cjpeg(im, fp, filename):
    if False:
        while True:
            i = 10
    tempfile = im._dump()
    subprocess.check_call(['cjpeg', '-outfile', filename, tempfile])
    try:
        os.unlink(tempfile)
    except OSError:
        pass

def jpeg_factory(fp=None, filename=None):
    if False:
        i = 10
        return i + 15
    im = JpegImageFile(fp, filename)
    try:
        mpheader = im._getmp()
        if mpheader[45057] > 1:
            from .MpoImagePlugin import MpoImageFile
            im = MpoImageFile.adopt(im, mpheader)
    except (TypeError, IndexError):
        pass
    except SyntaxError:
        warnings.warn('Image appears to be a malformed MPO file, it will be interpreted as a base JPEG file')
    return im
Image.register_open(JpegImageFile.format, jpeg_factory, _accept)
Image.register_save(JpegImageFile.format, _save)
Image.register_extensions(JpegImageFile.format, ['.jfif', '.jpe', '.jpg', '.jpeg'])
Image.register_mime(JpegImageFile.format, 'image/jpeg')