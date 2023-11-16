import io
import os
import re
import subprocess
import sys
import tempfile
from . import Image, ImageFile
from ._binary import i32le as i32
from ._deprecate import deprecate
split = re.compile('^%%([^:]*):[ \\t]*(.*)[ \\t]*$')
field = re.compile('^%[%!\\w]([^:]*)[ \\t]*$')
gs_binary = None
gs_windows_binary = None

def has_ghostscript():
    if False:
        i = 10
        return i + 15
    global gs_binary, gs_windows_binary
    if gs_binary is None:
        if sys.platform.startswith('win'):
            if gs_windows_binary is None:
                import shutil
                for binary in ('gswin32c', 'gswin64c', 'gs'):
                    if shutil.which(binary) is not None:
                        gs_windows_binary = binary
                        break
                else:
                    gs_windows_binary = False
            gs_binary = gs_windows_binary
        else:
            try:
                subprocess.check_call(['gs', '--version'], stdout=subprocess.DEVNULL)
                gs_binary = 'gs'
            except OSError:
                gs_binary = False
    return gs_binary is not False

def Ghostscript(tile, size, fp, scale=1, transparency=False):
    if False:
        return 10
    'Render an image using Ghostscript'
    global gs_binary
    if not has_ghostscript():
        msg = 'Unable to locate Ghostscript on paths'
        raise OSError(msg)
    (decoder, tile, offset, data) = tile[0]
    (length, bbox) = data
    scale = int(scale) or 1
    width = size[0] * scale
    height = size[1] * scale
    res_x = 72.0 * width / (bbox[2] - bbox[0])
    res_y = 72.0 * height / (bbox[3] - bbox[1])
    (out_fd, outfile) = tempfile.mkstemp()
    os.close(out_fd)
    infile_temp = None
    if hasattr(fp, 'name') and os.path.exists(fp.name):
        infile = fp.name
    else:
        (in_fd, infile_temp) = tempfile.mkstemp()
        os.close(in_fd)
        infile = infile_temp
        with open(infile_temp, 'wb') as f:
            fp.seek(0, io.SEEK_END)
            fsize = fp.tell()
            fp.seek(0)
            lengthfile = fsize
            while lengthfile > 0:
                s = fp.read(min(lengthfile, 100 * 1024))
                if not s:
                    break
                lengthfile -= len(s)
                f.write(s)
    device = 'pngalpha' if transparency else 'ppmraw'
    command = [gs_binary, '-q', f'-g{width:d}x{height:d}', f'-r{res_x:f}x{res_y:f}', '-dBATCH', '-dNOPAUSE', '-dSAFER', f'-sDEVICE={device}', f'-sOutputFile={outfile}', '-c', f'{-bbox[0]} {-bbox[1]} translate', '-f', infile, '-c', 'showpage']
    try:
        startupinfo = None
        if sys.platform.startswith('win'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.check_call(command, startupinfo=startupinfo)
        out_im = Image.open(outfile)
        out_im.load()
    finally:
        try:
            os.unlink(outfile)
            if infile_temp:
                os.unlink(infile_temp)
        except OSError:
            pass
    im = out_im.im.copy()
    out_im.close()
    return im

class PSFile:
    """
    Wrapper for bytesio object that treats either CR or LF as end of line.
    This class is no longer used internally, but kept for backwards compatibility.
    """

    def __init__(self, fp):
        if False:
            while True:
                i = 10
        deprecate('PSFile', 11, action='If you need the functionality of this class you will need to implement it yourself.')
        self.fp = fp
        self.char = None

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            for i in range(10):
                print('nop')
        self.char = None
        self.fp.seek(offset, whence)

    def readline(self):
        if False:
            print('Hello World!')
        s = [self.char or b'']
        self.char = None
        c = self.fp.read(1)
        while c not in b'\r\n' and len(c):
            s.append(c)
            c = self.fp.read(1)
        self.char = self.fp.read(1)
        if self.char in b'\r\n':
            self.char = None
        return b''.join(s).decode('latin-1')

def _accept(prefix):
    if False:
        return 10
    return prefix[:4] == b'%!PS' or (len(prefix) >= 4 and i32(prefix) == 3335770309)

class EpsImageFile(ImageFile.ImageFile):
    """EPS File Parser for the Python Imaging Library"""
    format = 'EPS'
    format_description = 'Encapsulated Postscript'
    mode_map = {1: 'L', 2: 'LAB', 3: 'RGB', 4: 'CMYK'}

    def _open(self):
        if False:
            i = 10
            return i + 15
        (length, offset) = self._find_offset(self.fp)
        self.fp.seek(offset)
        self._mode = 'RGB'
        self._size = None
        byte_arr = bytearray(255)
        bytes_mv = memoryview(byte_arr)
        bytes_read = 0
        reading_header_comments = True
        reading_trailer_comments = False
        trailer_reached = False

        def check_required_header_comments():
            if False:
                while True:
                    i = 10
            if 'PS-Adobe' not in self.info:
                msg = 'EPS header missing "%!PS-Adobe" comment'
                raise SyntaxError(msg)
            if 'BoundingBox' not in self.info:
                msg = 'EPS header missing "%%BoundingBox" comment'
                raise SyntaxError(msg)

        def _read_comment(s):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal reading_trailer_comments
            try:
                m = split.match(s)
            except re.error as e:
                msg = 'not an EPS file'
                raise SyntaxError(msg) from e
            if m:
                (k, v) = m.group(1, 2)
                self.info[k] = v
                if k == 'BoundingBox':
                    if v == '(atend)':
                        reading_trailer_comments = True
                    elif not self._size or (trailer_reached and reading_trailer_comments):
                        try:
                            box = [int(float(i)) for i in v.split()]
                            self._size = (box[2] - box[0], box[3] - box[1])
                            self.tile = [('eps', (0, 0) + self.size, offset, (length, box))]
                        except Exception:
                            pass
                return True
        while True:
            byte = self.fp.read(1)
            if byte == b'':
                if bytes_read == 0:
                    break
            elif byte in b'\r\n':
                if bytes_read == 0:
                    continue
            else:
                if bytes_read >= 255:
                    if byte_arr[0] == ord('%'):
                        msg = 'not an EPS file'
                        raise SyntaxError(msg)
                    else:
                        if reading_header_comments:
                            check_required_header_comments()
                            reading_header_comments = False
                        bytes_read = 0
                byte_arr[bytes_read] = byte[0]
                bytes_read += 1
                continue
            if reading_header_comments:
                if byte_arr[0] != ord('%') or bytes_mv[:13] == b'%%EndComments':
                    check_required_header_comments()
                    reading_header_comments = False
                    continue
                s = str(bytes_mv[:bytes_read], 'latin-1')
                if not _read_comment(s):
                    m = field.match(s)
                    if m:
                        k = m.group(1)
                        if k[:8] == 'PS-Adobe':
                            self.info['PS-Adobe'] = k[9:]
                        else:
                            self.info[k] = ''
                    elif s[0] == '%':
                        pass
                    else:
                        msg = 'bad EPS header'
                        raise OSError(msg)
            elif bytes_mv[:11] == b'%ImageData:':
                image_data_values = byte_arr[11:bytes_read].split(None, 7)
                (columns, rows, bit_depth, mode_id) = (int(value) for value in image_data_values[:4])
                if bit_depth == 1:
                    self._mode = '1'
                elif bit_depth == 8:
                    try:
                        self._mode = self.mode_map[mode_id]
                    except ValueError:
                        break
                else:
                    break
                self._size = (columns, rows)
                return
            elif trailer_reached and reading_trailer_comments:
                if bytes_mv[:5] == b'%%EOF':
                    break
                s = str(bytes_mv[:bytes_read], 'latin-1')
                _read_comment(s)
            elif bytes_mv[:9] == b'%%Trailer':
                trailer_reached = True
            bytes_read = 0
        check_required_header_comments()
        if not self._size:
            msg = 'cannot determine EPS bounding box'
            raise OSError(msg)

    def _find_offset(self, fp):
        if False:
            i = 10
            return i + 15
        s = fp.read(4)
        if s == b'%!PS':
            fp.seek(0, io.SEEK_END)
            length = fp.tell()
            offset = 0
        elif i32(s) == 3335770309:
            s = fp.read(8)
            offset = i32(s)
            length = i32(s, 4)
        else:
            msg = 'not an EPS file'
            raise SyntaxError(msg)
        return (length, offset)

    def load(self, scale=1, transparency=False):
        if False:
            for i in range(10):
                print('nop')
        if self.tile:
            self.im = Ghostscript(self.tile, self.size, self.fp, scale, transparency)
            self._mode = self.im.mode
            self._size = self.im.size
            self.tile = []
        return Image.Image.load(self)

    def load_seek(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

def _save(im, fp, filename, eps=1):
    if False:
        print('Hello World!')
    'EPS Writer for the Python Imaging Library.'
    im.load()
    if im.mode == 'L':
        operator = (8, 1, b'image')
    elif im.mode == 'RGB':
        operator = (8, 3, b'false 3 colorimage')
    elif im.mode == 'CMYK':
        operator = (8, 4, b'false 4 colorimage')
    else:
        msg = 'image mode is not supported'
        raise ValueError(msg)
    if eps:
        fp.write(b'%!PS-Adobe-3.0 EPSF-3.0\n')
        fp.write(b'%%Creator: PIL 0.1 EpsEncode\n')
        fp.write(b'%%%%BoundingBox: 0 0 %d %d\n' % im.size)
        fp.write(b'%%Pages: 1\n')
        fp.write(b'%%EndComments\n')
        fp.write(b'%%Page: 1 1\n')
        fp.write(b'%%ImageData: %d %d ' % im.size)
        fp.write(b'%d %d 0 1 1 "%s"\n' % operator)
    fp.write(b'gsave\n')
    fp.write(b'10 dict begin\n')
    fp.write(b'/buf %d string def\n' % (im.size[0] * operator[1]))
    fp.write(b'%d %d scale\n' % im.size)
    fp.write(b'%d %d 8\n' % im.size)
    fp.write(b'[%d 0 0 -%d 0 %d]\n' % (im.size[0], im.size[1], im.size[1]))
    fp.write(b'{ currentfile buf readhexstring pop } bind\n')
    fp.write(operator[2] + b'\n')
    if hasattr(fp, 'flush'):
        fp.flush()
    ImageFile._save(im, fp, [('eps', (0, 0) + im.size, 0, None)])
    fp.write(b'\n%%%%EndBinary\n')
    fp.write(b'grestore end\n')
    if hasattr(fp, 'flush'):
        fp.flush()
Image.register_open(EpsImageFile.format, EpsImageFile, _accept)
Image.register_save(EpsImageFile.format, _save)
Image.register_extensions(EpsImageFile.format, ['.ps', '.eps'])
Image.register_mime(EpsImageFile.format, 'application/postscript')