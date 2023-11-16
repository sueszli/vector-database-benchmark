import sys
from io import BytesIO
from . import Image
from ._util import is_path
qt_versions = [['6', 'PyQt6'], ['side6', 'PySide6']]
qt_versions.sort(key=lambda qt_version: qt_version[1] in sys.modules, reverse=True)
for (qt_version, qt_module) in qt_versions:
    try:
        if qt_module == 'PyQt6':
            from PyQt6.QtCore import QBuffer, QIODevice
            from PyQt6.QtGui import QImage, QPixmap, qRgba
        elif qt_module == 'PySide6':
            from PySide6.QtCore import QBuffer, QIODevice
            from PySide6.QtGui import QImage, QPixmap, qRgba
    except (ImportError, RuntimeError):
        continue
    qt_is_installed = True
    break
else:
    qt_is_installed = False
    qt_version = None

def rgb(r, g, b, a=255):
    if False:
        i = 10
        return i + 15
    '(Internal) Turns an RGB color into a Qt compatible color integer.'
    return qRgba(r, g, b, a) & 4294967295

def fromqimage(im):
    if False:
        print('Hello World!')
    '\n    :param im: QImage or PIL ImageQt object\n    '
    buffer = QBuffer()
    if qt_version == '6':
        try:
            qt_openmode = QIODevice.OpenModeFlag
        except AttributeError:
            qt_openmode = QIODevice.OpenMode
    else:
        qt_openmode = QIODevice
    buffer.open(qt_openmode.ReadWrite)
    if im.hasAlphaChannel():
        im.save(buffer, 'png')
    else:
        im.save(buffer, 'ppm')
    b = BytesIO()
    b.write(buffer.data())
    buffer.close()
    b.seek(0)
    return Image.open(b)

def fromqpixmap(im):
    if False:
        while True:
            i = 10
    return fromqimage(im)

def align8to32(bytes, width, mode):
    if False:
        print('Hello World!')
    '\n    converts each scanline of data from 8 bit to 32 bit aligned\n    '
    bits_per_pixel = {'1': 1, 'L': 8, 'P': 8, 'I;16': 16}[mode]
    bits_per_line = bits_per_pixel * width
    (full_bytes_per_line, remaining_bits_per_line) = divmod(bits_per_line, 8)
    bytes_per_line = full_bytes_per_line + (1 if remaining_bits_per_line else 0)
    extra_padding = -bytes_per_line % 4
    if not extra_padding:
        return bytes
    new_data = []
    for i in range(len(bytes) // bytes_per_line):
        new_data.append(bytes[i * bytes_per_line:(i + 1) * bytes_per_line] + b'\x00' * extra_padding)
    return b''.join(new_data)

def _toqclass_helper(im):
    if False:
        return 10
    data = None
    colortable = None
    exclusive_fp = False
    if hasattr(im, 'toUtf8'):
        im = str(im.toUtf8(), 'utf-8')
    if is_path(im):
        im = Image.open(im)
        exclusive_fp = True
    qt_format = QImage.Format if qt_version == '6' else QImage
    if im.mode == '1':
        format = qt_format.Format_Mono
    elif im.mode == 'L':
        format = qt_format.Format_Indexed8
        colortable = []
        for i in range(256):
            colortable.append(rgb(i, i, i))
    elif im.mode == 'P':
        format = qt_format.Format_Indexed8
        colortable = []
        palette = im.getpalette()
        for i in range(0, len(palette), 3):
            colortable.append(rgb(*palette[i:i + 3]))
    elif im.mode == 'RGB':
        im = im.convert('RGBA')
        data = im.tobytes('raw', 'BGRA')
        format = qt_format.Format_RGB32
    elif im.mode == 'RGBA':
        data = im.tobytes('raw', 'BGRA')
        format = qt_format.Format_ARGB32
    elif im.mode == 'I;16' and hasattr(qt_format, 'Format_Grayscale16'):
        im = im.point(lambda i: i * 256)
        format = qt_format.Format_Grayscale16
    else:
        if exclusive_fp:
            im.close()
        msg = f'unsupported image mode {repr(im.mode)}'
        raise ValueError(msg)
    size = im.size
    __data = data or align8to32(im.tobytes(), size[0], im.mode)
    if exclusive_fp:
        im.close()
    return {'data': __data, 'size': size, 'format': format, 'colortable': colortable}
if qt_is_installed:

    class ImageQt(QImage):

        def __init__(self, im):
            if False:
                print('Hello World!')
            "\n            An PIL image wrapper for Qt.  This is a subclass of PyQt's QImage\n            class.\n\n            :param im: A PIL Image object, or a file name (given either as\n                Python string or a PyQt string object).\n            "
            im_data = _toqclass_helper(im)
            self.__data = im_data['data']
            super().__init__(self.__data, im_data['size'][0], im_data['size'][1], im_data['format'])
            if im_data['colortable']:
                self.setColorTable(im_data['colortable'])

def toqimage(im):
    if False:
        i = 10
        return i + 15
    return ImageQt(im)

def toqpixmap(im):
    if False:
        return 10
    qimage = toqimage(im)
    return QPixmap.fromImage(qimage)