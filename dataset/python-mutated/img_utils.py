import struct
import imghdr

def isValidStream(data):
    if False:
        for i in range(10):
            print('nop')
    if data is None:
        return False
    format = imghdr.what(None, data)
    if format is None:
        return False
    return True

def getImgFormat(filepath):
    if False:
        while True:
            i = 10
    "\n\tRead header of an image file and try to determine it's format\n\tno requirements, support JPEG, JPEG2000, PNG, GIF, BMP, TIFF, EXR\n\t"
    format = None
    with open(filepath, 'rb') as fhandle:
        head = fhandle.read(32)
        if head[:6] in (b'GIF87a', b'GIF89a'):
            format = 'GIF'
        elif head.startswith(b'\x89PNG\r\n\x1a\n'):
            format = 'PNG'
        elif (b'JFIF' in head or b'Exif' in head or b'8BIM' in head) or head.startswith(b'\xff\xd8'):
            format = 'JPEG'
        elif head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
            format = 'JPEG2000'
        elif head.startswith(b'BM'):
            format = 'BMP'
        elif head[:2] in (b'MM', b'II'):
            format = 'TIFF'
        elif head.startswith(b'v/1\x01'):
            format = 'EXR'
    return format

def getImgDim(filepath):
    if False:
        for i in range(10):
            print('nop')
    '\n\tReturn (width, height) for a given img file content\n\tno requirements, support JPEG, JPEG2000, PNG, GIF, BMP\n\t'
    (width, height) = (None, None)
    with open(filepath, 'rb') as fhandle:
        head = fhandle.read(32)
        if head[:6] in (b'GIF87a', b'GIF89a'):
            try:
                (width, height) = struct.unpack('<hh', head[6:10])
            except struct.error:
                raise ValueError('Invalid GIF file')
        elif head.startswith(b'\x89PNG\r\n\x1a\n'):
            try:
                (width, height) = struct.unpack('>LL', head[16:24])
            except struct.error:
                try:
                    (width, height) = struct.unpack('>LL', head[8:16])
                except struct.error:
                    raise ValueError('Invalid PNG file')
        elif (b'JFIF' in head or b'Exif' in head or b'8BIM' in head) or head.startswith(b'\xff\xd8'):
            try:
                fhandle.seek(0)
                size = 2
                ftype = 0
                while not 192 <= ftype <= 207:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 255:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                fhandle.seek(1, 1)
                (height, width) = struct.unpack('>HH', fhandle.read(4))
            except struct.error:
                raise ValueError('Invalid JPEG file')
        elif head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
            fhandle.seek(48)
            try:
                (height, width) = struct.unpack('>LL', fhandle.read(8))
            except struct.error:
                raise ValueError('Invalid JPEG2000 file')
        elif head.startswith(b'BM'):
            imgtype = 'BMP'
            try:
                (width, height) = struct.unpack('<LL', head[18:26])
            except struct.error:
                raise ValueError('Invalid BMP file')
    return (width, height)