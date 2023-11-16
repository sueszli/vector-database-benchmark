from . import Image, ImageFile
from ._binary import i16le as word
from ._binary import si16le as short
from ._binary import si32le as _long
_handler = None

def register_handler(handler):
    if False:
        return 10
    '\n    Install application-specific WMF image handler.\n\n    :param handler: Handler object.\n    '
    global _handler
    _handler = handler
if hasattr(Image.core, 'drawwmf'):

    class WmfHandler:

        def open(self, im):
            if False:
                while True:
                    i = 10
            im._mode = 'RGB'
            self.bbox = im.info['wmf_bbox']

        def load(self, im):
            if False:
                while True:
                    i = 10
            im.fp.seek(0)
            return Image.frombytes('RGB', im.size, Image.core.drawwmf(im.fp.read(), im.size, self.bbox), 'raw', 'BGR', im.size[0] * 3 + 3 & -4, -1)
    register_handler(WmfHandler())

def _accept(prefix):
    if False:
        i = 10
        return i + 15
    return prefix[:6] == b'\xd7\xcd\xc6\x9a\x00\x00' or prefix[:4] == b'\x01\x00\x00\x00'

class WmfStubImageFile(ImageFile.StubImageFile):
    format = 'WMF'
    format_description = 'Windows Metafile'

    def _open(self):
        if False:
            return 10
        self._inch = None
        s = self.fp.read(80)
        if s[:6] == b'\xd7\xcd\xc6\x9a\x00\x00':
            self._inch = word(s, 14)
            x0 = short(s, 6)
            y0 = short(s, 8)
            x1 = short(s, 10)
            y1 = short(s, 12)
            self.info['dpi'] = 72
            size = ((x1 - x0) * self.info['dpi'] // self._inch, (y1 - y0) * self.info['dpi'] // self._inch)
            self.info['wmf_bbox'] = (x0, y0, x1, y1)
            if s[22:26] != b'\x01\x00\t\x00':
                msg = 'Unsupported WMF file format'
                raise SyntaxError(msg)
        elif s[:4] == b'\x01\x00\x00\x00' and s[40:44] == b' EMF':
            x0 = _long(s, 8)
            y0 = _long(s, 12)
            x1 = _long(s, 16)
            y1 = _long(s, 20)
            frame = (_long(s, 24), _long(s, 28), _long(s, 32), _long(s, 36))
            size = (x1 - x0, y1 - y0)
            xdpi = 2540.0 * (x1 - y0) / (frame[2] - frame[0])
            ydpi = 2540.0 * (y1 - y0) / (frame[3] - frame[1])
            self.info['wmf_bbox'] = (x0, y0, x1, y1)
            if xdpi == ydpi:
                self.info['dpi'] = xdpi
            else:
                self.info['dpi'] = (xdpi, ydpi)
        else:
            msg = 'Unsupported file format'
            raise SyntaxError(msg)
        self._mode = 'RGB'
        self._size = size
        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        if False:
            i = 10
            return i + 15
        return _handler

    def load(self, dpi=None):
        if False:
            i = 10
            return i + 15
        if dpi is not None and self._inch is not None:
            self.info['dpi'] = dpi
            (x0, y0, x1, y1) = self.info['wmf_bbox']
            self._size = ((x1 - x0) * self.info['dpi'] // self._inch, (y1 - y0) * self.info['dpi'] // self._inch)
        return super().load()

def _save(im, fp, filename):
    if False:
        while True:
            i = 10
    if _handler is None or not hasattr(_handler, 'save'):
        msg = 'WMF save handler not installed'
        raise OSError(msg)
    _handler.save(im, fp, filename)
Image.register_open(WmfStubImageFile.format, WmfStubImageFile, _accept)
Image.register_save(WmfStubImageFile.format, _save)
Image.register_extensions(WmfStubImageFile.format, ['.wmf', '.emf'])