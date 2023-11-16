from . import Image, ImageFile
_handler = None

def register_handler(handler):
    if False:
        while True:
            i = 10
    '\n    Install application-specific GRIB image handler.\n\n    :param handler: Handler object.\n    '
    global _handler
    _handler = handler

def _accept(prefix):
    if False:
        for i in range(10):
            print('nop')
    return prefix[:4] == b'GRIB' and prefix[7] == 1

class GribStubImageFile(ImageFile.StubImageFile):
    format = 'GRIB'
    format_description = 'GRIB'

    def _open(self):
        if False:
            print('Hello World!')
        offset = self.fp.tell()
        if not _accept(self.fp.read(8)):
            msg = 'Not a GRIB file'
            raise SyntaxError(msg)
        self.fp.seek(offset)
        self._mode = 'F'
        self._size = (1, 1)
        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        if False:
            print('Hello World!')
        return _handler

def _save(im, fp, filename):
    if False:
        while True:
            i = 10
    if _handler is None or not hasattr(_handler, 'save'):
        msg = 'GRIB save handler not installed'
        raise OSError(msg)
    _handler.save(im, fp, filename)
Image.register_open(GribStubImageFile.format, GribStubImageFile, _accept)
Image.register_save(GribStubImageFile.format, _save)
Image.register_extension(GribStubImageFile.format, '.grib')