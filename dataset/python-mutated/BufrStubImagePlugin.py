from . import Image, ImageFile
_handler = None

def register_handler(handler):
    if False:
        while True:
            i = 10
    '\n    Install application-specific BUFR image handler.\n\n    :param handler: Handler object.\n    '
    global _handler
    _handler = handler

def _accept(prefix):
    if False:
        return 10
    return prefix[:4] == b'BUFR' or prefix[:4] == b'ZCZC'

class BufrStubImageFile(ImageFile.StubImageFile):
    format = 'BUFR'
    format_description = 'BUFR'

    def _open(self):
        if False:
            i = 10
            return i + 15
        offset = self.fp.tell()
        if not _accept(self.fp.read(4)):
            msg = 'Not a BUFR file'
            raise SyntaxError(msg)
        self.fp.seek(offset)
        self._mode = 'F'
        self._size = (1, 1)
        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        if False:
            while True:
                i = 10
        return _handler

def _save(im, fp, filename):
    if False:
        for i in range(10):
            print('nop')
    if _handler is None or not hasattr(_handler, 'save'):
        msg = 'BUFR save handler not installed'
        raise OSError(msg)
    _handler.save(im, fp, filename)
Image.register_open(BufrStubImageFile.format, BufrStubImageFile, _accept)
Image.register_save(BufrStubImageFile.format, _save)
Image.register_extension(BufrStubImageFile.format, '.bufr')