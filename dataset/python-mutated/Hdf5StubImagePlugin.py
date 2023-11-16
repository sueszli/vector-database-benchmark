from . import Image, ImageFile
_handler = None

def register_handler(handler):
    if False:
        return 10
    '\n    Install application-specific HDF5 image handler.\n\n    :param handler: Handler object.\n    '
    global _handler
    _handler = handler

def _accept(prefix):
    if False:
        for i in range(10):
            print('nop')
    return prefix[:8] == b'\x89HDF\r\n\x1a\n'

class HDF5StubImageFile(ImageFile.StubImageFile):
    format = 'HDF5'
    format_description = 'HDF5'

    def _open(self):
        if False:
            print('Hello World!')
        offset = self.fp.tell()
        if not _accept(self.fp.read(8)):
            msg = 'Not an HDF file'
            raise SyntaxError(msg)
        self.fp.seek(offset)
        self._mode = 'F'
        self._size = (1, 1)
        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        if False:
            i = 10
            return i + 15
        return _handler

def _save(im, fp, filename):
    if False:
        while True:
            i = 10
    if _handler is None or not hasattr(_handler, 'save'):
        msg = 'HDF5 save handler not installed'
        raise OSError(msg)
    _handler.save(im, fp, filename)
Image.register_open(HDF5StubImageFile.format, HDF5StubImageFile, _accept)
Image.register_save(HDF5StubImageFile.format, _save)
Image.register_extensions(HDF5StubImageFile.format, ['.h5', '.hdf'])