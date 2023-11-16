import olefile
from . import Image, TiffImagePlugin

def _accept(prefix):
    if False:
        return 10
    return prefix[:8] == olefile.MAGIC

class MicImageFile(TiffImagePlugin.TiffImageFile):
    format = 'MIC'
    format_description = 'Microsoft Image Composer'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        if False:
            print('Hello World!')
        try:
            self.ole = olefile.OleFileIO(self.fp)
        except OSError as e:
            msg = 'not an MIC file; invalid OLE file'
            raise SyntaxError(msg) from e
        self.images = []
        for path in self.ole.listdir():
            if path[1:] and path[0][-4:] == '.ACI' and (path[1] == 'Image'):
                self.images.append(path)
        if not self.images:
            msg = 'not an MIC file; no image entries'
            raise SyntaxError(msg)
        self.frame = None
        self._n_frames = len(self.images)
        self.is_animated = self._n_frames > 1
        self.seek(0)

    def seek(self, frame):
        if False:
            return 10
        if not self._seek_check(frame):
            return
        try:
            filename = self.images[frame]
        except IndexError as e:
            msg = 'no such frame'
            raise EOFError(msg) from e
        self.fp = self.ole.openstream(filename)
        TiffImagePlugin.TiffImageFile._open(self)
        self.frame = frame

    def tell(self):
        if False:
            print('Hello World!')
        return self.frame

    def close(self):
        if False:
            return 10
        self.ole.close()
        super().close()

    def __exit__(self, *args):
        if False:
            return 10
        self.ole.close()
        super().__exit__()
Image.register_open(MicImageFile.format, MicImageFile, _accept)
Image.register_extension(MicImageFile.format, '.mic')