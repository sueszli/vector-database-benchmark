from . import Image, ImageFile

class PcdImageFile(ImageFile.ImageFile):
    format = 'PCD'
    format_description = 'Kodak PhotoCD'

    def _open(self):
        if False:
            return 10
        self.fp.seek(2048)
        s = self.fp.read(2048)
        if s[:4] != b'PCD_':
            msg = 'not a PCD file'
            raise SyntaxError(msg)
        orientation = s[1538] & 3
        self.tile_post_rotate = None
        if orientation == 1:
            self.tile_post_rotate = 90
        elif orientation == 3:
            self.tile_post_rotate = -90
        self._mode = 'RGB'
        self._size = (768, 512)
        self.tile = [('pcd', (0, 0) + self.size, 96 * 2048, None)]

    def load_end(self):
        if False:
            print('Hello World!')
        if self.tile_post_rotate:
            self.im = self.im.rotate(self.tile_post_rotate)
            self._size = self.im.size
Image.register_open(PcdImageFile.format, PcdImageFile)
Image.register_extension(PcdImageFile.format, '.pcd')