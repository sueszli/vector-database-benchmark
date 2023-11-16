"""
SDL2 image loader
=================
"""
__all__ = ('ImageLoaderSDL2',)
from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader
try:
    from kivy.core.image import _img_sdl2
except ImportError:
    from kivy.core import handle_win_lib_import_error
    handle_win_lib_import_error('image', 'sdl2', 'kivy.core.image._img_sdl2')
    raise

class ImageLoaderSDL2(ImageLoaderBase):
    """Image loader based on SDL2_image"""

    def _ensure_ext(self):
        if False:
            for i in range(10):
                print('nop')
        _img_sdl2.init()

    @staticmethod
    def extensions():
        if False:
            while True:
                i = 10
        'Return accepted extensions for this loader'
        return ('bmp', 'jpg', 'jpeg', 'jpe', 'lbm', 'pcx', 'png', 'pnm', 'tga', 'tiff', 'webp', 'xcf', 'xpm', 'xv')

    @staticmethod
    def can_save(fmt, is_bytesio):
        if False:
            return 10
        return fmt in ('jpg', 'png')

    @staticmethod
    def can_load_memory():
        if False:
            i = 10
            return i + 15
        return True

    def load(self, filename):
        if False:
            i = 10
            return i + 15
        if self._inline:
            data = filename.read()
            info = _img_sdl2.load_from_memory(data)
        else:
            info = _img_sdl2.load_from_filename(filename)
        if not info:
            Logger.warning('Image: Unable to load image <%s>' % filename)
            raise Exception('SDL2: Unable to load image')
        (w, h, fmt, pixels, rowlength) = info
        if not self._inline:
            self.filename = filename
        return [ImageData(w, h, fmt, pixels, source=filename, rowlength=rowlength)]

    @staticmethod
    def save(filename, width, height, pixelfmt, pixels, flipped, imagefmt):
        if False:
            i = 10
            return i + 15
        _img_sdl2.save(filename, width, height, pixelfmt, pixels, flipped, imagefmt)
        return True
ImageLoader.register(ImageLoaderSDL2)