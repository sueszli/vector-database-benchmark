"""
Tex: Compressed texture
"""
__all__ = ('ImageLoaderTex',)
import json
from struct import unpack
from kivy.logger import Logger
from kivy.core.image import ImageLoaderBase, ImageData, ImageLoader

class ImageLoaderTex(ImageLoaderBase):

    @staticmethod
    def extensions():
        if False:
            print('Hello World!')
        return ('tex',)

    def load(self, filename):
        if False:
            while True:
                i = 10
        try:
            fd = open(filename, 'rb')
            if fd.read(4) != 'KTEX':
                raise Exception('Invalid tex identifier')
            headersize = unpack('I', fd.read(4))[0]
            header = fd.read(headersize)
            if len(header) != headersize:
                raise Exception('Truncated tex header')
            info = json.loads(header)
            data = fd.read()
            if len(data) != info['datalen']:
                raise Exception('Truncated tex data')
        except:
            Logger.warning('Image: Image <%s> is corrupted' % filename)
            raise
        (width, height) = info['image_size']
        (tw, th) = info['texture_size']
        images = [data]
        im = ImageData(width, height, str(info['format']), images[0], source=filename)
        '\n        if len(dds.images) > 1:\n            images = dds.images\n            images_size = dds.images_size\n            for index in range(1, len(dds.images)):\n                w, h = images_size[index]\n                data = images[index]\n                im.add_mipmap(index, w, h, data)\n        '
        return [im]
ImageLoader.register(ImageLoaderTex)