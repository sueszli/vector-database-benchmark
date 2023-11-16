import skimage.transform
import skimage.color
import skimage.util
from PIL import Image
import io

class FrameTransformer:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def resize(frame, size_string=None):
        if False:
            for i in range(10):
                print('nop')
        (width, height) = size_string.lower().split('x')
        return skimage.util.img_as_ubyte(skimage.transform.resize(frame, (int(height), int(width)), order=0))

    @staticmethod
    def rescale(frame, scale=None):
        if False:
            return 10
        return skimage.util.img_as_ubyte(skimage.transform.rescale(frame, float(scale)))

    @staticmethod
    def crop(frame, y0, x0, y1, x1):
        if False:
            while True:
                i = 10
        return frame[int(y0):int(y1), int(x0):int(x1), :]

    @staticmethod
    def grayscale(frame):
        if False:
            for i in range(10):
                print('nop')
        return skimage.util.img_as_ubyte(skimage.color.rgb2gray(frame))

    @staticmethod
    def to_float(frame):
        if False:
            for i in range(10):
                print('nop')
        return skimage.util.img_as_float(frame)

    @staticmethod
    def to_png(frame):
        if False:
            for i in range(10):
                print('nop')
        pil_frame = Image.fromarray(skimage.util.img_as_ubyte(frame))
        pil_frame = pil_frame.convert('RGB')
        png_frame = io.BytesIO()
        pil_frame.save(png_frame, format='PNG', compress_level=3)
        png_frame.seek(0)
        return png_frame.read()