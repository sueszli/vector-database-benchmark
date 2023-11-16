"""
ImageHash module
"""
import numpy as np
try:
    from PIL import Image
    import imagehash
    PIL = True
except ImportError:
    PIL = False
from ..base import Pipeline

class ImageHash(Pipeline):
    """
    Generates perceptual image hashes. These hashes can be used to detect near-duplicate images. This method is not
    backed by machine learning models and not intended to find conceptually similar images.
    """

    def __init__(self, algorithm='average', size=8, strings=True):
        if False:
            i = 10
            return i + 15
        '\n        Creates an ImageHash pipeline.\n\n        Args:\n            algorithm: image hashing algorithm (average, perceptual, difference, wavelet, color)\n            size: hash size\n            strings: outputs hex strings if True (default), otherwise the pipeline returns numpy arrays\n        '
        if not PIL:
            raise ImportError('ImageHash pipeline is not available - install "pipeline" extra to enable')
        self.algorithm = algorithm
        self.size = size
        self.strings = strings

    def __call__(self, images):
        if False:
            i = 10
            return i + 15
        '\n        Generates perceptual image hashes.\n\n        Args:\n            images: image|list\n\n        Returns:\n            list of hashes\n        '
        values = [images] if not isinstance(images, list) else images
        values = [Image.open(image) if isinstance(image, str) else image for image in values]
        hashes = [self.ihash(image) for image in values]
        return hashes[0] if not isinstance(images, list) else hashes

    def ihash(self, image):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets an image hash for image.\n\n        Args:\n            image: PIL image\n\n        Returns:\n            hash as hex string\n        '
        if self.algorithm == 'perceptual':
            data = imagehash.phash(image, self.size)
        elif self.algorithm == 'difference':
            data = imagehash.dhash(image, self.size)
        elif self.algorithm == 'wavelet':
            data = imagehash.whash(image, self.size)
        elif self.algorithm == 'color':
            data = imagehash.colorhash(image, self.size)
        else:
            data = imagehash.average_hash(image, self.size)
        return str(data) if self.strings else data.hash.astype(np.float32).reshape(-1)