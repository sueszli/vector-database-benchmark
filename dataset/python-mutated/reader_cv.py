from typing import Optional, Tuple, Union
from catalyst.contrib.data.reader import IReader
from catalyst.contrib.utils.image import imread, mimread

class ImageReader(IReader):
    """Image reader abstraction. Reads images from a ``csv`` dataset."""

    def __init__(self, input_key: str, output_key: Optional[str]=None, rootpath: Optional[str]=None, grayscale: bool=False):
        if False:
            print('Hello World!')
        '\n        Args:\n            input_key: key to use from annotation dict\n            output_key: key to use to store the result,\n                default: ``input_key``\n            rootpath: path to images dataset root directory\n                (so your can use relative paths in annotations)\n            grayscale: flag if you need to work only\n                with grayscale images\n        '
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        self.grayscale = grayscale

    def __call__(self, element):
        if False:
            while True:
                i = 10
        'Reads a row from your annotations dict with filename and\n        transfer it to an image\n\n        Args:\n            element: elem in your dataset\n\n        Returns:\n            np.ndarray: Image\n        '
        image_name = str(element[self.input_key])
        img = imread(image_name, rootpath=self.rootpath, grayscale=self.grayscale)
        output = {self.output_key: img}
        return output

class MaskReader(IReader):
    """Mask reader abstraction. Reads masks from a `csv` dataset."""

    def __init__(self, input_key: str, output_key: Optional[str]=None, rootpath: Optional[str]=None, clip_range: Tuple[Union[int, float], Union[int, float]]=(0, 1)):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            input_key: key to use from annotation dict\n            output_key: key to use to store the result,\n                default: ``input_key``\n            rootpath: path to images dataset root directory\n                (so your can use relative paths in annotations)\n            clip_range (Tuple[int, int]): lower and upper interval edges,\n                image values outside the interval are clipped\n                to the interval edges\n        '
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        self.clip = clip_range

    def __call__(self, element):
        if False:
            print('Hello World!')
        'Reads a row from your annotations dict with filename and\n        transfer it to a mask\n\n        Args:\n            element: elem in your dataset.\n\n        Returns:\n            np.ndarray: Mask\n        '
        mask_name = str(element[self.input_key])
        mask = mimread(mask_name, rootpath=self.rootpath, clip_range=self.clip)
        output = {self.output_key: mask}
        return output
__all__ = ['ImageReader', 'MaskReader']