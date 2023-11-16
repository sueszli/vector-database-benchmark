import os
from typing import List
import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from .base import BaseSession

class DisSession(BaseSession):

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        if False:
            while True:
                i = 10
        '\n        Predicts the mask image for the input image.\n\n        This method takes a PILImage object as input and returns a list of PILImage objects as output. It performs several image processing operations to generate the mask image.\n\n        Parameters:\n            img (PILImage): The input image.\n\n        Returns:\n            List[PILImage]: A list of PILImage objects representing the generated mask image.\n        '
        ort_outs = self.inner_session.run(None, self.normalize(img, (0.485, 0.456, 0.406), (1.0, 1.0, 1.0), (1024, 1024)))
        pred = ort_outs[0][:, 0, :, :]
        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)
        mask = Image.fromarray((pred * 255).astype('uint8'), mode='L')
        mask = mask.resize(img.size, Image.LANCZOS)
        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Downloads the pre-trained model file.\n\n        This class method downloads the pre-trained model file from a specified URL using the pooch library.\n\n        Parameters:\n            args: Additional positional arguments.\n            kwargs: Additional keyword arguments.\n\n        Returns:\n            str: The path to the downloaded model file.\n        '
        fname = f'{cls.name(*args, **kwargs)}.onnx'
        pooch.retrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx', None if cls.checksum_disabled(*args, **kwargs) else 'md5:fc16ebd8b0c10d971d3513d564d01e29', fname=fname, path=cls.u2net_home(*args, **kwargs), progressbar=True)
        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the name of the model.\n\n        This class method returns the name of the model.\n\n        Parameters:\n            args: Additional positional arguments.\n            kwargs: Additional keyword arguments.\n\n        Returns:\n            str: The name of the model.\n        '
        return 'isnet-general-use'