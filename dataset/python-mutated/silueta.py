import os
from typing import List
import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from .base import BaseSession

class SiluetaSession(BaseSession):
    """This is a class representing a SiluetaSession object."""

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Predict the mask of the input image.\n\n        This method takes an image as input, preprocesses it, and performs a prediction to generate a mask. The generated mask is then post-processed and returned as a list of PILImage objects.\n\n        Parameters:\n            img (PILImage): The input image to be processed.\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            List[PILImage]: A list of post-processed masks.\n        '
        ort_outs = self.inner_session.run(None, self.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)))
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
            for i in range(10):
                print('nop')
        '\n        Download the pre-trained model file.\n\n        This method downloads the pre-trained model file from a specified URL. The file is saved to the U2NET home directory.\n\n        Parameters:\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            str: The path to the downloaded model file.\n        '
        fname = f'{cls.name()}.onnx'
        pooch.retrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx', None if cls.checksum_disabled(*args, **kwargs) else 'md5:55e59e0d8062d2f5d013f4725ee84782', fname=fname, path=cls.u2net_home(*args, **kwargs), progressbar=True)
        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        if False:
            return 10
        '\n        Return the name of the model.\n\n        This method returns the name of the Silueta model.\n\n        Parameters:\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            str: The name of the model.\n        '
        return 'silueta'