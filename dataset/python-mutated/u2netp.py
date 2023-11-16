import os
from typing import List
import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from .base import BaseSession

class U2netpSession(BaseSession):
    """This class represents a session for using the U2netp model."""

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        if False:
            return 10
        '\n        Predicts the mask for the given image using the U2netp model.\n\n        Parameters:\n            img (PILImage): The input image.\n\n        Returns:\n            List[PILImage]: The predicted mask.\n        '
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
            while True:
                i = 10
        '\n        Downloads the U2netp model.\n\n        Returns:\n            str: The path to the downloaded model.\n        '
        fname = f'{cls.name(*args, **kwargs)}.onnx'
        pooch.retrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx', None if cls.checksum_disabled(*args, **kwargs) else 'md5:8e83ca70e441ab06c318d82300c84806', fname=fname, path=cls.u2net_home(*args, **kwargs), progressbar=True)
        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the name of the U2netp model.\n\n        Returns:\n            str: The name of the model.\n        '
        return 'u2netp'