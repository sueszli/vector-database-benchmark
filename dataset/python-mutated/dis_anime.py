import os
from typing import List
import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from .base import BaseSession

class DisSession(BaseSession):
    """
    This class represents a session for object detection.
    """

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use a pre-trained model to predict the object in the given image.\n\n        Parameters:\n            img (PILImage): The input image.\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            List[PILImage]: A list of predicted mask images.\n        '
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
            i = 10
            return i + 15
        '\n        Download the pre-trained models.\n\n        Parameters:\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            str: The path of the downloaded model file.\n        '
        fname = f'{cls.name(*args, **kwargs)}.onnx'
        pooch.retrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx', None if cls.checksum_disabled(*args, **kwargs) else 'md5:6f184e756bb3bd901c8849220a83e38e', fname=fname, path=cls.u2net_home(*args, **kwargs), progressbar=True)
        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Get the name of the pre-trained model.\n\n        Parameters:\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            str: The name of the pre-trained model.\n        '
        return 'isnet-anime'