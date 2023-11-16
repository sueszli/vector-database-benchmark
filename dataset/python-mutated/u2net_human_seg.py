import os
from typing import List
import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from .base import BaseSession

class U2netHumanSegSession(BaseSession):
    """
    This class represents a session for performing human segmentation using the U2Net model.
    """

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        if False:
            return 10
        '\n        Predicts human segmentation masks for the input image.\n\n        Parameters:\n            img (PILImage): The input image.\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            List[PILImage]: A list of predicted masks.\n        '
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
            return 10
        '\n        Downloads the U2Net model weights.\n\n        Parameters:\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            str: The path to the downloaded model weights.\n        '
        fname = f'{cls.name(*args, **kwargs)}.onnx'
        pooch.retrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx', None if cls.checksum_disabled(*args, **kwargs) else 'md5:c09ddc2e0104f800e3e1bb4652583d1f', fname=fname, path=cls.u2net_home(*args, **kwargs), progressbar=True)
        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the name of the U2Net model.\n\n        Parameters:\n            *args: Variable length argument list.\n            **kwargs: Arbitrary keyword arguments.\n\n        Returns:\n            str: The name of the model.\n        '
        return 'u2net_human_seg'