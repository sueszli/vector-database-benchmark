import os
from typing import List
import numpy as np
import onnxruntime as ort
import pooch
from PIL import Image
from PIL.Image import Image as PILImage
from .base import BaseSession

class U2netCustomSession(BaseSession):
    """This is a class representing a custom session for the U2net model."""

    def __init__(self, model_name: str, sess_opts: ort.SessionOptions, providers=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize a new U2netCustomSession object.\n\n        Parameters:\n            model_name (str): The name of the model.\n            sess_opts (ort.SessionOptions): The session options.\n            providers: The providers.\n            *args: Additional positional arguments.\n            **kwargs: Additional keyword arguments.\n\n        Raises:\n            ValueError: If model_path is None.\n        '
        model_path = kwargs.get('model_path')
        if model_path is None:
            raise ValueError('model_path is required')
        super().__init__(model_name, sess_opts, providers, *args, **kwargs)

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        if False:
            print('Hello World!')
        '\n        Predict the segmentation mask for the input image.\n\n        Parameters:\n            img (PILImage): The input image.\n            *args: Additional positional arguments.\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            List[PILImage]: A list of PILImage objects representing the segmentation mask.\n        '
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
        '\n        Download the model files.\n\n        Parameters:\n            *args: Additional positional arguments.\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            str: The absolute path to the model files.\n        '
        model_path = kwargs.get('model_path')
        if model_path is None:
            return
        return os.path.abspath(os.path.expanduser(model_path))

    @classmethod
    def name(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Get the name of the model.\n\n        Parameters:\n            *args: Additional positional arguments.\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            str: The name of the model.\n        '
        return 'u2net_custom'