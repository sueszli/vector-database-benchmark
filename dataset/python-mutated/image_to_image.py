from typing import List, Union
import numpy as np
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES
logger = logging.get_logger(__name__)

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageToImagePipeline(Pipeline):
    """
    Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous
    image input.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests

    >>> from transformers import pipeline

    >>> upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
    >>> img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    >>> img = img.resize((64, 64))
    >>> upscaled_img = upscaler(img)
    >>> img.size
    (64, 64)

    >>> upscaled_img.size
    (144, 144)
    ```

    This image to image pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-to-image"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-to-image).
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        requires_backends(self, 'vision')
        self.check_model_type(MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)

    def _sanitize_parameters(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}
        if 'timeout' in kwargs:
            preprocess_params['timeout'] = kwargs['timeout']
        if 'head_mask' in kwargs:
            forward_params['head_mask'] = kwargs['head_mask']
        return (preprocess_params, forward_params, postprocess_params)

    def __call__(self, images: Union[str, List[str], 'Image.Image', List['Image.Image']], **kwargs) -> Union['Image.Image', List['Image.Image']]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform the image(s) passed as inputs.\n\n        Args:\n            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):\n                The pipeline handles three types of images:\n\n                - A string containing a http link pointing to an image\n                - A string containing a local path to an image\n                - An image loaded in PIL directly\n\n                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.\n                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL\n                images.\n            timeout (`float`, *optional*, defaults to None):\n                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and\n                the call may block forever.\n\n        Return:\n            An image (Image.Image) or a list of images (List["Image.Image"]) containing result(s). If the input is a\n            single image, the return will be also a single image, if the input is a list of several images, it will\n            return a list of transformed images.\n        '
        return super().__call__(images, **kwargs)

    def _forward(self, model_inputs):
        if False:
            return 10
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def preprocess(self, image, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        image = load_image(image, timeout=timeout)
        inputs = self.image_processor(images=[image], return_tensors='pt')
        return inputs

    def postprocess(self, model_outputs):
        if False:
            while True:
                i = 10
        images = []
        if 'reconstruction' in model_outputs.keys():
            outputs = model_outputs.reconstruction
        for output in outputs:
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.moveaxis(output, source=0, destination=-1)
            output = (output * 255.0).round().astype(np.uint8)
            images.append(Image.fromarray(output))
        return images if len(images) > 1 else images[0]