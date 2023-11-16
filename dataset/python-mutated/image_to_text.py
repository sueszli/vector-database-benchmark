from typing import List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image
if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
logger = logging.get_logger(__name__)

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageToTextPipeline(Pipeline):
    """
    Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'generated_text': 'two birds are standing next to each other '}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        requires_backends(self, 'vision')
        self.check_model_type(TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES if self.framework == 'tf' else MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)

    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, prompt=None, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        forward_kwargs = {}
        preprocess_params = {}
        if prompt is not None:
            preprocess_params['prompt'] = prompt
        if timeout is not None:
            preprocess_params['timeout'] = timeout
        if generate_kwargs is not None:
            forward_kwargs['generate_kwargs'] = generate_kwargs
        if max_new_tokens is not None:
            if 'generate_kwargs' not in forward_kwargs:
                forward_kwargs['generate_kwargs'] = {}
            if 'max_new_tokens' in forward_kwargs['generate_kwargs']:
                raise ValueError("'max_new_tokens' is defined twice, once in 'generate_kwargs' and once as a direct parameter, please use only one")
            forward_kwargs['generate_kwargs']['max_new_tokens'] = max_new_tokens
        return (preprocess_params, forward_kwargs, {})

    def __call__(self, images: Union[str, List[str], 'Image.Image', List['Image.Image']], **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assign labels to the image(s) passed as inputs.\n\n        Args:\n            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):\n                The pipeline handles three types of images:\n\n                - A string containing a HTTP(s) link pointing to an image\n                - A string containing a local path to an image\n                - An image loaded in PIL directly\n\n                The pipeline accepts either a single image or a batch of images.\n\n            max_new_tokens (`int`, *optional*):\n                The amount of maximum tokens to generate. By default it will use `generate` default.\n\n            generate_kwargs (`Dict`, *optional*):\n                Pass it to send all of these arguments directly to `generate` allowing full control of this function.\n            timeout (`float`, *optional*, defaults to None):\n                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and\n                the call may block forever.\n\n        Return:\n            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:\n\n            - **generated_text** (`str`) -- The generated text.\n        '
        return super().__call__(images, **kwargs)

    def preprocess(self, image, prompt=None, timeout=None):
        if False:
            return 10
        image = load_image(image, timeout=timeout)
        if prompt is not None:
            if not isinstance(prompt, str):
                raise ValueError(f'Received an invalid text input, got - {type(prompt)} - but expected a single string. Note also that one single text can be provided for conditional image to text generation.')
            model_type = self.model.config.model_type
            if model_type == 'git':
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                input_ids = self.tokenizer(text=prompt, add_special_tokens=False).input_ids
                input_ids = [self.tokenizer.cls_token_id] + input_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                model_inputs.update({'input_ids': input_ids})
            elif model_type == 'pix2struct':
                model_inputs = self.image_processor(images=image, header_text=prompt, return_tensors=self.framework)
            elif model_type != 'vision-encoder-decoder':
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                text_inputs = self.tokenizer(prompt, return_tensors=self.framework)
                model_inputs.update(text_inputs)
            else:
                raise ValueError(f'Model type {model_type} does not support conditional text generation')
        else:
            model_inputs = self.image_processor(images=image, return_tensors=self.framework)
        if self.model.config.model_type == 'git' and prompt is None:
            model_inputs['input_ids'] = None
        return model_inputs

    def _forward(self, model_inputs, generate_kwargs=None):
        if False:
            while True:
                i = 10
        if 'input_ids' in model_inputs and isinstance(model_inputs['input_ids'], list) and all((x is None for x in model_inputs['input_ids'])):
            model_inputs['input_ids'] = None
        if generate_kwargs is None:
            generate_kwargs = {}
        inputs = model_inputs.pop(self.model.main_input_name)
        model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
        return model_outputs

    def postprocess(self, model_outputs):
        if False:
            for i in range(10):
                print('nop')
        records = []
        for output_ids in model_outputs:
            record = {'generated_text': self.tokenizer.decode(output_ids, skip_special_tokens=True)}
            records.append(record)
        return records