from collections import UserDict
from typing import List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline
if is_vision_available():
    from PIL import Image
    from ..image_utils import load_image
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
    from ..tf_utils import stable_softmax
logger = logging.get_logger(__name__)

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotImageClassificationPipeline(Pipeline):
    """
    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
    provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="openai/clip-vit-large-patch14")
    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["animals", "humans", "landscape"],
    ... )
    [{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["black and white", "photorealist", "painting"],
    ... )
    [{'score': 0.996, 'label': 'black and white'}, {'score': 0.003, 'label': 'photorealist'}, {'score': 0.0, 'label': 'painting'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-image-classification).
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        requires_backends(self, 'vision')
        self.check_model_type(TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES if self.framework == 'tf' else MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES)

    def __call__(self, images: Union[str, List[str], 'Image', List['Image']], **kwargs):
        if False:
            print('Hello World!')
        '\n        Assign labels to the image(s) passed as inputs.\n\n        Args:\n            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):\n                The pipeline handles three types of images:\n\n                - A string containing a http link pointing to an image\n                - A string containing a local path to an image\n                - An image loaded in PIL directly\n\n            candidate_labels (`List[str]`):\n                The candidate labels for this image\n\n            hypothesis_template (`str`, *optional*, defaults to `"This is a photo of {}"`):\n                The sentence used in cunjunction with *candidate_labels* to attempt the image classification by\n                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using\n                logits_per_image\n\n            timeout (`float`, *optional*, defaults to None):\n                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and\n                the call may block forever.\n\n        Return:\n            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the\n            following keys:\n\n            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.\n            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).\n        '
        return super().__call__(images, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        if False:
            while True:
                i = 10
        preprocess_params = {}
        if 'candidate_labels' in kwargs:
            preprocess_params['candidate_labels'] = kwargs['candidate_labels']
        if 'timeout' in kwargs:
            preprocess_params['timeout'] = kwargs['timeout']
        if 'hypothesis_template' in kwargs:
            preprocess_params['hypothesis_template'] = kwargs['hypothesis_template']
        return (preprocess_params, {}, {})

    def preprocess(self, image, candidate_labels=None, hypothesis_template='This is a photo of {}.', timeout=None):
        if False:
            return 10
        image = load_image(image, timeout=timeout)
        inputs = self.image_processor(images=[image], return_tensors=self.framework)
        inputs['candidate_labels'] = candidate_labels
        sequences = [hypothesis_template.format(x) for x in candidate_labels]
        text_inputs = self.tokenizer(sequences, return_tensors=self.framework, padding=True)
        inputs['text_inputs'] = [text_inputs]
        return inputs

    def _forward(self, model_inputs):
        if False:
            while True:
                i = 10
        candidate_labels = model_inputs.pop('candidate_labels')
        text_inputs = model_inputs.pop('text_inputs')
        if isinstance(text_inputs[0], UserDict):
            text_inputs = text_inputs[0]
        else:
            text_inputs = text_inputs[0][0]
        outputs = self.model(**text_inputs, **model_inputs)
        model_outputs = {'candidate_labels': candidate_labels, 'logits': outputs.logits_per_image}
        return model_outputs

    def postprocess(self, model_outputs):
        if False:
            for i in range(10):
                print('nop')
        candidate_labels = model_outputs.pop('candidate_labels')
        logits = model_outputs['logits'][0]
        if self.framework == 'pt':
            probs = logits.softmax(dim=-1).squeeze(-1)
            scores = probs.tolist()
            if not isinstance(scores, list):
                scores = [scores]
        elif self.framework == 'tf':
            probs = stable_softmax(logits, axis=-1)
            scores = probs.numpy().tolist()
        else:
            raise ValueError(f'Unsupported framework: {self.framework}')
        result = [{'score': score, 'label': candidate_label} for (score, candidate_label) in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])]
        return result