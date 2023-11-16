from typing import Any, Dict
import torch
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor

class OfaTextToImageSynthesisPreprocessor(OfaBasePreprocessor):
    """
    OFA preprocessor for text to image synthesis tasks.
    """

    def __init__(self, cfg, model_dir, mode=ModeKeys.INFERENCE, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'preprocess the data\n\n        Args:\n            cfg(modelscope.utils.config.ConfigDict) : model config\n            model_dir (str): model path,\n            mode: preprocessor mode (model mode)\n        '
        super(OfaTextToImageSynthesisPreprocessor, self).__init__(cfg, model_dir, mode, *args, **kwargs)
        self.max_src_length = 64

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        "\n        Building samples for inference.\n\n        step 1. Preprocessing for str input.\n            - do lower, strip and restrict the total length by `max_src_length`.\n        step 2. Building text to image synthesis instruction. The template of\n            the instruction is like `what is the complete image? caption: {}`,\n            while the `{}` will be replaced by the result of step 1.\n        step 3. Tokenize the instruction as model's inputs.\n\n\n        Args:\n            data (`Dict[str, Any]`): Input data, should contains the key of `text`,\n                which refer to the description of synthesis image.\n        Return:\n            A dict object, contains source text input, patch images with `None` value\n            patch masks and code masks with `Tensor([False])` value.\n        "
        source = ' '.join(data['text'].lower().strip().split()[:self.max_src_length])
        source = 'what is the complete image? caption: {}'.format(source)
        inputs = self.tokenize_text(source)
        sample = {'source': inputs, 'patch_images': None, 'patch_masks': torch.tensor([False]), 'code_masks': torch.tensor([False])}
        return sample