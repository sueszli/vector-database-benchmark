from typing import Any, Dict, Union
from transformers import AutoTokenizer
from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type
from .transformers_tokenizer import NLPTokenizer

@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.siamese_uie_preprocessor)
class SiameseUiePreprocessor(Preprocessor):
    """The tokenizer preprocessor used in zero shot classification.
    """

    def __init__(self, model_dir: str, mode: str=ModeKeys.INFERENCE, **kwargs):
        if False:
            while True:
                i = 10
        'preprocess the data\n        Args:\n            model_dir (str): model path\n        '
        super().__init__(mode)
        self.model_dir: str = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def __call__(self, data: list, **kwargs) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "process the raw input data\n\n        Args:\n            data (str or dict): a sentence\n                Example:\n                    'you are so handsome.'\n\n        Returns:\n            Dict[str, Any]: the preprocessed data\n        "
        features = self.tokenizer(data, **kwargs)
        return features