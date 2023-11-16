from typing import Any, Dict
from transformers import AutoTokenizer
from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert

@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.re_tokenizer)
class RelationExtractionTransformersPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, mode: str=ModeKeys.INFERENCE, **kwargs):
        if False:
            while True:
                i = 10
        "The preprocessor for relation Extraction task, based on transformers' tokenizer.\n\n        Args:\n            model_dir: The model dir used to initialize the tokenizer.\n            mode: The mode for the preprocessor.\n        "
        super().__init__(mode)
        self.model_dir: str = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    @type_assert(object, str)
    def __call__(self, data: str, **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        "process the raw input data\n\n        Args:\n            data (str): a sentence\n                Example:\n                    'you are so handsome.'\n\n        Returns:\n            Dict[str, Any]: the preprocessed data\n        "
        text = data
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        output = self.tokenizer([text], **kwargs)
        return {'text': text, 'input_ids': output['input_ids'], 'attention_mask': output['attention_mask'], 'offsets': output[0].offsets}