from typing import Any, Dict, List, Tuple, Union
import numpy as np
from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger
from .transformers_tokenizer import NLPTokenizer
from .utils import labels_to_id, parse_text_and_label
logger = get_logger()

class TextClassificationPreprocessorBase(Preprocessor):

    def __init__(self, model_dir=None, first_sequence: str=None, second_sequence: str=None, label: str='label', label2id: Dict=None, mode: str=ModeKeys.INFERENCE, keep_original_columns: List[str]=None):
        if False:
            i = 10
            return i + 15
        'The base class for the text classification preprocessor.\n\n        Args:\n            model_dir(str, `optional`): The model dir used to parse the label mapping, can be None.\n            first_sequence(str, `optional`): The key of the first sequence.\n            second_sequence(str, `optional`): The key of the second sequence.\n            label(str, `optional`): The keys of the label columns, default is `label`\n            label2id: (dict, `optional`): The optional label2id mapping\n            mode(str, `optional`): The mode for the preprocessor\n            keep_original_columns(List[str], `optional`): The original columns to keep,\n                only available when the input is a `dict`, default None\n        '
        super().__init__(mode)
        self.model_dir = model_dir
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.label = label
        self.label2id = label2id
        self.keep_original_columns = keep_original_columns
        if self.label2id is None and self.model_dir is not None:
            self.label2id = parse_label_mapping(self.model_dir)
        logger.info(f'The key of sentence1: {self.first_sequence}, The key of sentence2: {self.second_sequence}, The key of label: {self.label}')
        if self.first_sequence is None:
            logger.warning('[Important] first_sequence attribute is not set, this will cause an error if your input is a dict.')

    @property
    def id2label(self):
        if False:
            i = 10
            return i + 15
        'Return the id2label mapping according to the label2id mapping.\n\n        @return: The id2label mapping if exists.\n        '
        if self.label2id is not None:
            return {id: label for (label, id) in self.label2id.items()}
        return None

    def __call__(self, data: Union[str, Tuple, Dict], **kwargs) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'process the raw input data\n\n        Args:\n            data (tuple): [sentence1, sentence2]\n                sentence1 (str): a sentence\n                sentence2 (str): a sentence\n\n        Returns:\n            Dict[str, Any]: the preprocessed data\n        '
        (text_a, text_b, labels) = parse_text_and_label(data, self.mode, self.first_sequence, self.second_sequence, self.label)
        output = self._tokenize_text(text_a, text_b, **kwargs)
        output = {k: np.array(v) if isinstance(v, list) else v for (k, v) in output.items()}
        labels_to_id(labels, output, self.label2id)
        if self.keep_original_columns and isinstance(data, dict):
            for column in self.keep_original_columns:
                output[column] = data[column]
        return output

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        if False:
            while True:
                i = 10
        'Tokenize the text.\n\n        Args:\n            sequence1: The first sequence.\n            sequence2: The second sequence which may be None.\n\n        Returns:\n            The encoded sequence.\n        '
        raise NotImplementedError()

@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.nli_tokenizer)
@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.sen_sim_tokenizer)
@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.bert_seq_cls_tokenizer)
@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.sen_cls_tokenizer)
class TextClassificationTransformersPreprocessor(TextClassificationPreprocessorBase):

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        return self.nlp_tokenizer(sequence1, sequence2, **kwargs)

    def __init__(self, model_dir=None, first_sequence: str=None, second_sequence: str=None, label: Union[str, List]='label', label2id: Dict=None, mode: str=ModeKeys.INFERENCE, max_length: int=None, use_fast: bool=None, keep_original_columns=None, **kwargs):
        if False:
            return 10
        "The tokenizer preprocessor used in sequence classification.\n\n        Args:\n            use_fast: Whether to use the fast tokenizer or not.\n            max_length: The max sequence length which the model supported,\n                will be passed into tokenizer as the 'max_length' param.\n            **kwargs: Extra args input into the tokenizer's __call__ method.\n        "
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['max_length'] = max_length if max_length is not None else kwargs.get('sequence_length', 128)
        kwargs.pop('sequence_length', None)
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)
        super().__init__(model_dir, first_sequence, second_sequence, label, label2id, mode, keep_original_columns)