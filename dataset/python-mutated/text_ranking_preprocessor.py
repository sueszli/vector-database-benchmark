from typing import Any, Dict
from transformers import AutoTokenizer
from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert

class TextRankingPreprocessorBase(Preprocessor):

    def __init__(self, mode: str=ModeKeys.INFERENCE, first_sequence='source_sentence', second_sequence='sentences_to_compare', label='labels', qid='qid'):
        if False:
            i = 10
            return i + 15
        'The tokenizer preprocessor class for the text ranking preprocessor.\n\n        Args:\n            first_sequence(str, `optional`): The key of the first sequence.\n            second_sequence(str, `optional`): The key of the second sequence.\n            label(str, `optional`): The keys of the label columns, default `labels`.\n            qid(str, `optional`): The qid info.\n            mode: The mode for the preprocessor.\n        '
        super().__init__(mode)
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.label = label
        self.qid = qid

@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.text_ranking)
class TextRankingTransformersPreprocessor(TextRankingPreprocessorBase):

    def __init__(self, model_dir: str, mode: str=ModeKeys.INFERENCE, first_sequence='source_sentence', second_sequence='sentences_to_compare', label='labels', qid='qid', max_length=None, padding='max_length', truncation=True, use_fast=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "The tokenizer preprocessor class for the text ranking preprocessor.\n\n        Args:\n            model_dir(str, `optional`): The model dir used to parse the label mapping, can be None.\n            max_length: The max sequence length which the model supported,\n                will be passed into tokenizer as the 'max_length' param.\n        "
        super().__init__(mode=mode, first_sequence=first_sequence, second_sequence=second_sequence, label=label, qid=qid)
        self.model_dir = model_dir
        self.sequence_length = max_length if max_length is not None else kwargs.get('sequence_length', 128)
        kwargs.pop('sequence_length', None)
        self.tokenize_kwargs = kwargs
        self.tokenize_kwargs['padding'] = padding
        self.tokenize_kwargs['truncation'] = truncation
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=use_fast)

    @type_assert(object, dict)
    def __call__(self, data: Dict, **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        sentence1 = data.get(self.first_sequence)
        sentence2 = data.get(self.second_sequence)
        labels = data.get(self.label)
        qid = data.get(self.qid)
        if isinstance(sentence2, str):
            sentence2 = [sentence2]
        if isinstance(sentence1, str):
            sentence1 = [sentence1]
        sentence1 = sentence1 * len(sentence2)
        kwargs['max_length'] = kwargs.get('max_length', kwargs.pop('sequence_length', self.sequence_length))
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        self.tokenize_kwargs.update(kwargs)
        feature = self.tokenizer(sentence1, sentence2, **self.tokenize_kwargs)
        if labels is not None:
            feature['labels'] = labels
        if qid is not None:
            feature['qid'] = qid
        return feature