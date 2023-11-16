"""
Processor class for MarkupLM.
"""
from typing import Optional, Union
from ...file_utils import TensorType
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy

class MarkupLMProcessor(ProcessorMixin):
    """
    Constructs a MarkupLM processor which combines a MarkupLM feature extractor and a MarkupLM tokenizer into a single
    processor.

    [`MarkupLMProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`MarkupLMFeatureExtractor`] to extract nodes and corresponding xpaths from one or more HTML strings.
    Next, these are provided to [`MarkupLMTokenizer`] or [`MarkupLMTokenizerFast`], which turns them into token-level
    `input_ids`, `attention_mask`, `token_type_ids`, `xpath_tags_seq` and `xpath_subs_seq`.

    Args:
        feature_extractor (`MarkupLMFeatureExtractor`):
            An instance of [`MarkupLMFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`MarkupLMTokenizer` or `MarkupLMTokenizerFast`):
            An instance of [`MarkupLMTokenizer`] or [`MarkupLMTokenizerFast`]. The tokenizer is a required input.
        parse_html (`bool`, *optional*, defaults to `True`):
            Whether or not to use `MarkupLMFeatureExtractor` to parse HTML strings into nodes and corresponding xpaths.
    """
    feature_extractor_class = 'MarkupLMFeatureExtractor'
    tokenizer_class = ('MarkupLMTokenizer', 'MarkupLMTokenizerFast')
    parse_html = True

    def __call__(self, html_strings=None, nodes=None, xpaths=None, node_labels=None, questions=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchEncoding:
        if False:
            print('Hello World!')
        '\n        This method first forwards the `html_strings` argument to [`~MarkupLMFeatureExtractor.__call__`]. Next, it\n        passes the `nodes` and `xpaths` along with the additional arguments to [`~MarkupLMTokenizer.__call__`] and\n        returns the output.\n\n        Optionally, one can also provide a `text` argument which is passed along as first sequence.\n\n        Please refer to the docstring of the above two methods for more information.\n        '
        if self.parse_html:
            if html_strings is None:
                raise ValueError('Make sure to pass HTML strings in case `parse_html` is set to `True`')
            if nodes is not None or xpaths is not None or node_labels is not None:
                raise ValueError("Please don't pass nodes, xpaths nor node labels in case `parse_html` is set to `True`")
            features = self.feature_extractor(html_strings)
            nodes = features['nodes']
            xpaths = features['xpaths']
        else:
            if html_strings is not None:
                raise ValueError('You have passed HTML strings but `parse_html` is set to `False`.')
            if nodes is None or xpaths is None:
                raise ValueError('Make sure to pass nodes and xpaths in case `parse_html` is set to `False`')
        if questions is not None and self.parse_html:
            if isinstance(questions, str):
                questions = [questions]
        encoded_inputs = self.tokenizer(text=questions if questions is not None else nodes, text_pair=nodes if questions is not None else None, xpaths=xpaths, node_labels=node_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
        return encoded_inputs

    def batch_decode(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer\n        to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the\n        docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            print('Hello World!')
        tokenizer_input_names = self.tokenizer.model_input_names
        return tokenizer_input_names