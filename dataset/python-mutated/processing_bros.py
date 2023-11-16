"""
Processor class for Bros.
"""
from typing import List, Optional, Union
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

class BrosProcessor(ProcessorMixin):
    """
    Constructs a Bros processor which wraps a BERT tokenizer.

    [`BrosProcessor`] offers all the functionalities of [`BertTokenizerFast`]. See the docstring of
    [`~BrosProcessor.__call__`] and [`~BrosProcessor.decode`] for more information.

    Args:
        tokenizer (`BertTokenizerFast`, *optional*):
            An instance of ['BertTokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ['tokenizer']
    tokenizer_class = ('BertTokenizer', 'BertTokenizerFast')

    def __init__(self, tokenizer=None, **kwargs):
        if False:
            while True:
                i = 10
        if tokenizer is None:
            raise ValueError('You need to specify a `tokenizer`.')
        super().__init__(tokenizer)

    def __call__(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchEncoding:
        if False:
            i = 10
            return i + 15
        '\n        This method uses [`BertTokenizerFast.__call__`] to prepare text for the model.\n\n        Please refer to the docstring of the above two methods for more information.\n        '
        encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
        return encoding

    def batch_decode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to\n        the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            i = 10
            return i + 15
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names))