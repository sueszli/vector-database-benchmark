"""
Processor class for BridgeTower.
"""
from typing import List, Optional, Union
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

class BridgeTowerProcessor(ProcessorMixin):
    """
    Constructs a BridgeTower processor which wraps a Roberta tokenizer and BridgeTower image processor into a single
    processor.

    [`BridgeTowerProcessor`] offers all the functionalities of [`BridgeTowerImageProcessor`] and
    [`RobertaTokenizerFast`]. See the docstring of [`~BridgeTowerProcessor.__call__`] and
    [`~BridgeTowerProcessor.decode`] for more information.

    Args:
        image_processor (`BridgeTowerImageProcessor`):
            An instance of [`BridgeTowerImageProcessor`]. The image processor is a required input.
        tokenizer (`RobertaTokenizerFast`):
            An instance of ['RobertaTokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'BridgeTowerImageProcessor'
    tokenizer_class = ('RobertaTokenizer', 'RobertaTokenizerFast')

    def __init__(self, image_processor, tokenizer):
        if False:
            print('Hello World!')
        super().__init__(image_processor, tokenizer)

    def __call__(self, images, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchEncoding:
        if False:
            print('Hello World!')
        '\n        This method uses [`BridgeTowerImageProcessor.__call__`] method to prepare image(s) for the model, and\n        [`RobertaTokenizerFast.__call__`] to prepare text for the model.\n\n        Please refer to the docstring of the above two methods for more information.\n        '
        encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors, do_normalize=True, do_center_crop=True, **kwargs)
        encoding.update(encoding_image_processor)
        return encoding

    def batch_decode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer\n        to the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            i = 10
            return i + 15
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))