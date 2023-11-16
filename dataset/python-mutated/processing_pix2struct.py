"""
Processor class for Pix2Struct.
"""
from typing import List, Optional, Union
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

class Pix2StructProcessor(ProcessorMixin):
    """
    Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single
    processor.

    [`Pix2StructProcessor`] offers all the functionalities of [`Pix2StructImageProcessor`] and [`T5TokenizerFast`]. See
    the docstring of [`~Pix2StructProcessor.__call__`] and [`~Pix2StructProcessor.decode`] for more information.

    Args:
        image_processor (`Pix2StructImageProcessor`):
            An instance of [`Pix2StructImageProcessor`]. The image processor is a required input.
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'Pix2StructImageProcessor'
    tokenizer_class = ('T5Tokenizer', 'T5TokenizerFast')

    def __init__(self, image_processor, tokenizer):
        if False:
            print('Hello World!')
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)

    def __call__(self, images=None, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, max_patches: Optional[int]=2048, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_token_type_ids: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchEncoding:
        if False:
            print('Hello World!')
        '\n        This method uses [`Pix2StructImageProcessor.preprocess`] method to prepare image(s) for the model, and\n        [`T5TokenizerFast.__call__`] to prepare text for the model.\n\n        Please refer to the docstring of the above two methods for more information.\n        '
        if images is None and text is None:
            raise ValueError('You have to specify either images or text.')
        if images is None and (not self.image_processor.is_vqa):
            self.current_processor = self.tokenizer
            text_encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_token_type_ids=return_token_type_ids, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
            return text_encoding
        if not self.image_processor.is_vqa:
            encoding_image_processor = self.image_processor(images, return_tensors=return_tensors, max_patches=max_patches, **kwargs)
        else:
            encoding_image_processor = self.image_processor(images, return_tensors=return_tensors, max_patches=max_patches, header_text=text, **kwargs)
        if text is not None and (not self.image_processor.is_vqa):
            text_encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_token_type_ids=return_token_type_ids, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
            if 'attention_mask' in text_encoding:
                text_encoding['decoder_attention_mask'] = text_encoding.pop('attention_mask')
            if 'input_ids' in text_encoding:
                text_encoding['decoder_input_ids'] = text_encoding.pop('input_ids')
        else:
            text_encoding = None
        if text_encoding is not None:
            encoding_image_processor.update(text_encoding)
        return encoding_image_processor

    def batch_decode(self, *args, **kwargs):
        if False:
            return 10
        "\n        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].\n        Please refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            print('Hello World!')
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))