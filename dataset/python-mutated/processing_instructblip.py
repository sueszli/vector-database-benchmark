"""
Processor class for InstructBLIP. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""
import os
from typing import List, Optional, Union
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType
from ..auto import AutoTokenizer

class InstructBlipProcessor(ProcessorMixin):
    """
    Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
    processor.

    [`InstructBlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the
    docstring of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        qformer_tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'BlipImageProcessor'
    tokenizer_class = 'AutoTokenizer'

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        if False:
            while True:
                i = 10
        super().__init__(image_processor, tokenizer)
        self.qformer_tokenizer = qformer_tokenizer

    def __call__(self, images: ImageInput=None, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_token_type_ids: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchFeature:
        if False:
            i = 10
            return i + 15
        '\n        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and\n        [`BertTokenizerFast.__call__`] to prepare text for the model.\n\n        Please refer to the docstring of the above two methods for more information.\n        '
        if images is None and text is None:
            raise ValueError('You have to specify at least images or text.')
        encoding = BatchFeature()
        if text is not None:
            text_encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_token_type_ids=return_token_type_ids, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
            encoding.update(text_encoding)
            qformer_text_encoding = self.qformer_tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_token_type_ids=return_token_type_ids, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
            encoding['qformer_input_ids'] = qformer_text_encoding.pop('input_ids')
            encoding['qformer_attention_mask'] = qformer_text_encoding.pop('attention_mask')
        if images is not None:
            image_encoding = self.image_processor(images, return_tensors=return_tensors)
            encoding.update(image_encoding)
        return encoding

    def batch_decode(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer\n        to the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            return 10
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        if False:
            i = 10
            return i + 15
        if os.path.isfile(save_directory):
            raise ValueError(f'Provided path ({save_directory}) should be a directory, not a file')
        os.makedirs(save_directory, exist_ok=True)
        qformer_tokenizer_path = os.path.join(save_directory, 'qformer_tokenizer')
        self.qformer_tokenizer.save_pretrained(qformer_tokenizer_path)
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if False:
            return 10
        qformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='qformer_tokenizer')
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        args.append(qformer_tokenizer)
        return cls(*args)