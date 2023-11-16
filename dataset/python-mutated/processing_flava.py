"""
Image/Text processor class for FLAVA
"""
import warnings
from typing import List, Optional, Union
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

class FlavaProcessor(ProcessorMixin):
    """
    Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

    [`FlavaProcessor`] offers all the functionalities of [`FlavaImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~FlavaProcessor.__call__`] and [`~FlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`FlavaImageProcessor`], *optional*): The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*): The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'FlavaImageProcessor'
    tokenizer_class = ('BertTokenizer', 'BertTokenizerFast')

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if False:
            print('Hello World!')
        feature_extractor = None
        if 'feature_extractor' in kwargs:
            warnings.warn('The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor` instead.', FutureWarning)
            feature_extractor = kwargs.pop('feature_extractor')
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError('You need to specify an `image_processor`.')
        if tokenizer is None:
            raise ValueError('You need to specify a `tokenizer`.')
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def __call__(self, images: Optional[ImageInput]=None, text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=False, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_image_mask: Optional[bool]=None, return_codebook_pixels: Optional[bool]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs):
        if False:
            return 10
        '\n        This method uses [`FlavaImageProcessor.__call__`] method to prepare image(s) for the model, and\n        [`BertTokenizerFast.__call__`] to prepare text for the model.\n\n        Please refer to the docstring of the above two methods for more information.\n        '
        if text is None and images is None:
            raise ValueError('You have to specify either text or images. Both cannot be none.')
        if text is not None:
            encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
        if images is not None:
            image_features = self.image_processor(images, return_image_mask=return_image_mask, return_codebook_pixels=return_codebook_pixels, return_tensors=return_tensors, **kwargs)
        if text is not None and images is not None:
            encoding.update(image_features)
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        if False:
            return 10
        "\n        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            return 10
        "\n        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to\n        the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            while True:
                i = 10
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @property
    def feature_extractor_class(self):
        if False:
            print('Hello World!')
        warnings.warn('`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.', FutureWarning)
        return self.image_processor_class

    @property
    def feature_extractor(self):
        if False:
            print('Hello World!')
        warnings.warn('`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.', FutureWarning)
        return self.image_processor