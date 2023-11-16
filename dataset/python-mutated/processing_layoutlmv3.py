"""
Processor class for LayoutLMv3.
"""
import warnings
from typing import List, Optional, Union
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

class LayoutLMv3Processor(ProcessorMixin):
    """
    Constructs a LayoutLMv3 processor which combines a LayoutLMv3 image processor and a LayoutLMv3 tokenizer into a
    single processor.

    [`LayoutLMv3Processor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv3ImageProcessor`] to resize and normalize document images, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutLMv3Tokenizer`] or
    [`LayoutLMv3TokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`LayoutLMv3ImageProcessor`, *optional*):
            An instance of [`LayoutLMv3ImageProcessor`]. The image processor is a required input.
        tokenizer (`LayoutLMv3Tokenizer` or `LayoutLMv3TokenizerFast`, *optional*):
            An instance of [`LayoutLMv3Tokenizer`] or [`LayoutLMv3TokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'LayoutLMv3ImageProcessor'
    tokenizer_class = ('LayoutLMv3Tokenizer', 'LayoutLMv3TokenizerFast')

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
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

    def __call__(self, images, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]=None, text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]]=None, boxes: Union[List[List[int]], List[List[List[int]]]]=None, word_labels: Optional[Union[List[int], List[List[int]]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchEncoding:
        if False:
            return 10
        '\n        This method first forwards the `images` argument to [`~LayoutLMv3ImageProcessor.__call__`]. In case\n        [`LayoutLMv3ImageProcessor`] was initialized with `apply_ocr` set to `True`, it passes the obtained words and\n        bounding boxes along with the additional arguments to [`~LayoutLMv3Tokenizer.__call__`] and returns the output,\n        together with resized and normalized `pixel_values`. In case [`LayoutLMv3ImageProcessor`] was initialized with\n        `apply_ocr` set to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along\n        with the additional arguments to [`~LayoutLMv3Tokenizer.__call__`] and returns the output, together with\n        resized and normalized `pixel_values`.\n\n        Please refer to the docstring of the above two methods for more information.\n        '
        if self.image_processor.apply_ocr and boxes is not None:
            raise ValueError('You cannot provide bounding boxes if you initialized the image processor with apply_ocr set to True.')
        if self.image_processor.apply_ocr and word_labels is not None:
            raise ValueError('You cannot provide word labels if you initialized the image processor with apply_ocr set to True.')
        features = self.image_processor(images=images, return_tensors=return_tensors)
        if text is not None and self.image_processor.apply_ocr and (text_pair is None):
            if isinstance(text, str):
                text = [text]
            text_pair = features['words']
        encoded_inputs = self.tokenizer(text=text if text is not None else features['words'], text_pair=text_pair if text_pair is not None else None, boxes=boxes if boxes is not None else features['boxes'], word_labels=word_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, return_tensors=return_tensors, **kwargs)
        images = features.pop('pixel_values')
        if return_overflowing_tokens is True:
            images = self.get_overflowing_images(images, encoded_inputs['overflow_to_sample_mapping'])
        encoded_inputs['pixel_values'] = images
        return encoded_inputs

    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        if False:
            print('Hello World!')
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])
        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(f'Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}')
        return images_with_overflow

    def batch_decode(self, *args, **kwargs):
        if False:
            return 10
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
            i = 10
            return i + 15
        return ['input_ids', 'bbox', 'attention_mask', 'pixel_values']

    @property
    def feature_extractor_class(self):
        if False:
            print('Hello World!')
        warnings.warn('`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.', FutureWarning)
        return self.image_processor_class

    @property
    def feature_extractor(self):
        if False:
            return 10
        warnings.warn('`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.', FutureWarning)
        return self.image_processor