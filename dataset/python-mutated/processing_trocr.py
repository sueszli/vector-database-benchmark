"""
Processor class for TrOCR.
"""
import warnings
from contextlib import contextmanager
from ...processing_utils import ProcessorMixin

class TrOCRProcessor(ProcessorMixin):
    """
    Constructs a TrOCR processor which wraps a vision image processor and a TrOCR tokenizer into a single processor.

    [`TrOCRProcessor`] offers all the functionalities of [`ViTImageProcessor`/`DeiTImageProcessor`] and
    [`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [`~TrOCRProcessor.__call__`] and [`~TrOCRProcessor.decode`] for
    more information.

    Args:
        image_processor ([`ViTImageProcessor`/`DeiTImageProcessor`], *optional*):
            An instance of [`ViTImageProcessor`/`DeiTImageProcessor`]. The image processor is a required input.
        tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`], *optional*):
            An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'AutoImageProcessor'
    tokenizer_class = 'AutoTokenizer'

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
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's\n        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context\n        [`~TrOCRProcessor.as_target_processor`] this method forwards all its arguments to TrOCRTokenizer's\n        [`~TrOCRTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.\n        "
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)
        images = kwargs.pop('images', None)
        text = kwargs.pop('text', None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]
        if images is None and text is None:
            raise ValueError('You need to specify either an `images` or `text` input to process.')
        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)
        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs['labels'] = encodings['input_ids']
            return inputs

    def batch_decode(self, *args, **kwargs):
        if False:
            return 10
        "\n        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer\n        to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the\n        docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        if False:
            i = 10
            return i + 15
        '\n        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.\n        '
        warnings.warn('`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your labels by using the argument `text` of the regular `__call__` method (either in the same call as your images inputs, or in a separate call.')
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    @property
    def feature_extractor_class(self):
        if False:
            return 10
        warnings.warn('`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.', FutureWarning)
        return self.image_processor_class

    @property
    def feature_extractor(self):
        if False:
            print('Hello World!')
        warnings.warn('`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.', FutureWarning)
        return self.image_processor