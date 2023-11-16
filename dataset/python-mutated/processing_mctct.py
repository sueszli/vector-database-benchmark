"""
Speech processor class for M-CTC-T
"""
import warnings
from contextlib import contextmanager
from ....processing_utils import ProcessorMixin

class MCTCTProcessor(ProcessorMixin):
    """
    Constructs a MCTCT processor which wraps a MCTCT feature extractor and a MCTCT tokenizer into a single processor.

    [`MCTCTProcessor`] offers all the functionalities of [`MCTCTFeatureExtractor`] and [`AutoTokenizer`]. See the
    [`~MCTCTProcessor.__call__`] and [`~MCTCTProcessor.decode`] for more information.

    Args:
        feature_extractor (`MCTCTFeatureExtractor`):
            An instance of [`MCTCTFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = 'MCTCTFeatureExtractor'
    tokenizer_class = 'AutoTokenizer'

    def __init__(self, feature_extractor, tokenizer):
        if False:
            while True:
                i = 10
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        When used in normal mode, this method forwards all its arguments to MCTCTFeatureExtractor's\n        [`~MCTCTFeatureExtractor.__call__`] and returns its output. If used in the context\n        [`~MCTCTProcessor.as_target_processor`] this method forwards all its arguments to AutoTokenizer's\n        [`~AutoTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.\n        "
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)
        if 'raw_speech' in kwargs:
            warnings.warn('Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.')
            audio = kwargs.pop('raw_speech')
        else:
            audio = kwargs.pop('audio', None)
        sampling_rate = kwargs.pop('sampling_rate', None)
        text = kwargs.pop('text', None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]
        if audio is None and text is None:
            raise ValueError('You need to specify either an `audio` or `text` input to process.')
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)
        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs['labels'] = encodings['input_ids']
            return inputs

    def batch_decode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        This method forwards all its arguments to AutoTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer\n        to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def pad(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        When used in normal mode, this method forwards all its arguments to MCTCTFeatureExtractor's\n        [`~MCTCTFeatureExtractor.pad`] and returns its output. If used in the context\n        [`~MCTCTProcessor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's\n        [`~PreTrainedTokenizer.pad`]. Please refer to the docstring of the above two methods for more information.\n        "
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)
        input_features = kwargs.pop('input_features', None)
        labels = kwargs.pop('labels', None)
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)
        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features['labels'] = labels['input_ids']
            return input_features

    def decode(self, *args, **kwargs):
        if False:
            return 10
        "\n        This method forwards all its arguments to AutoTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the\n        docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        if False:
            i = 10
            return i + 15
        '\n        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning MCTCT.\n        '
        warnings.warn('`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your labels by using the argument `text` of the regular `__call__` method (either in the same call as your audio inputs, or in a separate call.')
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False