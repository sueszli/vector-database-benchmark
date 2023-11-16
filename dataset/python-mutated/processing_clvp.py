"""
Processor class for CLVP
"""
from ...processing_utils import ProcessorMixin

class ClvpProcessor(ProcessorMixin):
    """
    Constructs a CLVP processor which wraps a CLVP Feature Extractor and a CLVP Tokenizer into a single processor.

    [`ClvpProcessor`] offers all the functionalities of [`ClvpFeatureExtractor`] and [`ClvpTokenizer`]. See the
    [`~ClvpProcessor.__call__`], [`~ClvpProcessor.decode`] and [`~ClvpProcessor.batch_decode`] for more information.

    Args:
        feature_extractor (`ClvpFeatureExtractor`):
            An instance of [`ClvpFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`ClvpTokenizer`):
            An instance of [`ClvpTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = 'ClvpFeatureExtractor'
    tokenizer_class = 'ClvpTokenizer'
    model_input_names = ['input_ids', 'input_features', 'attention_mask']

    def __init__(self, feature_extractor, tokenizer):
        if False:
            i = 10
            return i + 15
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        '\n        Forwards the `audio` and `sampling_rate` arguments to [`~ClvpFeatureExtractor.__call__`] and the `text`\n        argument to [`~ClvpTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more\n        information.\n        '
        raw_speech = kwargs.pop('raw_speech', None)
        sampling_rate = kwargs.pop('sampling_rate', None)
        text = kwargs.pop('text', None)
        if raw_speech is None and text is None:
            raise ValueError('You need to specify either an `raw_speech` or `text` input to process.')
        if raw_speech is not None:
            inputs = self.feature_extractor(raw_speech, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)
        if text is None:
            return inputs
        elif raw_speech is None:
            return encodings
        else:
            inputs['input_ids'] = encodings['input_ids']
            inputs['attention_mask'] = encodings['attention_mask']
            return inputs

    def batch_decode(self, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        This method forwards all its arguments to ClvpTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer\n        to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to ClvpTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the\n        docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)