"""
Audio/Text processor class for SeamlessM4T
"""
from ...processing_utils import ProcessorMixin

class SeamlessM4TProcessor(ProcessorMixin):
    """
    Constructs a SeamlessM4T processor which wraps a SeamlessM4T feature extractor and a SeamlessM4T tokenizer into a
    single processor.

    [`SeamlessM4TProcessor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and
    [`SeamlessM4TTokenizerFast`]. See the [`~SeamlessM4TProcessor.__call__`] and [`~SeamlessM4TProcessor.decode`] for
    more information.

    Args:
        feature_extractor ([`SeamlessM4TFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`SeamlessM4TTokenizerFast`]):
            The tokenizer is a required input.
    """
    feature_extractor_class = 'SeamlessM4TFeatureExtractor'
    tokenizer_class = ('SeamlessM4TTokenizer', 'SeamlessM4TTokenizerFast')

    def __init__(self, feature_extractor, tokenizer):
        if False:
            while True:
                i = 10
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, text=None, audios=None, src_lang=None, tgt_lang=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`\n        and `kwargs` arguments to SeamlessM4TTokenizerFast\'s [`~SeamlessM4TTokenizerFast.__call__`] if `text` is not\n        `None` to encode the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to\n        SeamlessM4TFeatureExtractor\'s [`~SeamlessM4TFeatureExtractor.__call__`] if `audios` is not `None`. Please refer\n        to the doctsring of the above two methods for more information.\n\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings\n                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set\n                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            audios (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):\n                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case\n                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,\n                and T the sample length of the audio.\n            src_lang (`str`, *optional*):\n                The language code of the input texts/audios. If not specified, the last `src_lang` specified will be\n                used.\n            tgt_lang (`str`, *optional*):\n                The code of the target language. If not specified, the last `tgt_lang` specified will be used.\n            kwargs (*optional*):\n                Remaining dictionary of keyword arguments that will be passed to the feature extractor and/or the\n                tokenizer.\n        Returns:\n            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:\n\n            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.\n            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when\n              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not\n              `None`).\n            - **input_features** -- Audio input features to be fed to a model. Returned when `audios` is not `None`.\n        '
        sampling_rate = kwargs.pop('sampling_rate', None)
        if text is None and audios is None:
            raise ValueError('You have to specify either text or audios. Both cannot be none.')
        elif text is not None and audios is not None:
            raise ValueError('Text and audios are mututally exclusive when passed to `SeamlessM4T`. Specify one or another.')
        elif text is not None:
            if tgt_lang is not None:
                self.tokenizer.tgt_lang = tgt_lang
            if src_lang is not None:
                self.tokenizer.src_lang = src_lang
            encoding = self.tokenizer(text, **kwargs)
            return encoding
        else:
            encoding = self.feature_extractor(audios, sampling_rate=sampling_rate, **kwargs)
            return encoding

    def batch_decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].\n        Please refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            return 10
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))