"""
Audio/Text processor class for CLAP
"""
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

class ClapProcessor(ProcessorMixin):
    """
    Constructs a CLAP processor which wraps a CLAP feature extractor and a RoBerta tokenizer into a single processor.

    [`ClapProcessor`] offers all the functionalities of [`ClapFeatureExtractor`] and [`RobertaTokenizerFast`]. See the
    [`~ClapProcessor.__call__`] and [`~ClapProcessor.decode`] for more information.

    Args:
        feature_extractor ([`ClapFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`RobertaTokenizerFast`]):
            The tokenizer is a required input.
    """
    feature_extractor_class = 'ClapFeatureExtractor'
    tokenizer_class = ('RobertaTokenizer', 'RobertaTokenizerFast')

    def __init__(self, feature_extractor, tokenizer):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, text=None, audios=None, return_tensors=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`\n        and `kwargs` arguments to RobertaTokenizerFast\'s [`~RobertaTokenizerFast.__call__`] if `text` is not `None` to\n        encode the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to\n        ClapFeatureExtractor\'s [`~ClapFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the\n        doctsring of the above two methods for more information.\n\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings\n                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set\n                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            audios (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):\n                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case\n                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,\n                and T the sample length of the audio.\n\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors of a particular framework. Acceptable values are:\n\n                - `\'tf\'`: Return TensorFlow `tf.constant` objects.\n                - `\'pt\'`: Return PyTorch `torch.Tensor` objects.\n                - `\'np\'`: Return NumPy `np.ndarray` objects.\n                - `\'jax\'`: Return JAX `jnp.ndarray` objects.\n\n        Returns:\n            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:\n\n            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.\n            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when\n              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not\n              `None`).\n            - **audio_features** -- Audio features to be fed to a model. Returned when `audios` is not `None`.\n        '
        sampling_rate = kwargs.pop('sampling_rate', None)
        if text is None and audios is None:
            raise ValueError('You have to specify either text or audios. Both cannot be none.')
        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        if audios is not None:
            audio_features = self.feature_extractor(audios, sampling_rate=sampling_rate, return_tensors=return_tensors, **kwargs)
        if text is not None and audios is not None:
            encoding['input_features'] = audio_features.input_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**audio_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            return 10
        "\n        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer\n        to the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            print('Hello World!')
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))