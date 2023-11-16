"""
Image/Text processor class for ALIGN
"""
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

class AlignProcessor(ProcessorMixin):
    """
    Constructs an ALIGN processor which wraps [`EfficientNetImageProcessor`] and
    [`BertTokenizer`]/[`BertTokenizerFast`] into a single processor that interits both the image processor and
    tokenizer functionalities. See the [`~AlignProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more
    information.

    Args:
        image_processor ([`EfficientNetImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`BertTokenizer`, `BertTokenizerFast`]):
            The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'EfficientNetImageProcessor'
    tokenizer_class = ('BertTokenizer', 'BertTokenizerFast')

    def __init__(self, image_processor, tokenizer):
        if False:
            while True:
                i = 10
        super().__init__(image_processor, tokenizer)

    def __call__(self, text=None, images=None, padding='max_length', max_length=64, return_tensors=None, **kwargs):
        if False:
            return 10
        '\n        Main method to prepare text(s) and image(s) to be fed as input to the model. This method forwards the `text`\n        and `kwargs` arguments to BertTokenizerFast\'s [`~BertTokenizerFast.__call__`] if `text` is not `None` to encode\n        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to\n        EfficientNetImageProcessor\'s [`~EfficientNetImageProcessor.__call__`] if `images` is not `None`. Please refer\n        to the doctsring of the above two methods for more information.\n\n        Args:\n            text (`str`, `List[str]`):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings\n                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set\n                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):\n                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch\n                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a\n                number of channels, H and W are image height and width.\n            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `max_length`):\n                Activates and controls padding for tokenization of input text. Choose between [`True` or `\'longest\'`,\n                `\'max_length\'`, `False` or `\'do_not_pad\'`]\n            max_length (`int`, *optional*, defaults to `max_length`):\n                Maximum padding value to use to pad the input text during tokenization.\n\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors of a particular framework. Acceptable values are:\n\n                - `\'tf\'`: Return TensorFlow `tf.constant` objects.\n                - `\'pt\'`: Return PyTorch `torch.Tensor` objects.\n                - `\'np\'`: Return NumPy `np.ndarray` objects.\n                - `\'jax\'`: Return JAX `jnp.ndarray` objects.\n\n        Returns:\n            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:\n\n            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.\n            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when\n              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not\n              `None`).\n            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.\n        '
        if text is None and images is None:
            raise ValueError('You have to specify either text or images. Both cannot be none.')
        if text is not None:
            encoding = self.tokenizer(text, padding=padding, max_length=max_length, return_tensors=return_tensors, **kwargs)
        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)
        if text is not None and images is not None:
            encoding['pixel_values'] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
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