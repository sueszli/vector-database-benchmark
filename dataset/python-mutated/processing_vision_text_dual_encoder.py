"""
Processor class for VisionTextDualEncoder
"""
import warnings
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

class VisionTextDualEncoderProcessor(ProcessorMixin):
    """
    Constructs a VisionTextDualEncoder processor which wraps an image processor and a tokenizer into a single
    processor.

    [`VisionTextDualEncoderProcessor`] offers all the functionalities of [`AutoImageProcessor`] and [`AutoTokenizer`].
    See the [`~VisionTextDualEncoderProcessor.__call__`] and [`~VisionTextDualEncoderProcessor.decode`] for more
    information.

    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'AutoImageProcessor'
    tokenizer_class = 'AutoTokenizer'

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if False:
            return 10
        feature_extractor = None
        if 'feature_extractor' in kwargs:
            warnings.warn('The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor` instead.', FutureWarning)
            feature_extractor = kwargs.pop('feature_extractor')
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError('You have to specify an image_processor.')
        if tokenizer is None:
            raise ValueError('You have to specify a tokenizer.')
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`\n        and `kwargs` arguments to VisionTextDualEncoderTokenizer\'s [`~PreTrainedTokenizer.__call__`] if `text` is not\n        `None` to encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to\n        AutoImageProcessor\'s [`~AutoImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring\n        of the above two methods for more information.\n\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings\n                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set\n                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).\n            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):\n                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch\n                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a\n                number of channels, H and W are image height and width.\n\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors of a particular framework. Acceptable values are:\n\n                - `\'tf\'`: Return TensorFlow `tf.constant` objects.\n                - `\'pt\'`: Return PyTorch `torch.Tensor` objects.\n                - `\'np\'`: Return NumPy `np.ndarray` objects.\n                - `\'jax\'`: Return JAX `jnp.ndarray` objects.\n\n        Returns:\n            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:\n\n            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.\n            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when\n              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not\n              `None`).\n            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.\n        '
        if text is None and images is None:
            raise ValueError('You have to specify either text or images. Both cannot be none.')
        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
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
            return 10
        "\n        This method forwards all its arguments to VisionTextDualEncoderTokenizer's\n        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to VisionTextDualEncoderTokenizer's [`~PreTrainedTokenizer.decode`].\n        Please refer to the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            i = 10
            return i + 15
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @property
    def feature_extractor_class(self):
        if False:
            i = 10
            return i + 15
        warnings.warn('`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.', FutureWarning)
        return self.image_processor_class

    @property
    def feature_extractor(self):
        if False:
            return 10
        warnings.warn('`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.', FutureWarning)
        return self.image_processor