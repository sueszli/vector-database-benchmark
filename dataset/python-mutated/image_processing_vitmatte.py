"""Image processor class for ViTMatte."""
from typing import List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import pad, to_channel_dimension_format
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, ImageInput, get_image_size, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, logging
logger = logging.get_logger(__name__)

class VitMatteImageProcessor(BaseImageProcessor):
    """
    Constructs a ViTMatte image processor.

    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to make the width and height divisible by `size_divisibility`. Can be overridden
            by the `do_pad` parameter in the `preprocess` method.
        size_divisibility (`int`, *optional*, defaults to 32):
            The width and height of the image will be padded to be divisible by this number.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: bool=True, size_divisibility: int=32, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_pad = do_pad
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.size_divisibility = size_divisibility

    def pad_image(self, image: np.ndarray, size_divisibility: int=32, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Args:\n            image (`np.ndarray`):\n                Image to pad.\n            size_divisibility (`int`, *optional*, defaults to 32):\n                The width and height of the image will be padded to be divisible by this number.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        (height, width) = get_image_size(image, input_data_format)
        if height % size_divisibility != 0 or width % size_divisibility != 0:
            pad_height = size_divisibility - height % size_divisibility
            pad_width = size_divisibility - width % size_divisibility
            padding = ((0, pad_height), (0, pad_width))
            image = pad(image, padding=padding, data_format=data_format, input_data_format=input_data_format)
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_data_format)
        return image

    def preprocess(self, images: ImageInput, trimaps: ImageInput, do_rescale: Optional[bool]=None, rescale_factor: Optional[float]=None, do_normalize: Optional[bool]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, size_divisibility: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: Union[str, ChannelDimension]=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Preprocess an image or batch of images.\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If\n                passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            trimaps (`ImageInput`):\n                Trimap to preprocess.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image values between [0 - 1].\n            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):\n                Rescale factor to rescale the image by if `do_rescale` is set to `True`.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Image mean to use if `do_normalize` is set to `True`.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Image standard deviation to use if `do_normalize` is set to `True`.\n            do_pad (`bool`, *optional*, defaults to `self.do_pad`):\n                Whether to pad the image.\n            size_divisibility (`int`, *optional*, defaults to `self.size_divisibility`):\n                The size divisibility to pad the image to if `do_pad` is set to `True`.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                - Unset: Return a list of `np.ndarray`.\n                - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_pad = do_pad if do_pad is not None else self.do_pad
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size_divisibility = size_divisibility if size_divisibility is not None else self.size_divisibility
        images = make_list_of_images(images)
        trimaps = make_list_of_images(trimaps, expected_ndims=2)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if not valid_images(trimaps):
            raise ValueError('Invalid trimap type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_pad and size_divisibility is None:
            raise ValueError('Size divisilibyt must be specified if do_pad is True.')
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        images = [to_numpy_array(image) for image in images]
        trimaps = [to_numpy_array(trimap) for trimap in trimaps]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
            trimaps = [self.rescale(image=trimap, scale=rescale_factor, input_data_format=input_data_format) for trimap in trimaps]
        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format) for image in images]
        images = [np.concatenate([image, np.expand_dims(trimap, axis=-1)], axis=-1) for (image, trimap) in zip(images, trimaps)]
        if do_pad:
            images = [self.pad_image(image, size_divisibility=size_divisibility, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image=image, channel_dim=data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)