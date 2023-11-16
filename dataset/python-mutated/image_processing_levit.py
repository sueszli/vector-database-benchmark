"""Image processor class for LeViT."""
from typing import Dict, Iterable, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ChannelDimension, ImageInput, PILImageResampling, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, logging
logger = logging.get_logger(__name__)

class LevitImageProcessor(BaseImageProcessor):
    """
    Constructs a LeViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Wwhether to resize the shortest edge of the input to int(256/224 *`size`). Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]`, *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the output image after resizing. If size is a dict with keys "width" and "height", the image will
            be resized to `(size["height"], size["width"])`. If size is a dict with key "shortest_edge", the shortest
            edge value `c` is rescaled to `int(c * (256/224))`. The smaller edge of the image will be matched to this
            value i.e, if height > width, then image will be rescaled to `(size["shortest_egde"] * height / width,
            size["shortest_egde"])`. Can be overridden by the `size` parameter in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether or not to center crop the input to `(crop_size["height"], crop_size["width"])`. Can be overridden
            by the `do_center_crop` parameter in the `preprocess` method.
        crop_size (`Dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Desired image size after `center_crop`. Can be overridden by the `crop_size` parameter in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`List[int]`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`List[int]`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BICUBIC, do_center_crop: bool=True, crop_size: Dict[str, int]=None, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Optional[Union[float, Iterable[float]]]=IMAGENET_DEFAULT_MEAN, image_std: Optional[Union[float, Iterable[float]]]=IMAGENET_DEFAULT_STD, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        size = size if size is not None else {'shortest_edge': 224}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {'height': 224, 'width': 224}
        crop_size = get_size_dict(crop_size, param_name='crop_size')
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Resize an image.\n\n        If size is a dict with keys "width" and "height", the image will be resized to `(size["height"],\n        size["width"])`.\n\n        If size is a dict with key "shortest_edge", the shortest edge value `c` is rescaled to `int(c * (256/224))`.\n        The smaller edge of the image will be matched to this value i.e, if height > width, then image will be rescaled\n        to `(size["shortest_egde"] * height / width, size["shortest_egde"])`.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Size of the output image after resizing. If size is a dict with keys "width" and "height", the image\n                will be resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value\n                `c` is rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value\n                i.e, if height > width, then image will be rescaled to (size * height / width, size).\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):\n                Resampling filter to use when resiizing the image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        size_dict = get_size_dict(size, default_to_square=False)
        if 'shortest_edge' in size:
            shortest_edge = int(256 / 224 * size['shortest_edge'])
            output_size = get_resize_output_image_size(image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format)
            size_dict = {'height': output_size[0], 'width': output_size[1]}
        if 'height' not in size_dict or 'width' not in size_dict:
            raise ValueError(f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {size_dict.keys()}")
        return resize(image, size=(size_dict['height'], size_dict['width']), resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def preprocess(self, images: ImageInput, do_resize: Optional[bool]=None, size: Optional[Dict[str, int]]=None, resample: PILImageResampling=None, do_center_crop: Optional[bool]=None, crop_size: Optional[Dict[str, int]]=None, do_rescale: Optional[bool]=None, rescale_factor: Optional[float]=None, do_normalize: Optional[bool]=None, image_mean: Optional[Union[float, Iterable[float]]]=None, image_std: Optional[Union[float, Iterable[float]]]=None, return_tensors: Optional[TensorType]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> BatchFeature:
        if False:
            for i in range(10):
                print('nop')
        '\n        Preprocess an image or batch of images to be used as input to a LeViT model.\n\n        Args:\n            images (`ImageInput`):\n                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging\n                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Size of the output image after resizing. If size is a dict with keys "width" and "height", the image\n                will be resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value\n                `c` is rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value\n                i.e, if height > width, then image will be rescaled to (size * height / width, size).\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):\n                Resampling filter to use when resiizing the image.\n            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):\n                Whether to center crop the image.\n            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):\n                Size of the output image after center cropping. Crops images to (crop_size["height"],\n                crop_size["width"]).\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image pixel values by `rescaling_factor` - typical to values between 0 and 1.\n            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):\n                Factor to rescale the image pixel values by.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image pixel values by `image_mean` and `image_std`.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Mean to normalize the image pixel values by.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Standard deviation to normalize the image pixel values by.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`str` or `ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name='crop_size')
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_resize and size is None:
            raise ValueError('Size must be specified if do_resize is True.')
        if do_center_crop and crop_size is None:
            raise ValueError('Crop size must be specified if do_center_crop is True.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if do_resize:
            images = [self.resize(image, size, resample, input_data_format=input_data_format) for image in images]
        if do_center_crop:
            images = [self.center_crop(image, crop_size, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)