"""Image processor class for PoolFormer."""
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ChannelDimension, ImageInput, PILImageResampling, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_vision_available, logging
if is_vision_available():
    import PIL
logger = logging.get_logger(__name__)

class PoolFormerImageProcessor(BaseImageProcessor):
    """
    Constructs a PoolFormer image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method. If crop_pct is
            unset:
            - size is `{"height": h, "width": w}`: the image is resized to `(h, w)`.
            - size is `{"shortest_edge": s}`: the shortest edge of the image is resized to s whilst maintaining the
              aspect ratio.

            If crop_pct is set:
            - size is `{"height": h, "width": w}`: the image is resized to `(int(floor(h/crop_pct)),
              int(floor(w/crop_pct)))`
            - size is `{"height": c, "width": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`
              whilst maintaining the aspect ratio.
            - size is `{"shortest_edge": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`
              whilst maintaining the aspect ratio.
        crop_pct (`float`, *optional*, defaults to 0.9):
            Percentage of the image to crop from the center. Can be overridden by `crop_pct` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image
            is padded with 0's and then center cropped. Can be overridden by `do_center_crop` in the `preprocess`
            method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying center crop. Only has an effect if `do_center_crop` is set to `True`. Can
            be overridden by the `crop_size` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_resize: bool=True, size: Dict[str, int]=None, crop_pct: int=0.9, resample: PILImageResampling=PILImageResampling.BICUBIC, do_center_crop: bool=True, crop_size: Dict[str, int]=None, rescale_factor: Union[int, float]=1 / 255, do_rescale: bool=True, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        size = size if size is not None else {'shortest_edge': 224}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {'height': 224, 'width': 224}
        crop_size = get_size_dict(crop_size, param_name='crop_size')
        self.do_resize = do_resize
        self.size = size
        self.crop_pct = crop_pct
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def resize(self, image: np.ndarray, size: Dict[str, int], crop_pct: Optional[float]=None, resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Resize an image.\n\n        If crop_pct is unset:\n            - size is `{"height": h, "width": w}`: the image is resized to `(h, w)`.\n            - size is `{"shortest_edge": s}`: the shortest edge of the image is resized to s whilst maintaining the\n              aspect ratio.\n\n        if crop_pct is set:\n            - size is `{"height": h, "width": w}`: the image is resized to `(int(floor(h/crop_pct)),\n              int(floor(w/crop_pct)))`\n            - size is `{"height": c, "width": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`\n              whilst maintaining the aspect ratio.\n            - size is `{"shortest_edge": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`\n              whilst maintaining the aspect ratio.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Size of the output image.\n            crop_pct (`float`, *optional*):\n                Percentage of the image that will be cropped from the center. If set, the image is resized\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):\n                Resampling filter to use when resizing the image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        size = get_size_dict(size, default_to_square=False)
        if 'shortest_edge' not in size and ('height' not in size or 'width' not in size):
            raise ValueError(f"size must contain 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        if crop_pct is not None:
            if 'shortest_edge' in size:
                scale_size = int(size['shortest_edge'] / crop_pct)
            elif 'height' in size and 'width' in size:
                if size['height'] == size['width']:
                    scale_size = int(size['height'] / crop_pct)
                else:
                    scale_size = (int(size['height'] / crop_pct), int(size['width'] / crop_pct))
            else:
                raise ValueError('Invalid size for resize: {}'.format(size))
            output_size = get_resize_output_image_size(image, size=scale_size, default_to_square=False, input_data_format=input_data_format)
        elif 'shortest_edge' in size:
            output_size = get_resize_output_image_size(image, size=size['shortest_edge'], default_to_square=False, input_data_format=input_data_format)
        elif 'height' in size and 'width' in size:
            output_size = (size['height'], size['width'])
        else:
            raise ValueError('Invalid size for resize: {}'.format(size))
        return resize(image, size=output_size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def preprocess(self, images: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, crop_pct: int=None, resample: PILImageResampling=None, do_center_crop: bool=None, crop_size: Dict[str, int]=None, do_rescale: bool=None, rescale_factor: float=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> PIL.Image.Image:
        if False:
            for i in range(10):
                print('nop')
        '\n        Preprocess an image or batch of images.\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If\n                passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Size of the image after applying resize.\n            crop_pct (`float`, *optional*, defaults to `self.crop_pct`):\n                Percentage of the image to crop. Only has an effect if `do_resize` is set to `True`.\n            resample (`int`, *optional*, defaults to `self.resample`):\n                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only\n                has an effect if `do_resize` is set to `True`.\n            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):\n                Whether to center crop the image.\n            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):\n                Size of the image after applying center crop.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image values between [0 - 1].\n            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):\n                Rescale factor to rescale the image by if `do_rescale` is set to `True`.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Image mean.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Image standard deviation.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_resize = do_resize if do_resize is not None else self.do_resize
        crop_pct = crop_pct if crop_pct is not None else self.crop_pct
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
        if do_resize and size is None or resample is None:
            raise ValueError('Size and resample must be specified if do_resize is True.')
        if do_center_crop and crop_pct is None:
            raise ValueError('Crop_pct must be specified if do_center_crop is True.')
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
            images = [self.resize(image=image, size=size, crop_pct=crop_pct, resample=resample, input_data_format=input_data_format) for image in images]
        if do_center_crop:
            images = [self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)