"""Image processor class for GLPN."""
from typing import List, Optional, Union
import numpy as np
import PIL.Image
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import ChannelDimension, PILImageResampling, get_image_size, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, logging
logger = logging.get_logger(__name__)

class GLPNImageProcessor(BaseImageProcessor):
    """
    Constructs a GLPN image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions, rounding them down to the closest multiple of
            `size_divisor`. Can be overridden by `do_resize` in `preprocess`.
        size_divisor (`int`, *optional*, defaults to 32):
            When `do_resize` is `True`, images are resized so their height and width are rounded down to the closest
            multiple of `size_divisor`. Can be overridden by `size_divisor` in `preprocess`.
        resample (`PIL.Image` resampling filter, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Can be
            overridden by `do_rescale` in `preprocess`.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_resize: bool=True, size_divisor: int=32, resample=PILImageResampling.BILINEAR, do_rescale: bool=True, **kwargs) -> None:
        if False:
            return 10
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.size_divisor = size_divisor
        self.resample = resample
        super().__init__(**kwargs)

    def resize(self, image: np.ndarray, size_divisor: int, resample: PILImageResampling=PILImageResampling.BILINEAR, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Resize the image, rounding the (height, width) dimensions down to the closest multiple of size_divisor.\n\n        If the image is of dimension (3, 260, 170) and size_divisor is 32, the image will be resized to (3, 256, 160).\n\n        Args:\n            image (`np.ndarray`):\n                The image to resize.\n            size_divisor (`int`):\n                The image is resized so its height and width are rounded down to the closest multiple of\n                `size_divisor`.\n            resample:\n                `PIL.Image` resampling filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.\n            data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the output image. If `None`, the channel dimension format of the input\n                image is used. Can be one of:\n                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not set, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n\n        Returns:\n            `np.ndarray`: The resized image.\n        '
        (height, width) = get_image_size(image, channel_dim=input_data_format)
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        image = resize(image, (new_h, new_w), resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)
        return image

    def preprocess(self, images: Union['PIL.Image.Image', TensorType, List['PIL.Image.Image'], List[TensorType]], do_resize: Optional[bool]=None, size_divisor: Optional[int]=None, resample=None, do_rescale: Optional[bool]=None, return_tensors: Optional[Union[TensorType, str]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> BatchFeature:
        if False:
            i = 10
            return i + 15
        '\n        Preprocess the given images.\n\n        Args:\n            images (`PIL.Image.Image` or `TensorType` or `List[np.ndarray]` or `List[TensorType]`):\n                Images to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If\n                passing in images with pixel values between 0 and 1, set `do_normalize=False`.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the input such that the (height, width) dimensions are a multiple of `size_divisor`.\n            size_divisor (`int`, *optional*, defaults to `self.size_divisor`):\n                When `do_resize` is `True`, images are resized so their height and width are rounded down to the\n                closest multiple of `size_divisor`.\n            resample (`PIL.Image` resampling filter, *optional*, defaults to `self.resample`):\n                `PIL.Image` resampling filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has\n                an effect if `do_resize` is set to `True`.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - `None`: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample
        if do_resize and size_divisor is None:
            raise ValueError('size_divisor is required for resizing')
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image(s)')
        images = [to_numpy_array(img) for img in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if do_resize:
            images = [self.resize(image, size_divisor=size_divisor, resample=resample, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image, scale=1 / 255, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)