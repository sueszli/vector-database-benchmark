"""Image processor class for Nougat."""
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import get_resize_output_image_size, pad, resize, to_channel_dimension_format, to_pil_image
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ChannelDimension, ImageInput, PILImageResampling, get_image_size, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, logging
from ...utils.import_utils import is_cv2_available, is_vision_available
logger = logging.get_logger(__name__)
if is_cv2_available():
    pass
if is_vision_available():
    import PIL

class NougatImageProcessor(BaseImageProcessor):
    """
    Constructs a Nougat image processor.

    Args:
        do_crop_margin (`bool`, *optional*, defaults to `True`):
            Whether to crop the image margins.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 896, "width": 672}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the images to the largest image size in the batch.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Image standard deviation.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_crop_margin: bool=True, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, do_thumbnail: bool=True, do_align_long_axis: bool=False, do_pad: bool=True, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        size = size if size is not None else {'height': 896, 'width': 672}
        size = get_size_dict(size)
        self.do_crop_margin = do_crop_margin
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def python_find_non_zero(self, image: np.array):
        if False:
            i = 10
            return i + 15
        'This is a reimplementation of a findNonZero function equivalent to cv2.'
        non_zero_indices = np.column_stack(np.nonzero(image))
        idxvec = non_zero_indices[:, [1, 0]]
        idxvec = idxvec.reshape(-1, 1, 2)
        return idxvec

    def python_bounding_rect(self, coordinates):
        if False:
            return 10
        'This is a reimplementation of a BoundingRect function equivalent to cv2.'
        min_values = np.min(coordinates, axis=(0, 1)).astype(int)
        max_values = np.max(coordinates, axis=(0, 1)).astype(int)
        (x_min, y_min) = (min_values[0], min_values[1])
        width = max_values[0] - x_min + 1
        height = max_values[1] - y_min + 1
        return (x_min, y_min, width, height)

    def crop_margin(self, image: np.array, gray_threshold: int=200, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.array:
        if False:
            return 10
        '\n        Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the\n        threshold).\n\n        Args:\n            image (`np.array`):\n                The image to be cropped.\n            gray_threshold (`int`, *optional*, defaults to `200`)\n                Value below which pixels are considered to be gray.\n            data_format (`ChannelDimension`, *optional*):\n                The channel dimension format of the output image. If unset, will use the inferred format from the\n                input.\n            input_data_format (`ChannelDimension`, *optional*):\n                The channel dimension format of the input image. If unset, will use the inferred format from the input.\n        '
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        image = to_pil_image(image, input_data_format=input_data_format)
        data = np.array(image.convert('L')).astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            image = np.array(image)
            image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
            return image
        data = (data - min_val) / (max_val - min_val) * 255
        gray = data < gray_threshold
        coords = self.python_find_non_zero(gray)
        (x_min, y_min, width, height) = self.python_bounding_rect(coords)
        image = image.crop((x_min, y_min, x_min + width, y_min + height))
        image = np.array(image).astype(np.uint8)
        image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
        image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        return image

    def align_long_axis(self, image: np.ndarray, size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Align the long axis of the image to the longest axis of the specified size.\n\n        Args:\n            image (`np.ndarray`):\n                The image to be aligned.\n            size (`Dict[str, int]`):\n                The size `{"height": h, "width": w}` to align the long axis to.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The data format of the output image. If unset, the same format as the input image is used.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n\n        Returns:\n            `np.ndarray`: The aligned image.\n        '
        (input_height, input_width) = get_image_size(image, channel_dim=input_data_format)
        (output_height, output_width) = (size['height'], size['width'])
        if output_width < output_height and input_width > input_height or (output_width > output_height and input_width < input_height):
            image = np.rot90(image, 3)
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def pad_image(self, image: np.ndarray, size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Pad the image to the specified size at the top, bottom, left and right.\n\n        Args:\n            image (`np.ndarray`):\n                The image to be padded.\n            size (`Dict[str, int]`):\n                The size `{"height": h, "width": w}` to pad the image to.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The data format of the output image. If unset, the same format as the input image is used.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        (output_height, output_width) = (size['height'], size['width'])
        (input_height, input_width) = get_image_size(image, channel_dim=input_data_format)
        delta_width = output_width - input_width
        delta_height = output_height - input_height
        pad_top = delta_height // 2
        pad_left = delta_width // 2
        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format)

    def thumbnail(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any\n        corresponding dimension of the specified size.\n\n        Args:\n            image (`np.ndarray`):\n                The image to be resized.\n            size (`Dict[str, int]`):\n                The size `{"height": h, "width": w}` to resize the image to.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):\n                The resampling filter to use.\n            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):\n                The data format of the output image. If unset, the same format as the input image is used.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        (input_height, input_width) = get_image_size(image, channel_dim=input_data_format)
        (output_height, output_width) = (size['height'], size['width'])
        height = min(input_height, output_height)
        width = min(input_width, output_width)
        if height == input_height and width == input_width:
            return image
        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)
        return resize(image, size=(height, width), resample=resample, reducing_gap=2.0, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Resizes `image` to `(height, width)` specified by `size` using the PIL library.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Size of the output image.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):\n                Resampling filter to use when resiizing the image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        size = get_size_dict(size)
        shortest_edge = min(size['height'], size['width'])
        output_size = get_resize_output_image_size(image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format)
        resized_image = resize(image, size=output_size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)
        return resized_image

    def preprocess(self, images: ImageInput, do_crop_margin: bool=None, do_resize: bool=None, size: Dict[str, int]=None, resample: PILImageResampling=None, do_thumbnail: bool=None, do_align_long_axis: bool=None, do_pad: bool=None, do_rescale: bool=None, rescale_factor: Union[int, float]=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: Optional[ChannelDimension]=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> PIL.Image.Image:
        if False:
            i = 10
            return i + 15
        '\n        Preprocess an image or batch of images.\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.\n            do_crop_margin (`bool`, *optional*, defaults to `self.do_crop_margin`):\n                Whether to crop the image margins.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Size of the image after resizing. Shortest edge of the image is resized to min(size["height"],\n                size["width"]) with the longest edge resized to keep the input aspect ratio.\n            resample (`int`, *optional*, defaults to `self.resample`):\n                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only\n                has an effect if `do_resize` is set to `True`.\n            do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`):\n                Whether to resize the image using thumbnail method.\n            do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`):\n                Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.\n            do_pad (`bool`, *optional*, defaults to `self.do_pad`):\n                Whether to pad the images to the largest image size in the batch.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image by the specified scale `rescale_factor`.\n            rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):\n                Scale factor to use if rescaling the image.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Image mean to use for normalization.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Image standard deviation to use for normalization.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                - Unset: Return a list of `np.ndarray`.\n                - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: defaults to the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_crop_margin = do_crop_margin if do_crop_margin is not None else self.do_crop_margin
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_thumbnail = do_thumbnail if do_thumbnail is not None else self.do_thumbnail
        do_align_long_axis = do_align_long_axis if do_align_long_axis is not None else self.do_align_long_axis
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_resize and size is None:
            raise ValueError('Size must be specified if do_resize is True.')
        if do_pad and size is None:
            raise ValueError('Size must be specified if do_pad is True.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if do_crop_margin:
            images = [self.crop_margin(image, input_data_format=input_data_format) for image in images]
        if do_align_long_axis:
            images = [self.align_long_axis(image, size=size, input_data_format=input_data_format) for image in images]
        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format) for image in images]
        if do_thumbnail:
            images = [self.thumbnail(image=image, size=size, input_data_format=input_data_format) for image in images]
        if do_pad:
            images = [self.pad_image(image=image, size=size, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)