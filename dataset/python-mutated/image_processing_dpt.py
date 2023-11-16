"""Image processor class for DPT."""
import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import pad, resize, to_channel_dimension_format
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, ImageInput, PILImageResampling, get_image_size, infer_channel_dimension_format, is_scaled_image, is_torch_available, is_torch_tensor, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_vision_available, logging
if is_torch_available():
    import torch
if is_vision_available():
    import PIL
logger = logging.get_logger(__name__)

def get_resize_output_image_size(input_image: np.ndarray, output_size: Union[int, Iterable[int]], keep_aspect_ratio: bool, multiple: int, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> Tuple[int, int]:
    if False:
        return 10

    def constraint_to_multiple_of(val, multiple, min_val=0, max_val=None):
        if False:
            while True:
                i = 10
        x = round(val / multiple) * multiple
        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple
        if x < min_val:
            x = math.ceil(val / multiple) * multiple
        return x
    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
    (input_height, input_width) = get_image_size(input_image, input_data_format)
    (output_height, output_width) = output_size
    scale_height = output_height / input_height
    scale_width = output_width / input_width
    if keep_aspect_ratio:
        if abs(1 - scale_width) < abs(1 - scale_height):
            scale_height = scale_width
        else:
            scale_width = scale_height
    new_height = constraint_to_multiple_of(scale_height * input_height, multiple=multiple)
    new_width = constraint_to_multiple_of(scale_width * input_width, multiple=multiple)
    return (new_height, new_width)

class DPTImageProcessor(BaseImageProcessor):
    """
    Constructs a DPT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions. Can be overidden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the image after resizing. Can be overidden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Defines the resampling filter to use if resizing the image. Can be overidden by `resample` in `preprocess`.
        keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
            If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
            be overidden by `keep_aspect_ratio` in `preprocess`.
        ensure_multiple_of (`int`, *optional*, defaults to 1):
            If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overidden
            by `ensure_multiple_of` in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overidden by `do_rescale` in
            `preprocess`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overidden by `rescale_factor` in `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `False`):
            Whether to apply center padding. This was introduced in the DINOv2 paper, which uses the model in
            combination with DPT.
        size_divisor (`int`, *optional*):
            If `do_pad` is `True`, pads the image dimensions to be divisible by this value. This was introduced in the
            DINOv2 paper, which uses the model in combination with DPT.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, keep_aspect_ratio: bool=False, ensure_multiple_of: int=1, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: bool=False, size_divisor: int=None, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        size = size if size is not None else {'height': 384, 'width': 384}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.size_divisor = size_divisor

    def resize(self, image: np.ndarray, size: Dict[str, int], keep_aspect_ratio: bool=False, ensure_multiple_of: int=1, resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Resize an image to target size `(size["height"], size["width"])`. If `keep_aspect_ratio` is `True`, the image\n        is resized to the largest possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is\n        set, the image is resized to a size that is a multiple of this value.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Target size of the output image.\n            keep_aspect_ratio (`bool`, *optional*, defaults to `False`):\n                If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.\n            ensure_multiple_of (`int`, *optional*, defaults to 1):\n                The image is resized to a size that is a multiple of this value.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):\n                Defines the resampling filter to use if resizing the image. Otherwise, the image is resized to size\n                specified in `size`.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):\n                Resampling filter to use when resiizing the image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        size = get_size_dict(size)
        if 'height' not in size or 'width' not in size:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size.keys()}")
        output_size = get_resize_output_image_size(image, output_size=(size['height'], size['width']), keep_aspect_ratio=keep_aspect_ratio, multiple=ensure_multiple_of, input_data_format=input_data_format)
        return resize(image, size=output_size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def pad_image(self, image: np.array, size_divisor: int, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None):
        if False:
            return 10
        '\n        Center pad an image to be a multiple of `multiple`.\n\n        Args:\n            image (`np.ndarray`):\n                Image to pad.\n            size_divisor (`int`):\n                The width and height of the image will be padded to a multiple of this number.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '

        def _get_pad(size, size_divisor):
            if False:
                return 10
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return (pad_size_left, pad_size_right)
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        (height, width) = get_image_size(image, input_data_format)
        (pad_size_left, pad_size_right) = _get_pad(height, size_divisor)
        (pad_size_top, pad_size_bottom) = _get_pad(width, size_divisor)
        return pad(image, ((pad_size_left, pad_size_right), (pad_size_top, pad_size_bottom)), data_format=data_format)

    def preprocess(self, images: ImageInput, do_resize: bool=None, size: int=None, keep_aspect_ratio: bool=None, ensure_multiple_of: int=None, resample: PILImageResampling=None, do_rescale: bool=None, rescale_factor: float=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: bool=None, size_divisor: int=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> PIL.Image.Image:
        if False:
            return 10
        '\n        Preprocess an image or batch of images.\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If\n                passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Size of the image after reszing. If `keep_aspect_ratio` is `True`, the image is resized to the largest\n                possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is set, the image is\n                resized to a size that is a multiple of this value.\n            keep_aspect_ratio (`bool`, *optional*, defaults to `self.keep_aspect_ratio`):\n                Whether to keep the aspect ratio of the image. If False, the image will be resized to (size, size). If\n                True, the image will be resized to keep the aspect ratio and the size will be the maximum possible.\n            ensure_multiple_of (`int`, *optional*, defaults to `self.ensure_multiple_of`):\n                Ensure that the image size is a multiple of this value.\n            resample (`int`, *optional*, defaults to `self.resample`):\n                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only\n                has an effect if `do_resize` is set to `True`.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image values between [0 - 1].\n            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):\n                Rescale factor to rescale the image by if `do_rescale` is set to `True`.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Image mean.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Image standard deviation.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size)
        keep_aspect_ratio = keep_aspect_ratio if keep_aspect_ratio is not None else self.keep_aspect_ratio
        ensure_multiple_of = ensure_multiple_of if ensure_multiple_of is not None else self.ensure_multiple_of
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_resize and size is None or resample is None:
            raise ValueError('Size and resample must be specified if do_resize is True.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        if do_pad and size_divisor is None:
            raise ValueError('Size divisibility must be specified if do_pad is True.')
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format) for image in images]
        if do_pad:
            images = [self.pad_image(image=image, size_divisor=size_divisor, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple]=None):
        if False:
            return 10
        '\n        Converts the output of [`DPTForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.\n\n        Args:\n            outputs ([`DPTForSemanticSegmentation`]):\n                Raw outputs of the model.\n            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):\n                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,\n                predictions will not be resized.\n\n        Returns:\n            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic\n            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is\n            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.\n        '
        logits = outputs.logits
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()
            semantic_segmentation = []
            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode='bilinear', align_corners=False)
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]
        return semantic_segmentation