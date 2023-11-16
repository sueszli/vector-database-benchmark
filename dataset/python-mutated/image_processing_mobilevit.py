"""Image processor class for MobileViT."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import flip_channel_order, get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging
if is_vision_available():
    import PIL
if is_torch_available():
    import torch
logger = logging.get_logger(__name__)

class MobileViTImageProcessor(BaseImageProcessor):
    """
    Constructs a MobileViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Controls the size of the output image after resizing. Can be overridden by the `size` parameter in the
            `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter
            in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped. Can be overridden by the `do_center_crop` parameter in
            the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Desired output size `(size["height"], size["width"])` when applying center-cropping. Can be overridden by
            the `crop_size` parameter in the `preprocess` method.
        do_flip_channel_order (`bool`, *optional*, defaults to `True`):
            Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order`
            parameter in the `preprocess` method.
        use_square_size (`bool`, *optional*, defaults to `False`):
            The value to be passed to `get_size_dict` as `default_to_square` when computing the image size. If the
            `size` argument in `get_size_dict` is an `int`, it determines whether to default to a square image or not.
            Note that this attribute is not used in computing `crop_size` via calling `get_size_dict`.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_center_crop: bool=True, crop_size: Dict[str, int]=None, do_flip_channel_order: bool=True, use_square_size: bool=False, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        size = size if size is not None else {'shortest_edge': 224}
        size = get_size_dict(size, default_to_square=use_square_size)
        crop_size = crop_size if crop_size is not None else {'height': 256, 'width': 256}
        crop_size = get_size_dict(crop_size, param_name='crop_size')
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_flip_channel_order = do_flip_channel_order
        self.use_square_size = use_square_size

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BILINEAR, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge\n        resized to keep the input aspect ratio.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Size of the output image.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):\n                Resampling filter to use when resiizing the image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        size = get_size_dict(size, default_to_square=self.use_square_size)
        if 'shortest_edge' not in size:
            raise ValueError(f'The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}')
        output_size = get_resize_output_image_size(image, size=size['shortest_edge'], default_to_square=self.use_square_size, input_data_format=input_data_format)
        return resize(image, size=output_size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def flip_channel_order(self, image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Flip the color channels from RGB to BGR or vice versa.\n\n        Args:\n            image (`np.ndarray`):\n                The image, represented as a numpy array.\n            data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        return flip_channel_order(image, data_format=data_format, input_data_format=input_data_format)

    def preprocess(self, images: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, resample: PILImageResampling=None, do_rescale: bool=None, rescale_factor: float=None, do_center_crop: bool=None, crop_size: Dict[str, int]=None, do_flip_channel_order: bool=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> PIL.Image.Image:
        if False:
            return 10
        '\n        Preprocess an image or batch of images.\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If\n                passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Size of the image after resizing.\n            resample (`int`, *optional*, defaults to `self.resample`):\n                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only\n                has an effect if `do_resize` is set to `True`.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image by rescale factor.\n            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):\n                Rescale factor to rescale the image by if `do_rescale` is set to `True`.\n            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):\n                Whether to center crop the image.\n            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):\n                Size of the center crop if `do_center_crop` is set to `True`.\n            do_flip_channel_order (`bool`, *optional*, defaults to `self.do_flip_channel_order`):\n                Whether to flip the channel order of the image.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_flip_channel_order = do_flip_channel_order if do_flip_channel_order is not None else self.do_flip_channel_order
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=self.use_square_size)
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name='crop_size')
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_resize and size is None:
            raise ValueError('Size must be specified if do_resize is True.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_center_crop and crop_size is None:
            raise ValueError('Crop size must be specified if do_center_crop is True.')
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format) for image in images]
        if do_center_crop:
            images = [self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
        if do_flip_channel_order:
            images = [self.flip_channel_order(image=image, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple]=None):
        if False:
            i = 10
            return i + 15
        '\n        Converts the output of [`MobileViTForSemanticSegmentation`] into semantic segmentation maps. Only supports\n        PyTorch.\n\n        Args:\n            outputs ([`MobileViTForSemanticSegmentation`]):\n                Raw outputs of the model.\n            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):\n                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,\n                predictions will not be resized.\n\n        Returns:\n            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic\n            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is\n            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.\n        '
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