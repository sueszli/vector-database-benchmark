"""Image processor class for TVLT."""
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, ImageInput, PILImageResampling, infer_channel_dimension_format, is_scaled_image, is_valid_image, to_numpy_array, valid_images
from ...utils import TensorType, logging
logger = logging.get_logger(__name__)

def make_batched(videos) -> List[List[ImageInput]]:
    if False:
        while True:
            i = 10
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)):
        return videos
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        videos_dim = np.array(videos[0]).ndim
        if videos_dim == 3:
            return [videos]
        elif videos_dim == 4:
            return videos
    elif is_valid_image(videos):
        videos_dim = np.array(videos).ndim
        if videos_dim == 3:
            return [[videos]]
        elif videos_dim == 4:
            return [videos]
        elif videos_dim == 5:
            return videos
    raise ValueError(f'Could not make batched video from {videos}')

class TvltImageProcessor(BaseImageProcessor):
    """
    Constructs a TVLT image processor.

    This processor can be used to prepare either videos or images for the model by converting images to 1-frame videos.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the output image after resizing. The shortest edge of the image will be resized to
            `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overriden by
            `size` in the `preprocess` method.
        patch_size (`List[int]` *optional*, defaults to [16,16]):
            The patch size of image patch embedding.
        num_frames (`int` *optional*, defaults to 8):
            The maximum number of video frames.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to 1/255):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    model_input_names = ['pixel_values', 'pixel_mask', 'pixel_values_mixed', 'pixel_mask_mixed']

    def __init__(self, do_resize: bool=True, size: Dict[str, int]=None, patch_size: List[int]=[16, 16], num_frames: int=8, resample: PILImageResampling=PILImageResampling.BILINEAR, do_center_crop: bool=True, crop_size: Dict[str, int]=None, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=IMAGENET_STANDARD_MEAN, image_std: Optional[Union[float, List[float]]]=IMAGENET_STANDARD_STD, init_mask_generator=False, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        size = size if size is not None else {'shortest_edge': 224}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {'height': 224, 'width': 224}
        crop_size = get_size_dict(crop_size, param_name='crop_size')
        self.do_resize = do_resize
        self.size = size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BILINEAR, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Resize an image.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will\n                have the size `(h, w)`. If `size` is of the form `{"shortest_edge": s}`, the output image will have its\n                shortest edge of length `s` while keeping the aspect ratio of the original image.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):\n                Resampling filter to use when resiizing the image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        size = get_size_dict(size, default_to_square=False)
        if 'shortest_edge' in size:
            output_size = get_resize_output_image_size(image, size['shortest_edge'], default_to_square=False, input_data_format=input_data_format)
        elif 'height' in size and 'width' in size:
            output_size = (size['height'], size['width'])
        else:
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        return resize(image, size=output_size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def _preprocess_image(self, image: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, resample: PILImageResampling=None, do_center_crop: bool=None, crop_size: Dict[str, int]=None, do_rescale: bool=None, rescale_factor: float=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, data_format: Optional[ChannelDimension]=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Preprocesses a single image.'
        if do_resize and size is None or resample is None:
            raise ValueError('Size and resample must be specified if do_resize is True.')
        if do_center_crop and crop_size is None:
            raise ValueError('Crop size must be specified if do_center_crop is True.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        image = to_numpy_array(image)
        if is_scaled_image(image) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def preprocess(self, videos: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, patch_size: List[int]=None, num_frames: int=None, resample: PILImageResampling=None, do_center_crop: bool=None, crop_size: Dict[str, int]=None, do_rescale: bool=None, rescale_factor: float=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, is_mixed: bool=False, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> BatchFeature:
        if False:
            print('Hello World!')
        '\n        Preprocess an videos or image or batch of videos or images.\n\n        Args:\n            videos (`ImageInput`):\n                Images or videos to preprocess. Expects a single or batch of frames with pixel values ranging from 0 to\n                255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Size of the image after applying resize.\n            patch_size (`List[int]` *optional*, defaults to self.patch_size):\n                The patch size of image patch embedding.\n            num_frames (`int` *optional*, defaults to self.num_frames):\n                The maximum number of video frames.\n            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):\n                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only\n                has an effect if `do_resize` is set to `True`.\n            do_center_crop (`bool`, *optional*, defaults to `self.do_centre_crop`):\n                Whether to centre crop the image.\n            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):\n                Size of the image after applying the centre crop.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image values between [0 - 1].\n            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):\n                Rescale factor to rescale the image by if `do_rescale` is set to `True`.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Image mean.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Image standard deviation.\n            is_mixed (`bool`, *optional*):\n                If the input video has negative samples.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                    - Unset: Use the inferred channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n\n        Returns:\n            [`BatchFeature`]: A [`BatchFeature`] with the following fields:\n\n            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,\n              width).\n\n            - **pixel_mask** -- Pixel masks to be fed to a model, of shape (batch_size, num_pixel_patches).\n\n            - **pixel_values_mixed** -- Pixel values with both postive or negative to be fed to a model, of shape\n              (batch_size, num_channels, height, width).\n\n            - **pixel_mask_mixed** -- Pixel masks with both postive or negative to be fed to a model, of shape\n              (batch_size, num_pixel_patches).\n        '
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
        patch_size = patch_size if patch_size is not None else self.patch_size
        num_frames = num_frames if patch_size is not None else self.num_frames
        if not valid_images(videos):
            raise ValueError('Invalid image or video type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        videos = make_batched(videos)
        for video in videos:
            if len(video) > self.num_frames:
                raise ValueError(f'number of frames must not be greater than the maximum frames of the model {self.num_frames}.')
        max_num_frames = max([len(video) for video in videos])
        num_patches_per_image = (size['shortest_edge'] // patch_size[0]) ** 2
        video_masks = np.array([len(video) * num_patches_per_image * [1] + (max_num_frames - len(video)) * num_patches_per_image * [0] for video in videos])
        videos = [[self._preprocess_image(image=img, do_resize=do_resize, size=size, resample=resample, do_center_crop=do_center_crop, crop_size=crop_size, do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, data_format=data_format, input_data_format=input_data_format) for img in video] for video in videos]
        if is_mixed:
            data = {'pixel_values_mixed': videos, 'pixel_mask_mixed': video_masks}
        else:
            data = {'pixel_values': videos, 'pixel_mask': video_masks}
        return BatchFeature(data=data, tensor_type=return_tensors)