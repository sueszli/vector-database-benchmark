"""Image processor class for SAM."""
import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ChannelDimension, ImageInput, PILImageResampling, get_image_size, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_tf_available, is_torch_available, is_torchvision_available, logging, requires_backends
if is_torch_available():
    import torch
    import torch.nn.functional as F
if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms
if is_tf_available():
    import tensorflow as tf
    from tensorflow.experimental import numpy as tnp
    from ...tf_utils import flatten, shape_list
logger = logging.get_logger(__name__)

class SamImageProcessor(BaseImageProcessor):
    """
    Constructs a SAM image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"longest_edge": 1024}`):
            Size of the output image after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `size` parameter in the
            `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to the specified `pad_size`. Can be overridden by the `do_pad` parameter in the
            `preprocess` method.
        pad_size (`dict`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
            Size of the output image after padding. Can be overridden by the `pad_size` parameter in the `preprocess`
            method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: bool=True, pad_size: int=None, do_convert_rgb: bool=True, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        size = size if size is not None else {'longest_edge': 1024}
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size
        pad_size = pad_size if pad_size is not None else {'height': 1024, 'width': 1024}
        pad_size = get_size_dict(pad_size, default_to_square=True)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.do_convert_rgb = do_convert_rgb

    def pad_image(self, image: np.ndarray, pad_size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.\n\n        Args:\n            image (`np.ndarray`):\n                Image to pad.\n            pad_size (`Dict[str, int]`):\n                Size of the output image after padding.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The data format of the image. Can be either "channels_first" or "channels_last". If `None`, the\n                `data_format` of the `image` will be used.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        (output_height, output_width) = (pad_size['height'], pad_size['width'])
        (input_height, input_width) = get_image_size(image, channel_dim=input_data_format)
        pad_width = output_width - input_width
        pad_height = output_height - input_height
        padded_image = pad(image, ((0, pad_height), (0, pad_width)), data_format=data_format, input_data_format=input_data_format, **kwargs)
        return padded_image

    def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int):
        if False:
            i = 10
            return i + 15
        '\n        Compute the output size given input size and target long side length.\n        '
        (oldh, oldw) = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        (newh, neww) = (oldh * scale, oldw * scale)
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Resize an image to `(size["height"], size["width"])`.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest\n                edge of the image will be resized to the specified size, while the other edge will be resized to\n                maintain the aspect ratio.\n            resample:\n                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.\n            data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n\n        Returns:\n            `np.ndarray`: The resized image.\n        '
        size = get_size_dict(size)
        if 'longest_edge' not in size:
            raise ValueError(f'The `size` dictionary must contain the key `longest_edge`. Got {size.keys()}')
        input_size = get_image_size(image, channel_dim=input_data_format)
        (output_height, output_width) = self._get_preprocess_shape(input_size, size['longest_edge'])
        return resize(image, size=(output_height, output_width), resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def preprocess(self, images: ImageInput, do_resize: Optional[bool]=None, size: Optional[Dict[str, int]]=None, resample: Optional['PILImageResampling']=None, do_rescale: Optional[bool]=None, rescale_factor: Optional[Union[int, float]]=None, do_normalize: Optional[bool]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, pad_size: Optional[Dict[str, int]]=None, do_convert_rgb: bool=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs):
        if False:
            return 10
        '\n        Preprocess an image or batch of images.\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If\n                passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Controls the size of the image after `resize`. The longest edge of the image is resized to\n                `size["longest_edge"]` whilst preserving the aspect ratio.\n            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):\n                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image pixel values by rescaling factor.\n            rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):\n                Rescale factor to apply to the image pixel values.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Image mean to normalize the image by if `do_normalize` is set to `True`.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.\n            do_pad (`bool`, *optional*, defaults to `self.do_pad`):\n                Whether to pad the image.\n            pad_size (`Dict[str, int]`, *optional*, defaults to `self.pad_size`):\n                Controls the size of the padding applied to the image. The image is padded to `pad_size["height"]` and\n                `pad_size["width"]` if `do_pad` is set to `True`.\n            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):\n                Whether to convert the image to RGB.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size = pad_size if pad_size is not None else self.pad_size
        pad_size = get_size_dict(pad_size, default_to_square=True)
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_resize and (size is None or resample is None):
            raise ValueError('Size and resample must be specified if do_resize is True.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        if do_pad and pad_size is None:
            raise ValueError('Pad size must be specified if do_pad is True.')
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        original_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format) for image in images]
        reshaped_input_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format) for image in images]
        if do_pad:
            images = [self.pad_image(image=image, pad_size=pad_size, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        encoded_outputs = BatchFeature(data={'pixel_values': images, 'original_sizes': original_sizes, 'reshaped_input_sizes': reshaped_input_sizes}, tensor_type=return_tensors)
        return encoded_outputs

    def post_process_masks(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None, return_tensors='pt'):
        if False:
            print('Hello World!')
        '\n        Remove padding and upscale masks to the original image size.\n\n        Args:\n            masks (`Union[List[torch.Tensor], List[np.ndarray], List[tf.Tensor]]`):\n                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.\n            original_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):\n                The original sizes of each image before it was resized to the model\'s expected input shape, in (height,\n                width) format.\n            reshaped_input_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):\n                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.\n            mask_threshold (`float`, *optional*, defaults to 0.0):\n                The threshold to use for binarizing the masks.\n            binarize (`bool`, *optional*, defaults to `True`):\n                Whether to binarize the masks.\n            pad_size (`int`, *optional*, defaults to `self.pad_size`):\n                The target size the images were padded to before being passed to the model. If None, the target size is\n                assumed to be the processor\'s `pad_size`.\n            return_tensors (`str`, *optional*, defaults to `"pt"`):\n                If `"pt"`, return PyTorch tensors. If `"tf"`, return TensorFlow tensors.\n        Returns:\n            (`Union[torch.Tensor, tf.Tensor]`): Batched masks in batch_size, num_channels, height, width) format, where\n            (height, width) is given by original_size.\n        '
        if return_tensors == 'pt':
            return self._post_process_masks_pt(masks=masks, original_sizes=original_sizes, reshaped_input_sizes=reshaped_input_sizes, mask_threshold=mask_threshold, binarize=binarize, pad_size=pad_size)
        elif return_tensors == 'tf':
            return self._post_process_masks_tf(masks=masks, original_sizes=original_sizes, reshaped_input_sizes=reshaped_input_sizes, mask_threshold=mask_threshold, binarize=binarize, pad_size=pad_size)
        else:
            raise ValueError("return_tensors must be either 'pt' or 'tf'")

    def _post_process_masks_pt(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Remove padding and upscale masks to the original image size.\n\n        Args:\n            masks (`Union[List[torch.Tensor], List[np.ndarray]]`):\n                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.\n            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):\n                The original sizes of each image before it was resized to the model's expected input shape, in (height,\n                width) format.\n            reshaped_input_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):\n                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.\n            mask_threshold (`float`, *optional*, defaults to 0.0):\n                The threshold to use for binarizing the masks.\n            binarize (`bool`, *optional*, defaults to `True`):\n                Whether to binarize the masks.\n            pad_size (`int`, *optional*, defaults to `self.pad_size`):\n                The target size the images were padded to before being passed to the model. If None, the target size is\n                assumed to be the processor's `pad_size`.\n        Returns:\n            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)\n            is given by original_size.\n        "
        requires_backends(self, ['torch'])
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size['height'], pad_size['width'])
        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        if isinstance(reshaped_input_sizes, (torch.Tensor, np.ndarray)):
            reshaped_input_sizes = reshaped_input_sizes.tolist()
        output_masks = []
        for (i, original_size) in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            elif not isinstance(masks[i], torch.Tensor):
                raise ValueError('Input masks should be a list of `torch.tensors` or a list of `np.ndarray`')
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode='bilinear', align_corners=False)
            interpolated_mask = interpolated_mask[..., :reshaped_input_sizes[i][0], :reshaped_input_sizes[i][1]]
            interpolated_mask = F.interpolate(interpolated_mask, original_size, mode='bilinear', align_corners=False)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)
        return output_masks

    def _post_process_masks_tf(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None):
        if False:
            i = 10
            return i + 15
        "\n        Remove padding and upscale masks to the original image size.\n\n        Args:\n            masks (`tf.Tensor`):\n                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.\n            original_sizes (`tf.Tensor`):\n                The original size of the images before resizing for input to the model, in (height, width) format.\n            reshaped_input_sizes (`tf.Tensor`):\n                The size of the image input to the model, in (height, width) format. Used to remove padding.\n            mask_threshold (`float`, *optional*, defaults to 0.0):\n                The threshold to use for binarizing the masks.\n            binarize (`bool`, *optional*, defaults to `True`):\n                Whether to binarize the masks.\n            pad_size (`int`, *optional*, defaults to `self.pad_size`):\n                The target size the images were padded to before being passed to the model. If None, the target size is\n                assumed to be the processor's `pad_size`.\n        Returns:\n            (`tf.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width) is\n            given by original_size.\n        "
        requires_backends(self, ['tf'])
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size['height'], pad_size['width'])
        output_masks = []
        for (i, original_size) in enumerate(original_sizes):
            mask = tf.transpose(masks[i], perm=[0, 2, 3, 1])
            interpolated_mask = tf.image.resize(mask, target_image_size, method='bilinear')
            interpolated_mask = interpolated_mask[:, :reshaped_input_sizes[i][0], :reshaped_input_sizes[i][1], :]
            interpolated_mask = tf.image.resize(interpolated_mask, original_size, method='bilinear')
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(tf.transpose(interpolated_mask, perm=[0, 3, 1, 2]))
        return output_masks

    def post_process_for_mask_generation(self, all_masks, all_scores, all_boxes, crops_nms_thresh, return_tensors='pt'):
        if False:
            i = 10
            return i + 15
        '\n        Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.\n\n        Args:\n            all_masks (`Union[List[torch.Tensor], List[tf.Tensor]]`):\n                List of all predicted segmentation masks\n            all_scores (`Union[List[torch.Tensor], List[tf.Tensor]]`):\n                List of all predicted iou scores\n            all_boxes (`Union[List[torch.Tensor], List[tf.Tensor]]`):\n                List of all bounding boxes of the predicted masks\n            crops_nms_thresh (`float`):\n                Threshold for NMS (Non Maximum Suppression) algorithm.\n            return_tensors (`str`, *optional*, defaults to `pt`):\n                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.\n        '
        if return_tensors == 'pt':
            return _postprocess_for_mg(all_masks, all_scores, all_boxes, crops_nms_thresh)
        elif return_tensors == 'tf':
            return _postprocess_for_mg_tf(all_masks, all_scores, all_boxes, crops_nms_thresh)

    def generate_crop_boxes(self, image, target_size, crop_n_layers: int=0, overlap_ratio: float=512 / 1500, points_per_crop: Optional[int]=32, crop_n_points_downscale_factor: Optional[List[int]]=1, device: Optional['torch.device']=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, return_tensors: str='pt'):
        if False:
            return 10
        '\n        Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.\n\n        Args:\n            image (`np.array`):\n                Input original image\n            target_size (`int`):\n                Target size of the resized image\n            crop_n_layers (`int`, *optional*, defaults to 0):\n                If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where\n                each layer has 2**i_layer number of image crops.\n            overlap_ratio (`float`, *optional*, defaults to 512/1500):\n                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of\n                the image length. Later layers with more crops scale down this overlap.\n            points_per_crop (`int`, *optional*, defaults to 32):\n                Number of points to sample from each crop.\n            crop_n_points_downscale_factor (`List[int]`, *optional*, defaults to 1):\n                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.\n            device (`torch.device`, *optional*, defaults to None):\n                Device to use for the computation. If None, cpu will be used.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n            return_tensors (`str`, *optional*, defaults to `pt`):\n                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.\n        '
        (crop_boxes, points_per_crop, cropped_images, input_labels) = _generate_crop_boxes(image, target_size, crop_n_layers, overlap_ratio, points_per_crop, crop_n_points_downscale_factor, input_data_format)
        if return_tensors == 'pt':
            if device is None:
                device = torch.device('cpu')
            crop_boxes = torch.tensor(crop_boxes, device=device)
            points_per_crop = torch.tensor(points_per_crop, device=device)
            input_labels = torch.tensor(input_labels, device=device)
        elif return_tensors == 'tf':
            if device is not None:
                raise ValueError('device is not a supported argument when return_tensors is tf!')
            crop_boxes = tf.convert_to_tensor(crop_boxes)
            points_per_crop = tf.convert_to_tensor(points_per_crop)
            input_labels = tf.convert_to_tensor(input_labels)
        else:
            raise ValueError("return_tensors must be either 'pt' or 'tf'.")
        return (crop_boxes, points_per_crop, cropped_images, input_labels)

    def filter_masks(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1, return_tensors='pt'):
        if False:
            i = 10
            return i + 15
        '\n        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being\n        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability\n        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to\n        bounding boxes and pad the predicted masks if necessary.\n\n        Args:\n            masks (`Union[torch.Tensor, tf.Tensor]`):\n                Input masks.\n            iou_scores (`Union[torch.Tensor, tf.Tensor]`):\n                List of IoU scores.\n            original_size (`Tuple[int,int]`):\n                Size of the orginal image.\n            cropped_box_image (`np.array`):\n                The cropped image.\n            pred_iou_thresh (`float`, *optional*, defaults to 0.88):\n                The threshold for the iou scores.\n            stability_score_thresh (`float`, *optional*, defaults to 0.95):\n                The threshold for the stability score.\n            mask_threshold (`float`, *optional*, defaults to 0):\n                The threshold for the predicted masks.\n            stability_score_offset (`float`, *optional*, defaults to 1):\n                The offset for the stability score used in the `_compute_stability_score` method.\n            return_tensors (`str`, *optional*, defaults to `pt`):\n                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.\n        '
        if return_tensors == 'pt':
            return self._filter_masks_pt(masks=masks, iou_scores=iou_scores, original_size=original_size, cropped_box_image=cropped_box_image, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, mask_threshold=mask_threshold, stability_score_offset=stability_score_offset)
        elif return_tensors == 'tf':
            return self._filter_masks_tf(masks=masks, iou_scores=iou_scores, original_size=original_size, cropped_box_image=cropped_box_image, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, mask_threshold=mask_threshold, stability_score_offset=stability_score_offset)

    def _filter_masks_pt(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being\n        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability\n        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to\n        bounding boxes and pad the predicted masks if necessary.\n\n        Args:\n            masks (`torch.Tensor`):\n                Input masks.\n            iou_scores (`torch.Tensor`):\n                List of IoU scores.\n            original_size (`Tuple[int,int]`):\n                Size of the orginal image.\n            cropped_box_image (`np.array`):\n                The cropped image.\n            pred_iou_thresh (`float`, *optional*, defaults to 0.88):\n                The threshold for the iou scores.\n            stability_score_thresh (`float`, *optional*, defaults to 0.95):\n                The threshold for the stability score.\n            mask_threshold (`float`, *optional*, defaults to 0):\n                The threshold for the predicted masks.\n            stability_score_offset (`float`, *optional*, defaults to 1):\n                The offset for the stability score used in the `_compute_stability_score` method.\n\n        '
        requires_backends(self, ['torch'])
        (original_height, original_width) = original_size
        iou_scores = iou_scores.flatten(0, 1)
        masks = masks.flatten(0, 1)
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError('masks and iou_scores must have the same batch size.')
        if masks.device != iou_scores.device:
            iou_scores = iou_scores.to(masks.device)
        batch_size = masks.shape[0]
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_pt(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box(masks)
        keep_mask = ~_is_box_near_crop_edge(converted_boxes, cropped_box_image, [0, 0, original_width, original_height])
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]
        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        masks = _mask_to_rle_pytorch(masks)
        return (masks, scores, converted_boxes)

    def _filter_masks_tf(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1):
        if False:
            while True:
                i = 10
        '\n        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being\n        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability\n        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to\n        bounding boxes and pad the predicted masks if necessary.\n\n        Args:\n            masks (`tf.Tensor`):\n                Input masks.\n            iou_scores (`tf.Tensor`):\n                List of IoU scores.\n            original_size (`Tuple[int,int]`):\n                Size of the orginal image.\n            cropped_box_image (`np.array`):\n                The cropped image.\n            pred_iou_thresh (`float`, *optional*, defaults to 0.88):\n                The threshold for the iou scores.\n            stability_score_thresh (`float`, *optional*, defaults to 0.95):\n                The threshold for the stability score.\n            mask_threshold (`float`, *optional*, defaults to 0):\n                The threshold for the predicted masks.\n            stability_score_offset (`float`, *optional*, defaults to 1):\n                The offset for the stability score used in the `_compute_stability_score` method.\n\n        '
        requires_backends(self, ['tf'])
        (original_height, original_width) = original_size
        iou_scores = tf.reshape(iou_scores, [iou_scores.shape[0] * iou_scores.shape[1], iou_scores.shape[2:]])
        masks = tf.reshape(masks, [masks.shape[0] * masks.shape[1], masks.shape[2:]])
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError('masks and iou_scores must have the same batch size.')
        batch_size = masks.shape[0]
        keep_mask = tf.ones(batch_size, dtype=tf.bool)
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_tf(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box_tf(masks)
        keep_mask = ~_is_box_near_crop_edge_tf(converted_boxes, cropped_box_image, [0, 0, original_width, original_height])
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]
        masks = _pad_masks_tf(masks, cropped_box_image, original_height, original_width)
        masks = _mask_to_rle_tf(masks)
        return (masks, scores, converted_boxes)

def _compute_stability_score_pt(masks: 'torch.Tensor', mask_threshold: float, stability_score_offset: int):
    if False:
        i = 10
        return i + 15
    intersections = (masks > mask_threshold + stability_score_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > mask_threshold - stability_score_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    stability_scores = intersections / unions
    return stability_scores

def _compute_stability_score_tf(masks: 'tf.Tensor', mask_threshold: float, stability_score_offset: int):
    if False:
        return 10
    intersections = tf.count_nonzero(masks > mask_threshold + stability_score_offset, axis=[-1, -2], dtype=tf.float32)
    unions = tf.count_nonzero(masks > mask_threshold - stability_score_offset, axis=[-1, -2], dtype=tf.float32)
    stability_scores = intersections / unions
    return stability_scores

def _build_point_grid(n_per_side: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Generates a 2D grid of points evenly spaced in [0,1]x[0,1].'
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def _normalize_coordinates(target_size: int, coords: np.ndarray, original_size: Tuple[int, int], is_bounding_box=False) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Expects a numpy array of length 2 in the final dimension. Requires the original image size in (height, width)\n    format.\n    '
    (old_height, old_width) = original_size
    scale = target_size * 1.0 / max(old_height, old_width)
    (new_height, new_width) = (old_height * scale, old_width * scale)
    new_width = int(new_width + 0.5)
    new_height = int(new_height + 0.5)
    coords = deepcopy(coords).astype(float)
    if is_bounding_box:
        coords = coords.reshape(-1, 2, 2)
    coords[..., 0] = coords[..., 0] * (new_width / old_width)
    coords[..., 1] = coords[..., 1] * (new_height / old_height)
    if is_bounding_box:
        coords = coords.reshape(-1, 4)
    return coords

def _generate_crop_boxes(image, target_size: int, crop_n_layers: int=0, overlap_ratio: float=512 / 1500, points_per_crop: Optional[int]=32, crop_n_points_downscale_factor: Optional[List[int]]=1, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> Tuple[List[List[int]], List[int]]:
    if False:
        print('Hello World!')
    '\n    Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.\n\n    Args:\n        image (Union[`numpy.ndarray`, `PIL.Image`, `torch.Tensor`]):\n            Image to generate crops for.\n        target_size (`int`):\n            Size of the smallest crop.\n        crop_n_layers (`int`, *optional*):\n            If `crops_n_layers>0`, mask prediction will be run again on crops of the image. Sets the number of layers\n            to run, where each layer has 2**i_layer number of image crops.\n        overlap_ratio (`int`, *optional*):\n            Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the\n            image length. Later layers with more crops scale down this overlap.\n        points_per_crop (`int`, *optional*):\n            Number of points to sample per crop.\n        crop_n_points_downscale_factor (`int`, *optional*):\n            The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.\n        input_data_format (`str` or `ChannelDimension`, *optional*):\n            The channel dimension format of the input image. If not provided, it will be inferred.\n    '
    if isinstance(image, list):
        raise ValueError('Only one image is allowed for crop generation.')
    image = to_numpy_array(image)
    original_size = get_image_size(image, input_data_format)
    points_grid = []
    for i in range(crop_n_layers + 1):
        n_points = int(points_per_crop / crop_n_points_downscale_factor ** i)
        points_grid.append(_build_point_grid(n_points))
    (crop_boxes, layer_idxs) = _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size)
    (cropped_images, point_grid_per_crop) = _generate_crop_images(crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format)
    crop_boxes = np.array(crop_boxes)
    crop_boxes = crop_boxes.astype(np.float32)
    points_per_crop = np.array([point_grid_per_crop])
    points_per_crop = np.transpose(points_per_crop, axes=(0, 2, 1, 3))
    input_labels = np.ones_like(points_per_crop[:, :, :, 0], dtype=np.int64)
    return (crop_boxes, points_per_crop, cropped_images, input_labels)

def _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size):
    if False:
        while True:
            i = 10
    '\n    Generates 2 ** (layers idx + 1) crops for each crop_n_layers. Crops are in the XYWH format : The XYWH format\n    consists of the following required indices:\n        - X: X coordinate of the top left of the bounding box\n        - Y: Y coordinate of the top left of the bounding box\n        - W: width of the bounding box\n        - H: height of the bounding box\n    '
    (crop_boxes, layer_idxs) = ([], [])
    (im_height, im_width) = original_size
    short_side = min(im_height, im_width)
    crop_boxes.append([0, 0, im_width, im_height])
    layer_idxs.append(0)
    for i_layer in range(crop_n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
        crop_width = int(math.ceil((overlap * (n_crops_per_side - 1) + im_width) / n_crops_per_side))
        crop_height = int(math.ceil((overlap * (n_crops_per_side - 1) + im_height) / n_crops_per_side))
        crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_height - overlap) * i) for i in range(n_crops_per_side)]
        for (left, top) in product(crop_box_x0, crop_box_y0):
            box = [left, top, min(left + crop_width, im_width), min(top + crop_height, im_height)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)
    return (crop_boxes, layer_idxs)

def _generate_crop_images(crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes as an input bounding boxes that are used to crop the image. Based in the crops, the corresponding points are\n    also passed.\n    '
    cropped_images = []
    total_points_per_crop = []
    for (i, crop_box) in enumerate(crop_boxes):
        (left, top, right, bottom) = crop_box
        channel_dim = infer_channel_dimension_format(image, input_data_format)
        if channel_dim == ChannelDimension.LAST:
            cropped_im = image[top:bottom, left:right, :]
        else:
            cropped_im = image[:, top:bottom, left:right]
        cropped_images.append(cropped_im)
        cropped_im_size = get_image_size(cropped_im, channel_dim)
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points = points_grid[layer_idxs[i]] * points_scale
        normalized_points = _normalize_coordinates(target_size, points, original_size)
        total_points_per_crop.append(normalized_points)
    return (cropped_images, total_points_per_crop)

def _pad_masks(masks, crop_box: List[int], orig_height: int, orig_width: int):
    if False:
        while True:
            i = 10
    (left, top, right, bottom) = crop_box
    if left == 0 and top == 0 and (right == orig_width) and (bottom == orig_height):
        return masks
    (pad_x, pad_y) = (orig_width - (right - left), orig_height - (bottom - top))
    pad = (left, pad_x - left, top, pad_y - top)
    return torch.nn.functional.pad(masks, pad, value=0)

def _pad_masks_tf(masks, crop_box: List[int], orig_height: int, orig_width: int):
    if False:
        print('Hello World!')
    (left, top, right, bottom) = crop_box
    if left == 0 and top == 0 and (right == orig_width) and (bottom == orig_height):
        return masks
    (pad_x, pad_y) = (orig_width - (right - left), orig_height - (bottom - top))
    pad = (left, pad_x - left, top, pad_y - top)
    return tf.pad(masks, pad, constant_values=0)

def _is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    if False:
        return 10
    'Filter masks at the edge of a crop, but not at the edge of the original image.'
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    (left, top, _, _) = crop_box
    offset = torch.tensor([[left, top, left, top]], device=boxes.device)
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    boxes = (boxes + offset).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)

def _is_box_near_crop_edge_tf(boxes, crop_box, orig_box, atol=20.0):
    if False:
        i = 10
        return i + 15
    'Filter masks at the edge of a crop, but not at the edge of the original image.'
    crop_box_tf = tf.convert_to_tensor(crop_box, dtype=tf.float32)
    orig_box_tf = tf.convert_to_tensor(orig_box, dtype=tf.float32)
    (left, top, _, _) = crop_box
    offset = tf.convert_to_tensor([[left, top, left, top]])
    if len(boxes.shape) == 3:
        offset = tf.expand_dims(offset, 1)
    boxes = tf.cast(boxes + offset, tf.float32)
    near_crop_edge = tnp.isclose(boxes, crop_box_tf[None, :], atol=atol, rtol=0)
    near_image_edge = tnp.isclose(boxes, orig_box_tf[None, :], atol=atol, rtol=0)
    near_crop_edge = tf.math.logical_and(near_crop_edge, ~near_image_edge)
    return tf.reduce_any(near_crop_edge, axis=1)

def _batched_mask_to_box(masks: 'torch.Tensor'):
    if False:
        while True:
            i = 10
    '\n    Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format which\n    corresponds the following required indices:\n        - LEFT: left hand side of the bounding box\n        - TOP: top of the bounding box\n        - RIGHT: right of the bounding box\n        - BOTTOM: bottom of the bounding box\n\n    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x ... x height x width, the output shape\n    is channel_1 x channel_2 x ... x 4.\n\n    Args:\n        - masks (`torch.Tensor` of shape `(batch, nb_mask, height, width)`)\n    '
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    (height, width) = shape[-2:]
    (in_height, _) = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(height, device=in_height.device)[None, :]
    (bottom_edges, _) = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + height * ~in_height
    (top_edges, _) = torch.min(in_height_coords, dim=-1)
    (in_width, _) = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(width, device=in_width.device)[None, :]
    (right_edges, _) = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + width * ~in_width
    (left_edges, _) = torch.min(in_width_coords, dim=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)
    out = out.reshape(*shape[:-2], 4)
    return out

def _batched_mask_to_box_tf(masks: 'tf.Tensor'):
    if False:
        return 10
    '\n    Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format which\n    corresponds the following required indices:\n        - LEFT: left hand side of the bounding box\n        - TOP: top of the bounding box\n        - RIGHT: right of the bounding box\n        - BOTTOM: bottom of the bounding box\n\n    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x ... x height x width, the output shape\n    is channel_1 x channel_2 x ... x 4.\n\n    Args:\n        - masks (`tf.Tensor` of shape `(batch, nb_mask, height, width)`)\n    '
    if tf.size(masks) == 0:
        return tf.zeros([*masks.shape[:-2], 4])
    shape = shape_list(masks)
    (height, width) = shape[-2:]
    in_height = tf.reduce_max(masks, axis=-1)
    in_height_coords = in_height * tf.range(height)[None, :]
    bottom_edges = tf.reduce_max(in_height_coords, axis=-1)
    in_height_coords = in_height_coords + height * ~in_height
    top_edges = tf.reduce_min(in_height_coords, axis=-1)
    (in_width, _) = tf.reduce_max(masks, axis=-2)
    in_width_coords = in_width * tf.range(width)[None, :]
    (right_edges, _) = tf.reduce_max(in_width_coords, axis=-1)
    in_width_coords = in_width_coords + width * ~in_width
    (left_edges, _) = tf.reduce_min(in_width_coords, axis=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = tf.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = out * tf.expand_dims(~empty_filter, -1)
    out = tf.reshape(out, *shape[:-2], 4)
    return out

def _mask_to_rle_pytorch(input_mask: 'torch.Tensor'):
    if False:
        return 10
    '\n    Encodes masks the run-length encoding (RLE), in the format expected by pycoco tools.\n    '
    (batch_size, height, width) = input_mask.shape
    input_mask = input_mask.permute(0, 2, 1).flatten(1)
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    change_indices = diff.nonzero()
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if input_mask[i, 0] == 0 else [0]
        counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1]]
        out.append({'size': [height, width], 'counts': counts})
    return out

def _mask_to_rle_tf(input_mask: 'tf.Tensor'):
    if False:
        return 10
    '\n    Encodes masks the run-length encoding (RLE), in the format expected by pycoco tools.\n    '
    (batch_size, height, width) = input_mask.shape
    input_mask = flatten(tf.transpose(input_mask, perm=(0, 2, 1)), 1)
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    change_indices = tf.where(diff)
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if input_mask[i, 0] == 0 else [0]
        counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1]]
        out.append({'size': [height, width], 'counts': counts})
    return out

def _rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Compute a binary mask from an uncompressed RLE.'
    (height, width) = rle['size']
    mask = np.empty(height * width, dtype=bool)
    idx = 0
    parity = False
    for count in rle['counts']:
        mask[idx:idx + count] = parity
        idx += count
        parity = not parity
    mask = mask.reshape(width, height)
    return mask.transpose()

def _postprocess_for_mg(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
    if False:
        while True:
            i = 10
    '\n    Perform NMS (Non Maximum Suppression) on the outputs.\n\n    Args:\n            rle_masks (`torch.Tensor`):\n                binary masks in the RLE format\n            iou_scores (`torch.Tensor` of shape (nb_masks, 1)):\n                iou_scores predicted by the model\n            mask_boxes (`torch.Tensor`):\n                The bounding boxes corresponding to segmentation masks\n            amg_crops_nms_thresh (`float`, *optional*, defaults to 0.7):\n                NMS threshold.\n    '
    keep_by_nms = batched_nms(boxes=mask_boxes.float(), scores=iou_scores, idxs=torch.zeros(mask_boxes.shape[0]), iou_threshold=amg_crops_nms_thresh)
    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]
    return (masks, iou_scores, rle_masks, mask_boxes)

def _postprocess_for_mg_tf(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform NMS (Non Maximum Suppression) on the outputs.\n\n    Args:\n            rle_masks (`tf.Tensor`):\n                binary masks in the RLE format\n            iou_scores (`tf.Tensor` of shape (nb_masks, 1)):\n                iou_scores predicted by the model\n            mask_boxes (`tf.Tensor`):\n                The bounding boxes corresponding to segmentation masks\n            amg_crops_nms_thresh (`float`, *optional*, defaults to 0.7):\n                NMS threshold.\n    '
    keep_by_nms = tf.image.combined_non_max_suppression(boxes=mask_boxes.float(), scores=iou_scores, idxs=torch.zeros(mask_boxes.shape[0]), iou_threshold=amg_crops_nms_thresh)
    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]
    return (masks, iou_scores, rle_masks, mask_boxes)