"""Image processor class for OWLv2."""
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import center_to_corners_format, pad, to_channel_dimension_format
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ChannelDimension, ImageInput, PILImageResampling, get_image_size, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_scipy_available, is_torch_available, is_vision_available, logging, requires_backends
if is_torch_available():
    import torch
if is_vision_available():
    import PIL
if is_scipy_available():
    from scipy import ndimage as ndi
logger = logging.get_logger(__name__)

def _upcast(t):
    if False:
        while True:
            i = 10
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def box_area(boxes):
    if False:
        return 10
    '\n    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.\n\n    Args:\n        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):\n            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1\n            < x2` and `0 <= y1 < y2`.\n    Returns:\n        `torch.FloatTensor`: a tensor containing the area for each box.\n    '
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    if False:
        return 10
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    inter = width_height[:, :, 0] * width_height[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return (iou, union)

def _preprocess_resize_output_shape(image, output_shape):
    if False:
        print('Hello World!')
    'Validate resize output shape according to input image.\n\n    Args:\n        image (`np.ndarray`):\n         Image to be resized.\n        output_shape (`iterable`):\n            Size of the generated output image `(rows, cols[, ...][, dim])`. If `dim` is not provided, the number of\n            channels is preserved.\n\n    Returns\n        image (`np.ndarray):\n            The input image, but with additional singleton dimensions appended in the case where `len(output_shape) >\n            input.ndim`.\n        output_shape (`Tuple`):\n            The output shape converted to tuple.\n\n    Raises ------ ValueError:\n        If output_shape length is smaller than the image number of dimensions.\n\n    Notes ----- The input image is reshaped if its number of dimensions is not equal to output_shape_length.\n\n    '
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        input_shape += (1,) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        output_shape = output_shape + (image.shape[-1],)
    elif output_ndim < image.ndim:
        raise ValueError('output_shape length cannot be smaller than the image number of dimensions')
    return (image, output_shape)

def _clip_warp_output(input_image, output_image):
    if False:
        return 10
    'Clip output image to range of values of input image.\n\n    Note that this function modifies the values of *output_image* in-place.\n\n    Taken from:\n    https://github.com/scikit-image/scikit-image/blob/b4b521d6f0a105aabeaa31699949f78453ca3511/skimage/transform/_warps.py#L640.\n\n    Args:\n        input_image : ndarray\n            Input image.\n        output_image : ndarray\n            Output image, which is modified in-place.\n    '
    min_val = np.min(input_image)
    if np.isnan(min_val):
        min_func = np.nanmin
        max_func = np.nanmax
        min_val = min_func(input_image)
    else:
        min_func = np.min
        max_func = np.max
    max_val = max_func(input_image)
    output_image = np.clip(output_image, min_val, max_val)
    return output_image

class Owlv2ImageProcessor(BaseImageProcessor):
    """
    Constructs an OWLv2 image processor.

    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overriden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overriden by `rescale_factor` in the `preprocess`
            method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to a square with gray pixels on the bottom and the right. Can be overriden by
            `do_pad` in the `preprocess` method.
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be overriden
            by `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 960, "width": 960}`):
            Size to resize the image to. Can be overriden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling method to use if resizing the image. Can be overriden by `resample` in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_pad: bool=True, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.do_resize = do_resize
        self.size = size if size is not None else {'height': 960, 'width': 960}
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def pad(self, image: np.array, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None):
        if False:
            print('Hello World!')
        '\n        Pad an image to a square with gray pixels on the bottom and the right, as per the original OWLv2\n        implementation.\n\n        Args:\n            image (`np.ndarray`):\n                Image to pad.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred from the input\n                image.\n        '
        (height, width) = get_image_size(image)
        size = max(height, width)
        image = pad(image=image, padding=((0, size - height), (0, size - width)), constant_values=0.5, data_format=data_format, input_data_format=input_data_format)
        return image

    def resize(self, image: np.ndarray, size: Dict[str, int], anti_aliasing: bool=True, anti_aliasing_sigma=None, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Resize an image as per the original implementation.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Dictionary containing the height and width to resize the image to.\n            anti_aliasing (`bool`, *optional*, defaults to `True`):\n                Whether to apply anti-aliasing when downsampling the image.\n            anti_aliasing_sigma (`float`, *optional*, defaults to `None`):\n                Standard deviation for Gaussian kernel when downsampling the image. If `None`, it will be calculated\n                automatically.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred from the input\n                image.\n        '
        requires_backends(self, 'scipy')
        output_shape = (size['height'], size['width'])
        image = to_channel_dimension_format(image, ChannelDimension.LAST)
        (image, output_shape) = _preprocess_resize_output_shape(image, output_shape)
        input_shape = image.shape
        factors = np.divide(input_shape, output_shape)
        ndi_mode = 'mirror'
        cval = 0
        order = 1
        if anti_aliasing:
            if anti_aliasing_sigma is None:
                anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
            else:
                anti_aliasing_sigma = np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
                if np.any(anti_aliasing_sigma < 0):
                    raise ValueError('Anti-aliasing standard deviation must be greater than or equal to zero')
                elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                    warnings.warn('Anti-aliasing standard deviation greater than zero but not down-sampling along all axes')
            filtered = ndi.gaussian_filter(image, anti_aliasing_sigma, cval=cval, mode=ndi_mode)
        else:
            filtered = image
        zoom_factors = [1 / f for f in factors]
        out = ndi.zoom(filtered, zoom_factors, order=order, mode=ndi_mode, cval=cval, grid_mode=True)
        image = _clip_warp_output(image, out)
        image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
        image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        return image

    def preprocess(self, images: ImageInput, do_pad: bool=None, do_resize: bool=None, size: Dict[str, int]=None, do_rescale: bool=None, rescale_factor: float=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> PIL.Image.Image:
        if False:
            return 10
        '\n        Preprocess an image or batch of images.\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If\n                passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            do_pad (`bool`, *optional*, defaults to `self.do_pad`):\n                Whether to pad the image to a square with gray pixels on the bottom and the right.\n            do_resize (`bool`, *optional*, defaults to `self.do_resize`):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to `self.size`):\n                Size to resize the image to.\n            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):\n                Whether to rescale the image values between [0 - 1].\n            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):\n                Rescale factor to rescale the image by if `do_rescale` is set to `True`.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):\n                Image mean.\n            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):\n                Image standard deviation.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = size if size is not None else self.size
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_resize and size is None:
            raise ValueError('Size must be specified if do_resize is True.')
        if do_rescale and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
        if do_pad:
            images = [self.pad(image=image, input_data_format=input_data_format) for image in images]
        if do_resize:
            images = [self.resize(image=image, size=size, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format) for image in images]
        images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
        data = {'pixel_values': images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_object_detection(self, outputs, threshold: float=0.1, target_sizes: Union[TensorType, List[Tuple]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,\n        bottom_right_x, bottom_right_y) format.\n\n        Args:\n            outputs ([`OwlViTObjectDetectionOutput`]):\n                Raw outputs of the model.\n            threshold (`float`, *optional*):\n                Score threshold to keep object detection predictions.\n            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):\n                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size\n                `(height, width)` of each image in the batch. If unset, predictions will not be resized.\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image\n            in the batch as predicted by the model.\n        '
        (logits, boxes) = (outputs.logits, outputs.pred_boxes)
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices
        boxes = center_to_corners_format(boxes)
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                (img_h, img_w) = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
        results = []
        for (s, l, b) in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({'scores': score, 'labels': label, 'boxes': box})
        return results

    def post_process_image_guided_detection(self, outputs, threshold=0.0, nms_threshold=0.3, target_sizes=None):
        if False:
            return 10
        '\n        Converts the output of [`OwlViTForObjectDetection.image_guided_detection`] into the format expected by the COCO\n        api.\n\n        Args:\n            outputs ([`OwlViTImageGuidedObjectDetectionOutput`]):\n                Raw outputs of the model.\n            threshold (`float`, *optional*, defaults to 0.0):\n                Minimum confidence threshold to use to filter out predicted boxes.\n            nms_threshold (`float`, *optional*, defaults to 0.3):\n                IoU threshold for non-maximum suppression of overlapping boxes.\n            target_sizes (`torch.Tensor`, *optional*):\n                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in\n                the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to\n                None, predictions will not be unnormalized.\n\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image\n            in the batch as predicted by the model. All labels are set to None as\n            `OwlViTForObjectDetection.image_guided_detection` perform one-shot object detection.\n        '
        (logits, target_boxes) = (outputs.logits, outputs.target_pred_boxes)
        if len(logits) != len(target_sizes):
            raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
        if target_sizes.shape[1] != 2:
            raise ValueError('Each element of target_sizes must contain the size (h, w) of each image of the batch')
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        target_boxes = center_to_corners_format(target_boxes)
        if nms_threshold < 1.0:
            for idx in range(target_boxes.shape[0]):
                for i in torch.argsort(-scores[idx]):
                    if not scores[idx][i]:
                        continue
                    ious = box_iou(target_boxes[idx][i, :].unsqueeze(0), target_boxes[idx])[0][0]
                    ious[i] = -1.0
                    scores[idx][ious > nms_threshold] = 0.0
        (img_h, img_w) = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
        target_boxes = target_boxes * scale_fct[:, None, :]
        results = []
        alphas = torch.zeros_like(scores)
        for idx in range(target_boxes.shape[0]):
            query_scores = scores[idx]
            if not query_scores.nonzero().numel():
                continue
            query_scores[query_scores < threshold] = 0.0
            max_score = torch.max(query_scores) + 1e-06
            query_alphas = (query_scores - max_score * 0.1) / (max_score * 0.9)
            query_alphas = torch.clip(query_alphas, 0.0, 1.0)
            alphas[idx] = query_alphas
            mask = alphas[idx] > 0
            box_scores = alphas[idx][mask]
            boxes = target_boxes[idx][mask]
            results.append({'scores': box_scores, 'labels': None, 'boxes': boxes})
        return results