"""Image processor class for Deformable DETR."""
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import PaddingMode, center_to_corners_format, corners_to_center_format, pad, rescale, resize, rgb_to_id, to_channel_dimension_format
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ChannelDimension, ImageInput, PILImageResampling, get_image_size, infer_channel_dimension_format, is_batched, is_scaled_image, to_numpy_array, valid_coco_detection_annotations, valid_coco_panoptic_annotations, valid_images
from ...utils import is_flax_available, is_jax_tensor, is_tf_available, is_tf_tensor, is_torch_available, is_torch_tensor, is_torchvision_available, is_vision_available, logging
from ...utils.generic import ExplicitEnum, TensorType
if is_torch_available():
    import torch
if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms
if is_vision_available():
    import PIL
logger = logging.get_logger(__name__)

class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = 'coco_detection'
    COCO_PANOPTIC = 'coco_panoptic'
SUPPORTED_ANNOTATION_FORMATS = (AnnotionFormat.COCO_DETECTION, AnnotionFormat.COCO_PANOPTIC)

def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the output image size given the input image size and the desired output size.\n\n    Args:\n        image_size (`Tuple[int, int]`):\n            The input image size.\n        size (`int`):\n            The desired output size.\n        max_size (`int`, *optional*):\n            The maximum allowed output size.\n    '
    (height, width) = image_size
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if height <= width and height == size or (width <= height and width == size):
        return (height, width)
    if width < height:
        ow = size
        oh = int(size * height / width)
    else:
        oh = size
        ow = int(size * width / height)
    return (oh, ow)

def get_resize_output_image_size(input_image: np.ndarray, size: Union[int, Tuple[int, int], List[int]], max_size: Optional[int]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> Tuple[int, int]:
    if False:
        print('Hello World!')
    '\n    Computes the output image size given the input image size and the desired output size. If the desired output size\n    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output\n    image size is computed by keeping the aspect ratio of the input image size.\n\n    Args:\n        input_image (`np.ndarray`):\n            The image to resize.\n        size (`int` or `Tuple[int, int]` or `List[int]`):\n            The desired output size.\n        max_size (`int`, *optional*):\n            The maximum allowed output size.\n        input_data_format (`ChannelDimension` or `str`, *optional*):\n            The channel dimension format of the input image. If not provided, it will be inferred from the input image.\n    '
    image_size = get_image_size(input_image, input_data_format)
    if isinstance(size, (list, tuple)):
        return size
    return get_size_with_aspect_ratio(image_size, size, max_size)

def get_numpy_to_framework_fn(arr) -> Callable:
    if False:
        i = 10
        return i + 15
    '\n    Returns a function that converts a numpy array to the framework of the input array.\n\n    Args:\n        arr (`np.ndarray`): The array to convert.\n    '
    if isinstance(arr, np.ndarray):
        return np.array
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf
        return tf.convert_to_tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch
        return torch.tensor
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp
        return jnp.array
    raise ValueError(f'Cannot convert arrays of type {type(arr)}')

def safe_squeeze(arr: np.ndarray, axis: Optional[int]=None) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Squeezes an array, but only if the axis specified has dim 1.\n    '
    if axis is None:
        return arr.squeeze()
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr

def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    if False:
        while True:
            i = 10
    (image_height, image_width) = image_size
    norm_annotation = {}
    for (key, value) in annotation.items():
        if key == 'boxes':
            boxes = value
            boxes = corners_to_center_format(boxes)
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            norm_annotation[key] = boxes
        else:
            norm_annotation[key] = value
    return norm_annotation

def max_across_indices(values: Iterable[Any]) -> List[Any]:
    if False:
        return 10
    '\n    Return the maximum value across all indices of an iterable of values.\n    '
    return [max(values_i) for values_i in zip(*values)]

def get_max_height_width(images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]]=None) -> List[int]:
    if False:
        while True:
            i = 10
    '\n    Get the maximum height and width across all images in a batch.\n    '
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])
    if input_data_format == ChannelDimension.FIRST:
        (_, max_height, max_width) = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        (max_height, max_width, _) = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f'Invalid channel dimension format: {input_data_format}')
    return (max_height, max_width)

def make_pixel_mask(image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.\n\n    Args:\n        image (`np.ndarray`):\n            Image to make the pixel mask for.\n        output_size (`Tuple[int, int]`):\n            Output size of the mask.\n    '
    (input_height, input_width) = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask

def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Convert a COCO polygon annotation to a mask.\n\n    Args:\n        segmentations (`List[List[float]]`):\n            List of polygons, each polygon represented by a list of x-y coordinates.\n        height (`int`):\n            Height of the mask.\n        width (`int`):\n            Width of the mask.\n    '
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError('Pycocotools is not installed in your environment.')
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)
    return masks

def prepare_coco_detection_annotation(image, target, return_segmentation_masks: bool=False, input_data_format: Optional[Union[ChannelDimension, str]]=None):
    if False:
        i = 10
        return i + 15
    '\n    Convert the target in COCO format into the format expected by DETA.\n    '
    (image_height, image_width) = get_image_size(image, channel_dim=input_data_format)
    image_id = target['image_id']
    image_id = np.asarray([image_id], dtype=np.int64)
    annotations = target['annotations']
    annotations = [obj for obj in annotations if 'iscrowd' not in obj or obj['iscrowd'] == 0]
    classes = [obj['category_id'] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)
    area = np.asarray([obj['area'] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj['iscrowd'] if 'iscrowd' in obj else 0 for obj in annotations], dtype=np.int64)
    boxes = [obj['bbox'] for obj in annotations]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    new_target = {}
    new_target['image_id'] = image_id
    new_target['class_labels'] = classes[keep]
    new_target['boxes'] = boxes[keep]
    new_target['area'] = area[keep]
    new_target['iscrowd'] = iscrowd[keep]
    new_target['orig_size'] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)
    if annotations and 'keypoints' in annotations[0]:
        keypoints = [obj['keypoints'] for obj in annotations]
        keypoints = np.asarray(keypoints, dtype=np.float32)
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target['keypoints'] = keypoints[keep]
    if return_segmentation_masks:
        segmentation_masks = [obj['segmentation'] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target['masks'] = masks[keep]
    return new_target

def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    '\n    Compute the bounding boxes around the provided panoptic segmentation masks.\n\n    Args:\n        masks: masks in format `[number_masks, height, width]` where N is the number of masks\n\n    Returns:\n        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format\n    '
    if masks.size == 0:
        return np.zeros((0, 4))
    (h, w) = masks.shape[-2:]
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    (y, x) = np.meshgrid(y, x, indexing='ij')
    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~np.array(masks, dtype=bool))
    x_min = x.filled(fill_value=100000000.0)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)
    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~np.array(masks, dtype=bool))
    y_min = y.filled(fill_value=100000000.0)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)
    return np.stack([x_min, y_min, x_max, y_max], 1)

def prepare_coco_panoptic_annotation(image: np.ndarray, target: Dict, masks_path: Union[str, pathlib.Path], return_masks: bool=True, input_data_format: Union[ChannelDimension, str]=None) -> Dict:
    if False:
        return 10
    '\n    Prepare a coco panoptic annotation for DETA.\n    '
    (image_height, image_width) = get_image_size(image, channel_dim=input_data_format)
    annotation_path = pathlib.Path(masks_path) / target['file_name']
    new_target = {}
    new_target['image_id'] = np.asarray([target['image_id'] if 'image_id' in target else target['id']], dtype=np.int64)
    new_target['size'] = np.asarray([image_height, image_width], dtype=np.int64)
    new_target['orig_size'] = np.asarray([image_height, image_width], dtype=np.int64)
    if 'segments_info' in target:
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)
        ids = np.array([segment_info['id'] for segment_info in target['segments_info']])
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        if return_masks:
            new_target['masks'] = masks
        new_target['boxes'] = masks_to_boxes(masks)
        new_target['class_labels'] = np.array([segment_info['category_id'] for segment_info in target['segments_info']], dtype=np.int64)
        new_target['iscrowd'] = np.asarray([segment_info['iscrowd'] for segment_info in target['segments_info']], dtype=np.int64)
        new_target['area'] = np.asarray([segment_info['area'] for segment_info in target['segments_info']], dtype=np.float32)
    return new_target

def resize_annotation(annotation: Dict[str, Any], orig_size: Tuple[int, int], target_size: Tuple[int, int], threshold: float=0.5, resample: PILImageResampling=PILImageResampling.NEAREST):
    if False:
        while True:
            i = 10
    '\n    Resizes an annotation to a target size.\n\n    Args:\n        annotation (`Dict[str, Any]`):\n            The annotation dictionary.\n        orig_size (`Tuple[int, int]`):\n            The original size of the input image.\n        target_size (`Tuple[int, int]`):\n            The target size of the image, as returned by the preprocessing `resize` step.\n        threshold (`float`, *optional*, defaults to 0.5):\n            The threshold used to binarize the segmentation masks.\n        resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):\n            The resampling filter to use when resizing the masks.\n    '
    ratios = tuple((float(s) / float(s_orig) for (s, s_orig) in zip(target_size, orig_size)))
    (ratio_height, ratio_width) = ratios
    new_annotation = {}
    new_annotation['size'] = target_size
    for (key, value) in annotation.items():
        if key == 'boxes':
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            new_annotation['boxes'] = scaled_boxes
        elif key == 'area':
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            new_annotation['area'] = scaled_area
        elif key == 'masks':
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold
            new_annotation['masks'] = masks
        elif key == 'size':
            new_annotation['size'] = target_size
        else:
            new_annotation[key] = value
    return new_annotation

class DetaImageProcessor(BaseImageProcessor):
    """
    Constructs a Deformable DETR image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's (height, width) dimensions after resizing. Can be overridden by the `size` parameter in
            the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image to the largest image in a batch and create a pixel mask. Can be
            overridden by the `do_pad` parameter in the `preprocess` method.
    """
    model_input_names = ['pixel_values', 'pixel_mask']

    def __init__(self, format: Union[str, AnnotionFormat]=AnnotionFormat.COCO_DETECTION, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Union[float, List[float]]=None, image_std: Union[float, List[float]]=None, do_pad: bool=True, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        if 'pad_and_return_pixel_mask' in kwargs:
            do_pad = kwargs.pop('pad_and_return_pixel_mask')
        size = size if size is not None else {'shortest_edge': 800, 'longest_edge': 1333}
        size = get_size_dict(size, default_to_square=False)
        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad

    def prepare_annotation(self, image: np.ndarray, target: Dict, format: Optional[AnnotionFormat]=None, return_segmentation_masks: bool=None, masks_path: Optional[Union[str, pathlib.Path]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> Dict:
        if False:
            return 10
        '\n        Prepare an annotation for feeding into DETA model.\n        '
        format = format if format is not None else self.format
        if format == AnnotionFormat.COCO_DETECTION:
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_detection_annotation(image, target, return_segmentation_masks, input_data_format=input_data_format)
        elif format == AnnotionFormat.COCO_PANOPTIC:
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_panoptic_annotation(image, target, masks_path=masks_path, return_masks=return_segmentation_masks, input_data_format=input_data_format)
        else:
            raise ValueError(f'Format {format} is not supported.')
        return target

    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        if False:
            while True:
                i = 10
        logger.warning_once('The `prepare` method is deprecated and will be removed in a v4.33. Please use `prepare_annotation` instead. Note: the `prepare_annotation` method does not return the image anymore.')
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return (image, target)

    def convert_coco_poly_to_mask(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        logger.warning_once('The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ')
        return convert_coco_poly_to_mask(*args, **kwargs)

    def prepare_coco_detection(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        logger.warning_once('The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ')
        return prepare_coco_detection_annotation(*args, **kwargs)

    def prepare_coco_panoptic(self, *args, **kwargs):
        if False:
            return 10
        logger.warning_once('The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ')
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BILINEAR, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an\n        int, smaller edge of the image will be matched to this number.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                The desired output size. Can contain keys `shortest_edge` and `longest_edge` or `height` and `width`.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):\n                Resampling filter to use if resizing the image.\n            data_format (`ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred from the input\n                image.\n        '
        size = get_size_dict(size, default_to_square=False)
        if 'shortest_edge' in size and 'longest_edge' in size:
            size = get_resize_output_image_size(image, size['shortest_edge'], size['longest_edge'], input_data_format=input_data_format)
        elif 'height' in size and 'width' in size:
            size = (size['height'], size['width'])
        else:
            raise ValueError(f"Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got {size.keys()}.")
        image = resize(image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format)
        return image

    def resize_annotation(self, annotation, orig_size, size, resample: PILImageResampling=PILImageResampling.NEAREST) -> Dict:
        if False:
            return 10
        '\n        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched\n        to this number.\n        '
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    def rescale(self, image: np.ndarray, rescale_factor: float, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            return 10
        '\n        Rescale the image by the given factor. image = image * rescale_factor.\n\n        Args:\n            image (`np.ndarray`):\n                Image to rescale.\n            rescale_factor (`float`):\n                The value to use for rescaling.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the input image. If unset, is inferred from the input image. Can be\n                one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n        '
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        if False:
            print('Hello World!')
        '\n        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to\n        `[center_x, center_y, width, height]` format.\n        '
        return normalize_annotation(annotation, image_size=image_size)

    def _pad_image(self, image: np.ndarray, output_size: Tuple[int, int], constant_values: Union[float, Iterable[float]]=0, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Pad an image with zeros to the given size.\n        '
        (input_height, input_width) = get_image_size(image, channel_dim=input_data_format)
        (output_height, output_width) = output_size
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(image, padding, mode=PaddingMode.CONSTANT, constant_values=constant_values, data_format=data_format, input_data_format=input_data_format)
        return padded_image

    def pad(self, images: List[np.ndarray], constant_values: Union[float, Iterable[float]]=0, return_pixel_mask: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> BatchFeature:
        if False:
            return 10
        '\n        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width\n        in the batch and optionally returns their corresponding pixel mask.\n\n        Args:\n            image (`np.ndarray`):\n                Image to pad.\n            constant_values (`float` or `Iterable[float]`, *optional*):\n                The value to use for the padding if `mode` is `"constant"`.\n            return_pixel_mask (`bool`, *optional*, defaults to `True`):\n                Whether to return a pixel mask.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        pad_size = get_max_height_width(images, input_data_format=input_data_format)
        padded_images = [self._pad_image(image, pad_size, constant_values=constant_values, data_format=data_format, input_data_format=input_data_format) for image in images]
        data = {'pixel_values': padded_images}
        if return_pixel_mask:
            masks = [make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format) for image in images]
            data['pixel_mask'] = masks
        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess(self, images: ImageInput, annotations: Optional[Union[List[Dict], List[List[Dict]]]]=None, return_segmentation_masks: bool=None, masks_path: Optional[Union[str, pathlib.Path]]=None, do_resize: Optional[bool]=None, size: Optional[Dict[str, int]]=None, resample=None, do_rescale: Optional[bool]=None, rescale_factor: Optional[Union[int, float]]=None, do_normalize: Optional[bool]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, format: Optional[Union[str, AnnotionFormat]]=None, return_tensors: Optional[Union[TensorType, str]]=None, data_format: Union[str, ChannelDimension]=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> BatchFeature:
        if False:
            while True:
                i = 10
        '\n        Preprocess an image or a batch of images so that it can be used by the model.\n\n        Args:\n            images (`ImageInput`):\n                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging\n                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            annotations (`List[Dict]` or `List[List[Dict]]`, *optional*):\n                List of annotations associated with the image or batch of images. If annotionation is for object\n                detection, the annotations should be a dictionary with the following keys:\n                - "image_id" (`int`): The image id.\n                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a\n                  dictionary. An image can have no annotations, in which case the list should be empty.\n                If annotionation is for segmentation, the annotations should be a dictionary with the following keys:\n                - "image_id" (`int`): The image id.\n                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.\n                  An image can have no segments, in which case the list should be empty.\n                - "file_name" (`str`): The file name of the image.\n            return_segmentation_masks (`bool`, *optional*, defaults to self.return_segmentation_masks):\n                Whether to return segmentation masks.\n            masks_path (`str` or `pathlib.Path`, *optional*):\n                Path to the directory containing the segmentation masks.\n            do_resize (`bool`, *optional*, defaults to self.do_resize):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to self.size):\n                Size of the image after resizing.\n            resample (`PILImageResampling`, *optional*, defaults to self.resample):\n                Resampling filter to use when resizing the image.\n            do_rescale (`bool`, *optional*, defaults to self.do_rescale):\n                Whether to rescale the image.\n            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):\n                Rescale factor to use when rescaling the image.\n            do_normalize (`bool`, *optional*, defaults to self.do_normalize):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):\n                Mean to use when normalizing the image.\n            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):\n                Standard deviation to use when normalizing the image.\n            do_pad (`bool`, *optional*, defaults to self.do_pad):\n                Whether to pad the image.\n            format (`str` or `AnnotionFormat`, *optional*, defaults to self.format):\n                Format of the annotations.\n            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):\n                Type of tensors to return. If `None`, will return the list of images.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        if 'pad_and_return_pixel_mask' in kwargs:
            logger.warning_once('The `pad_and_return_pixel_mask` argument is deprecated and will be removed in a future version, use `do_pad` instead.')
            do_pad = kwargs.pop('pad_and_return_pixel_mask')
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, default_to_square=False)
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_pad = self.do_pad if do_pad is None else do_pad
        format = self.format if format is None else format
        if do_resize is not None and size is None:
            raise ValueError('Size and max_size must be specified if do_resize is True.')
        if do_rescale is not None and rescale_factor is None:
            raise ValueError('Rescale factor must be specified if do_rescale is True.')
        if do_normalize is not None and (image_mean is None or image_std is None):
            raise ValueError('Image mean and std must be specified if do_normalize is True.')
        if not is_batched(images):
            images = [images]
            annotations = [annotations] if annotations is not None else None
        if annotations is not None and len(images) != len(annotations):
            raise ValueError(f'The number of images ({len(images)}) and annotations ({len(annotations)}) do not match.')
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        format = AnnotionFormat(format)
        if annotations is not None:
            if format == AnnotionFormat.COCO_DETECTION and (not valid_coco_detection_annotations(annotations)):
                raise ValueError('Invalid COCO detection annotations. Annotations must a dict (single image) of list of dicts (batch of images) with the following keys: `image_id` and `annotations`, with the latter being a list of annotations in the COCO format.')
            elif format == AnnotionFormat.COCO_PANOPTIC and (not valid_coco_panoptic_annotations(annotations)):
                raise ValueError('Invalid COCO panoptic annotations. Annotations must a dict (single image) of list of dicts (batch of images) with the following keys: `image_id`, `file_name` and `segments_info`, with the latter being a list of annotations in the COCO format.')
            elif format not in SUPPORTED_ANNOTATION_FORMATS:
                raise ValueError(f'Unsupported annotation format: {format} must be one of {SUPPORTED_ANNOTATION_FORMATS}')
        if masks_path is not None and format == AnnotionFormat.COCO_PANOPTIC and (not isinstance(masks_path, (pathlib.Path, str))):
            raise ValueError(f'The path to the directory containing the mask PNG files should be provided as a `pathlib.Path` or string object, but is {type(masks_path)} instead.')
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if annotations is not None:
            prepared_images = []
            prepared_annotations = []
            for (image, target) in zip(images, annotations):
                target = self.prepare_annotation(image, target, format, return_segmentation_masks=return_segmentation_masks, masks_path=masks_path, input_data_format=input_data_format)
                prepared_images.append(image)
                prepared_annotations.append(target)
            images = prepared_images
            annotations = prepared_annotations
            del prepared_images, prepared_annotations
        if do_resize:
            if annotations is not None:
                (resized_images, resized_annotations) = ([], [])
                for (image, target) in zip(images, annotations):
                    orig_size = get_image_size(image, input_data_format)
                    resized_image = self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
                    resized_annotation = self.resize_annotation(target, orig_size, get_image_size(resized_image, input_data_format))
                    resized_images.append(resized_image)
                    resized_annotations.append(resized_annotation)
                images = resized_images
                annotations = resized_annotations
                del resized_images, resized_annotations
            else:
                images = [self.resize(image, size=size, resample=resample, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images]
            if annotations is not None:
                annotations = [self.normalize_annotation(annotation, get_image_size(image, input_data_format)) for (annotation, image) in zip(annotations, images)]
        if do_pad:
            data = self.pad(images, return_pixel_mask=True, data_format=data_format, input_data_format=input_data_format)
        else:
            images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
            data = {'pixel_values': images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        if annotations is not None:
            encoded_inputs['labels'] = [BatchFeature(annotation, tensor_type=return_tensors) for annotation in annotations]
        return encoded_inputs

    def post_process_object_detection(self, outputs, threshold: float=0.5, target_sizes: Union[TensorType, List[Tuple]]=None, nms_threshold: float=0.7):
        if False:
            print('Hello World!')
        '\n        Converts the output of [`DetaForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,\n        bottom_right_x, bottom_right_y) format. Only supports PyTorch.\n\n        Args:\n            outputs ([`DetrObjectDetectionOutput`]):\n                Raw outputs of the model.\n            threshold (`float`, *optional*, defaults to 0.5):\n                Score threshold to keep object detection predictions.\n            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):\n                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size\n                (height, width) of each image in the batch. If left to None, predictions will not be resized.\n            nms_threshold (`float`, *optional*, defaults to 0.7):\n                NMS threshold.\n\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image\n            in the batch as predicted by the model.\n        '
        (out_logits, out_bbox) = (outputs.logits, outputs.pred_boxes)
        (batch_size, num_queries, num_labels) = out_logits.shape
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
        prob = out_logits.sigmoid()
        all_scores = prob.view(batch_size, num_queries * num_labels).to(out_logits.device)
        all_indexes = torch.arange(num_queries * num_labels)[None].repeat(batch_size, 1).to(out_logits.device)
        all_boxes = torch.div(all_indexes, out_logits.shape[2], rounding_mode='floor')
        all_labels = all_indexes % out_logits.shape[2]
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                (img_h, img_w) = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
        results = []
        for b in range(batch_size):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]
            pre_topk = score.topk(min(10000, len(score))).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]
            keep_inds = batched_nms(box, score, lbls, nms_threshold)[:100]
            score = score[keep_inds]
            lbls = lbls[keep_inds]
            box = box[keep_inds]
            results.append({'scores': score[score > threshold], 'labels': lbls[score > threshold], 'boxes': box[score > threshold]})
        return results