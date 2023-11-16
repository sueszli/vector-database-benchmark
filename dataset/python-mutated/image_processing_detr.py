"""Image processor class for DETR."""
import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import PaddingMode, center_to_corners_format, corners_to_center_format, id_to_rgb, pad, rescale, resize, rgb_to_id, to_channel_dimension_format
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ChannelDimension, ImageInput, PILImageResampling, get_image_size, infer_channel_dimension_format, is_scaled_image, make_list_of_images, to_numpy_array, valid_coco_detection_annotations, valid_coco_panoptic_annotations, valid_images
from ...utils import ExplicitEnum, TensorType, is_flax_available, is_jax_tensor, is_scipy_available, is_tf_available, is_tf_tensor, is_torch_available, is_torch_tensor, is_vision_available, logging
if is_torch_available():
    import torch
    from torch import nn
if is_vision_available():
    import PIL
if is_scipy_available():
    import scipy.special
    import scipy.stats
logger = logging.get_logger(__name__)
AnnotationType = Dict[str, Union[int, str, List[Dict]]]

class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = 'coco_detection'
    COCO_PANOPTIC = 'coco_panoptic'
SUPPORTED_ANNOTATION_FORMATS = (AnnotionFormat.COCO_DETECTION, AnnotionFormat.COCO_PANOPTIC)

def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    if False:
        return 10
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
        return 10
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
        print('Hello World!')
    '\n    Return the maximum value across all indices of an iterable of values.\n    '
    return [max(values_i) for values_i in zip(*values)]

def get_max_height_width(images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]]=None) -> List[int]:
    if False:
        return 10
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
    '\n    Convert the target in COCO format into the format expected by DETR.\n    '
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
        print('Hello World!')
    '\n    Prepare a coco panoptic annotation for DETR.\n    '
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

def get_segmentation_image(masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False):
    if False:
        print('Hello World!')
    (h, w) = input_size
    (final_h, final_w) = target_size
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)
    if m_id.shape[-1] == 0:
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        m_id = m_id.argmax(-1).reshape(h, w)
    if deduplicate:
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]
    seg_img = id_to_rgb(m_id)
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img

def get_mask_area(seg_img: np.ndarray, target_size: Tuple[int, int], n_classes: int) -> np.ndarray:
    if False:
        print('Hello World!')
    (final_h, final_w) = target_size
    np_seg_img = seg_img.astype(np.uint8)
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    m_id = rgb_to_id(np_seg_img)
    area = [(m_id == i).sum() for i in range(n_classes)]
    return area

def score_labels_from_class_probabilities(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    probs = scipy.special.softmax(logits, axis=-1)
    labels = probs.argmax(-1, keepdims=True)
    scores = np.take_along_axis(probs, labels, axis=-1)
    (scores, labels) = (scores.squeeze(-1), labels.squeeze(-1))
    return (scores, labels)

def post_process_panoptic_sample(out_logits: np.ndarray, masks: np.ndarray, boxes: np.ndarray, processed_size: Tuple[int, int], target_size: Tuple[int, int], is_thing_map: Dict, threshold=0.85) -> Dict:
    if False:
        return 10
    '\n    Converts the output of [`DetrForSegmentation`] into panoptic segmentation predictions for a single sample.\n\n    Args:\n        out_logits (`torch.Tensor`):\n            The logits for this sample.\n        masks (`torch.Tensor`):\n            The predicted segmentation masks for this sample.\n        boxes (`torch.Tensor`):\n            The prediced bounding boxes for this sample. The boxes are in the normalized format `(center_x, center_y,\n            width, height)` and values between `[0, 1]`, relative to the size the image (disregarding padding).\n        processed_size (`Tuple[int, int]`):\n            The processed size of the image `(height, width)`, as returned by the preprocessing step i.e. the size\n            after data augmentation but before batching.\n        target_size (`Tuple[int, int]`):\n            The target size of the image, `(height, width)` corresponding to the requested final size of the\n            prediction.\n        is_thing_map (`Dict`):\n            A dictionary mapping class indices to a boolean value indicating whether the class is a thing or not.\n        threshold (`float`, *optional*, defaults to 0.85):\n            The threshold used to binarize the segmentation masks.\n    '
    (scores, labels) = score_labels_from_class_probabilities(out_logits)
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])
    if len(cur_boxes) != len(cur_classes):
        raise ValueError('Not as many boxes as there are classes')
    cur_masks = masks[keep]
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PILImageResampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    (b, h, w) = cur_masks.shape
    cur_masks = cur_masks.reshape(b, -1)
    stuff_equiv_classes = defaultdict(list)
    for (k, label) in enumerate(cur_classes):
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)
    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))
    if cur_classes.size() > 0:
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        while filtered_small.any():
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        cur_classes = np.ones((1, 1), dtype=np.int64)
    segments_info = [{'id': i, 'isthing': is_thing_map[cat], 'category_id': int(cat), 'area': a} for (i, (cat, a)) in enumerate(zip(cur_classes, area))]
    del cur_classes
    with io.BytesIO() as out:
        PIL.Image.fromarray(seg_img).save(out, format='PNG')
        predictions = {'png_string': out.getvalue(), 'segments_info': segments_info}
    return predictions

def resize_annotation(annotation: Dict[str, Any], orig_size: Tuple[int, int], target_size: Tuple[int, int], threshold: float=0.5, resample: PILImageResampling=PILImageResampling.NEAREST):
    if False:
        print('Hello World!')
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

def binary_mask_to_rle(mask):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.\n\n    Args:\n        mask (`torch.Tensor` or `numpy.array`):\n            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target\n            segment_id or class_id.\n    Returns:\n        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE\n        format.\n    '
    if is_torch_tensor(mask):
        mask = mask.numpy()
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)

def convert_segmentation_to_rle(segmentation):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.\n\n    Args:\n        segmentation (`torch.Tensor` or `numpy.array`):\n            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.\n    Returns:\n        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.\n    '
    segment_ids = torch.unique(segmentation)
    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)
    return run_length_encodings

def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    if False:
        return 10
    "\n    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and\n    `labels`.\n\n    Args:\n        masks (`torch.Tensor`):\n            A tensor of shape `(num_queries, height, width)`.\n        scores (`torch.Tensor`):\n            A tensor of shape `(num_queries)`.\n        labels (`torch.Tensor`):\n            A tensor of shape `(num_queries)`.\n        object_mask_threshold (`float`):\n            A number between 0 and 1 used to binarize the masks.\n    Raises:\n        `ValueError`: Raised when the first dimension doesn't match in all input tensors.\n    Returns:\n        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region\n        < `object_mask_threshold`.\n    "
    if not masks.shape[0] == scores.shape[0] == labels.shape[0]:
        raise ValueError('mask, scores and labels must have the same shape!')
    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)
    return (masks[to_keep], scores[to_keep], labels[to_keep])

def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    if False:
        print('Hello World!')
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False
    return (mask_exists, mask_k)

def compute_segments(mask_probs, pred_scores, pred_labels, mask_threshold: float=0.5, overlap_mask_area_threshold: float=0.8, label_ids_to_fuse: Optional[Set[int]]=None, target_size: Tuple[int, int]=None):
    if False:
        i = 10
        return i + 15
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]
    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []
    if target_size is not None:
        mask_probs = nn.functional.interpolate(mask_probs.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
    current_segment_id = 0
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse
        (mask_exists, mask_k) = check_segment_validity(mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold)
        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append({'id': current_segment_id, 'label_id': pred_class, 'was_fused': should_fuse, 'score': segment_score})
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id
    return (segmentation, segments)

class DetrImageProcessor(BaseImageProcessor):
    """
    Constructs a Detr image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's `(height, width)` dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
            in the `preprocess` method.
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
            print('Hello World!')
        if 'pad_and_return_pixel_mask' in kwargs:
            do_pad = kwargs.pop('pad_and_return_pixel_mask')
        if 'max_size' in kwargs:
            logger.warning_once("The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.")
            max_size = kwargs.pop('max_size')
        else:
            max_size = None if size is None else 1333
        size = size if size is not None else {'shortest_edge': 800, 'longest_edge': 1333}
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
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

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        if False:
            return 10
        '\n        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is\n        created using from_dict and kwargs e.g. `DetrImageProcessor.from_pretrained(checkpoint, size=600,\n        max_size=800)`\n        '
        image_processor_dict = image_processor_dict.copy()
        if 'max_size' in kwargs:
            image_processor_dict['max_size'] = kwargs.pop('max_size')
        if 'pad_and_return_pixel_mask' in kwargs:
            image_processor_dict['pad_and_return_pixel_mask'] = kwargs.pop('pad_and_return_pixel_mask')
        return super().from_dict(image_processor_dict, **kwargs)

    def prepare_annotation(self, image: np.ndarray, target: Dict, format: Optional[AnnotionFormat]=None, return_segmentation_masks: bool=None, masks_path: Optional[Union[str, pathlib.Path]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> Dict:
        if False:
            return 10
        '\n        Prepare an annotation for feeding into DETR model.\n        '
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
            for i in range(10):
                print('nop')
        logger.warning_once('The `prepare` method is deprecated and will be removed in a v4.33. Please use `prepare_annotation` instead. Note: the `prepare_annotation` method does not return the image anymore.')
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return (image, target)

    def convert_coco_poly_to_mask(self, *args, **kwargs):
        if False:
            print('Hello World!')
        logger.warning_once('The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ')
        return convert_coco_poly_to_mask(*args, **kwargs)

    def prepare_coco_detection(self, *args, **kwargs):
        if False:
            return 10
        logger.warning_once('The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ')
        return prepare_coco_detection_annotation(*args, **kwargs)

    def prepare_coco_panoptic(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        logger.warning_once('The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ')
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BILINEAR, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an\n        int, smaller edge of the image will be matched to this number.\n\n        Args:\n            image (`np.ndarray`):\n                Image to resize.\n            size (`Dict[str, int]`):\n                Dictionary containing the size to resize to. Can contain the keys `shortest_edge` and `longest_edge` or\n                `height` and `width`.\n            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):\n                Resampling filter to use if resizing the image.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        if 'max_size' in kwargs:
            logger.warning_once("The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.")
            max_size = kwargs.pop('max_size')
        else:
            max_size = None
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        if 'shortest_edge' in size and 'longest_edge' in size:
            size = get_resize_output_image_size(image, size['shortest_edge'], size['longest_edge'], input_data_format=input_data_format)
        elif 'height' in size and 'width' in size:
            size = (size['height'], size['width'])
        else:
            raise ValueError(f"Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got {size.keys()}.")
        image = resize(image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)
        return image

    def resize_annotation(self, annotation, orig_size, size, resample: PILImageResampling=PILImageResampling.NEAREST) -> Dict:
        if False:
            while True:
                i = 10
        '\n        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched\n        to this number.\n        '
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    def rescale(self, image: np.ndarray, rescale_factor: float, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Rescale the image by the given factor. image = image * rescale_factor.\n\n        Args:\n            image (`np.ndarray`):\n                Image to rescale.\n            rescale_factor (`float`):\n                The value to use for rescaling.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the input image. If unset, is inferred from the input image. Can be\n                one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n        '
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to\n        `[center_x, center_y, width, height]` format.\n        '
        return normalize_annotation(annotation, image_size=image_size)

    def _pad_image(self, image: np.ndarray, output_size: Tuple[int, int], constant_values: Union[float, Iterable[float]]=0, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        if False:
            return 10
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
            while True:
                i = 10
        '\n        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width\n        in the batch and optionally returns their corresponding pixel mask.\n\n        Args:\n            image (`np.ndarray`):\n                Image to pad.\n            constant_values (`float` or `Iterable[float]`, *optional*):\n                The value to use for the padding if `mode` is `"constant"`.\n            return_pixel_mask (`bool`, *optional*, defaults to `True`):\n                Whether to return a pixel mask.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the image. If not provided, it will be the same as the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        pad_size = get_max_height_width(images, input_data_format=input_data_format)
        padded_images = [self._pad_image(image, pad_size, constant_values=constant_values, data_format=data_format, input_data_format=input_data_format) for image in images]
        data = {'pixel_values': padded_images}
        if return_pixel_mask:
            masks = [make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format) for image in images]
            data['pixel_mask'] = masks
        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess(self, images: ImageInput, annotations: Optional[Union[AnnotationType, List[AnnotationType]]]=None, return_segmentation_masks: bool=None, masks_path: Optional[Union[str, pathlib.Path]]=None, do_resize: Optional[bool]=None, size: Optional[Dict[str, int]]=None, resample=None, do_rescale: Optional[bool]=None, rescale_factor: Optional[Union[int, float]]=None, do_normalize: Optional[bool]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, format: Optional[Union[str, AnnotionFormat]]=None, return_tensors: Optional[Union[TensorType, str]]=None, data_format: Union[str, ChannelDimension]=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> BatchFeature:
        if False:
            print('Hello World!')
        '\n        Preprocess an image or a batch of images so that it can be used by the model.\n\n        Args:\n            images (`ImageInput`):\n                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging\n                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.\n            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):\n                List of annotations associated with the image or batch of images. If annotation is for object\n                detection, the annotations should be a dictionary with the following keys:\n                - "image_id" (`int`): The image id.\n                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a\n                  dictionary. An image can have no annotations, in which case the list should be empty.\n                If annotation is for segmentation, the annotations should be a dictionary with the following keys:\n                - "image_id" (`int`): The image id.\n                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.\n                  An image can have no segments, in which case the list should be empty.\n                - "file_name" (`str`): The file name of the image.\n            return_segmentation_masks (`bool`, *optional*, defaults to self.return_segmentation_masks):\n                Whether to return segmentation masks.\n            masks_path (`str` or `pathlib.Path`, *optional*):\n                Path to the directory containing the segmentation masks.\n            do_resize (`bool`, *optional*, defaults to self.do_resize):\n                Whether to resize the image.\n            size (`Dict[str, int]`, *optional*, defaults to self.size):\n                Size of the image after resizing.\n            resample (`PILImageResampling`, *optional*, defaults to self.resample):\n                Resampling filter to use when resizing the image.\n            do_rescale (`bool`, *optional*, defaults to self.do_rescale):\n                Whether to rescale the image.\n            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):\n                Rescale factor to use when rescaling the image.\n            do_normalize (`bool`, *optional*, defaults to self.do_normalize):\n                Whether to normalize the image.\n            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):\n                Mean to use when normalizing the image.\n            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):\n                Standard deviation to use when normalizing the image.\n            do_pad (`bool`, *optional*, defaults to self.do_pad):\n                Whether to pad the image.\n            format (`str` or `AnnotionFormat`, *optional*, defaults to self.format):\n                Format of the annotations.\n            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):\n                Type of tensors to return. If `None`, will return the list of images.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        if 'pad_and_return_pixel_mask' in kwargs:
            logger.warning_once('The `pad_and_return_pixel_mask` argument is deprecated and will be removed in a future version, use `do_pad` instead.')
            do_pad = kwargs.pop('pad_and_return_pixel_mask')
        max_size = None
        if 'max_size' in kwargs:
            logger.warning_once("The `max_size` argument is deprecated and will be removed in a future version, use `size['longest_edge']` instead.")
            size = kwargs.pop('max_size')
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, max_size=max_size, default_to_square=False)
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
        images = make_list_of_images(images)
        if annotations is not None and isinstance(annotations, dict):
            annotations = [annotations]
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
                    resized_image = self.resize(image, size=size, max_size=max_size, resample=resample, input_data_format=input_data_format)
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

    def post_process(self, outputs, target_sizes):
        if False:
            i = 10
            return i + 15
        '\n        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,\n        bottom_right_x, bottom_right_y) format. Only supports PyTorch.\n\n        Args:\n            outputs ([`DetrObjectDetectionOutput`]):\n                Raw outputs of the model.\n            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):\n                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the\n                original image size (before any data augmentation). For visualization, this should be the image size\n                after data augment, but before padding.\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image\n            in the batch as predicted by the model.\n        '
        logger.warning_once('`post_process` is deprecated and will be removed in v5 of Transformers, please use `post_process_object_detection` instead, with `threshold=0.` for equivalent results.')
        (out_logits, out_bbox) = (outputs.logits, outputs.pred_boxes)
        if len(out_logits) != len(target_sizes):
            raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
        if target_sizes.shape[1] != 2:
            raise ValueError('Each element of target_sizes must contain the size (h, w) of each image of the batch')
        prob = nn.functional.softmax(out_logits, -1)
        (scores, labels) = prob[..., :-1].max(-1)
        boxes = center_to_corners_format(out_bbox)
        (img_h, img_w) = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for (s, l, b) in zip(scores, labels, boxes)]
        return results

    def post_process_segmentation(self, outputs, target_sizes, threshold=0.9, mask_threshold=0.5):
        if False:
            return 10
        '\n        Converts the output of [`DetrForSegmentation`] into image segmentation predictions. Only supports PyTorch.\n\n        Args:\n            outputs ([`DetrSegmentationOutput`]):\n                Raw outputs of the model.\n            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):\n                Torch Tensor (or list) corresponding to the requested final size (h, w) of each prediction.\n            threshold (`float`, *optional*, defaults to 0.9):\n                Threshold to use to filter out queries.\n            mask_threshold (`float`, *optional*, defaults to 0.5):\n                Threshold to use when turning the predicted masks into binary values.\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, and masks for an image\n            in the batch as predicted by the model.\n        '
        logger.warning_once('`post_process_segmentation` is deprecated and will be removed in v5 of Transformers, please use `post_process_semantic_segmentation`.')
        (out_logits, raw_masks) = (outputs.logits, outputs.pred_masks)
        empty_label = out_logits.shape[-1] - 1
        preds = []

        def to_tuple(tup):
            if False:
                return 10
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())
        for (cur_logits, cur_masks, size) in zip(out_logits, raw_masks, target_sizes):
            (cur_scores, cur_labels) = cur_logits.softmax(-1).max(-1)
            keep = cur_labels.ne(empty_label) & (cur_scores > threshold)
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]
            cur_masks = cur_masks[keep]
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode='bilinear').squeeze(1)
            cur_masks = (cur_masks.sigmoid() > mask_threshold) * 1
            predictions = {'scores': cur_scores, 'labels': cur_labels, 'masks': cur_masks}
            preds.append(predictions)
        return preds

    def post_process_instance(self, results, outputs, orig_target_sizes, max_target_sizes, threshold=0.5):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts the output of [`DetrForSegmentation`] into actual instance segmentation predictions. Only supports\n        PyTorch.\n\n        Args:\n            results (`List[Dict]`):\n                Results list obtained by [`~DetrImageProcessor.post_process`], to which "masks" results will be added.\n            outputs ([`DetrSegmentationOutput`]):\n                Raw outputs of the model.\n            orig_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):\n                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original\n                image size (before any data augmentation).\n            max_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):\n                Tensor containing the maximum size (h, w) of each image of the batch. For evaluation, this must be the\n                original image size (before any data augmentation).\n            threshold (`float`, *optional*, defaults to 0.5):\n                Threshold to use when turning the predicted masks into binary values.\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, boxes and masks for an\n            image in the batch as predicted by the model.\n        '
        logger.warning_once('`post_process_instance` is deprecated and will be removed in v5 of Transformers, please use `post_process_instance_segmentation`.')
        if len(orig_target_sizes) != len(max_target_sizes):
            raise ValueError('Make sure to pass in as many orig_target_sizes as max_target_sizes')
        (max_h, max_w) = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs.pred_masks.squeeze(2)
        outputs_masks = nn.functional.interpolate(outputs_masks, size=(max_h, max_w), mode='bilinear', align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > threshold).cpu()
        for (i, (cur_mask, t, tt)) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            (img_h, img_w) = (t[0], t[1])
            results[i]['masks'] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]['masks'] = nn.functional.interpolate(results[i]['masks'].float(), size=tuple(tt.tolist()), mode='nearest').byte()
        return results

    def post_process_panoptic(self, outputs, processed_sizes, target_sizes=None, is_thing_map=None, threshold=0.85):
        if False:
            while True:
                i = 10
        '\n        Converts the output of [`DetrForSegmentation`] into actual panoptic predictions. Only supports PyTorch.\n\n        Args:\n            outputs ([`DetrSegmentationOutput`]):\n                Raw outputs of the model.\n            processed_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):\n                Torch Tensor (or list) containing the size (h, w) of each image of the batch, i.e. the size after data\n                augmentation but before batching.\n            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`, *optional*):\n                Torch Tensor (or list) corresponding to the requested final size `(height, width)` of each prediction.\n                If left to None, it will default to the `processed_sizes`.\n            is_thing_map (`torch.Tensor` of shape `(batch_size, 2)`, *optional*):\n                Dictionary mapping class indices to either True or False, depending on whether or not they are a thing.\n                If not set, defaults to the `is_thing_map` of COCO panoptic.\n            threshold (`float`, *optional*, defaults to 0.85):\n                Threshold to use to filter out queries.\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing a PNG string and segments_info values for\n            an image in the batch as predicted by the model.\n        '
        logger.warning_once('`post_process_panoptic is deprecated and will be removed in v5 of Transformers, please use `post_process_panoptic_segmentation`.')
        if target_sizes is None:
            target_sizes = processed_sizes
        if len(processed_sizes) != len(target_sizes):
            raise ValueError('Make sure to pass in as many processed_sizes as target_sizes')
        if is_thing_map is None:
            is_thing_map = {i: i <= 90 for i in range(201)}
        (out_logits, raw_masks, raw_boxes) = (outputs.logits, outputs.pred_masks, outputs.pred_boxes)
        if not len(out_logits) == len(raw_masks) == len(target_sizes):
            raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits and masks')
        empty_label = out_logits.shape[-1] - 1
        preds = []

        def to_tuple(tup):
            if False:
                print('Hello World!')
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())
        for (cur_logits, cur_masks, cur_boxes, size, target_size) in zip(out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes):
            (cur_scores, cur_labels) = cur_logits.softmax(-1).max(-1)
            keep = cur_labels.ne(empty_label) & (cur_scores > threshold)
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]
            cur_masks = cur_masks[keep]
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode='bilinear').squeeze(1)
            cur_boxes = center_to_corners_format(cur_boxes[keep])
            (h, w) = cur_masks.shape[-2:]
            if len(cur_boxes) != len(cur_labels):
                raise ValueError('Not as many boxes as there are classes')
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda : [])
            for (k, label) in enumerate(cur_labels):
                if not is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                if False:
                    print('Hello World!')
                m_id = masks.transpose(0, 1).softmax(-1)
                if m_id.shape[-1] == 0:
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)
                if dedup:
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                (final_h, final_w) = to_tuple(target_size)
                seg_img = PIL.Image.fromarray(id_to_rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=PILImageResampling.NEAREST)
                np_seg_img = torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes()))
                np_seg_img = np_seg_img.view(final_h, final_w, 3)
                np_seg_img = np_seg_img.numpy()
                m_id = torch.from_numpy(rgb_to_id(np_seg_img))
                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return (area, seg_img)
            (area, seg_img) = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_labels.numel() > 0:
                while True:
                    filtered_small = torch.as_tensor([area[i] <= 4 for (i, c) in enumerate(cur_labels)], dtype=torch.bool, device=keep.device)
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_labels = cur_labels[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        (area, seg_img) = get_ids_area(cur_masks, cur_scores)
                    else:
                        break
            else:
                cur_labels = torch.ones(1, dtype=torch.long, device=cur_labels.device)
            segments_info = []
            for (i, a) in enumerate(area):
                cat = cur_labels[i].item()
                segments_info.append({'id': i, 'isthing': is_thing_map[cat], 'category_id': cat, 'area': a})
            del cur_labels
            with io.BytesIO() as out:
                seg_img.save(out, format='PNG')
                predictions = {'png_string': out.getvalue(), 'segments_info': segments_info}
            preds.append(predictions)
        return preds

    def post_process_object_detection(self, outputs, threshold: float=0.5, target_sizes: Union[TensorType, List[Tuple]]=None):
        if False:
            return 10
        '\n        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,\n        bottom_right_x, bottom_right_y) format. Only supports PyTorch.\n\n        Args:\n            outputs ([`DetrObjectDetectionOutput`]):\n                Raw outputs of the model.\n            threshold (`float`, *optional*):\n                Score threshold to keep object detection predictions.\n            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):\n                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size\n                `(height, width)` of each image in the batch. If unset, predictions will not be resized.\n        Returns:\n            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image\n            in the batch as predicted by the model.\n        '
        (out_logits, out_bbox) = (outputs.logits, outputs.pred_boxes)
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
        prob = nn.functional.softmax(out_logits, -1)
        (scores, labels) = prob[..., :-1].max(-1)
        boxes = center_to_corners_format(out_bbox)
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

    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple[int, int]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts the output of [`DetrForSegmentation`] into semantic segmentation maps. Only supports PyTorch.\n\n        Args:\n            outputs ([`DetrForSegmentation`]):\n                Raw outputs of the model.\n            target_sizes (`List[Tuple[int, int]]`, *optional*):\n                A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the\n                batch. If unset, predictions will not be resized.\n        Returns:\n            `List[torch.Tensor]`:\n                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)\n                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each\n                `torch.Tensor` correspond to a semantic class id.\n        '
        class_queries_logits = outputs.logits
        masks_queries_logits = outputs.pred_masks
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        segmentation = torch.einsum('bqc, bqhw -> bchw', masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = nn.functional.interpolate(segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode='bilinear', align_corners=False)
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]
        return semantic_segmentation

    def post_process_instance_segmentation(self, outputs, threshold: float=0.5, mask_threshold: float=0.5, overlap_mask_area_threshold: float=0.8, target_sizes: Optional[List[Tuple[int, int]]]=None, return_coco_annotation: Optional[bool]=False) -> List[Dict]:
        if False:
            i = 10
            return i + 15
        '\n        Converts the output of [`DetrForSegmentation`] into instance segmentation predictions. Only supports PyTorch.\n\n        Args:\n            outputs ([`DetrForSegmentation`]):\n                Raw outputs of the model.\n            threshold (`float`, *optional*, defaults to 0.5):\n                The probability score threshold to keep predicted instance masks.\n            mask_threshold (`float`, *optional*, defaults to 0.5):\n                Threshold to use when turning the predicted masks into binary values.\n            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):\n                The overlap mask area threshold to merge or discard small disconnected parts within each binary\n                instance mask.\n            target_sizes (`List[Tuple]`, *optional*):\n                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested\n                final size (height, width) of each prediction. If unset, predictions will not be resized.\n            return_coco_annotation (`bool`, *optional*):\n                Defaults to `False`. If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE)\n                format.\n        Returns:\n            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:\n            - **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id` or\n              `List[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to\n              `True`. Set to `None` if no mask if found above `threshold`.\n            - **segments_info** -- A dictionary that contains additional information on each segment.\n                - **id** -- An integer representing the `segment_id`.\n                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.\n                - **score** -- Prediction score of segment with `segment_id`.\n        '
        class_queries_logits = outputs.logits
        masks_queries_logits = outputs.pred_masks
        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1
        mask_probs = masks_queries_logits.sigmoid()
        (pred_scores, pred_labels) = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)
        results: List[Dict[str, TensorType]] = []
        for i in range(batch_size):
            (mask_probs_item, pred_scores_item, pred_labels_item) = remove_low_and_no_objects(mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels)
            if mask_probs_item.shape[0] <= 0:
                (height, width) = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({'segmentation': segmentation, 'segments_info': []})
                continue
            target_size = target_sizes[i] if target_sizes is not None else None
            (segmentation, segments) = compute_segments(mask_probs=mask_probs_item, pred_scores=pred_scores_item, pred_labels=pred_labels_item, mask_threshold=mask_threshold, overlap_mask_area_threshold=overlap_mask_area_threshold, label_ids_to_fuse=[], target_size=target_size)
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)
            results.append({'segmentation': segmentation, 'segments_info': segments})
        return results

    def post_process_panoptic_segmentation(self, outputs, threshold: float=0.5, mask_threshold: float=0.5, overlap_mask_area_threshold: float=0.8, label_ids_to_fuse: Optional[Set[int]]=None, target_sizes: Optional[List[Tuple[int, int]]]=None) -> List[Dict]:
        if False:
            while True:
                i = 10
        '\n        Converts the output of [`DetrForSegmentation`] into image panoptic segmentation predictions. Only supports\n        PyTorch.\n\n        Args:\n            outputs ([`DetrForSegmentation`]):\n                The outputs from [`DetrForSegmentation`].\n            threshold (`float`, *optional*, defaults to 0.5):\n                The probability score threshold to keep predicted instance masks.\n            mask_threshold (`float`, *optional*, defaults to 0.5):\n                Threshold to use when turning the predicted masks into binary values.\n            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):\n                The overlap mask area threshold to merge or discard small disconnected parts within each binary\n                instance mask.\n            label_ids_to_fuse (`Set[int]`, *optional*):\n                The labels in this state will have all their instances be fused together. For instance we could say\n                there can only be one sky in an image, but several persons, so the label ID for sky would be in that\n                set, but not the one for person.\n            target_sizes (`List[Tuple]`, *optional*):\n                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested\n                final size (height, width) of each prediction in batch. If unset, predictions will not be resized.\n        Returns:\n            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:\n            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id` or\n              `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized to\n              the corresponding `target_sizes` entry.\n            - **segments_info** -- A dictionary that contains additional information on each segment.\n                - **id** -- an integer representing the `segment_id`.\n                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.\n                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.\n                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.\n                - **score** -- Prediction score of segment with `segment_id`.\n        '
        if label_ids_to_fuse is None:
            logger.warning_once('`label_ids_to_fuse` unset. No instance will be fused.')
            label_ids_to_fuse = set()
        class_queries_logits = outputs.logits
        masks_queries_logits = outputs.pred_masks
        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1
        mask_probs = masks_queries_logits.sigmoid()
        (pred_scores, pred_labels) = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)
        results: List[Dict[str, TensorType]] = []
        for i in range(batch_size):
            (mask_probs_item, pred_scores_item, pred_labels_item) = remove_low_and_no_objects(mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels)
            if mask_probs_item.shape[0] <= 0:
                (height, width) = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({'segmentation': segmentation, 'segments_info': []})
                continue
            target_size = target_sizes[i] if target_sizes is not None else None
            (segmentation, segments) = compute_segments(mask_probs=mask_probs_item, pred_scores=pred_scores_item, pred_labels=pred_labels_item, mask_threshold=mask_threshold, overlap_mask_area_threshold=overlap_mask_area_threshold, label_ids_to_fuse=label_ids_to_fuse, target_size=target_size)
            results.append({'segmentation': segmentation, 'segments_info': segments})
        return results