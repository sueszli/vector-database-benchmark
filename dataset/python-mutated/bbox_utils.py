from __future__ import division
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, cast
import numpy as np
from .transforms_interface import BoxInternalType, BoxType
from .utils import DataProcessor, Params
__all__ = ['normalize_bbox', 'denormalize_bbox', 'normalize_bboxes', 'denormalize_bboxes', 'calculate_bbox_area', 'filter_bboxes_by_visibility', 'convert_bbox_to_albumentations', 'convert_bbox_from_albumentations', 'convert_bboxes_to_albumentations', 'convert_bboxes_from_albumentations', 'check_bbox', 'check_bboxes', 'filter_bboxes', 'union_of_bboxes', 'BboxProcessor', 'BboxParams']
TBox = TypeVar('TBox', BoxType, BoxInternalType)

class BboxParams(Params):
    """
    Parameters of bounding boxes

    Args:
        format (str): format of bounding boxes. Should be 'coco', 'pascal_voc', 'albumentations' or 'yolo'.

            The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
            The `pascal_voc` format
                `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
            The `albumentations` format
                is like `pascal_voc`, but normalized,
                in other words: `[x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
            The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        label_fields (list): list of fields that are joined with boxes, e.g labels.
            Should be same type as boxes.
        min_area (float): minimum area of a bounding box. All bounding boxes whose
            visible area in pixels is less than this value will be removed. Default: 0.0.
        min_visibility (float): minimum fraction of area for a bounding box
            to remain this box in list. Default: 0.0.
        min_width (float): Minimum width of a bounding box. All bounding boxes whose width is
            less than this value will be removed. Default: 0.0.
        min_height (float): Minimum height of a bounding box. All bounding boxes whose height is
            less than this value will be removed. Default: 0.0.
        check_each_transform (bool): if `True`, then bboxes will be checked after each dual transform.
            Default: `True`
    """

    def __init__(self, format: str, label_fields: Optional[Sequence[str]]=None, min_area: float=0.0, min_visibility: float=0.0, min_width: float=0.0, min_height: float=0.0, check_each_transform: bool=True):
        if False:
            while True:
                i = 10
        super(BboxParams, self).__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.check_each_transform = check_each_transform

    def _to_dict(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        data = super(BboxParams, self)._to_dict()
        data.update({'min_area': self.min_area, 'min_visibility': self.min_visibility, 'min_width': self.min_width, 'min_height': self.min_height, 'check_each_transform': self.check_each_transform})
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        if False:
            while True:
                i = 10
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        if False:
            i = 10
            return i + 15
        return 'BboxParams'

class BboxProcessor(DataProcessor):

    def __init__(self, params: BboxParams, additional_targets: Optional[Dict[str, str]]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'bboxes'

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        for data_name in self.data_fields:
            data_exists = data_name in data and len(data[data_name])
            if data_exists and len(data[data_name][0]) < 5:
                if self.params.label_fields is None:
                    raise ValueError("Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox because bboxes must have labels")
        if self.params.label_fields:
            if not all((i in data.keys() for i in self.params.label_fields)):
                raise ValueError("Your 'label_fields' are not valid - them must have same names as params in dict")

    def filter(self, data: Sequence, rows: int, cols: int) -> List:
        if False:
            i = 10
            return i + 15
        self.params: BboxParams
        return filter_bboxes(data, rows, cols, min_area=self.params.min_area, min_visibility=self.params.min_visibility, min_width=self.params.min_width, min_height=self.params.min_height)

    def check(self, data: Sequence, rows: int, cols: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        check_bboxes(data)

    def convert_from_albumentations(self, data: Sequence, rows: int, cols: int) -> List[BoxType]:
        if False:
            print('Hello World!')
        return convert_bboxes_from_albumentations(data, self.params.format, rows, cols, check_validity=True)

    def convert_to_albumentations(self, data: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
        if False:
            while True:
                i = 10
        return convert_bboxes_to_albumentations(data, self.params.format, rows, cols, check_validity=True)

def normalize_bbox(bbox: TBox, rows: int, cols: int) -> TBox:
    if False:
        print('Hello World!')
    'Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates\n    by image height.\n\n    Args:\n        bbox: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.\n        rows: Image height.\n        cols: Image width.\n\n    Returns:\n        Normalized bounding box `(x_min, y_min, x_max, y_max)`.\n\n    Raises:\n        ValueError: If rows or cols is less or equal zero\n\n    '
    if rows <= 0:
        raise ValueError('Argument rows must be positive integer')
    if cols <= 0:
        raise ValueError('Argument cols must be positive integer')
    tail: Tuple[Any, ...]
    ((x_min, y_min, x_max, y_max), tail) = (bbox[:4], tuple(bbox[4:]))
    (x_min, x_max) = (x_min / cols, x_max / cols)
    (y_min, y_max) = (y_min / rows, y_max / rows)
    return cast(BoxType, (x_min, y_min, x_max, y_max) + tail)

def denormalize_bbox(bbox: TBox, rows: int, cols: int) -> TBox:
    if False:
        return 10
    'Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates\n    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.\n\n    Args:\n        bbox: Normalized bounding box `(x_min, y_min, x_max, y_max)`.\n        rows: Image height.\n        cols: Image width.\n\n    Returns:\n        Denormalized bounding box `(x_min, y_min, x_max, y_max)`.\n\n    Raises:\n        ValueError: If rows or cols is less or equal zero\n\n    '
    tail: Tuple[Any, ...]
    ((x_min, y_min, x_max, y_max), tail) = (bbox[:4], tuple(bbox[4:]))
    if rows <= 0:
        raise ValueError('Argument rows must be positive integer')
    if cols <= 0:
        raise ValueError('Argument cols must be positive integer')
    (x_min, x_max) = (x_min * cols, x_max * cols)
    (y_min, y_max) = (y_min * rows, y_max * rows)
    return cast(BoxType, (x_min, y_min, x_max, y_max) + tail)

def normalize_bboxes(bboxes: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
    if False:
        while True:
            i = 10
    'Normalize a list of bounding boxes.\n\n    Args:\n        bboxes: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.\n        rows: Image height.\n        cols: Image width.\n\n    Returns:\n        Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.\n\n    '
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]

def denormalize_bboxes(bboxes: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
    if False:
        while True:
            i = 10
    'Denormalize a list of bounding boxes.\n\n    Args:\n        bboxes: Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.\n        rows: Image height.\n        cols: Image width.\n\n    Returns:\n        List: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.\n\n    '
    return [denormalize_bbox(bbox, rows, cols) for bbox in bboxes]

def calculate_bbox_area(bbox: BoxType, rows: int, cols: int) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Calculate the area of a bounding box in (fractional) pixels.\n\n    Args:\n        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.\n        rows: Image height.\n        cols: Image width.\n\n    Return:\n        Area in (fractional) pixels of the (denormalized) bounding box.\n\n    '
    bbox = denormalize_bbox(bbox, rows, cols)
    (x_min, y_min, x_max, y_max) = bbox[:4]
    area = (x_max - x_min) * (y_max - y_min)
    return area

def filter_bboxes_by_visibility(original_shape: Sequence[int], bboxes: Sequence[BoxType], transformed_shape: Sequence[int], transformed_bboxes: Sequence[BoxType], threshold: float=0.0, min_area: float=0.0) -> List[BoxType]:
    if False:
        for i in range(10):
            print('nop')
    'Filter bounding boxes and return only those boxes whose visibility after transformation is above\n    the threshold and minimal area of bounding box in pixels is more then min_area.\n\n    Args:\n        original_shape: Original image shape `(height, width, ...)`.\n        bboxes: Original bounding boxes `[(x_min, y_min, x_max, y_max)]`.\n        transformed_shape: Transformed image shape `(height, width)`.\n        transformed_bboxes: Transformed bounding boxes `[(x_min, y_min, x_max, y_max)]`.\n        threshold: visibility threshold. Should be a value in the range [0.0, 1.0].\n        min_area: Minimal area threshold.\n\n    Returns:\n        Filtered bounding boxes `[(x_min, y_min, x_max, y_max)]`.\n\n    '
    (img_height, img_width) = original_shape[:2]
    (transformed_img_height, transformed_img_width) = transformed_shape[:2]
    visible_bboxes = []
    for (bbox, transformed_bbox) in zip(bboxes, transformed_bboxes):
        if not all((0.0 <= value <= 1.0 for value in transformed_bbox[:4])):
            continue
        bbox_area = calculate_bbox_area(bbox, img_height, img_width)
        transformed_bbox_area = calculate_bbox_area(transformed_bbox, transformed_img_height, transformed_img_width)
        if transformed_bbox_area < min_area:
            continue
        visibility = transformed_bbox_area / bbox_area
        if visibility >= threshold:
            visible_bboxes.append(transformed_bbox)
    return visible_bboxes

def convert_bbox_to_albumentations(bbox: BoxType, source_format: str, rows: int, cols: int, check_validity: bool=False) -> BoxType:
    if False:
        while True:
            i = 10
    "Convert a bounding box from a format specified in `source_format` to the format used by albumentations:\n    normalized coordinates of top-left and bottom-right corners of the bounding box in a form of\n    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.\n\n    Args:\n        bbox: A bounding box tuple.\n        source_format: format of the bounding box. Should be 'coco', 'pascal_voc', or 'yolo'.\n        check_validity: Check if all boxes are valid boxes.\n        rows: Image height.\n        cols: Image width.\n\n    Returns:\n        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.\n\n    Note:\n        The `coco` format of a bounding box looks like `(x_min, y_min, width, height)`, e.g. (97, 12, 150, 200).\n        The `pascal_voc` format of a bounding box looks like `(x_min, y_min, x_max, y_max)`, e.g. (97, 12, 247, 212).\n        The `yolo` format of a bounding box looks like `(x, y, width, height)`, e.g. (0.3, 0.1, 0.05, 0.07);\n        where `x`, `y` coordinates of the center of the box, all values normalized to 1 by image height and width.\n\n    Raises:\n        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`, or `yolo`.\n        ValueError: If in YOLO format all labels not in range (0, 1).\n\n    "
    if source_format not in {'coco', 'pascal_voc', 'yolo'}:
        raise ValueError(f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'")
    if source_format == 'coco':
        ((x_min, y_min, width, height), tail) = (bbox[:4], bbox[4:])
        x_max = x_min + width
        y_max = y_min + height
    elif source_format == 'yolo':
        _bbox = np.array(bbox[:4])
        if check_validity and np.any((_bbox <= 0) | (_bbox > 1)):
            raise ValueError('In YOLO format all coordinates must be float and in range (0, 1]')
        ((x, y, w, h), tail) = (bbox[:4], bbox[4:])
        (w_half, h_half) = (w / 2, h / 2)
        x_min = x - w_half
        y_min = y - h_half
        x_max = x_min + w
        y_max = y_min + h
    else:
        ((x_min, y_min, x_max, y_max), tail) = (bbox[:4], bbox[4:])
    bbox = (x_min, y_min, x_max, y_max) + tuple(tail)
    if source_format != 'yolo':
        bbox = normalize_bbox(bbox, rows, cols)
    if check_validity:
        check_bbox(bbox)
    return bbox

def convert_bbox_from_albumentations(bbox: BoxType, target_format: str, rows: int, cols: int, check_validity: bool=False) -> BoxType:
    if False:
        i = 10
        return i + 15
    "Convert a bounding box from the format used by albumentations to a format, specified in `target_format`.\n\n    Args:\n        bbox: An albumentations bounding box `(x_min, y_min, x_max, y_max)`.\n        target_format: required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.\n        rows: Image height.\n        cols: Image width.\n        check_validity: Check if all boxes are valid boxes.\n\n    Returns:\n        tuple: A bounding box.\n\n    Note:\n        The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].\n        The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].\n        The `yolo` format of a bounding box looks like `[x, y, width, height]`, e.g. [0.3, 0.1, 0.05, 0.07].\n\n    Raises:\n        ValueError: if `target_format` is not equal to `coco`, `pascal_voc` or `yolo`.\n\n    "
    if target_format not in {'coco', 'pascal_voc', 'yolo'}:
        raise ValueError(f"Unknown target_format {target_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'")
    if check_validity:
        check_bbox(bbox)
    if target_format != 'yolo':
        bbox = denormalize_bbox(bbox, rows, cols)
    if target_format == 'coco':
        ((x_min, y_min, x_max, y_max), tail) = (bbox[:4], tuple(bbox[4:]))
        width = x_max - x_min
        height = y_max - y_min
        bbox = cast(BoxType, (x_min, y_min, width, height) + tail)
    elif target_format == 'yolo':
        ((x_min, y_min, x_max, y_max), tail) = (bbox[:4], bbox[4:])
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        bbox = cast(BoxType, (x, y, w, h) + tail)
    return bbox

def convert_bboxes_to_albumentations(bboxes: Sequence[BoxType], source_format, rows, cols, check_validity=False) -> List[BoxType]:
    if False:
        for i in range(10):
            print('nop')
    'Convert a list bounding boxes from a format specified in `source_format` to the format used by albumentations'
    return [convert_bbox_to_albumentations(bbox, source_format, rows, cols, check_validity) for bbox in bboxes]

def convert_bboxes_from_albumentations(bboxes: Sequence[BoxType], target_format: str, rows: int, cols: int, check_validity: bool=False) -> List[BoxType]:
    if False:
        i = 10
        return i + 15
    "Convert a list of bounding boxes from the format used by albumentations to a format, specified\n    in `target_format`.\n\n    Args:\n        bboxes: List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.\n        target_format: required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.\n        rows: Image height.\n        cols: Image width.\n        check_validity: Check if all boxes are valid boxes.\n\n    Returns:\n        List of bounding boxes.\n\n    "
    return [convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity) for bbox in bboxes]

def check_bbox(bbox: BoxType) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums'
    for (name, value) in zip(['x_min', 'y_min', 'x_max', 'y_max'], bbox[:4]):
        if not 0 <= value <= 1 and (not np.isclose(value, 0)) and (not np.isclose(value, 1)):
            raise ValueError(f'Expected {name} for bbox {bbox} to be in the range [0.0, 1.0], got {value}.')
    (x_min, y_min, x_max, y_max) = bbox[:4]
    if x_max <= x_min:
        raise ValueError(f'x_max is less than or equal to x_min for bbox {bbox}.')
    if y_max <= y_min:
        raise ValueError(f'y_max is less than or equal to y_min for bbox {bbox}.')

def check_bboxes(bboxes: Sequence[BoxType]) -> None:
    if False:
        print('Hello World!')
    'Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums'
    for bbox in bboxes:
        check_bbox(bbox)

def filter_bboxes(bboxes: Sequence[BoxType], rows: int, cols: int, min_area: float=0.0, min_visibility: float=0.0, min_width: float=0.0, min_height: float=0.0) -> List[BoxType]:
    if False:
        while True:
            i = 10
    'Remove bounding boxes that either lie outside of the visible area by more then min_visibility\n    or whose area in pixels is under the threshold set by `min_area`. Also it crops boxes to final image size.\n\n    Args:\n        bboxes: List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.\n        rows: Image height.\n        cols: Image width.\n        min_area: Minimum area of a bounding box. All bounding boxes whose visible area in pixels.\n            is less than this value will be removed. Default: 0.0.\n        min_visibility: Minimum fraction of area for a bounding box to remain this box in list. Default: 0.0.\n        min_width: Minimum width of a bounding box. All bounding boxes whose width is\n            less than this value will be removed. Default: 0.0.\n        min_height: Minimum height of a bounding box. All bounding boxes whose height is\n            less than this value will be removed. Default: 0.0.\n\n    Returns:\n        List of bounding boxes.\n\n    '
    resulting_boxes: List[BoxType] = []
    for bbox in bboxes:
        transformed_box_area = calculate_bbox_area(bbox, rows, cols)
        (bbox, tail) = (cast(BoxType, tuple(np.clip(bbox[:4], 0, 1.0))), tuple(bbox[4:]))
        clipped_box_area = calculate_bbox_area(bbox, rows, cols)
        (x_min, y_min, x_max, y_max) = denormalize_bbox(bbox, rows, cols)[:4]
        (clipped_width, clipped_height) = (x_max - x_min, y_max - y_min)
        if clipped_box_area != 0 and clipped_box_area >= min_area and (clipped_box_area / transformed_box_area >= min_visibility) and (clipped_width >= min_width) and (clipped_height >= min_height):
            resulting_boxes.append(cast(BoxType, bbox + tail))
    return resulting_boxes

def union_of_bboxes(height: int, width: int, bboxes: Sequence[BoxType], erosion_rate: float=0.0) -> BoxType:
    if False:
        return 10
    'Calculate union of bounding boxes.\n\n    Args:\n        height (float): Height of image or space.\n        width (float): Width of image or space.\n        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.\n        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.\n            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.\n\n    Returns:\n        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.\n\n    '
    (x1, y1) = (width, height)
    (x2, y2) = (0, 0)
    for bbox in bboxes:
        (x_min, y_min, x_max, y_max) = bbox[:4]
        (w, h) = (x_max - x_min, y_max - y_min)
        (lim_x1, lim_y1) = (x_min + erosion_rate * w, y_min + erosion_rate * h)
        (lim_x2, lim_y2) = (x_max - erosion_rate * w, y_max - erosion_rate * h)
        (x1, y1) = (np.min([x1, lim_x1]), np.min([y1, lim_y1]))
        (x2, y2) = (np.max([x2, lim_x2]), np.max([y2, lim_y2]))
    return (x1, y1, x2, y2)