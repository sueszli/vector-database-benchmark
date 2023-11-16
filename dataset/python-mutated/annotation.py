import copy
from typing import Dict, List, Optional
import numpy as np
from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_bool_mask_from_coco_segmentation, get_coco_segmentation_from_bool_mask
from sahi.utils.shapely import ShapelyAnnotation
try:
    from pycocotools import mask as mask_utils
    use_rle = True
except ImportError:
    use_rle = False

class BoundingBox:
    """
    Bounding box of the annotation.
    """

    def __init__(self, box: List[float], shift_amount: List[int]=[0, 0]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            box: List[float]\n                [minx, miny, maxx, maxy]\n            shift_amount: List[int]\n                To shift the box and mask predictions from sliced image\n                to full sized image, should be in the form of [shift_x, shift_y]\n        '
        if box[0] < 0 or box[1] < 0 or box[2] < 0 or (box[3] < 0):
            raise Exception('Box coords [minx, miny, maxx, maxy] cannot be negative')
        self.minx = box[0]
        self.miny = box[1]
        self.maxx = box[2]
        self.maxy = box[3]
        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]

    @property
    def shift_amount(self):
        if False:
            while True:
                i = 10
        '\n        Returns the shift amount of the bbox slice as [shift_x, shift_y]\n        '
        return [self.shift_x, self.shift_y]

    @property
    def area(self):
        if False:
            i = 10
            return i + 15
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def get_expanded_box(self, ratio=0.1, max_x=None, max_y=None):
        if False:
            print('Hello World!')
        w = self.maxx - self.minx
        h = self.maxy - self.miny
        y_mar = int(w * ratio)
        x_mar = int(h * ratio)
        maxx = min(max_x, self.maxx + x_mar) if max_x else self.maxx + x_mar
        minx = max(0, self.minx - x_mar)
        maxy = min(max_y, self.maxy + y_mar) if max_y else self.maxy + y_mar
        miny = max(0, self.miny - y_mar)
        box = [minx, miny, maxx, maxy]
        return BoundingBox(box)

    def to_xywh(self):
        if False:
            return 10
        '\n        Returns: [xmin, ymin, width, height]\n        '
        return [self.minx, self.miny, self.maxx - self.minx, self.maxy - self.miny]

    def to_coco_bbox(self):
        if False:
            while True:
                i = 10
        '\n        Returns: [xmin, ymin, width, height]\n        '
        return self.to_xywh()

    def to_xyxy(self):
        if False:
            return 10
        '\n        Returns: [xmin, ymin, xmax, ymax]\n        '
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_voc_bbox(self):
        if False:
            print('Hello World!')
        '\n        Returns: [xmin, ymin, xmax, ymax]\n        '
        return self.to_xyxy()

    def get_shifted_box(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns: shifted BoundingBox\n        '
        box = [self.minx + self.shift_x, self.miny + self.shift_y, self.maxx + self.shift_x, self.maxy + self.shift_y]
        return BoundingBox(box)

    def __repr__(self):
        if False:
            return 10
        return f'BoundingBox: <{(self.minx, self.miny, self.maxx, self.maxy)}, w: {self.maxx - self.minx}, h: {self.maxy - self.miny}>'

class Category:
    """
    Category of the annotation.
    """

    def __init__(self, id=None, name=None):
        if False:
            while True:
                i = 10
        '\n        Args:\n            id: int\n                ID of the object category\n            name: str\n                Name of the object category\n        '
        if not isinstance(id, int):
            raise TypeError('id should be integer')
        if not isinstance(name, str):
            raise TypeError('name should be string')
        self.id = id
        self.name = name

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Category: <id: {self.id}, name: {self.name}>'

class Mask:

    @classmethod
    def from_float_mask(cls, mask, full_shape=None, mask_threshold: float=0.5, shift_amount: list=[0, 0]):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            mask: np.ndarray of np.float elements\n                Mask values between 0 and 1 (should have a shape of height*width)\n            mask_threshold: float\n                Value to threshold mask pixels between 0 and 1\n            shift_amount: List\n                To shift the box and mask predictions from sliced image\n                to full sized image, should be in the form of [shift_x, shift_y]\n            full_shape: List\n                Size of the full image after shifting, should be in the form of [height, width]\n        '
        bool_mask = mask > mask_threshold
        return cls(bool_mask=bool_mask, shift_amount=shift_amount, full_shape=full_shape)

    @classmethod
    def from_coco_segmentation(cls, segmentation, full_shape=None, shift_amount: list=[0, 0]):
        if False:
            print('Hello World!')
        '\n        Init Mask from coco segmentation representation.\n\n        Args:\n            segmentation : List[List]\n                [\n                    [x1, y1, x2, y2, x3, y3, ...],\n                    [x1, y1, x2, y2, x3, y3, ...],\n                    ...\n                ]\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n        '
        if full_shape is None:
            raise ValueError('full_shape must be provided')
        bool_mask = get_bool_mask_from_coco_segmentation(segmentation, height=full_shape[0], width=full_shape[1])
        return cls(bool_mask=bool_mask, shift_amount=shift_amount, full_shape=full_shape)

    def __init__(self, bool_mask=None, full_shape=None, shift_amount: list=[0, 0]):
        if False:
            print('Hello World!')
        '\n        Args:\n            bool_mask: np.ndarray with bool elements\n                2D mask of object, should have a shape of height*width\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n        '
        if len(bool_mask) > 0:
            has_bool_mask = True
        else:
            has_bool_mask = False
        if has_bool_mask:
            self._mask = self.encode_bool_mask(bool_mask)
        else:
            self._mask = None
        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]
        if full_shape:
            self.full_shape_height = full_shape[0]
            self.full_shape_width = full_shape[1]
        elif has_bool_mask:
            self.full_shape_height = self.bool_mask.shape[0]
            self.full_shape_width = self.bool_mask.shape[1]
        else:
            self.full_shape_height = None
            self.full_shape_width = None

    def encode_bool_mask(self, bool_mask):
        if False:
            return 10
        _mask = bool_mask
        if use_rle:
            _mask = mask_utils.encode(np.asfortranarray(bool_mask.astype(np.uint8)))
        return _mask

    def decode_bool_mask(self, bool_mask):
        if False:
            while True:
                i = 10
        _mask = bool_mask
        if use_rle:
            _mask = mask_utils.decode(bool_mask).astype(bool)
        return _mask

    @property
    def bool_mask(self):
        if False:
            for i in range(10):
                print('nop')
        return self.decode_bool_mask(self._mask)

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns mask shape as [height, width]\n        '
        return [self.bool_mask.shape[0], self.bool_mask.shape[1]]

    @property
    def full_shape(self):
        if False:
            while True:
                i = 10
        '\n        Returns full mask shape after shifting as [height, width]\n        '
        return [self.full_shape_height, self.full_shape_width]

    @property
    def shift_amount(self):
        if False:
            while True:
                i = 10
        '\n        Returns the shift amount of the mask slice as [shift_x, shift_y]\n        '
        return [self.shift_x, self.shift_y]

    def get_shifted_mask(self):
        if False:
            i = 10
            return i + 15
        if self.full_shape_height is None or self.full_shape_width is None:
            raise ValueError('full_shape is None')
        mask_fullsized = np.full((self.full_shape_height, self.full_shape_width), 0, dtype='float32')
        starting_pixel = [self.shift_x, self.shift_y]
        ending_pixel = [min(starting_pixel[0] + self.bool_mask.shape[1], self.full_shape_width), min(starting_pixel[1] + self.bool_mask.shape[0], self.full_shape_height)]
        mask_fullsized[starting_pixel[1]:ending_pixel[1], starting_pixel[0]:ending_pixel[0]] = self.bool_mask[:ending_pixel[1] - starting_pixel[1], :ending_pixel[0] - starting_pixel[0]]
        return Mask(mask_fullsized, shift_amount=[0, 0], full_shape=self.full_shape)

    def to_coco_segmentation(self):
        if False:
            return 10
        '\n        Returns boolean mask as coco segmentation:\n        [\n            [x1, y1, x2, y2, x3, y3, ...],\n            [x1, y1, x2, y2, x3, y3, ...],\n            ...\n        ]\n        '
        coco_segmentation = get_coco_segmentation_from_bool_mask(self.bool_mask)
        return coco_segmentation

class ObjectAnnotation:
    """
    All about an annotation such as Mask, Category, BoundingBox.
    """

    @classmethod
    def from_bool_mask(cls, bool_mask, category_id: Optional[int]=None, category_name: Optional[str]=None, shift_amount: Optional[List[int]]=[0, 0], full_shape: Optional[List[int]]=None):
        if False:
            return 10
        '\n        Creates ObjectAnnotation from bool_mask (2D np.ndarray)\n\n        Args:\n            bool_mask: np.ndarray with bool elements\n                2D mask of object, should have a shape of height*width\n            category_id: int\n                ID of the object category\n            category_name: str\n                Name of the object category\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n        '
        return cls(category_id=category_id, bool_mask=bool_mask, category_name=category_name, shift_amount=shift_amount, full_shape=full_shape)

    @classmethod
    def from_coco_segmentation(cls, segmentation, full_shape: List[int], category_id: Optional[int]=None, category_name: Optional[str]=None, shift_amount: Optional[List[int]]=[0, 0]):
        if False:
            print('Hello World!')
        '\n        Creates ObjectAnnotation from coco segmentation:\n        [\n            [x1, y1, x2, y2, x3, y3, ...],\n            [x1, y1, x2, y2, x3, y3, ...],\n            ...\n        ]\n\n        Args:\n            segmentation: List[List]\n                [\n                    [x1, y1, x2, y2, x3, y3, ...],\n                    [x1, y1, x2, y2, x3, y3, ...],\n                    ...\n                ]\n            category_id: int\n                ID of the object category\n            category_name: str\n                Name of the object category\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n        '
        bool_mask = get_bool_mask_from_coco_segmentation(segmentation, width=full_shape[1], height=full_shape[0])
        return cls(category_id=category_id, bool_mask=bool_mask, category_name=category_name, shift_amount=shift_amount, full_shape=full_shape)

    @classmethod
    def from_coco_bbox(cls, bbox: List[int], category_id: Optional[int]=None, category_name: Optional[str]=None, shift_amount: Optional[List[int]]=[0, 0], full_shape: Optional[List[int]]=None):
        if False:
            i = 10
            return i + 15
        '\n        Creates ObjectAnnotation from coco bbox [minx, miny, width, height]\n\n        Args:\n            bbox: List\n                [minx, miny, width, height]\n            category_id: int\n                ID of the object category\n            category_name: str\n                Name of the object category\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n        '
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        return cls(category_id=category_id, bbox=bbox, category_name=category_name, shift_amount=shift_amount, full_shape=full_shape)

    @classmethod
    def from_coco_annotation_dict(cls, annotation_dict: Dict, full_shape: List[int], category_name: str=None, shift_amount: Optional[List[int]]=[0, 0]):
        if False:
            i = 10
            return i + 15
        '\n        Creates ObjectAnnotation object from category name and COCO formatted\n        annotation dict (with fields "bbox", "segmentation", "category_id").\n\n        Args:\n            annotation_dict: dict\n                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")\n            category_name: str\n                Category name of the annotation\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n        '
        if annotation_dict['segmentation']:
            return cls.from_coco_segmentation(segmentation=annotation_dict['segmentation'], category_id=annotation_dict['category_id'], category_name=category_name, shift_amount=shift_amount, full_shape=full_shape)
        else:
            return cls.from_coco_bbox(bbox=annotation_dict['bbox'], category_id=annotation_dict['category_id'], category_name=category_name, shift_amount=shift_amount, full_shape=full_shape)

    @classmethod
    def from_shapely_annotation(cls, annotation, full_shape: List[int], category_id: Optional[int]=None, category_name: Optional[str]=None, shift_amount: Optional[List[int]]=[0, 0]):
        if False:
            return 10
        '\n        Creates ObjectAnnotation from shapely_utils.ShapelyAnnotation\n\n        Args:\n            annotation: shapely_utils.ShapelyAnnotation\n            category_id: int\n                ID of the object category\n            category_name: str\n                Name of the object category\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n        '
        bool_mask = get_bool_mask_from_coco_segmentation(annotation.to_coco_segmentation(), width=full_shape[1], height=full_shape[0])
        return cls(category_id=category_id, bool_mask=bool_mask, category_name=category_name, shift_amount=shift_amount, full_shape=full_shape)

    @classmethod
    def from_imantics_annotation(cls, annotation, shift_amount: Optional[List[int]]=[0, 0], full_shape: Optional[List[int]]=None):
        if False:
            return 10
        '\n        Creates ObjectAnnotation from imantics.annotation.Annotation\n\n        Args:\n            annotation: imantics.annotation.Annotation\n            shift_amount: List\n                To shift the box and mask predictions from sliced image to full\n                sized image, should be in the form of [shift_x, shift_y]\n            full_shape: List\n                Size of the full image, should be in the form of [height, width]\n        '
        return cls(category_id=annotation.category.id, bool_mask=annotation.mask.array, category_name=annotation.category.name, shift_amount=shift_amount, full_shape=full_shape)

    def __init__(self, bbox: Optional[List[int]]=None, bool_mask: Optional[np.ndarray]=None, category_id: Optional[int]=None, category_name: Optional[str]=None, shift_amount: Optional[List[int]]=[0, 0], full_shape: Optional[List[int]]=None):
        if False:
            while True:
                i = 10
        '\n        Args:\n            bbox: List\n                [minx, miny, maxx, maxy]\n            bool_mask: np.ndarray with bool elements\n                2D mask of object, should have a shape of height*width\n            category_id: int\n                ID of the object category\n            category_name: str\n                Name of the object category\n            shift_amount: List\n                To shift the box and mask predictions from sliced image\n                to full sized image, should be in the form of [shift_x, shift_y]\n            full_shape: List\n                Size of the full image after shifting, should be in\n                the form of [height, width]\n        '
        if not isinstance(category_id, int):
            raise ValueError('category_id must be an integer')
        if bbox is None and bool_mask is None:
            raise ValueError('you must provide a bbox or bool_mask')
        if bool_mask is not None:
            self.mask = Mask(bool_mask=bool_mask, shift_amount=shift_amount, full_shape=full_shape)
            bbox_from_bool_mask = get_bbox_from_bool_mask(bool_mask)
            if bbox_from_bool_mask is not None:
                bbox = bbox_from_bool_mask
            else:
                raise ValueError('Invalid boolean mask.')
        else:
            self.mask = None
        if type(bbox).__module__ == 'numpy':
            bbox = copy.deepcopy(bbox).tolist()
        xmin = max(bbox[0], 0)
        ymin = max(bbox[1], 0)
        if full_shape:
            xmax = min(bbox[2], full_shape[1])
            ymax = min(bbox[3], full_shape[0])
        else:
            xmax = bbox[2]
            ymax = bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        self.bbox = BoundingBox(bbox, shift_amount)
        category_name = category_name if category_name else str(category_id)
        self.category = Category(id=category_id, name=category_name)
        self.merged = None

    def to_coco_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns sahi.utils.coco.CocoAnnotation representation of ObjectAnnotation.\n        '
        if self.mask:
            coco_annotation = CocoAnnotation.from_coco_segmentation(segmentation=self.mask.to_coco_segmentation(), category_id=self.category.id, category_name=self.category.name)
        else:
            coco_annotation = CocoAnnotation.from_coco_bbox(bbox=self.bbox.to_xywh(), category_id=self.category.id, category_name=self.category.name)
        return coco_annotation

    def to_coco_prediction(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns sahi.utils.coco.CocoPrediction representation of ObjectAnnotation.\n        '
        if self.mask:
            coco_prediction = CocoPrediction.from_coco_segmentation(segmentation=self.mask.to_coco_segmentation(), category_id=self.category.id, category_name=self.category.name, score=1)
        else:
            coco_prediction = CocoPrediction.from_coco_bbox(bbox=self.bbox.to_xywh(), category_id=self.category.id, category_name=self.category.name, score=1)
        return coco_prediction

    def to_shapely_annotation(self):
        if False:
            return 10
        '\n        Returns sahi.utils.shapely.ShapelyAnnotation representation of ObjectAnnotation.\n        '
        if self.mask:
            shapely_annotation = ShapelyAnnotation.from_coco_segmentation(segmentation=self.mask.to_coco_segmentation())
        else:
            shapely_annotation = ShapelyAnnotation.from_coco_bbox(bbox=self.bbox.to_xywh())
        return shapely_annotation

    def to_imantics_annotation(self):
        if False:
            while True:
                i = 10
        '\n        Returns imantics.annotation.Annotation representation of ObjectAnnotation.\n        '
        try:
            import imantics
        except ImportError:
            raise ImportError('Please run "pip install -U imantics" to install imantics first for imantics conversion.')
        imantics_category = imantics.Category(id=self.category.id, name=self.category.name)
        if self.mask is not None:
            imantics_mask = imantics.Mask.create(self.mask.bool_mask)
            imantics_annotation = imantics.annotation.Annotation.from_mask(mask=imantics_mask, category=imantics_category)
        else:
            imantics_bbox = imantics.BBox.create(self.bbox.to_xyxy())
            imantics_annotation = imantics.annotation.Annotation.from_bbox(bbox=imantics_bbox, category=imantics_category)
        return imantics_annotation

    def deepcopy(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns: deepcopy of current ObjectAnnotation instance\n        '
        return copy.deepcopy(self)

    @classmethod
    def get_empty_mask(cls):
        if False:
            return 10
        return Mask(bool_mask=None)

    def get_shifted_object_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        if self.mask:
            return ObjectAnnotation(bbox=self.bbox.get_shifted_box().to_xyxy(), category_id=self.category.id, bool_mask=self.mask.get_shifted_mask().bool_mask, category_name=self.category.name, shift_amount=[0, 0], full_shape=self.mask.get_shifted_mask().full_shape)
        else:
            return ObjectAnnotation(bbox=self.bbox.get_shifted_box().to_xyxy(), category_id=self.category.id, bool_mask=None, category_name=self.category.name, shift_amount=[0, 0], full_shape=None)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'ObjectAnnotation<\n    bbox: {self.bbox},\n    mask: {self.mask},\n    category: {self.category}>'