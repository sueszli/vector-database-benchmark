"""Wrappers for third party pycocotools to be used within object_detection.

Note that nothing in this file is tensorflow related and thus cannot
be called directly as a slim metric, for example.

TODO(jonathanhuang): wrap as a slim metric in metrics.py


Usage example: given a set of images with ids in the list image_ids
and corresponding lists of numpy arrays encoding groundtruth (boxes and classes)
and detections (boxes, scores and classes), where elements of each list
correspond to detections/annotations of a single image,
then evaluation (in multi-class mode) can be invoked as follows:

  groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
      image_ids, groundtruth_boxes_list, groundtruth_classes_list,
      max_num_classes, output_path=None)
  detections_list = coco_tools.ExportDetectionsToCOCO(
      image_ids, detection_boxes_list, detection_scores_list,
      detection_classes_list, output_path=None)
  groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
  detections = groundtruth.LoadAnnotations(detections_list)
  evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                         agnostic_mode=False)
  metrics = evaluator.ComputeMetrics()

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
import copy
import time
import numpy as np
from pycocotools import coco
from pycocotools import cocoeval
from pycocotools import mask
from six.moves import range
from six.moves import zip
import tensorflow as tf
from object_detection.utils import json_utils

class COCOWrapper(coco.COCO):
    """Wrapper for the pycocotools COCO class."""

    def __init__(self, dataset, detection_type='bbox'):
        if False:
            print('Hello World!')
        "COCOWrapper constructor.\n\n    See http://mscoco.org/dataset/#format for a description of the format.\n    By default, the coco.COCO class constructor reads from a JSON file.\n    This function duplicates the same behavior but loads from a dictionary,\n    allowing us to perform evaluation without writing to external storage.\n\n    Args:\n      dataset: a dictionary holding bounding box annotations in the COCO format.\n      detection_type: type of detections being wrapped. Can be one of ['bbox',\n        'segmentation']\n\n    Raises:\n      ValueError: if detection_type is unsupported.\n    "
        supported_detection_types = ['bbox', 'segmentation']
        if detection_type not in supported_detection_types:
            raise ValueError('Unsupported detection type: {}. Supported values are: {}'.format(detection_type, supported_detection_types))
        self._detection_type = detection_type
        coco.COCO.__init__(self)
        self.dataset = dataset
        self.createIndex()

    def LoadAnnotations(self, annotations):
        if False:
            while True:
                i = 10
        "Load annotations dictionary into COCO datastructure.\n\n    See http://mscoco.org/dataset/#format for a description of the annotations\n    format.  As above, this function replicates the default behavior of the API\n    but does not require writing to external storage.\n\n    Args:\n      annotations: python list holding object detection results where each\n        detection is encoded as a dict with required keys ['image_id',\n        'category_id', 'score'] and one of ['bbox', 'segmentation'] based on\n        `detection_type`.\n\n    Returns:\n      a coco.COCO datastructure holding object detection annotations results\n\n    Raises:\n      ValueError: if annotations is not a list\n      ValueError: if annotations do not correspond to the images contained\n        in self.\n    "
        results = coco.COCO()
        results.dataset['images'] = [img for img in self.dataset['images']]
        tf.logging.info('Loading and preparing annotation results...')
        tic = time.time()
        if not isinstance(annotations, list):
            raise ValueError('annotations is not a list of objects')
        annotation_img_ids = [ann['image_id'] for ann in annotations]
        if set(annotation_img_ids) != set(annotation_img_ids) & set(self.getImgIds()):
            raise ValueError('Results do not correspond to current coco set')
        results.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        if self._detection_type == 'bbox':
            for (idx, ann) in enumerate(annotations):
                bb = ann['bbox']
                ann['area'] = bb[2] * bb[3]
                ann['id'] = idx + 1
                ann['iscrowd'] = 0
        elif self._detection_type == 'segmentation':
            for (idx, ann) in enumerate(annotations):
                ann['area'] = mask.area(ann['segmentation'])
                ann['bbox'] = mask.toBbox(ann['segmentation'])
                ann['id'] = idx + 1
                ann['iscrowd'] = 0
        tf.logging.info('DONE (t=%0.2fs)', time.time() - tic)
        results.dataset['annotations'] = annotations
        results.createIndex()
        return results

class COCOEvalWrapper(cocoeval.COCOeval):
    """Wrapper for the pycocotools COCOeval class.

  To evaluate, create two objects (groundtruth_dict and detections_list)
  using the conventions listed at http://mscoco.org/dataset/#format.
  Then call evaluation as follows:

    groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    detections = groundtruth.LoadAnnotations(detections_list)
    evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                           agnostic_mode=False)

    metrics = evaluator.ComputeMetrics()
  """

    def __init__(self, groundtruth=None, detections=None, agnostic_mode=False, iou_type='bbox'):
        if False:
            return 10
        'COCOEvalWrapper constructor.\n\n    Note that for the area-based metrics to be meaningful, detection and\n    groundtruth boxes must be in image coordinates measured in pixels.\n\n    Args:\n      groundtruth: a coco.COCO (or coco_tools.COCOWrapper) object holding\n        groundtruth annotations\n      detections: a coco.COCO (or coco_tools.COCOWrapper) object holding\n        detections\n      agnostic_mode: boolean (default: False).  If True, evaluation ignores\n        class labels, treating all detections as proposals.\n      iou_type: IOU type to use for evaluation. Supports `bbox` or `segm`.\n    '
        cocoeval.COCOeval.__init__(self, groundtruth, detections, iouType=iou_type)
        if agnostic_mode:
            self.params.useCats = 0

    def GetCategory(self, category_id):
        if False:
            return 10
        "Fetches dictionary holding category information given category id.\n\n    Args:\n      category_id: integer id\n    Returns:\n      dictionary holding 'id', 'name'.\n    "
        return self.cocoGt.cats[category_id]

    def GetAgnosticMode(self):
        if False:
            while True:
                i = 10
        'Returns true if COCO Eval is configured to evaluate in agnostic mode.'
        return self.params.useCats == 0

    def GetCategoryIdList(self):
        if False:
            return 10
        'Returns list of valid category ids.'
        return self.params.catIds

    def ComputeMetrics(self, include_metrics_per_category=False, all_metrics_per_category=False):
        if False:
            while True:
                i = 10
        "Computes detection metrics.\n\n    Args:\n      include_metrics_per_category: If True, will include metrics per category.\n      all_metrics_per_category: If true, include all the summery metrics for\n        each category in per_category_ap. Be careful with setting it to true if\n        you have more than handful of categories, because it will pollute\n        your mldash.\n\n    Returns:\n      1. summary_metrics: a dictionary holding:\n        'Precision/mAP': mean average precision over classes averaged over IOU\n          thresholds ranging from .5 to .95 with .05 increments\n        'Precision/mAP@.50IOU': mean average precision at 50% IOU\n        'Precision/mAP@.75IOU': mean average precision at 75% IOU\n        'Precision/mAP (small)': mean average precision for small objects\n                        (area < 32^2 pixels)\n        'Precision/mAP (medium)': mean average precision for medium sized\n                        objects (32^2 pixels < area < 96^2 pixels)\n        'Precision/mAP (large)': mean average precision for large objects\n                        (96^2 pixels < area < 10000^2 pixels)\n        'Recall/AR@1': average recall with 1 detection\n        'Recall/AR@10': average recall with 10 detections\n        'Recall/AR@100': average recall with 100 detections\n        'Recall/AR@100 (small)': average recall for small objects with 100\n          detections\n        'Recall/AR@100 (medium)': average recall for medium objects with 100\n          detections\n        'Recall/AR@100 (large)': average recall for large objects with 100\n          detections\n      2. per_category_ap: a dictionary holding category specific results with\n        keys of the form: 'Precision mAP ByCategory/category'\n        (without the supercategory part if no supercategories exist).\n        For backward compatibility 'PerformanceByCategory' is included in the\n        output regardless of all_metrics_per_category.\n        If evaluating class-agnostic mode, per_category_ap is an empty\n        dictionary.\n\n    Raises:\n      ValueError: If category_stats does not exist.\n    "
        self.evaluate()
        self.accumulate()
        self.summarize()
        summary_metrics = OrderedDict([('Precision/mAP', self.stats[0]), ('Precision/mAP@.50IOU', self.stats[1]), ('Precision/mAP@.75IOU', self.stats[2]), ('Precision/mAP (small)', self.stats[3]), ('Precision/mAP (medium)', self.stats[4]), ('Precision/mAP (large)', self.stats[5]), ('Recall/AR@1', self.stats[6]), ('Recall/AR@10', self.stats[7]), ('Recall/AR@100', self.stats[8]), ('Recall/AR@100 (small)', self.stats[9]), ('Recall/AR@100 (medium)', self.stats[10]), ('Recall/AR@100 (large)', self.stats[11])])
        if not include_metrics_per_category:
            return (summary_metrics, {})
        if not hasattr(self, 'category_stats'):
            raise ValueError('Category stats do not exist')
        per_category_ap = OrderedDict([])
        if self.GetAgnosticMode():
            return (summary_metrics, per_category_ap)
        for (category_index, category_id) in enumerate(self.GetCategoryIdList()):
            category = self.GetCategory(category_id)['name']
            per_category_ap['PerformanceByCategory/mAP/{}'.format(category)] = self.category_stats[0][category_index]
            if all_metrics_per_category:
                per_category_ap['Precision mAP ByCategory/{}'.format(category)] = self.category_stats[0][category_index]
                per_category_ap['Precision mAP@.50IOU ByCategory/{}'.format(category)] = self.category_stats[1][category_index]
                per_category_ap['Precision mAP@.75IOU ByCategory/{}'.format(category)] = self.category_stats[2][category_index]
                per_category_ap['Precision mAP (small) ByCategory/{}'.format(category)] = self.category_stats[3][category_index]
                per_category_ap['Precision mAP (medium) ByCategory/{}'.format(category)] = self.category_stats[4][category_index]
                per_category_ap['Precision mAP (large) ByCategory/{}'.format(category)] = self.category_stats[5][category_index]
                per_category_ap['Recall AR@1 ByCategory/{}'.format(category)] = self.category_stats[6][category_index]
                per_category_ap['Recall AR@10 ByCategory/{}'.format(category)] = self.category_stats[7][category_index]
                per_category_ap['Recall AR@100 ByCategory/{}'.format(category)] = self.category_stats[8][category_index]
                per_category_ap['Recall AR@100 (small) ByCategory/{}'.format(category)] = self.category_stats[9][category_index]
                per_category_ap['Recall AR@100 (medium) ByCategory/{}'.format(category)] = self.category_stats[10][category_index]
                per_category_ap['Recall AR@100 (large) ByCategory/{}'.format(category)] = self.category_stats[11][category_index]
        return (summary_metrics, per_category_ap)

def _ConvertBoxToCOCOFormat(box):
    if False:
        while True:
            i = 10
    'Converts a box in [ymin, xmin, ymax, xmax] format to COCO format.\n\n  This is a utility function for converting from our internal\n  [ymin, xmin, ymax, xmax] convention to the convention used by the COCO API\n  i.e., [xmin, ymin, width, height].\n\n  Args:\n    box: a [ymin, xmin, ymax, xmax] numpy array\n\n  Returns:\n    a list of floats representing [xmin, ymin, width, height]\n  '
    return [float(box[1]), float(box[0]), float(box[3] - box[1]), float(box[2] - box[0])]

def _RleCompress(masks):
    if False:
        while True:
            i = 10
    'Compresses mask using Run-length encoding provided by pycocotools.\n\n  Args:\n    masks: uint8 numpy array of shape [mask_height, mask_width] with values in\n    {0, 1}.\n\n  Returns:\n    A pycocotools Run-length encoding of the mask.\n  '
    return mask.encode(np.asfortranarray(masks))

def ExportSingleImageGroundtruthToCoco(image_id, next_annotation_id, category_id_set, groundtruth_boxes, groundtruth_classes, groundtruth_masks=None, groundtruth_is_crowd=None):
    if False:
        i = 10
        return i + 15
    'Export groundtruth of a single image to COCO format.\n\n  This function converts groundtruth detection annotations represented as numpy\n  arrays to dictionaries that can be ingested by the COCO evaluation API. Note\n  that the image_ids provided here must match the ones given to\n  ExportSingleImageDetectionsToCoco. We assume that boxes and classes are in\n  correspondence - that is: groundtruth_boxes[i, :], and\n  groundtruth_classes[i] are associated with the same groundtruth annotation.\n\n  In the exported result, "area" fields are always set to the area of the\n  groundtruth bounding box.\n\n  Args:\n    image_id: a unique image identifier either of type integer or string.\n    next_annotation_id: integer specifying the first id to use for the\n      groundtruth annotations. All annotations are assigned a continuous integer\n      id starting from this value.\n    category_id_set: A set of valid class ids. Groundtruth with classes not in\n      category_id_set are dropped.\n    groundtruth_boxes: numpy array (float32) with shape [num_gt_boxes, 4]\n    groundtruth_classes: numpy array (int) with shape [num_gt_boxes]\n    groundtruth_masks: optional uint8 numpy array of shape [num_detections,\n      image_height, image_width] containing detection_masks.\n    groundtruth_is_crowd: optional numpy array (int) with shape [num_gt_boxes]\n      indicating whether groundtruth boxes are crowd.\n\n  Returns:\n    a list of groundtruth annotations for a single image in the COCO format.\n\n  Raises:\n    ValueError: if (1) groundtruth_boxes and groundtruth_classes do not have the\n      right lengths or (2) if each of the elements inside these lists do not\n      have the correct shapes or (3) if image_ids are not integers\n  '
    if len(groundtruth_classes.shape) != 1:
        raise ValueError('groundtruth_classes is expected to be of rank 1.')
    if len(groundtruth_boxes.shape) != 2:
        raise ValueError('groundtruth_boxes is expected to be of rank 2.')
    if groundtruth_boxes.shape[1] != 4:
        raise ValueError('groundtruth_boxes should have shape[1] == 4.')
    num_boxes = groundtruth_classes.shape[0]
    if num_boxes != groundtruth_boxes.shape[0]:
        raise ValueError('Corresponding entries in groundtruth_classes, and groundtruth_boxes should have compatible shapes (i.e., agree on the 0th dimension).Classes shape: %d. Boxes shape: %d. Image ID: %s' % (groundtruth_classes.shape[0], groundtruth_boxes.shape[0], image_id))
    has_is_crowd = groundtruth_is_crowd is not None
    if has_is_crowd and len(groundtruth_is_crowd.shape) != 1:
        raise ValueError('groundtruth_is_crowd is expected to be of rank 1.')
    groundtruth_list = []
    for i in range(num_boxes):
        if groundtruth_classes[i] in category_id_set:
            iscrowd = groundtruth_is_crowd[i] if has_is_crowd else 0
            export_dict = {'id': next_annotation_id + i, 'image_id': image_id, 'category_id': int(groundtruth_classes[i]), 'bbox': list(_ConvertBoxToCOCOFormat(groundtruth_boxes[i, :])), 'area': float((groundtruth_boxes[i, 2] - groundtruth_boxes[i, 0]) * (groundtruth_boxes[i, 3] - groundtruth_boxes[i, 1])), 'iscrowd': iscrowd}
            if groundtruth_masks is not None:
                export_dict['segmentation'] = _RleCompress(groundtruth_masks[i])
            groundtruth_list.append(export_dict)
    return groundtruth_list

def ExportGroundtruthToCOCO(image_ids, groundtruth_boxes, groundtruth_classes, categories, output_path=None):
    if False:
        print('Hello World!')
    'Export groundtruth detection annotations in numpy arrays to COCO API.\n\n  This function converts a set of groundtruth detection annotations represented\n  as numpy arrays to dictionaries that can be ingested by the COCO API.\n  Inputs to this function are three lists: image ids for each groundtruth image,\n  groundtruth boxes for each image and groundtruth classes respectively.\n  Note that the image_ids provided here must match the ones given to the\n  ExportDetectionsToCOCO function in order for evaluation to work properly.\n  We assume that for each image, boxes, scores and classes are in\n  correspondence --- that is: image_id[i], groundtruth_boxes[i, :] and\n  groundtruth_classes[i] are associated with the same groundtruth annotation.\n\n  In the exported result, "area" fields are always set to the area of the\n  groundtruth bounding box and "iscrowd" fields are always set to 0.\n  TODO(jonathanhuang): pass in "iscrowd" array for evaluating on COCO dataset.\n\n  Args:\n    image_ids: a list of unique image identifier either of type integer or\n      string.\n    groundtruth_boxes: list of numpy arrays with shape [num_gt_boxes, 4]\n      (note that num_gt_boxes can be different for each entry in the list)\n    groundtruth_classes: list of numpy arrays (int) with shape [num_gt_boxes]\n      (note that num_gt_boxes can be different for each entry in the list)\n    categories: a list of dictionaries representing all possible categories.\n        Each dict in this list has the following keys:\n          \'id\': (required) an integer id uniquely identifying this category\n          \'name\': (required) string representing category name\n            e.g., \'cat\', \'dog\', \'pizza\'\n          \'supercategory\': (optional) string representing the supercategory\n            e.g., \'animal\', \'vehicle\', \'food\', etc\n    output_path: (optional) path for exporting result to JSON\n  Returns:\n    dictionary that can be read by COCO API\n  Raises:\n    ValueError: if (1) groundtruth_boxes and groundtruth_classes do not have the\n      right lengths or (2) if each of the elements inside these lists do not\n      have the correct shapes or (3) if image_ids are not integers\n  '
    category_id_set = set([cat['id'] for cat in categories])
    groundtruth_export_list = []
    image_export_list = []
    if not len(image_ids) == len(groundtruth_boxes) == len(groundtruth_classes):
        raise ValueError('Input lists must have the same length')
    annotation_id = 1
    for (image_id, boxes, classes) in zip(image_ids, groundtruth_boxes, groundtruth_classes):
        image_export_list.append({'id': image_id})
        groundtruth_export_list.extend(ExportSingleImageGroundtruthToCoco(image_id, annotation_id, category_id_set, boxes, classes))
        num_boxes = classes.shape[0]
        annotation_id += num_boxes
    groundtruth_dict = {'annotations': groundtruth_export_list, 'images': image_export_list, 'categories': categories}
    if output_path:
        with tf.gfile.GFile(output_path, 'w') as fid:
            json_utils.Dump(groundtruth_dict, fid, float_digits=4, indent=2)
    return groundtruth_dict

def ExportSingleImageDetectionBoxesToCoco(image_id, category_id_set, detection_boxes, detection_scores, detection_classes):
    if False:
        while True:
            i = 10
    'Export detections of a single image to COCO format.\n\n  This function converts detections represented as numpy arrays to dictionaries\n  that can be ingested by the COCO evaluation API. Note that the image_ids\n  provided here must match the ones given to the\n  ExporSingleImageDetectionBoxesToCoco. We assume that boxes, and classes are in\n  correspondence - that is: boxes[i, :], and classes[i]\n  are associated with the same groundtruth annotation.\n\n  Args:\n    image_id: unique image identifier either of type integer or string.\n    category_id_set: A set of valid class ids. Detections with classes not in\n      category_id_set are dropped.\n    detection_boxes: float numpy array of shape [num_detections, 4] containing\n      detection boxes.\n    detection_scores: float numpy array of shape [num_detections] containing\n      scored for the detection boxes.\n    detection_classes: integer numpy array of shape [num_detections] containing\n      the classes for detection boxes.\n\n  Returns:\n    a list of detection annotations for a single image in the COCO format.\n\n  Raises:\n    ValueError: if (1) detection_boxes, detection_scores and detection_classes\n      do not have the right lengths or (2) if each of the elements inside these\n      lists do not have the correct shapes or (3) if image_ids are not integers.\n  '
    if len(detection_classes.shape) != 1 or len(detection_scores.shape) != 1:
        raise ValueError('All entries in detection_classes and detection_scoresexpected to be of rank 1.')
    if len(detection_boxes.shape) != 2:
        raise ValueError('All entries in detection_boxes expected to be of rank 2.')
    if detection_boxes.shape[1] != 4:
        raise ValueError('All entries in detection_boxes should have shape[1] == 4.')
    num_boxes = detection_classes.shape[0]
    if not num_boxes == detection_boxes.shape[0] == detection_scores.shape[0]:
        raise ValueError('Corresponding entries in detection_classes, detection_scores and detection_boxes should have compatible shapes (i.e., agree on the 0th dimension). Classes shape: %d. Boxes shape: %d. Scores shape: %d' % (detection_classes.shape[0], detection_boxes.shape[0], detection_scores.shape[0]))
    detections_list = []
    for i in range(num_boxes):
        if detection_classes[i] in category_id_set:
            detections_list.append({'image_id': image_id, 'category_id': int(detection_classes[i]), 'bbox': list(_ConvertBoxToCOCOFormat(detection_boxes[i, :])), 'score': float(detection_scores[i])})
    return detections_list

def ExportSingleImageDetectionMasksToCoco(image_id, category_id_set, detection_masks, detection_scores, detection_classes):
    if False:
        return 10
    'Export detection masks of a single image to COCO format.\n\n  This function converts detections represented as numpy arrays to dictionaries\n  that can be ingested by the COCO evaluation API. We assume that\n  detection_masks, detection_scores, and detection_classes are in correspondence\n  - that is: detection_masks[i, :], detection_classes[i] and detection_scores[i]\n    are associated with the same annotation.\n\n  Args:\n    image_id: unique image identifier either of type integer or string.\n    category_id_set: A set of valid class ids. Detections with classes not in\n      category_id_set are dropped.\n    detection_masks: uint8 numpy array of shape [num_detections, image_height,\n      image_width] containing detection_masks.\n    detection_scores: float numpy array of shape [num_detections] containing\n      scores for detection masks.\n    detection_classes: integer numpy array of shape [num_detections] containing\n      the classes for detection masks.\n\n  Returns:\n    a list of detection mask annotations for a single image in the COCO format.\n\n  Raises:\n    ValueError: if (1) detection_masks, detection_scores and detection_classes\n      do not have the right lengths or (2) if each of the elements inside these\n      lists do not have the correct shapes or (3) if image_ids are not integers.\n  '
    if len(detection_classes.shape) != 1 or len(detection_scores.shape) != 1:
        raise ValueError('All entries in detection_classes and detection_scoresexpected to be of rank 1.')
    num_boxes = detection_classes.shape[0]
    if not num_boxes == len(detection_masks) == detection_scores.shape[0]:
        raise ValueError('Corresponding entries in detection_classes, detection_scores and detection_masks should have compatible lengths and shapes Classes length: %d.  Masks length: %d. Scores length: %d' % (detection_classes.shape[0], len(detection_masks), detection_scores.shape[0]))
    detections_list = []
    for i in range(num_boxes):
        if detection_classes[i] in category_id_set:
            detections_list.append({'image_id': image_id, 'category_id': int(detection_classes[i]), 'segmentation': _RleCompress(detection_masks[i]), 'score': float(detection_scores[i])})
    return detections_list

def ExportDetectionsToCOCO(image_ids, detection_boxes, detection_scores, detection_classes, categories, output_path=None):
    if False:
        for i in range(10):
            print('nop')
    "Export detection annotations in numpy arrays to COCO API.\n\n  This function converts a set of predicted detections represented\n  as numpy arrays to dictionaries that can be ingested by the COCO API.\n  Inputs to this function are lists, consisting of boxes, scores and\n  classes, respectively, corresponding to each image for which detections\n  have been produced.  Note that the image_ids provided here must\n  match the ones given to the ExportGroundtruthToCOCO function in order\n  for evaluation to work properly.\n\n  We assume that for each image, boxes, scores and classes are in\n  correspondence --- that is: detection_boxes[i, :], detection_scores[i] and\n  detection_classes[i] are associated with the same detection.\n\n  Args:\n    image_ids: a list of unique image identifier either of type integer or\n      string.\n    detection_boxes: list of numpy arrays with shape [num_detection_boxes, 4]\n    detection_scores: list of numpy arrays (float) with shape\n      [num_detection_boxes]. Note that num_detection_boxes can be different\n      for each entry in the list.\n    detection_classes: list of numpy arrays (int) with shape\n      [num_detection_boxes]. Note that num_detection_boxes can be different\n      for each entry in the list.\n    categories: a list of dictionaries representing all possible categories.\n      Each dict in this list must have an integer 'id' key uniquely identifying\n      this category.\n    output_path: (optional) path for exporting result to JSON\n\n  Returns:\n    list of dictionaries that can be read by COCO API, where each entry\n    corresponds to a single detection and has keys from:\n    ['image_id', 'category_id', 'bbox', 'score'].\n  Raises:\n    ValueError: if (1) detection_boxes and detection_classes do not have the\n      right lengths or (2) if each of the elements inside these lists do not\n      have the correct shapes or (3) if image_ids are not integers.\n  "
    category_id_set = set([cat['id'] for cat in categories])
    detections_export_list = []
    if not len(image_ids) == len(detection_boxes) == len(detection_scores) == len(detection_classes):
        raise ValueError('Input lists must have the same length')
    for (image_id, boxes, scores, classes) in zip(image_ids, detection_boxes, detection_scores, detection_classes):
        detections_export_list.extend(ExportSingleImageDetectionBoxesToCoco(image_id, category_id_set, boxes, scores, classes))
    if output_path:
        with tf.gfile.GFile(output_path, 'w') as fid:
            json_utils.Dump(detections_export_list, fid, float_digits=4, indent=2)
    return detections_export_list

def ExportSegmentsToCOCO(image_ids, detection_masks, detection_scores, detection_classes, categories, output_path=None):
    if False:
        return 10
    "Export segmentation masks in numpy arrays to COCO API.\n\n  This function converts a set of predicted instance masks represented\n  as numpy arrays to dictionaries that can be ingested by the COCO API.\n  Inputs to this function are lists, consisting of segments, scores and\n  classes, respectively, corresponding to each image for which detections\n  have been produced.\n\n  Note this function is recommended to use for small dataset.\n  For large dataset, it should be used with a merge function\n  (e.g. in map reduce), otherwise the memory consumption is large.\n\n  We assume that for each image, masks, scores and classes are in\n  correspondence --- that is: detection_masks[i, :, :, :], detection_scores[i]\n  and detection_classes[i] are associated with the same detection.\n\n  Args:\n    image_ids: list of image ids (typically ints or strings)\n    detection_masks: list of numpy arrays with shape [num_detection, h, w, 1]\n      and type uint8. The height and width should match the shape of\n      corresponding image.\n    detection_scores: list of numpy arrays (float) with shape\n      [num_detection]. Note that num_detection can be different\n      for each entry in the list.\n    detection_classes: list of numpy arrays (int) with shape\n      [num_detection]. Note that num_detection can be different\n      for each entry in the list.\n    categories: a list of dictionaries representing all possible categories.\n      Each dict in this list must have an integer 'id' key uniquely identifying\n      this category.\n    output_path: (optional) path for exporting result to JSON\n\n  Returns:\n    list of dictionaries that can be read by COCO API, where each entry\n    corresponds to a single detection and has keys from:\n    ['image_id', 'category_id', 'segmentation', 'score'].\n\n  Raises:\n    ValueError: if detection_masks and detection_classes do not have the\n      right lengths or if each of the elements inside these lists do not\n      have the correct shapes.\n  "
    if not len(image_ids) == len(detection_masks) == len(detection_scores) == len(detection_classes):
        raise ValueError('Input lists must have the same length')
    segment_export_list = []
    for (image_id, masks, scores, classes) in zip(image_ids, detection_masks, detection_scores, detection_classes):
        if len(classes.shape) != 1 or len(scores.shape) != 1:
            raise ValueError('All entries in detection_classes and detection_scoresexpected to be of rank 1.')
        if len(masks.shape) != 4:
            raise ValueError('All entries in masks expected to be of rank 4. Given {}'.format(masks.shape))
        num_boxes = classes.shape[0]
        if not num_boxes == masks.shape[0] == scores.shape[0]:
            raise ValueError('Corresponding entries in segment_classes, detection_scores and detection_boxes should have compatible shapes (i.e., agree on the 0th dimension).')
        category_id_set = set([cat['id'] for cat in categories])
        segment_export_list.extend(ExportSingleImageDetectionMasksToCoco(image_id, category_id_set, np.squeeze(masks, axis=3), scores, classes))
    if output_path:
        with tf.gfile.GFile(output_path, 'w') as fid:
            json_utils.Dump(segment_export_list, fid, float_digits=4, indent=2)
    return segment_export_list

def ExportKeypointsToCOCO(image_ids, detection_keypoints, detection_scores, detection_classes, categories, output_path=None):
    if False:
        i = 10
        return i + 15
    "Exports keypoints in numpy arrays to COCO API.\n\n  This function converts a set of predicted keypoints represented\n  as numpy arrays to dictionaries that can be ingested by the COCO API.\n  Inputs to this function are lists, consisting of keypoints, scores and\n  classes, respectively, corresponding to each image for which detections\n  have been produced.\n\n  We assume that for each image, keypoints, scores and classes are in\n  correspondence --- that is: detection_keypoints[i, :, :, :],\n  detection_scores[i] and detection_classes[i] are associated with the same\n  detection.\n\n  Args:\n    image_ids: list of image ids (typically ints or strings)\n    detection_keypoints: list of numpy arrays with shape\n      [num_detection, num_keypoints, 2] and type float32 in absolute\n      x-y coordinates.\n    detection_scores: list of numpy arrays (float) with shape\n      [num_detection]. Note that num_detection can be different\n      for each entry in the list.\n    detection_classes: list of numpy arrays (int) with shape\n      [num_detection]. Note that num_detection can be different\n      for each entry in the list.\n    categories: a list of dictionaries representing all possible categories.\n      Each dict in this list must have an integer 'id' key uniquely identifying\n      this category and an integer 'num_keypoints' key specifying the number of\n      keypoints the category has.\n    output_path: (optional) path for exporting result to JSON\n\n  Returns:\n    list of dictionaries that can be read by COCO API, where each entry\n    corresponds to a single detection and has keys from:\n    ['image_id', 'category_id', 'keypoints', 'score'].\n\n  Raises:\n    ValueError: if detection_keypoints and detection_classes do not have the\n      right lengths or if each of the elements inside these lists do not\n      have the correct shapes.\n  "
    if not len(image_ids) == len(detection_keypoints) == len(detection_scores) == len(detection_classes):
        raise ValueError('Input lists must have the same length')
    keypoints_export_list = []
    for (image_id, keypoints, scores, classes) in zip(image_ids, detection_keypoints, detection_scores, detection_classes):
        if len(classes.shape) != 1 or len(scores.shape) != 1:
            raise ValueError('All entries in detection_classes and detection_scoresexpected to be of rank 1.')
        if len(keypoints.shape) != 3:
            raise ValueError('All entries in keypoints expected to be of rank 3. Given {}'.format(keypoints.shape))
        num_boxes = classes.shape[0]
        if not num_boxes == keypoints.shape[0] == scores.shape[0]:
            raise ValueError('Corresponding entries in detection_classes, detection_keypoints, and detection_scores should have compatible shapes (i.e., agree on the 0th dimension).')
        category_id_set = set([cat['id'] for cat in categories])
        category_id_to_num_keypoints_map = {cat['id']: cat['num_keypoints'] for cat in categories if 'num_keypoints' in cat}
        for i in range(num_boxes):
            if classes[i] not in category_id_set:
                raise ValueError('class id should be in category_id_set\n')
            if classes[i] in category_id_to_num_keypoints_map:
                num_keypoints = category_id_to_num_keypoints_map[classes[i]]
                instance_keypoints = np.concatenate([keypoints[i, 0:num_keypoints, :], np.expand_dims(np.ones(num_keypoints), axis=1)], axis=1).astype(int)
                instance_keypoints = instance_keypoints.flatten().tolist()
                keypoints_export_list.append({'image_id': image_id, 'category_id': int(classes[i]), 'keypoints': instance_keypoints, 'score': float(scores[i])})
    if output_path:
        with tf.gfile.GFile(output_path, 'w') as fid:
            json_utils.Dump(keypoints_export_list, fid, float_digits=4, indent=2)
    return keypoints_export_list