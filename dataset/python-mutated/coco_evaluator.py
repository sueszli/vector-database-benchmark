"""The COCO-style evaluator.

The following snippet demonstrates the use of interfaces:

  evaluator = COCOEvaluator(...)
  for _ in range(num_evals):
    for _ in range(num_batches_per_eval):
      predictions, groundtruth = predictor.predict(...)  # pop a batch.
      evaluator.update(predictions, groundtruths)  # aggregate internal stats.
    evaluator.evaluate()  # finish one full eval.

See also: https://github.com/cocodataset/cocoapi/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import atexit
import tempfile
import numpy as np
from absl import logging
from pycocotools import cocoeval
import six
import tensorflow.compat.v2 as tf
from official.vision.detection.evaluation import coco_utils
from official.vision.detection.utils import class_utils

class COCOEvaluator(object):
    """COCO evaluation metric class."""

    def __init__(self, annotation_file, include_mask, need_rescale_bboxes=True):
        if False:
            print('Hello World!')
        'Constructs COCO evaluation class.\n\n    The class provides the interface to metrics_fn in TPUEstimator. The\n    _update_op() takes detections from each image and push them to\n    self.detections. The _evaluate() loads a JSON file in COCO annotation format\n    as the groundtruths and runs COCO evaluation.\n\n    Args:\n      annotation_file: a JSON file that stores annotations of the eval dataset.\n        If `annotation_file` is None, groundtruth annotations will be loaded\n        from the dataloader.\n      include_mask: a boolean to indicate whether or not to include the mask\n        eval.\n      need_rescale_bboxes: If true bboxes in `predictions` will be rescaled back\n        to absolute values (`image_info` is needed in this case).\n    '
        if annotation_file:
            if annotation_file.startswith('gs://'):
                (_, local_val_json) = tempfile.mkstemp(suffix='.json')
                tf.io.gfile.remove(local_val_json)
                tf.io.gfile.copy(annotation_file, local_val_json)
                atexit.register(tf.io.gfile.remove, local_val_json)
            else:
                local_val_json = annotation_file
            self._coco_gt = coco_utils.COCOWrapper(eval_type='mask' if include_mask else 'box', annotation_file=local_val_json)
        self._annotation_file = annotation_file
        self._include_mask = include_mask
        self._metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
        self._required_prediction_fields = ['source_id', 'num_detections', 'detection_classes', 'detection_scores', 'detection_boxes']
        self._need_rescale_bboxes = need_rescale_bboxes
        if self._need_rescale_bboxes:
            self._required_prediction_fields.append('image_info')
        self._required_groundtruth_fields = ['source_id', 'height', 'width', 'classes', 'boxes']
        if self._include_mask:
            mask_metric_names = ['mask_' + x for x in self._metric_names]
            self._metric_names.extend(mask_metric_names)
            self._required_prediction_fields.extend(['detection_masks'])
            self._required_groundtruth_fields.extend(['masks'])
        self.reset()

    def reset(self):
        if False:
            return 10
        'Resets internal states for a fresh run.'
        self._predictions = {}
        if not self._annotation_file:
            self._groundtruths = {}

    def evaluate(self):
        if False:
            while True:
                i = 10
        'Evaluates with detections from all images with COCO API.\n\n    Returns:\n      coco_metric: float numpy array with shape [24] representing the\n        coco-style evaluation metrics (box and mask).\n    '
        if not self._annotation_file:
            logging.info('Thre is no annotation_file in COCOEvaluator.')
            gt_dataset = coco_utils.convert_groundtruths_to_coco_dataset(self._groundtruths)
            coco_gt = coco_utils.COCOWrapper(eval_type='mask' if self._include_mask else 'box', gt_dataset=gt_dataset)
        else:
            logging.info('Using annotation file: %s', self._annotation_file)
            coco_gt = self._coco_gt
        coco_predictions = coco_utils.convert_predictions_to_coco_annotations(self._predictions)
        coco_dt = coco_gt.loadRes(predictions=coco_predictions)
        image_ids = [ann['image_id'] for ann in coco_predictions]
        coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats
        if self._include_mask:
            mcoco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='segm')
            mcoco_eval.params.imgIds = image_ids
            mcoco_eval.evaluate()
            mcoco_eval.accumulate()
            mcoco_eval.summarize()
            mask_coco_metrics = mcoco_eval.stats
        if self._include_mask:
            metrics = np.hstack((coco_metrics, mask_coco_metrics))
        else:
            metrics = coco_metrics
        self.reset()
        metrics_dict = {}
        for (i, name) in enumerate(self._metric_names):
            metrics_dict[name] = metrics[i].astype(np.float32)
        return metrics_dict

    def _process_predictions(self, predictions):
        if False:
            i = 10
            return i + 15
        image_scale = np.tile(predictions['image_info'][:, 2:3, :], (1, 1, 2))
        predictions['detection_boxes'] = predictions['detection_boxes'].astype(np.float32)
        predictions['detection_boxes'] /= image_scale
        if 'detection_outer_boxes' in predictions:
            predictions['detection_outer_boxes'] = predictions['detection_outer_boxes'].astype(np.float32)
            predictions['detection_outer_boxes'] /= image_scale

    def update(self, predictions, groundtruths=None):
        if False:
            for i in range(10):
                print('nop')
        'Update and aggregate detection results and groundtruth data.\n\n    Args:\n      predictions: a dictionary of numpy arrays including the fields below.\n        See different parsers under `../dataloader` for more details.\n        Required fields:\n          - source_id: a numpy array of int or string of shape [batch_size].\n          - image_info [if `need_rescale_bboxes` is True]: a numpy array of\n            float of shape [batch_size, 4, 2].\n          - num_detections: a numpy array of\n            int of shape [batch_size].\n          - detection_boxes: a numpy array of float of shape [batch_size, K, 4].\n          - detection_classes: a numpy array of int of shape [batch_size, K].\n          - detection_scores: a numpy array of float of shape [batch_size, K].\n        Optional fields:\n          - detection_masks: a numpy array of float of shape\n              [batch_size, K, mask_height, mask_width].\n      groundtruths: a dictionary of numpy arrays including the fields below.\n        See also different parsers under `../dataloader` for more details.\n        Required fields:\n          - source_id: a numpy array of int or string of shape [batch_size].\n          - height: a numpy array of int of shape [batch_size].\n          - width: a numpy array of int of shape [batch_size].\n          - num_detections: a numpy array of int of shape [batch_size].\n          - boxes: a numpy array of float of shape [batch_size, K, 4].\n          - classes: a numpy array of int of shape [batch_size, K].\n        Optional fields:\n          - is_crowds: a numpy array of int of shape [batch_size, K]. If the\n              field is absent, it is assumed that this instance is not crowd.\n          - areas: a numy array of float of shape [batch_size, K]. If the\n              field is absent, the area is calculated using either boxes or\n              masks depending on which one is available.\n          - masks: a numpy array of float of shape\n              [batch_size, K, mask_height, mask_width],\n\n    Raises:\n      ValueError: if the required prediction or groundtruth fields are not\n        present in the incoming `predictions` or `groundtruths`.\n    '
        for k in self._required_prediction_fields:
            if k not in predictions:
                raise ValueError('Missing the required key `{}` in predictions!'.format(k))
        if self._need_rescale_bboxes:
            self._process_predictions(predictions)
        for (k, v) in six.iteritems(predictions):
            if k not in self._predictions:
                self._predictions[k] = [v]
            else:
                self._predictions[k].append(v)
        if not self._annotation_file:
            assert groundtruths
            for k in self._required_groundtruth_fields:
                if k not in groundtruths:
                    raise ValueError('Missing the required key `{}` in groundtruths!'.format(k))
            for (k, v) in six.iteritems(groundtruths):
                if k not in self._groundtruths:
                    self._groundtruths[k] = [v]
                else:
                    self._groundtruths[k].append(v)

class ShapeMaskCOCOEvaluator(COCOEvaluator):
    """COCO evaluation metric class for ShapeMask."""

    def __init__(self, mask_eval_class, **kwargs):
        if False:
            while True:
                i = 10
        'Constructs COCO evaluation class.\n\n    The class provides the interface to metrics_fn in TPUEstimator. The\n    _update_op() takes detections from each image and push them to\n    self.detections. The _evaluate() loads a JSON file in COCO annotation format\n    as the groundtruths and runs COCO evaluation.\n\n    Args:\n      mask_eval_class: the set of classes for mask evaluation.\n      **kwargs: other keyword arguments passed to the parent class initializer.\n    '
        super(ShapeMaskCOCOEvaluator, self).__init__(**kwargs)
        self._mask_eval_class = mask_eval_class
        self._eval_categories = class_utils.coco_split_class_ids(mask_eval_class)
        if mask_eval_class != 'all':
            self._metric_names = [x.replace('mask', 'novel_mask') for x in self._metric_names]

    def evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates with detections from all images with COCO API.\n\n    Returns:\n      coco_metric: float numpy array with shape [24] representing the\n        coco-style evaluation metrics (box and mask).\n    '
        if not self._annotation_file:
            gt_dataset = coco_utils.convert_groundtruths_to_coco_dataset(self._groundtruths)
            coco_gt = coco_utils.COCOWrapper(eval_type='mask' if self._include_mask else 'box', gt_dataset=gt_dataset)
        else:
            coco_gt = self._coco_gt
        coco_predictions = coco_utils.convert_predictions_to_coco_annotations(self._predictions)
        coco_dt = coco_gt.loadRes(predictions=coco_predictions)
        image_ids = [ann['image_id'] for ann in coco_predictions]
        coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats
        if self._include_mask:
            mcoco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='segm')
            mcoco_eval.params.imgIds = image_ids
            mcoco_eval.evaluate()
            mcoco_eval.accumulate()
            mcoco_eval.summarize()
            if self._mask_eval_class == 'all':
                metrics = np.hstack((coco_metrics, mcoco_eval.stats))
            else:
                mask_coco_metrics = mcoco_eval.category_stats
                val_catg_idx = np.isin(mcoco_eval.params.catIds, self._eval_categories)
                if np.any(val_catg_idx):
                    mean_val_metrics = []
                    for mid in range(len(self._metric_names) // 2):
                        mean_val_metrics.append(np.nanmean(mask_coco_metrics[mid][val_catg_idx]))
                    mean_val_metrics = np.array(mean_val_metrics)
                else:
                    mean_val_metrics = np.zeros(len(self._metric_names) // 2)
                metrics = np.hstack((coco_metrics, mean_val_metrics))
        else:
            metrics = coco_metrics
        self.reset()
        metrics_dict = {}
        for (i, name) in enumerate(self._metric_names):
            metrics_dict[name] = metrics[i].astype(np.float32)
        return metrics_dict