"""Class for evaluating object detections with calibration metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list
from object_detection.core import region_similarity_calculator
from object_detection.core import standard_fields
from object_detection.core import target_assigner
from object_detection.matchers import argmax_matcher
from object_detection.metrics import calibration_metrics
from object_detection.utils import object_detection_evaluation

class CalibrationDetectionEvaluator(object_detection_evaluation.DetectionEvaluator):
    """Class to evaluate calibration detection metrics."""

    def __init__(self, categories, iou_threshold=0.5):
        if False:
            print('Hello World!')
        "Constructor.\n\n    Args:\n      categories: A list of dicts, each of which has the following keys -\n        'id': (required) an integer id uniquely identifying this category.\n        'name': (required) string representing category name e.g., 'cat', 'dog'.\n      iou_threshold: Threshold above which to consider a box as matched during\n        evaluation.\n    "
        super(CalibrationDetectionEvaluator, self).__init__(categories)
        similarity_calc = region_similarity_calculator.IouSimilarity()
        matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=iou_threshold, unmatched_threshold=iou_threshold)
        box_coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
        self._target_assigner = target_assigner.TargetAssigner(similarity_calc, matcher, box_coder)

    def match_single_image_info(self, image_info):
        if False:
            return 10
        'Match detections to groundtruth for a single image.\n\n    Detections are matched to available groundtruth in the image based on the\n    IOU threshold from the constructor.  The classes of the detections and\n    groundtruth matches are then compared. Detections that do not have IOU above\n    the required threshold or have different classes from their match are\n    considered negative matches. All inputs in `image_info` originate or are\n    inferred from the eval_dict passed to class method\n    `get_estimator_eval_metric_ops`.\n\n    Args:\n      image_info: a tuple or list containing the following (in order):\n        - gt_boxes: tf.float32 tensor of groundtruth boxes.\n        - gt_classes: tf.int64 tensor of groundtruth classes associated with\n            groundtruth boxes.\n        - num_gt_box: scalar indicating the number of groundtruth boxes per\n            image.\n        - det_boxes: tf.float32 tensor of detection boxes.\n        - det_classes: tf.int64 tensor of detection classes associated with\n            detection boxes.\n        - num_det_box: scalar indicating the number of detection boxes per\n            image.\n    Returns:\n      is_class_matched: tf.int64 tensor identical in shape to det_boxes,\n        indicating whether detection boxes matched with and had the same\n        class as groundtruth annotations.\n    '
        (gt_boxes, gt_classes, num_gt_box, det_boxes, det_classes, num_det_box) = image_info
        detection_boxes = det_boxes[:num_det_box]
        detection_classes = det_classes[:num_det_box]
        groundtruth_boxes = gt_boxes[:num_gt_box]
        groundtruth_classes = gt_classes[:num_gt_box]
        det_boxlist = box_list.BoxList(detection_boxes)
        gt_boxlist = box_list.BoxList(groundtruth_boxes)
        one_hot_depth = tf.cast(tf.add(tf.reduce_max(groundtruth_classes), 1), dtype=tf.int32)
        gt_classes_one_hot = tf.one_hot(groundtruth_classes, one_hot_depth, dtype=tf.float32)
        (one_hot_cls_targets, _, _, _, _) = self._target_assigner.assign(det_boxlist, gt_boxlist, gt_classes_one_hot, unmatched_class_label=tf.zeros(shape=one_hot_depth, dtype=tf.float32))
        cls_targets = tf.argmax(one_hot_cls_targets, axis=1)
        is_class_matched = tf.cast(tf.equal(tf.cast(cls_targets, tf.int64), detection_classes), dtype=tf.int64)
        return is_class_matched

    def get_estimator_eval_metric_ops(self, eval_dict):
        if False:
            while True:
                i = 10
        "Returns a dictionary of eval metric ops.\n\n    Note that once value_op is called, the detections and groundtruth added via\n    update_op are cleared.\n\n    This function can take in groundtruth and detections for a batch of images,\n    or for a single image. For the latter case, the batch dimension for input\n    tensors need not be present.\n\n    Args:\n      eval_dict: A dictionary that holds tensors for evaluating object detection\n        performance. For single-image evaluation, this dictionary may be\n        produced from eval_util.result_dict_for_single_example(). If multi-image\n        evaluation, `eval_dict` should contain the fields\n        'num_groundtruth_boxes_per_image' and 'num_det_boxes_per_image' to\n        properly unpad the tensors from the batch.\n\n    Returns:\n      a dictionary of metric names to tuple of value_op and update_op that can\n      be used as eval metric ops in tf.estimator.EstimatorSpec. Note that all\n      update ops must be run together and similarly all value ops must be run\n      together to guarantee correct behaviour.\n    "
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        image_id = eval_dict[input_data_fields.key]
        groundtruth_boxes = eval_dict[input_data_fields.groundtruth_boxes]
        groundtruth_classes = eval_dict[input_data_fields.groundtruth_classes]
        detection_boxes = eval_dict[detection_fields.detection_boxes]
        detection_scores = eval_dict[detection_fields.detection_scores]
        detection_classes = eval_dict[detection_fields.detection_classes]
        num_gt_boxes_per_image = eval_dict.get('num_groundtruth_boxes_per_image', None)
        num_det_boxes_per_image = eval_dict.get('num_det_boxes_per_image', None)
        is_annotated_batched = eval_dict.get('is_annotated', None)
        if not image_id.shape.as_list():
            image_id = tf.expand_dims(image_id, 0)
            groundtruth_boxes = tf.expand_dims(groundtruth_boxes, 0)
            groundtruth_classes = tf.expand_dims(groundtruth_classes, 0)
            detection_boxes = tf.expand_dims(detection_boxes, 0)
            detection_scores = tf.expand_dims(detection_scores, 0)
            detection_classes = tf.expand_dims(detection_classes, 0)
            if num_gt_boxes_per_image is None:
                num_gt_boxes_per_image = tf.shape(groundtruth_boxes)[1:2]
            else:
                num_gt_boxes_per_image = tf.expand_dims(num_gt_boxes_per_image, 0)
            if num_det_boxes_per_image is None:
                num_det_boxes_per_image = tf.shape(detection_boxes)[1:2]
            else:
                num_det_boxes_per_image = tf.expand_dims(num_det_boxes_per_image, 0)
            if is_annotated_batched is None:
                is_annotated_batched = tf.constant([True])
            else:
                is_annotated_batched = tf.expand_dims(is_annotated_batched, 0)
        else:
            if num_gt_boxes_per_image is None:
                num_gt_boxes_per_image = tf.tile(tf.shape(groundtruth_boxes)[1:2], multiples=tf.shape(groundtruth_boxes)[0:1])
            if num_det_boxes_per_image is None:
                num_det_boxes_per_image = tf.tile(tf.shape(detection_boxes)[1:2], multiples=tf.shape(detection_boxes)[0:1])
            if is_annotated_batched is None:
                is_annotated_batched = tf.ones_like(image_id, dtype=tf.bool)
        image_info = [tf.boolean_mask(tensor, is_annotated_batched) for tensor in [groundtruth_boxes, groundtruth_classes, num_gt_boxes_per_image, detection_boxes, detection_classes, num_det_boxes_per_image]]
        is_class_matched = tf.map_fn(self.match_single_image_info, image_info, dtype=tf.int64)
        y_true = tf.squeeze(is_class_matched)
        y_pred = tf.squeeze(tf.boolean_mask(detection_scores, is_annotated_batched))
        (ece, update_op) = calibration_metrics.expected_calibration_error(y_true, y_pred)
        return {'CalibrationError/ExpectedCalibrationError': (ece, update_op)}

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        if False:
            while True:
                i = 10
        'Adds groundtruth for a single image to be used for evaluation.\n\n    Args:\n      image_id: A unique string/integer identifier for the image.\n      groundtruth_dict: A dictionary of groundtruth numpy arrays required\n        for evaluations.\n    '
        raise NotImplementedError

    def add_single_detected_image_info(self, image_id, detections_dict):
        if False:
            for i in range(10):
                print('nop')
        'Adds detections for a single image to be used for evaluation.\n\n    Args:\n      image_id: A unique string/integer identifier for the image.\n      detections_dict: A dictionary of detection numpy arrays required for\n        evaluation.\n    '
        raise NotImplementedError

    def evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates detections and returns a dictionary of metrics.'
        raise NotImplementedError

    def clear(self):
        if False:
            print('Hello World!')
        'Clears the state to prepare for a fresh evaluation.'
        raise NotImplementedError