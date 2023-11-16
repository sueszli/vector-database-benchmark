"""Evaluates Visual Relations Detection(VRD) result evaluation on an image.

Annotate each VRD result as true positives or false positive according to
a predefined IOU ratio. Multi-class detection is supported by default.
Based on the settings, per image evaluation is performed either on phrase
detection subtask or on relation detection subtask.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops

class PerImageVRDEvaluation(object):
    """Evaluate vrd result of a single image."""

    def __init__(self, matching_iou_threshold=0.5):
        if False:
            return 10
        'Initialized PerImageVRDEvaluation by evaluation parameters.\n\n    Args:\n      matching_iou_threshold: A ratio of area intersection to union, which is\n          the threshold to consider whether a detection is true positive or not;\n          in phrase detection subtask.\n    '
        self.matching_iou_threshold = matching_iou_threshold

    def compute_detection_tp_fp(self, detected_box_tuples, detected_scores, detected_class_tuples, groundtruth_box_tuples, groundtruth_class_tuples):
        if False:
            i = 10
            return i + 15
        'Evaluates VRD as being tp, fp from a single image.\n\n    Args:\n      detected_box_tuples: A numpy array of structures with shape [N,],\n          representing N tuples, each tuple containing the same number of named\n          bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max].\n      detected_scores: A float numpy array of shape [N,], representing\n          the confidence scores of the detected N object instances.\n      detected_class_tuples: A numpy array of structures shape [N,],\n          representing the class labels of the corresponding bounding boxes and\n          possibly additional classes.\n      groundtruth_box_tuples: A float numpy array of structures with the shape\n          [M,], representing M tuples, each tuple containing the same number\n          of named bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max].\n      groundtruth_class_tuples: A numpy array of structures shape [M,],\n          representing  the class labels of the corresponding bounding boxes and\n          possibly additional classes.\n\n    Returns:\n      scores: A single numpy array with shape [N,], representing N scores\n          detected with object class, sorted in descentent order.\n      tp_fp_labels: A single boolean numpy array of shape [N,], representing N\n          True/False positive label, one label per tuple. The labels are sorted\n          so that the order of the labels matches the order of the scores.\n      result_mapping: A numpy array with shape [N,] with original index of each\n          entry.\n    '
        (scores, tp_fp_labels, result_mapping) = self._compute_tp_fp(detected_box_tuples=detected_box_tuples, detected_scores=detected_scores, detected_class_tuples=detected_class_tuples, groundtruth_box_tuples=groundtruth_box_tuples, groundtruth_class_tuples=groundtruth_class_tuples)
        return (scores, tp_fp_labels, result_mapping)

    def _compute_tp_fp(self, detected_box_tuples, detected_scores, detected_class_tuples, groundtruth_box_tuples, groundtruth_class_tuples):
        if False:
            for i in range(10):
                print('nop')
        'Labels as true/false positives detection tuples across all classes.\n\n    Args:\n      detected_box_tuples: A numpy array of structures with shape [N,],\n          representing N tuples, each tuple containing the same number of named\n          bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max]\n      detected_scores: A float numpy array of shape [N,], representing\n          the confidence scores of the detected N object instances.\n      detected_class_tuples: A numpy array of structures shape [N,],\n          representing the class labels of the corresponding bounding boxes and\n          possibly additional classes.\n      groundtruth_box_tuples: A float numpy array of structures with the shape\n          [M,], representing M tuples, each tuple containing the same number\n          of named bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max]\n      groundtruth_class_tuples: A numpy array of structures shape [M,],\n          representing  the class labels of the corresponding bounding boxes and\n          possibly additional classes.\n\n    Returns:\n      scores: A single numpy array with shape [N,], representing N scores\n          detected with object class, sorted in descentent order.\n      tp_fp_labels: A single boolean numpy array of shape [N,], representing N\n          True/False positive label, one label per tuple. The labels are sorted\n          so that the order of the labels matches the order of the scores.\n      result_mapping: A numpy array with shape [N,] with original index of each\n          entry.\n    '
        unique_gt_tuples = np.unique(np.concatenate((groundtruth_class_tuples, detected_class_tuples)))
        result_scores = []
        result_tp_fp_labels = []
        result_mapping = []
        for unique_tuple in unique_gt_tuples:
            detections_selector = detected_class_tuples == unique_tuple
            gt_selector = groundtruth_class_tuples == unique_tuple
            selector_mapping = np.where(detections_selector)[0]
            detection_scores_per_tuple = detected_scores[detections_selector]
            detection_box_per_tuple = detected_box_tuples[detections_selector]
            sorted_indices = np.argsort(detection_scores_per_tuple)
            sorted_indices = sorted_indices[::-1]
            tp_fp_labels = self._compute_tp_fp_for_single_class(detected_box_tuples=detection_box_per_tuple[sorted_indices], groundtruth_box_tuples=groundtruth_box_tuples[gt_selector])
            result_scores.append(detection_scores_per_tuple[sorted_indices])
            result_tp_fp_labels.append(tp_fp_labels)
            result_mapping.append(selector_mapping[sorted_indices])
        if result_scores:
            result_scores = np.concatenate(result_scores)
            result_tp_fp_labels = np.concatenate(result_tp_fp_labels)
            result_mapping = np.concatenate(result_mapping)
        else:
            result_scores = np.array([], dtype=float)
            result_tp_fp_labels = np.array([], dtype=bool)
            result_mapping = np.array([], dtype=int)
        sorted_indices = np.argsort(result_scores)
        sorted_indices = sorted_indices[::-1]
        return (result_scores[sorted_indices], result_tp_fp_labels[sorted_indices], result_mapping[sorted_indices])

    def _get_overlaps_and_scores_relation_tuples(self, detected_box_tuples, groundtruth_box_tuples):
        if False:
            print('Hello World!')
        'Computes overlaps and scores between detected and groundtruth tuples.\n\n    Both detections and groundtruth boxes have the same class tuples.\n\n    Args:\n      detected_box_tuples: A numpy array of structures with shape [N,],\n          representing N tuples, each tuple containing the same number of named\n          bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max]\n      groundtruth_box_tuples: A float numpy array of structures with the shape\n          [M,], representing M tuples, each tuple containing the same number\n          of named bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max]\n\n    Returns:\n      result_iou: A float numpy array of size\n        [num_detected_tuples, num_gt_box_tuples].\n    '
        result_iou = np.ones((detected_box_tuples.shape[0], groundtruth_box_tuples.shape[0]), dtype=float)
        for field in detected_box_tuples.dtype.fields:
            detected_boxlist_field = np_box_list.BoxList(detected_box_tuples[field])
            gt_boxlist_field = np_box_list.BoxList(groundtruth_box_tuples[field])
            iou_field = np_box_list_ops.iou(detected_boxlist_field, gt_boxlist_field)
            result_iou = np.minimum(iou_field, result_iou)
        return result_iou

    def _compute_tp_fp_for_single_class(self, detected_box_tuples, groundtruth_box_tuples):
        if False:
            while True:
                i = 10
        'Labels boxes detected with the same class from the same image as tp/fp.\n\n    Detection boxes are expected to be already sorted by score.\n    Args:\n      detected_box_tuples: A numpy array of structures with shape [N,],\n          representing N tuples, each tuple containing the same number of named\n          bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max]\n      groundtruth_box_tuples: A float numpy array of structures with the shape\n          [M,], representing M tuples, each tuple containing the same number\n          of named bounding boxes.\n          Each box is of the format [y_min, x_min, y_max, x_max]\n\n    Returns:\n      tp_fp_labels: a boolean numpy array indicating whether a detection is a\n          true positive.\n    '
        if detected_box_tuples.size == 0:
            return np.array([], dtype=bool)
        min_iou = self._get_overlaps_and_scores_relation_tuples(detected_box_tuples, groundtruth_box_tuples)
        num_detected_tuples = detected_box_tuples.shape[0]
        tp_fp_labels = np.zeros(num_detected_tuples, dtype=bool)
        if min_iou.shape[1] > 0:
            max_overlap_gt_ids = np.argmax(min_iou, axis=1)
            is_gt_tuple_detected = np.zeros(min_iou.shape[1], dtype=bool)
            for i in range(num_detected_tuples):
                gt_id = max_overlap_gt_ids[i]
                if min_iou[i, gt_id] >= self.matching_iou_threshold:
                    if not is_gt_tuple_detected[gt_id]:
                        tp_fp_labels[i] = True
                        is_gt_tuple_detected[gt_id] = True
        return tp_fp_labels