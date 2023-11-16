"""Evaluate Object Detection result on a single image.

Annotate each detected result as true positives or false positive according to
a predefined IOU ratio. Non Maximum Supression is used by default. Multi class
detection is supported by default.
Based on the settings, per image evaluation is either performed on boxes or
on object masks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
from object_detection.utils import np_box_mask_list
from object_detection.utils import np_box_mask_list_ops

class PerImageEvaluation(object):
    """Evaluate detection result of a single image."""

    def __init__(self, num_groundtruth_classes, matching_iou_threshold=0.5, nms_iou_threshold=0.3, nms_max_output_boxes=50, group_of_weight=0.0):
        if False:
            while True:
                i = 10
        'Initialized PerImageEvaluation by evaluation parameters.\n\n    Args:\n      num_groundtruth_classes: Number of ground truth object classes\n      matching_iou_threshold: A ratio of area intersection to union, which is\n        the threshold to consider whether a detection is true positive or not\n      nms_iou_threshold: IOU threshold used in Non Maximum Suppression.\n      nms_max_output_boxes: Number of maximum output boxes in NMS.\n      group_of_weight: Weight of the group-of boxes.\n    '
        self.matching_iou_threshold = matching_iou_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_max_output_boxes = nms_max_output_boxes
        self.num_groundtruth_classes = num_groundtruth_classes
        self.group_of_weight = group_of_weight

    def compute_object_detection_metrics(self, detected_boxes, detected_scores, detected_class_labels, groundtruth_boxes, groundtruth_class_labels, groundtruth_is_difficult_list, groundtruth_is_group_of_list, detected_masks=None, groundtruth_masks=None):
        if False:
            while True:
                i = 10
        'Evaluates detections as being tp, fp or weighted from a single image.\n\n    The evaluation is done in two stages:\n     1. All detections are matched to non group-of boxes; true positives are\n        determined and detections matched to difficult boxes are ignored.\n     2. Detections that are determined as false positives are matched against\n        group-of boxes and weighted if matched.\n\n    Args:\n      detected_boxes: A float numpy array of shape [N, 4], representing N\n        regions of detected object regions. Each row is of the format [y_min,\n        x_min, y_max, x_max]\n      detected_scores: A float numpy array of shape [N, 1], representing the\n        confidence scores of the detected N object instances.\n      detected_class_labels: A integer numpy array of shape [N, 1], repreneting\n        the class labels of the detected N object instances.\n      groundtruth_boxes: A float numpy array of shape [M, 4], representing M\n        regions of object instances in ground truth\n      groundtruth_class_labels: An integer numpy array of shape [M, 1],\n        representing M class labels of object instances in ground truth\n      groundtruth_is_difficult_list: A boolean numpy array of length M denoting\n        whether a ground truth box is a difficult instance or not\n      groundtruth_is_group_of_list: A boolean numpy array of length M denoting\n        whether a ground truth box has group-of tag\n      detected_masks: (optional) A uint8 numpy array of shape [N, height,\n        width]. If not None, the metrics will be computed based on masks.\n      groundtruth_masks: (optional) A uint8 numpy array of shape [M, height,\n        width]. Can have empty masks, i.e. where all values are 0.\n\n    Returns:\n      scores: A list of C float numpy arrays. Each numpy array is of\n          shape [K, 1], representing K scores detected with object class\n          label c\n      tp_fp_labels: A list of C boolean numpy arrays. Each numpy array\n          is of shape [K, 1], representing K True/False positive label of\n          object instances detected with class label c\n      is_class_correctly_detected_in_image: a numpy integer array of\n          shape [C, 1], indicating whether the correponding class has a least\n          one instance being correctly detected in the image\n    '
        (detected_boxes, detected_scores, detected_class_labels, detected_masks) = self._remove_invalid_boxes(detected_boxes, detected_scores, detected_class_labels, detected_masks)
        (scores, tp_fp_labels) = self._compute_tp_fp(detected_boxes=detected_boxes, detected_scores=detected_scores, detected_class_labels=detected_class_labels, groundtruth_boxes=groundtruth_boxes, groundtruth_class_labels=groundtruth_class_labels, groundtruth_is_difficult_list=groundtruth_is_difficult_list, groundtruth_is_group_of_list=groundtruth_is_group_of_list, detected_masks=detected_masks, groundtruth_masks=groundtruth_masks)
        is_class_correctly_detected_in_image = self._compute_cor_loc(detected_boxes=detected_boxes, detected_scores=detected_scores, detected_class_labels=detected_class_labels, groundtruth_boxes=groundtruth_boxes, groundtruth_class_labels=groundtruth_class_labels, detected_masks=detected_masks, groundtruth_masks=groundtruth_masks)
        return (scores, tp_fp_labels, is_class_correctly_detected_in_image)

    def _compute_cor_loc(self, detected_boxes, detected_scores, detected_class_labels, groundtruth_boxes, groundtruth_class_labels, detected_masks=None, groundtruth_masks=None):
        if False:
            while True:
                i = 10
        'Compute CorLoc score for object detection result.\n\n    Args:\n      detected_boxes: A float numpy array of shape [N, 4], representing N\n        regions of detected object regions. Each row is of the format [y_min,\n        x_min, y_max, x_max]\n      detected_scores: A float numpy array of shape [N, 1], representing the\n        confidence scores of the detected N object instances.\n      detected_class_labels: A integer numpy array of shape [N, 1], repreneting\n        the class labels of the detected N object instances.\n      groundtruth_boxes: A float numpy array of shape [M, 4], representing M\n        regions of object instances in ground truth\n      groundtruth_class_labels: An integer numpy array of shape [M, 1],\n        representing M class labels of object instances in ground truth\n      detected_masks: (optional) A uint8 numpy array of shape [N, height,\n        width]. If not None, the scores will be computed based on masks.\n      groundtruth_masks: (optional) A uint8 numpy array of shape [M, height,\n        width].\n\n    Returns:\n      is_class_correctly_detected_in_image: a numpy integer array of\n          shape [C, 1], indicating whether the correponding class has a least\n          one instance being correctly detected in the image\n\n    Raises:\n      ValueError: If detected masks is not None but groundtruth masks are None,\n        or the other way around.\n    '
        if detected_masks is not None and groundtruth_masks is None or (detected_masks is None and groundtruth_masks is not None):
            raise ValueError('If `detected_masks` is provided, then `groundtruth_masks` should also be provided.')
        is_class_correctly_detected_in_image = np.zeros(self.num_groundtruth_classes, dtype=int)
        for i in range(self.num_groundtruth_classes):
            (gt_boxes_at_ith_class, gt_masks_at_ith_class, detected_boxes_at_ith_class, detected_scores_at_ith_class, detected_masks_at_ith_class) = self._get_ith_class_arrays(detected_boxes, detected_scores, detected_masks, detected_class_labels, groundtruth_boxes, groundtruth_masks, groundtruth_class_labels, i)
            is_class_correctly_detected_in_image[i] = self._compute_is_class_correctly_detected_in_image(detected_boxes=detected_boxes_at_ith_class, detected_scores=detected_scores_at_ith_class, groundtruth_boxes=gt_boxes_at_ith_class, detected_masks=detected_masks_at_ith_class, groundtruth_masks=gt_masks_at_ith_class)
        return is_class_correctly_detected_in_image

    def _compute_is_class_correctly_detected_in_image(self, detected_boxes, detected_scores, groundtruth_boxes, detected_masks=None, groundtruth_masks=None):
        if False:
            print('Hello World!')
        'Compute CorLoc score for a single class.\n\n    Args:\n      detected_boxes: A numpy array of shape [N, 4] representing detected box\n        coordinates\n      detected_scores: A 1-d numpy array of length N representing classification\n        score\n      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth\n        box coordinates\n      detected_masks: (optional) A np.uint8 numpy array of shape [N, height,\n        width]. If not None, the scores will be computed based on masks.\n      groundtruth_masks: (optional) A np.uint8 numpy array of shape [M, height,\n        width].\n\n    Returns:\n      is_class_correctly_detected_in_image: An integer 1 or 0 denoting whether a\n          class is correctly detected in the image or not\n    '
        if detected_boxes.size > 0:
            if groundtruth_boxes.size > 0:
                max_score_id = np.argmax(detected_scores)
                mask_mode = False
                if detected_masks is not None and groundtruth_masks is not None:
                    mask_mode = True
                if mask_mode:
                    detected_boxlist = np_box_mask_list.BoxMaskList(box_data=np.expand_dims(detected_boxes[max_score_id], axis=0), mask_data=np.expand_dims(detected_masks[max_score_id], axis=0))
                    gt_boxlist = np_box_mask_list.BoxMaskList(box_data=groundtruth_boxes, mask_data=groundtruth_masks)
                    iou = np_box_mask_list_ops.iou(detected_boxlist, gt_boxlist)
                else:
                    detected_boxlist = np_box_list.BoxList(np.expand_dims(detected_boxes[max_score_id, :], axis=0))
                    gt_boxlist = np_box_list.BoxList(groundtruth_boxes)
                    iou = np_box_list_ops.iou(detected_boxlist, gt_boxlist)
                if np.max(iou) >= self.matching_iou_threshold:
                    return 1
        return 0

    def _compute_tp_fp(self, detected_boxes, detected_scores, detected_class_labels, groundtruth_boxes, groundtruth_class_labels, groundtruth_is_difficult_list, groundtruth_is_group_of_list, detected_masks=None, groundtruth_masks=None):
        if False:
            i = 10
            return i + 15
        'Labels true/false positives of detections of an image across all classes.\n\n    Args:\n      detected_boxes: A float numpy array of shape [N, 4], representing N\n        regions of detected object regions. Each row is of the format [y_min,\n        x_min, y_max, x_max]\n      detected_scores: A float numpy array of shape [N, 1], representing the\n        confidence scores of the detected N object instances.\n      detected_class_labels: A integer numpy array of shape [N, 1], repreneting\n        the class labels of the detected N object instances.\n      groundtruth_boxes: A float numpy array of shape [M, 4], representing M\n        regions of object instances in ground truth\n      groundtruth_class_labels: An integer numpy array of shape [M, 1],\n        representing M class labels of object instances in ground truth\n      groundtruth_is_difficult_list: A boolean numpy array of length M denoting\n        whether a ground truth box is a difficult instance or not\n      groundtruth_is_group_of_list: A boolean numpy array of length M denoting\n        whether a ground truth box has group-of tag\n      detected_masks: (optional) A np.uint8 numpy array of shape [N, height,\n        width]. If not None, the scores will be computed based on masks.\n      groundtruth_masks: (optional) A np.uint8 numpy array of shape [M, height,\n        width].\n\n    Returns:\n      result_scores: A list of float numpy arrays. Each numpy array is of\n          shape [K, 1], representing K scores detected with object class\n          label c\n      result_tp_fp_labels: A list of boolean numpy array. Each numpy array is of\n          shape [K, 1], representing K True/False positive label of object\n          instances detected with class label c\n\n    Raises:\n      ValueError: If detected masks is not None but groundtruth masks are None,\n        or the other way around.\n    '
        if detected_masks is not None and groundtruth_masks is None:
            raise ValueError('Detected masks is available but groundtruth masks is not.')
        if detected_masks is None and groundtruth_masks is not None:
            raise ValueError('Groundtruth masks is available but detected masks is not.')
        result_scores = []
        result_tp_fp_labels = []
        for i in range(self.num_groundtruth_classes):
            groundtruth_is_difficult_list_at_ith_class = groundtruth_is_difficult_list[groundtruth_class_labels == i]
            groundtruth_is_group_of_list_at_ith_class = groundtruth_is_group_of_list[groundtruth_class_labels == i]
            (gt_boxes_at_ith_class, gt_masks_at_ith_class, detected_boxes_at_ith_class, detected_scores_at_ith_class, detected_masks_at_ith_class) = self._get_ith_class_arrays(detected_boxes, detected_scores, detected_masks, detected_class_labels, groundtruth_boxes, groundtruth_masks, groundtruth_class_labels, i)
            (scores, tp_fp_labels) = self._compute_tp_fp_for_single_class(detected_boxes=detected_boxes_at_ith_class, detected_scores=detected_scores_at_ith_class, groundtruth_boxes=gt_boxes_at_ith_class, groundtruth_is_difficult_list=groundtruth_is_difficult_list_at_ith_class, groundtruth_is_group_of_list=groundtruth_is_group_of_list_at_ith_class, detected_masks=detected_masks_at_ith_class, groundtruth_masks=gt_masks_at_ith_class)
            result_scores.append(scores)
            result_tp_fp_labels.append(tp_fp_labels)
        return (result_scores, result_tp_fp_labels)

    def _get_overlaps_and_scores_mask_mode(self, detected_boxes, detected_scores, detected_masks, groundtruth_boxes, groundtruth_masks, groundtruth_is_group_of_list):
        if False:
            for i in range(10):
                print('nop')
        'Computes overlaps and scores between detected and groudntruth masks.\n\n    Args:\n      detected_boxes: A numpy array of shape [N, 4] representing detected box\n        coordinates\n      detected_scores: A 1-d numpy array of length N representing classification\n        score\n      detected_masks: A uint8 numpy array of shape [N, height, width]. If not\n        None, the scores will be computed based on masks.\n      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth\n        box coordinates\n      groundtruth_masks: A uint8 numpy array of shape [M, height, width].\n      groundtruth_is_group_of_list: A boolean numpy array of length M denoting\n        whether a ground truth box has group-of tag. If a groundtruth box is\n        group-of box, every detection matching this box is ignored.\n\n    Returns:\n      iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If\n          gt_non_group_of_boxlist.num_boxes() == 0 it will be None.\n      ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If\n          gt_group_of_boxlist.num_boxes() == 0 it will be None.\n      scores: The score of the detected boxlist.\n      num_boxes: Number of non-maximum suppressed detected boxes.\n    '
        detected_boxlist = np_box_mask_list.BoxMaskList(box_data=detected_boxes, mask_data=detected_masks)
        detected_boxlist.add_field('scores', detected_scores)
        detected_boxlist = np_box_mask_list_ops.non_max_suppression(detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
        gt_non_group_of_boxlist = np_box_mask_list.BoxMaskList(box_data=groundtruth_boxes[~groundtruth_is_group_of_list], mask_data=groundtruth_masks[~groundtruth_is_group_of_list])
        gt_group_of_boxlist = np_box_mask_list.BoxMaskList(box_data=groundtruth_boxes[groundtruth_is_group_of_list], mask_data=groundtruth_masks[groundtruth_is_group_of_list])
        iou = np_box_mask_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
        ioa = np.transpose(np_box_mask_list_ops.ioa(gt_group_of_boxlist, detected_boxlist))
        scores = detected_boxlist.get_field('scores')
        num_boxes = detected_boxlist.num_boxes()
        return (iou, ioa, scores, num_boxes)

    def _get_overlaps_and_scores_box_mode(self, detected_boxes, detected_scores, groundtruth_boxes, groundtruth_is_group_of_list):
        if False:
            print('Hello World!')
        'Computes overlaps and scores between detected and groudntruth boxes.\n\n    Args:\n      detected_boxes: A numpy array of shape [N, 4] representing detected box\n        coordinates\n      detected_scores: A 1-d numpy array of length N representing classification\n        score\n      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth\n        box coordinates\n      groundtruth_is_group_of_list: A boolean numpy array of length M denoting\n        whether a ground truth box has group-of tag. If a groundtruth box is\n        group-of box, every detection matching this box is ignored.\n\n    Returns:\n      iou: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If\n          gt_non_group_of_boxlist.num_boxes() == 0 it will be None.\n      ioa: A float numpy array of size [num_detected_boxes, num_gt_boxes]. If\n          gt_group_of_boxlist.num_boxes() == 0 it will be None.\n      scores: The score of the detected boxlist.\n      num_boxes: Number of non-maximum suppressed detected boxes.\n    '
        detected_boxlist = np_box_list.BoxList(detected_boxes)
        detected_boxlist.add_field('scores', detected_scores)
        detected_boxlist = np_box_list_ops.non_max_suppression(detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
        gt_non_group_of_boxlist = np_box_list.BoxList(groundtruth_boxes[~groundtruth_is_group_of_list])
        gt_group_of_boxlist = np_box_list.BoxList(groundtruth_boxes[groundtruth_is_group_of_list])
        iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
        ioa = np.transpose(np_box_list_ops.ioa(gt_group_of_boxlist, detected_boxlist))
        scores = detected_boxlist.get_field('scores')
        num_boxes = detected_boxlist.num_boxes()
        return (iou, ioa, scores, num_boxes)

    def _compute_tp_fp_for_single_class(self, detected_boxes, detected_scores, groundtruth_boxes, groundtruth_is_difficult_list, groundtruth_is_group_of_list, detected_masks=None, groundtruth_masks=None):
        if False:
            for i in range(10):
                print('nop')
        'Labels boxes detected with the same class from the same image as tp/fp.\n\n    Args:\n      detected_boxes: A numpy array of shape [N, 4] representing detected box\n        coordinates\n      detected_scores: A 1-d numpy array of length N representing classification\n        score\n      groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth\n        box coordinates\n      groundtruth_is_difficult_list: A boolean numpy array of length M denoting\n        whether a ground truth box is a difficult instance or not. If a\n        groundtruth box is difficult, every detection matching this box is\n        ignored.\n      groundtruth_is_group_of_list: A boolean numpy array of length M denoting\n        whether a ground truth box has group-of tag. If a groundtruth box is\n        group-of box, every detection matching this box is ignored.\n      detected_masks: (optional) A uint8 numpy array of shape [N, height,\n        width]. If not None, the scores will be computed based on masks.\n      groundtruth_masks: (optional) A uint8 numpy array of shape [M, height,\n        width].\n\n    Returns:\n      Two arrays of the same size, containing all boxes that were evaluated as\n      being true positives or false positives; if a box matched to a difficult\n      box or to a group-of box, it is ignored.\n\n      scores: A numpy array representing the detection scores.\n      tp_fp_labels: a boolean numpy array indicating whether a detection is a\n          true positive.\n    '
        if detected_boxes.size == 0:
            return (np.array([], dtype=float), np.array([], dtype=bool))
        mask_mode = False
        if detected_masks is not None and groundtruth_masks is not None:
            mask_mode = True
        iou = np.ndarray([0, 0])
        ioa = np.ndarray([0, 0])
        iou_mask = np.ndarray([0, 0])
        ioa_mask = np.ndarray([0, 0])
        if mask_mode:
            mask_presence_indicator = np.sum(groundtruth_masks, axis=(1, 2)) > 0
            (iou_mask, ioa_mask, scores, num_detected_boxes) = self._get_overlaps_and_scores_mask_mode(detected_boxes=detected_boxes, detected_scores=detected_scores, detected_masks=detected_masks, groundtruth_boxes=groundtruth_boxes[mask_presence_indicator, :], groundtruth_masks=groundtruth_masks[mask_presence_indicator, :], groundtruth_is_group_of_list=groundtruth_is_group_of_list[mask_presence_indicator])
            if sum(mask_presence_indicator) < len(mask_presence_indicator):
                (iou, ioa, _, num_detected_boxes) = self._get_overlaps_and_scores_box_mode(detected_boxes=detected_boxes, detected_scores=detected_scores, groundtruth_boxes=groundtruth_boxes[~mask_presence_indicator, :], groundtruth_is_group_of_list=groundtruth_is_group_of_list[~mask_presence_indicator])
            num_detected_boxes = detected_boxes.shape[0]
        else:
            mask_presence_indicator = np.zeros(groundtruth_is_group_of_list.shape, dtype=bool)
            (iou, ioa, scores, num_detected_boxes) = self._get_overlaps_and_scores_box_mode(detected_boxes=detected_boxes, detected_scores=detected_scores, groundtruth_boxes=groundtruth_boxes, groundtruth_is_group_of_list=groundtruth_is_group_of_list)
        if groundtruth_boxes.size == 0:
            return (scores, np.zeros(num_detected_boxes, dtype=bool))
        tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)
        is_matched_to_box = np.zeros(num_detected_boxes, dtype=bool)
        is_matched_to_difficult = np.zeros(num_detected_boxes, dtype=bool)
        is_matched_to_group_of = np.zeros(num_detected_boxes, dtype=bool)

        def compute_match_iou(iou, groundtruth_nongroup_of_is_difficult_list, is_box):
            if False:
                i = 10
                return i + 15
            'Computes TP/FP for non group-of box matching.\n\n      The function updates the following local variables:\n        tp_fp_labels - if a box is matched to group-of\n        is_matched_to_difficult - the detections that were processed at this are\n          matched to difficult box.\n        is_matched_to_box - the detections that were processed at this stage are\n          marked as is_box.\n\n      Args:\n        iou: intersection-over-union matrix [num_gt_boxes]x[num_det_boxes].\n        groundtruth_nongroup_of_is_difficult_list: boolean that specifies if gt\n          box is difficult.\n        is_box: boolean that specifies if currently boxes or masks are\n          processed.\n      '
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            is_gt_detected = np.zeros(iou.shape[1], dtype=bool)
            for i in range(num_detected_boxes):
                gt_id = max_overlap_gt_ids[i]
                is_evaluatable = not tp_fp_labels[i] and (not is_matched_to_difficult[i]) and (iou[i, gt_id] >= self.matching_iou_threshold) and (not is_matched_to_group_of[i])
                if is_evaluatable:
                    if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
                        if not is_gt_detected[gt_id]:
                            tp_fp_labels[i] = True
                            is_gt_detected[gt_id] = True
                            is_matched_to_box[i] = is_box
                    else:
                        is_matched_to_difficult[i] = True

        def compute_match_ioa(ioa, is_box):
            if False:
                i = 10
                return i + 15
            'Computes TP/FP for group-of box matching.\n\n      The function updates the following local variables:\n        is_matched_to_group_of - if a box is matched to group-of\n        is_matched_to_box - the detections that were processed at this stage are\n          marked as is_box.\n\n      Args:\n        ioa: intersection-over-area matrix [num_gt_boxes]x[num_det_boxes].\n        is_box: boolean that specifies if currently boxes or masks are\n          processed.\n\n      Returns:\n        scores_group_of: of detections matched to group-of boxes\n        [num_groupof_matched].\n        tp_fp_labels_group_of: boolean array of size [num_groupof_matched], all\n          values are True.\n      '
            scores_group_of = np.zeros(ioa.shape[1], dtype=float)
            tp_fp_labels_group_of = self.group_of_weight * np.ones(ioa.shape[1], dtype=float)
            max_overlap_group_of_gt_ids = np.argmax(ioa, axis=1)
            for i in range(num_detected_boxes):
                gt_id = max_overlap_group_of_gt_ids[i]
                is_evaluatable = not tp_fp_labels[i] and (not is_matched_to_difficult[i]) and (ioa[i, gt_id] >= self.matching_iou_threshold) and (not is_matched_to_group_of[i])
                if is_evaluatable:
                    is_matched_to_group_of[i] = True
                    is_matched_to_box[i] = is_box
                    scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])
            selector = np.where((scores_group_of > 0) & (tp_fp_labels_group_of > 0))
            scores_group_of = scores_group_of[selector]
            tp_fp_labels_group_of = tp_fp_labels_group_of[selector]
            return (scores_group_of, tp_fp_labels_group_of)
        if iou_mask.shape[1] > 0:
            groundtruth_is_difficult_mask_list = groundtruth_is_difficult_list[mask_presence_indicator]
            groundtruth_is_group_of_mask_list = groundtruth_is_group_of_list[mask_presence_indicator]
            compute_match_iou(iou_mask, groundtruth_is_difficult_mask_list[~groundtruth_is_group_of_mask_list], is_box=False)
        scores_mask_group_of = np.ndarray([0], dtype=float)
        tp_fp_labels_mask_group_of = np.ndarray([0], dtype=float)
        if ioa_mask.shape[1] > 0:
            (scores_mask_group_of, tp_fp_labels_mask_group_of) = compute_match_ioa(ioa_mask, is_box=False)
        if iou.shape[1] > 0:
            groundtruth_is_difficult_box_list = groundtruth_is_difficult_list[~mask_presence_indicator]
            groundtruth_is_group_of_box_list = groundtruth_is_group_of_list[~mask_presence_indicator]
            compute_match_iou(iou, groundtruth_is_difficult_box_list[~groundtruth_is_group_of_box_list], is_box=True)
        scores_box_group_of = np.ndarray([0], dtype=float)
        tp_fp_labels_box_group_of = np.ndarray([0], dtype=float)
        if ioa.shape[1] > 0:
            (scores_box_group_of, tp_fp_labels_box_group_of) = compute_match_ioa(ioa, is_box=True)
        if mask_mode:
            valid_entries = ~is_matched_to_difficult & ~is_matched_to_group_of & ~is_matched_to_box
            return (np.concatenate((scores[valid_entries], scores_mask_group_of)), np.concatenate((tp_fp_labels[valid_entries].astype(float), tp_fp_labels_mask_group_of)))
        else:
            valid_entries = ~is_matched_to_difficult & ~is_matched_to_group_of
            return (np.concatenate((scores[valid_entries], scores_box_group_of)), np.concatenate((tp_fp_labels[valid_entries].astype(float), tp_fp_labels_box_group_of)))

    def _get_ith_class_arrays(self, detected_boxes, detected_scores, detected_masks, detected_class_labels, groundtruth_boxes, groundtruth_masks, groundtruth_class_labels, class_index):
        if False:
            print('Hello World!')
        'Returns numpy arrays belonging to class with index `class_index`.\n\n    Args:\n      detected_boxes: A numpy array containing detected boxes.\n      detected_scores: A numpy array containing detected scores.\n      detected_masks: A numpy array containing detected masks.\n      detected_class_labels: A numpy array containing detected class labels.\n      groundtruth_boxes: A numpy array containing groundtruth boxes.\n      groundtruth_masks: A numpy array containing groundtruth masks.\n      groundtruth_class_labels: A numpy array containing groundtruth class\n        labels.\n      class_index: An integer index.\n\n    Returns:\n      gt_boxes_at_ith_class: A numpy array containing groundtruth boxes labeled\n        as ith class.\n      gt_masks_at_ith_class: A numpy array containing groundtruth masks labeled\n        as ith class.\n      detected_boxes_at_ith_class: A numpy array containing detected boxes\n        corresponding to the ith class.\n      detected_scores_at_ith_class: A numpy array containing detected scores\n        corresponding to the ith class.\n      detected_masks_at_ith_class: A numpy array containing detected masks\n        corresponding to the ith class.\n    '
        selected_groundtruth = groundtruth_class_labels == class_index
        gt_boxes_at_ith_class = groundtruth_boxes[selected_groundtruth]
        if groundtruth_masks is not None:
            gt_masks_at_ith_class = groundtruth_masks[selected_groundtruth]
        else:
            gt_masks_at_ith_class = None
        selected_detections = detected_class_labels == class_index
        detected_boxes_at_ith_class = detected_boxes[selected_detections]
        detected_scores_at_ith_class = detected_scores[selected_detections]
        if detected_masks is not None:
            detected_masks_at_ith_class = detected_masks[selected_detections]
        else:
            detected_masks_at_ith_class = None
        return (gt_boxes_at_ith_class, gt_masks_at_ith_class, detected_boxes_at_ith_class, detected_scores_at_ith_class, detected_masks_at_ith_class)

    def _remove_invalid_boxes(self, detected_boxes, detected_scores, detected_class_labels, detected_masks=None):
        if False:
            while True:
                i = 10
        'Removes entries with invalid boxes.\n\n    A box is invalid if either its xmax is smaller than its xmin, or its ymax\n    is smaller than its ymin.\n\n    Args:\n      detected_boxes: A float numpy array of size [num_boxes, 4] containing box\n        coordinates in [ymin, xmin, ymax, xmax] format.\n      detected_scores: A float numpy array of size [num_boxes].\n      detected_class_labels: A int32 numpy array of size [num_boxes].\n      detected_masks: A uint8 numpy array of size [num_boxes, height, width].\n\n    Returns:\n      valid_detected_boxes: A float numpy array of size [num_valid_boxes, 4]\n        containing box coordinates in [ymin, xmin, ymax, xmax] format.\n      valid_detected_scores: A float numpy array of size [num_valid_boxes].\n      valid_detected_class_labels: A int32 numpy array of size\n        [num_valid_boxes].\n      valid_detected_masks: A uint8 numpy array of size\n        [num_valid_boxes, height, width].\n    '
        valid_indices = np.logical_and(detected_boxes[:, 0] < detected_boxes[:, 2], detected_boxes[:, 1] < detected_boxes[:, 3])
        detected_boxes = detected_boxes[valid_indices]
        detected_scores = detected_scores[valid_indices]
        detected_class_labels = detected_class_labels[valid_indices]
        if detected_masks is not None:
            detected_masks = detected_masks[valid_indices]
        return [detected_boxes, detected_scores, detected_class_labels, detected_masks]