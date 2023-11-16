"""ROI-related ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf
from official.vision.detection.ops import nms
from official.vision.detection.utils import box_utils

def multilevel_propose_rois(rpn_boxes, rpn_scores, anchor_boxes, image_shape, rpn_pre_nms_top_k=2000, rpn_post_nms_top_k=1000, rpn_nms_threshold=0.7, rpn_score_threshold=0.0, rpn_min_size_threshold=0.0, decode_boxes=True, clip_boxes=True, use_batched_nms=False, apply_sigmoid_to_score=True):
    if False:
        return 10
    'Proposes RoIs given a group of candidates from different FPN levels.\n\n  The following describes the steps:\n    1. For each individual level:\n      a. Apply sigmoid transform if specified.\n      b. Decode boxes if specified.\n      c. Clip boxes if specified.\n      d. Filter small boxes and those fall outside image if specified.\n      e. Apply pre-NMS filtering including pre-NMS top k and score thresholding.\n      f. Apply NMS.\n    2. Aggregate post-NMS boxes from each level.\n    3. Apply an overall top k to generate the final selected RoIs.\n\n  Args:\n    rpn_boxes: a dict with keys representing FPN levels and values representing\n      box tenors of shape [batch_size, feature_h, feature_w, num_anchors * 4].\n    rpn_scores: a dict with keys representing FPN levels and values representing\n      logit tensors of shape [batch_size, feature_h, feature_w, num_anchors].\n    anchor_boxes: a dict with keys representing FPN levels and values\n      representing anchor box tensors of shape\n      [batch_size, feature_h, feature_w, num_anchors * 4].\n    image_shape: a tensor of shape [batch_size, 2] where the last dimension are\n      [height, width] of the scaled image.\n    rpn_pre_nms_top_k: an integer of top scoring RPN proposals *per level* to\n      keep before applying NMS. Default: 2000.\n    rpn_post_nms_top_k: an integer of top scoring RPN proposals *in total* to\n      keep after applying NMS. Default: 1000.\n    rpn_nms_threshold: a float between 0 and 1 representing the IoU threshold\n      used for NMS. If 0.0, no NMS is applied. Default: 0.7.\n    rpn_score_threshold: a float between 0 and 1 representing the minimal box\n      score to keep before applying NMS. This is often used as a pre-filtering\n      step for better performance. If 0, no filtering is applied. Default: 0.\n    rpn_min_size_threshold: a float representing the minimal box size in each\n      side (w.r.t. the scaled image) to keep before applying NMS. This is often\n      used as a pre-filtering step for better performance. If 0, no filtering is\n      applied. Default: 0.\n    decode_boxes: a boolean indicating whether `rpn_boxes` needs to be decoded\n      using `anchor_boxes`. If False, use `rpn_boxes` directly and ignore\n      `anchor_boxes`. Default: True.\n    clip_boxes: a boolean indicating whether boxes are first clipped to the\n      scaled image size before appliying NMS. If False, no clipping is applied\n      and `image_shape` is ignored. Default: True.\n    use_batched_nms: a boolean indicating whether NMS is applied in batch using\n      `tf.image.combined_non_max_suppression`. Currently only available in\n      CPU/GPU. Default: False.\n    apply_sigmoid_to_score: a boolean indicating whether apply sigmoid to\n      `rpn_scores` before applying NMS. Default: True.\n\n  Returns:\n    selected_rois: a tensor of shape [batch_size, rpn_post_nms_top_k, 4],\n      representing the box coordinates of the selected proposals w.r.t. the\n      scaled image.\n    selected_roi_scores: a tensor of shape [batch_size, rpn_post_nms_top_k, 1],\n      representing the scores of the selected proposals.\n  '
    with tf.name_scope('multilevel_propose_rois'):
        rois = []
        roi_scores = []
        image_shape = tf.expand_dims(image_shape, axis=1)
        for level in sorted(rpn_scores.keys()):
            with tf.name_scope('level_%d' % level):
                (_, feature_h, feature_w, num_anchors_per_location) = rpn_scores[level].get_shape().as_list()
                num_boxes = feature_h * feature_w * num_anchors_per_location
                this_level_scores = tf.reshape(rpn_scores[level], [-1, num_boxes])
                this_level_boxes = tf.reshape(rpn_boxes[level], [-1, num_boxes, 4])
                this_level_anchors = tf.cast(tf.reshape(anchor_boxes[level], [-1, num_boxes, 4]), dtype=this_level_scores.dtype)
                if apply_sigmoid_to_score:
                    this_level_scores = tf.sigmoid(this_level_scores)
                if decode_boxes:
                    this_level_boxes = box_utils.decode_boxes(this_level_boxes, this_level_anchors)
                if clip_boxes:
                    this_level_boxes = box_utils.clip_boxes(this_level_boxes, image_shape)
                if rpn_min_size_threshold > 0.0:
                    (this_level_boxes, this_level_scores) = box_utils.filter_boxes(this_level_boxes, this_level_scores, image_shape, rpn_min_size_threshold)
                this_level_pre_nms_top_k = min(num_boxes, rpn_pre_nms_top_k)
                this_level_post_nms_top_k = min(num_boxes, rpn_post_nms_top_k)
                if rpn_nms_threshold > 0.0:
                    if use_batched_nms:
                        (this_level_rois, this_level_roi_scores, _, _) = tf.image.combined_non_max_suppression(tf.expand_dims(this_level_boxes, axis=2), tf.expand_dims(this_level_scores, axis=-1), max_output_size_per_class=this_level_pre_nms_top_k, max_total_size=this_level_post_nms_top_k, iou_threshold=rpn_nms_threshold, score_threshold=rpn_score_threshold, pad_per_class=False, clip_boxes=False)
                    else:
                        if rpn_score_threshold > 0.0:
                            (this_level_boxes, this_level_scores) = box_utils.filter_boxes_by_scores(this_level_boxes, this_level_scores, rpn_score_threshold)
                        (this_level_boxes, this_level_scores) = box_utils.top_k_boxes(this_level_boxes, this_level_scores, k=this_level_pre_nms_top_k)
                        (this_level_roi_scores, this_level_rois) = nms.sorted_non_max_suppression_padded(this_level_scores, this_level_boxes, max_output_size=this_level_post_nms_top_k, iou_threshold=rpn_nms_threshold)
                else:
                    (this_level_rois, this_level_roi_scores) = box_utils.top_k_boxes(this_level_rois, this_level_scores, k=this_level_post_nms_top_k)
                rois.append(this_level_rois)
                roi_scores.append(this_level_roi_scores)
        all_rois = tf.concat(rois, axis=1)
        all_roi_scores = tf.concat(roi_scores, axis=1)
        with tf.name_scope('top_k_rois'):
            (_, num_valid_rois) = all_roi_scores.get_shape().as_list()
            overall_top_k = min(num_valid_rois, rpn_post_nms_top_k)
            (selected_rois, selected_roi_scores) = box_utils.top_k_boxes(all_rois, all_roi_scores, k=overall_top_k)
        return (selected_rois, selected_roi_scores)

class ROIGenerator(object):
    """Proposes RoIs for the second stage processing."""

    def __init__(self, params):
        if False:
            print('Hello World!')
        self._rpn_pre_nms_top_k = params.rpn_pre_nms_top_k
        self._rpn_post_nms_top_k = params.rpn_post_nms_top_k
        self._rpn_nms_threshold = params.rpn_nms_threshold
        self._rpn_score_threshold = params.rpn_score_threshold
        self._rpn_min_size_threshold = params.rpn_min_size_threshold
        self._test_rpn_pre_nms_top_k = params.test_rpn_pre_nms_top_k
        self._test_rpn_post_nms_top_k = params.test_rpn_post_nms_top_k
        self._test_rpn_nms_threshold = params.test_rpn_nms_threshold
        self._test_rpn_score_threshold = params.test_rpn_score_threshold
        self._test_rpn_min_size_threshold = params.test_rpn_min_size_threshold
        self._use_batched_nms = params.use_batched_nms

    def __call__(self, boxes, scores, anchor_boxes, image_shape, is_training):
        if False:
            while True:
                i = 10
        'Generates RoI proposals.\n\n    Args:\n      boxes: a dict with keys representing FPN levels and values representing\n        box tenors of shape [batch_size, feature_h, feature_w, num_anchors * 4].\n      scores: a dict with keys representing FPN levels and values representing\n        logit tensors of shape [batch_size, feature_h, feature_w, num_anchors].\n      anchor_boxes: a dict with keys representing FPN levels and values\n        representing anchor box tensors of shape\n        [batch_size, feature_h, feature_w, num_anchors * 4].\n      image_shape: a tensor of shape [batch_size, 2] where the last dimension\n        are [height, width] of the scaled image.\n      is_training: a bool indicating whether it is in training or inference\n        mode.\n\n    Returns:\n      proposed_rois: a tensor of shape [batch_size, rpn_post_nms_top_k, 4],\n        representing the box coordinates of the proposed RoIs w.r.t. the\n        scaled image.\n      proposed_roi_scores: a tensor of shape\n        [batch_size, rpn_post_nms_top_k, 1], representing the scores of the\n        proposed RoIs.\n\n    '
        (proposed_rois, proposed_roi_scores) = multilevel_propose_rois(boxes, scores, anchor_boxes, image_shape, rpn_pre_nms_top_k=self._rpn_pre_nms_top_k if is_training else self._test_rpn_pre_nms_top_k, rpn_post_nms_top_k=self._rpn_post_nms_top_k if is_training else self._test_rpn_post_nms_top_k, rpn_nms_threshold=self._rpn_nms_threshold if is_training else self._test_rpn_nms_threshold, rpn_score_threshold=self._rpn_score_threshold if is_training else self._test_rpn_score_threshold, rpn_min_size_threshold=self._rpn_min_size_threshold if is_training else self._test_rpn_min_size_threshold, decode_boxes=True, clip_boxes=True, use_batched_nms=self._use_batched_nms, apply_sigmoid_to_score=True)
        return (proposed_rois, proposed_roi_scores)