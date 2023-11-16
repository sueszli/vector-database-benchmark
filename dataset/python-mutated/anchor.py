"""Anchor box and labeler definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow.compat.v2 as tf
from official.vision.detection.utils.object_detection import argmax_matcher
from official.vision.detection.utils.object_detection import balanced_positive_negative_sampler
from official.vision.detection.utils.object_detection import box_list
from official.vision.detection.utils.object_detection import faster_rcnn_box_coder
from official.vision.detection.utils.object_detection import region_similarity_calculator
from official.vision.detection.utils.object_detection import target_assigner

class Anchor(object):
    """Anchor class for anchor-based object detectors."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_size, image_size):
        if False:
            while True:
                i = 10
        'Constructs multiscale anchors.\n\n    Args:\n      min_level: integer number of minimum level of the output feature pyramid.\n      max_level: integer number of maximum level of the output feature pyramid.\n      num_scales: integer number representing intermediate scales added\n        on each level. For instances, num_scales=2 adds one additional\n        intermediate anchor scales [2^0, 2^0.5] on each level.\n      aspect_ratios: list of float numbers representing the aspect raito anchors\n        added on each level. The number indicates the ratio of width to height.\n        For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each\n        scale level.\n      anchor_size: float number representing the scale of size of the base\n        anchor to the feature stride 2^level.\n      image_size: a list of integer numbers or Tensors representing\n        [height, width] of the input image size.The image_size should be divided\n        by the largest feature stride 2^max_level.\n    '
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size
        self.image_size = image_size
        self.boxes = self._generate_boxes()

    def _generate_boxes(self):
        if False:
            return 10
        'Generates multiscale anchor boxes.\n\n    Returns:\n      a Tensor of shape [N, 4], represneting anchor boxes of all levels\n      concatenated together.\n    '
        boxes_all = []
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            for scale in range(self.num_scales):
                for aspect_ratio in self.aspect_ratios:
                    stride = 2 ** level
                    intermidate_scale = 2 ** (scale / float(self.num_scales))
                    base_anchor_size = self.anchor_size * stride * intermidate_scale
                    aspect_x = aspect_ratio ** 0.5
                    aspect_y = aspect_ratio ** (-0.5)
                    half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                    half_anchor_size_y = base_anchor_size * aspect_y / 2.0
                    x = tf.range(stride / 2, self.image_size[1], stride)
                    y = tf.range(stride / 2, self.image_size[0], stride)
                    (xv, yv) = tf.meshgrid(x, y)
                    xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
                    yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
                    boxes = tf.stack([yv - half_anchor_size_y, xv - half_anchor_size_x, yv + half_anchor_size_y, xv + half_anchor_size_x], axis=1)
                    boxes_l.append(boxes)
            boxes_l = tf.stack(boxes_l, axis=1)
            boxes_l = tf.reshape(boxes_l, [-1, 4])
            boxes_all.append(boxes_l)
        return tf.concat(boxes_all, axis=0)

    def unpack_labels(self, labels):
        if False:
            return 10
        'Unpacks an array of labels into multiscales labels.'
        unpacked_labels = collections.OrderedDict()
        count = 0
        for level in range(self.min_level, self.max_level + 1):
            feat_size_y = tf.cast(self.image_size[0] / 2 ** level, tf.int32)
            feat_size_x = tf.cast(self.image_size[1] / 2 ** level, tf.int32)
            steps = feat_size_y * feat_size_x * self.anchors_per_location
            unpacked_labels[level] = tf.reshape(labels[count:count + steps], [feat_size_y, feat_size_x, -1])
            count += steps
        return unpacked_labels

    @property
    def anchors_per_location(self):
        if False:
            return 10
        return self.num_scales * len(self.aspect_ratios)

    @property
    def multilevel_boxes(self):
        if False:
            return 10
        return self.unpack_labels(self.boxes)

class AnchorLabeler(object):
    """Labeler for dense object detector."""

    def __init__(self, anchor, match_threshold=0.5, unmatched_threshold=0.5):
        if False:
            while True:
                i = 10
        'Constructs anchor labeler to assign labels to anchors.\n\n    Args:\n      anchor: an instance of class Anchors.\n      match_threshold: a float number between 0 and 1 representing the\n        lower-bound threshold to assign positive labels for anchors. An anchor\n        with a score over the threshold is labeled positive.\n      unmatched_threshold: a float number between 0 and 1 representing the\n        upper-bound threshold to assign negative labels for anchors. An anchor\n        with a score below the threshold is labeled negative.\n    '
        similarity_calc = region_similarity_calculator.IouSimilarity()
        matcher = argmax_matcher.ArgMaxMatcher(match_threshold, unmatched_threshold=unmatched_threshold, negatives_lower_than_unmatched=True, force_match_for_each_row=True)
        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
        self._target_assigner = target_assigner.TargetAssigner(similarity_calc, matcher, box_coder)
        self._anchor = anchor
        self._match_threshold = match_threshold
        self._unmatched_threshold = unmatched_threshold

    def label_anchors(self, gt_boxes, gt_labels):
        if False:
            print('Hello World!')
        'Labels anchors with ground truth inputs.\n\n    Args:\n      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.\n        For each row, it stores [y0, x0, y1, x1] for four corners of a box.\n      gt_labels: A integer tensor with shape [N, 1] representing groundtruth\n        classes.\n    Returns:\n      cls_targets_dict: ordered dictionary with keys\n        [min_level, min_level+1, ..., max_level]. The values are tensor with\n        shape [height_l, width_l, num_anchors_per_location]. The height_l and\n        width_l represent the dimension of class logits at l-th level.\n      box_targets_dict: ordered dictionary with keys\n        [min_level, min_level+1, ..., max_level]. The values are tensor with\n        shape [height_l, width_l, num_anchors_per_location * 4]. The height_l\n        and width_l represent the dimension of bounding box regression output at\n        l-th level.\n      num_positives: scalar tensor storing number of positives in an image.\n    '
        gt_box_list = box_list.BoxList(gt_boxes)
        anchor_box_list = box_list.BoxList(self._anchor.boxes)
        (cls_targets, _, box_targets, _, matches) = self._target_assigner.assign(anchor_box_list, gt_box_list, gt_labels)
        match_results = tf.expand_dims(matches.match_results, axis=1)
        cls_targets = tf.cast(cls_targets, tf.int32)
        cls_targets = tf.where(tf.equal(match_results, -1), -tf.ones_like(cls_targets), cls_targets)
        cls_targets = tf.where(tf.equal(match_results, -2), -2 * tf.ones_like(cls_targets), cls_targets)
        cls_targets_dict = self._anchor.unpack_labels(cls_targets)
        box_targets_dict = self._anchor.unpack_labels(box_targets)
        num_positives = tf.reduce_sum(input_tensor=tf.cast(tf.greater(matches.match_results, -1), tf.float32))
        return (cls_targets_dict, box_targets_dict, num_positives)

class RpnAnchorLabeler(AnchorLabeler):
    """Labeler for Region Proposal Network."""

    def __init__(self, anchor, match_threshold=0.7, unmatched_threshold=0.3, rpn_batch_size_per_im=256, rpn_fg_fraction=0.5):
        if False:
            return 10
        AnchorLabeler.__init__(self, anchor, match_threshold=0.7, unmatched_threshold=0.3)
        self._rpn_batch_size_per_im = rpn_batch_size_per_im
        self._rpn_fg_fraction = rpn_fg_fraction

    def _get_rpn_samples(self, match_results):
        if False:
            i = 10
            return i + 15
        "Computes anchor labels.\n\n    This function performs subsampling for foreground (fg) and background (bg)\n    anchors.\n    Args:\n      match_results: A integer tensor with shape [N] representing the\n        matching results of anchors. (1) match_results[i]>=0,\n        meaning that column i is matched with row match_results[i].\n        (2) match_results[i]=-1, meaning that column i is not matched.\n        (3) match_results[i]=-2, meaning that column i is ignored.\n    Returns:\n      score_targets: a integer tensor with the a shape of [N].\n        (1) score_targets[i]=1, the anchor is a positive sample.\n        (2) score_targets[i]=0, negative. (3) score_targets[i]=-1, the anchor is\n        don't care (ignore).\n    "
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(positive_fraction=self._rpn_fg_fraction, is_static=False)
        indicator = tf.greater(match_results, -2)
        labels = tf.greater(match_results, -1)
        samples = sampler.subsample(indicator, self._rpn_batch_size_per_im, labels)
        positive_labels = tf.where(tf.logical_and(samples, labels), tf.constant(2, dtype=tf.int32, shape=match_results.shape), tf.constant(0, dtype=tf.int32, shape=match_results.shape))
        negative_labels = tf.where(tf.logical_and(samples, tf.logical_not(labels)), tf.constant(1, dtype=tf.int32, shape=match_results.shape), tf.constant(0, dtype=tf.int32, shape=match_results.shape))
        ignore_labels = tf.fill(match_results.shape, -1)
        return (ignore_labels + positive_labels + negative_labels, positive_labels, negative_labels)

    def label_anchors(self, gt_boxes, gt_labels):
        if False:
            print('Hello World!')
        'Labels anchors with ground truth inputs.\n\n    Args:\n      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.\n        For each row, it stores [y0, x0, y1, x1] for four corners of a box.\n      gt_labels: A integer tensor with shape [N, 1] representing groundtruth\n        classes.\n    Returns:\n      score_targets_dict: ordered dictionary with keys\n        [min_level, min_level+1, ..., max_level]. The values are tensor with\n        shape [height_l, width_l, num_anchors]. The height_l and width_l\n        represent the dimension of class logits at l-th level.\n      box_targets_dict: ordered dictionary with keys\n        [min_level, min_level+1, ..., max_level]. The values are tensor with\n        shape [height_l, width_l, num_anchors * 4]. The height_l and\n        width_l represent the dimension of bounding box regression output at\n        l-th level.\n    '
        gt_box_list = box_list.BoxList(gt_boxes)
        anchor_box_list = box_list.BoxList(self._anchor.boxes)
        (_, _, box_targets, _, matches) = self._target_assigner.assign(anchor_box_list, gt_box_list, gt_labels)
        (score_targets, _, _) = self._get_rpn_samples(matches.match_results)
        score_targets_dict = self._anchor.unpack_labels(score_targets)
        box_targets_dict = self._anchor.unpack_labels(box_targets)
        return (score_targets_dict, box_targets_dict)