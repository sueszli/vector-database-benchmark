"""Base target assigner module.

The job of a TargetAssigner is, for a given set of anchors (bounding boxes) and
groundtruth detections (bounding boxes), to assign classification and regression
targets to each anchor as well as weights to each anchor (specifying, e.g.,
which anchors should not contribute to training loss).

It assigns classification/regression targets by performing the following steps:
1) Computing pairwise similarity between anchors and groundtruth boxes using a
  provided RegionSimilarity Calculator
2) Computing a matching based on the similarity matrix using a provided Matcher
3) Assigning regression targets based on the matching and a provided BoxCoder
4) Assigning classification targets based on the matching and groundtruth labels

Note that TargetAssigners only operate on detections from a single
image at a time, so any logic for applying a TargetAssigner to multiple
images must be handled externally.
"""
import tensorflow.compat.v2 as tf
from official.vision.detection.utils.object_detection import box_list
from official.vision.detection.utils.object_detection import shape_utils
KEYPOINTS_FIELD_NAME = 'keypoints'

class TargetAssigner(object):
    """Target assigner to compute classification and regression targets."""

    def __init__(self, similarity_calc, matcher, box_coder, negative_class_weight=1.0, unmatched_cls_target=None):
        if False:
            for i in range(10):
                print('nop')
        'Construct Object Detection Target Assigner.\n\n    Args:\n      similarity_calc: a RegionSimilarityCalculator\n      matcher: Matcher used to match groundtruth to anchors.\n      box_coder: BoxCoder used to encode matching groundtruth boxes with\n        respect to anchors.\n      negative_class_weight: classification weight to be associated to negative\n        anchors (default: 1.0). The weight must be in [0., 1.].\n      unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]\n        which is consistent with the classification target for each\n        anchor (and can be empty for scalar targets).  This shape must thus be\n        compatible with the groundtruth labels that are passed to the "assign"\n        function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).\n        If set to None, unmatched_cls_target is set to be [0] for each anchor.\n\n    Raises:\n      ValueError: if similarity_calc is not a RegionSimilarityCalculator or\n        if matcher is not a Matcher or if box_coder is not a BoxCoder\n    '
        self._similarity_calc = similarity_calc
        self._matcher = matcher
        self._box_coder = box_coder
        self._negative_class_weight = negative_class_weight
        if unmatched_cls_target is None:
            self._unmatched_cls_target = tf.constant([0], tf.float32)
        else:
            self._unmatched_cls_target = unmatched_cls_target

    @property
    def box_coder(self):
        if False:
            return 10
        return self._box_coder

    def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None, groundtruth_weights=None, **params):
        if False:
            while True:
                i = 10
        'Assign classification and regression targets to each anchor.\n\n    For a given set of anchors and groundtruth detections, match anchors\n    to groundtruth_boxes and assign classification and regression targets to\n    each anchor as well as weights based on the resulting match (specifying,\n    e.g., which anchors should not contribute to training loss).\n\n    Anchors that are not matched to anything are given a classification target\n    of self._unmatched_cls_target which can be specified via the constructor.\n\n    Args:\n      anchors: a BoxList representing N anchors\n      groundtruth_boxes: a BoxList representing M groundtruth boxes\n      groundtruth_labels:  a tensor of shape [M, d_1, ... d_k]\n        with labels for each of the ground_truth boxes. The subshape\n        [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set\n        to None, groundtruth_labels assumes a binary problem where all\n        ground_truth boxes get a positive label (of 1).\n      groundtruth_weights: a float tensor of shape [M] indicating the weight to\n        assign to all anchors match to a particular groundtruth box. The weights\n        must be in [0., 1.]. If None, all weights are set to 1.\n      **params: Additional keyword arguments for specific implementations of\n              the Matcher.\n\n    Returns:\n      cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],\n        where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels\n        which has shape [num_gt_boxes, d_1, d_2, ... d_k].\n      cls_weights: a float32 tensor with shape [num_anchors]\n      reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]\n      reg_weights: a float32 tensor with shape [num_anchors]\n      match: a matcher.Match object encoding the match between anchors and\n        groundtruth boxes, with rows corresponding to groundtruth boxes\n        and columns corresponding to anchors.\n\n    Raises:\n      ValueError: if anchors or groundtruth_boxes are not of type\n        box_list.BoxList\n    '
        if not isinstance(anchors, box_list.BoxList):
            raise ValueError('anchors must be an BoxList')
        if not isinstance(groundtruth_boxes, box_list.BoxList):
            raise ValueError('groundtruth_boxes must be an BoxList')
        if groundtruth_labels is None:
            groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(), 0))
            groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)
        unmatched_shape_assert = shape_utils.assert_shape_equal(shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[1:], shape_utils.combined_static_and_dynamic_shape(self._unmatched_cls_target))
        labels_and_box_shapes_assert = shape_utils.assert_shape_equal(shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[:1], shape_utils.combined_static_and_dynamic_shape(groundtruth_boxes.get())[:1])
        if groundtruth_weights is None:
            num_gt_boxes = groundtruth_boxes.num_boxes_static()
            if not num_gt_boxes:
                num_gt_boxes = groundtruth_boxes.num_boxes()
            groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)
        with tf.control_dependencies([unmatched_shape_assert, labels_and_box_shapes_assert]):
            match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes, anchors)
            match = self._matcher.match(match_quality_matrix, **params)
            reg_targets = self._create_regression_targets(anchors, groundtruth_boxes, match)
            cls_targets = self._create_classification_targets(groundtruth_labels, match)
            reg_weights = self._create_regression_weights(match, groundtruth_weights)
            cls_weights = self._create_classification_weights(match, groundtruth_weights)
        num_anchors = anchors.num_boxes_static()
        if num_anchors is not None:
            reg_targets = self._reset_target_shape(reg_targets, num_anchors)
            cls_targets = self._reset_target_shape(cls_targets, num_anchors)
            reg_weights = self._reset_target_shape(reg_weights, num_anchors)
            cls_weights = self._reset_target_shape(cls_weights, num_anchors)
        return (cls_targets, cls_weights, reg_targets, reg_weights, match)

    def _reset_target_shape(self, target, num_anchors):
        if False:
            print('Hello World!')
        "Sets the static shape of the target.\n\n    Args:\n      target: the target tensor. Its first dimension will be overwritten.\n      num_anchors: the number of anchors, which is used to override the target's\n        first dimension.\n\n    Returns:\n      A tensor with the shape info filled in.\n    "
        target_shape = target.get_shape().as_list()
        target_shape[0] = num_anchors
        target.set_shape(target_shape)
        return target

    def _create_regression_targets(self, anchors, groundtruth_boxes, match):
        if False:
            for i in range(10):
                print('nop')
        'Returns a regression target for each anchor.\n\n    Args:\n      anchors: a BoxList representing N anchors\n      groundtruth_boxes: a BoxList representing M groundtruth_boxes\n      match: a matcher.Match object\n\n    Returns:\n      reg_targets: a float32 tensor with shape [N, box_code_dimension]\n    '
        matched_gt_boxes = match.gather_based_on_match(groundtruth_boxes.get(), unmatched_value=tf.zeros(4), ignored_value=tf.zeros(4))
        matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
        if groundtruth_boxes.has_field(KEYPOINTS_FIELD_NAME):
            groundtruth_keypoints = groundtruth_boxes.get_field(KEYPOINTS_FIELD_NAME)
            matched_keypoints = match.gather_based_on_match(groundtruth_keypoints, unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]), ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
            matched_gt_boxlist.add_field(KEYPOINTS_FIELD_NAME, matched_keypoints)
        matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
        match_results_shape = shape_utils.combined_static_and_dynamic_shape(match.match_results)
        unmatched_ignored_reg_targets = tf.tile(self._default_regression_target(), [match_results_shape[0], 1])
        matched_anchors_mask = match.matched_column_indicator()
        matched_anchors_mask = tf.tile(tf.expand_dims(matched_anchors_mask, 1), [1, tf.shape(matched_reg_targets)[1]])
        reg_targets = tf.where(matched_anchors_mask, matched_reg_targets, unmatched_ignored_reg_targets)
        return reg_targets

    def _default_regression_target(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the default target for anchors to regress to.\n\n    Default regression targets are set to zero (though in\n    this implementation what these targets are set to should\n    not matter as the regression weight of any box set to\n    regress to the default target is zero).\n\n    Returns:\n      default_target: a float32 tensor with shape [1, box_code_dimension]\n    '
        return tf.constant([self._box_coder.code_size * [0]], tf.float32)

    def _create_classification_targets(self, groundtruth_labels, match):
        if False:
            return 10
        'Create classification targets for each anchor.\n\n    Assign a classification target of for each anchor to the matching\n    groundtruth label that is provided by match.  Anchors that are not matched\n    to anything are given the target self._unmatched_cls_target\n\n    Args:\n      groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]\n        with labels for each of the ground_truth boxes. The subshape\n        [d_1, ... d_k] can be empty (corresponding to scalar labels).\n      match: a matcher.Match object that provides a matching between anchors\n        and groundtruth boxes.\n\n    Returns:\n      a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the\n      subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has\n      shape [num_gt_boxes, d_1, d_2, ... d_k].\n    '
        return match.gather_based_on_match(groundtruth_labels, unmatched_value=self._unmatched_cls_target, ignored_value=self._unmatched_cls_target)

    def _create_regression_weights(self, match, groundtruth_weights):
        if False:
            print('Hello World!')
        'Set regression weight for each anchor.\n\n    Only positive anchors are set to contribute to the regression loss, so this\n    method returns a weight of 1 for every positive anchor and 0 for every\n    negative anchor.\n\n    Args:\n      match: a matcher.Match object that provides a matching between anchors\n        and groundtruth boxes.\n      groundtruth_weights: a float tensor of shape [M] indicating the weight to\n        assign to all anchors match to a particular groundtruth box.\n\n    Returns:\n      a float32 tensor with shape [num_anchors] representing regression weights.\n    '
        return match.gather_based_on_match(groundtruth_weights, ignored_value=0.0, unmatched_value=0.0)

    def _create_classification_weights(self, match, groundtruth_weights):
        if False:
            i = 10
            return i + 15
        'Create classification weights for each anchor.\n\n    Positive (matched) anchors are associated with a weight of\n    positive_class_weight and negative (unmatched) anchors are associated with\n    a weight of negative_class_weight. When anchors are ignored, weights are set\n    to zero. By default, both positive/negative weights are set to 1.0,\n    but they can be adjusted to handle class imbalance (which is almost always\n    the case in object detection).\n\n    Args:\n      match: a matcher.Match object that provides a matching between anchors\n        and groundtruth boxes.\n      groundtruth_weights: a float tensor of shape [M] indicating the weight to\n        assign to all anchors match to a particular groundtruth box.\n\n    Returns:\n      a float32 tensor with shape [num_anchors] representing classification\n      weights.\n    '
        return match.gather_based_on_match(groundtruth_weights, ignored_value=0.0, unmatched_value=self._negative_class_weight)

    def get_box_coder(self):
        if False:
            print('Hello World!')
        'Get BoxCoder of this TargetAssigner.\n\n    Returns:\n      BoxCoder object.\n    '
        return self._box_coder