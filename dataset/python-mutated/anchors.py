"""Anchor definition."""
import collections
import numpy as np
import tensorflow as tf
import utils
from .anchors_utils import argmax_matcher
from .anchors_utils import box_list
from .anchors_utils import faster_rcnn_box_coder
from .anchors_utils import region_similarity_calculator
from .anchors_utils import target_assigner
MAX_DETECTION_POINTS = 5000

def decode_box_outputs(pred_boxes, anchor_boxes):
    if False:
        for i in range(10):
            print('nop')
    'Transforms relative regression coordinates to absolute positions.\n\n    Network predictions are normalized and relative to a given anchor; this\n    reverses the transformation and outputs absolute coordinates for the input\n    image.\n\n    Args:\n      pred_boxes: predicted box regression targets.\n      anchor_boxes: anchors on all feature levels.\n    Returns:\n      outputs: bounding boxes.\n    '
    anchor_boxes = tf.cast(anchor_boxes, pred_boxes.dtype)
    ycenter_a = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2
    xcenter_a = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2
    ha = anchor_boxes[..., 2] - anchor_boxes[..., 0]
    wa = anchor_boxes[..., 3] - anchor_boxes[..., 1]
    (ty, tx, th, tw) = tf.unstack(pred_boxes, num=4, axis=-1)
    w = tf.math.exp(tw) * wa
    h = tf.math.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

class Anchors:
    """Multi-scale anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size):
        if False:
            while True:
                i = 10
        'Constructs multiscale anchors.\n\n        Args:\n          min_level: integer number of minimum level of the output feature pyramid.\n          max_level: integer number of maximum level of the output feature pyramid.\n          num_scales: integer number representing intermediate scales added\n            on each level. For instances, num_scales=2 adds two additional\n            anchor scales [2^0, 2^0.5] on each level.\n          aspect_ratios: list of representing the aspect ratio anchors added\n            on each level. For instances, aspect_ratios = [1.0, 2.0, 0..5]\n            adds three anchors on each level.\n          anchor_scale: float number representing the scale of size of the base\n            anchor to the feature stride 2^level. Or a list, one value per layer.\n          image_size: integer number or tuple of integer number of input image size.\n        '
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        if isinstance(anchor_scale, (list, tuple)):
            assert len(anchor_scale) == max_level - min_level + 1
            self.anchor_scales = anchor_scale
        else:
            self.anchor_scales = [anchor_scale] * (max_level - min_level + 1)
        self.image_size = utils.parse_image_size(image_size)
        self.feat_sizes = utils.get_feat_sizes(image_size, max_level)
        self.config = self._generate_configs()
        self.boxes = self._generate_boxes()

    def _generate_configs(self):
        if False:
            while True:
                i = 10
        'Generate configurations of anchor boxes.'
        anchor_configs = {}
        feat_sizes = self.feat_sizes
        for level in range(self.min_level, self.max_level + 1):
            anchor_configs[level] = []
            for scale_octave in range(self.num_scales):
                for aspect in self.aspect_ratios:
                    anchor_configs[level].append(((feat_sizes[0]['height'] / float(feat_sizes[level]['height']), feat_sizes[0]['width'] / float(feat_sizes[level]['width'])), scale_octave / float(self.num_scales), aspect, self.anchor_scales[level - self.min_level]))
        return anchor_configs

    def _generate_boxes(self):
        if False:
            return 10
        'Generates multiscale anchor boxes.'
        boxes_all = []
        for (_, configs) in self.config.items():
            boxes_level = []
            for config in configs:
                (stride, octave_scale, aspect, anchor_scale) = config
                base_anchor_size_x = anchor_scale * stride[1] * 2 ** octave_scale
                base_anchor_size_y = anchor_scale * stride[0] * 2 ** octave_scale
                if isinstance(aspect, list):
                    (aspect_x, aspect_y) = aspect
                else:
                    aspect_x = np.sqrt(aspect)
                    aspect_y = 1.0 / aspect_x
                anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0
                anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0
                x = np.arange(stride[1] / 2, self.image_size[1], stride[1])
                y = np.arange(stride[0] / 2, self.image_size[0], stride[0])
                (xv, yv) = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2, yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
        return anchor_boxes

    def get_anchors_per_location(self):
        if False:
            print('Hello World!')
        return self.num_scales * len(self.aspect_ratios)

class AnchorLabeler(object):
    """Labeler for multiscale anchor boxes."""

    def __init__(self, anchors, num_classes, match_threshold=0.5):
        if False:
            for i in range(10):
                print('nop')
        'Constructs anchor labeler to assign labels to anchors.\n\n        Args:\n          anchors: an instance of class Anchors.\n          num_classes: integer number representing number of classes in the dataset.\n          match_threshold: float number between 0 and 1 representing the threshold\n            to assign positive labels for anchors.\n        '
        similarity_calc = region_similarity_calculator.IouSimilarity()
        matcher = argmax_matcher.ArgMaxMatcher(match_threshold, unmatched_threshold=match_threshold, negatives_lower_than_unmatched=True, force_match_for_each_row=True)
        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
        self._target_assigner = target_assigner.TargetAssigner(similarity_calc, matcher, box_coder)
        self._anchors = anchors
        self._match_threshold = match_threshold
        self._num_classes = num_classes

    def _unpack_labels(self, labels):
        if False:
            return 10
        'Unpacks an array of labels into multiscales labels.'
        labels_unpacked = collections.OrderedDict()
        anchors = self._anchors
        count = 0
        for level in range(anchors.min_level, anchors.max_level + 1):
            feat_size = anchors.feat_sizes[level]
            steps = feat_size['height'] * feat_size['width'] * anchors.get_anchors_per_location()
            indices = tf.range(count, count + steps)
            count += steps
            labels_unpacked[level] = tf.reshape(tf.gather(labels, indices), [feat_size['height'], feat_size['width'], -1])
        return labels_unpacked

    def label_anchors(self, gt_boxes, gt_labels):
        if False:
            print('Hello World!')
        'Labels anchors with ground truth inputs.\n\n        Args:\n          gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.\n            For each row, it stores [y0, x0, y1, x1] for four corners of a box.\n          gt_labels: A integer tensor with shape [N, 1] representing groundtruth\n            classes.\n        Returns:\n          cls_targets_dict: ordered dictionary with keys\n            [min_level, min_level+1, ..., max_level]. The values are tensor with\n            shape [height_l, width_l, num_anchors]. The height_l and width_l\n            represent the dimension of class logits at l-th level.\n          box_targets_dict: ordered dictionary with keys\n            [min_level, min_level+1, ..., max_level]. The values are tensor with\n            shape [height_l, width_l, num_anchors * 4]. The height_l and\n            width_l represent the dimension of bounding box regression output at\n            l-th level.\n          num_positives: scalar tensor storing number of positives in an image.\n        '
        gt_box_list = box_list.BoxList(gt_boxes)
        anchor_box_list = box_list.BoxList(self._anchors.boxes)
        (cls_targets, _, box_targets, _, matches) = self._target_assigner.assign(anchor_box_list, gt_box_list, gt_labels)
        cls_targets -= 1
        cls_targets = tf.cast(cls_targets, tf.int32)
        cls_targets_dict = self._unpack_labels(cls_targets)
        box_targets_dict = self._unpack_labels(box_targets)
        num_positives = tf.reduce_sum(tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))
        return (cls_targets_dict, box_targets_dict, num_positives)