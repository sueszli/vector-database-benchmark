"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""
import tensorflow.compat.v2 as tf
from official.vision.detection.utils.object_detection import box_coder
from official.vision.detection.utils.object_detection import box_list
EPSILON = 1e-08

class FasterRcnnBoxCoder(box_coder.BoxCoder):
    """Faster RCNN box coder."""

    def __init__(self, scale_factors=None):
        if False:
            print('Hello World!')
        'Constructor for FasterRcnnBoxCoder.\n\n    Args:\n      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.\n        If set to None, does not perform scaling. For Faster RCNN,\n        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].\n    '
        if scale_factors:
            assert len(scale_factors) == 4
            for scalar in scale_factors:
                assert scalar > 0
        self._scale_factors = scale_factors

    @property
    def code_size(self):
        if False:
            i = 10
            return i + 15
        return 4

    def _encode(self, boxes, anchors):
        if False:
            i = 10
            return i + 15
        'Encode a box collection with respect to anchor collection.\n\n    Args:\n      boxes: BoxList holding N boxes to be encoded.\n      anchors: BoxList of anchors.\n\n    Returns:\n      a tensor representing N anchor-encoded boxes of the format\n      [ty, tx, th, tw].\n    '
        (ycenter_a, xcenter_a, ha, wa) = anchors.get_center_coordinates_and_sizes()
        (ycenter, xcenter, h, w) = boxes.get_center_coordinates_and_sizes()
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON
        tx = (xcenter - xcenter_a) / wa
        ty = (ycenter - ycenter_a) / ha
        tw = tf.math.log(w / wa)
        th = tf.math.log(h / ha)
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
        return tf.transpose(a=tf.stack([ty, tx, th, tw]))

    def _decode(self, rel_codes, anchors):
        if False:
            while True:
                i = 10
        'Decode relative codes to boxes.\n\n    Args:\n      rel_codes: a tensor representing N anchor-encoded boxes.\n      anchors: BoxList of anchors.\n\n    Returns:\n      boxes: BoxList holding N bounding boxes.\n    '
        (ycenter_a, xcenter_a, ha, wa) = anchors.get_center_coordinates_and_sizes()
        (ty, tx, th, tw) = tf.unstack(tf.transpose(a=rel_codes))
        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0
        return box_list.BoxList(tf.transpose(a=tf.stack([ymin, xmin, ymax, xmax])))