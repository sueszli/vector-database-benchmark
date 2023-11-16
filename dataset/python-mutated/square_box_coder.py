"""Square box coder.

Square box coder follows the coding schema described below:
l = sqrt(h * w)
la = sqrt(ha * wa)
ty = (y - ya) / la
tx = (x - xa) / la
tl = log(l / la)
where x, y, w, h denote the box's center coordinates, width, and height,
respectively. Similarly, xa, ya, wa, ha denote the anchor's center
coordinates, width and height. tx, ty, tl denote the anchor-encoded
center, and length, respectively. Because the encoded box is a square, only
one length is encoded.

This has shown to provide performance improvements over the Faster RCNN box
coder when the objects being detected tend to be square (e.g. faces) and when
the input images are not distorted via resizing.
"""
import tensorflow as tf
from object_detection.core import box_coder
from object_detection.core import box_list
EPSILON = 1e-08

class SquareBoxCoder(box_coder.BoxCoder):
    """Encodes a 3-scalar representation of a square box."""

    def __init__(self, scale_factors=None):
        if False:
            while True:
                i = 10
        'Constructor for SquareBoxCoder.\n\n    Args:\n      scale_factors: List of 3 positive scalars to scale ty, tx, and tl.\n        If set to None, does not perform scaling. For faster RCNN,\n        the open-source implementation recommends using [10.0, 10.0, 5.0].\n\n    Raises:\n      ValueError: If scale_factors is not length 3 or contains values less than\n        or equal to 0.\n    '
        if scale_factors:
            if len(scale_factors) != 3:
                raise ValueError('The argument scale_factors must be a list of length 3.')
            if any((scalar <= 0 for scalar in scale_factors)):
                raise ValueError('The values in scale_factors must all be greater than 0.')
        self._scale_factors = scale_factors

    @property
    def code_size(self):
        if False:
            while True:
                i = 10
        return 3

    def _encode(self, boxes, anchors):
        if False:
            while True:
                i = 10
        'Encodes a box collection with respect to an anchor collection.\n\n    Args:\n      boxes: BoxList holding N boxes to be encoded.\n      anchors: BoxList of anchors.\n\n    Returns:\n      a tensor representing N anchor-encoded boxes of the format\n      [ty, tx, tl].\n    '
        (ycenter_a, xcenter_a, ha, wa) = anchors.get_center_coordinates_and_sizes()
        la = tf.sqrt(ha * wa)
        (ycenter, xcenter, h, w) = boxes.get_center_coordinates_and_sizes()
        l = tf.sqrt(h * w)
        la += EPSILON
        l += EPSILON
        tx = (xcenter - xcenter_a) / la
        ty = (ycenter - ycenter_a) / la
        tl = tf.log(l / la)
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            tl *= self._scale_factors[2]
        return tf.transpose(tf.stack([ty, tx, tl]))

    def _decode(self, rel_codes, anchors):
        if False:
            print('Hello World!')
        'Decodes relative codes to boxes.\n\n    Args:\n      rel_codes: a tensor representing N anchor-encoded boxes.\n      anchors: BoxList of anchors.\n\n    Returns:\n      boxes: BoxList holding N bounding boxes.\n    '
        (ycenter_a, xcenter_a, ha, wa) = anchors.get_center_coordinates_and_sizes()
        la = tf.sqrt(ha * wa)
        (ty, tx, tl) = tf.unstack(tf.transpose(rel_codes))
        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            tl /= self._scale_factors[2]
        l = tf.exp(tl) * la
        ycenter = ty * la + ycenter_a
        xcenter = tx * la + xcenter_a
        ymin = ycenter - l / 2.0
        xmin = xcenter - l / 2.0
        ymax = ycenter + l / 2.0
        xmax = xcenter + l / 2.0
        return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))