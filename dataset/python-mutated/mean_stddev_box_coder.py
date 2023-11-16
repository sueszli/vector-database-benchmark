"""Mean stddev box coder.

This box coder use the following coding schema to encode boxes:
rel_code = (box_corner - anchor_corner_mean) / anchor_corner_stddev.
"""
from object_detection.core import box_coder
from object_detection.core import box_list

class MeanStddevBoxCoder(box_coder.BoxCoder):
    """Mean stddev box coder."""

    def __init__(self, stddev=0.01):
        if False:
            print('Hello World!')
        'Constructor for MeanStddevBoxCoder.\n\n    Args:\n      stddev: The standard deviation used to encode and decode boxes.\n    '
        self._stddev = stddev

    @property
    def code_size(self):
        if False:
            while True:
                i = 10
        return 4

    def _encode(self, boxes, anchors):
        if False:
            for i in range(10):
                print('nop')
        'Encode a box collection with respect to anchor collection.\n\n    Args:\n      boxes: BoxList holding N boxes to be encoded.\n      anchors: BoxList of N anchors.\n\n    Returns:\n      a tensor representing N anchor-encoded boxes\n\n    Raises:\n      ValueError: if the anchors still have deprecated stddev field.\n    '
        box_corners = boxes.get()
        if anchors.has_field('stddev'):
            raise ValueError("'stddev' is a parameter of MeanStddevBoxCoder and should not be specified in the box list.")
        means = anchors.get()
        return (box_corners - means) / self._stddev

    def _decode(self, rel_codes, anchors):
        if False:
            return 10
        'Decode.\n\n    Args:\n      rel_codes: a tensor representing N anchor-encoded boxes.\n      anchors: BoxList of anchors.\n\n    Returns:\n      boxes: BoxList holding N bounding boxes\n\n    Raises:\n      ValueError: if the anchors still have deprecated stddev field and expects\n        the decode method to use stddev value from that field.\n    '
        means = anchors.get()
        if anchors.has_field('stddev'):
            raise ValueError("'stddev' is a parameter of MeanStddevBoxCoder and should not be specified in the box list.")
        box_corners = rel_codes * self._stddev + means
        return box_list.BoxList(box_corners)