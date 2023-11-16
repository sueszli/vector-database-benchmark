"""Generates grid anchors on the fly as used in Faster RCNN.

Generates grid anchors on the fly as described in:
"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
"""
import tensorflow as tf
from object_detection.core import anchor_generator
from object_detection.core import box_list
from object_detection.utils import ops

class GridAnchorGenerator(anchor_generator.AnchorGenerator):
    """Generates a grid of anchors at given scales and aspect ratios."""

    def __init__(self, scales=(0.5, 1.0, 2.0), aspect_ratios=(0.5, 1.0, 2.0), base_anchor_size=None, anchor_stride=None, anchor_offset=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructs a GridAnchorGenerator.\n\n    Args:\n      scales: a list of (float) scales, default=(0.5, 1.0, 2.0)\n      aspect_ratios: a list of (float) aspect ratios, default=(0.5, 1.0, 2.0)\n      base_anchor_size: base anchor size as height, width (\n                        (length-2 float32 list or tensor, default=[256, 256])\n      anchor_stride: difference in centers between base anchors for adjacent\n                     grid positions (length-2 float32 list or tensor,\n                     default=[16, 16])\n      anchor_offset: center of the anchor with scale and aspect ratio 1 for the\n                     upper left element of the grid, this should be zero for\n                     feature networks with only VALID padding and even receptive\n                     field size, but may need additional calculation if other\n                     padding is used (length-2 float32 list or tensor,\n                     default=[0, 0])\n    '
        if base_anchor_size is None:
            base_anchor_size = [256, 256]
        if anchor_stride is None:
            anchor_stride = [16, 16]
        if anchor_offset is None:
            anchor_offset = [0, 0]
        self._scales = scales
        self._aspect_ratios = aspect_ratios
        self._base_anchor_size = base_anchor_size
        self._anchor_stride = anchor_stride
        self._anchor_offset = anchor_offset

    def name_scope(self):
        if False:
            for i in range(10):
                print('nop')
        return 'GridAnchorGenerator'

    def num_anchors_per_location(self):
        if False:
            while True:
                i = 10
        'Returns the number of anchors per spatial location.\n\n    Returns:\n      a list of integers, one for each expected feature map to be passed to\n      the `generate` function.\n    '
        return [len(self._scales) * len(self._aspect_ratios)]

    def _generate(self, feature_map_shape_list):
        if False:
            for i in range(10):
                print('nop')
        'Generates a collection of bounding boxes to be used as anchors.\n\n    Args:\n      feature_map_shape_list: list of pairs of convnet layer resolutions in the\n        format [(height_0, width_0)].  For example, setting\n        feature_map_shape_list=[(8, 8)] asks for anchors that correspond\n        to an 8x8 layer.  For this anchor generator, only lists of length 1 are\n        allowed.\n\n    Returns:\n      boxes_list: a list of BoxLists each holding anchor boxes corresponding to\n        the input feature map shapes.\n\n    Raises:\n      ValueError: if feature_map_shape_list, box_specs_list do not have the same\n        length.\n      ValueError: if feature_map_shape_list does not consist of pairs of\n        integers\n    '
        if not (isinstance(feature_map_shape_list, list) and len(feature_map_shape_list) == 1):
            raise ValueError('feature_map_shape_list must be a list of length 1.')
        if not all([isinstance(list_item, tuple) and len(list_item) == 2 for list_item in feature_map_shape_list]):
            raise ValueError('feature_map_shape_list must be a list of pairs.')
        with tf.init_scope():
            self._base_anchor_size = tf.cast(tf.convert_to_tensor(self._base_anchor_size), dtype=tf.float32)
            self._anchor_stride = tf.cast(tf.convert_to_tensor(self._anchor_stride), dtype=tf.float32)
            self._anchor_offset = tf.cast(tf.convert_to_tensor(self._anchor_offset), dtype=tf.float32)
        (grid_height, grid_width) = feature_map_shape_list[0]
        (scales_grid, aspect_ratios_grid) = ops.meshgrid(self._scales, self._aspect_ratios)
        scales_grid = tf.reshape(scales_grid, [-1])
        aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
        anchors = tile_anchors(grid_height, grid_width, scales_grid, aspect_ratios_grid, self._base_anchor_size, self._anchor_stride, self._anchor_offset)
        num_anchors = anchors.num_boxes_static()
        if num_anchors is None:
            num_anchors = anchors.num_boxes()
        anchor_indices = tf.zeros([num_anchors])
        anchors.add_field('feature_map_index', anchor_indices)
        return [anchors]

def tile_anchors(grid_height, grid_width, scales, aspect_ratios, base_anchor_size, anchor_stride, anchor_offset):
    if False:
        while True:
            i = 10
    'Create a tiled set of anchors strided along a grid in image space.\n\n  This op creates a set of anchor boxes by placing a "basis" collection of\n  boxes with user-specified scales and aspect ratios centered at evenly\n  distributed points along a grid.  The basis collection is specified via the\n  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]\n  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale\n  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2\n  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before\n  placing it over its respective center.\n\n  Grid points are specified via grid_height, grid_width parameters as well as\n  the anchor_stride and anchor_offset parameters.\n\n  Args:\n    grid_height: size of the grid in the y direction (int or int scalar tensor)\n    grid_width: size of the grid in the x direction (int or int scalar tensor)\n    scales: a 1-d  (float) tensor representing the scale of each box in the\n      basis set.\n    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each\n      box in the basis set.  The length of the scales and aspect_ratios tensors\n      must be equal.\n    base_anchor_size: base anchor size as [height, width]\n      (float tensor of shape [2])\n    anchor_stride: difference in centers between base anchors for adjacent grid\n                   positions (float tensor of shape [2])\n    anchor_offset: center of the anchor with scale and aspect ratio 1 for the\n                   upper left element of the grid, this should be zero for\n                   feature networks with only VALID padding and even receptive\n                   field size, but may need some additional calculation if other\n                   padding is used (float tensor of shape [2])\n  Returns:\n    a BoxList holding a collection of N anchor boxes\n  '
    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]
    y_centers = tf.cast(tf.range(grid_height), dtype=tf.float32)
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = tf.cast(tf.range(grid_width), dtype=tf.float32)
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    (x_centers, y_centers) = ops.meshgrid(x_centers, y_centers)
    (widths_grid, x_centers_grid) = ops.meshgrid(widths, x_centers)
    (heights_grid, y_centers_grid) = ops.meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    return box_list.BoxList(bbox_corners)

def _center_size_bbox_to_corners_bbox(centers, sizes):
    if False:
        print('Hello World!')
    'Converts bbox center-size representation to corners representation.\n\n  Args:\n    centers: a tensor with shape [N, 2] representing bounding box centers\n    sizes: a tensor with shape [N, 2] representing bounding boxes\n\n  Returns:\n    corners: tensor with shape [N, 4] representing bounding boxes in corners\n      representation\n  '
    return tf.concat([centers - 0.5 * sizes, centers + 0.5 * sizes], 1)