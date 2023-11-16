"""Generates grid anchors on the fly corresponding to multiple CNN layers.

Generates grid anchors on the fly corresponding to multiple CNN layers as
described in:
"SSD: Single Shot MultiBox Detector"
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
(see Section 2.2: Choosing scales and aspect ratios for default boxes)
"""
import numpy as np
import tensorflow as tf
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops

class MultipleGridAnchorGenerator(anchor_generator.AnchorGenerator):
    """Generate a grid of anchors for multiple CNN layers."""

    def __init__(self, box_specs_list, base_anchor_size=None, anchor_strides=None, anchor_offsets=None, clip_window=None):
        if False:
            return 10
        'Constructs a MultipleGridAnchorGenerator.\n\n    To construct anchors, at multiple grid resolutions, one must provide a\n    list of feature_map_shape_list (e.g., [(8, 8), (4, 4)]), and for each grid\n    size, a corresponding list of (scale, aspect ratio) box specifications.\n\n    For example:\n    box_specs_list = [[(.1, 1.0), (.1, 2.0)],  # for 8x8 grid\n                      [(.2, 1.0), (.3, 1.0), (.2, 2.0)]]  # for 4x4 grid\n\n    To support the fully convolutional setting, we pass grid sizes in at\n    generation time, while scale and aspect ratios are fixed at construction\n    time.\n\n    Args:\n      box_specs_list: list of list of (scale, aspect ratio) pairs with the\n        outside list having the same number of entries as feature_map_shape_list\n        (which is passed in at generation time).\n      base_anchor_size: base anchor size as [height, width]\n                        (length-2 float numpy or Tensor, default=[1.0, 1.0]).\n                        The height and width values are normalized to the\n                        minimum dimension of the input height and width, so that\n                        when the base anchor height equals the base anchor\n                        width, the resulting anchor is square even if the input\n                        image is not square.\n      anchor_strides: list of pairs of strides in pixels (in y and x directions\n        respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]\n        means that we want the anchors corresponding to the first layer to be\n        strided by 25 pixels and those in the second layer to be strided by 50\n        pixels in both y and x directions. If anchor_strides=None, they are set\n        to be the reciprocal of the corresponding feature map shapes.\n      anchor_offsets: list of pairs of offsets in pixels (in y and x directions\n        respectively). The offset specifies where we want the center of the\n        (0, 0)-th anchor to lie for each layer. For example, setting\n        anchor_offsets=[(10, 10), (20, 20)]) means that we want the\n        (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space\n        and likewise that we want the (0, 0)-th anchor of the second layer to\n        lie at (25, 25) in pixel space. If anchor_offsets=None, then they are\n        set to be half of the corresponding anchor stride.\n      clip_window: a tensor of shape [4] specifying a window to which all\n        anchors should be clipped. If clip_window is None, then no clipping\n        is performed.\n\n    Raises:\n      ValueError: if box_specs_list is not a list of list of pairs\n      ValueError: if clip_window is not either None or a tensor of shape [4]\n    '
        if isinstance(box_specs_list, list) and all([isinstance(list_item, list) for list_item in box_specs_list]):
            self._box_specs = box_specs_list
        else:
            raise ValueError('box_specs_list is expected to be a list of lists of pairs')
        if base_anchor_size is None:
            base_anchor_size = [256, 256]
        self._base_anchor_size = base_anchor_size
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        if clip_window is not None and clip_window.get_shape().as_list() != [4]:
            raise ValueError('clip_window must either be None or a shape [4] tensor')
        self._clip_window = clip_window
        self._scales = []
        self._aspect_ratios = []
        for box_spec in self._box_specs:
            if not all([isinstance(entry, tuple) and len(entry) == 2 for entry in box_spec]):
                raise ValueError('box_specs_list is expected to be a list of lists of pairs')
            (scales, aspect_ratios) = zip(*box_spec)
            self._scales.append(scales)
            self._aspect_ratios.append(aspect_ratios)
        for (arg, arg_name) in zip([self._anchor_strides, self._anchor_offsets], ['anchor_strides', 'anchor_offsets']):
            if arg and (not (isinstance(arg, list) and len(arg) == len(self._box_specs))):
                raise ValueError('%s must be a list with the same length as self._box_specs' % arg_name)
            if arg and (not all([isinstance(list_item, tuple) and len(list_item) == 2 for list_item in arg])):
                raise ValueError('%s must be a list of pairs.' % arg_name)

    def name_scope(self):
        if False:
            while True:
                i = 10
        return 'MultipleGridAnchorGenerator'

    def num_anchors_per_location(self):
        if False:
            return 10
        'Returns the number of anchors per spatial location.\n\n    Returns:\n      a list of integers, one for each expected feature map to be passed to\n      the Generate function.\n    '
        return [len(box_specs) for box_specs in self._box_specs]

    def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
        if False:
            i = 10
            return i + 15
        'Generates a collection of bounding boxes to be used as anchors.\n\n    The number of anchors generated for a single grid with shape MxM where we\n    place k boxes over each grid center is k*M^2 and thus the total number of\n    anchors is the sum over all grids. In our box_specs_list example\n    (see the constructor docstring), we would place two boxes over each grid\n    point on an 8x8 grid and three boxes over each grid point on a 4x4 grid and\n    thus end up with 2*8^2 + 3*4^2 = 176 anchors in total. The layout of the\n    output anchors follows the order of how the grid sizes and box_specs are\n    specified (with box_spec index varying the fastest, followed by width\n    index, then height index, then grid index).\n\n    Args:\n      feature_map_shape_list: list of pairs of convnet layer resolutions in the\n        format [(height_0, width_0), (height_1, width_1), ...]. For example,\n        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that\n        correspond to an 8x8 layer followed by a 7x7 layer.\n      im_height: the height of the image to generate the grid for. If both\n        im_height and im_width are 1, the generated anchors default to\n        absolute coordinates, otherwise normalized coordinates are produced.\n      im_width: the width of the image to generate the grid for. If both\n        im_height and im_width are 1, the generated anchors default to\n        absolute coordinates, otherwise normalized coordinates are produced.\n\n    Returns:\n      boxes_list: a list of BoxLists each holding anchor boxes corresponding to\n        the input feature map shapes.\n\n    Raises:\n      ValueError: if feature_map_shape_list, box_specs_list do not have the same\n        length.\n      ValueError: if feature_map_shape_list does not consist of pairs of\n        integers\n    '
        if not (isinstance(feature_map_shape_list, list) and len(feature_map_shape_list) == len(self._box_specs)):
            raise ValueError('feature_map_shape_list must be a list with the same length as self._box_specs')
        if not all([isinstance(list_item, tuple) and len(list_item) == 2 for list_item in feature_map_shape_list]):
            raise ValueError('feature_map_shape_list must be a list of pairs.')
        im_height = tf.cast(im_height, dtype=tf.float32)
        im_width = tf.cast(im_width, dtype=tf.float32)
        if not self._anchor_strides:
            anchor_strides = [(1.0 / tf.cast(pair[0], dtype=tf.float32), 1.0 / tf.cast(pair[1], dtype=tf.float32)) for pair in feature_map_shape_list]
        else:
            anchor_strides = [(tf.cast(stride[0], dtype=tf.float32) / im_height, tf.cast(stride[1], dtype=tf.float32) / im_width) for stride in self._anchor_strides]
        if not self._anchor_offsets:
            anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1]) for stride in anchor_strides]
        else:
            anchor_offsets = [(tf.cast(offset[0], dtype=tf.float32) / im_height, tf.cast(offset[1], dtype=tf.float32) / im_width) for offset in self._anchor_offsets]
        for (arg, arg_name) in zip([anchor_strides, anchor_offsets], ['anchor_strides', 'anchor_offsets']):
            if not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
                raise ValueError('%s must be a list with the same length as self._box_specs' % arg_name)
            if not all([isinstance(list_item, tuple) and len(list_item) == 2 for list_item in arg]):
                raise ValueError('%s must be a list of pairs.' % arg_name)
        anchor_grid_list = []
        min_im_shape = tf.minimum(im_height, im_width)
        scale_height = min_im_shape / im_height
        scale_width = min_im_shape / im_width
        if not tf.contrib.framework.is_tensor(self._base_anchor_size):
            base_anchor_size = [scale_height * tf.constant(self._base_anchor_size[0], dtype=tf.float32), scale_width * tf.constant(self._base_anchor_size[1], dtype=tf.float32)]
        else:
            base_anchor_size = [scale_height * self._base_anchor_size[0], scale_width * self._base_anchor_size[1]]
        for (feature_map_index, (grid_size, scales, aspect_ratios, stride, offset)) in enumerate(zip(feature_map_shape_list, self._scales, self._aspect_ratios, anchor_strides, anchor_offsets)):
            tiled_anchors = grid_anchor_generator.tile_anchors(grid_height=grid_size[0], grid_width=grid_size[1], scales=scales, aspect_ratios=aspect_ratios, base_anchor_size=base_anchor_size, anchor_stride=stride, anchor_offset=offset)
            if self._clip_window is not None:
                tiled_anchors = box_list_ops.clip_to_window(tiled_anchors, self._clip_window, filter_nonoverlapping=False)
            num_anchors_in_layer = tiled_anchors.num_boxes_static()
            if num_anchors_in_layer is None:
                num_anchors_in_layer = tiled_anchors.num_boxes()
            anchor_indices = feature_map_index * tf.ones([num_anchors_in_layer])
            tiled_anchors.add_field('feature_map_index', anchor_indices)
            anchor_grid_list.append(tiled_anchors)
        return anchor_grid_list

def create_ssd_anchors(num_layers=6, min_scale=0.2, max_scale=0.95, scales=None, aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3), interpolated_scale_aspect_ratio=1.0, base_anchor_size=None, anchor_strides=None, anchor_offsets=None, reduce_boxes_in_lowest_layer=True):
    if False:
        for i in range(10):
            print('nop')
    'Creates MultipleGridAnchorGenerator for SSD anchors.\n\n  This function instantiates a MultipleGridAnchorGenerator that reproduces\n  ``default box`` construction proposed by Liu et al in the SSD paper.\n  See Section 2.2 for details. Grid sizes are assumed to be passed in\n  at generation time from finest resolution to coarsest resolution --- this is\n  used to (linearly) interpolate scales of anchor boxes corresponding to the\n  intermediate grid sizes.\n\n  Anchors that are returned by calling the `generate` method on the returned\n  MultipleGridAnchorGenerator object are always in normalized coordinates\n  and clipped to the unit square: (i.e. all coordinates lie in [0, 1]x[0, 1]).\n\n  Args:\n    num_layers: integer number of grid layers to create anchors for (actual\n      grid sizes passed in at generation time)\n    min_scale: scale of anchors corresponding to finest resolution (float)\n    max_scale: scale of anchors corresponding to coarsest resolution (float)\n    scales: As list of anchor scales to use. When not None and not empty,\n      min_scale and max_scale are not used.\n    aspect_ratios: list or tuple of (float) aspect ratios to place on each\n      grid point.\n    interpolated_scale_aspect_ratio: An additional anchor is added with this\n      aspect ratio and a scale interpolated between the scale for a layer\n      and the scale for the next layer (1.0 for the last layer).\n      This anchor is not included if this value is 0.\n    base_anchor_size: base anchor size as [height, width].\n      The height and width values are normalized to the minimum dimension of the\n      input height and width, so that when the base anchor height equals the\n      base anchor width, the resulting anchor is square even if the input image\n      is not square.\n    anchor_strides: list of pairs of strides in pixels (in y and x directions\n      respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]\n      means that we want the anchors corresponding to the first layer to be\n      strided by 25 pixels and those in the second layer to be strided by 50\n      pixels in both y and x directions. If anchor_strides=None, they are set to\n      be the reciprocal of the corresponding feature map shapes.\n    anchor_offsets: list of pairs of offsets in pixels (in y and x directions\n      respectively). The offset specifies where we want the center of the\n      (0, 0)-th anchor to lie for each layer. For example, setting\n      anchor_offsets=[(10, 10), (20, 20)]) means that we want the\n      (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space\n      and likewise that we want the (0, 0)-th anchor of the second layer to lie\n      at (25, 25) in pixel space. If anchor_offsets=None, then they are set to\n      be half of the corresponding anchor stride.\n    reduce_boxes_in_lowest_layer: a boolean to indicate whether the fixed 3\n      boxes per location is used in the lowest layer.\n\n  Returns:\n    a MultipleGridAnchorGenerator\n  '
    if base_anchor_size is None:
        base_anchor_size = [1.0, 1.0]
    box_specs_list = []
    if scales is None or not scales:
        scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)] + [1.0]
    else:
        scales += [1.0]
    for (layer, scale, scale_next) in zip(range(num_layers), scales[:-1], scales[1:]):
        layer_box_specs = []
        if layer == 0 and reduce_boxes_in_lowest_layer:
            layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
        else:
            for aspect_ratio in aspect_ratios:
                layer_box_specs.append((scale, aspect_ratio))
            if interpolated_scale_aspect_ratio > 0.0:
                layer_box_specs.append((np.sqrt(scale * scale_next), interpolated_scale_aspect_ratio))
        box_specs_list.append(layer_box_specs)
    return MultipleGridAnchorGenerator(box_specs_list, base_anchor_size, anchor_strides, anchor_offsets)