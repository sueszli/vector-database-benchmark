"""Generates grid anchors on the fly corresponding to multiple CNN layers."""
import tensorflow as tf
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops

class FlexibleGridAnchorGenerator(anchor_generator.AnchorGenerator):
    """Generate a grid of anchors for multiple CNN layers of different scale."""

    def __init__(self, base_sizes, aspect_ratios, anchor_strides, anchor_offsets, normalize_coordinates=True):
        if False:
            i = 10
            return i + 15
        'Constructs a FlexibleGridAnchorGenerator.\n\n    This generator is more flexible than the multiple_grid_anchor_generator\n    and multiscale_grid_anchor_generator, and can generate any of the anchors\n    that they can generate, plus additional anchor configurations. In\n    particular, it allows the explicit specification of scale and aspect ratios\n    at each layer without making any assumptions between the relationship\n    between scales and aspect ratios between layers.\n\n    Args:\n      base_sizes: list of tuples of anchor base sizes. For example, setting\n        base_sizes=[(1, 2, 3), (4, 5)] means that we want 3 anchors at each\n        grid point on the first layer with the base sizes of 1, 2, and 3, and 2\n        anchors at each grid point on the second layer with the base sizes of\n        4 and 5.\n      aspect_ratios: list or tuple of aspect ratios. For example, setting\n        aspect_ratios=[(1.0, 2.0, 0.5), (1.0, 2.0)] means that we want 3 anchors\n        at each grid point on the first layer with aspect ratios of 1.0, 2.0,\n        and 0.5, and 2 anchors at each grid point on the sercond layer with the\n        base sizes of 1.0 and 2.0.\n      anchor_strides: list of pairs of strides in pixels (in y and x directions\n        respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]\n        means that we want the anchors corresponding to the first layer to be\n        strided by 25 pixels and those in the second layer to be strided by 50\n        pixels in both y and x directions.\n      anchor_offsets: list of pairs of offsets in pixels (in y and x directions\n        respectively). The offset specifies where we want the center of the\n        (0, 0)-th anchor to lie for each layer. For example, setting\n        anchor_offsets=[(10, 10), (20, 20)]) means that we want the\n        (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space\n        and likewise that we want the (0, 0)-th anchor of the second layer to\n        lie at (25, 25) in pixel space.\n      normalize_coordinates: whether to produce anchors in normalized\n        coordinates. (defaults to True).\n    '
        self._base_sizes = base_sizes
        self._aspect_ratios = aspect_ratios
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._normalize_coordinates = normalize_coordinates

    def name_scope(self):
        if False:
            return 10
        return 'FlexibleGridAnchorGenerator'

    def num_anchors_per_location(self):
        if False:
            return 10
        'Returns the number of anchors per spatial location.\n\n    Returns:\n      a list of integers, one for each expected feature map to be passed to\n      the Generate function.\n    '
        return [len(size) for size in self._base_sizes]

    def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
        if False:
            print('Hello World!')
        'Generates a collection of bounding boxes to be used as anchors.\n\n    Currently we require the input image shape to be statically defined.  That\n    is, im_height and im_width should be integers rather than tensors.\n\n    Args:\n      feature_map_shape_list: list of pairs of convnet layer resolutions in the\n        format [(height_0, width_0), (height_1, width_1), ...]. For example,\n        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that\n        correspond to an 8x8 layer followed by a 7x7 layer.\n      im_height: the height of the image to generate the grid for. If both\n        im_height and im_width are 1, anchors can only be generated in\n        absolute coordinates.\n      im_width: the width of the image to generate the grid for. If both\n        im_height and im_width are 1, anchors can only be generated in\n        absolute coordinates.\n\n    Returns:\n      boxes_list: a list of BoxLists each holding anchor boxes corresponding to\n        the input feature map shapes.\n    Raises:\n      ValueError: if im_height and im_width are 1, but normalized coordinates\n        were requested.\n    '
        anchor_grid_list = []
        for (feat_shape, base_sizes, aspect_ratios, anchor_stride, anchor_offset) in zip(feature_map_shape_list, self._base_sizes, self._aspect_ratios, self._anchor_strides, self._anchor_offsets):
            anchor_grid = grid_anchor_generator.tile_anchors(feat_shape[0], feat_shape[1], tf.cast(tf.convert_to_tensor(base_sizes), dtype=tf.float32), tf.cast(tf.convert_to_tensor(aspect_ratios), dtype=tf.float32), tf.constant([1.0, 1.0]), tf.cast(tf.convert_to_tensor(anchor_stride), dtype=tf.float32), tf.cast(tf.convert_to_tensor(anchor_offset), dtype=tf.float32))
            num_anchors = anchor_grid.num_boxes_static()
            if num_anchors is None:
                num_anchors = anchor_grid.num_boxes()
            anchor_indices = tf.zeros([num_anchors])
            anchor_grid.add_field('feature_map_index', anchor_indices)
            if self._normalize_coordinates:
                if im_height == 1 or im_width == 1:
                    raise ValueError('Normalized coordinates were requested upon construction of the FlexibleGridAnchorGenerator, but a subsequent call to generate did not supply dimension information.')
                anchor_grid = box_list_ops.to_normalized_coordinates(anchor_grid, im_height, im_width, check_range=False)
            anchor_grid_list.append(anchor_grid)
        return anchor_grid_list