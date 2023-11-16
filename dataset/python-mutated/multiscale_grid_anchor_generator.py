"""Generates grid anchors on the fly corresponding to multiple CNN layers.

Generates grid anchors on the fly corresponding to multiple CNN layers as
described in:
"Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollar
"""
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops

class MultiscaleGridAnchorGenerator(anchor_generator.AnchorGenerator):
    """Generate a grid of anchors for multiple CNN layers of different scale."""

    def __init__(self, min_level, max_level, anchor_scale, aspect_ratios, scales_per_octave, normalize_coordinates=True):
        if False:
            return 10
        'Constructs a MultiscaleGridAnchorGenerator.\n\n    To construct anchors, at multiple scale resolutions, one must provide a\n    the minimum level and maximum levels on a scale pyramid. To define the size\n    of anchor, the anchor scale is provided to decide the size relatively to the\n    stride of the corresponding feature map. The generator allows one pixel\n    location on feature map maps to multiple anchors, that have different aspect\n    ratios and intermediate scales.\n\n    Args:\n      min_level: minimum level in feature pyramid.\n      max_level: maximum level in feature pyramid.\n      anchor_scale: anchor scale and feature stride define the size of the base\n        anchor on an image. For example, given a feature pyramid with strides\n        [2^3, ..., 2^7] and anchor scale 4. The base anchor size is\n        4 * [2^3, ..., 2^7].\n      aspect_ratios: list or tuple of (float) aspect ratios to place on each\n        grid point.\n      scales_per_octave: integer number of intermediate scales per scale octave.\n      normalize_coordinates: whether to produce anchors in normalized\n        coordinates. (defaults to True).\n    '
        self._anchor_grid_info = []
        self._aspect_ratios = aspect_ratios
        self._scales_per_octave = scales_per_octave
        self._normalize_coordinates = normalize_coordinates
        scales = [2 ** (float(scale) / scales_per_octave) for scale in range(scales_per_octave)]
        aspects = list(aspect_ratios)
        for level in range(min_level, max_level + 1):
            anchor_stride = [2 ** level, 2 ** level]
            base_anchor_size = [2 ** level * anchor_scale, 2 ** level * anchor_scale]
            self._anchor_grid_info.append({'level': level, 'info': [scales, aspects, base_anchor_size, anchor_stride]})

    def name_scope(self):
        if False:
            i = 10
            return i + 15
        return 'MultiscaleGridAnchorGenerator'

    def num_anchors_per_location(self):
        if False:
            while True:
                i = 10
        'Returns the number of anchors per spatial location.\n\n    Returns:\n      a list of integers, one for each expected feature map to be passed to\n      the Generate function.\n    '
        return len(self._anchor_grid_info) * [len(self._aspect_ratios) * self._scales_per_octave]

    def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
        if False:
            i = 10
            return i + 15
        'Generates a collection of bounding boxes to be used as anchors.\n\n    Currently we require the input image shape to be statically defined.  That\n    is, im_height and im_width should be integers rather than tensors.\n\n    Args:\n      feature_map_shape_list: list of pairs of convnet layer resolutions in the\n        format [(height_0, width_0), (height_1, width_1), ...]. For example,\n        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that\n        correspond to an 8x8 layer followed by a 7x7 layer.\n      im_height: the height of the image to generate the grid for. If both\n        im_height and im_width are 1, anchors can only be generated in\n        absolute coordinates.\n      im_width: the width of the image to generate the grid for. If both\n        im_height and im_width are 1, anchors can only be generated in\n        absolute coordinates.\n\n    Returns:\n      boxes_list: a list of BoxLists each holding anchor boxes corresponding to\n        the input feature map shapes.\n    Raises:\n      ValueError: if im_height and im_width are not integers.\n      ValueError: if im_height and im_width are 1, but normalized coordinates\n        were requested.\n    '
        anchor_grid_list = []
        for (feat_shape, grid_info) in zip(feature_map_shape_list, self._anchor_grid_info):
            level = grid_info['level']
            stride = 2 ** level
            (scales, aspect_ratios, base_anchor_size, anchor_stride) = grid_info['info']
            feat_h = feat_shape[0]
            feat_w = feat_shape[1]
            anchor_offset = [0, 0]
            if isinstance(im_height, int) and isinstance(im_width, int):
                if im_height % 2.0 ** level == 0 or im_height == 1:
                    anchor_offset[0] = stride / 2.0
                if im_width % 2.0 ** level == 0 or im_width == 1:
                    anchor_offset[1] = stride / 2.0
            ag = grid_anchor_generator.GridAnchorGenerator(scales, aspect_ratios, base_anchor_size=base_anchor_size, anchor_stride=anchor_stride, anchor_offset=anchor_offset)
            (anchor_grid,) = ag.generate(feature_map_shape_list=[(feat_h, feat_w)])
            if self._normalize_coordinates:
                if im_height == 1 or im_width == 1:
                    raise ValueError('Normalized coordinates were requested upon construction of the MultiscaleGridAnchorGenerator, but a subsequent call to generate did not supply dimension information.')
                anchor_grid = box_list_ops.to_normalized_coordinates(anchor_grid, im_height, im_width, check_range=False)
            anchor_grid_list.append(anchor_grid)
        return anchor_grid_list