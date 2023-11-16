"""Base anchor generator.

The job of the anchor generator is to create (or load) a collection
of bounding boxes to be used as anchors.

Generated anchors are assumed to match some convolutional grid or list of grid
shapes.  For example, we might want to generate anchors matching an 8x8
feature map and a 4x4 feature map.  If we place 3 anchors per grid location
on the first feature map and 6 anchors per grid location on the second feature
map, then 3*8*8 + 6*4*4 = 288 anchors are generated in total.

To support fully convolutional settings, feature map shapes are passed
dynamically at generation time.  The number of anchors to place at each location
is static --- implementations of AnchorGenerator must always be able return
the number of anchors that it uses per location for each feature map.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
import six
from six.moves import zip
import tensorflow as tf

class AnchorGenerator(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for anchor generators."""

    @abstractmethod
    def name_scope(self):
        if False:
            print('Hello World!')
        'Name scope.\n\n    Must be defined by implementations.\n\n    Returns:\n      a string representing the name scope of the anchor generation operation.\n    '
        pass

    @property
    def check_num_anchors(self):
        if False:
            while True:
                i = 10
        'Whether to dynamically check the number of anchors generated.\n\n    Can be overridden by implementations that would like to disable this\n    behavior.\n\n    Returns:\n      a boolean controlling whether the Generate function should dynamically\n      check the number of anchors generated against the mathematically\n      expected number of anchors.\n    '
        return True

    @abstractmethod
    def num_anchors_per_location(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of anchors per spatial location.\n\n    Returns:\n      a list of integers, one for each expected feature map to be passed to\n      the `generate` function.\n    '
        pass

    def generate(self, feature_map_shape_list, **params):
        if False:
            i = 10
            return i + 15
        'Generates a collection of bounding boxes to be used as anchors.\n\n    TODO(rathodv): remove **params from argument list and make stride and\n      offsets (for multiple_grid_anchor_generator) constructor arguments.\n\n    Args:\n      feature_map_shape_list: list of (height, width) pairs in the format\n        [(height_0, width_0), (height_1, width_1), ...] that the generated\n        anchors must align with.  Pairs can be provided as 1-dimensional\n        integer tensors of length 2 or simply as tuples of integers.\n      **params: parameters for anchor generation op\n\n    Returns:\n      boxes_list: a list of BoxLists each holding anchor boxes corresponding to\n        the input feature map shapes.\n\n    Raises:\n      ValueError: if the number of feature map shapes does not match the length\n        of NumAnchorsPerLocation.\n    '
        if self.check_num_anchors and len(feature_map_shape_list) != len(self.num_anchors_per_location()):
            raise ValueError('Number of feature maps is expected to equal the length of `num_anchors_per_location`.')
        with tf.name_scope(self.name_scope()):
            anchors_list = self._generate(feature_map_shape_list, **params)
            if self.check_num_anchors:
                with tf.control_dependencies([self._assert_correct_number_of_anchors(anchors_list, feature_map_shape_list)]):
                    for item in anchors_list:
                        item.set(tf.identity(item.get()))
            return anchors_list

    @abstractmethod
    def _generate(self, feature_map_shape_list, **params):
        if False:
            return 10
        'To be overridden by implementations.\n\n    Args:\n      feature_map_shape_list: list of (height, width) pairs in the format\n        [(height_0, width_0), (height_1, width_1), ...] that the generated\n        anchors must align with.\n      **params: parameters for anchor generation op\n\n    Returns:\n      boxes_list: a list of BoxList, each holding a collection of N anchor\n        boxes.\n    '
        pass

    def anchor_index_to_feature_map_index(self, boxlist_list):
        if False:
            while True:
                i = 10
        'Returns a 1-D array of feature map indices for each anchor.\n\n    Args:\n      boxlist_list: a list of Boxlist, each holding a collection of N anchor\n        boxes. This list is produced in self.generate().\n\n    Returns:\n      A [num_anchors] integer array, where each element indicates which feature\n      map index the anchor belongs to.\n    '
        feature_map_indices_list = []
        for (i, boxes) in enumerate(boxlist_list):
            feature_map_indices_list.append(i * tf.ones([boxes.num_boxes()], dtype=tf.int32))
        return tf.concat(feature_map_indices_list, axis=0)

    def _assert_correct_number_of_anchors(self, anchors_list, feature_map_shape_list):
        if False:
            return 10
        'Assert that correct number of anchors was generated.\n\n    Args:\n      anchors_list: A list of box_list.BoxList object holding anchors generated.\n      feature_map_shape_list: list of (height, width) pairs in the format\n        [(height_0, width_0), (height_1, width_1), ...] that the generated\n        anchors must align with.\n    Returns:\n      Op that raises InvalidArgumentError if the number of anchors does not\n        match the number of expected anchors.\n    '
        expected_num_anchors = 0
        actual_num_anchors = 0
        for (num_anchors_per_location, feature_map_shape, anchors) in zip(self.num_anchors_per_location(), feature_map_shape_list, anchors_list):
            expected_num_anchors += num_anchors_per_location * feature_map_shape[0] * feature_map_shape[1]
            actual_num_anchors += anchors.num_boxes()
        return tf.assert_equal(expected_num_anchors, actual_num_anchors)