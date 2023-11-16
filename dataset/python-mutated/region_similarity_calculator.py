"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from abc import ABCMeta
from abc import abstractmethod
import tensorflow.compat.v2 as tf

def area(boxlist, scope=None):
    if False:
        while True:
            i = 10
    'Computes area of boxes.\n\n  Args:\n    boxlist: BoxList holding N boxes\n    scope: name scope.\n\n  Returns:\n    a tensor with shape [N] representing box areas.\n  '
    if not scope:
        scope = 'Area'
    with tf.name_scope(scope):
        (y_min, x_min, y_max, x_max) = tf.split(value=boxlist.get(), num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def intersection(boxlist1, boxlist2, scope=None):
    if False:
        return 10
    'Compute pairwise intersection areas between boxes.\n\n  Args:\n    boxlist1: BoxList holding N boxes\n    boxlist2: BoxList holding M boxes\n    scope: name scope.\n\n  Returns:\n    a tensor with shape [N, M] representing pairwise intersections\n  '
    if not scope:
        scope = 'Intersection'
    with tf.name_scope(scope):
        (y_min1, x_min1, y_max1, x_max1) = tf.split(value=boxlist1.get(), num_or_size_splits=4, axis=1)
        (y_min2, x_min2, y_max2, x_max2) = tf.split(value=boxlist2.get(), num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(a=y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(a=y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(a=x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(a=x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths

def iou(boxlist1, boxlist2, scope=None):
    if False:
        print('Hello World!')
    'Computes pairwise intersection-over-union between box collections.\n\n  Args:\n    boxlist1: BoxList holding N boxes\n    boxlist2: BoxList holding M boxes\n    scope: name scope.\n\n  Returns:\n    a tensor with shape [N, M] representing pairwise iou scores.\n  '
    if not scope:
        scope = 'IOU'
    with tf.name_scope(scope):
        intersections = intersection(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)
        unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections
        return tf.where(tf.equal(intersections, 0.0), tf.zeros_like(intersections), tf.truediv(intersections, unions))

class RegionSimilarityCalculator(object):
    """Abstract base class for region similarity calculator."""
    __metaclass__ = ABCMeta

    def compare(self, boxlist1, boxlist2, scope=None):
        if False:
            return 10
        "Computes matrix of pairwise similarity between BoxLists.\n\n    This op (to be overriden) computes a measure of pairwise similarity between\n    the boxes in the given BoxLists. Higher values indicate more similarity.\n\n    Note that this method simply measures similarity and does not explicitly\n    perform a matching.\n\n    Args:\n      boxlist1: BoxList holding N boxes.\n      boxlist2: BoxList holding M boxes.\n      scope: Op scope name. Defaults to 'Compare' if None.\n\n    Returns:\n      a (float32) tensor of shape [N, M] with pairwise similarity score.\n    "
        if not scope:
            scope = 'Compare'
        with tf.name_scope(scope) as scope:
            return self._compare(boxlist1, boxlist2)

    @abstractmethod
    def _compare(self, boxlist1, boxlist2):
        if False:
            for i in range(10):
                print('nop')
        pass

class IouSimilarity(RegionSimilarityCalculator):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

  This class computes pairwise similarity between two BoxLists based on IOU.
  """

    def _compare(self, boxlist1, boxlist2):
        if False:
            while True:
                i = 10
        'Compute pairwise IOU similarity between the two BoxLists.\n\n    Args:\n      boxlist1: BoxList holding N boxes.\n      boxlist2: BoxList holding M boxes.\n\n    Returns:\n      A tensor with shape [N, M] representing pairwise iou scores.\n    '
        return iou(boxlist1, boxlist2)