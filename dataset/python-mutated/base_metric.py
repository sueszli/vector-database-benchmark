"""Defines the top-level interface for evaluating segmentations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import numpy as np
import six
_EPSILON = 1e-10

def realdiv_maybe_zero(x, y):
    if False:
        print('Hello World!')
    'Element-wise x / y where y may contain zeros, for those returns 0 too.'
    return np.where(np.less(np.abs(y), _EPSILON), np.zeros_like(x), np.divide(x, y))

@six.add_metaclass(abc.ABCMeta)
class SegmentationMetric(object):
    """Abstract base class for computers of segmentation metrics.

  Subclasses will implement both:
  1. Comparing the predicted segmentation for an image with the groundtruth.
  2. Computing the final metric over a set of images.
  These are often done as separate steps, due to the need to accumulate
  intermediate values other than the metric itself across images, computing the
  actual metric value only on these accumulations after all the images have been
  compared.

  A simple usage would be:

    metric = MetricImplementation(...)
    for <image>, <groundtruth> in evaluation_set:
      <prediction> = run_segmentation(<image>)
      metric.compare_and_accumulate(<prediction>, <groundtruth>)
    print(metric.result())

  """

    def __init__(self, num_categories, ignored_label, max_instances_per_category, offset):
        if False:
            print('Hello World!')
        'Base initialization for SegmentationMetric.\n\n    Args:\n      num_categories: The number of segmentation categories (or "classes" in the\n        dataset.\n      ignored_label: A category id that is ignored in evaluation, e.g. the void\n        label as defined in COCO panoptic segmentation dataset.\n      max_instances_per_category: The maximum number of instances for each\n        category. Used in ensuring unique instance labels.\n      offset: The maximum number of unique labels. This is used, by multiplying\n        the ground-truth labels, to generate unique ids for individual regions\n        of overlap between groundtruth and predicted segments.\n    '
        self.num_categories = num_categories
        self.ignored_label = ignored_label
        self.max_instances_per_category = max_instances_per_category
        self.offset = offset
        self.reset()

    def _naively_combine_labels(self, category_array, instance_array):
        if False:
            while True:
                i = 10
        'Naively creates a combined label array from categories and instances.'
        return category_array.astype(np.uint32) * self.max_instances_per_category + instance_array.astype(np.uint32)

    @abc.abstractmethod
    def compare_and_accumulate(self, groundtruth_category_array, groundtruth_instance_array, predicted_category_array, predicted_instance_array):
        if False:
            i = 10
            return i + 15
        'Compares predicted segmentation with groundtruth, accumulates its metric.\n\n    It is not assumed that instance ids are unique across different categories.\n    See for example combine_semantic_and_instance_predictions.py in official\n    PanopticAPI evaluation code for issues to consider when fusing category\n    and instance labels.\n\n    Instances ids of the ignored category have the meaning that id 0 is "void"\n    and remaining ones are crowd instances.\n\n    Args:\n      groundtruth_category_array: A 2D numpy uint16 array of groundtruth\n        per-pixel category labels.\n      groundtruth_instance_array: A 2D numpy uint16 array of groundtruth\n        instance labels.\n      predicted_category_array: A 2D numpy uint16 array of predicted per-pixel\n        category labels.\n      predicted_instance_array: A 2D numpy uint16 array of predicted instance\n        labels.\n\n    Returns:\n      The value of the metric over all comparisons done so far, including this\n      one, as a float scalar.\n    '
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def result(self):
        if False:
            print('Hello World!')
        'Computes the metric over all comparisons done so far.'
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def detailed_results(self, is_thing=None):
        if False:
            for i in range(10):
                print('nop')
        'Computes and returns the detailed final metric results.\n\n    Args:\n      is_thing: A boolean array of length `num_categories`. The entry\n        `is_thing[category_id]` is True iff that category is a "thing" category\n        instead of "stuff."\n\n    Returns:\n      A dictionary with a breakdown of metrics and/or metric factors by things,\n      stuff, and all categories.\n    '
        raise NotImplementedError('Not implemented in subclasses.')

    @abc.abstractmethod
    def result_per_category(self):
        if False:
            return 10
        'For supported metrics, return individual per-category metric values.\n\n    Returns:\n      A numpy array of shape `[self.num_categories]`, where index `i` is the\n      metrics value over only that category.\n    '
        raise NotImplementedError('Not implemented in subclass.')

    def print_detailed_results(self, is_thing=None, print_digits=3):
        if False:
            i = 10
            return i + 15
        'Prints out a detailed breakdown of metric results.\n\n    Args:\n      is_thing: A boolean array of length num_categories.\n        `is_thing[category_id]` will say whether that category is a "thing"\n        rather than "stuff."\n      print_digits: Number of significant digits to print in computed metrics.\n    '
        raise NotImplementedError('Not implemented in subclass.')

    @abc.abstractmethod
    def merge(self, other_instance):
        if False:
            return 10
        'Combines the accumulated results of another instance into self.\n\n    The following two cases should put `metric_a` into an equivalent state.\n\n    Case 1 (with merge):\n\n      metric_a = MetricsSubclass(...)\n      metric_a.compare_and_accumulate(<comparison 1>)\n      metric_a.compare_and_accumulate(<comparison 2>)\n\n      metric_b = MetricsSubclass(...)\n      metric_b.compare_and_accumulate(<comparison 3>)\n      metric_b.compare_and_accumulate(<comparison 4>)\n\n      metric_a.merge(metric_b)\n\n    Case 2 (without merge):\n\n      metric_a = MetricsSubclass(...)\n      metric_a.compare_and_accumulate(<comparison 1>)\n      metric_a.compare_and_accumulate(<comparison 2>)\n      metric_a.compare_and_accumulate(<comparison 3>)\n      metric_a.compare_and_accumulate(<comparison 4>)\n\n    Args:\n      other_instance: Another compatible instance of the same metric subclass.\n    '
        raise NotImplementedError('Not implemented in subclass.')

    @abc.abstractmethod
    def reset(self):
        if False:
            while True:
                i = 10
        "Resets the accumulation to the metric class's state at initialization.\n\n    Note that this function will be called in SegmentationMetric.__init__.\n    "
        raise NotImplementedError('Must be implemented in subclasses.')