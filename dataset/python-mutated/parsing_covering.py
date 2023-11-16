"""Implementation of the Parsing Covering metric.

Parsing Covering is a region-based metric for evaluating the task of
image parsing, aka panoptic segmentation.

Please see the paper for details:
"DeeperLab: Single-Shot Image Parser", Tien-Ju Yang, Maxwell D. Collins,
Yukun Zhu, Jyh-Jing Hwang, Ting Liu, Xiao Zhang, Vivienne Sze,
George Papandreou, Liang-Chieh Chen. arXiv: 1902.05093, 2019.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import prettytable
import six
from deeplab.evaluation import base_metric

class ParsingCovering(base_metric.SegmentationMetric):
    """Metric class for Parsing Covering.

  Computes segmentation covering metric introduced in (Arbelaez, et al., 2010)
  with extension to handle multi-class semantic labels (a.k.a. parsing
  covering). Specifically, segmentation covering (SC) is defined in Eq. (8) in
  (Arbelaez et al., 2010) as:

  SC(c) = \\sum_{R\\in S}(|R| * \\max_{R'\\in S'}O(R,R')) / \\sum_{R\\in S}|R|,

  where S are the groundtruth instance regions and S' are the predicted
  instance regions. The parsing covering is simply:

  PC = \\sum_{c=1}^{C}SC(c) / C,

  where C is the number of classes.
  """

    def __init__(self, num_categories, ignored_label, max_instances_per_category, offset, normalize_by_image_size=True):
        if False:
            print('Hello World!')
        'Initialization for ParsingCovering.\n\n    Args:\n      num_categories: The number of segmentation categories (or "classes" in the\n        dataset.\n      ignored_label: A category id that is ignored in evaluation, e.g. the void\n        label as defined in COCO panoptic segmentation dataset.\n      max_instances_per_category: The maximum number of instances for each\n        category. Used in ensuring unique instance labels.\n      offset: The maximum number of unique labels. This is used, by multiplying\n        the ground-truth labels, to generate unique ids for individual regions\n        of overlap between groundtruth and predicted segments.\n      normalize_by_image_size: Whether to normalize groundtruth instance region\n        areas by image size. If True, groundtruth instance areas and weighted\n        IoUs will be divided by the size of the corresponding image before\n        accumulated across the dataset.\n    '
        super(ParsingCovering, self).__init__(num_categories, ignored_label, max_instances_per_category, offset)
        self.normalize_by_image_size = normalize_by_image_size

    def compare_and_accumulate(self, groundtruth_category_array, groundtruth_instance_array, predicted_category_array, predicted_instance_array):
        if False:
            print('Hello World!')
        'See base class.'
        max_ious = np.zeros([self.num_categories, self.max_instances_per_category], dtype=np.float64)
        gt_areas = np.zeros([self.num_categories, self.max_instances_per_category], dtype=np.float64)
        pred_areas = np.zeros([self.num_categories, self.max_instances_per_category], dtype=np.float64)
        intersections = collections.defaultdict(list)
        pred_segment_id = self._naively_combine_labels(predicted_category_array, predicted_instance_array)
        gt_segment_id = self._naively_combine_labels(groundtruth_category_array, groundtruth_instance_array)
        intersection_id_array = gt_segment_id.astype(np.uint32) * self.offset + pred_segment_id.astype(np.uint32)
        (intersection_ids, intersection_areas) = np.unique(intersection_id_array, return_counts=True)
        for (intersection_id, intersection_area) in six.moves.zip(intersection_ids, intersection_areas):
            gt_segment_id = intersection_id // self.offset
            gt_category = gt_segment_id // self.max_instances_per_category
            if gt_category == self.ignored_label:
                continue
            gt_instance = gt_segment_id % self.max_instances_per_category
            gt_areas[gt_category, gt_instance] += intersection_area
            pred_segment_id = intersection_id % self.offset
            pred_category = pred_segment_id // self.max_instances_per_category
            pred_instance = pred_segment_id % self.max_instances_per_category
            pred_areas[pred_category, pred_instance] += intersection_area
            if pred_category != gt_category:
                continue
            intersections[gt_category, gt_instance].append((pred_instance, intersection_area))
        for (gt_label, instance_intersections) in six.iteritems(intersections):
            (category, gt_instance) = gt_label
            gt_area = gt_areas[category, gt_instance]
            ious = []
            for (pred_instance, intersection_area) in instance_intersections:
                pred_area = pred_areas[category, pred_instance]
                union = gt_area + pred_area - intersection_area
                ious.append(intersection_area / union)
            max_ious[category, gt_instance] = max(ious)
        if self.normalize_by_image_size:
            gt_areas /= groundtruth_category_array.size
        self.weighted_iou_per_class += np.sum(max_ious * gt_areas, axis=-1)
        self.gt_area_per_class += np.sum(gt_areas, axis=-1)
        return self.result()

    def result_per_category(self):
        if False:
            while True:
                i = 10
        'See base class.'
        return base_metric.realdiv_maybe_zero(self.weighted_iou_per_class, self.gt_area_per_class)

    def _valid_categories(self):
        if False:
            for i in range(10):
                print('nop')
        'Categories with a "valid" value for the metric, have > 0 instances.\n\n    We will ignore the `ignore_label` class and other classes which have\n    groundtruth area of 0.\n\n    Returns:\n      Boolean array of shape `[num_categories]`.\n    '
        valid_categories = np.not_equal(self.gt_area_per_class, 0)
        if self.ignored_label >= 0 and self.ignored_label < self.num_categories:
            valid_categories[self.ignored_label] = False
        return valid_categories

    def detailed_results(self, is_thing=None):
        if False:
            print('Hello World!')
        'See base class.'
        valid_categories = self._valid_categories()
        category_sets = collections.OrderedDict()
        category_sets['All'] = valid_categories
        if is_thing is not None:
            category_sets['Things'] = np.logical_and(valid_categories, is_thing)
            category_sets['Stuff'] = np.logical_and(valid_categories, np.logical_not(is_thing))
        covering_per_class = self.result_per_category()
        results = {}
        for (category_set_name, in_category_set) in six.iteritems(category_sets):
            if np.any(in_category_set):
                results[category_set_name] = {'pc': np.mean(covering_per_class[in_category_set]), 'n': np.sum(in_category_set.astype(np.int32))}
            else:
                results[category_set_name] = {'pc': 0, 'n': 0}
        return results

    def print_detailed_results(self, is_thing=None, print_digits=3):
        if False:
            while True:
                i = 10
        'See base class.'
        results = self.detailed_results(is_thing=is_thing)
        tab = prettytable.PrettyTable()
        tab.add_column('', [], align='l')
        for fieldname in ['PC', 'N']:
            tab.add_column(fieldname, [], align='r')
        for (category_set, subset_results) in six.iteritems(results):
            data_cols = [round(subset_results['pc'], print_digits) * 100, subset_results['n']]
            tab.add_row([category_set] + data_cols)
        print(tab)

    def result(self):
        if False:
            print('Hello World!')
        'See base class.'
        covering_per_class = self.result_per_category()
        valid_categories = self._valid_categories()
        if not np.any(valid_categories):
            return 0.0
        return np.mean(covering_per_class[valid_categories])

    def merge(self, other_instance):
        if False:
            print('Hello World!')
        'See base class.'
        self.weighted_iou_per_class += other_instance.weighted_iou_per_class
        self.gt_area_per_class += other_instance.gt_area_per_class

    def reset(self):
        if False:
            while True:
                i = 10
        'See base class.'
        self.weighted_iou_per_class = np.zeros(self.num_categories, dtype=np.float64)
        self.gt_area_per_class = np.zeros(self.num_categories, dtype=np.float64)