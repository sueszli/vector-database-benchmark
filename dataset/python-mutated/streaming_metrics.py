"""Code to compute segmentation in a "streaming" pattern in Tensorflow.

These aggregate the metric over examples of the evaluation set. Each example is
assumed to be fed in in a stream, and the metric implementation accumulates
across them.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from deeplab.evaluation import panoptic_quality
from deeplab.evaluation import parsing_covering
_EPSILON = 1e-10

def _realdiv_maybe_zero(x, y):
    if False:
        print('Hello World!')
    'Support tf.realdiv(x, y) where y may contain zeros.'
    return tf.where(tf.less(y, _EPSILON), tf.zeros_like(x), tf.realdiv(x, y))

def _running_total(value, shape, name=None):
    if False:
        print('Hello World!')
    'Maintains a running total of tensor `value` between calls.'
    with tf.variable_scope(name, 'running_total', [value]):
        total_var = tf.get_variable('total', shape, value.dtype, initializer=tf.zeros_initializer(), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])
        updated_total = tf.assign_add(total_var, value, use_locking=True)
    return (total_var, updated_total)

def _panoptic_quality_helper(groundtruth_category_array, groundtruth_instance_array, predicted_category_array, predicted_instance_array, num_classes, max_instances_per_category, ignored_label, offset):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to compute panoptic quality.'
    pq = panoptic_quality.PanopticQuality(num_classes, ignored_label, max_instances_per_category, offset)
    pq.compare_and_accumulate(groundtruth_category_array, groundtruth_instance_array, predicted_category_array, predicted_instance_array)
    return (pq.iou_per_class, pq.tp_per_class, pq.fn_per_class, pq.fp_per_class)

def streaming_panoptic_quality(groundtruth_categories, groundtruth_instances, predicted_categories, predicted_instances, num_classes, max_instances_per_category, ignored_label, offset, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Aggregates the panoptic metric across calls with different input tensors.\n\n  See tf.metrics.* functions for comparable functionality and usage.\n\n  Args:\n    groundtruth_categories: A 2D uint16 tensor of groundtruth category labels.\n    groundtruth_instances: A 2D uint16 tensor of groundtruth instance labels.\n    predicted_categories: A 2D uint16 tensor of predicted category labels.\n    predicted_instances: A 2D uint16 tensor of predicted instance labels.\n    num_classes: Number of classes in the dataset as an integer.\n    max_instances_per_category: The maximum number of instances for each class\n      as an integer or integer tensor.\n    ignored_label: The class id to be ignored in evaluation as an integer or\n      integer tensor.\n    offset: The maximum number of unique labels as an integer or integer tensor.\n    name: An optional variable_scope name.\n\n  Returns:\n    qualities: A tensor of shape `[6, num_classes]`, where (1) panoptic quality,\n      (2) segmentation quality, (3) recognition quality, (4) total_tp,\n      (5) total_fn and (6) total_fp are saved in the respective rows.\n    update_ops: List of operations that update the running overall panoptic\n      quality.\n\n  Raises:\n    RuntimeError: If eager execution is enabled.\n  '
    if tf.executing_eagerly():
        raise RuntimeError('Cannot aggregate when eager execution is enabled.')
    input_args = [tf.convert_to_tensor(groundtruth_categories, tf.uint16), tf.convert_to_tensor(groundtruth_instances, tf.uint16), tf.convert_to_tensor(predicted_categories, tf.uint16), tf.convert_to_tensor(predicted_instances, tf.uint16), tf.convert_to_tensor(num_classes, tf.int32), tf.convert_to_tensor(max_instances_per_category, tf.int32), tf.convert_to_tensor(ignored_label, tf.int32), tf.convert_to_tensor(offset, tf.int32)]
    return_types = [tf.float64, tf.float64, tf.float64, tf.float64]
    with tf.variable_scope(name, 'streaming_panoptic_quality', input_args):
        panoptic_results = tf.py_func(_panoptic_quality_helper, input_args, return_types, stateful=False)
        (iou, tp, fn, fp) = tuple(panoptic_results)
        (total_iou, updated_iou) = _running_total(iou, [num_classes], name='iou_total')
        (total_tp, updated_tp) = _running_total(tp, [num_classes], name='tp_total')
        (total_fn, updated_fn) = _running_total(fn, [num_classes], name='fn_total')
        (total_fp, updated_fp) = _running_total(fp, [num_classes], name='fp_total')
        update_ops = [updated_iou, updated_tp, updated_fn, updated_fp]
        sq = _realdiv_maybe_zero(total_iou, total_tp)
        rq = _realdiv_maybe_zero(total_tp, total_tp + 0.5 * total_fn + 0.5 * total_fp)
        pq = tf.multiply(sq, rq)
        qualities = tf.stack([pq, sq, rq, total_tp, total_fn, total_fp], axis=0)
    return (qualities, update_ops)

def _parsing_covering_helper(groundtruth_category_array, groundtruth_instance_array, predicted_category_array, predicted_instance_array, num_classes, max_instances_per_category, ignored_label, offset, normalize_by_image_size):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to compute parsing covering.'
    pc = parsing_covering.ParsingCovering(num_classes, ignored_label, max_instances_per_category, offset, normalize_by_image_size)
    pc.compare_and_accumulate(groundtruth_category_array, groundtruth_instance_array, predicted_category_array, predicted_instance_array)
    return (pc.weighted_iou_per_class, pc.gt_area_per_class)

def streaming_parsing_covering(groundtruth_categories, groundtruth_instances, predicted_categories, predicted_instances, num_classes, max_instances_per_category, ignored_label, offset, normalize_by_image_size=True, name=None):
    if False:
        print('Hello World!')
    'Aggregates the covering across calls with different input tensors.\n\n  See tf.metrics.* functions for comparable functionality and usage.\n\n  Args:\n    groundtruth_categories: A 2D uint16 tensor of groundtruth category labels.\n    groundtruth_instances: A 2D uint16 tensor of groundtruth instance labels.\n    predicted_categories: A 2D uint16 tensor of predicted category labels.\n    predicted_instances: A 2D uint16 tensor of predicted instance labels.\n    num_classes: Number of classes in the dataset as an integer.\n    max_instances_per_category: The maximum number of instances for each class\n      as an integer or integer tensor.\n    ignored_label: The class id to be ignored in evaluation as an integer or\n      integer tensor.\n    offset: The maximum number of unique labels as an integer or integer tensor.\n    normalize_by_image_size: Whether to normalize groundtruth region areas by\n      image size. If True, groundtruth instance areas and weighted IoUs will be\n      divided by the size of the corresponding image before accumulated across\n      the dataset.\n    name: An optional variable_scope name.\n\n  Returns:\n    coverings: A tensor of shape `[3, num_classes]`, where (1) per class\n      coverings, (2) per class sum of weighted IoUs, and (3) per class sum of\n      groundtruth region areas are saved in the perspective rows.\n    update_ops: List of operations that update the running overall parsing\n      covering.\n\n  Raises:\n    RuntimeError: If eager execution is enabled.\n  '
    if tf.executing_eagerly():
        raise RuntimeError('Cannot aggregate when eager execution is enabled.')
    input_args = [tf.convert_to_tensor(groundtruth_categories, tf.uint16), tf.convert_to_tensor(groundtruth_instances, tf.uint16), tf.convert_to_tensor(predicted_categories, tf.uint16), tf.convert_to_tensor(predicted_instances, tf.uint16), tf.convert_to_tensor(num_classes, tf.int32), tf.convert_to_tensor(max_instances_per_category, tf.int32), tf.convert_to_tensor(ignored_label, tf.int32), tf.convert_to_tensor(offset, tf.int32), tf.convert_to_tensor(normalize_by_image_size, tf.bool)]
    return_types = [tf.float64, tf.float64]
    with tf.variable_scope(name, 'streaming_parsing_covering', input_args):
        covering_results = tf.py_func(_parsing_covering_helper, input_args, return_types, stateful=False)
        (weighted_iou_per_class, gt_area_per_class) = tuple(covering_results)
        (total_weighted_iou_per_class, updated_weighted_iou_per_class) = _running_total(weighted_iou_per_class, [num_classes], name='weighted_iou_per_class_total')
        (total_gt_area_per_class, updated_gt_area_per_class) = _running_total(gt_area_per_class, [num_classes], name='gt_area_per_class_total')
        covering_per_class = _realdiv_maybe_zero(total_weighted_iou_per_class, total_gt_area_per_class)
        coverings = tf.stack([covering_per_class, total_weighted_iou_per_class, total_gt_area_per_class], axis=0)
        update_ops = [updated_weighted_iou_per_class, updated_gt_area_per_class]
    return (coverings, update_ops)