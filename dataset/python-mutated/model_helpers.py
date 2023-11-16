"""Miscellaneous functions that can be called by models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numbers
import tensorflow as tf
from tensorflow.python.util import nest

def past_stop_threshold(stop_threshold, eval_metric):
    if False:
        while True:
            i = 10
    'Return a boolean representing whether a model should be stopped.\n\n  Args:\n    stop_threshold: float, the threshold above which a model should stop\n      training.\n    eval_metric: float, the current value of the relevant metric to check.\n\n  Returns:\n    True if training should stop, False otherwise.\n\n  Raises:\n    ValueError: if either stop_threshold or eval_metric is not a number\n  '
    if stop_threshold is None:
        return False
    if not isinstance(stop_threshold, numbers.Number):
        raise ValueError('Threshold for checking stop conditions must be a number.')
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError('Eval metric being checked against stop conditions must be a number.')
    if eval_metric >= stop_threshold:
        tf.compat.v1.logging.info('Stop threshold of {} was passed with metric value {}.'.format(stop_threshold, eval_metric))
        return True
    return False

def generate_synthetic_data(input_shape, input_value=0, input_dtype=None, label_shape=None, label_value=0, label_dtype=None):
    if False:
        while True:
            i = 10
    'Create a repeating dataset with constant values.\n\n  Args:\n    input_shape: a tf.TensorShape object or nested tf.TensorShapes. The shape of\n      the input data.\n    input_value: Value of each input element.\n    input_dtype: Input dtype. If None, will be inferred by the input value.\n    label_shape: a tf.TensorShape object or nested tf.TensorShapes. The shape of\n      the label data.\n    label_value: Value of each input element.\n    label_dtype: Input dtype. If None, will be inferred by the target value.\n\n  Returns:\n    Dataset of tensors or tuples of tensors (if label_shape is set).\n  '
    element = input_element = nest.map_structure(lambda s: tf.constant(input_value, input_dtype, s), input_shape)
    if label_shape:
        label_element = nest.map_structure(lambda s: tf.constant(label_value, label_dtype, s), label_shape)
        element = (input_element, label_element)
    return tf.data.Dataset.from_tensors(element).repeat()

def apply_clean(flags_obj):
    if False:
        return 10
    if flags_obj.clean and tf.io.gfile.exists(flags_obj.model_dir):
        tf.compat.v1.logging.info('--clean flag set. Removing existing model dir: {}'.format(flags_obj.model_dir))
        tf.io.gfile.rmtree(flags_obj.model_dir)