"""Schema and transform definition for the Criteo dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_transform as tft

def _get_raw_categorical_column_name(column_idx):
    if False:
        i = 10
        return i + 15
    return 'categorical-feature-{}'.format(column_idx)

def get_transformed_categorical_column_name(column_name_or_id):
    if False:
        while True:
            i = 10
    if isinstance(column_name_or_id, bytes):
        column_name = column_name_or_id
    else:
        column_name = _get_raw_categorical_column_name(column_name_or_id)
    return column_name + '_id'
_INTEGER_COLUMN_NAMES = ['int-feature-{}'.format(column_idx) for column_idx in range(1, 14)]
_CATEGORICAL_COLUMN_NAMES = [_get_raw_categorical_column_name(column_idx) for column_idx in range(14, 40)]
DEFAULT_DELIMITER = '\t'
_NUM_BUCKETS = 10
tft.common.IS_ANNOTATIONS_PB_AVAILABLE = False

def make_ordered_column_names(include_label=True):
    if False:
        i = 10
        return i + 15
    'Returns the column names in the dataset in the order as they appear.\n\n  Args:\n    include_label: Indicates whether the label feature should be included.\n  Returns:\n    A list of column names in the dataset.\n  '
    result = ['clicked'] if include_label else []
    for name in _INTEGER_COLUMN_NAMES:
        result.append(name)
    for name in _CATEGORICAL_COLUMN_NAMES:
        result.append(name)
    return result

def make_legacy_input_feature_spec(include_label=True):
    if False:
        return 10
    'Input schema definition.\n\n  Args:\n    include_label: Indicates whether the label feature should be included.\n  Returns:\n    A `Schema` object.\n  '
    result = {}
    if include_label:
        result['clicked'] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    for name in _INTEGER_COLUMN_NAMES:
        result[name] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1)
    for name in _CATEGORICAL_COLUMN_NAMES:
        result[name] = tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value='')
    return result

def make_input_feature_spec(include_label=True):
    if False:
        print('Hello World!')
    'Input schema definition.\n\n  Args:\n    include_label: Indicates whether the label feature should be included.\n\n  Returns:\n    A `Schema` object.\n  '
    result = {}
    if include_label:
        result['clicked'] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    for name in _INTEGER_COLUMN_NAMES:
        result[name] = tf.io.VarLenFeature(dtype=tf.int64)
    for name in _CATEGORICAL_COLUMN_NAMES:
        result[name] = tf.io.VarLenFeature(dtype=tf.string)
    return result

def make_preprocessing_fn(frequency_threshold):
    if False:
        i = 10
        return i + 15
    'Creates a preprocessing function for criteo.\n\n  Args:\n    frequency_threshold: The frequency_threshold used when generating\n      vocabularies for the categorical features.\n\n  Returns:\n    A preprocessing function.\n  '

    def preprocessing_fn(inputs):
        if False:
            i = 10
            return i + 15
        'User defined preprocessing function for criteo columns.\n\n    Args:\n      inputs: dictionary of input `tensorflow_transform.Column`.\n    Returns:\n      A dictionary of `tensorflow_transform.Column` representing the transformed\n          columns.\n    '
        result = {'clicked': inputs['clicked']}
        for name in _INTEGER_COLUMN_NAMES:
            feature = inputs[name]
            feature = tft.sparse_tensor_to_dense_with_shape(feature, [None, 1], default_value=-1)
            feature = tf.squeeze(feature, axis=1)
            result[name] = feature
            result[name + '_bucketized'] = tft.bucketize(feature, _NUM_BUCKETS)
        for name in _CATEGORICAL_COLUMN_NAMES:
            feature = inputs[name]
            feature = tft.sparse_tensor_to_dense_with_shape(feature, [None, 1], default_value='')
            feature = tf.squeeze(feature, axis=1)
            result[get_transformed_categorical_column_name(name)] = tft.compute_and_apply_vocabulary(feature, frequency_threshold=frequency_threshold)
        return result
    return preprocessing_fn