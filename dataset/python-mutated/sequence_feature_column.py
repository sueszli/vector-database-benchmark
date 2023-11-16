"""This API defines FeatureColumn for sequential input.

NOTE: This API is a work in progress and will likely be changing frequently.
"""
import collections
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
_FEATURE_COLUMN_DEPRECATION_WARNING = '    Warning: tf.feature_column is not recommended for new code. Instead,\n    feature preprocessing can be done directly using either [Keras preprocessing\n    layers](https://www.tensorflow.org/guide/migrate/migrating_feature_columns)\n    or through the one-stop utility [`tf.keras.utils.FeatureSpace`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/FeatureSpace)\n    built on top of them. See the [migration guide](https://tensorflow.org/guide/migrate)\n    for details.\n    '
_FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING = 'Use Keras preprocessing layers instead, either directly or via the `tf.keras.utils.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.'

def concatenate_context_input(context_input, sequence_input):
    if False:
        print('Hello World!')
    'Replicates `context_input` across all timesteps of `sequence_input`.\n\n  Expands dimension 1 of `context_input` then tiles it `sequence_length` times.\n  This value is appended to `sequence_input` on dimension 2 and the result is\n  returned.\n\n  Args:\n    context_input: A `Tensor` of dtype `float32` and shape `[batch_size, d1]`.\n    sequence_input: A `Tensor` of dtype `float32` and shape `[batch_size,\n      padded_length, d0]`.\n\n  Returns:\n    A `Tensor` of dtype `float32` and shape `[batch_size, padded_length,\n    d0 + d1]`.\n\n  Raises:\n    ValueError: If `sequence_input` does not have rank 3 or `context_input` does\n      not have rank 2.\n  '
    seq_rank_check = check_ops.assert_rank(sequence_input, 3, message='sequence_input must have rank 3', data=[array_ops.shape(sequence_input)])
    seq_type_check = check_ops.assert_type(sequence_input, dtypes.float32, message='sequence_input must have dtype float32; got {}.'.format(sequence_input.dtype))
    ctx_rank_check = check_ops.assert_rank(context_input, 2, message='context_input must have rank 2', data=[array_ops.shape(context_input)])
    ctx_type_check = check_ops.assert_type(context_input, dtypes.float32, message='context_input must have dtype float32; got {}.'.format(context_input.dtype))
    with ops.control_dependencies([seq_rank_check, seq_type_check, ctx_rank_check, ctx_type_check]):
        padded_length = array_ops.shape(sequence_input)[1]
        tiled_context_input = array_ops.tile(array_ops.expand_dims(context_input, 1), array_ops.concat([[1], [padded_length], [1]], 0))
    return array_ops.concat([sequence_input, tiled_context_input], 2)

@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_identity')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_identity(key, num_buckets, default_value=None):
    if False:
        i = 10
        return i + 15
    "Returns a feature column that represents sequences of integers.\n\n  Pass this to `embedding_column` or `indicator_column` to convert sequence\n  categorical data into dense representation for input to sequence NN, such as\n  RNN.\n\n  Example:\n\n  ```python\n  watches = sequence_categorical_column_with_identity(\n      'watches', num_buckets=1000)\n  watches_embedding = embedding_column(watches, dimension=10)\n  columns = [watches_embedding]\n\n  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))\n  sequence_feature_layer = SequenceFeatures(columns)\n  sequence_input, sequence_length = sequence_feature_layer(features)\n  sequence_length_mask = tf.sequence_mask(sequence_length)\n\n  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)\n  rnn_layer = tf.keras.layers.RNN(rnn_cell)\n  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)\n  ```\n\n  Args:\n    key: A unique string identifying the input feature.\n    num_buckets: Range of inputs. Namely, inputs are expected to be in the range\n      `[0, num_buckets)`.\n    default_value: If `None`, this column's graph operations will fail for\n      out-of-range inputs. Otherwise, this value must be in the range `[0,\n      num_buckets)`, and will replace out-of-range inputs.\n\n  Returns:\n    A `SequenceCategoricalColumn`.\n\n  Raises:\n    ValueError: if `num_buckets` is less than one.\n    ValueError: if `default_value` is not in range `[0, num_buckets)`.\n  "
    return fc.SequenceCategoricalColumn(fc.categorical_column_with_identity(key=key, num_buckets=num_buckets, default_value=default_value))

@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_hash_bucket')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=dtypes.string):
    if False:
        return 10
    "A sequence of categorical terms where ids are set by hashing.\n\n  Pass this to `embedding_column` or `indicator_column` to convert sequence\n  categorical data into dense representation for input to sequence NN, such as\n  RNN.\n\n  Example:\n\n  ```python\n  tokens = sequence_categorical_column_with_hash_bucket(\n      'tokens', hash_bucket_size=1000)\n  tokens_embedding = embedding_column(tokens, dimension=10)\n  columns = [tokens_embedding]\n\n  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))\n  sequence_feature_layer = SequenceFeatures(columns)\n  sequence_input, sequence_length = sequence_feature_layer(features)\n  sequence_length_mask = tf.sequence_mask(sequence_length)\n\n  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)\n  rnn_layer = tf.keras.layers.RNN(rnn_cell)\n  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)\n  ```\n\n  Args:\n    key: A unique string identifying the input feature.\n    hash_bucket_size: An int > 1. The number of buckets.\n    dtype: The type of features. Only string and integer types are supported.\n\n  Returns:\n    A `SequenceCategoricalColumn`.\n\n  Raises:\n    ValueError: `hash_bucket_size` is not greater than 1.\n    ValueError: `dtype` is neither string nor integer.\n  "
    return fc.SequenceCategoricalColumn(fc.categorical_column_with_hash_bucket(key=key, hash_bucket_size=hash_bucket_size, dtype=dtype))

@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_vocabulary_file')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_vocabulary_file(key, vocabulary_file, vocabulary_size=None, num_oov_buckets=0, default_value=None, dtype=dtypes.string):
    if False:
        print('Hello World!')
    "A sequence of categorical terms where ids use a vocabulary file.\n\n  Pass this to `embedding_column` or `indicator_column` to convert sequence\n  categorical data into dense representation for input to sequence NN, such as\n  RNN.\n\n  Example:\n\n  ```python\n  states = sequence_categorical_column_with_vocabulary_file(\n      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,\n      num_oov_buckets=5)\n  states_embedding = embedding_column(states, dimension=10)\n  columns = [states_embedding]\n\n  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))\n  sequence_feature_layer = SequenceFeatures(columns)\n  sequence_input, sequence_length = sequence_feature_layer(features)\n  sequence_length_mask = tf.sequence_mask(sequence_length)\n\n  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)\n  rnn_layer = tf.keras.layers.RNN(rnn_cell)\n  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)\n  ```\n\n  Args:\n    key: A unique string identifying the input feature.\n    vocabulary_file: The vocabulary file name.\n    vocabulary_size: Number of the elements in the vocabulary. This must be no\n      greater than length of `vocabulary_file`, if less than length, later\n      values are ignored. If None, it is set to the length of `vocabulary_file`.\n    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary\n      buckets. All out-of-vocabulary inputs will be assigned IDs in the range\n      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of\n      the input value. A positive `num_oov_buckets` can not be specified with\n      `default_value`.\n    default_value: The integer ID value to return for out-of-vocabulary feature\n      values, defaults to `-1`. This can not be specified with a positive\n      `num_oov_buckets`.\n    dtype: The type of features. Only string and integer types are supported.\n\n  Returns:\n    A `SequenceCategoricalColumn`.\n\n  Raises:\n    ValueError: `vocabulary_file` is missing or cannot be opened.\n    ValueError: `vocabulary_size` is missing or < 1.\n    ValueError: `num_oov_buckets` is a negative integer.\n    ValueError: `num_oov_buckets` and `default_value` are both specified.\n    ValueError: `dtype` is neither string nor integer.\n  "
    return fc.SequenceCategoricalColumn(fc.categorical_column_with_vocabulary_file(key=key, vocabulary_file=vocabulary_file, vocabulary_size=vocabulary_size, num_oov_buckets=num_oov_buckets, default_value=default_value, dtype=dtype))

@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_vocabulary_list')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_vocabulary_list(key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0):
    if False:
        print('Hello World!')
    "A sequence of categorical terms where ids use an in-memory list.\n\n  Pass this to `embedding_column` or `indicator_column` to convert sequence\n  categorical data into dense representation for input to sequence NN, such as\n  RNN.\n\n  Example:\n\n  ```python\n  colors = sequence_categorical_column_with_vocabulary_list(\n      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),\n      num_oov_buckets=2)\n  colors_embedding = embedding_column(colors, dimension=3)\n  columns = [colors_embedding]\n\n  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))\n  sequence_feature_layer = SequenceFeatures(columns)\n  sequence_input, sequence_length = sequence_feature_layer(features)\n  sequence_length_mask = tf.sequence_mask(sequence_length)\n\n  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)\n  rnn_layer = tf.keras.layers.RNN(rnn_cell)\n  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)\n  ```\n\n  Args:\n    key: A unique string identifying the input feature.\n    vocabulary_list: An ordered iterable defining the vocabulary. Each feature\n      is mapped to the index of its value (if present) in `vocabulary_list`.\n      Must be castable to `dtype`.\n    dtype: The type of features. Only string and integer types are supported. If\n      `None`, it will be inferred from `vocabulary_list`.\n    default_value: The integer ID value to return for out-of-vocabulary feature\n      values, defaults to `-1`. This can not be specified with a positive\n      `num_oov_buckets`.\n    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary\n      buckets. All out-of-vocabulary inputs will be assigned IDs in the range\n      `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a\n      hash of the input value. A positive `num_oov_buckets` can not be specified\n      with `default_value`.\n\n  Returns:\n    A `SequenceCategoricalColumn`.\n\n  Raises:\n    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.\n    ValueError: `num_oov_buckets` is a negative integer.\n    ValueError: `num_oov_buckets` and `default_value` are both specified.\n    ValueError: if `dtype` is not integer or string.\n  "
    return fc.SequenceCategoricalColumn(fc.categorical_column_with_vocabulary_list(key=key, vocabulary_list=vocabulary_list, dtype=dtype, default_value=default_value, num_oov_buckets=num_oov_buckets))

@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_numeric_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_numeric_column(key, shape=(1,), default_value=0.0, dtype=dtypes.float32, normalizer_fn=None):
    if False:
        return 10
    "Returns a feature column that represents sequences of numeric data.\n\n  Example:\n\n  ```python\n  temperature = sequence_numeric_column('temperature')\n  columns = [temperature]\n\n  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))\n  sequence_feature_layer = SequenceFeatures(columns)\n  sequence_input, sequence_length = sequence_feature_layer(features)\n  sequence_length_mask = tf.sequence_mask(sequence_length)\n\n  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)\n  rnn_layer = tf.keras.layers.RNN(rnn_cell)\n  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)\n  ```\n\n  Args:\n    key: A unique string identifying the input features.\n    shape: The shape of the input data per sequence id. E.g. if `shape=(2,)`,\n      each example must contain `2 * sequence_length` values.\n    default_value: A single value compatible with `dtype` that is used for\n      padding the sparse data into a dense `Tensor`.\n    dtype: The type of values.\n    normalizer_fn: If not `None`, a function that can be used to normalize the\n      value of the tensor after `default_value` is applied for parsing.\n      Normalizer function takes the input `Tensor` as its argument, and returns\n      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that\n      even though the most common use case of this function is normalization, it\n      can be used for any kind of Tensorflow transformations.\n\n  Returns:\n    A `SequenceNumericColumn`.\n\n  Raises:\n    TypeError: if any dimension in shape is not an int.\n    ValueError: if any dimension in shape is not a positive integer.\n    ValueError: if `dtype` is not convertible to `tf.float32`.\n  "
    shape = fc._check_shape(shape=shape, key=key)
    if not (dtype.is_integer or dtype.is_floating):
        raise ValueError('dtype must be convertible to float. dtype: {}, key: {}'.format(dtype, key))
    if normalizer_fn is not None and (not callable(normalizer_fn)):
        raise TypeError('normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))
    return SequenceNumericColumn(key, shape=shape, default_value=default_value, dtype=dtype, normalizer_fn=normalizer_fn)

def _assert_all_equal_and_return(tensors, name=None):
    if False:
        return 10
    'Asserts that all tensors are equal and returns the first one.'
    with ops.name_scope(name, 'assert_all_equal', values=tensors):
        if len(tensors) == 1:
            return tensors[0]
        assert_equal_ops = []
        for t in tensors[1:]:
            assert_equal_ops.append(check_ops.assert_equal(tensors[0], t))
        with ops.control_dependencies(assert_equal_ops):
            return array_ops.identity(tensors[0])

@serialization.register_feature_column
class SequenceNumericColumn(fc.SequenceDenseColumn, collections.namedtuple('SequenceNumericColumn', ('key', 'shape', 'default_value', 'dtype', 'normalizer_fn'))):
    """Represents sequences of numeric data."""

    @property
    def _is_v2_column(self):
        if False:
            print('Hello World!')
        return True

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'See `FeatureColumn` base class.'
        return self.key

    @property
    def parse_example_spec(self):
        if False:
            for i in range(10):
                print('nop')
        'See `FeatureColumn` base class.'
        return {self.key: parsing_ops.VarLenFeature(self.dtype)}

    def transform_feature(self, transformation_cache, state_manager):
        if False:
            print('Hello World!')
        'See `FeatureColumn` base class.\n\n    In this case, we apply the `normalizer_fn` to the input tensor.\n\n    Args:\n      transformation_cache: A `FeatureTransformationCache` object to access\n        features.\n      state_manager: A `StateManager` to create / access resources such as\n        lookup tables.\n\n    Returns:\n      Normalized input tensor.\n    '
        input_tensor = transformation_cache.get(self.key, state_manager)
        if self.normalizer_fn is not None:
            input_tensor = self.normalizer_fn(input_tensor)
        return input_tensor

    @property
    def variable_shape(self):
        if False:
            print('Hello World!')
        'Returns a `TensorShape` representing the shape of sequence input.'
        return tensor_shape.TensorShape(self.shape)

    def get_sequence_dense_tensor(self, transformation_cache, state_manager):
        if False:
            while True:
                i = 10
        'Returns a `TensorSequenceLengthPair`.\n\n    Args:\n      transformation_cache: A `FeatureTransformationCache` object to access\n        features.\n      state_manager: A `StateManager` to create / access resources such as\n        lookup tables.\n    '
        sp_tensor = transformation_cache.get(self, state_manager)
        dense_tensor = sparse_ops.sparse_tensor_to_dense(sp_tensor, default_value=self.default_value)
        dense_shape = array_ops.concat([array_ops.shape(dense_tensor)[:1], [-1], self.variable_shape], axis=0)
        dense_tensor = array_ops.reshape(dense_tensor, shape=dense_shape)
        if sp_tensor.shape.ndims == 2:
            num_elements = self.variable_shape.num_elements()
        else:
            num_elements = 1
        seq_length = fc_utils.sequence_length_from_sparse_tensor(sp_tensor, num_elements=num_elements)
        return fc.SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=seq_length)

    @property
    def parents(self):
        if False:
            i = 10
            return i + 15
        "See 'FeatureColumn` base class."
        return [self.key]

    def get_config(self):
        if False:
            return 10
        "See 'FeatureColumn` base class."
        config = dict(zip(self._fields, self))
        config['dtype'] = self.dtype.name
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        if False:
            while True:
                i = 10
        "See 'FeatureColumn` base class."
        fc._check_config_keys(config, cls._fields)
        kwargs = fc._standardize_and_copy_config(config)
        kwargs['dtype'] = dtypes.as_dtype(config['dtype'])
        return cls(**kwargs)