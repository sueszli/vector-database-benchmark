"""Parsing Ops."""
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops import parsing_grad
from tensorflow.python.ops.gen_parsing_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
VarLenFeature = parsing_config.VarLenFeature
RaggedFeature = parsing_config.RaggedFeature
SparseFeature = parsing_config.SparseFeature
FixedLenFeature = parsing_config.FixedLenFeature
FixedLenSequenceFeature = parsing_config.FixedLenSequenceFeature
_ParseOpParams = parsing_config._ParseOpParams
_construct_tensors_for_composite_features = parsing_config._construct_tensors_for_composite_features
_construct_sparse_tensors_for_sparse_features = _construct_tensors_for_composite_features

def _prepend_none_dimension(features):
    if False:
        return 10
    'Returns a copy of features with adjusted FixedLenSequenceFeature shapes.'
    if features:
        modified_features = dict(features)
        for (key, feature) in features.items():
            if isinstance(feature, FixedLenSequenceFeature):
                if not feature.allow_missing:
                    raise ValueError('Unsupported: FixedLenSequenceFeature requires allow_missing to be True.')
                modified_features[key] = FixedLenSequenceFeature([None] + list(feature.shape), feature.dtype, feature.allow_missing, feature.default_value)
        return modified_features
    else:
        return features

@tf_export('io.parse_example', v1=[])
@dispatch.add_dispatch_support
def parse_example_v2(serialized, features, example_names=None, name=None):
    if False:
        while True:
            i = 10
    'Parses `Example` protos into a `dict` of tensors.\n\n  Parses a number of serialized [`Example`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)\n  protos given in `serialized`. We refer to `serialized` as a batch with\n  `batch_size` many entries of individual `Example` protos.\n\n  `example_names` may contain descriptive names for the corresponding serialized\n  protos. These may be useful for debugging purposes, but they have no effect on\n  the output. If not `None`, `example_names` must be the same length as\n  `serialized`.\n\n  This op parses serialized examples into a dictionary mapping keys to `Tensor`\n  `SparseTensor`, and `RaggedTensor` objects. `features` is a Mapping from keys\n  to `VarLenFeature`, `SparseFeature`, `RaggedFeature`, and `FixedLenFeature`\n  objects. Each `VarLenFeature` and `SparseFeature` is mapped to a\n  `SparseTensor`; each `FixedLenFeature` is mapped to a `Tensor`; and each\n  `RaggedFeature` is mapped to a `RaggedTensor`.\n\n  Each `VarLenFeature` maps to a `SparseTensor` of the specified type\n  representing a ragged matrix. Its indices are `[batch, index]` where `batch`\n  identifies the example in `serialized`, and `index` is the value\'s index in\n  the list of values associated with that feature and example.\n\n  Each `SparseFeature` maps to a `SparseTensor` of the specified type\n  representing a Tensor of `dense_shape` `[batch_size] + SparseFeature.size`.\n  Its `values` come from the feature in the examples with key `value_key`.\n  A `values[i]` comes from a position `k` in the feature of an example at batch\n  entry `batch`. This positional information is recorded in `indices[i]` as\n  `[batch, index_0, index_1, ...]` where `index_j` is the `k-th` value of\n  the feature in the example at with key `SparseFeature.index_key[j]`.\n  In other words, we split the indices (except the first index indicating the\n  batch entry) of a `SparseTensor` by dimension into different features of the\n  `Example`. Due to its complexity a `VarLenFeature` should be preferred over a\n  `SparseFeature` whenever possible.\n\n  Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or\n  `tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.\n\n  `FixedLenFeature` entries with a `default_value` are optional. With no default\n  value, we will fail if that `Feature` is missing from any example in\n  `serialized`.\n\n  Each `FixedLenSequenceFeature` `df` maps to a `Tensor` of the specified type\n  (or `tf.float32` if not specified) and shape\n  `(serialized.size(), None) + df.shape`.\n  All examples in `serialized` will be padded with `default_value` along the\n  second dimension.\n\n  Each `RaggedFeature` maps to a `RaggedTensor` of the specified type.  It\n  is formed by stacking the `RaggedTensor` for each example, where the\n  `RaggedTensor` for each individual example is constructed using the tensors\n  specified by `RaggedTensor.values_key` and `RaggedTensor.partition`.  See\n  the `tf.io.RaggedFeature` documentation for details and examples.\n\n  Examples:\n\n  For example, if one expects a `tf.float32` `VarLenFeature` `ft` and three\n  serialized `Example`s are provided:\n\n  ```\n  serialized = [\n    features\n      { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },\n    features\n      { feature []},\n    features\n      { feature { key: "ft" value { float_list { value: [3.0] } } }\n  ]\n  ```\n\n  then the output will look like:\n\n  ```python\n  {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],\n                      values=[1.0, 2.0, 3.0],\n                      dense_shape=(3, 2)) }\n  ```\n\n  If instead a `FixedLenSequenceFeature` with `default_value = -1.0` and\n  `shape=[]` is used then the output will look like:\n\n  ```python\n  {"ft": [[1.0, 2.0], [3.0, -1.0]]}\n  ```\n\n  Given two `Example` input protos in `serialized`:\n\n  ```\n  [\n    features {\n      feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }\n      feature { key: "gps" value { float_list { value: [] } } }\n    },\n    features {\n      feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }\n      feature { key: "dank" value { int64_list { value: [ 42 ] } } }\n      feature { key: "gps" value { } }\n    }\n  ]\n  ```\n\n  And arguments\n\n  ```\n  example_names: ["input0", "input1"],\n  features: {\n      "kw": VarLenFeature(tf.string),\n      "dank": VarLenFeature(tf.int64),\n      "gps": VarLenFeature(tf.float32),\n  }\n  ```\n\n  Then the output is a dictionary:\n\n  ```python\n  {\n    "kw": SparseTensor(\n        indices=[[0, 0], [0, 1], [1, 0]],\n        values=["knit", "big", "emmy"]\n        dense_shape=[2, 2]),\n    "dank": SparseTensor(\n        indices=[[1, 0]],\n        values=[42],\n        dense_shape=[2, 1]),\n    "gps": SparseTensor(\n        indices=[],\n        values=[],\n        dense_shape=[2, 0]),\n  }\n  ```\n\n  For dense results in two serialized `Example`s:\n\n  ```\n  [\n    features {\n      feature { key: "age" value { int64_list { value: [ 0 ] } } }\n      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }\n     },\n     features {\n      feature { key: "age" value { int64_list { value: [] } } }\n      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }\n    }\n  ]\n  ```\n\n  We can use arguments:\n\n  ```\n  example_names: ["input0", "input1"],\n  features: {\n      "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),\n      "gender": FixedLenFeature([], dtype=tf.string),\n  }\n  ```\n\n  And the expected output is:\n\n  ```python\n  {\n    "age": [[0], [-1]],\n    "gender": [["f"], ["f"]],\n  }\n  ```\n\n  An alternative to `VarLenFeature` to obtain a `SparseTensor` is\n  `SparseFeature`. For example, given two `Example` input protos in\n  `serialized`:\n\n  ```\n  [\n    features {\n      feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }\n      feature { key: "ix" value { int64_list { value: [ 3, 20 ] } } }\n    },\n    features {\n      feature { key: "val" value { float_list { value: [ 0.0 ] } } }\n      feature { key: "ix" value { int64_list { value: [ 42 ] } } }\n    }\n  ]\n  ```\n\n  And arguments\n\n  ```\n  example_names: ["input0", "input1"],\n  features: {\n      "sparse": SparseFeature(\n          index_key="ix", value_key="val", dtype=tf.float32, size=100),\n  }\n  ```\n\n  Then the output is a dictionary:\n\n  ```python\n  {\n    "sparse": SparseTensor(\n        indices=[[0, 3], [0, 20], [1, 42]],\n        values=[0.5, -1.0, 0.0]\n        dense_shape=[2, 100]),\n  }\n  ```\n\n  See the `tf.io.RaggedFeature` documentation for examples showing how\n  `RaggedFeature` can be used to obtain `RaggedTensor`s.\n\n  Args:\n    serialized: A vector (1-D Tensor) of strings, a batch of binary\n      serialized `Example` protos.\n    features: A mapping of feature keys to `FixedLenFeature`,\n      `VarLenFeature`, `SparseFeature`, and `RaggedFeature` values.\n    example_names: A vector (1-D Tensor) of strings (optional), the names of\n      the serialized protos in the batch.\n    name: A name for this operation (optional).\n\n  Returns:\n    A `dict` mapping feature keys to `Tensor`, `SparseTensor`, and\n    `RaggedTensor` values.\n\n  Raises:\n    ValueError: if any feature is invalid.\n  '
    if not features:
        raise ValueError('Argument `features` cannot be None or falsy. Got %s' % features)
    features = _prepend_none_dimension(features)
    params = _ParseOpParams.from_features(features, [VarLenFeature, SparseFeature, FixedLenFeature, FixedLenSequenceFeature, RaggedFeature])
    outputs = _parse_example_raw(serialized, example_names, params, name=name)
    return _construct_tensors_for_composite_features(features, outputs)

@tf_export(v1=['io.parse_example', 'parse_example'])
@dispatch.add_dispatch_support
def parse_example(serialized, features, name=None, example_names=None):
    if False:
        print('Hello World!')
    return parse_example_v2(serialized, features, example_names, name)
parse_example.__doc__ = parse_example_v2.__doc__

def _parse_example_raw(serialized, names, params, name):
    if False:
        print('Hello World!')
    'Parses `Example` protos.\n\n  Args:\n    serialized: A vector (1-D Tensor) of strings, a batch of binary\n      serialized `Example` protos.\n    names: A vector (1-D Tensor) of strings (optional), the names of\n      the serialized protos.\n    params: A `ParseOpParams` containing the parameters for the parse op.\n    name: A name for this operation (optional).\n\n  Returns:\n    A `dict` mapping keys to `Tensor`s and `SparseTensor`s and `RaggedTensor`s.\n\n  '
    if params.num_features == 0:
        raise ValueError('Must provide at least one feature key.')
    with ops.name_scope(name, 'ParseExample', [serialized, names]):
        names = [] if names is None else names
        serialized = ops.convert_to_tensor(serialized, name='serialized')
        if params.ragged_keys and serialized.shape.ndims is None:
            raise ValueError('serialized must have statically-known rank to parse ragged features.')
        outputs = gen_parsing_ops.parse_example_v2(serialized=serialized, names=names, sparse_keys=params.sparse_keys, dense_keys=params.dense_keys, ragged_keys=params.ragged_keys, dense_defaults=params.dense_defaults_vec, num_sparse=len(params.sparse_keys), sparse_types=params.sparse_types, ragged_value_types=params.ragged_value_types, ragged_split_types=params.ragged_split_types, dense_shapes=params.dense_shapes_as_proto, name=name)
        (sparse_indices, sparse_values, sparse_shapes, dense_values, ragged_values, ragged_row_splits) = outputs
        ragged_tensors = parsing_config._build_ragged_tensors(serialized.shape, ragged_values, ragged_row_splits)
        sparse_tensors = [sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(sparse_indices, sparse_values, sparse_shapes)]
        return dict(zip(params.sparse_keys + params.dense_keys + params.ragged_keys, sparse_tensors + dense_values + ragged_tensors))

@tf_export(v1=['io.parse_single_example', 'parse_single_example'])
@dispatch.add_dispatch_support
def parse_single_example(serialized, features, name=None, example_names=None):
    if False:
        for i in range(10):
            print('nop')
    'Parses a single `Example` proto.\n\n  Similar to `parse_example`, except:\n\n  For dense tensors, the returned `Tensor` is identical to the output of\n  `parse_example`, except there is no batch dimension, the output shape is the\n  same as the shape given in `dense_shape`.\n\n  For `SparseTensor`s, the first (batch) column of the indices matrix is removed\n  (the indices matrix is a column vector), the values vector is unchanged, and\n  the first (`batch_size`) entry of the shape vector is removed (it is now a\n  single element vector).\n\n  One might see performance advantages by batching `Example` protos with\n  `parse_example` instead of using this function directly.\n\n  Args:\n    serialized: A scalar string Tensor, a single serialized Example.\n    features: A mapping of feature keys to `FixedLenFeature` or\n      `VarLenFeature` values.\n    name: A name for this operation (optional).\n    example_names: (Optional) A scalar string Tensor, the associated name.\n\n  Returns:\n    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.\n\n  Raises:\n    ValueError: if any feature is invalid.\n  '
    return parse_single_example_v2(serialized, features, example_names, name)

@tf_export('io.parse_single_example', v1=[])
@dispatch.add_dispatch_support
def parse_single_example_v2(serialized, features, example_names=None, name=None):
    if False:
        while True:
            i = 10
    'Parses a single `Example` proto.\n\n  Similar to `parse_example`, except:\n\n  For dense tensors, the returned `Tensor` is identical to the output of\n  `parse_example`, except there is no batch dimension, the output shape is the\n  same as the shape given in `dense_shape`.\n\n  For `SparseTensor`s, the first (batch) column of the indices matrix is removed\n  (the indices matrix is a column vector), the values vector is unchanged, and\n  the first (`batch_size`) entry of the shape vector is removed (it is now a\n  single element vector).\n\n  One might see performance advantages by batching `Example` protos with\n  `parse_example` instead of using this function directly.\n\n  Args:\n    serialized: A scalar string Tensor, a single serialized Example.\n    features: A mapping of feature keys to `FixedLenFeature` or\n      `VarLenFeature` values.\n    example_names: (Optional) A scalar string Tensor, the associated name.\n    name: A name for this operation (optional).\n\n  Returns:\n    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.\n\n  Raises:\n    ValueError: if any feature is invalid.\n  '
    if not features:
        raise ValueError('Invalid argument: features cannot be None.')
    with ops.name_scope(name, 'ParseSingleExample', [serialized, example_names]):
        serialized = ops.convert_to_tensor(serialized, name='serialized')
        serialized = _assert_scalar(serialized, 'serialized')
        return parse_example_v2(serialized, features, example_names, name)

@tf_export('io.parse_sequence_example')
@dispatch.add_dispatch_support
def parse_sequence_example(serialized, context_features=None, sequence_features=None, example_names=None, name=None):
    if False:
        i = 10
        return i + 15
    "Parses a batch of `SequenceExample` protos.\n\n  Parses a vector of serialized\n  [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)\n  protos given in `serialized`.\n\n  This op parses serialized sequence examples into a tuple of dictionaries,\n  each mapping keys to `Tensor` and `SparseTensor` objects.\n  The first dictionary contains mappings for keys appearing in\n  `context_features`, and the second dictionary contains mappings for keys\n  appearing in `sequence_features`.\n\n  At least one of `context_features` and `sequence_features` must be provided\n  and non-empty.\n\n  The `context_features` keys are associated with a `SequenceExample` as a\n  whole, independent of time / frame.  In contrast, the `sequence_features` keys\n  provide a way to access variable-length data within the `FeatureList` section\n  of the `SequenceExample` proto.  While the shapes of `context_features` values\n  are fixed with respect to frame, the frame dimension (the first dimension)\n  of `sequence_features` values may vary between `SequenceExample` protos,\n  and even between `feature_list` keys within the same `SequenceExample`.\n\n  `context_features` contains `VarLenFeature`, `RaggedFeature`, and\n  `FixedLenFeature`  objects. Each `VarLenFeature` is mapped to a\n  `SparseTensor`; each `RaggedFeature` is  mapped to a `RaggedTensor`; and each\n  `FixedLenFeature` is mapped to a `Tensor`, of the specified type, shape, and\n  default value.\n\n  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and\n  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a\n  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and\n  each `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified\n  type. The shape will be `(B,T,) + df.dense_shape` for\n  `FixedLenSequenceFeature` `df`, where `B` is the batch size, and `T` is the\n  length of the associated `FeatureList` in the `SequenceExample`. For instance,\n  `FixedLenSequenceFeature([])` yields a scalar 2-D `Tensor` of static shape\n  `[None, None]` and dynamic shape `[B, T]`, while\n  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 3-D matrix `Tensor`\n  of static shape `[None, None, k]` and dynamic shape `[B, T, k]`.\n\n  Like the input, the resulting output tensors have a batch dimension. This\n  means that the original per-example shapes of `VarLenFeature`s and\n  `FixedLenSequenceFeature`s can be lost. To handle that situation, this op also\n  provides dicts of shape tensors as part of the output. There is one dict for\n  the context features, and one for the feature_list features. Context features\n  of type `FixedLenFeature`s will not be present, since their shapes are already\n  known by the caller. In situations where the input `FixedLenSequenceFeature`s\n  are of different sequence lengths across examples, the shorter examples will\n  be padded with default datatype values: 0 for numeric types, and the empty\n  string for string types.\n\n  Each `SparseTensor` corresponding to `sequence_features` represents a ragged\n  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`\n  entry and `index` is the value's index in the list of values associated with\n  that time.\n\n  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`\n  entries with `allow_missing=True` are optional; otherwise, we will fail if\n  that `Feature` or `FeatureList` is missing from any example in `serialized`.\n\n  `example_name` may contain a descriptive name for the corresponding serialized\n  proto. This may be useful for debugging purposes, but it has no effect on the\n  output. If not `None`, `example_name` must be a scalar.\n\n  Args:\n    serialized: A vector (1-D Tensor) of type string containing binary\n      serialized `SequenceExample` protos.\n    context_features: A mapping of feature keys to `FixedLenFeature` or\n      `VarLenFeature` or `RaggedFeature` values. These features are associated\n      with a `SequenceExample` as a whole.\n    sequence_features: A mapping of feature keys to\n      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.\n      These features are associated with data within the `FeatureList` section\n      of the `SequenceExample` proto.\n    example_names: A vector (1-D Tensor) of strings (optional), the name of the\n      serialized protos.\n    name: A name for this operation (optional).\n\n  Returns:\n    A tuple of three `dict`s, each mapping keys to `Tensor`s,\n    `SparseTensor`s, and `RaggedTensor`. The first dict contains the context\n    key/values, the second dict contains the feature_list key/values, and the\n    final dict contains the lengths of any dense feature_list features.\n\n  Raises:\n    ValueError: if any feature is invalid.\n  "
    if not (context_features or sequence_features):
        raise ValueError('Both `context_features` and `sequence_features` argument are None, but at least one should have values.')
    context_params = _ParseOpParams.from_features(context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
    feature_list_params = _ParseOpParams.from_features(sequence_features, [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])
    with ops.name_scope(name, 'ParseSequenceExample', [serialized, example_names]):
        outputs = _parse_sequence_example_raw(serialized, example_names, context_params, feature_list_params, name)
        (context_output, feature_list_output, feature_list_lengths) = outputs
        if context_params.ragged_keys:
            context_output = _construct_tensors_for_composite_features(context_features, context_output)
        if feature_list_params.ragged_keys:
            feature_list_output = _construct_tensors_for_composite_features(sequence_features, feature_list_output)
        return (context_output, feature_list_output, feature_list_lengths)

def _parse_sequence_example_raw(serialized, debug_name, context, feature_list, name=None):
    if False:
        while True:
            i = 10
    'Parses a vector of `SequenceExample` protos.\n\n  Args:\n    serialized: A vector (1-D Tensor) of type string, containing binary\n      serialized `SequenceExample` protos.\n    debug_name: A vector (1-D Tensor) of strings (optional), the names of the\n      serialized protos.\n    context: A `ParseOpParams` containing the parameters for the parse\n      op for the context features.\n    feature_list: A `ParseOpParams` containing the parameters for the\n      parse op for the feature_list features.\n    name: A name for this operation (optional).\n\n  Returns:\n    A tuple of three `dict`s, each mapping keys to `Tensor`s, `SparseTensor`s,\n    and `RaggedTensor`s. The first dict contains the context key/values, the\n    second dict contains the feature_list key/values, and the final dict\n    contains the lengths of any dense feature_list features.\n\n  Raises:\n    TypeError: if feature_list.dense_defaults is not either None or a dict.\n  '
    if context.num_features + feature_list.num_features == 0:
        raise ValueError('Must provide at least one feature key.')
    with ops.name_scope(name, 'ParseSequenceExample', [serialized]):
        debug_name = [] if debug_name is None else debug_name
        feature_list_dense_missing_assumed_empty = []
        for (k, v) in feature_list.dense_defaults.items():
            if v is not None:
                raise ValueError('Value feature_list.dense_defaults[%s] must be None' % k)
            feature_list_dense_missing_assumed_empty.append(k)
        has_ragged = context.ragged_keys or feature_list.ragged_keys
        serialized = ops.convert_to_tensor(serialized, name='serialized')
        if has_ragged and serialized.shape.ndims is None:
            raise ValueError('serialized must have statically-known rank to parse ragged features.')
        feature_list_dense_missing_assumed_empty_vector = [key in feature_list_dense_missing_assumed_empty for key in feature_list.dense_keys]
        outputs = gen_parsing_ops.parse_sequence_example_v2(serialized=serialized, debug_name=debug_name, context_sparse_keys=context.sparse_keys, context_dense_keys=context.dense_keys, context_ragged_keys=context.ragged_keys, feature_list_sparse_keys=feature_list.sparse_keys, feature_list_dense_keys=feature_list.dense_keys, feature_list_ragged_keys=feature_list.ragged_keys, feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty_vector, context_dense_defaults=context.dense_defaults_vec, Ncontext_sparse=len(context.sparse_keys), Nfeature_list_sparse=len(feature_list.sparse_keys), Nfeature_list_dense=len(feature_list.dense_keys), context_sparse_types=context.sparse_types, context_ragged_value_types=context.ragged_value_types, context_ragged_split_types=context.ragged_split_types, feature_list_dense_types=feature_list.dense_types, feature_list_sparse_types=feature_list.sparse_types, feature_list_ragged_value_types=feature_list.ragged_value_types, feature_list_ragged_split_types=feature_list.ragged_split_types, context_dense_shapes=context.dense_shapes_as_proto, feature_list_dense_shapes=feature_list.dense_shapes, name=name)
        (context_sparse_indices, context_sparse_values, context_sparse_shapes, context_dense_values, context_ragged_values, context_ragged_row_splits, feature_list_sparse_indices, feature_list_sparse_values, feature_list_sparse_shapes, feature_list_dense_values, feature_list_dense_lengths, feature_list_ragged_values, feature_list_ragged_outer_splits, feature_list_ragged_inner_splits) = outputs
        context_ragged_tensors = parsing_config._build_ragged_tensors(serialized.shape, context_ragged_values, context_ragged_row_splits)
        feature_list_ragged_tensors = parsing_config._build_ragged_tensors(serialized.shape, feature_list_ragged_values, feature_list_ragged_outer_splits, feature_list_ragged_inner_splits)
        context_sparse_tensors = [sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(context_sparse_indices, context_sparse_values, context_sparse_shapes)]
        feature_list_sparse_tensors = [sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(feature_list_sparse_indices, feature_list_sparse_values, feature_list_sparse_shapes)]
        context_output = dict(zip(context.sparse_keys + context.dense_keys + context.ragged_keys, context_sparse_tensors + context_dense_values + context_ragged_tensors))
        feature_list_output = dict(zip(feature_list.sparse_keys + feature_list.dense_keys + feature_list.ragged_keys, feature_list_sparse_tensors + feature_list_dense_values + feature_list_ragged_tensors))
        feature_list_lengths = dict(zip(feature_list.dense_keys, feature_list_dense_lengths))
        return (context_output, feature_list_output, feature_list_lengths)

@tf_export('io.parse_single_sequence_example', v1=['io.parse_single_sequence_example', 'parse_single_sequence_example'])
@dispatch.add_dispatch_support
def parse_single_sequence_example(serialized, context_features=None, sequence_features=None, example_name=None, name=None):
    if False:
        i = 10
        return i + 15
    "Parses a single `SequenceExample` proto.\n\n  Parses a single serialized [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)\n  proto given in `serialized`.\n\n  This op parses a serialized sequence example into a tuple of dictionaries,\n  each mapping keys to `Tensor` and `SparseTensor` objects.\n  The first dictionary contains mappings for keys appearing in\n  `context_features`, and the second dictionary contains mappings for keys\n  appearing in `sequence_features`.\n\n  At least one of `context_features` and `sequence_features` must be provided\n  and non-empty.\n\n  The `context_features` keys are associated with a `SequenceExample` as a\n  whole, independent of time / frame.  In contrast, the `sequence_features` keys\n  provide a way to access variable-length data within the `FeatureList` section\n  of the `SequenceExample` proto.  While the shapes of `context_features` values\n  are fixed with respect to frame, the frame dimension (the first dimension)\n  of `sequence_features` values may vary between `SequenceExample` protos,\n  and even between `feature_list` keys within the same `SequenceExample`.\n\n  `context_features` contains `VarLenFeature`, `RaggedFeature`, and\n  `FixedLenFeature` objects. Each `VarLenFeature` is mapped to a `SparseTensor`;\n  each `RaggedFeature` is mapped to a `RaggedTensor`; and each `FixedLenFeature`\n  is mapped to a `Tensor`, of the specified type, shape, and default value.\n\n  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and\n  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a\n  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and each\n  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.\n  The shape will be `(T,) + df.dense_shape` for `FixedLenSequenceFeature` `df`,\n  where `T` is the length of the associated `FeatureList` in the\n  `SequenceExample`. For instance, `FixedLenSequenceFeature([])` yields a scalar\n  1-D `Tensor` of static shape `[None]` and dynamic shape `[T]`, while\n  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 2-D matrix `Tensor`\n  of static shape `[None, k]` and dynamic shape `[T, k]`.\n\n  Each `SparseTensor` corresponding to `sequence_features` represents a ragged\n  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`\n  entry and `index` is the value's index in the list of values associated with\n  that time.\n\n  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`\n  entries with `allow_missing=True` are optional; otherwise, we will fail if\n  that `Feature` or `FeatureList` is missing from any example in `serialized`.\n\n  `example_name` may contain a descriptive name for the corresponding serialized\n  proto. This may be useful for debugging purposes, but it has no effect on the\n  output. If not `None`, `example_name` must be a scalar.\n\n  Note that the batch version of this function, `tf.parse_sequence_example`,\n  is written for better memory efficiency and will be faster on large\n  `SequenceExample`s.\n\n  Args:\n    serialized: A scalar (0-D Tensor) of type string, a single binary\n      serialized `SequenceExample` proto.\n    context_features: A mapping of feature keys to `FixedLenFeature` or\n      `VarLenFeature` or `RaggedFeature` values. These features are associated\n      with a `SequenceExample` as a whole.\n    sequence_features: A mapping of feature keys to\n      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.\n      These features are associated with data within the `FeatureList` section\n      of the `SequenceExample` proto.\n    example_name: A scalar (0-D Tensor) of strings (optional), the name of\n      the serialized proto.\n    name: A name for this operation (optional).\n\n  Returns:\n    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s\n    and `RaggedTensor`s.\n\n    * The first dict contains the context key/values.\n    * The second dict contains the feature_list key/values.\n\n  Raises:\n    ValueError: if any feature is invalid.\n  "
    if not (context_features or sequence_features):
        raise ValueError('Both context_features and sequence_features are None, but at least one should have values.')
    context_params = _ParseOpParams.from_features(context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
    feature_list_params = _ParseOpParams.from_features(sequence_features, [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])
    with ops.name_scope(name, 'ParseSingleSequenceExample', [serialized, example_name]):
        (context_output, feature_list_output) = _parse_single_sequence_example_raw(serialized, context_params, feature_list_params, example_name, name)
        if context_params.ragged_keys:
            context_output = _construct_tensors_for_composite_features(context_features, context_output)
        if feature_list_params.ragged_keys:
            feature_list_output = _construct_tensors_for_composite_features(sequence_features, feature_list_output)
        return (context_output, feature_list_output)

def _parse_single_sequence_example_raw(serialized, context, feature_list, debug_name, name=None):
    if False:
        return 10
    'Parses a single `SequenceExample` proto.\n\n  Args:\n    serialized: A scalar (0-D Tensor) of type string, a single binary serialized\n      `SequenceExample` proto.\n    context: A `ParseOpParams` containing the parameters for the parse op for\n      the context features.\n    feature_list: A `ParseOpParams` containing the parameters for the parse op\n      for the feature_list features.\n    debug_name: A scalar (0-D Tensor) of strings (optional), the name of the\n      serialized proto.\n    name: A name for this operation (optional).\n\n  Returns:\n    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.\n    The first dict contains the context key/values.\n    The second dict contains the feature_list key/values.\n\n  Raises:\n    TypeError: if feature_list.dense_defaults is not either None or a dict.\n  '
    with ops.name_scope(name, 'ParseSingleExample', [serialized, debug_name]):
        serialized = ops.convert_to_tensor(serialized, name='serialized')
        serialized = _assert_scalar(serialized, 'serialized')
    return _parse_sequence_example_raw(serialized, debug_name, context, feature_list, name)[:2]

@tf_export('io.decode_raw', v1=[])
@dispatch.add_dispatch_support
def decode_raw(input_bytes, out_type, little_endian=True, fixed_length=None, name=None):
    if False:
        return 10
    'Convert raw bytes from input tensor into numeric tensors.\n\n  Every component of the input tensor is interpreted as a sequence of bytes.\n  These bytes are then decoded as numbers in the format specified by `out_type`.\n\n  >>> tf.io.decode_raw(tf.constant("1"), tf.uint8)\n  <tf.Tensor: shape=(1,), dtype=uint8, numpy=array([49], dtype=uint8)>\n  >>> tf.io.decode_raw(tf.constant("1,2"), tf.uint8)\n  <tf.Tensor: shape=(3,), dtype=uint8, numpy=array([49, 44, 50], dtype=uint8)>\n\n  Note that the rank of the output tensor is always one more than the input one:\n\n  >>> tf.io.decode_raw(tf.constant(["1","2"]), tf.uint8).shape\n  TensorShape([2, 1])\n  >>> tf.io.decode_raw(tf.constant([["1"],["2"]]), tf.uint8).shape\n  TensorShape([2, 1, 1])\n\n  This is because each byte in the input is converted to a new value on the\n  output (if output type is `uint8` or `int8`, otherwise chunks of inputs get\n  coverted to a new value):\n\n  >>> tf.io.decode_raw(tf.constant("123"), tf.uint8)\n  <tf.Tensor: shape=(3,), dtype=uint8, numpy=array([49, 50, 51], dtype=uint8)>\n  >>> tf.io.decode_raw(tf.constant("1234"), tf.uint8)\n  <tf.Tensor: shape=(4,), dtype=uint8, numpy=array([49, 50, 51, 52], ...\n  >>> # chuncked output\n  >>> tf.io.decode_raw(tf.constant("12"), tf.uint16)\n  <tf.Tensor: shape=(1,), dtype=uint16, numpy=array([12849], dtype=uint16)>\n  >>> tf.io.decode_raw(tf.constant("1234"), tf.uint16)\n  <tf.Tensor: shape=(2,), dtype=uint16, numpy=array([12849, 13363], ...\n  >>> # int64 output\n  >>> tf.io.decode_raw(tf.constant("12345678"), tf.int64)\n  <tf.Tensor: ... numpy=array([4050765991979987505])>\n  >>> tf.io.decode_raw(tf.constant("1234567887654321"), tf.int64)\n  <tf.Tensor: ... numpy=array([4050765991979987505, 3544952156018063160])>\n\n  The operation allows specifying endianness via the `little_endian` parameter.\n\n  >>> tf.io.decode_raw(tf.constant("\\x0a\\x0b"), tf.int16)\n  <tf.Tensor: shape=(1,), dtype=int16, numpy=array([2826], dtype=int16)>\n  >>> hex(2826)\n  \'0xb0a\'\n  >>> tf.io.decode_raw(tf.constant("\\x0a\\x0b"), tf.int16, little_endian=False)\n  <tf.Tensor: shape=(1,), dtype=int16, numpy=array([2571], dtype=int16)>\n  >>> hex(2571)\n  \'0xa0b\'\n\n  If the elements of `input_bytes` are of different length, you must specify\n  `fixed_length`:\n\n  >>> tf.io.decode_raw(tf.constant([["1"],["23"]]), tf.uint8, fixed_length=4)\n  <tf.Tensor: shape=(2, 1, 4), dtype=uint8, numpy=\n  array([[[49,  0,  0,  0]],\n         [[50, 51,  0,  0]]], dtype=uint8)>\n\n  If the `fixed_length` value is larger that the length of the `out_type` dtype,\n  multiple values are generated:\n\n  >>> tf.io.decode_raw(tf.constant(["1212"]), tf.uint16, fixed_length=4)\n  <tf.Tensor: shape=(1, 2), dtype=uint16, numpy=array([[12849, 12849]], ...\n\n  If the input value is larger than `fixed_length`, it is truncated:\n\n  >>> x=\'\'.join([chr(1), chr(2), chr(3), chr(4)])\n  >>> tf.io.decode_raw(x, tf.uint16, fixed_length=2)\n  <tf.Tensor: shape=(1,), dtype=uint16, numpy=array([513], dtype=uint16)>\n  >>> hex(513)\n  \'0x201\'\n\n  If `little_endian` and `fixed_length` are specified, truncation to the fixed\n  length occurs before endianness conversion:\n\n  >>> x=\'\'.join([chr(1), chr(2), chr(3), chr(4)])\n  >>> tf.io.decode_raw(x, tf.uint16, fixed_length=2, little_endian=False)\n  <tf.Tensor: shape=(1,), dtype=uint16, numpy=array([258], dtype=uint16)>\n  >>> hex(258)\n  \'0x102\'\n\n  If input values all have the same length, then specifying `fixed_length`\n  equal to the size of the strings should not change output:\n\n  >>> x = ["12345678", "87654321"]\n  >>> tf.io.decode_raw(x, tf.int16)\n  <tf.Tensor: shape=(2, 4), dtype=int16, numpy=\n  array([[12849, 13363, 13877, 14391],\n         [14136, 13622, 13108, 12594]], dtype=int16)>\n  >>> tf.io.decode_raw(x, tf.int16, fixed_length=len(x[0]))\n  <tf.Tensor: shape=(2, 4), dtype=int16, numpy=\n  array([[12849, 13363, 13877, 14391],\n         [14136, 13622, 13108, 12594]], dtype=int16)>\n\n  Args:\n    input_bytes:\n      Each element of the input Tensor is converted to an array of bytes.\n\n      Currently, this must be a tensor of strings (bytes), although semantically\n      the operation should support any input.\n    out_type:\n      `DType` of the output. Acceptable types are `half`, `float`, `double`,\n      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.\n    little_endian:\n      Whether the `input_bytes` data is in little-endian format. Data will be\n      converted into host byte order if necessary.\n    fixed_length:\n      If set, the first `fixed_length` bytes of each element will be converted.\n      Data will be zero-padded or truncated to the specified length.\n\n      `fixed_length` must be a multiple of the size of `out_type`.\n\n      `fixed_length` must be specified if the elements of `input_bytes` are of\n      variable length.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` object storing the decoded bytes.\n  '
    if fixed_length is not None:
        return gen_parsing_ops.decode_padded_raw(input_bytes, fixed_length=fixed_length, out_type=out_type, little_endian=little_endian, name=name)
    else:
        return gen_parsing_ops.decode_raw(input_bytes, out_type, little_endian=little_endian, name=name)

@tf_export(v1=['decode_raw', 'io.decode_raw'])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, 'bytes is deprecated, use input_bytes instead', 'bytes')
def decode_raw_v1(input_bytes=None, out_type=None, little_endian=True, name=None, bytes=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert raw byte strings into tensors.\n\n  Args:\n    input_bytes:\n      Each element of the input Tensor is converted to an array of bytes.\n    out_type:\n      `DType` of the output. Acceptable types are `half`, `float`, `double`,\n      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.\n    little_endian:\n      Whether the `input_bytes` data is in little-endian format. Data will be\n      converted into host byte order if necessary.\n    name: A name for the operation (optional).\n    bytes: Deprecated parameter. Use `input_bytes` instead.\n\n  Returns:\n    A `Tensor` object storing the decoded bytes.\n  '
    input_bytes = deprecation.deprecated_argument_lookup('input_bytes', input_bytes, 'bytes', bytes)
    if out_type is None:
        raise ValueError("decode_raw_v1() missing 1 positional argument: 'out_type'")
    return gen_parsing_ops.decode_raw(input_bytes, out_type, little_endian=little_endian, name=name)

@tf_export(v1=['io.decode_csv', 'decode_csv'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('decode_csv')
def decode_csv(records, record_defaults, field_delim=',', use_quote_delim=True, name=None, na_value='', select_cols=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert CSV records to tensors. Each column maps to one tensor.\n\n  RFC 4180 format is expected for the CSV records.\n  (https://tools.ietf.org/html/rfc4180)\n  Note that we allow leading and trailing spaces with int or float field.\n\n  Args:\n    records: A `Tensor` of type `string`.\n      Each string is a record/row in the csv and all records should have\n      the same format.\n    record_defaults: A list of `Tensor` objects with specific types.\n      Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.\n      One tensor per column of the input record, with either a\n      scalar default value for that column or an empty vector if the column is\n      required.\n    field_delim: An optional `string`. Defaults to `","`.\n      char delimiter to separate fields in a record.\n    use_quote_delim: An optional `bool`. Defaults to `True`.\n      If false, treats double quotation marks as regular\n      characters inside of the string fields (ignoring RFC 4180, Section 2,\n      Bullet 5).\n    name: A name for the operation (optional).\n    na_value: Additional string to recognize as NA/NaN.\n    select_cols: Optional sorted list of column indices to select. If specified,\n      only this subset of columns will be parsed and returned.\n\n  Returns:\n    A list of `Tensor` objects. Has the same type as `record_defaults`.\n    Each tensor will have the same shape as records.\n\n  Raises:\n    ValueError: If any of the arguments is malformed.\n  '
    return decode_csv_v2(records, record_defaults, field_delim, use_quote_delim, na_value, select_cols, name)

@tf_export('io.decode_csv', v1=[])
@dispatch.add_dispatch_support
def decode_csv_v2(records, record_defaults, field_delim=',', use_quote_delim=True, na_value='', select_cols=None, name=None):
    if False:
        i = 10
        return i + 15
    'Convert CSV records to tensors. Each column maps to one tensor.\n\n  RFC 4180 format is expected for the CSV records.\n  (https://tools.ietf.org/html/rfc4180)\n  Note that we allow leading and trailing spaces with int or float field.\n\n  Args:\n    records: A `Tensor` of type `string`.\n      Each string is a record/row in the csv and all records should have\n      the same format.\n    record_defaults: A list of `Tensor` objects with specific types.\n      Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.\n      One tensor per column of the input record, with either a\n      scalar default value for that column or an empty vector if the column is\n      required.\n    field_delim: An optional `string`. Defaults to `","`.\n      char delimiter to separate fields in a record.\n    use_quote_delim: An optional `bool`. Defaults to `True`.\n      If false, treats double quotation marks as regular\n      characters inside of the string fields (ignoring RFC 4180, Section 2,\n      Bullet 5).\n    na_value: Additional string to recognize as NA/NaN.\n    select_cols: Optional sorted list of column indices to select. If specified,\n      only this subset of columns will be parsed and returned.\n    name: A name for the operation (optional).\n\n  Returns:\n    A list of `Tensor` objects. Has the same type as `record_defaults`.\n    Each tensor will have the same shape as records.\n\n  Raises:\n    ValueError: If any of the arguments is malformed.\n  '
    if select_cols is not None and any((select_cols[i] >= select_cols[i + 1] for i in range(len(select_cols) - 1))):
        raise ValueError('select_cols is not strictly increasing.')
    if select_cols is not None and select_cols[0] < 0:
        raise ValueError('select_cols contains negative values.')
    if select_cols is not None and len(select_cols) != len(record_defaults):
        raise ValueError('Length of select_cols and record_defaults do not match.')
    return gen_parsing_ops.decode_csv(records=records, record_defaults=record_defaults, field_delim=field_delim, use_quote_delim=use_quote_delim, na_value=na_value, name=name, select_cols=select_cols)

def _assert_scalar(value, name):
    if False:
        for i in range(10):
            print('nop')
    'Asserts that `value` is scalar, and returns `value`.'
    value_rank = value.shape.rank
    if value_rank is None:
        check = control_flow_assert.Assert(math_ops.equal(array_ops.rank(value), 0), ['Input %s must be a scalar' % name], name='%sIsScalar' % name.capitalize())
        result = control_flow_ops.with_dependencies([check], value, name='%sDependencies' % name)
        result.set_shape([])
        return result
    elif value_rank == 0:
        return value
    else:
        raise ValueError('Input %s must be a scalar' % name)

@tf_export('io.decode_json_example', v1=['decode_json_example', 'io.decode_json_example'])
def decode_json_example(json_examples, name=None):
    if False:
        while True:
            i = 10
    'Convert JSON-encoded Example records to binary protocol buffer strings.\n\n  Note: This is **not** a general purpose JSON parsing op.\n\n  This op converts JSON-serialized `tf.train.Example` (maybe created with\n  `json_format.MessageToJson`, following the\n  [standard JSON mapping](\n  https://developers.google.com/protocol-buffers/docs/proto3#json))\n  to a binary-serialized `tf.train.Example` (equivalent to\n  `Example.SerializeToString()`) suitable for conversion to tensors with\n  `tf.io.parse_example`.\n\n  Here is a `tf.train.Example` proto:\n\n  >>> example = tf.train.Example(\n  ...   features=tf.train.Features(\n  ...       feature={\n  ...           "a": tf.train.Feature(\n  ...               int64_list=tf.train.Int64List(\n  ...                   value=[1, 1, 3]))}))\n\n  Here it is converted to JSON:\n\n  >>> from google.protobuf import json_format\n  >>> example_json = json_format.MessageToJson(example)\n  >>> print(example_json)\n  {\n    "features": {\n      "feature": {\n        "a": {\n          "int64List": {\n            "value": [\n              "1",\n              "1",\n              "3"\n            ]\n          }\n        }\n      }\n    }\n  }\n\n  This op converts the above json string to a binary proto:\n\n  >>> example_binary = tf.io.decode_json_example(example_json)\n  >>> example_binary.numpy()\n  b\'\\n\\x0f\\n\\r\\n\\x01a\\x12\\x08\\x1a\\x06\\x08\\x01\\x08\\x01\\x08\\x03\'\n\n  The OP works on string tensors of andy shape:\n\n  >>> tf.io.decode_json_example([\n  ...     [example_json, example_json],\n  ...     [example_json, example_json]]).shape.as_list()\n  [2, 2]\n\n  This resulting binary-string is equivalent to `Example.SerializeToString()`,\n  and can be converted to Tensors using `tf.io.parse_example` and related\n  functions:\n\n  >>> tf.io.parse_example(\n  ...   serialized=[example_binary.numpy(),\n  ...              example.SerializeToString()],\n  ...   features = {\'a\': tf.io.FixedLenFeature(shape=[3], dtype=tf.int64)})\n  {\'a\': <tf.Tensor: shape=(2, 3), dtype=int64, numpy=\n   array([[1, 1, 3],\n          [1, 1, 3]])>}\n\n  Args:\n    json_examples: A string tensor containing json-serialized `tf.Example`\n      protos.\n    name: A name for the op.\n\n  Returns:\n    A string Tensor containing the binary-serialized `tf.Example` protos.\n\n  Raises:\n     `tf.errors.InvalidArgumentError`: If the JSON could not be converted to a\n     `tf.Example`\n  '
    return gen_parsing_ops.decode_json_example(json_examples, name=name)
dispatch.register_unary_elementwise_api(gen_parsing_ops.decode_compressed)