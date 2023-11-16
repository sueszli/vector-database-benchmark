"""Helper functions for creating partitioned variables.

This is a convenient abstraction to partition a large variable across
multiple smaller variables that can be assigned to different devices.

The full variable can be reconstructed by concatenating the smaller variables.
Using partitioned variables instead of a single variable is mostly a
performance choice.  It however also has an impact on:

1. Random initialization, as the random number generator is called once per
   slice
2. Updates, as they happen in parallel across slices

A key design goal is to allow a different graph to repartition a variable
with the same name but different slicings, including possibly no partitions.

TODO(touts): If an initializer provides a seed, the seed must be changed
deterministically for each slice, maybe by adding one to it, otherwise each
slice will use the same values.  Maybe this can be done by passing the
slice offsets to the initializer functions.

Typical usage:

```python
# Create a list of partitioned variables with:
vs = create_partitioned_variables(
    <shape>, <slicing>, <initializer>, name=<optional-name>)

# Pass the list as inputs to embedding_lookup for sharded, parallel lookup:
y = embedding_lookup(vs, ids, partition_strategy="div")

# Or fetch the variables in parallel to speed up large matmuls:
z = matmul(x, concat(slice_dim, vs))
```
"""
import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
__all__ = ['create_partitioned_variables', 'variable_axis_size_partitioner', 'min_max_variable_partitioner', 'fixed_size_partitioner']

@tf_export(v1=['variable_axis_size_partitioner'])
def variable_axis_size_partitioner(max_shard_bytes, axis=0, bytes_per_string_element=16, max_shards=None):
    if False:
        while True:
            i = 10
    'Get a partitioner for VariableScope to keep shards below `max_shard_bytes`.\n\n  This partitioner will shard a Variable along one axis, attempting to keep\n  the maximum shard size below `max_shard_bytes`.  In practice, this is not\n  always possible when sharding along only one axis.  When this happens,\n  this axis is sharded as much as possible (i.e., every dimension becomes\n  a separate shard).\n\n  If the partitioner hits the `max_shards` limit, then each shard may end up\n  larger than `max_shard_bytes`. By default `max_shards` equals `None` and no\n  limit on the number of shards is enforced.\n\n  One reasonable value for `max_shard_bytes` is `(64 << 20) - 1`, or almost\n  `64MB`, to keep below the protobuf byte limit.\n\n  Args:\n    max_shard_bytes: The maximum size any given shard is allowed to be.\n    axis: The axis to partition along.  Default: outermost axis.\n    bytes_per_string_element: If the `Variable` is of type string, this provides\n      an estimate of how large each scalar in the `Variable` is.\n    max_shards: The maximum number of shards in int created taking precedence\n      over `max_shard_bytes`.\n\n  Returns:\n    A partition function usable as the `partitioner` argument to\n    `variable_scope` and `get_variable`.\n\n  Raises:\n    ValueError: If any of the byte counts are non-positive.\n  '
    if max_shard_bytes < 1 or bytes_per_string_element < 1:
        raise ValueError(f'Both max_shard_bytes and bytes_per_string_element must be positive. Currently, max_shard_bytes is {max_shard_bytes} andbytes_per_string_element is {bytes_per_string_element}')
    if max_shards and max_shards < 1:
        raise ValueError('max_shards must be positive.')

    def _partitioner(shape, dtype):
        if False:
            i = 10
            return i + 15
        'Partitioner that partitions shards to have max_shard_bytes total size.\n\n    Args:\n      shape: A `TensorShape`.\n      dtype: A `DType`.\n\n    Returns:\n      A tuple representing how much to slice each axis in shape.\n\n    Raises:\n      ValueError: If shape is not a fully defined `TensorShape` or dtype is not\n        a `DType`.\n    '
        if not isinstance(shape, tensor_shape.TensorShape):
            raise ValueError(f'shape is not a TensorShape: {shape}')
        if not shape.is_fully_defined():
            raise ValueError(f'shape is not fully defined: {shape}')
        if not isinstance(dtype, dtypes.DType):
            raise ValueError(f'dtype is not a DType: {dtype}')
        if dtype.base_dtype == dtypes.string:
            element_size = bytes_per_string_element
        else:
            element_size = dtype.size
        partitions = [1] * shape.ndims
        bytes_per_slice = 1.0 * (shape.num_elements() / shape.dims[axis].value) * element_size
        slices_per_shard = max(1, math.floor(max_shard_bytes / bytes_per_slice))
        axis_shards = int(math.ceil(1.0 * shape.dims[axis].value / slices_per_shard))
        if max_shards:
            axis_shards = min(max_shards, axis_shards)
        partitions[axis] = axis_shards
        return partitions
    return _partitioner

@tf_export(v1=['min_max_variable_partitioner'])
def min_max_variable_partitioner(max_partitions=1, axis=0, min_slice_size=256 << 10, bytes_per_string_element=16):
    if False:
        i = 10
        return i + 15
    'Partitioner to allocate minimum size per slice.\n\n  Returns a partitioner that partitions the variable of given shape and dtype\n  such that each partition has a minimum of `min_slice_size` slice of the\n  variable. The maximum number of such partitions (upper bound) is given by\n  `max_partitions`.\n\n  Args:\n    max_partitions: Upper bound on the number of partitions. Defaults to 1.\n    axis: Axis along which to partition the variable. Defaults to 0.\n    min_slice_size: Minimum size of the variable slice per partition. Defaults\n      to 256K.\n    bytes_per_string_element: If the `Variable` is of type string, this provides\n      an estimate of how large each scalar in the `Variable` is.\n\n  Returns:\n    A partition function usable as the `partitioner` argument to\n    `variable_scope` and `get_variable`.\n\n  '

    def _partitioner(shape, dtype):
        if False:
            print('Hello World!')
        'Partitioner that partitions list for a variable of given shape and type.\n\n    Ex: Consider partitioning a variable of type float32 with\n      shape=[1024, 1024].\n      If `max_partitions` >= 16, this function would return\n        [(1024 * 1024 * 4) / (256 * 1024), 1] = [16, 1].\n      If `max_partitions` < 16, this function would return\n        [`max_partitions`, 1].\n\n    Args:\n      shape: Shape of the variable.\n      dtype: Type of the variable.\n\n    Returns:\n      List of partitions for each axis (currently only one axis can be\n      partitioned).\n\n    Raises:\n      ValueError: If axis to partition along does not exist for the variable.\n    '
        if axis >= len(shape):
            raise ValueError(f'Cannot partition variable along axis {axis} when shape is only {shape}')
        if dtype.base_dtype == dtypes.string:
            bytes_per_element = bytes_per_string_element
        else:
            bytes_per_element = dtype.size
        total_size_bytes = shape.num_elements() * bytes_per_element
        partitions = total_size_bytes / min_slice_size
        partitions_list = [1] * len(shape)
        partitions_list[axis] = max(1, min(shape.dims[axis].value, max_partitions, int(math.ceil(partitions))))
        return partitions_list
    return _partitioner

@tf_export(v1=['fixed_size_partitioner'])
def fixed_size_partitioner(num_shards, axis=0):
    if False:
        return 10
    'Partitioner to specify a fixed number of shards along given axis.\n\n  @compatibility(TF2)\n  This API is deprecated in TF2. In TF2, partitioner is no longer part of\n  the variable declaration via `tf.Variable`.\n  [ParameterServer Training]\n  (https://www.tensorflow.org/tutorials/distribute/parameter_server_training)\n  handles partitioning of variables. The corresponding TF2 partitioner class of\n  `fixed_size_partitioner` is\n  `tf.distribute.experimental.partitioners.FixedShardsPartitioner`.\n\n  Check the [migration guide]\n  (https://www.tensorflow.org/guide/migrate#2_use_python_objects_to_track_variables_and_losses)\n  on the differences in treatment of variables and losses between TF1 and TF2.\n\n  Before:\n\n    ```\n    x = tf.compat.v1.get_variable(\n      "x", shape=(2,), partitioner=tf.compat.v1.fixed_size_partitioner(2)\n    )\n    ```\n  After:\n\n    ```\n    partitioner = (\n        tf.distribute.experimental.partitioners.FixedShardsPartitioner(\n            num_shards=2)\n    )\n    strategy = tf.distribute.experimental.ParameterServerStrategy(\n                   cluster_resolver=cluster_resolver,\n                   variable_partitioner=partitioner)\n\n    with strategy.scope():\n      x = tf.Variable([1.0, 2.0])\n    ```\n  @end_compatibility\n\n  Args:\n    num_shards: `int`, number of shards to partition variable.\n    axis: `int`, axis to partition on.\n\n  Returns:\n    A partition function usable as the `partitioner` argument to\n    `variable_scope` and `get_variable`.\n  '

    def _partitioner(shape, **unused_args):
        if False:
            i = 10
            return i + 15
        partitions_list = [1] * len(shape)
        partitions_list[axis] = min(num_shards, shape.dims[axis].value)
        return partitions_list
    return _partitioner

@tf_export(v1=['create_partitioned_variables'])
@deprecation.deprecated(date=None, instructions='Use `tf.get_variable` with a partitioner set.')
def create_partitioned_variables(shape, slicing, initializer, dtype=dtypes.float32, trainable=True, collections=None, name=None, reuse=None):
    if False:
        for i in range(10):
            print('nop')
    'Create a list of partitioned variables according to the given `slicing`.\n\n  Currently only one dimension of the full variable can be sliced, and the\n  full variable can be reconstructed by the concatenation of the returned\n  list along that dimension.\n\n  Args:\n    shape: List of integers.  The shape of the full variable.\n    slicing: List of integers.  How to partition the variable.\n      Must be of the same length as `shape`.  Each value\n      indicate how many slices to create in the corresponding\n      dimension.  Presently only one of the values can be more than 1;\n      that is, the variable can only be sliced along one dimension.\n\n      For convenience, The requested number of partitions does not have to\n      divide the corresponding dimension evenly.  If it does not, the\n      shapes of the partitions are incremented by 1 starting from partition\n      0 until all slack is absorbed.  The adjustment rules may change in the\n      future, but as you can save/restore these variables with different\n      slicing specifications this should not be a problem.\n    initializer: A `Tensor` of shape `shape` or a variable initializer\n      function.  If a function, it will be called once for each slice,\n      passing the shape and data type of the slice as parameters.  The\n      function must return a tensor with the same shape as the slice.\n    dtype: Type of the variables. Ignored if `initializer` is a `Tensor`.\n    trainable: If True also add all the variables to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES`.\n    collections: List of graph collections keys to add the variables to.\n      Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.\n    name: Optional name for the full variable.  Defaults to\n      `"PartitionedVariable"` and gets uniquified automatically.\n    reuse: Boolean or `None`; if `True` and name is set, it would reuse\n      previously created variables. if `False` it will create new variables.\n      if `None`, it would inherit the parent scope reuse.\n\n  Returns:\n    A list of Variables corresponding to the slicing.\n\n  Raises:\n    ValueError: If any of the arguments is malformed.\n  '
    if len(shape) != len(slicing):
        raise ValueError(f"The 'shape' and 'slicing' of a partitioned Variable must have the length: shape: {shape}, slicing: {slicing}")
    if len(shape) < 1:
        raise ValueError(f'A partitioned Variable must have rank at least 1: shape: {shape}')
    partitioner = lambda **unused_kwargs: slicing
    with variable_scope.variable_scope(name, 'PartitionedVariable', reuse=reuse):
        partitioned_var = variable_scope._get_partitioned_variable(name=None, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, partitioner=partitioner, collections=collections)
        return list(partitioned_var)