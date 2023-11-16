"""Classes for storing ragged tensors and their values."""
import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
_convert_row_partition = RowPartition._convert_row_partition

@tf_export('RaggedTensor')
class RaggedTensor(composite_tensor.CompositeTensor, internal_types.NativeObject, internal_types.RaggedTensor):
    """Represents a ragged tensor.

  A `RaggedTensor` is a tensor with one or more *ragged dimensions*, which are
  dimensions whose slices may have different lengths.  For example, the inner
  (column) dimension of `rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged,
  since the column slices (`rt[0, :]`, ..., `rt[4, :]`) have different lengths.
  Dimensions whose slices all have the same length are called *uniform
  dimensions*.  The outermost dimension of a `RaggedTensor` is always uniform,
  since it consists of a single slice (and so there is no possibility for
  differing slice lengths).

  The total number of dimensions in a `RaggedTensor` is called its *rank*,
  and the number of ragged dimensions in a `RaggedTensor` is called its
  *ragged-rank*.  A `RaggedTensor`'s ragged-rank is fixed at graph creation
  time: it can't depend on the runtime values of `Tensor`s, and can't vary
  dynamically for different session runs.

  Note that the `__init__` constructor is private. Please use one of the
  following methods to construct a `RaggedTensor`:

  * `tf.RaggedTensor.from_row_lengths`
  * `tf.RaggedTensor.from_value_rowids`
  * `tf.RaggedTensor.from_row_splits`
  * `tf.RaggedTensor.from_row_starts`
  * `tf.RaggedTensor.from_row_limits`
  * `tf.RaggedTensor.from_nested_row_splits`
  * `tf.RaggedTensor.from_nested_row_lengths`
  * `tf.RaggedTensor.from_nested_value_rowids`

  ### Potentially Ragged Tensors

  Many ops support both `Tensor`s and `RaggedTensor`s
  (see [tf.ragged](https://www.tensorflow.org/api_docs/python/tf/ragged) for a
  full listing). The term "potentially ragged tensor" may be used to refer to a
  tensor that might be either a `Tensor` or a `RaggedTensor`.  The ragged-rank
  of a `Tensor` is zero.

  ### Documenting RaggedTensor Shapes

  When documenting the shape of a RaggedTensor, ragged dimensions can be
  indicated by enclosing them in parentheses.  For example, the shape of
  a 3-D `RaggedTensor` that stores the fixed-size word embedding for each
  word in a sentence, for each sentence in a batch, could be written as
  `[num_sentences, (num_words), embedding_size]`.  The parentheses around
  `(num_words)` indicate that dimension is ragged, and that the length
  of each element list in that dimension may vary for each item.

  ### Component Tensors

  Internally, a `RaggedTensor` consists of a concatenated list of values that
  are partitioned into variable-length rows.  In particular, each `RaggedTensor`
  consists of:

    * A `values` tensor, which concatenates the variable-length rows into a
      flattened list.  For example, the `values` tensor for
      `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is `[3, 1, 4, 1, 5, 9, 2, 6]`.

    * A `row_splits` vector, which indicates how those flattened values are
      divided into rows.  In particular, the values for row `rt[i]` are stored
      in the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

  Example:

  >>> print(tf.RaggedTensor.from_row_splits(
  ...       values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...       row_splits=[0, 4, 4, 7, 8, 8]))
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

  ### Alternative Row-Partitioning Schemes

  In addition to `row_splits`, ragged tensors provide support for five other
  row-partitioning schemes:

    * `row_lengths`: a vector with shape `[nrows]`, which specifies the length
      of each row.

    * `value_rowids` and `nrows`: `value_rowids` is a vector with shape
      `[nvals]`, corresponding one-to-one with `values`, which specifies
      each value's row index.  In particular, the row `rt[row]` consists of the
      values `rt.values[j]` where `value_rowids[j]==row`.  `nrows` is an
      integer scalar that specifies the number of rows in the
      `RaggedTensor`. (`nrows` is used to indicate trailing empty rows.)

    * `row_starts`: a vector with shape `[nrows]`, which specifies the start
      offset of each row.  Equivalent to `row_splits[:-1]`.

    * `row_limits`: a vector with shape `[nrows]`, which specifies the stop
      offset of each row.  Equivalent to `row_splits[1:]`.

    * `uniform_row_length`: A scalar tensor, specifying the length of every
      row.  This row-partitioning scheme may only be used if all rows have
      the same length.

  Example: The following ragged tensors are equivalent, and all represent the
  nested list `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]`.

  >>> values = [3, 1, 4, 1, 5, 9, 2, 6]
  >>> RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_value_rowids(
  ...     values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_uniform_row_length(values, uniform_row_length=2)
  <tf.RaggedTensor [[3, 1], [4, 1], [5, 9], [2, 6]]>

  ### Multiple Ragged Dimensions

  `RaggedTensor`s with multiple ragged dimensions can be defined by using
  a nested `RaggedTensor` for the `values` tensor.  Each nested `RaggedTensor`
  adds a single ragged dimension.

  >>> inner_rt = RaggedTensor.from_row_splits(  # =rt1 from above
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
  >>> outer_rt = RaggedTensor.from_row_splits(
  ...     values=inner_rt, row_splits=[0, 3, 3, 5])
  >>> print(outer_rt.to_list())
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  >>> print(outer_rt.ragged_rank)
  2

  The factory function `RaggedTensor.from_nested_row_splits` may be used to
  construct a `RaggedTensor` with multiple ragged dimensions directly, by
  providing a list of `row_splits` tensors:

  >>> RaggedTensor.from_nested_row_splits(
  ...     flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])).to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]

  ### Uniform Inner Dimensions

  `RaggedTensor`s with uniform inner dimensions can be defined
  by using a multidimensional `Tensor` for `values`.

  >>> rt = RaggedTensor.from_row_splits(values=tf.ones([5, 3], tf.int32),
  ...                                   row_splits=[0, 2, 5])
  >>> print(rt.to_list())
  [[[1, 1, 1], [1, 1, 1]],
   [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
  >>> print(rt.shape)
  (2, None, 3)

  ### Uniform Outer Dimensions

  `RaggedTensor`s with uniform outer dimensions can be defined by using
  one or more `RaggedTensor` with a `uniform_row_length` row-partitioning
  tensor.  For example, a `RaggedTensor` with shape `[2, 2, None]` can be
  constructed with this method from a `RaggedTensor` values with shape
  `[4, None]`:

  >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
  >>> print(values.shape)
  (4, None)
  >>> rt6 = tf.RaggedTensor.from_uniform_row_length(values, 2)
  >>> print(rt6)
  <tf.RaggedTensor [[[1, 2, 3], [4]], [[5, 6], [7, 8, 9, 10]]]>
  >>> print(rt6.shape)
  (2, 2, None)

  Note that `rt6` only contains one ragged dimension (the innermost
  dimension). In contrast, if `from_row_splits` is used to construct a similar
  `RaggedTensor`, then that `RaggedTensor` will have two ragged dimensions:

  >>> rt7 = tf.RaggedTensor.from_row_splits(values, [0, 2, 4])
  >>> print(rt7.shape)
  (2, None, None)

  Uniform and ragged outer dimensions may be interleaved, meaning that a
  tensor with any combination of ragged and uniform dimensions may be created.
  For example, a RaggedTensor `t4` with shape `[3, None, 4, 8, None, 2]` could
  be constructed as follows:

  ```python
  t0 = tf.zeros([1000, 2])                           # Shape:         [1000, 2]
  t1 = RaggedTensor.from_row_lengths(t0, [...])      #           [160, None, 2]
  t2 = RaggedTensor.from_uniform_row_length(t1, 8)   #         [20, 8, None, 2]
  t3 = RaggedTensor.from_uniform_row_length(t2, 4)   #       [5, 4, 8, None, 2]
  t4 = RaggedTensor.from_row_lengths(t3, [...])      # [3, None, 4, 8, None, 2]
  ```

  """

    @doc_controls.do_not_generate_docs
    def __init__(self, values, row_partition, internal=False):
        if False:
            i = 10
            return i + 15
        'Creates a `RaggedTensor` with a specified partitioning for `values`.\n\n    This constructor is private -- please use one of the following ops to\n    build `RaggedTensor`s:\n\n      * `tf.RaggedTensor.from_row_lengths`\n      * `tf.RaggedTensor.from_value_rowids`\n      * `tf.RaggedTensor.from_row_splits`\n      * `tf.RaggedTensor.from_row_starts`\n      * `tf.RaggedTensor.from_row_limits`\n      * `tf.RaggedTensor.from_nested_row_splits`\n      * `tf.RaggedTensor.from_nested_row_lengths`\n      * `tf.RaggedTensor.from_nested_value_rowids`\n\n    Args:\n      values: A potentially ragged tensor of any dtype and shape `[nvals, ...]`.\n      row_partition: A `RowPartition` object, representing the arrangement of\n        the lists at the top level.\n      internal: True if the constructor is being called by one of the factory\n        methods.  If false, an exception will be raised.\n\n    Raises:\n      ValueError: If internal = False. Note that this method is intended only\n                 for internal use.\n      TypeError: If values is not a `RaggedTensor` or `Tensor`, or\n                 row_partition is not a `RowPartition`.\n    '
        if not internal:
            raise ValueError('RaggedTensor constructor is private; please use one of the factory methods instead (e.g., RaggedTensor.from_row_lengths())')
        _assert_is_supported_ragged_values_type(values)
        if not isinstance(row_partition, RowPartition):
            raise TypeError(f'Argument `row_partition` must be a RowPartition. Received {row_partition}.')
        values.shape.with_rank_at_least(1)
        if isinstance(values, RaggedTensor):
            assert row_partition.dtype == values._row_partition.dtype
        self._values = values
        self._row_partition = row_partition

    @classmethod
    def _from_row_partition(cls, values, row_partition, validate=True):
        if False:
            i = 10
            return i + 15
        'Creates a `RaggedTensor` with a row partition.\n\n    This is used as a way for RaggedTensors to share row partitions.\n\n    The outer dimension of values must be equal to `partition.nvals()`.\n\n    Args:\n      values: A potentially ragged tensor.\n      row_partition: a `RowPartition`: can be shared between tensors.\n      validate: If true, then use assertions to check that the arguments form a\n        valid `RaggedTensor`.\n\n    Returns:\n      A `RaggedTensor`.  `result.rank = values.rank + 1`.\n      `result.ragged_rank = values.ragged_rank + 1`.\n\n    Raises:\n      ValueError: If partition.nvals() != _nrows(values)\n    '
        if not isinstance(row_partition, RowPartition):
            raise TypeError(f'Argument `row_partition` must be a RowPartition. Received {row_partition}.')
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        (values, row_partition) = cls._convert_values_and_partition(values, row_partition, 'partition')
        if row_partition._has_precomputed_value_rowids():
            value_rowids_shape = row_partition.value_rowids().shape
            values.shape[:1].assert_is_compatible_with(value_rowids_shape)
        if validate:
            msg = 'Arguments to _from_row_partition do not form a valid RaggedTensor'
            nvals = _nrows(values, row_partition.dtype)
            checks = [check_ops.assert_equal(math_ops.cast(row_partition.nvals(), row_partition.dtype), nvals, message=msg)]
            if not isinstance(values, RaggedTensor):
                checks.append(check_ops.assert_rank_at_least(values, 1))
            row_partition = row_partition._with_dependencies(checks)
        return cls(values=values, internal=True, row_partition=row_partition)

    @classmethod
    @dispatch.add_dispatch_support
    def from_value_rowids(cls, values, value_rowids, nrows=None, name=None, validate=True):
        if False:
            return 10
        "Creates a `RaggedTensor` with rows partitioned by `value_rowids`.\n\n    The returned `RaggedTensor` corresponds with the python list defined by:\n\n    ```python\n    result = [[values[i] for i in range(len(values)) if value_rowids[i] == row]\n              for row in range(nrows)]\n    ```\n\n    Args:\n      values: A potentially ragged tensor with shape `[nvals, ...]`.\n      value_rowids: A 1-D integer tensor with shape `[nvals]`, which corresponds\n        one-to-one with `values`, and specifies each value's row index.  Must be\n        nonnegative, and must be sorted in ascending order.\n      nrows: An integer scalar specifying the number of rows.  This should be\n        specified if the `RaggedTensor` may containing empty training rows. Must\n        be greater than `value_rowids[-1]` (or zero if `value_rowids` is empty).\n        Defaults to `value_rowids[-1] + 1` (or zero if `value_rowids` is empty).\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor`.  `result.rank = values.rank + 1`.\n      `result.ragged_rank = values.ragged_rank + 1`.\n\n    Raises:\n      ValueError: If `nrows` is incompatible with `value_rowids`.\n\n    #### Example:\n\n    >>> print(tf.RaggedTensor.from_value_rowids(\n    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],\n    ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],\n    ...     nrows=5))\n    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>\n\n    "
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        with ops.name_scope(name, 'RaggedFromValueRowIds', [values, value_rowids, nrows]):
            row_partition = RowPartition.from_value_rowids(value_rowids=value_rowids, nrows=nrows, validate=validate, dtype_hint=_get_optional_partition_dtype(values))
            return cls._from_row_partition(values, row_partition, validate=validate)

    @classmethod
    @dispatch.add_dispatch_support
    def from_row_splits(cls, values, row_splits, name=None, validate=True):
        if False:
            while True:
                i = 10
        'Creates a `RaggedTensor` with rows partitioned by `row_splits`.\n\n    The returned `RaggedTensor` corresponds with the python list defined by:\n\n    ```python\n    result = [values[row_splits[i]:row_splits[i + 1]]\n              for i in range(len(row_splits) - 1)]\n    ```\n\n    Args:\n      values: A potentially ragged tensor with shape `[nvals, ...]`.\n      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be\n        empty, and must be sorted in ascending order.  `row_splits[0]` must be\n        zero and `row_splits[-1]` must be `nvals`.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor`.  `result.rank = values.rank + 1`.\n      `result.ragged_rank = values.ragged_rank + 1`.\n\n    Raises:\n      ValueError: If `row_splits` is an empty list.\n\n    #### Example:\n\n    >>> print(tf.RaggedTensor.from_row_splits(\n    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],\n    ...     row_splits=[0, 4, 4, 7, 8, 8]))\n    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>\n\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        with ops.name_scope(name, 'RaggedFromRowSplits', [values, row_splits]):
            row_partition = RowPartition.from_row_splits(row_splits=row_splits, validate=validate, dtype_hint=_get_optional_partition_dtype(values))
            return cls._from_row_partition(values, row_partition, validate=validate)

    @classmethod
    @dispatch.add_dispatch_support
    def from_row_lengths(cls, values, row_lengths, name=None, validate=True):
        if False:
            return 10
        'Creates a `RaggedTensor` with rows partitioned by `row_lengths`.\n\n    The returned `RaggedTensor` corresponds with the python list defined by:\n\n    ```python\n    result = [[values.pop(0) for i in range(length)]\n              for length in row_lengths]\n    ```\n\n    Args:\n      values: A potentially ragged tensor with shape `[nvals, ...]`.\n      row_lengths: A 1-D integer tensor with shape `[nrows]`.  Must be\n        nonnegative.  `sum(row_lengths)` must be `nvals`.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor`.  `result.rank = values.rank + 1`.\n      `result.ragged_rank = values.ragged_rank + 1`.\n\n    #### Example:\n\n    >>> print(tf.RaggedTensor.from_row_lengths(\n    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],\n    ...     row_lengths=[4, 0, 3, 1, 0]))\n    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>\n\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        with ops.name_scope(name, 'RaggedFromRowLengths', [values, row_lengths]):
            row_partition = RowPartition.from_row_lengths(row_lengths=row_lengths, validate=validate, dtype_hint=_get_optional_partition_dtype(values))
            return cls._from_row_partition(values, row_partition, validate=validate)

    @classmethod
    @dispatch.add_dispatch_support
    def from_row_starts(cls, values, row_starts, name=None, validate=True):
        if False:
            i = 10
            return i + 15
        'Creates a `RaggedTensor` with rows partitioned by `row_starts`.\n\n    Equivalent to: `from_row_splits(values, concat([row_starts, nvals]))`.\n\n    Args:\n      values: A potentially ragged tensor with shape `[nvals, ...]`.\n      row_starts: A 1-D integer tensor with shape `[nrows]`.  Must be\n        nonnegative and sorted in ascending order.  If `nrows>0`, then\n        `row_starts[0]` must be zero.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor`.  `result.rank = values.rank + 1`.\n      `result.ragged_rank = values.ragged_rank + 1`.\n\n    #### Example:\n\n    >>> print(tf.RaggedTensor.from_row_starts(\n    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],\n    ...     row_starts=[0, 4, 4, 7, 8]))\n    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>\n\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        with ops.name_scope(name, 'RaggedFromRowStarts', [values, row_starts]):
            values = _convert_to_ragged_tensor_values(values)
            row_partition = RowPartition.from_row_starts(row_starts=row_starts, nvals=_nrows(values), validate=validate, dtype_hint=_get_optional_partition_dtype(values))
            return cls._from_row_partition(values, row_partition, validate=validate)

    @classmethod
    @dispatch.add_dispatch_support
    def from_row_limits(cls, values, row_limits, name=None, validate=True):
        if False:
            for i in range(10):
                print('nop')
        'Creates a `RaggedTensor` with rows partitioned by `row_limits`.\n\n    Equivalent to: `from_row_splits(values, concat([0, row_limits]))`.\n\n    Args:\n      values: A potentially ragged tensor with shape `[nvals, ...]`.\n      row_limits: A 1-D integer tensor with shape `[nrows]`.  Must be sorted in\n        ascending order.  If `nrows>0`, then `row_limits[-1]` must be `nvals`.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor`.  `result.rank = values.rank + 1`.\n      `result.ragged_rank = values.ragged_rank + 1`.\n\n    #### Example:\n\n    >>> print(tf.RaggedTensor.from_row_limits(\n    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],\n    ...     row_limits=[4, 4, 7, 8, 8]))\n    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>\n\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        with ops.name_scope(name, 'RaggedFromRowLimits', [values, row_limits]):
            values = _convert_to_ragged_tensor_values(values)
            row_partition = RowPartition.from_row_limits(row_limits=row_limits, validate=validate, dtype_hint=_get_optional_partition_dtype(values))
            return cls._from_row_partition(values, row_partition, validate=validate)

    @classmethod
    @dispatch.add_dispatch_support
    def from_uniform_row_length(cls, values, uniform_row_length, nrows=None, validate=True, name=None):
        if False:
            while True:
                i = 10
        'Creates a `RaggedTensor` with rows partitioned by `uniform_row_length`.\n\n    This method can be used to create `RaggedTensor`s with multiple uniform\n    outer dimensions.  For example, a `RaggedTensor` with shape `[2, 2, None]`\n    can be constructed with this method from a `RaggedTensor` values with shape\n    `[4, None]`:\n\n    >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])\n    >>> print(values.shape)\n    (4, None)\n    >>> rt1 = tf.RaggedTensor.from_uniform_row_length(values, 2)\n    >>> print(rt1)\n    <tf.RaggedTensor [[[1, 2, 3], [4]], [[5, 6], [7, 8, 9, 10]]]>\n    >>> print(rt1.shape)\n    (2, 2, None)\n\n    Note that `rt1` only contains one ragged dimension (the innermost\n    dimension). In contrast, if `from_row_splits` is used to construct a similar\n    `RaggedTensor`, then that `RaggedTensor` will have two ragged dimensions:\n\n    >>> rt2 = tf.RaggedTensor.from_row_splits(values, [0, 2, 4])\n    >>> print(rt2.shape)\n    (2, None, None)\n\n    Args:\n      values: A potentially ragged tensor with shape `[nvals, ...]`.\n      uniform_row_length: A scalar integer tensor.  Must be nonnegative. The\n        size of the outer axis of `values` must be evenly divisible by\n        `uniform_row_length`.\n      nrows: The number of rows in the constructed RaggedTensor.  If not\n        specified, then it defaults to `nvals/uniform_row_length` (or `0` if\n        `uniform_row_length==0`).  `nrows` only needs to be specified if\n        `uniform_row_length` might be zero.  `uniform_row_length*nrows` must be\n        `nvals`.\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n      name: A name prefix for the RaggedTensor (optional).\n\n    Returns:\n      A `RaggedTensor` that corresponds with the python list defined by:\n\n      ```python\n      result = [[values.pop(0) for i in range(uniform_row_length)]\n                for _ in range(nrows)]\n      ```\n\n      `result.rank = values.rank + 1`.\n      `result.ragged_rank = values.ragged_rank + 1`.\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        with ops.name_scope(name, 'RaggedFromUniformRowLength', [values, uniform_row_length, nrows]):
            values = _convert_to_ragged_tensor_values(values)
            uniform_row_length = _convert_row_partition(uniform_row_length, 'UniformRowLength', _get_optional_partition_dtype(values))
            nvals = _nvals_uniform_row_length(values, uniform_row_length)
            row_partition = RowPartition.from_uniform_row_length(uniform_row_length=uniform_row_length, nvals=nvals, nrows=nrows, validate=validate, dtype_hint=_get_optional_partition_dtype(values))
            return cls._from_row_partition(values, row_partition, validate=validate)

    @classmethod
    @dispatch.add_dispatch_support
    def from_nested_value_rowids(cls, flat_values, nested_value_rowids, nested_nrows=None, name=None, validate=True):
        if False:
            print('Hello World!')
        'Creates a `RaggedTensor` from a nested list of `value_rowids` tensors.\n\n    Equivalent to:\n\n    ```python\n    result = flat_values\n    for (rowids, nrows) in reversed(zip(nested_value_rowids, nested_nrows)):\n      result = from_value_rowids(result, rowids, nrows)\n    ```\n\n    Args:\n      flat_values: A potentially ragged tensor.\n      nested_value_rowids: A list of 1-D integer tensors.  The `i`th tensor is\n        used as the `value_rowids` for the `i`th ragged dimension.\n      nested_nrows: A list of integer scalars.  The `i`th scalar is used as the\n        `nrows` for the `i`th ragged dimension.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor` (or `flat_values` if `nested_value_rowids` is empty).\n\n    Raises:\n      ValueError: If `len(nested_values_rowids) != len(nested_nrows)`.\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        if isinstance(nested_value_rowids, tensor_lib.Tensor):
            raise TypeError(f'Argument `nested_value_rowids` must be a list of Tensors. Received {nested_value_rowids}.')
        if nested_nrows is None:
            nested_nrows = [None] * len(nested_value_rowids)
        else:
            if isinstance(nested_nrows, tensor_lib.Tensor):
                raise TypeError(f'Argument `nested_nrows` must be a list of Tensors. Received {nested_nrows}.')
            if len(nested_nrows) != len(nested_value_rowids):
                raise ValueError(f'Argument `nested_nrows` must have the same length as argument `nested_value_rowids`. len(nested_nrows) = {len(nested_nrows)} vs. len(nested_values_rowids) = {len(nested_value_rowids)}.')
        with ops.name_scope(name, 'RaggedFromNestedValueRowIds', [flat_values] + list(nested_value_rowids) + list(nested_nrows)):
            result = flat_values
            for (value_rowids, nrows) in reversed(list(zip(nested_value_rowids, nested_nrows))):
                result = cls.from_value_rowids(result, value_rowids, nrows, validate=validate)
            return result

    @classmethod
    @dispatch.add_dispatch_support
    def from_nested_row_splits(cls, flat_values, nested_row_splits, name=None, validate=True):
        if False:
            i = 10
            return i + 15
        'Creates a `RaggedTensor` from a nested list of `row_splits` tensors.\n\n    Equivalent to:\n\n    ```python\n    result = flat_values\n    for row_splits in reversed(nested_row_splits):\n      result = from_row_splits(result, row_splits)\n    ```\n\n    Args:\n      flat_values: A potentially ragged tensor.\n      nested_row_splits: A list of 1-D integer tensors.  The `i`th tensor is\n        used as the `row_splits` for the `i`th ragged dimension.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor` (or `flat_values` if `nested_row_splits` is empty).\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        if isinstance(nested_row_splits, tensor_lib.Tensor):
            raise TypeError(f'Argument `nested_row_splits` must be a list of Tensors. Received {nested_row_splits}.')
        with ops.name_scope(name, 'RaggedFromNestedRowSplits', [flat_values] + list(nested_row_splits)):
            result = flat_values
            for splits in reversed(nested_row_splits):
                result = cls.from_row_splits(result, splits, validate=validate)
            return result

    @classmethod
    @dispatch.add_dispatch_support
    def from_nested_row_lengths(cls, flat_values, nested_row_lengths, name=None, validate=True):
        if False:
            i = 10
            return i + 15
        'Creates a `RaggedTensor` from a nested list of `row_lengths` tensors.\n\n    Equivalent to:\n\n    ```python\n    result = flat_values\n    for row_lengths in reversed(nested_row_lengths):\n      result = from_row_lengths(result, row_lengths)\n    ```\n\n    Args:\n      flat_values: A potentially ragged tensor.\n      nested_row_lengths: A list of 1-D integer tensors.  The `i`th tensor is\n        used as the `row_lengths` for the `i`th ragged dimension.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor` (or `flat_values` if `nested_row_lengths` is empty).\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        if isinstance(nested_row_lengths, tensor_lib.Tensor):
            raise TypeError(f'Argument `nested_row_lengths` must be a list of Tensors. Received {nested_row_lengths}.')
        with ops.name_scope(name, 'RaggedFromNestedRowlengths', [flat_values] + list(nested_row_lengths)):
            result = flat_values
            for lengths in reversed(nested_row_lengths):
                result = cls.from_row_lengths(result, lengths, validate=validate)
            return result

    @classmethod
    def _from_nested_row_partitions(cls, flat_values, nested_row_partitions, name=None, validate=True):
        if False:
            print('Hello World!')
        'Creates a `RaggedTensor` from a nested list of row partitions.\n\n    Equivalent to:\n\n    ```python\n    result = flat_values\n    for row_partition in reversed(nested_row_partitions):\n      result = _from_row_partition(result, row_partition)\n    ```\n\n    Args:\n      flat_values: A potentially ragged tensor.\n      nested_row_partitions: A list of row partitions.  The `i`th element is\n        used as the row partition for the `i`th ragged dimension.\n      name: A name prefix for the RaggedTensor (optional).\n      validate: If true, then use assertions to check that the arguments form\n        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,\n          since they must be checked for each tensor value.\n\n    Returns:\n      A `RaggedTensor` (or `flat_values` if `nested_row_lengths` is empty).\n    '
        if not isinstance(validate, bool):
            raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
        if isinstance(nested_row_partitions, RowPartition):
            raise TypeError(f'Argument `nested_row_partitions` must be a list of RowPartitions. Received {nested_row_partitions}.')
        if isinstance(nested_row_partitions, tensor_lib.Tensor):
            raise TypeError(f'Argument `nested_row_partitions` must be a list of RowPartitions. Received {nested_row_partitions}.')
        with ops.name_scope(name, 'RaggedFromNestedRowPartitions', [flat_values] + list(nested_row_partitions)):
            result = flat_values
            for partition in reversed(nested_row_partitions):
                result = cls._from_row_partition(result, partition, validate=validate)
            return result

    @classmethod
    def _convert_values_and_partition(cls, values, row_partition, name):
        if False:
            return 10
        'Converts `values` and `partition` to Tensors.\n\n    If `values` is a `RaggedTensor`, then converts `values` and `partition`\n    to have compatible row-partitioning dtypes.  In particular, if any of the\n    row partitioning tensors are `int64`, then all of the other row\n    partitioning tensors wil be cast to `int64` (if auto_cast_partition_dtype()\n    is true) or an error will be raised (if auto_cast_partition_dtype() is\n    false).\n\n    Args:\n      values: The `values` for the `RaggedTensor` being constructed.\n      row_partition: A RowPartition object for the `RaggedTensor` being\n        constructed.\n      name: The name of the RowPartition object.\n\n    Returns:\n      A tuple (values, partition).\n    '
        if not isinstance(row_partition, RowPartition):
            raise TypeError(f'Argument `row_partition` must be a RowPartition. Received {row_partition}.')
        if isinstance(values, RaggedTensor):
            if values._row_partition.dtype != row_partition.dtype:
                if not ragged_config.auto_cast_partition_dtype():
                    raise ValueError(f'Argument `row_partition` of RaggedTensor with name: {name} must have same dtype as Argument `values`. ({row_partition.dtype} vs. {values._row_partition.dtype}).')
                values = values.with_row_splits_dtype(row_partition.dtype)
        else:
            values = _convert_to_ragged_tensor_values(values)
        return (values, row_partition)

    @property
    def dtype(self):
        if False:
            return 10
        'The `DType` of values in this tensor.'
        return self._values.dtype

    @property
    def shape(self):
        if False:
            print('Hello World!')
        'The statically known shape of this ragged tensor.\n\n    Returns:\n      A `TensorShape` containing the statically known shape of this ragged\n      tensor.  Ragged dimensions have a size of `None`.\n\n    Examples:\n\n    >>> tf.ragged.constant([[0], [1, 2]]).shape\n    TensorShape([2, None])\n\n    >>> tf.ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1).shape\n    TensorShape([2, None, 2])\n\n    '
        nrows = self._row_partition.static_nrows
        ncols = self._row_partition.static_uniform_row_length
        value_shape = self._values.shape[1:]
        return tensor_shape.TensorShape([nrows, ncols]).concatenate(value_shape)

    def get_shape(self) -> tensor_shape.TensorShape:
        if False:
            while True:
                i = 10
        'The statically known shape of this ragged tensor.\n\n    Returns:\n      A `TensorShape` containing the statically known shape of this ragged\n      tensor.  Ragged dimensions have a size of `None`.\n\n    Alias for `shape` property.\n\n    Examples:\n\n    >>> tf.ragged.constant([[0], [1, 2]]).get_shape()\n    TensorShape([2, None])\n\n    >>> tf.ragged.constant(\n    ...    [[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1).get_shape()\n    TensorShape([2, None, 2])\n\n    '
        return self.shape

    @property
    def ragged_rank(self):
        if False:
            print('Hello World!')
        "The number of times the RaggedTensor's flat_values is partitioned.\n\n    Examples:\n\n    >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])\n    >>> values.ragged_rank\n    1\n\n    >>> rt = tf.RaggedTensor.from_uniform_row_length(values, 2)\n    >>> rt.ragged_rank\n    2\n\n    Returns:\n      A Python `int` indicating the number of times the underlying `flat_values`\n      Tensor has been partitioned to add a new dimension.\n      I.e., `tf.rank(rt) = tf.rank(rt.flat_values) + rt.ragged_rank`.\n    "
        values_is_ragged = isinstance(self._values, RaggedTensor)
        return self._values.ragged_rank + 1 if values_is_ragged else 1

    @property
    def values(self):
        if False:
            i = 10
            return i + 15
        'The concatenated rows for this ragged tensor.\n\n    `rt.values` is a potentially ragged tensor formed by flattening the two\n    outermost dimensions of `rt` into a single dimension.\n\n    `rt.values.shape = [nvals] + rt.shape[2:]` (where `nvals` is the\n    number of items in the outer two dimensions of `rt`).\n\n    `rt.ragged_rank = self.ragged_rank - 1`\n\n    Returns:\n      A potentially ragged tensor.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])\n    >>> print(rt.values)\n    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)\n\n    '
        return self._values

    @property
    def _nested_row_partitions(self):
        if False:
            while True:
                i = 10
        'Returns the row partitions for this `RaggedTensor`.'
        partitions = [self._row_partition]
        rt_values = self.values
        while isinstance(rt_values, RaggedTensor):
            partitions.append(rt_values._row_partition)
            rt_values = rt_values.values
        return tuple(partitions)

    @property
    def row_splits(self):
        if False:
            return 10
        "The row-split indices for this ragged tensor's `values`.\n\n    `rt.row_splits` specifies where the values for each row begin and end in\n    `rt.values`.  In particular, the values for row `rt[i]` are stored in\n    the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.\n\n    Returns:\n      A 1-D integer `Tensor` with shape `[self.nrows+1]`.\n      The returned tensor is non-empty, and is sorted in ascending order.\n      `self.row_splits[0]` is zero, and `self.row_splits[-1]` is equal to\n      `self.values.shape[0]`.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])\n    >>> print(rt.row_splits)  # indices of row splits in rt.values\n    tf.Tensor([0 4 4 7 8 8], shape=(6,), dtype=int64)\n\n    "
        return self._row_partition.row_splits()

    @property
    def uniform_row_length(self):
        if False:
            for i in range(10):
                print('nop')
        "The length of each row in this ragged tensor, or None if rows are ragged.\n\n    >>> rt1 = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])\n    >>> print(rt1.uniform_row_length)  # rows are ragged.\n    None\n\n    >>> rt2 = tf.RaggedTensor.from_uniform_row_length(\n    ...     values=rt1, uniform_row_length=2)\n    >>> print(rt2)\n    <tf.RaggedTensor [[[1, 2, 3], [4]], [[5, 6], [7, 8, 9, 10]]]>\n    >>> print(rt2.uniform_row_length)  # rows are not ragged (all have size 2).\n    tf.Tensor(2, shape=(), dtype=int64)\n\n    A RaggedTensor's rows are only considered to be uniform (i.e. non-ragged)\n    if it can be determined statically (at graph construction time) that the\n    rows all have the same length.\n\n    Returns:\n      A scalar integer `Tensor`, specifying the length of every row in this\n      ragged tensor (for ragged tensors whose rows are uniform); or `None`\n      (for ragged tensors whose rows are ragged).\n    "
        return self._row_partition.uniform_row_length()

    @property
    def flat_values(self):
        if False:
            for i in range(10):
                print('nop')
        'The innermost `values` tensor for this ragged tensor.\n\n    Concretely, if `rt.values` is a `Tensor`, then `rt.flat_values` is\n    `rt.values`; otherwise, `rt.flat_values` is `rt.values.flat_values`.\n\n    Conceptually, `flat_values` is the tensor formed by flattening the\n    outermost dimension and all of the ragged dimensions into a single\n    dimension.\n\n    `rt.flat_values.shape = [nvals] + rt.shape[rt.ragged_rank + 1:]`\n    (where `nvals` is the number of items in the flattened dimensions).\n\n    Returns:\n      A `Tensor`.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])\n    >>> print(rt.flat_values)\n    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)\n\n    '
        rt_values = self.values
        while isinstance(rt_values, RaggedTensor):
            rt_values = rt_values.values
        return rt_values

    @property
    def nested_row_splits(self):
        if False:
            return 10
        "A tuple containing the row_splits for all ragged dimensions.\n\n    `rt.nested_row_splits` is a tuple containing the `row_splits` tensors for\n    all ragged dimensions in `rt`, ordered from outermost to innermost.  In\n    particular, `rt.nested_row_splits = (rt.row_splits,) + value_splits` where:\n\n        * `value_splits = ()` if `rt.values` is a `Tensor`.\n        * `value_splits = rt.values.nested_row_splits` otherwise.\n\n    Returns:\n      A `tuple` of 1-D integer `Tensor`s.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant(\n    ...     [[[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]])\n    >>> for i, splits in enumerate(rt.nested_row_splits):\n    ...   print('Splits for dimension %d: %s' % (i+1, splits.numpy()))\n    Splits for dimension 1: [0 3]\n    Splits for dimension 2: [0 3 3 5]\n    Splits for dimension 3: [0 4 4 7 8 8]\n\n    "
        rt_nested_splits = [self.row_splits]
        rt_values = self.values
        while isinstance(rt_values, RaggedTensor):
            rt_nested_splits.append(rt_values.row_splits)
            rt_values = rt_values.values
        return tuple(rt_nested_splits)

    def value_rowids(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the row indices for the `values` in this ragged tensor.\n\n    `rt.value_rowids()` corresponds one-to-one with the outermost dimension of\n    `rt.values`, and specifies the row containing each value.  In particular,\n    the row `rt[row]` consists of the values `rt.values[j]` where\n    `rt.value_rowids()[j] == row`.\n\n    Args:\n      name: A name prefix for the returned tensor (optional).\n\n    Returns:\n      A 1-D integer `Tensor` with shape `self.values.shape[:1]`.\n      The returned tensor is nonnegative, and is sorted in ascending order.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])\n    >>> print(rt.values)\n    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)\n    >>> print(rt.value_rowids())  # corresponds 1:1 with rt.values\n    tf.Tensor([0 0 0 0 2 2 2 3], shape=(8,), dtype=int64)\n\n    '
        with ops.name_scope(name, 'RaggedValueRowIds', [self]):
            return self._row_partition.value_rowids()

    def nested_value_rowids(self, name=None):
        if False:
            print('Hello World!')
        "Returns a tuple containing the value_rowids for all ragged dimensions.\n\n    `rt.nested_value_rowids` is a tuple containing the `value_rowids` tensors\n    for\n    all ragged dimensions in `rt`, ordered from outermost to innermost.  In\n    particular, `rt.nested_value_rowids = (rt.value_rowids(),) + value_ids`\n    where:\n\n    * `value_ids = ()` if `rt.values` is a `Tensor`.\n    * `value_ids = rt.values.nested_value_rowids` otherwise.\n\n    Args:\n      name: A name prefix for the returned tensors (optional).\n\n    Returns:\n      A `tuple` of 1-D integer `Tensor`s.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant(\n    ...     [[[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]])\n    >>> for i, ids in enumerate(rt.nested_value_rowids()):\n    ...   print('row ids for dimension %d: %s' % (i+1, ids.numpy()))\n    row ids for dimension 1: [0 0 0]\n    row ids for dimension 2: [0 0 0 2 2]\n    row ids for dimension 3: [0 0 0 0 2 2 2 3]\n\n    "
        with ops.name_scope(name, 'RaggedNestedValueRowIds', [self]):
            rt_nested_ids = [self.value_rowids()]
            rt_values = self.values
            while isinstance(rt_values, RaggedTensor):
                rt_nested_ids.append(rt_values.value_rowids())
                rt_values = rt_values.values
            return tuple(rt_nested_ids)

    def nrows(self, out_type=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of rows in this ragged tensor.\n\n    I.e., the size of the outermost dimension of the tensor.\n\n    Args:\n      out_type: `dtype` for the returned tensor.  Defaults to\n        `self.row_splits.dtype`.\n      name: A name prefix for the returned tensor (optional).\n\n    Returns:\n      A scalar `Tensor` with dtype `out_type`.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])\n    >>> print(rt.nrows())  # rt has 5 rows.\n    tf.Tensor(5, shape=(), dtype=int64)\n\n    '
        with ops.name_scope(name, 'RaggedNRows', [self]):
            if out_type is None:
                return self._row_partition.nrows()
            else:
                return math_ops.cast(self._row_partition.nrows(), dtype=out_type)

    def row_starts(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the start indices for rows in this ragged tensor.\n\n    These indices specify where the values for each row begin in\n    `self.values`.  `rt.row_starts()` is equal to `rt.row_splits[:-1]`.\n\n    Args:\n      name: A name prefix for the returned tensor (optional).\n\n    Returns:\n      A 1-D integer Tensor with shape `[nrows]`.\n      The returned tensor is nonnegative, and is sorted in ascending order.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])\n    >>> print(rt.values)\n    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)\n    >>> print(rt.row_starts())  # indices of row starts in rt.values\n    tf.Tensor([0 4 4 7 8], shape=(5,), dtype=int64)\n\n    '
        with ops.name_scope(name, 'RaggedRowStarts', [self]):
            return self._row_partition.row_starts()

    def row_limits(self, name=None):
        if False:
            print('Hello World!')
        'Returns the limit indices for rows in this ragged tensor.\n\n    These indices specify where the values for each row end in\n    `self.values`.  `rt.row_limits(self)` is equal to `rt.row_splits[:-1]`.\n\n    Args:\n      name: A name prefix for the returned tensor (optional).\n\n    Returns:\n      A 1-D integer Tensor with shape `[nrows]`.\n      The returned tensor is nonnegative, and is sorted in ascending order.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])\n    >>> print(rt.values)\n    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)\n    >>> print(rt.row_limits())  # indices of row limits in rt.values\n    tf.Tensor([4 4 7 8 8], shape=(5,), dtype=int64)\n\n    '
        with ops.name_scope(name, 'RaggedRowLimits', [self]):
            return self._row_partition.row_limits()

    def row_lengths(self, axis=1, name=None):
        if False:
            while True:
                i = 10
        'Returns the lengths of the rows in this ragged tensor.\n\n    `rt.row_lengths()[i]` indicates the number of values in the\n    `i`th row of `rt`.\n\n    Args:\n      axis: An integer constant indicating the axis whose row lengths should be\n        returned.\n      name: A name prefix for the returned tensor (optional).\n\n    Returns:\n      A potentially ragged integer Tensor with shape `self.shape[:axis]`.\n\n    Raises:\n      ValueError: If `axis` is out of bounds.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant(\n    ...     [[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []])\n    >>> print(rt.row_lengths())  # lengths of rows in rt\n    tf.Tensor([2 0 2 1 0], shape=(5,), dtype=int64)\n    >>> print(rt.row_lengths(axis=2))  # lengths of axis=2 rows.\n    <tf.RaggedTensor [[3, 1], [], [2, 1], [1], []]>\n\n    '
        if axis == 0:
            return self._row_partition.nrows()
        if axis == 1:
            return self._row_partition.row_lengths()
        with ops.name_scope(name, 'RaggedRowLengths', [self]):
            axis = array_ops.get_positive_axis(axis, self.shape.rank, ndims_name='rank(self)')
            if axis == 0:
                return self.nrows()
            elif axis == 1:
                splits = self.row_splits
                return splits[1:] - splits[:-1]
            elif isinstance(self.values, RaggedTensor):
                return self.with_values(self.values.row_lengths(axis - 1))
            else:
                shape = array_ops.shape(self.values, out_type=self._row_partition.dtype)
                return self.with_values(array_ops.ones(shape[:axis - 1], self._row_partition.dtype) * shape[axis - 1])

    def nested_row_lengths(self, name=None):
        if False:
            i = 10
            return i + 15
        'Returns a tuple containing the row_lengths for all ragged dimensions.\n\n    `rt.nested_row_lengths()` is a tuple containing the `row_lengths` tensors\n    for all ragged dimensions in `rt`, ordered from outermost to innermost.\n\n    Args:\n      name: A name prefix for the returned tensors (optional).\n\n    Returns:\n      A `tuple` of 1-D integer `Tensors`.  The length of the tuple is equal to\n      `self.ragged_rank`.\n    '
        with ops.name_scope(name, 'RaggedNestedRowLengths', [self]):
            rt_nested_row_lengths = []
            rt = self
            while isinstance(rt, RaggedTensor):
                rt_nested_row_lengths.append(rt.row_lengths())
                rt = rt.values
            return tuple(rt_nested_row_lengths)

    def bounding_shape(self, axis=None, name=None, out_type=None):
        if False:
            return 10
        'Returns the tight bounding box shape for this `RaggedTensor`.\n\n    Args:\n      axis: An integer scalar or vector indicating which axes to return the\n        bounding box for.  If not specified, then the full bounding box is\n        returned.\n      name: A name prefix for the returned tensor (optional).\n      out_type: `dtype` for the returned tensor.  Defaults to\n        `self.row_splits.dtype`.\n\n    Returns:\n      An integer `Tensor` (`dtype=self.row_splits.dtype`).  If `axis` is not\n      specified, then `output` is a vector with\n      `output.shape=[self.shape.ndims]`.  If `axis` is a scalar, then the\n      `output` is a scalar.  If `axis` is a vector, then `output` is a vector,\n      where `output[i]` is the bounding size for dimension `axis[i]`.\n\n    #### Example:\n\n    >>> rt = tf.ragged.constant([[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]])\n    >>> rt.bounding_shape().numpy()\n    array([5, 4])\n\n    '
        if out_type is None:
            out_type = self._row_partition.dtype
        else:
            out_type = dtypes.as_dtype(out_type)
        with ops.name_scope(name, 'RaggedBoundingBox', [self, axis]):
            nested_splits = self.nested_row_splits
            rt_flat_values = self.flat_values
            if isinstance(axis, int):
                if axis == 0:
                    return array_ops.shape(nested_splits[0], out_type=out_type)[0] - 1
                elif axis == 1:
                    result = math_ops.maximum(math_ops.reduce_max(self.row_lengths()), 0)
                    if out_type != self._row_partition.dtype:
                        result = math_ops.cast(result, out_type)
                    return result
            splits_shape = array_ops.shape(self.row_splits, out_type=out_type)
            flat_values_shape = array_ops.shape(rt_flat_values, out_type=out_type)
            ragged_dimensions = [splits_shape[0] - 1] + [math_ops.maximum(math_ops.reduce_max(splits[1:] - splits[:-1]), 0) for splits in nested_splits]
            inner_dimensions = flat_values_shape[1:]
            if out_type != self._row_partition.dtype:
                ragged_dimensions = [math_ops.cast(d, out_type) for d in ragged_dimensions]
            bbox = array_ops.concat([array_ops_stack.stack(ragged_dimensions), inner_dimensions], axis=0)
            return bbox if axis is None else array_ops.gather(bbox, axis)

    def with_values(self, new_values):
        if False:
            for i in range(10):
                print('nop')
        'Returns a copy of `self` with `values` replaced by `new_value`.\n\n    Preserves cached row-partitioning tensors such as `self.cached_nrows` and\n    `self.cached_value_rowids` if they have values.\n\n    Args:\n      new_values: Potentially ragged tensor to use as the `values` for the\n        returned `RaggedTensor`.  Must have `rank > 0`, and must have the same\n        number of rows as `self.values`.\n\n    Returns:\n      A `RaggedTensor`.  `result.rank = 1 + new_values.rank`.\n      `result.ragged_rank = 1 + new_values.ragged_rank`\n    '
        new_values = _convert_to_ragged_tensor_values(new_values)
        new_values.shape.with_rank_at_least(1)
        self.values.shape[:1].assert_is_compatible_with(new_values.shape[:1])
        if isinstance(new_values, RaggedTensor) and self._row_partition.dtype != new_values.row_splits.dtype:
            if not ragged_config.auto_cast_partition_dtype():
                raise ValueError('self and new_values have mismatched row_splits dtypes; use RaggedTensor.with_row_splits_dtype() to convert them to compatible dtypes.')
            new_values = new_values.with_row_splits_dtype(dtypes.int64)
            return self.with_row_splits_dtype(dtypes.int64).with_values(new_values)
        return RaggedTensor(values=new_values, row_partition=self._row_partition, internal=True)

    def with_flat_values(self, new_values):
        if False:
            return 10
        'Returns a copy of `self` with `flat_values` replaced by `new_value`.\n\n    Preserves cached row-partitioning tensors such as `self.cached_nrows` and\n    `self.cached_value_rowids` if they have values.\n\n    Args:\n      new_values: Potentially ragged tensor that should replace\n        `self.flat_values`.  Must have `rank > 0`, and must have the same number\n        of rows as `self.flat_values`.\n\n    Returns:\n      A `RaggedTensor`.\n      `result.rank = self.ragged_rank + new_values.rank`.\n      `result.ragged_rank = self.ragged_rank + new_values.ragged_rank`.\n    '
        if isinstance(self._values, RaggedTensor):
            return self.with_values(self.values.with_flat_values(new_values))
        else:
            new_values = _convert_to_ragged_tensor_values(new_values)
        return self.with_values(new_values)

    def with_row_splits_dtype(self, dtype):
        if False:
            print('Hello World!')
        'Returns a copy of this RaggedTensor with the given `row_splits` dtype.\n\n    For RaggedTensors with multiple ragged dimensions, the `row_splits` for all\n    nested `RaggedTensor` objects are cast to the given dtype.\n\n    Args:\n      dtype: The dtype for `row_splits`.  One of `tf.int32` or `tf.int64`.\n\n    Returns:\n      A copy of this RaggedTensor, with the `row_splits` cast to the given\n      type.\n    '
        dtype = dtypes.as_dtype(dtype)
        if dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError(f'Argument `row_splits` dtype must be int32 or int64. Received {dtype}.')
        if self._row_partition.dtype == dtype:
            return self
        current_values = self._values
        if isinstance(current_values, RaggedTensor):
            return RaggedTensor(values=current_values.with_row_splits_dtype(dtype), row_partition=self._row_partition.with_dtype(dtype), internal=True)
        else:
            return RaggedTensor(values=current_values, row_partition=self._row_partition.with_dtype(dtype), internal=True)

    def merge_dims(self, outer_axis, inner_axis):
        if False:
            print('Hello World!')
        'Merges outer_axis...inner_axis into a single dimension.\n\n    Returns a copy of this RaggedTensor with the specified range of dimensions\n    flattened into a single dimension, with elements in row-major order.\n\n    #### Examples:\n\n    >>> rt = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6]]])\n    >>> print(rt.merge_dims(0, 1))\n    <tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>\n    >>> print(rt.merge_dims(1, 2))\n    <tf.RaggedTensor [[1, 2, 3], [4, 5, 6]]>\n    >>> print(rt.merge_dims(0, 2))\n    tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32)\n\n    To mimic the behavior of `np.flatten` (which flattens all dimensions), use\n    `rt.merge_dims(0, -1).  To mimic the behavior of `tf.layers.Flatten` (which\n    flattens all dimensions except the outermost batch dimension), use\n    `rt.merge_dims(1, -1)`.\n\n    Args:\n      outer_axis: `int`: The first dimension in the range of dimensions to\n        merge. May be negative if `self.shape.rank` is statically known.\n      inner_axis: `int`: The last dimension in the range of dimensions to merge.\n        May be negative if `self.shape.rank` is statically known.\n\n    Returns:\n      A copy of this tensor, with the specified dimensions merged into a\n      single dimension.  The shape of the returned tensor will be\n      `self.shape[:outer_axis] + [N] + self.shape[inner_axis + 1:]`, where `N`\n      is the total number of slices in the merged dimensions.\n    '
        outer_axis = array_ops.get_positive_axis(outer_axis, self.shape.rank, axis_name='outer_axis', ndims_name='rank(self)')
        inner_axis = array_ops.get_positive_axis(inner_axis, self.shape.rank, axis_name='inner_axis', ndims_name='rank(self)')
        if not outer_axis <= inner_axis:
            raise ValueError(f'Expected outer_axis ({outer_axis}) to be less than or equal to inner_axis ({inner_axis}).')
        return merge_dims(self, outer_axis, inner_axis)

    def _set_shape(self, shape):
        if False:
            return 10
        "Updates the static shape of `self` to be `shape`.\n\n    * If a dimension of `shape` has known rank, and is encoded via\n      partitioning, then this will update the corresponding partition to\n      define `_uniform_row_length` and `nrows`.\n    * If a dimension of `shape` has a known rank, and is encoded as one\n      of the `flat_values` dimensions, then `flat_values.set_shape()` will\n      be used to update its shape.\n\n    Warning: Using this method to assert an incorrect shape for a RaggedTensor\n    (i.e., one that's not consistent with its actual shape) can cause\n    segmentation faults and very difficult-to-diagnose behavior.  Only use this\n    method if you are certain that the shape is correct.\n\n    Args:\n      shape: `tf.TensorShape` specifying the shape for this `RaggedTensor`.\n    "
        shape = tensor_shape.as_shape(shape)
        if shape.rank is None:
            return
        shape = shape.as_list()
        if shape[0] is not None:
            self._row_partition._row_splits.set_shape(shape[0] + 1)
        dtype = self._row_partition.dtype
        for (i, partition) in enumerate(self._nested_row_partitions):
            size = shape[i + 1]
            if size is not None:
                if partition._uniform_row_length is not None:
                    old_row_length = tensor_util.constant_value(partition._uniform_row_length)
                    if old_row_length is not None:
                        if size == old_row_length:
                            continue
                        else:
                            raise ValueError(f'Inconsistent size for axis {i + 1}: {old_row_length} vs. {size}.')
                partition._uniform_row_length = ops.convert_to_tensor(size, dtype)
                if partition._nrows is None:
                    partition._nrows = array_ops.size(partition._row_splits, out_type=dtype) - 1
        if hasattr(self.flat_values, 'set_shape'):
            flat_shape = tensor_shape.as_shape([None] + shape[self.ragged_rank + 1:])
            self.flat_values.set_shape(flat_shape)

    @classmethod
    @dispatch.add_dispatch_support
    def from_tensor(cls, tensor, lengths=None, padding=None, ragged_rank=1, name=None, row_splits_dtype=dtypes.int64):
        if False:
            for i in range(10):
                print('nop')
        "Converts a `tf.Tensor` into a `RaggedTensor`.\n\n    The set of absent/default values may be specified using a vector of lengths\n    or a padding value (but not both).  If `lengths` is specified, then the\n    output tensor will satisfy `output[row] = tensor[row][:lengths[row]]`. If\n    'lengths' is a list of lists or tuple of lists, those lists will be used\n    as nested row lengths. If `padding` is specified, then any row *suffix*\n    consisting entirely of `padding` will be excluded from the returned\n    `RaggedTensor`.  If neither `lengths` nor `padding` is specified, then the\n    returned `RaggedTensor` will have no absent/default values.\n\n    Examples:\n\n    >>> dt = tf.constant([[5, 7, 0], [0, 3, 0], [6, 0, 0]])\n    >>> tf.RaggedTensor.from_tensor(dt)\n    <tf.RaggedTensor [[5, 7, 0], [0, 3, 0], [6, 0, 0]]>\n    >>> tf.RaggedTensor.from_tensor(dt, lengths=[1, 0, 3])\n    <tf.RaggedTensor [[5], [], [6, 0, 0]]>\n\n    >>> tf.RaggedTensor.from_tensor(dt, padding=0)\n    <tf.RaggedTensor [[5, 7], [0, 3], [6]]>\n\n    >>> dt = tf.constant([[[5, 0], [7, 0], [0, 0]],\n    ...                   [[0, 0], [3, 0], [0, 0]],\n    ...                   [[6, 0], [0, 0], [0, 0]]])\n    >>> tf.RaggedTensor.from_tensor(dt, lengths=([2, 0, 3], [1, 1, 2, 0, 1]))\n    <tf.RaggedTensor [[[5], [7]], [], [[6, 0], [], [0]]]>\n\n    Args:\n      tensor: The `Tensor` to convert.  Must have rank `ragged_rank + 1` or\n        higher.\n      lengths: An optional set of row lengths, specified using a 1-D integer\n        `Tensor` whose length is equal to `tensor.shape[0]` (the number of rows\n        in `tensor`).  If specified, then `output[row]` will contain\n        `tensor[row][:lengths[row]]`.  Negative lengths are treated as zero. You\n          may optionally pass a list or tuple of lengths to this argument, which\n          will be used as nested row lengths to construct a ragged tensor with\n          multiple ragged dimensions.\n      padding: An optional padding value.  If specified, then any row suffix\n        consisting entirely of `padding` will be excluded from the returned\n        RaggedTensor.  `padding` is a `Tensor` with the same dtype as `tensor`\n        and with `shape=tensor.shape[ragged_rank + 1:]`.\n      ragged_rank: Integer specifying the ragged rank for the returned\n        `RaggedTensor`.  Must be greater than zero.\n      name: A name prefix for the returned tensors (optional).\n      row_splits_dtype: `dtype` for the returned `RaggedTensor`'s `row_splits`\n        tensor.  One of `tf.int32` or `tf.int64`.\n\n    Returns:\n      A `RaggedTensor` with the specified `ragged_rank`.  The shape of the\n      returned ragged tensor is compatible with the shape of `tensor`.\n\n    Raises:\n      ValueError: If both `lengths` and `padding` are specified.\n      ValueError: If the rank of `tensor` is 0 or 1.\n    "
        row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
        if lengths is not None and padding is not None:
            raise ValueError('Specify argument `lengths` or `padding`, but not both.')
        if not isinstance(ragged_rank, int):
            raise TypeError(f'Argument `ragged_rank` must be an int. Received {ragged_rank}.')
        if ragged_rank <= 0:
            raise ValueError(f'Argument `ragged_rank` must be greater than 0. Received {ragged_rank}.')
        with ops.name_scope(name, 'RaggedFromTensor', [tensor, lengths, padding]):
            tensor = ops.convert_to_tensor(tensor, name='tensor')
            if tensor.shape.rank is not None and tensor.shape.rank < 2:
                raise ValueError(f"The rank of a RaggedTensor must be greater than 1, i.e., a list of scalars won't have ragged dimensions. Received argument `tensor` with rank {tensor.shape.rank}.")
            tensor.shape.with_rank_at_least(ragged_rank + 1)
            input_shape = array_ops.shape(tensor, out_type=row_splits_dtype)
            ncols = input_shape[1]
            if lengths is not None and isinstance(lengths, (list, tuple)) and len(lengths) and (not isinstance(lengths[0], (int, float))):
                if ragged_rank not in (1, len(lengths)):
                    raise ValueError(f'If Argument `lengths` is a tuple of row_lengths, argument `ragged_rank` must be len(lengths): {len(lengths)}. Received ragged_rank: {ragged_rank}.')
                tensor.shape.with_rank_at_least(len(lengths) + 1)
                num_tokens = math_ops.reduce_sum(lengths[-1])
                ones_mask = array_ops.ones([num_tokens], dtype=dtypes.bool)
                ragged_mask = cls.from_nested_row_lengths(ones_mask, lengths, validate=False)
                dense_ragged_mask = ragged_mask.to_tensor(default_value=False)
                masked_data = array_ops.boolean_mask(tensor, dense_ragged_mask)
                return cls.from_nested_row_lengths(masked_data, lengths, validate=False)
            if ragged_rank > 1:
                if tensor.shape.is_fully_defined():
                    input_shape = tensor.shape.as_list()
                    dim_size = np.cumprod(input_shape)
                    new_shape = [dim_size[ragged_rank - 1]] + input_shape[ragged_rank:]
                else:
                    dim_size = math_ops.cumprod(input_shape)
                    new_shape = array_ops.concat([[dim_size[ragged_rank - 1]], input_shape[ragged_rank:]], axis=0)
                flattened = array_ops.reshape(tensor, new_shape)
                result = cls.from_tensor(flattened, lengths, padding, row_splits_dtype=row_splits_dtype)
                for axis in range(ragged_rank - 1, 0, -1):
                    dim_len = tensor_shape.dimension_at_index(tensor.shape, axis).value
                    if dim_len is None:
                        dim_len = input_shape[axis]
                    else:
                        dim_len = constant_op.constant(dim_len, row_splits_dtype)
                    result = RaggedTensor.from_uniform_row_length(values=result, uniform_row_length=dim_len, nrows=dim_size[axis - 1], validate=False)
                return result
            if padding is not None:
                padding = ops.convert_to_tensor(padding, name='padding', dtype=tensor.dtype)
                padding.shape.assert_is_compatible_with(tensor.shape[2:])
                has_default_value = math_ops.equal(padding, tensor)
                tensor_rank = array_ops.rank(tensor)
                reduce_axis = math_ops.range(2, tensor_rank)
                has_default = cond.cond(tensor_rank > 2, lambda : math_ops.reduce_all(has_default_value, axis=reduce_axis), lambda : has_default_value)
                has_default.set_shape(tensor_shape.TensorShape([None, None]))
                has_default.set_shape(tensor.shape[:2])
                has_nondefault = math_ops.logical_not(has_default)
                has_nondefault = math_ops.cast(has_nondefault, row_splits_dtype)
                length_for_nondefault_value = has_nondefault * array_ops.expand_dims(math_ops.range(1, ncols + 1), 0)
                lengths = math_ops.reduce_max(length_for_nondefault_value, axis=1)
            if lengths is not None:
                lengths = ragged_util.convert_to_int_tensor(lengths, 'lengths', row_splits_dtype)
                lengths.shape.assert_has_rank(1)
                lengths = math_ops.minimum(lengths, ncols)
                lengths = math_ops.maximum(lengths, 0)
                limits = math_ops.cumsum(lengths)
                splits = array_ops.concat([array_ops.zeros([1], row_splits_dtype), limits], axis=0)
                mask = array_ops.sequence_mask(lengths, maxlen=ncols)
                values = array_ops.boolean_mask(tensor, mask)
                return cls.from_row_splits(values, splits, validate=False)
            values_shape = array_ops.concat([[input_shape[0] * input_shape[1]], input_shape[2:]], axis=0)
            values = array_ops.reshape(tensor, values_shape)
            const_nrows = tensor_shape.dimension_at_index(tensor.shape, 0).value
            const_ncols = tensor_shape.dimension_at_index(tensor.shape, 1).value
            if const_nrows is not None:
                nrows = constant_op.constant(const_nrows, row_splits_dtype)
            else:
                nrows = input_shape[0]
            if const_ncols is not None:
                ncols = constant_op.constant(const_ncols, row_splits_dtype)
            else:
                ncols = input_shape[1]
            return RaggedTensor.from_uniform_row_length(values=values, uniform_row_length=ncols, nrows=nrows, validate=False)

    def to_tensor(self, default_value=None, name=None, shape=None):
        if False:
            print('Hello World!')
        'Converts this `RaggedTensor` into a `tf.Tensor`.\n\n    If `shape` is specified, then the result is padded and/or truncated to\n    the specified shape.\n\n    Examples:\n\n    >>> rt = tf.ragged.constant([[9, 8, 7], [], [6, 5], [4]])\n    >>> print(rt.to_tensor())\n    tf.Tensor(\n        [[9 8 7] [0 0 0] [6 5 0] [4 0 0]], shape=(4, 3), dtype=int32)\n    >>> print(rt.to_tensor(shape=[5, 2]))\n    tf.Tensor(\n        [[9 8] [0 0] [6 5] [4 0] [0 0]], shape=(5, 2), dtype=int32)\n\n    Args:\n      default_value: Value to set for indices not specified in `self`. Defaults\n        to zero.  `default_value` must be broadcastable to\n        `self.shape[self.ragged_rank + 1:]`.\n      name: A name prefix for the returned tensors (optional).\n      shape: The shape of the resulting dense tensor.  In particular,\n        `result.shape[i]` is `shape[i]` (if `shape[i]` is not None), or\n        `self.bounding_shape(i)` (otherwise).`shape.rank` must be `None` or\n        equal to `self.rank`.\n\n    Returns:\n      A `Tensor` with shape `ragged.bounding_shape(self)` and the\n      values specified by the non-empty values in `self`.  Empty values are\n      assigned `default_value`.\n    '
        with ops.name_scope(name, 'RaggedToTensor', [self, default_value, shape]):
            if default_value is not None:
                default_value = ops.convert_to_tensor(default_value, name='default_value', dtype=self.dtype)
            type_tensor_pairs = _get_row_partition_type_tensor_pairs(self)
            row_partition_types = [x[0] for x in type_tensor_pairs]
            row_partition_tensors = [x[1] for x in type_tensor_pairs]
            if default_value is None:
                default_value = array_ops.zeros((), self.dtype)
            if isinstance(shape, (list, tuple)) and any((isinstance(v, tensor_lib.Tensor) for v in shape)) and all((isinstance(v, (int, tensor_lib.Tensor)) for v in shape)):
                shape = array_ops_stack.stack(shape)
            shape_tensor = _shape_as_tensor(shape, row_partition_tensors[0].dtype)
            tensor = gen_ragged_conversion_ops.ragged_tensor_to_tensor(shape=shape_tensor, values=self.flat_values, default_value=default_value, row_partition_types=row_partition_types, row_partition_tensors=row_partition_tensors)
            ragged_shape = self.shape
            if ragged_shape.rank is not None and (not isinstance(shape, tensor_lib.Tensor)):
                shape = tensor_shape.as_shape(shape)
                if shape.rank is None:
                    output_shape = ragged_shape
                else:
                    output_shape = [s1 if s1 is not None else s2 for (s1, s2) in zip(shape.as_list(), ragged_shape.as_list())]
                tensor.set_shape(output_shape)
            return tensor

    @classmethod
    @dispatch.add_dispatch_support
    def from_sparse(cls, st_input, name=None, row_splits_dtype=dtypes.int64):
        if False:
            while True:
                i = 10
        "Converts a 2D `tf.sparse.SparseTensor` to a `RaggedTensor`.\n\n    Each row of the `output` `RaggedTensor` will contain the explicit values\n    from the same row in `st_input`.  `st_input` must be ragged-right.  If not\n    it is not ragged-right, then an error will be generated.\n\n    Example:\n\n    >>> indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0]]\n    >>> st = tf.sparse.SparseTensor(indices=indices,\n    ...                             values=[1, 2, 3, 4, 5],\n    ...                             dense_shape=[4, 3])\n    >>> tf.RaggedTensor.from_sparse(st).to_list()\n    [[1, 2, 3], [4], [], [5]]\n\n    Currently, only two-dimensional `SparseTensors` are supported.\n\n    Args:\n      st_input: The sparse tensor to convert.  Must have rank 2.\n      name: A name prefix for the returned tensors (optional).\n      row_splits_dtype: `dtype` for the returned `RaggedTensor`'s `row_splits`\n        tensor.  One of `tf.int32` or `tf.int64`.\n\n    Returns:\n      A `RaggedTensor` with the same values as `st_input`.\n      `output.ragged_rank = rank(st_input) - 1`.\n      `output.shape = [st_input.dense_shape[0], None]`.\n    Raises:\n      ValueError: If the number of dimensions in `st_input` is not known\n        statically, or is not two.\n    "
        row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
        if not sparse_tensor.is_sparse(st_input):
            raise TypeError(f'Argument `st_input` must be of type SparseTensor, but is of type {type(st_input).__name__}.')
        with ops.name_scope(name, 'RaggedFromSparse', [st_input]):
            st_input = sparse_tensor.convert_to_tensor_or_sparse_tensor(st_input, name='st_input')
            if st_input.dense_shape.shape.ndims is None:
                static_rank_from_dense_shape = None
            else:
                static_rank_from_dense_shape = st_input.dense_shape.shape.dims[0].value
            if st_input.indices.shape.ndims is None:
                static_rank_from_indices = None
            else:
                static_rank_from_indices = st_input.indices.shape.dims[1].value
            if static_rank_from_dense_shape != 2 and static_rank_from_indices != 2:
                raise ValueError('rank(st_input) must be 2.')
            with ops.control_dependencies(_assert_sparse_indices_are_ragged_right(st_input.indices)):
                segment_ids = math_ops.cast(st_input.indices[:, 0], row_splits_dtype)
                num_segments = math_ops.cast(st_input.dense_shape[0], row_splits_dtype)
                return cls.from_value_rowids(st_input.values, segment_ids, num_segments, validate=False)

    def to_sparse(self, name=None):
        if False:
            while True:
                i = 10
        'Converts this `RaggedTensor` into a `tf.sparse.SparseTensor`.\n\n    Example:\n\n    >>> rt = tf.ragged.constant([[1, 2, 3], [4], [], [5, 6]])\n    >>> print(rt.to_sparse())\n    SparseTensor(indices=tf.Tensor(\n                     [[0 0] [0 1] [0 2] [1 0] [3 0] [3 1]],\n                     shape=(6, 2), dtype=int64),\n                 values=tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32),\n                 dense_shape=tf.Tensor([4 3], shape=(2,), dtype=int64))\n\n    Args:\n      name: A name prefix for the returned tensors (optional).\n\n    Returns:\n      A SparseTensor with the same values as `self`.\n    '
        with ops.name_scope(name, 'RaggedToSparse', [self]):
            result = gen_ragged_conversion_ops.ragged_tensor_to_sparse(self.nested_row_splits, self.flat_values, name=name)
            return sparse_tensor.SparseTensor(result.sparse_indices, result.sparse_values, result.sparse_dense_shape)

    @classmethod
    def _from_variant(cls, variant, dtype, output_ragged_rank, input_ragged_rank=None, row_splits_dtype=dtypes.int64, name=None):
        if False:
            return 10
        "Converts a `variant` Tensor into a `RaggedTensor`.\n\n    The input `variant` could be a scalar, meaning it encodes a single\n    `RaggedTensor` with ragged_rank `output_ragged_rank`. Alternatively it could\n    have an arbitrary rank, in which case each element is decoded into a\n    `RaggedTensor` with ragged_rank `input_ragged_rank` and these are then\n    stacked according to the input shape to output a single `RaggedTensor`\n    with ragged_rank `output_ragged_rank`. If `input_ragged_rank` is not\n    provided, it is inferred dynamically as `output_ragged_rank` -\n    `rank(variant)`. If `input_ragged_rank` is provided, the following must be\n    true: `output_ragged_rank` = `input_ragged_rank` + `rank(variant)`.\n\n    Example:\n\n    >>> rt = tf.ragged.constant([[0], [1, 2]])\n    >>> et = rt._to_variant()\n    >>> stacked_et = tf.stack([et, et])\n    >>> tf.RaggedTensor._from_variant(  # scalar input.\n    ...     et, dtype=tf.int32, output_ragged_rank=1).to_list()\n    [[0], [1, 2]]\n    >>> tf.RaggedTensor._from_variant(  # batched input.\n    ...     stacked_et, dtype=tf.int32, output_ragged_rank=2).to_list()\n    [[[0], [1, 2]], [[0], [1, 2]]]\n\n    Args:\n      variant: A `variant` Tensor representing an encoded (possibly\n        nested-batched) `RaggedTensor`.\n      dtype: The dtype of the encoded `RaggedTensor`.\n      output_ragged_rank: The expected ragged rank of the output `RaggedTensor`.\n      input_ragged_rank: The ragged rank of each encoded `RaggedTensor`. This is\n        optional and inferred dynamically if not provided.\n      row_splits_dtype: `dtype` for the RaggedTensor's `row_splits` tensor. One\n        of `tf.int32` or `tf.int64`.\n      name: A name prefix for the returned tensors (optional).\n\n    Returns:\n      A `RaggedTensor` of dtype `dtype` and ragged rank `output_ragged_rank`.\n\n    Raises:\n      ValueError: If the input rank is known, `input_ragged_rank` is provided\n          and `output_ragged_rank` = `input_ragged_rank` + `rank(variant)` does\n          not hold.\n    "
        variant = ops.convert_to_tensor(variant, name='variant', dtype=dtypes.variant)
        if variant.shape.ndims is not None and input_ragged_rank is not None and (output_ragged_rank != input_ragged_rank + variant.shape.ndims):
            raise ValueError(f'Argument `output_ragged_rank` ({output_ragged_rank}) must be equal to `input_ragged_rank` + `variant.shape.ndims` ({input_ragged_rank} + {variant.shape.ndims}).')
        input_ragged_rank = -1 if input_ragged_rank is None else input_ragged_rank
        with ops.name_scope(name, 'RaggedFromVariant', [variant, dtype, input_ragged_rank, output_ragged_rank]):
            result = gen_ragged_conversion_ops.ragged_tensor_from_variant(variant, input_ragged_rank, max(output_ragged_rank, 0), dtype, row_splits_dtype, name)
            return cls.from_nested_row_splits(result.output_dense_values, result.output_nested_splits, validate=False)

    def _to_variant(self, batched_input=False, name=None):
        if False:
            while True:
                i = 10
        'Converts this `RaggedTensor` into a `variant` Tensor.\n\n    If `batched_input` is `True`, then the `RaggedTensor` is unbatched along the\n    zero-th dimension, each component `RaggedTensor` is encoded into a scalar\n    `variant` Tensor, and these are stacked to return a 1-D `variant` Tensor.\n    If `batched_input` is `False`, then the `RaggedTensor` is encoded as is and\n    a scalar `variant` Tensor is returned.\n\n    Example:\n    >>> rt = tf.ragged.constant([[[0]], [[1]], [[2]]])\n    >>> rt._to_variant().shape.as_list()\n    []\n    >>> rt._to_variant(batched_input=True).shape.as_list()\n    [3]\n\n    Args:\n      batched_input: If `True`, the `RaggedTensor` is unbatched and converted to\n        a `variant` vector. Set to `False` by default.\n      name: A name prefix for the returned tensors (optional).\n\n    Returns:\n      A `variant` Tensor that encodes this `RaggedTensor`.\n    '
        with ops.name_scope(name, 'RaggedToVariant', [self, batched_input]):
            return gen_ragged_conversion_ops.ragged_tensor_to_variant(self.nested_row_splits, self.flat_values, batched_input, name)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._is_eager():
            with np.printoptions(formatter={'all': _formatter}):
                value_text = _formatter(self.numpy())
            return f'<tf.RaggedTensor {value_text}>'
        else:
            return 'tf.RaggedTensor(values=%s, row_splits=%s)' % (self.values, self.row_splits)

    def numpy(self):
        if False:
            i = 10
            return i + 15
        'Returns a numpy `array` with the values for this `RaggedTensor`.\n\n    Requires that this `RaggedTensor` was constructed in eager execution mode.\n\n    Ragged dimensions are encoded using numpy `arrays` with `dtype=object` and\n    `rank=1`, where each element is a single row.\n\n    #### Examples\n\n    In the following example, the value returned by `RaggedTensor.numpy()`\n    contains three numpy `array` objects: one for each row (with `rank=1` and\n    `dtype=int64`), and one to combine them (with `rank=1` and `dtype=object`):\n\n    >>> tf.ragged.constant([[1, 2, 3], [4, 5]], dtype=tf.int64).numpy()\n    array([array([1, 2, 3]), array([4, 5])], dtype=object)\n\n    Uniform dimensions are encoded using multidimensional numpy `array`s.  In\n    the following example, the value returned by `RaggedTensor.numpy()` contains\n    a single numpy `array` object, with `rank=2` and `dtype=int64`:\n\n    >>> tf.ragged.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64).numpy()\n    array([[1, 2, 3], [4, 5, 6]])\n\n    Returns:\n      A numpy `array`.\n    '
        if not self._is_eager():
            raise ValueError('RaggedTensor.numpy() is only supported in eager mode.')
        values = self.values.numpy()
        splits = self.row_splits.numpy()
        rows = [values[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]
        if not rows:
            return np.zeros((0, 0) + values.shape[1:], dtype=values.dtype)
        has_variable_length_rows = any((len(row) != len(rows[0]) for row in rows))
        dtype = np.object_ if has_variable_length_rows else None
        return np.array(rows, dtype=dtype)

    def to_list(self):
        if False:
            return 10
        'Returns a nested Python `list` with the values for this `RaggedTensor`.\n\n    Requires that `rt` was constructed in eager execution mode.\n\n    Returns:\n      A nested Python `list`.\n    '
        if not isinstance(self.row_splits, ops.EagerTensor):
            raise ValueError('to_list can only be used in eager mode.')
        row_splits = self.row_splits.numpy().tolist()
        values = self.values
        if isinstance(values, RaggedTensor):
            return [values[row_splits[i]:row_splits[i + 1]].to_list() for i in range(len(row_splits) - 1)]
        else:
            if hasattr(values, 'numpy'):
                values_as_list = values.numpy().tolist()
            elif hasattr(values, 'to_list'):
                values_as_list = values.to_list()
            else:
                raise ValueError('values must be convertible to a list')
            return [values_as_list[row_splits[i]:row_splits[i + 1]] for i in range(len(row_splits) - 1)]

    def _eager_value(self):
        if False:
            return 10
        'Returns a RaggedTensorValue for self.  Requires self._is_eager()=true.'
        value = self.flat_values.numpy()
        for row_splits in reversed(self.nested_row_splits):
            value = ragged_tensor_value.RaggedTensorValue(value, row_splits.numpy())
        return value

    def _is_eager(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if values & row_splits Tensors are all `EagerTensor`s.'
        rt = self
        while isinstance(rt, RaggedTensor):
            if not isinstance(rt.row_splits, ops.EagerTensor):
                return False
            rt = rt.values
        return isinstance(rt, ops.EagerTensor)

    def _overloaded_operator(name):
        if False:
            return 10

        def stub(*args, **kwargs):
            if False:
                return 10
            del args, kwargs
            raise ValueError(f"You must import 'tensorflow.python.ops.ragged.ragged_ops' before using RaggedTensor.{name}.")
        return stub
    __getitem__ = _overloaded_operator('__getitem__')
    __ge__ = _overloaded_operator('__ge__')
    __gt__ = _overloaded_operator('__gt__')
    __le__ = _overloaded_operator('__le__')
    __lt__ = _overloaded_operator('__lt__')
    __and__ = _overloaded_operator('__and__')
    __rand__ = _overloaded_operator('__rand__')
    __invert__ = _overloaded_operator('__invert__')
    __ror__ = _overloaded_operator('__ror__')
    __or__ = _overloaded_operator('__or__')
    __xor__ = _overloaded_operator('__xor__')
    __rxor__ = _overloaded_operator('__rxor__')
    __abs__ = _overloaded_operator('__abs__')
    __add__ = _overloaded_operator('__add__')
    __radd__ = _overloaded_operator('__radd__')
    __div__ = _overloaded_operator('__div__')
    __rdiv__ = _overloaded_operator('__rdiv__')
    __floordiv__ = _overloaded_operator('__floordiv__')
    __rfloordiv__ = _overloaded_operator('__rfloordiv__')
    __mod__ = _overloaded_operator('__mod__')
    __rmod__ = _overloaded_operator('__rmod__')
    __mul__ = _overloaded_operator('__mul__')
    __rmul__ = _overloaded_operator('__rmul__')
    __neg__ = _overloaded_operator('__neg__')
    __pow__ = _overloaded_operator('__pow__')
    __rpow__ = _overloaded_operator('__rpow__')
    __sub__ = _overloaded_operator('__sub__')
    __rsub__ = _overloaded_operator('__rsub__')
    __truediv__ = _overloaded_operator('__truediv__')
    __rtruediv__ = _overloaded_operator('__rtruediv__')
    del _overloaded_operator

    def _as_graph_element(self):
        if False:
            i = 10
            return i + 15
        'Convert `self` to a graph element.'
        values = self.values
        while isinstance(values, RaggedTensor):
            values = values.values
        return values

    @property
    def _type_spec(self):
        if False:
            i = 10
            return i + 15
        return RaggedTensorSpec.from_value(self)

    def _shape_invariant_to_type_spec(self, shape):
        if False:
            for i in range(10):
                print('nop')
        return RaggedTensorSpec(shape, self.dtype, self.ragged_rank, self.row_splits.dtype)

    def consumers(self):
        if False:
            i = 10
            return i + 15
        return self._consumers()
    __composite_gradient__ = composite_tensor_gradient.WithValuesCompositeTensorGradient()

def is_ragged(value):
    if False:
        print('Hello World!')
    'Returns true if `value` is a ragged tensor or ragged tensor value.'
    return isinstance(value, (RaggedTensor, ragged_tensor_value.RaggedTensorValue))

def match_row_splits_dtypes(*tensors, **kwargs):
    if False:
        while True:
            i = 10
    "Return a copy of `tensors` with row_splits all having the same dtype.\n\n  Args:\n    *tensors: A list of Tensors or RaggedTensors.\n    **kwargs: If 'return_dtype=True', then return a tuple (dtype, tensors),\n      where `dtype` is the data type used by row-splits, and `tensors` is the\n      converted list of `Tensors` and `RaggedTensors`.\n\n  Returns:\n    The converted list of `Tensors` and `RaggedTensors`.\n  "
    return_dtype = kwargs.pop('return_dtype', False)
    if kwargs:
        raise ValueError(f'Unexpected keyword args {kwargs}.')
    has_int32 = False
    has_int64 = False
    for tensor in tensors:
        if isinstance(tensor, RaggedTensor):
            if tensor.row_splits.dtype == dtypes.int32:
                has_int32 = True
            else:
                has_int64 = True
    if has_int32 and has_int64:
        if not ragged_config.auto_cast_partition_dtype():
            raise ValueError('Input RaggedTensors have mismatched row_splits dtypes; use RaggedTensor.with_row_splits_dtype() to convert them to compatible dtypes.')
        dtype = dtypes.int64
        tensors = tuple((t.with_row_splits_dtype(dtypes.int64) if isinstance(t, RaggedTensor) else t for t in tensors))
    elif has_int32:
        dtype = dtypes.int32
    else:
        dtype = dtypes.int64
    if return_dtype:
        return (dtype, tensors)
    else:
        return tensors

@tf_export('RaggedTensorSpec')
@type_spec_registry.register('tf.RaggedTensorSpec')
class RaggedTensorSpec(type_spec.BatchableTypeSpec, internal_types.RaggedTensorSpec):
    """Type specification for a `tf.RaggedTensor`."""
    __slots__ = ['_shape', '_dtype', '_ragged_rank', '_row_splits_dtype', '_flat_values_spec']

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        'The `tf.dtypes.DType` specified by this type for the RaggedTensor.\n\n    Examples:\n\n    >>> rt = tf.ragged.constant([["a"], ["b", "c"]], dtype=tf.string)\n    >>> tf.type_spec_from_value(rt).dtype\n    tf.string\n\n    Returns:\n      A `tf.dtypes.DType` of the values in the RaggedTensor.\n    '
        return self._dtype

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        'The statically known shape of the RaggedTensor.\n\n    Examples:\n\n    >>> rt = tf.ragged.constant([[0], [1, 2]])\n    >>> tf.type_spec_from_value(rt).shape\n    TensorShape([2, None])\n\n    >>> rt = tf.ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1)\n    >>> tf.type_spec_from_value(rt).shape\n    TensorShape([2, None, 2])\n\n    Returns:\n      A `tf.TensorShape` containing the statically known shape of the\n      RaggedTensor. Ragged dimensions have a size of `None`.\n    '
        return self._shape

    @property
    def ragged_rank(self):
        if False:
            return 10
        "The number of times the RaggedTensor's flat_values is partitioned.\n\n    Defaults to `shape.ndims - 1`.\n\n    Examples:\n\n    >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])\n    >>> tf.type_spec_from_value(values).ragged_rank\n    1\n\n    >>> rt1 = tf.RaggedTensor.from_uniform_row_length(values, 2)\n    >>> tf.type_spec_from_value(rt1).ragged_rank\n    2\n\n    Returns:\n      A Python `int` indicating the number of times the underlying `flat_values`\n      Tensor has been partitioned to add a new dimension.\n      I.e., `tf.rank(rt) = tf.rank(rt.flat_values) + rt.ragged_rank`.\n    "
        return self._ragged_rank

    @property
    def row_splits_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        "The `tf.dtypes.DType` of the RaggedTensor's `row_splits`.\n\n    Examples:\n\n    >>> rt = tf.ragged.constant([[1, 2, 3], [4]], row_splits_dtype=tf.int64)\n    >>> tf.type_spec_from_value(rt).row_splits_dtype\n    tf.int64\n\n    Returns:\n      A `tf.dtypes.DType` for the RaggedTensor's `row_splits` tensor. One\n      of `tf.int32` or `tf.int64`.\n    "
        return self._row_splits_dtype

    @property
    def flat_values_spec(self):
        if False:
            while True:
                i = 10
        'The `TypeSpec` of the flat_values of RaggedTensor.\n\n    Returns:\n      - The TypeSpec of flat_values.\n      - None when the flat_values is a Tensor.\n    '
        return self._flat_values_spec

    @property
    def value_type(self):
        if False:
            i = 10
            return i + 15
        return RaggedTensor if self._ragged_rank > 0 else tensor_lib.Tensor

    def __init__(self, shape=None, dtype=dtypes.float32, ragged_rank=None, row_splits_dtype=dtypes.int64, flat_values_spec=None):
        if False:
            while True:
                i = 10
        "Constructs a type specification for a `tf.RaggedTensor`.\n\n    Args:\n      shape: The shape of the RaggedTensor, or `None` to allow any shape.  If a\n        shape is specified, then all ragged dimensions must have size `None`.\n      dtype: `tf.DType` of values in the RaggedTensor.\n      ragged_rank: Python integer, the number of times the RaggedTensor's\n        flat_values is partitioned.  Defaults to `shape.ndims - 1`.\n      row_splits_dtype: `dtype` for the RaggedTensor's `row_splits` tensor. One\n        of `tf.int32` or `tf.int64`.\n      flat_values_spec: TypeSpec for flat_value of the RaggedTensor. It shall be\n        provided when the flat_values is a CompositeTensor rather then Tensor.\n        If both `dtype` and `flat_values_spec` and  are provided, `dtype` must\n        be the same as `flat_values_spec.dtype`. (experimental)\n    "
        self._shape = tensor_shape.as_shape(shape)
        self._row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
        if flat_values_spec is not None:
            if dtype is None:
                dtype = flat_values_spec.dtype
            elif dtype != flat_values_spec.dtype:
                raise ValueError('dtype must be the same as flat_values_spec.dtype')
        elif dtype is None:
            raise ValueError('At least one of dtype or flat_values_spec must be provided')
        self._dtype = dtypes.as_dtype(dtype)
        self._flat_values_spec = flat_values_spec
        rank = self._shape.ndims
        if ragged_rank is None:
            if rank is None:
                raise ValueError('Must specify ragged_rank or a shape with a known rank.')
            ragged_rank = rank - 1
        self._ragged_rank = ragged_rank
        if not isinstance(self._ragged_rank, int):
            raise TypeError(f'Argument `ragged_rank` must be an int. Received {ragged_rank}.')
        if rank is not None:
            if ragged_rank >= rank:
                raise ValueError(f'Argument `ragged_rank` ({ragged_rank}) must be less than rank ({rank}).')

    def is_compatible_with(self, spec_or_value):
        if False:
            for i in range(10):
                print('nop')
        if self._ragged_rank == 0:
            if self._flat_values_spec is None:
                if isinstance(spec_or_value, (tensor_lib.Tensor, tensor_lib.TensorSpec)):
                    return tensor_lib.TensorSpec(self._shape, self._dtype).is_compatible_with(spec_or_value)
            elif not isinstance(spec_or_value, (RaggedTensor, RaggedTensorSpec)):
                return self._flat_values_spec.is_compatible_with(spec_or_value)
        return super(RaggedTensorSpec, self).is_compatible_with(spec_or_value)

    def _serialize(self):
        if False:
            return 10
        if self._flat_values_spec is None:
            return (self._shape, self._dtype, self._ragged_rank, self._row_splits_dtype)
        else:
            return (self._shape, self._dtype, self._ragged_rank, self._row_splits_dtype, self._flat_values_spec)

    @property
    def _component_specs(self):
        if False:
            print('Hello World!')
        if self._ragged_rank <= 0:
            if self._flat_values_spec is not None:
                return [self._flat_values_spec]
            else:
                return [tensor_lib.TensorSpec(self._shape, self._dtype)]
        flat_values_spec = self._flat_values_spec
        if flat_values_spec is None:
            flat_values_shape = tensor_shape.TensorShape([None]).concatenate(self._shape[self._ragged_rank + 1:])
            flat_values_spec = tensor_lib.TensorSpec(flat_values_shape, self._dtype)
        outer_dim = tensor_shape.dimension_at_index(self._shape, 0)
        outer_splits_shape = [None if outer_dim is None else outer_dim + 1]
        inner_splits_spec = tensor_lib.TensorSpec([None], self._row_splits_dtype)
        specs = [flat_values_spec, tensor_lib.TensorSpec(outer_splits_shape, self._row_splits_dtype)] + [inner_splits_spec for _ in range(self._ragged_rank - 1)]
        return specs

    def _to_components(self, value):
        if False:
            return 10
        if is_ragged(value):
            return [value.flat_values] + list(value.nested_row_splits)
        else:
            return [value]

    def _from_components(self, tensor_list):
        if False:
            print('Hello World!')
        result = tensor_list[0]
        if all((isinstance(t, np.ndarray) for t in tensor_list)) and (not tf2.enabled()):
            for row_splits in reversed(tensor_list[1:]):
                result = ragged_tensor_value.RaggedTensorValue(result, row_splits)
        else:
            if isinstance(tensor_list[0], np.ndarray):
                tensor_list = [ops.convert_to_tensor(t) for t in tensor_list]
                result = tensor_list[0]
            for row_splits in reversed(tensor_list[1:]):
                result = RaggedTensor(result, RowPartition.from_row_splits(row_splits, validate=False), internal=True)
        if self._shape.ndims is not None:
            if isinstance(result, RaggedTensor):
                result._set_shape(self._shape)
                if self.flat_values_spec is not None and hasattr(result.flat_values, 'set_shape'):
                    result.flat_values.set_shape(self.flat_values_spec.shape)
            elif isinstance(result, tensor_lib.Tensor):
                result.set_shape(self._shape)
        return result

    @property
    def _flat_tensor_specs(self):
        if False:
            while True:
                i = 10
        return [tensor_lib.TensorSpec(None, dtypes.variant)]

    def _to_tensor_list(self, value):
        if False:
            print('Hello World!')
        if self._flat_values_spec is not None:
            raise ValueError('Customized value_type is not supported.')
        if isinstance(value, RaggedTensor):
            if value.ragged_rank != self._ragged_rank:
                raise ValueError(f'Ragged rank of value {value.ragged_rank} does not match ragged rank of type {self._ragged_rank}.')
            return [value._to_variant(batched_input=False)]
        else:
            if self._ragged_rank > 0:
                raise ValueError(f'Expected a RaggedTensor if ragged rank={self._ragged_rank} but got {type(value).__name__}.')
            return [gen_ragged_conversion_ops.ragged_tensor_to_variant((), value, batched_input=False)]

    def _to_batched_tensor_list(self, value):
        if False:
            return 10
        if self._flat_values_spec is not None:
            raise ValueError('Customized value_type is not supported.')
        if isinstance(value, RaggedTensor):
            if value.ragged_rank != self._ragged_rank:
                raise ValueError(f'Ragged rank of value {value.ragged_rank} does not match ragged rank of type {self._ragged_rank}.')
            return [value._to_variant(batched_input=True)]
        else:
            if self._ragged_rank > 0:
                raise ValueError(f'Expected a RaggedTensor if ragged rank={self._ragged_rank} but got {type(value).__name__}.')
            return [gen_ragged_conversion_ops.ragged_tensor_to_variant(rt_nested_splits=(), rt_dense_values=value, batched_input=True)]

    def _from_compatible_tensor_list(self, tensor_list):
        if False:
            i = 10
            return i + 15
        if self._flat_values_spec is not None:
            raise ValueError('Customized value_type is not supported.')
        result = RaggedTensor._from_variant(tensor_list[0], dtype=self._dtype, row_splits_dtype=self._row_splits_dtype, output_ragged_rank=self._ragged_rank)
        if self._shape.ndims is not None:
            if isinstance(result, RaggedTensor):
                result._set_shape(self._shape)
                if self.flat_values_spec is not None and hasattr(self.flat_values, 'set_shape'):
                    result.flat_values.set_shape(self.flat_values_spec.shape)
            else:
                result.set_shape(self._shape)
        return result

    def _batch(self, batch_size):
        if False:
            i = 10
            return i + 15
        if self._flat_values_spec is not None:
            raise ValueError('Customized value_type is not supported.')
        return RaggedTensorSpec(tensor_shape.TensorShape([batch_size]).concatenate(self._shape), self._dtype, self._ragged_rank + 1, self._row_splits_dtype)

    def _unbatch(self):
        if False:
            while True:
                i = 10
        if self._flat_values_spec is not None:
            raise ValueError('Customized value_type is not supported.')
        return RaggedTensorSpec(self._shape[1:], self._dtype, self._ragged_rank - 1, self._row_splits_dtype)

    def _to_legacy_output_types(self):
        if False:
            return 10
        return self._dtype

    def _to_legacy_output_shapes(self):
        if False:
            for i in range(10):
                print('nop')
        return self._shape

    def _to_legacy_output_classes(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    @classmethod
    def from_value(cls, value):
        if False:
            while True:
                i = 10
        if isinstance(value, ragged_tensor_value.RaggedTensorValue) or isinstance(value.flat_values, tensor_lib.Tensor):
            return cls(shape=value.shape, dtype=value.values.dtype, ragged_rank=value.ragged_rank, row_splits_dtype=value.row_splits.dtype)
        else:
            flat_values_spec = type_spec.type_spec_from_value(value.flat_values)
            flat_values_spec = flat_values_spec._unbatch()._batch(None)
            return cls(shape=value.shape, dtype=value.values.dtype, ragged_rank=value.ragged_rank, row_splits_dtype=value.row_splits.dtype, flat_values_spec=flat_values_spec)
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(RaggedTensorSpec, struct_pb2.TypeSpecProto.RAGGED_TENSOR_SPEC))
type_spec.register_type_spec_from_value_converter(ragged_tensor_value.RaggedTensorValue, RaggedTensorSpec.from_value)

def convert_to_tensor_or_ragged_tensor(value, dtype=None, preferred_dtype=None, name=None):
    if False:
        print('Hello World!')
    'Converts value to a `RaggedTensor` or `Tensor`.\n\n  * If `value` is a `RaggedTensor`, then return it as-is.\n  * If `value` is a `RaggedTensorValue`, return a corresponding constant\n    `RaggedTensor`.\n  * Otherwise, use `convert_to_tensor` to convert `value` to a `Tensor`.\n\n  Args:\n    value: A `RaggedTensor`, a `RaggedTensorValue`, or an object whose type has\n      a registered `Tensor` conversion function.\n    dtype: Optional element type for the returned tensor.  If missing the type\n      is inferred from the type of `value`.\n    preferred_dtype: Optional element type for the returned tensor, used when\n      dtype is None.  This argument has no effect if `value` is already a\n      tensor, or when conversion is not possible.\n    name: Optional name to use if a new `Tensor` is created.\n\n  Returns:\n    A `Tensor` or `RaggedTensor`.\n  '
    if isinstance(value, RaggedTensor):
        if dtype and (not dtype.is_compatible_with(value.dtype)):
            raise ValueError(f'Tensor conversion requested dtype {dtype.name} for RaggedTensor with dtype {value.dtype.name}: {value}.')
        return value
    elif isinstance(value, ragged_tensor_value.RaggedTensorValue):
        with ops.name_scope(name, 'ConvertToTensorOrRaggedTensor', []):
            flat_values = ops.convert_to_tensor(value=value.flat_values, dtype=dtype, dtype_hint=preferred_dtype, name='flat_values')
            return RaggedTensor.from_nested_row_splits(flat_values, value.nested_row_splits, validate=False)
    else:
        return tensor_conversion.convert_to_tensor_v2_with_dispatch(value=value, dtype=dtype, dtype_hint=preferred_dtype, name=name)

def _convert_to_ragged_tensor_values(value):
    if False:
        i = 10
        return i + 15
    'Converts value to supported RaggedTensor value.\n\n  * If `value` is an object of supported value type, then return it as-is.\n  * Otherwise convert it to Tensor or RaggedTensor.\n\n  Args:\n    value: An object of `Tensor`, `RaggedTensor` or registerred RaggedTensor\n      value types, or an object whose type has a registered `Tensor` conversion\n      function.\n\n  Returns:\n    An object of `Tensor`, `RaggedTensor` or registerred RaggedTensor\n    value types\n  '
    if _is_supported_ragged_values_type(value):
        return value
    else:
        return convert_to_tensor_or_ragged_tensor(value, name='values')

def _ragged_tensor_value_from_components(components):
    if False:
        i = 10
        return i + 15
    components = list(components)
    value = components.pop()
    while components:
        value = ragged_tensor_value.RaggedTensorValue(value, components.pop())
    return value

def _ragged_tensor_session_fetch(rt):
    if False:
        for i in range(10):
            print('nop')
    components = rt.nested_row_splits + (rt.flat_values,)
    return (components, _ragged_tensor_value_from_components)

def _ragged_tensor_session_feed(feed_key, feed_val):
    if False:
        for i in range(10):
            print('nop')
    key_components = feed_key.nested_row_splits + (feed_key.flat_values,)
    val_components = feed_val.nested_row_splits + (feed_val.flat_values,)
    return zip(key_components, val_components)

def _ragged_tensor_session_feed_for_partial_run(feed_key):
    if False:
        for i in range(10):
            print('nop')
    return feed_key.nested_row_splits + (feed_key.flat_values,)
session.register_session_run_conversion_functions(RaggedTensor, _ragged_tensor_session_fetch, _ragged_tensor_session_feed, _ragged_tensor_session_feed_for_partial_run)

class RaggedTensorType:
    """Encoding of a static type for a `RaggedTensor`.

  Use this type to express/declare that an output must have the type of
  `RaggedTensor`.
  """

    def __init__(self, dtype, ragged_rank, row_splits_dtype=dtypes.int64):
        if False:
            return 10
        "Initializes a RaggedTensorType object.\n\n    Args:\n      dtype: data type of the `RaggedTensor`'s inner values.\n      ragged_rank: ragged_rank of the declared `RaggedTensor`.\n      row_splits_dtype: data type for the `RaggedTensor`'s row splits.\n        One of: `tf.int32` or `tf.int64`.\n    "
        row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
        self._dtype = dtype
        self._ragged_rank = ragged_rank
        self._row_splits_dtype = row_splits_dtype
    dtype = property(lambda self: self._dtype)
    ragged_rank = property(lambda self: self._ragged_rank)
    row_splits_dtype = property(lambda self: self._row_splits_dtype)

    def __repr__(self):
        if False:
            return 10
        return 'RaggedTensorType(%r, %r, %r)' % (self.dtype, self.ragged_rank, self.row_splits_dtype)

def _assert_sparse_indices_are_ragged_right(indices):
    if False:
        i = 10
        return i + 15
    'Checks that the given SparseTensor.indices tensor is ragged-right.\n\n  Example: `indices = [[0, 0], [0, 1], [2, 0], [3, 1]]` is not ragged right\n  because the entry `[3, 1]` skips a cell.\n\n  Args:\n    indices: The SparseTensor indices to check.\n\n  Returns:\n    A list of control dependency op tensors.\n  '
    index_prefix = indices[:, :-1]
    index_suffix = indices[:, -1]
    index_prefix_changed = math_ops.reduce_any(math_ops.not_equal(index_prefix[1:], index_prefix[:-1]), axis=1)
    index_ok = array_ops.where(index_prefix_changed, math_ops.equal(index_suffix[1:], 0), math_ops.equal(index_suffix[1:], index_suffix[:-1] + 1))
    sparse_indices_are_ragged_right = math_ops.logical_and(math_ops.reduce_all(math_ops.equal(index_suffix[:1], 0)), math_ops.reduce_all(index_ok))
    message = ['SparseTensor is not right-ragged', 'SparseTensor.indices =', indices]
    return [control_flow_assert.Assert(sparse_indices_are_ragged_right, message)]

@ops.RegisterGradient('RaggedTensorToSparse')
def _ragged_tensor_to_sparse_gradient(op, unused_sparse_indices_grad, sparse_values_grad, unused_sparse_shape_grad):
    if False:
        while True:
            i = 10
    'Gradient for RaggedTensorToSparse.'
    op_inputs_nested_row_splits = op.inputs[:-1]
    op_inputs_flat_values = op.inputs[-1]
    nested_row_splits_gradient = [None] * len(op_inputs_nested_row_splits)
    flat_values_shape = array_ops.shape(op_inputs_flat_values)
    flat_values_gradient = array_ops.reshape(sparse_values_grad, flat_values_shape)
    return nested_row_splits_gradient + [flat_values_gradient]

def _assert_monotonic_increasing(tensor, message=None):
    if False:
        for i in range(10):
            print('nop')
    return check_ops.assert_non_negative(tensor[1:] - tensor[:-1], message=message)

def _assert_zero(tensor, message=None):
    if False:
        i = 10
        return i + 15
    return check_ops.assert_equal(tensor, constant_op.constant(0, dtype=tensor.dtype), message=message)

def _nrows(tensor, out_type=dtypes.int32):
    if False:
        print('Hello World!')
    if isinstance(tensor, RaggedTensor):
        return tensor.nrows(out_type=out_type)
    else:
        return array_ops.shape(tensor, out_type=out_type)[0]

def merge_dims(value, outer_axis, inner_axis):
    if False:
        for i in range(10):
            print('nop')
    'Merges value[outer_axis...inner_axis] into a single dimension.\n\n  See `RaggedTensor.merge_dims()` for more details.  This helper differs from\n  `RaggedTensor.merge_dims()` in that `value` may be a dense or ragged tensor.\n\n  Args:\n    value: A `RaggedTensor` or `Tensor`\n    outer_axis: `int`\n    inner_axis: `int`\n\n  Returns:\n    A flattened `RaggedTensor` or `Tensor`.\n  '
    if outer_axis == inner_axis:
        return value
    while outer_axis == 0 and isinstance(value, RaggedTensor):
        value = value.values
        inner_axis -= 1
        if inner_axis == 0:
            return value
    if not isinstance(value, RaggedTensor):
        if value.shape.is_fully_defined():
            old_shape = value.shape.as_list()
            new_shape = old_shape[:outer_axis] + [-1] + old_shape[inner_axis + 1:]
        else:
            old_shape = array_ops.shape(value)
            new_shape = array_ops.concat([old_shape[:outer_axis], [-1], old_shape[inner_axis + 1:]], axis=0)
        return array_ops.reshape(value, new_shape)
    if outer_axis > 1:
        return value.with_values(merge_dims(value.values, outer_axis - 1, inner_axis - 1))
    new_values = value.values
    new_splits = value.row_splits
    for axis in range(outer_axis, inner_axis):
        if isinstance(new_values, RaggedTensor):
            new_splits = array_ops.gather(new_values.row_splits, new_splits)
            new_values = new_values.values
        else:
            shape_split = inner_axis - axis + 1
            if new_values.shape.is_fully_defined():
                old_shape = new_values.shape.as_list()
                new_shape = [-1] + old_shape[shape_split:]
                flat_size = _prod(old_shape[1:shape_split])
            else:
                old_shape = array_ops.shape(new_values)
                new_shape = array_ops.concat([[-1], old_shape[shape_split:]], axis=0)
                flat_size = math_ops.cast(math_ops.reduce_prod(old_shape[1:shape_split]), new_splits.dtype)
            new_values = array_ops.reshape(new_values, new_shape)
            new_splits = new_splits * flat_size
            break
    return RaggedTensor.from_row_splits(new_values, new_splits)

def _prod(lst):
    if False:
        while True:
            i = 10
    'Returns the product of the numbers in a list.'
    return functools.reduce(operator.mul, lst, 1)

def _get_row_partition_type_tensor_pairs_tail(partition):
    if False:
        for i in range(10):
            print('nop')
    'Gets a row partition type tensor pair for the tail.\n\n  If value_rowid is defined, then it is used. Otherwise, row_splits\n  are used.\n\n  Args:\n    partition: a RowPartition.\n\n  Returns:\n    A list of (row_partition_type, row_partition_tensor) pairs.\n  '
    if partition._has_precomputed_value_rowids():
        return ('VALUE_ROWIDS', partition.value_rowids())
    else:
        return ('ROW_SPLITS', partition.row_splits())

def _get_row_partition_type_tensor_pairs(rt_input):
    if False:
        for i in range(10):
            print('nop')
    'Gets a list of the row partitions for rt_input.\n\n  If value_rowids are defined, then they are used. Otherwise, row_splits\n  are used. If the outermost level has value_rowids defind, then nrows is\n  also added.\n\n  Args:\n    rt_input: a ragged tensor.\n\n  Returns:\n    A list of (row_partition_type, row_partition_tensor) pairs.\n  '
    partitions = rt_input._nested_row_partitions
    tail = [_get_row_partition_type_tensor_pairs_tail(x) for x in partitions[1:]]
    if partitions[0]._value_rowids is not None:
        return [('FIRST_DIM_SIZE', partitions[0].nrows()), ('VALUE_ROWIDS', partitions[0].value_rowids())] + tail
    else:
        return [('ROW_SPLITS', partitions[0].row_splits())] + tail

def _shape_as_tensor(shape, dtype):
    if False:
        print('Hello World!')
    'Takes shape and coerces it to a shape as a tensor.\n\n  If the object is already a tensor, simply passes it on (result is guaranteed\n  to be int64 or int32, but not necessarily dtype).\n  If not, creates a tensor of type dtype.\n\n  Result is either a scalar equal to -1 if the shape is unknown_rank.\n  Otherwise, it is a vector, where unknown dimensions are represented with a\n  value of -1.\n\n  In C++, see TensorShapeFromTensor for parsing shapes in kernels, and\n  InferenceContext::MakeShapeFromShapeTensorTreatScalarAsUnknownShape, for\n  use in the shape inference function.\n\n  Args:\n    shape: input to coerce from TensorShape, Tensor, None, List[Optional[Int]],\n      Tuple[Optional[Int]].\n    dtype: tf.int64 or tf.int32\n\n  Returns:\n    a scalar or vector tensor of dtype tf.int32 or tf.int64.\n  '
    if dtype != dtypes.int64 and dtype != dtypes.int32:
        raise ValueError(f'Expected int64 or int32 for dtype: got {dtype}.')
    if isinstance(shape, tensor_lib.Tensor):
        if shape.dtype != dtypes.int64 and shape.dtype != dtypes.int32:
            return math_ops.cast(shape, dtype)
        return shape
    shape = tensor_shape.as_shape(shape)
    if not shape:
        return constant_op.constant(-1, dtype=dtype)
    shape = [-1 if x is None else x for x in shape.as_list()]
    return constant_op.constant(shape, dtype=dtype)

def _nvals_uniform_row_length(values, uniform_row_length):
    if False:
        while True:
            i = 10
    'Get the number of values for uniform row length constructor.'
    const_nvals = tensor_shape.dimension_at_index(values.shape, 0).value
    if const_nvals is not None:
        nvals = constant_op.constant(const_nvals, uniform_row_length.dtype)
    elif isinstance(values, RaggedTensor):
        nvals = values.nrows(out_type=uniform_row_length.dtype)
    else:
        nvals = array_ops.shape(values, out_type=uniform_row_length.dtype)[0]
    return nvals

def _get_optional_partition_dtype(values):
    if False:
        for i in range(10):
            print('nop')
    'Returns the partition dtype, or None if None exists.'
    if isinstance(values, RaggedTensor):
        return values._row_partition.dtype
    return None
_SUPPORTED_RAGGED_VALUE_TYPES = (tensor_lib.Tensor, RaggedTensor)

def _add_supported_value_type(cls):
    if False:
        for i in range(10):
            print('nop')
    'Register the `cls` as supported value type of RaggedTenosr.\n\n  The cls must be a subclass of CompositeTensor, and must support:\n   - Spec:\n     The Spec must be a `BatchableTypeSpec`\n   - Properties:\n     - x.shape\n     - x.dtype\n   - Methods:\n     - x.__getitem__(idx) (method: returns a supported value type)\n     - x.set_shape(shape)\n   - Ops:\n     - tf.shape(x) -- tf.shape(x)[0] must be a tf.Tensor.\n     - tf.tile(x)\n     - assert_rank_at_least(x)\n     - tf.ones_like(x)\n     - tf.gather(params=x, indices=Tensor)\n     - tf.add(x, y)\n     - tf.boolean_mask(x, ...)\n     - @TODO(edloper): Complete this list\n\n   Note: the following RaggedTensor, RaggedTensorSpec methods & ops are not\n   currently supported unless `rt.values` is a RaggedTensor or a tf.Tensor:\n     - rt.to_tensor()\n     - rt.to_sparse_tensor()\n     - rt._to_variant()\n     - rt._from_variant()\n     - tf.ragged.cross([rt])\n     - tf.gather(params=x, indices=rt)  # rt used for indices\n     - RaggedTensorSpec methods:\n       - _batch\n       - _unbatch\n       - _to_tensor_list\n       - _to_batched_tensor_list\n       - _from_compatible_tensor_list\n\n  Args:\n    cls: The type to be added to supported value types.\n  '
    if not issubclass(cls, composite_tensor.CompositeTensor):
        raise ValueError(f'cls ({cls}) must be a subclass of CompositeTensor.')
    if not hasattr(cls, 'shape'):
        raise ValueError('cls must support the `shape` property.')
    if not hasattr(cls, 'dtype'):
        raise ValueError('cls must support the `dtype` property.')
    global _SUPPORTED_RAGGED_VALUE_TYPES
    _SUPPORTED_RAGGED_VALUE_TYPES += (cls,)

def _is_supported_ragged_values_type(value):
    if False:
        return 10
    return isinstance(value, _SUPPORTED_RAGGED_VALUE_TYPES)

def _assert_is_supported_ragged_values_type(value):
    if False:
        return 10
    if not _is_supported_ragged_values_type(value):
        ok_types = ', '.join((cls.__name__ for cls in _SUPPORTED_RAGGED_VALUE_TYPES))
        raise TypeError(f'type(values) must be one of: {ok_types}, got {value}.')

def _formatter(x):
    if False:
        print('Hello World!')
    'Separate Numpy array elements with comma.'
    if isinstance(x, np.ndarray):
        if x.size != 0:
            return np.array2string(x, separator=', ')
        else:
            return repr(x.tolist())
    else:
        return str(x)
Ragged = typing.Union[RaggedTensor, ragged_tensor_value.RaggedTensorValue]
RaggedOrDense = typing.Union[Ragged, core_types.TensorLike]
from tensorflow.python.ops.ragged import ragged_ops