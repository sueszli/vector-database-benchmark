"""A class used to partition a sequence into contiguous subsequences ("rows").
"""
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export

@tf_export('experimental.RowPartition')
class RowPartition(composite_tensor.CompositeTensor):
    """Partitioning of a sequence of values into contiguous subsequences ("rows").

  A `RowPartition` describes how a sequence with `nvals` items should be
  divided into `nrows` contiguous subsequences ("rows").  For example, a
  `RowPartition` could be used to partition the vector `[1, 2, 3, 4, 5]` into
  subsequences `[[1, 2], [3], [], [4, 5]]`.  Note that `RowPartition` stores
  information about how values are partitioned, but does not include the
  partitioned values themselves.  `tf.RaggedTensor` is used to pair a `values`
  tensor with one or more `RowPartition`s, providing a complete encoding for a
  ragged tensor (i.e. a tensor with variable-length dimensions).

  `RowPartition`s may be defined using several different schemes:

    * `row_lengths`: an integer vector with shape `[nrows]`, which specifies
      the length of each row.

    * `row_splits`: an integer vector with shape `[nrows+1]`, specifying the
      "split points" between each row.

    * `row_starts`: an integer vector with shape `[nrows]`, which specifies
      the start offset for each row.  Equivalent to `row_splits[:-1]`.

    * `row_limits`: an integer vector with shape `[nrows]`, which specifies
      the stop offset for each row.  Equivalent to `row_splits[1:]`.

    * `value_rowids` is an integer vector with shape `[nvals]`, corresponding
      one-to-one with sequence values, which specifies the row that each value
      belongs to.  If the partition has empty trailing rows, then `nrows`
      must also be specified.

    * `uniform_row_length` is an integer scalar, specifying the length of every
      row.  This scheme may only be used if all rows have the same length.

  For example, the following `RowPartition`s all represent the partitioning of
  8 values into 5 sublists as follows: `[[*, *, *, *], [], [*, *, *], [*], []]`.

  >>> p1 = RowPartition.from_row_lengths([4, 0, 3, 1, 0])
  >>> p2 = RowPartition.from_row_splits([0, 4, 4, 7, 8, 8])
  >>> p3 = RowPartition.from_row_starts([0, 4, 4, 7, 8], nvals=8)
  >>> p4 = RowPartition.from_row_limits([4, 4, 7, 8, 8])
  >>> p5 = RowPartition.from_value_rowids([0, 0, 0, 0, 2, 2, 2, 3], nrows=5)

  For more information about each scheme, see the documentation for the
  its factory method.  For additional examples, see the documentation on
  `tf.RaggedTensor`.

  ### Precomputed Encodings

  `RowPartition` always stores at least one encoding of the partitioning, but
  it can be configured to cache additional encodings as well.  This can
  avoid unnecessary recomputation in eager mode.  (In graph mode, optimizations
  such as common subexpression elimination will typically prevent these
  unnecessary recomputations.)  To check which encodings are precomputed, use
  `RowPartition.has_precomputed_<encoding>`.  To cache an additional
  encoding, use `RowPartition.with_precomputed_<encoding>`.
  """

    def __init__(self, row_splits, row_lengths=None, value_rowids=None, nrows=None, uniform_row_length=None, nvals=None, internal=False):
        if False:
            i = 10
            return i + 15
        'Creates a `RowPartition` from the specified encoding tensor(s).\n\n    This constructor is private -- please use one of the following ops to\n    build `RowPartition`s:\n\n      * `RowPartition.from_row_lengths`\n      * `RowPartition.from_value_rowids`\n      * `RowPartition.from_row_splits`\n      * `RowPartition.from_row_starts`\n      * `RowPartition.from_row_limits`\n      * `RowPartition.from_uniform_row_length`\n\n    If row_splits is has a constant value, then all other arguments should\n    have a constant value.\n\n    Args:\n      row_splits: A 1-D integer tensor with shape `[nrows+1]`.\n      row_lengths: A 1-D integer tensor with shape `[nrows]`\n      value_rowids: A 1-D integer tensor with shape `[nvals]`.\n      nrows: A 1-D integer scalar tensor.\n      uniform_row_length: A scalar tensor.\n      nvals: A scalar tensor.\n      internal: Private key value, required to ensure that this private\n        constructor is *only* called from the factory methods.\n\n    Raises:\n      TypeError: If a row partitioning tensor has an inappropriate dtype.\n      TypeError: If exactly one row partitioning argument was not specified.\n      ValueError: If a row partitioning tensor has an inappropriate shape.\n      ValueError: If multiple partitioning arguments are specified.\n      ValueError: If nrows is specified but value_rowids is not None.\n    '
        if internal is not _row_partition_factory_key:
            raise ValueError('RowPartition constructor is private; please use one of the factory methods instead (e.g., RowPartition.from_row_lengths())')
        if not isinstance(row_splits, tensor_lib.Tensor):
            raise TypeError('Row-partitioning argument must be a Tensor, got %r' % row_splits)
        if row_splits.dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('Row-partitioning argument must be int32 or int64')
        row_splits.shape.assert_has_rank(1)
        row_splits.set_shape([None])
        self._row_splits = row_splits
        for tensor in [row_lengths, value_rowids, nrows, uniform_row_length, nvals]:
            if tensor is not None:
                if not isinstance(tensor, tensor_lib.Tensor):
                    raise TypeError('Cached value must be a Tensor or None.')
                elif tensor.dtype != row_splits.dtype:
                    raise ValueError(f'Inconsistent dtype for encoding tensors: {tensor} vs {row_splits}')
        self._row_lengths = row_lengths
        self._value_rowids = value_rowids
        self._nrows = nrows
        self._uniform_row_length = uniform_row_length
        self._nvals = nvals

    @classmethod
    def from_value_rowids(cls, value_rowids, nrows=None, validate=True, dtype=None, dtype_hint=None):
        if False:
            for i in range(10):
                print('nop')
        "Creates a `RowPartition` with rows partitioned by `value_rowids`.\n\n    This `RowPartition` divides a sequence `values` into rows by specifying\n    which row each value should be added to:\n\n    ```python\n    partitioned_rows = [[] for _ in nrows]\n    for (value, rowid) in zip(values, value_rowids):\n      partitioned_rows[rowid].append(value)\n    ```\n\n    Args:\n      value_rowids: A 1-D integer tensor with shape `[nvals]`, which corresponds\n        one-to-one with `values`, and specifies each value's row index.  Must be\n        nonnegative, and must be sorted in ascending order.\n      nrows: An integer scalar specifying the number of rows.  This should be\n        specified if the `RowPartition` may containing empty training rows. Must\n        be greater than `value_rowids[-1]` (or greater than or equal to zero if\n        `value_rowids` is empty). Defaults to `value_rowids[-1] + 1` (or zero if\n        `value_rowids` is empty).\n      validate: If true, then use assertions to check that the arguments form a\n        valid `RowPartition`.\n      dtype: Optional dtype for the RowPartition. If missing, the type\n        is inferred from the type of `value_rowids`, dtype_hint, or tf.int64.\n      dtype_hint: Optional dtype for the RowPartition, used when dtype\n        is None. In some cases, a caller may not have a dtype in mind when\n        converting to a tensor, so dtype_hint can be used as a soft preference.\n        If the conversion to `dtype_hint` is not possible, this argument has no\n        effect.\n\n    Returns:\n      A `RowPartition`.\n\n    Raises:\n      ValueError: If `nrows` is incompatible with `value_rowids`.\n\n    #### Example:\n\n    >>> print(RowPartition.from_value_rowids(\n    ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],\n    ...     nrows=4))\n    tf.RowPartition(row_splits=[0 4 4 7 8])\n    "
        from tensorflow.python.ops import bincount_ops
        if not isinstance(validate, bool):
            raise TypeError('validate must have type bool')
        with ops.name_scope(None, 'RowPartitionFromValueRowIds', [value_rowids, nrows]):
            value_rowids = cls._convert_row_partition(value_rowids, 'value_rowids', dtype_hint=dtype_hint, dtype=dtype)
            if nrows is None:
                const_rowids = tensor_util.constant_value(value_rowids)
                if const_rowids is None:
                    nrows = array_ops.concat([value_rowids[-1:], [-1]], axis=0)[0] + 1
                    const_nrows = None
                else:
                    const_nrows = const_rowids[-1] + 1 if const_rowids.size > 0 else 0
                    nrows = ops.convert_to_tensor(const_nrows, value_rowids.dtype, name='nrows')
            else:
                nrows = ops.convert_to_tensor(nrows, value_rowids.dtype, 'nrows')
                const_nrows = tensor_util.constant_value(nrows)
                if const_nrows is not None:
                    if const_nrows < 0:
                        raise ValueError('Expected nrows >= 0; got %d' % const_nrows)
                    const_rowids = tensor_util.constant_value(value_rowids)
                    if const_rowids is not None and const_rowids.size > 0:
                        if not const_nrows >= const_rowids[-1] + 1:
                            raise ValueError('Expected nrows >= value_rowids[-1] + 1; got nrows=%d, value_rowids[-1]=%d' % (const_nrows, const_rowids[-1]))
            value_rowids.shape.assert_has_rank(1)
            nrows.shape.assert_has_rank(0)
            if validate:
                msg = 'Arguments to from_value_rowids do not form a valid RowPartition'
                checks = [check_ops.assert_rank(value_rowids, 1, message=msg), check_ops.assert_rank(nrows, 0, message=msg), check_ops.assert_non_negative(value_rowids[:1], message=msg), _assert_monotonic_increasing(value_rowids, message=msg), check_ops.assert_less(value_rowids[-1:], nrows, message=msg)]
                value_rowids = control_flow_ops.with_dependencies(checks, value_rowids)
            value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32)
            nrows_int32 = math_ops.cast(nrows, dtypes.int32)
            row_lengths = bincount_ops.bincount(value_rowids_int32, minlength=nrows_int32, maxlength=nrows_int32, dtype=value_rowids.dtype)
            row_splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
            if const_nrows is not None:
                row_lengths.set_shape([const_nrows])
                row_splits.set_shape([const_nrows + 1])
            return cls(row_splits=row_splits, row_lengths=row_lengths, value_rowids=value_rowids, nrows=nrows, internal=_row_partition_factory_key)

    @classmethod
    def from_row_splits(cls, row_splits, validate=True, dtype=None, dtype_hint=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a `RowPartition` with rows partitioned by `row_splits`.\n\n    This `RowPartition` divides a sequence `values` into rows by indicating\n    where each row begins and ends:\n\n    ```python\n    partitioned_rows = []\n    for i in range(len(row_splits) - 1):\n      row_start = row_splits[i]\n      row_end = row_splits[i + 1]\n      partitioned_rows.append(values[row_start:row_end])\n    ```\n\n    Args:\n      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be\n        empty, and must be sorted in ascending order.  `row_splits[0]` must be\n        zero.\n      validate: If true, then use assertions to check that the arguments form a\n        valid `RowPartition`.\n      dtype: Optional dtype for the RowPartition. If missing, the type\n        is inferred from the type of `row_splits`, dtype_hint, or tf.int64.\n      dtype_hint: Optional dtype for the RowPartition, used when dtype\n        is None. In some cases, a caller may not have a dtype in mind when\n        converting to a tensor, so dtype_hint can be used as a soft preference.\n        If the conversion to `dtype_hint` is not possible, this argument has no\n        effect.\n\n    Returns:\n      A `RowPartition`.\n\n    Raises:\n      ValueError: If `row_splits` is an empty list.\n    '
        if not isinstance(validate, bool):
            raise TypeError('validate must have type bool')
        if isinstance(row_splits, (list, tuple)) and (not row_splits):
            raise ValueError('row_splits tensor may not be empty.')
        if isinstance(row_splits, tensor_lib.TensorSpec):
            return cls(row_splits=row_splits, internal=_row_partition_factory_key)
        with ops.name_scope(None, 'RowPartitionFromRowSplits', [row_splits]):
            row_splits = cls._convert_row_partition(row_splits, 'row_splits', dtype_hint=dtype_hint, dtype=dtype)
            row_splits.shape.assert_has_rank(1)
            if validate:
                msg = 'Arguments to from_row_splits do not form a valid RaggedTensor:'
                checks = [check_ops.assert_rank(row_splits, 1, message=msg + 'rank'), _assert_zero(row_splits[0], message=msg + 'zero'), _assert_monotonic_increasing(row_splits, message=msg + 'monotonic')]
                row_splits = control_flow_ops.with_dependencies(checks, row_splits)
            return cls(row_splits=row_splits, internal=_row_partition_factory_key)

    @classmethod
    def from_row_lengths(cls, row_lengths, validate=True, dtype=None, dtype_hint=None):
        if False:
            while True:
                i = 10
        'Creates a `RowPartition` with rows partitioned by `row_lengths`.\n\n    This `RowPartition` divides a sequence `values` into rows by indicating\n    the length of each row:\n\n    ```python\n    partitioned_rows = [[values.pop(0) for _ in range(length)]\n                        for length in row_lengths]\n    ```\n\n    Args:\n      row_lengths: A 1-D integer tensor with shape `[nrows]`.  Must be\n        nonnegative.\n      validate: If true, then use assertions to check that the arguments form a\n        valid `RowPartition`.\n\n      dtype: Optional dtype for the RowPartition. If missing, the type\n        is inferred from the type of `row_lengths`, dtype_hint, or tf.int64.\n      dtype_hint: Optional dtype for the RowPartition, used when dtype\n        is None. In some cases, a caller may not have a dtype in mind when\n        converting to a tensor, so dtype_hint can be used as a soft preference.\n        If the conversion to `dtype_hint` is not possible, this argument has no\n        effect.\n\n    Returns:\n      A `RowPartition`.\n    '
        if not isinstance(validate, bool):
            raise TypeError('validate must have type bool')
        with ops.name_scope(None, 'RowPartitionFromRowLengths', [row_lengths]):
            row_lengths = cls._convert_row_partition(row_lengths, 'row_lengths', dtype_hint=dtype_hint, dtype=dtype)
            row_lengths.shape.assert_has_rank(1)
            if validate:
                msg = 'Arguments to from_row_lengths do not form a valid RowPartition'
                checks = [check_ops.assert_rank(row_lengths, 1, message=msg), check_ops.assert_non_negative(row_lengths, message=msg)]
                row_lengths = control_flow_ops.with_dependencies(checks, row_lengths)
            row_limits = math_ops.cumsum(row_lengths)
            row_splits = array_ops.concat([[0], row_limits], axis=0)
            return cls(row_splits=row_splits, row_lengths=row_lengths, internal=_row_partition_factory_key)

    @classmethod
    def from_row_starts(cls, row_starts, nvals, validate=True, dtype=None, dtype_hint=None):
        if False:
            return 10
        'Creates a `RowPartition` with rows partitioned by `row_starts`.\n\n    Equivalent to: `from_row_splits(concat([row_starts, nvals], axis=0))`.\n\n    Args:\n      row_starts: A 1-D integer tensor with shape `[nrows]`.  Must be\n        nonnegative and sorted in ascending order.  If `nrows>0`, then\n        `row_starts[0]` must be zero.\n      nvals: A scalar tensor indicating the number of values.\n      validate: If true, then use assertions to check that the arguments form a\n        valid `RowPartition`.\n      dtype: Optional dtype for the RowPartition. If missing, the type\n        is inferred from the type of `row_starts`, dtype_hint, or tf.int64.\n      dtype_hint: Optional dtype for the RowPartition, used when dtype\n        is None. In some cases, a caller may not have a dtype in mind when\n        converting to a tensor, so dtype_hint can be used as a soft preference.\n        If the conversion to `dtype_hint` is not possible, this argument has no\n        effect.\n\n    Returns:\n      A `RowPartition`.\n    '
        if not isinstance(validate, bool):
            raise TypeError('validate must have type bool')
        with ops.name_scope(None, 'RowPartitionFromRowStarts', [row_starts]):
            row_starts = cls._convert_row_partition(row_starts, 'row_starts', dtype_hint=dtype_hint, dtype=dtype)
            row_starts.shape.assert_has_rank(1)
            nvals = math_ops.cast(nvals, row_starts.dtype)
            if validate:
                msg = 'Arguments to from_row_starts do not form a valid RaggedTensor'
                checks = [check_ops.assert_rank(row_starts, 1, message=msg), _assert_zero(row_starts[:1], message=msg), _assert_monotonic_increasing(row_starts, message=msg), check_ops.assert_less_equal(row_starts[-1:], nvals, message=msg)]
                row_starts = control_flow_ops.with_dependencies(checks, row_starts)
            row_splits = array_ops.concat([row_starts, [nvals]], axis=0)
            return cls(row_splits=row_splits, nvals=nvals, internal=_row_partition_factory_key)

    @classmethod
    def from_row_limits(cls, row_limits, validate=True, dtype=None, dtype_hint=None):
        if False:
            i = 10
            return i + 15
        'Creates a `RowPartition` with rows partitioned by `row_limits`.\n\n    Equivalent to: `from_row_splits(values, concat([0, row_limits], axis=0))`.\n\n    Args:\n      row_limits: A 1-D integer tensor with shape `[nrows]`.  Must be sorted in\n        ascending order.\n      validate: If true, then use assertions to check that the arguments form a\n        valid `RowPartition`.\n      dtype: Optional dtype for the RowPartition. If missing, the type\n        is inferred from the type of `row_limits`, dtype_hint, or tf.int64.\n      dtype_hint: Optional dtype for the RowPartition, used when dtype\n        is None. In some cases, a caller may not have a dtype in mind when\n        converting to a tensor, so dtype_hint can be used as a soft preference.\n        If the conversion to `dtype_hint` is not possible, this argument has no\n        effect.\n\n    Returns:\n      A `RowPartition`.\n    '
        if not isinstance(validate, bool):
            raise TypeError('validate must have type bool')
        with ops.name_scope(None, 'RowPartitionFromRowLimits', [row_limits]):
            row_limits = cls._convert_row_partition(row_limits, 'row_limits', dtype_hint=dtype_hint, dtype=dtype)
            row_limits.shape.assert_has_rank(1)
            if validate:
                msg = 'Arguments to from_row_limits do not form a valid RaggedTensor'
                checks = [check_ops.assert_rank(row_limits, 1, message=msg), check_ops.assert_non_negative(row_limits[:1], message=msg), _assert_monotonic_increasing(row_limits, message=msg)]
                row_limits = control_flow_ops.with_dependencies(checks, row_limits)
            zero = array_ops.zeros([1], row_limits.dtype)
            row_splits = array_ops.concat([zero, row_limits], axis=0)
            return cls(row_splits=row_splits, internal=_row_partition_factory_key)

    @classmethod
    def from_uniform_row_length(cls, uniform_row_length, nvals=None, nrows=None, validate=True, dtype=None, dtype_hint=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a `RowPartition` with rows partitioned by `uniform_row_length`.\n\n    This `RowPartition` divides a sequence `values` into rows that all have\n    the same length:\n\n    ```python\n    partitioned_rows = [[values.pop(0) for _ in range(uniform_row_length)]\n             for _ in range(nrows)]\n    ```\n\n    Note that either or both of nvals and nrows must be specified.\n\n    Args:\n      uniform_row_length: A scalar integer tensor.  Must be nonnegative. The\n        size of the outer axis of `values` must be evenly divisible by\n        `uniform_row_length`.\n      nvals: a non-negative scalar integer tensor for the number of values.\n        Must be specified if nrows is not specified. If not specified,\n        defaults to uniform_row_length*nrows\n      nrows: The number of rows in the constructed RowPartition.  If not\n        specified, then it defaults to `nvals/uniform_row_length` (or `0` if\n        `uniform_row_length==0`).  `nrows` only needs to be specified if\n        `uniform_row_length` might be zero.  `uniform_row_length*nrows` must be\n        `nvals`.\n      validate: If true, then use assertions to check that the arguments form a\n        valid `RowPartition`.\n      dtype: Optional dtype for the RowPartition. If missing, the type\n        is inferred from the type of `uniform_row_length`, dtype_hint,\n        or tf.int64.\n      dtype_hint: Optional dtype for the RowPartition, used when dtype\n        is None. In some cases, a caller may not have a dtype in mind when\n        converting to a tensor, so dtype_hint can be used as a soft preference.\n        If the conversion to `dtype_hint` is not possible, this argument has no\n        effect.\n\n    Returns:\n      A `RowPartition`.\n    '
        if not isinstance(validate, bool):
            raise TypeError('validate must have type bool')
        if nrows is None and nvals is None:
            raise ValueError('Either (or both) of nvals and nrows must be specified')
        with ops.name_scope(None, 'RowPartitionFromUniformRowLength', [uniform_row_length, nrows]):
            [uniform_row_length, nvals, nrows] = _convert_all_to_tensors([(uniform_row_length, 'uniform_row_length'), (nvals, 'nvals'), (nrows, 'nrows')], dtype=dtype, dtype_hint=dtype_hint)
            uniform_row_length.shape.assert_has_rank(0)
            const_row_length = tensor_util.constant_value(uniform_row_length)
            if nrows is None:
                if const_row_length is None:
                    rowlen_or_1 = math_ops.maximum(uniform_row_length, constant_op.constant(1, uniform_row_length.dtype))
                    nrows = nvals // rowlen_or_1
                elif const_row_length == 0:
                    nrows = constant_op.constant(0, dtype=uniform_row_length.dtype)
                else:
                    nrows = nvals // const_row_length
            const_nrows = None if nrows is None else tensor_util.constant_value(nrows)
            const_nvals = None if nvals is None else tensor_util.constant_value(nvals)
            const_uniform_row_length = tensor_util.constant_value(uniform_row_length)
            checks = []
            if const_nvals is None and const_nrows is not None and (const_uniform_row_length is not None):
                const_nvals = const_nrows * const_uniform_row_length
                if nvals is not None and validate:
                    checks.append(check_ops.assert_equal(nvals, const_nvals))
                nvals = constant_op.constant(const_nvals, uniform_row_length.dtype)
            if nvals is None:
                nvals = nrows * uniform_row_length
            if const_nrows is not None and const_row_length is not None:
                row_splits = [v * const_row_length for v in range(const_nrows + 1)]
                row_splits = constant_op.constant(row_splits, uniform_row_length.dtype)
            else:
                row_splits = math_ops.range(nrows + 1, dtype=uniform_row_length.dtype) * uniform_row_length
            if validate:
                if const_nrows is None or const_row_length is None or const_nvals is None:
                    checks.append(check_ops.assert_equal(nrows * uniform_row_length, nvals, ('uniform_row_length', uniform_row_length, 'times nrows', nrows, 'must equal nvals', nvals)))
                elif const_nrows * const_row_length != const_nvals:
                    raise ValueError('uniform_row_length=%d times nrows=%d must equal nvals=%d' % (const_row_length, const_nrows, const_nvals))
                if uniform_row_length.shape.rank is None:
                    checks.append(check_ops.assert_rank(uniform_row_length, 0, message='uniform_row_length must be a scalar.'))
                const_row_length = tensor_util.constant_value(uniform_row_length)
                if const_row_length is None:
                    checks.append(check_ops.assert_greater_equal(uniform_row_length, constant_op.constant(0, uniform_row_length.dtype), message='uniform_row_length must be >= 0.'))
                elif const_row_length < 0:
                    raise ValueError('uniform_row_length must be >= 0.')
                row_splits = control_flow_ops.with_dependencies(checks, row_splits)
            return cls(row_splits=row_splits, uniform_row_length=uniform_row_length, nrows=nrows, nvals=nvals, internal=_row_partition_factory_key)

    @classmethod
    def _convert_row_partition(cls, partition, name, dtype=None, dtype_hint=None):
        if False:
            for i in range(10):
                print('nop')
        'Converts `partition` to Tensors.\n\n    Args:\n      partition: A row-partitioning tensor for the `RowPartition` being\n        constructed.  I.e., one of: row_splits, row_lengths, row_starts,\n        row_limits, value_rowids, uniform_row_length.\n      name: The name of the row-partitioning tensor.\n      dtype: Optional dtype for the RowPartition. If missing, the type\n        is inferred from the type of `uniform_row_length`, dtype_hint,\n        or tf.int64.\n      dtype_hint: Optional dtype for the RowPartition, used when dtype\n        is None. In some cases, a caller may not have a dtype in mind when\n        converting to a tensor, so dtype_hint can be used as a soft preference.\n        If the conversion to `dtype_hint` is not possible, this argument has no\n        effect.\n\n    Returns:\n      A tensor equivalent to partition.\n\n    Raises:\n      ValueError: if dtype is not int32 or int64.\n    '
        if dtype_hint is None:
            dtype_hint = dtypes.int64
        if isinstance(partition, np.ndarray) and partition.dtype == np.int32 and (dtype is None):
            partition = ops.convert_to_tensor(partition, name=name)
        else:
            partition = tensor_conversion.convert_to_tensor_v2(partition, dtype_hint=dtype_hint, dtype=dtype, name=name)
        if partition.dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('%s must have dtype int32 or int64' % name)
        return partition

    def _with_dependencies(self, dependencies):
        if False:
            while True:
                i = 10
        'Returns a new RowPartition equal to self with control dependencies.\n\n    Specifically, self._row_splits is gated by the given control dependencies.\n    Used to add sanity checks to the constructors.\n\n    Args:\n      dependencies: a list of tensors to use as dependencies.\n\n    Returns:\n      A new RowPartition object.\n    '
        new_row_splits = control_flow_ops.with_dependencies(dependencies, self._row_splits)
        return RowPartition(row_splits=new_row_splits, row_lengths=self._row_lengths, value_rowids=self._value_rowids, nrows=self._nrows, uniform_row_length=self._uniform_row_length, internal=_row_partition_factory_key)

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        'The `DType` used to encode the row partition (either int32 or int64).'
        return self._row_splits.dtype

    def row_splits(self):
        if False:
            i = 10
            return i + 15
        'Returns the row-split indices for this row partition.\n\n    `row_splits` specifies where the values for each row begin and end.\n    In particular, the values for row `i` are stored in the slice\n    `values[row_splits[i]:row_splits[i+1]]`.\n\n    Returns:\n      A 1-D integer `Tensor` with shape `[self.nrows+1]`.\n      The returned tensor is non-empty, and is sorted in ascending order.\n      `self.row_splits()[0] == 0`.\n      `self.row_splits()[-1] == self.nvals()`.\n    '
        return self._row_splits

    def value_rowids(self):
        if False:
            print('Hello World!')
        'Returns the row indices for this row partition.\n\n    `value_rowids` specifies the row index fo reach value.  In particular,\n    `value_rowids[i]` is the row index for `values[i]`.\n\n    Returns:\n      A 1-D integer `Tensor` with shape `[self.nvals()]`.\n      The returned tensor is nonnegative, and is sorted in ascending order.\n    '
        if self._value_rowids is not None:
            return self._value_rowids
        return segment_id_ops.row_splits_to_segment_ids(self._row_splits)

    def nvals(self):
        if False:
            i = 10
            return i + 15
        "Returns the number of values partitioned by this `RowPartition`.\n\n    If the sequence partitioned by this `RowPartition` is a tensor, then\n    `nvals` is the size of that tensor's outermost dimension -- i.e.,\n    `nvals == values.shape[0]`.\n\n    Returns:\n      scalar integer Tensor\n    "
        return self._row_splits[-1]

    def nrows(self):
        if False:
            print('Hello World!')
        'Returns the number of rows created by this `RowPartition`.\n\n    Returns:\n      scalar integer Tensor\n    '
        if self._nrows is not None:
            return self._nrows
        nsplits = tensor_shape.dimension_at_index(self._row_splits.shape, 0)
        if nsplits.value is None:
            return array_ops.shape(self._row_splits, out_type=self.dtype)[0] - 1
        else:
            return constant_op.constant(nsplits.value - 1, dtype=self.dtype)

    def uniform_row_length(self):
        if False:
            while True:
                i = 10
        'Returns the length of each row in this partition, if rows are uniform.\n\n    If all rows in this `RowPartition` have the same length, then this returns\n    that length as a scalar integer `Tensor`.  Otherwise, it returns `None`.\n\n    Returns:\n      scalar Tensor with `type=self.dtype`, or `None`.\n    '
        return self._uniform_row_length

    def row_starts(self):
        if False:
            i = 10
            return i + 15
        'Returns the start indices for rows in this row partition.\n\n    These indices specify where the values for each row begin.\n    `partition.row_starts()` is equal to `partition.row_splits()[:-1]`.\n\n    Returns:\n      A 1-D integer Tensor with shape `[self.nrows()]`.\n      The returned tensor is nonnegative, and is sorted in ascending order.\n      `self.row_starts()[0] == 0`.\n      `self.row_starts()[-1] <= self.nvals()`.\n    '
        return self._row_splits[:-1]

    def row_limits(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the limit indices for rows in this row partition.\n\n    These indices specify where the values for each row end.\n    `partition.row_limits()` is equal to `partition.row_splits()[:-1]`.\n\n    Returns:\n      A 1-D integer Tensor with shape `[self.nrows]`.\n      The returned tensor is nonnegative, and is sorted in ascending order.\n      `self.row_limits()[-1] == self.nvals()`.\n    '
        return self._row_splits[1:]

    def row_lengths(self):
        if False:
            while True:
                i = 10
        'Returns the lengths of rows in this `RowPartition`.\n\n    Returns:\n      A 1-D integer Tensor with shape `[self.nrows]`.\n      The returned tensor is nonnegative.\n      `tf.reduce_sum(self.row_lengths) == self.nvals()`.\n    '
        if self._row_lengths is not None:
            return self._row_lengths
        splits = self._row_splits
        return splits[1:] - splits[:-1]

    @property
    def static_nrows(self):
        if False:
            for i in range(10):
                print('nop')
        'The number of rows in this partition, if statically known.\n\n    ```python\n    self.row_lengths().shape == [self.static_nrows]\n    self.row_starts().shape == [self.static_nrows]\n    self.row_limits().shape == [self.static_nrows]\n    self.row_splits().shape == [self.static_nrows + 1]\n    ```\n\n    Returns:\n      The number of rows in this partition as an `int` (if statically known);\n      or `None` (otherwise).\n    '
        if self._row_splits is not None:
            nrows_plus_one = tensor_shape.dimension_value(self._row_splits.shape[0])
            if nrows_plus_one is not None:
                return nrows_plus_one - 1
        if self._row_lengths is not None:
            nrows = tensor_shape.dimension_value(self._row_lengths.shape[0])
            if nrows is not None:
                return nrows
        if self._nrows is not None:
            return tensor_util.constant_value(self._nrows)
        return None

    @property
    def static_nvals(self):
        if False:
            return 10
        'The number of values in this partition, if statically known.\n\n    ```python\n    self.value_rowids().shape == [self.static_vals]\n    ```\n\n    Returns:\n      The number of values in this partition as an `int` (if statically known);\n      or `None` (otherwise).\n    '
        if self._nvals is not None:
            nvals = tensor_util.constant_value(self._nvals)
            if nvals is not None:
                return nvals
        if self._value_rowids is not None:
            nvals = tensor_shape.dimension_at_index(self._value_rowids.shape, 0)
            if nvals.value is not None:
                return nvals.value
        return None

    @property
    def static_uniform_row_length(self):
        if False:
            for i in range(10):
                print('nop')
        'The number of values in each row of this partition, if statically known.\n\n    Returns:\n      The number of values in each row of this partition as an `int` (if\n      statically known); or `None` (otherwise).\n    '
        if self._uniform_row_length is not None:
            return tensor_util.constant_value(self._uniform_row_length)
        return None

    def offsets_in_rows(self):
        if False:
            i = 10
            return i + 15
        'Return the offset of each value.\n\n    RowPartition takes an array x and converts it into sublists.\n    offsets[i] is the index of x[i] in its sublist.\n    Given a shape, such as:\n    [*,*,*],[*,*],[],[*,*]\n    This returns:\n    0,1,2,0,1,0,1\n\n    Returns:\n      an offset for every value.\n    '
        return gen_ragged_math_ops.ragged_range(starts=constant_op.constant(0, self.dtype), limits=self.row_lengths(), deltas=constant_op.constant(1, self.dtype)).rt_dense_values

    def is_uniform(self):
        if False:
            print('Hello World!')
        'Returns true if the partition is known to be uniform statically.\n\n    This is based upon the existence of self._uniform_row_length. For example:\n    RowPartition.from_row_lengths([3,3,3]).is_uniform()==false\n    RowPartition.from_uniform_row_length(5, nvals=20).is_uniform()==true\n    RowPartition.from_row_lengths([2,0,2]).is_uniform()==false\n\n    Returns:\n      Whether a RowPartition is known to be uniform statically.\n    '
        return self._uniform_row_length is not None

    def _static_check(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks if the object is internally consistent.\n\n    Raises:\n      ValueError if inconsistent.\n    '
        my_dtype = self.dtype
        if self._uniform_row_length is not None:
            if self._uniform_row_length.dtype != my_dtype:
                raise ValueError('_uniform_row_length.dtype=' + str(self._uniform_row_length.dtype) + ', not ' + str(my_dtype))
        if self._row_lengths is not None and self._row_lengths.dtype != my_dtype:
            raise ValueError('_row_lengths.dtype=' + str(self._row_lengths.dtype) + ', not ' + str(my_dtype))
        if self._value_rowids is not None and self._value_rowids.dtype != my_dtype:
            raise ValueError('_value_rowids.dtype=' + str(self._value_rowids.dtype) + ', not ' + str(my_dtype))
        if self._nrows is not None and self._nrows.dtype != my_dtype:
            raise ValueError('_nrows.dtype=' + str(self._nrows.dtype) + ', not ' + str(my_dtype))

    def with_dtype(self, dtype):
        if False:
            return 10
        'Returns a copy of this RowPartition with the given encoding dtype.\n\n    Args:\n      dtype: The dtype for encoding tensors, such as `row_splits` and `nrows`.\n      One of `tf.int32` or `tf.int64`.\n\n    Returns:\n      A copy of this RowPartition, with the encoding tensors cast to the given\n      type.\n    '
        dtype = dtypes.as_dtype(dtype)
        if dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('dtype must be int32 or int64')
        if self.dtype == dtype:
            return self
        return RowPartition(row_splits=_cast_if_not_none(self._row_splits, dtype), row_lengths=_cast_if_not_none(self._row_lengths, dtype), value_rowids=_cast_if_not_none(self._value_rowids, dtype), nrows=_cast_if_not_none(self._nrows, dtype), uniform_row_length=_cast_if_not_none(self._uniform_row_length, dtype), internal=_row_partition_factory_key)

    def __repr__(self):
        if False:
            print('Hello World!')
        if self._uniform_row_length is not None:
            return f'tf.RowPartition(nrows={self._nrows}, uniform_row_length={self._uniform_row_length})'
        else:
            return f'tf.RowPartition(row_splits={self._row_splits})'

    def _has_precomputed_row_splits(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns true if `row_splits` has already been computed.\n\n    If true, then `self.row_splits()` will return its value without calling\n    any TensorFlow ops.\n    '
        return self._row_splits is not None

    def _has_precomputed_row_lengths(self):
        if False:
            i = 10
            return i + 15
        'Returns true if `row_lengths` has already been computed.\n\n    If true, then `self.row_lengths()` will return its value without calling\n    any TensorFlow ops.\n    '
        return self._row_lengths is not None

    def _has_precomputed_value_rowids(self):
        if False:
            i = 10
            return i + 15
        'Returns true if `value_rowids` has already been computed.\n\n    If true, then `self.value_rowids()` will return its value without calling\n    any TensorFlow ops.\n    '
        return self._value_rowids is not None

    def _has_precomputed_nrows(self):
        if False:
            while True:
                i = 10
        'Returns true if `nrows` has already been computed.\n\n    If true, then `self.nrows()` will return its value without calling\n    any TensorFlow ops.\n    '
        return self._nrows is not None

    def _has_precomputed_nvals(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns true if `nvals` has already been computed.\n\n    If true, then `self.nvals()` will return its value without calling\n    any TensorFlow ops.\n    '
        return self._nvals is not None

    def _with_precomputed_row_splits(self):
        if False:
            return 10
        'Returns a copy of `self` with `row_splits` precomputed.'
        return RowPartition(row_splits=self.row_splits(), row_lengths=self._row_lengths, value_rowids=self._value_rowids, nrows=self._nrows, uniform_row_length=self._uniform_row_length, nvals=self._nvals, internal=_row_partition_factory_key)

    def _with_precomputed_row_lengths(self):
        if False:
            while True:
                i = 10
        'Returns a copy of `self` with `row_lengths` precomputed.'
        return RowPartition(row_splits=self._row_splits, row_lengths=self.row_lengths(), value_rowids=self._value_rowids, nrows=self._nrows, nvals=self._nvals, uniform_row_length=self._uniform_row_length, internal=_row_partition_factory_key)

    def _with_precomputed_value_rowids(self):
        if False:
            while True:
                i = 10
        'Returns a copy of `self` with `value_rowids` precomputed.'
        return RowPartition(row_splits=self._row_splits, row_lengths=self._row_lengths, value_rowids=self.value_rowids(), nrows=self._nrows, nvals=self._nvals, uniform_row_length=self._uniform_row_length, internal=_row_partition_factory_key)

    def _with_precomputed_nrows(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a copy of `self` with `nrows` precomputed.'
        return RowPartition(row_splits=self._row_splits, row_lengths=self._row_lengths, value_rowids=self._value_rowids, nrows=self.nrows(), nvals=self._nvals, uniform_row_length=self._uniform_row_length, internal=_row_partition_factory_key)

    def _with_precomputed_nvals(self):
        if False:
            i = 10
            return i + 15
        'Returns a copy of `self` with `row_splits` precomputed.'
        return RowPartition(row_splits=self.row_splits(), row_lengths=self._row_lengths, value_rowids=self._value_rowids, nrows=self._nrows, nvals=self.nvals(), uniform_row_length=self._uniform_row_length, internal=_row_partition_factory_key)

    def _merge_with_spec(self, b):
        if False:
            while True:
                i = 10
        'Merge with a TypeSpec to create a new RowPartition.'
        a_spec = self._type_spec
        if not a_spec.is_compatible_with(b):
            raise ValueError('RowPartition and RowPartitionSpec are not compatible')
        nrows = constant_op.constant(b.nrows, self.dtype) if b.nrows is not None else self._nrows
        nvals = constant_op.constant(b.nvals, self.dtype) if b.nvals is not None else self._nvals
        uniform_row_length = constant_op.constant(b.uniform_row_length, self.dtype) if b.uniform_row_length is not None else self._uniform_row_length
        return RowPartition(row_splits=self._row_splits, row_lengths=self._row_lengths, value_rowids=self._value_rowids, nvals=nvals, uniform_row_length=uniform_row_length, nrows=nrows, internal=_row_partition_factory_key)

    def _merge_precomputed_encodings(self, other, validate=True):
        if False:
            i = 10
            return i + 15
        'Returns a RowPartition that merges encodings from `self` and `other`.\n\n    Requires that `self` and `other` describe the same partition.\n\n    Args:\n      other: A `RowPartition` that encodes the same partition as `self`.\n      validate: If true, then add runtime checks to verify that `self` and\n        `other` encode the same row partition.\n\n    Returns:\n      A `RowPartition`.\n    '
        if self is other or (self._row_splits is other._row_splits and self._row_lengths is other._row_lengths and (self._value_rowids is other._value_rowids) and (self._nrows is other._nrows) and (self._nvals is other._nvals) and (self._uniform_row_length is other._uniform_row_length)):
            return self
        (nrows, nrows_validated) = _merge_tensors(self._nrows, other._nrows, 'nrows', validate)
        (nvals, _) = _merge_tensors(self._nvals, other._nvals, 'nvals', validate)
        (uniform_row_length, uniform_row_length_validated) = _merge_tensors(self._uniform_row_length, other._uniform_row_length, 'uniform_row_length', validate)
        if uniform_row_length_validated and nrows_validated:
            validate = False
        (row_splits, row_splits_validated) = _merge_tensors(self._row_splits, other._row_splits, 'row_splits', validate)
        if row_splits_validated:
            validate = False
        (row_lengths, row_lengths_validated) = _merge_tensors(self._row_lengths, other._row_lengths, 'row_lengths', validate)
        if row_lengths_validated:
            validate = False
        (value_rowids, value_rowids_validated) = _merge_tensors(self._value_rowids, other._value_rowids, 'value_rowids', validate)
        if value_rowids_validated and nrows_validated:
            validate = False
        if row_splits is self._row_splits and row_lengths is self._row_lengths and (value_rowids is self._value_rowids) and (nrows is self._nrows) and (uniform_row_length is self._uniform_row_length):
            return self
        if row_splits is other._row_splits and row_lengths is other._row_lengths and (value_rowids is other._value_rowids) and (nrows is other._nrows) and (uniform_row_length is other._uniform_row_length):
            return other
        return RowPartition(row_splits=row_splits, row_lengths=row_lengths, value_rowids=value_rowids, nrows=nrows, uniform_row_length=uniform_row_length, nvals=nvals, internal=_row_partition_factory_key)

    @property
    def _type_spec(self):
        if False:
            i = 10
            return i + 15
        return RowPartitionSpec.from_value(self)

@type_spec_registry.register('tf.RowPartitionSpec')
class RowPartitionSpec(type_spec.TypeSpec):
    """Type specification for a `tf.RowPartition`."""
    __slots__ = ['_nrows', '_nvals', '_uniform_row_length', '_dtype']
    value_type = property(lambda self: RowPartition)

    def __init__(self, nrows=None, nvals=None, uniform_row_length=None, dtype=dtypes.int64):
        if False:
            print('Hello World!')
        'Constructs a new RowPartitionSpec.\n\n    Args:\n      nrows: The number of rows in the RowPartition, or `None` if unspecified.\n      nvals: The number of values partitioned by the RowPartition, or `None` if\n        unspecified.\n      uniform_row_length: The number of values in each row for this\n        RowPartition, or `None` if rows are ragged or row length is unspecified.\n      dtype: The data type used to encode the partition.  One of `tf.int64` or\n        `tf.int32`.\n    '
        nrows = tensor_shape.TensorShape([nrows])
        nvals = tensor_shape.TensorShape([nvals])
        if not isinstance(uniform_row_length, tensor_shape.TensorShape):
            uniform_row_length = tensor_shape.TensorShape([uniform_row_length])
        else:
            uniform_row_length = uniform_row_length.with_rank(1)
        self._nrows = nrows
        self._nvals = nvals
        self._uniform_row_length = uniform_row_length
        self._dtype = dtypes.as_dtype(dtype)
        if self._dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('dtype must be tf.int32 or tf.int64')
        nrows = tensor_shape.dimension_value(nrows[0])
        nvals = tensor_shape.dimension_value(nvals[0])
        ncols = tensor_shape.dimension_value(uniform_row_length[0])
        if nrows == 0:
            if nvals is None:
                self._nvals = tensor_shape.TensorShape([0])
            elif nvals != 0:
                raise ValueError('nvals=%s is not compatible with nrows=%s' % (nvals, nrows))
        if ncols == 0:
            if nvals is None:
                self._nvals = tensor_shape.TensorShape([0])
            elif nvals != 0:
                raise ValueError('nvals=%s is not compatible with uniform_row_length=%s' % (nvals, uniform_row_length))
        if ncols is not None and nvals is not None:
            if ncols != 0 and nvals % ncols != 0:
                raise ValueError("nvals=%s is not compatible with uniform_row_length=%s (doesn't divide evenly)" % (nvals, ncols))
            if nrows is not None and nvals != ncols * nrows:
                raise ValueError('nvals=%s is not compatible with nrows=%s and uniform_row_length=%s' % (nvals, nrows, ncols))
            if nrows is None and ncols != 0:
                self._nrows = tensor_shape.TensorShape([nvals // ncols])
        if ncols is not None and nrows is not None and (nvals is None):
            self._nvals = tensor_shape.TensorShape([ncols * nrows])

    def is_compatible_with(self, other):
        if False:
            print('Hello World!')
        if not super(RowPartitionSpec, self).is_compatible_with(other):
            return False
        nrows = self._nrows.merge_with(other.nrows)
        nvals = self._nvals.merge_with(other.nvals)
        ncols = self._uniform_row_length.merge_with(other.uniform_row_length)
        return self._dimensions_compatible(nrows, nvals, ncols)

    def _serialize(self):
        if False:
            return 10
        return (self._nrows, self._nvals, self._uniform_row_length, self._dtype)

    @classmethod
    def _deserialize(cls, serialization):
        if False:
            print('Hello World!')
        (nrows, nvals, uniform_row_length, dtype) = serialization
        nrows = tensor_shape.dimension_value(nrows[0])
        nvals = tensor_shape.dimension_value(nvals[0])
        return cls(nrows, nvals, uniform_row_length, dtype)

    @property
    def nrows(self):
        if False:
            for i in range(10):
                print('nop')
        return tensor_shape.dimension_value(self._nrows[0])

    @property
    def nvals(self):
        if False:
            while True:
                i = 10
        return tensor_shape.dimension_value(self._nvals[0])

    @property
    def uniform_row_length(self):
        if False:
            print('Hello World!')
        return tensor_shape.dimension_value(self._uniform_row_length[0])

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        return self._dtype

    @property
    def _component_specs(self):
        if False:
            for i in range(10):
                print('nop')
        row_splits_shape = tensor_shape.TensorShape([tensor_shape.dimension_at_index(self._nrows, 0) + 1])
        return tensor_lib.TensorSpec(row_splits_shape, self._dtype)

    def _to_components(self, value):
        if False:
            i = 10
            return i + 15
        return value.row_splits()

    def _from_components(self, tensor):
        if False:
            while True:
                i = 10
        return RowPartition.from_row_splits(tensor, validate=False)

    @classmethod
    def from_value(cls, value):
        if False:
            print('Hello World!')
        if not isinstance(value, RowPartition):
            raise TypeError('Expected `value` to be a `RowPartition`')
        return cls(value.static_nrows, value.static_nvals, value.static_uniform_row_length, value.dtype)

    def __repr__(self):
        if False:
            return 10
        return 'RowPartitionSpec(nrows=%s, nvals=%s, uniform_row_length=%s, dtype=%r)' % (self.nrows, self.nvals, self.uniform_row_length, self.dtype)

    @staticmethod
    def _dimensions_compatible(nrows, nvals, uniform_row_length):
        if False:
            while True:
                i = 10
        'Returns true if the given dimensions are compatible.'
        nrows = tensor_shape.dimension_value(nrows[0])
        nvals = tensor_shape.dimension_value(nvals[0])
        ncols = tensor_shape.dimension_value(uniform_row_length[0])
        if nrows == 0 and nvals not in (0, None):
            return False
        if ncols == 0 and nvals not in (0, None):
            return False
        if ncols is not None and nvals is not None:
            if ncols != 0 and nvals % ncols != 0:
                return False
            if nrows is not None and nvals != ncols * nrows:
                return False
        return True

    def _merge_with(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Merge two RowPartitionSpecs.'
        nrows = self._nrows.merge_with(other.nrows)
        nvals = self._nvals.merge_with(other.nvals)
        ncols = self._uniform_row_length.merge_with(other.uniform_row_length)
        if not RowPartitionSpec._dimensions_compatible(nrows, nvals, ncols):
            raise ValueError('Merging incompatible RowPartitionSpecs')
        if self.dtype != other.dtype:
            raise ValueError('Merging RowPartitionSpecs with incompatible dtypes')
        return RowPartitionSpec(nrows=nrows[0], nvals=nvals[0], uniform_row_length=ncols[0], dtype=self.dtype)

    def with_dtype(self, dtype):
        if False:
            while True:
                i = 10
        nrows = tensor_shape.dimension_value(self._nrows[0])
        nvals = tensor_shape.dimension_value(self._nvals[0])
        return RowPartitionSpec(nrows, nvals, self._uniform_row_length, dtype)

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        del memo
        dtype = self.dtype
        nrows = tensor_shape.dimension_value(self._nrows[0])
        nvals = tensor_shape.dimension_value(self._nvals[0])
        uniform_row_length = None if self._uniform_row_length is None else tensor_shape.dimension_value(self._uniform_row_length[0])
        return RowPartitionSpec(nrows, nvals, uniform_row_length, dtype)
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(RowPartitionSpec, struct_pb2.TypeSpecProto.ROW_PARTITION_SPEC))

def _assert_monotonic_increasing(tensor, message=None):
    if False:
        return 10
    return check_ops.assert_non_negative(tensor[1:] - tensor[:-1], message=message)

def _assert_zero(tensor, message=None):
    if False:
        for i in range(10):
            print('nop')
    return check_ops.assert_equal(tensor, constant_op.constant(0, dtype=tensor.dtype), message=message)

def _cast_if_not_none(tensor, dtype):
    if False:
        while True:
            i = 10
    return None if tensor is None else math_ops.cast(tensor, dtype)

def _merge_tensors(t1, t2, name, validate):
    if False:
        while True:
            i = 10
    'Merge two optional Tensors with equal values into a single Tensor.\n\n  Args:\n    t1: tf.Tensor or None\n    t2: tf.Tensor or None\n    name: A name for the tensors (for error messages)\n    validate: If true, then check that `t1` is compatible with `t2` (if both are\n      non-None).\n\n  Returns:\n    A pair `(merged_value, validated)`:\n      * `merged_value` is `t1` if it is not None; or `t2` otherwise.\n      * `validated` is true if we validated that t1 and t2 are equal (either\n        by adding a check, or because t1 is t2).\n  '
    if t1 is None:
        return (t2, False)
    elif t2 is None:
        return (t1, False)
    elif t1 is t2:
        return (t1, True)
    else:
        err_msg = 'RowPartition._merge_precomputed_encodings: partitions have incompatible %s' % name
        if not t1.shape.is_compatible_with(t2.shape):
            raise ValueError(err_msg)
        if validate:
            checks = [check_ops.assert_equal(t1, t2, message=err_msg)]
            return (control_flow_ops.with_dependencies(checks, t1), True)
        else:
            return (t1, False)
_row_partition_factory_key = object()

def _get_dtype_or_none(value):
    if False:
        while True:
            i = 10
    if isinstance(value, tensor_lib.Tensor):
        return value.dtype
    return None

def _get_target_dtype(values, dtype=None, dtype_hint=None):
    if False:
        print('Hello World!')
    'Gets the target dtype of a family of values.'
    if dtype is not None:
        return dtype
    for value in values:
        if isinstance(value, tensor_lib.Tensor):
            return value.dtype
    for value in values:
        if isinstance(value, np.ndarray):
            return dtypes.as_dtype(value.dtype)
    if dtype_hint is not None:
        return dtype_hint
    return dtypes.int64

def _convert_all_to_tensors(values, dtype=None, dtype_hint=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert a list of objects to tensors of the same dtype.'
    target_dtype = _get_target_dtype([x for (x, _) in values], dtype, dtype_hint)
    convert_behavior = dtype is None
    if convert_behavior:
        return [None if x is None else ops.convert_to_tensor(x, dtype=target_dtype, name=name) for (x, name) in values]
    else:
        return [None if x is None else math_ops.cast(x, dtype=target_dtype, name=name) for (x, name) in values]