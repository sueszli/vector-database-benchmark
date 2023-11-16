"""Shapes & broadcasting for RaggedTensors."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util

class RaggedTensorDynamicShape:
    """A collection of tensors encoding the shape of a potentially ragged tensor.

  Each `RaggedTensorDynamicShape` consists of an ordered list of dimension
  sizes.  There are two dimension types:

    * "Uniform dimensions" are dimensions where all slices have the same
      length.  `RaggedTensorDynamicShape` records the size of each uniform
      dimension using a single scalar integer.

    * "Ragged dimensions" are dimensions whose slices may have different
      lengths.  `RaggedTensorDynamicShape` records the size of each ragged
      dimension using an integer vector containing the slice lengths for all
      the slices across that dimension.

  Furthermore, there are two ways a dimension might be encoded:

    * "Partitioned dimensions" are dimensions that are encoded using a
      `RaggedTensor`'s `nested_row_splits`.  The outermostmost partitioned
      dimension must be uniform, and the innermost partitioned dimension must
      be ragged.

    * "Inner dimensions" are dimensions that are encoded using a
      `RaggedTensor`'s `flat_values`.  Inner dimensions are always uniform.

  The sizes of partitioned dimensions are recorded using `partitioned_dim_sizes`
  and `inner_dim_sizes`:

    * `partitioned_dim_sizes` is a list of tensors (one for each partitioned
      dimension).

      * For uniform dimensions, the tensor is an integer scalar specifying the
        size of all slices across that dimension.
      * For ragged dimensions, the tensor is an integer vector specifying the
        size of each slice across that dimension.

    * `inner_dim_sizes` is a single integer vector, where each element
      specifies the size of a single inner dimension.

  Examples:

  Tensor                         | Ragged | Partitioned Dim Sizes  | Inner Dim
                                 : Rank   :                        : Sizes
  ------------------------------ | ------ | ---------------------- | ----------
  `[[1, 2, 3], [4, 5, 6]]`       |      0 |                        | `2, 3`
  `[[1, 2], [], [3, 4, 5]]`      |      1 | `3, (2, 0, 3)`         |
  `[[[1, 2], [3, 4]], [[5, 6]]]` |      1 | `2, (2, 1)`            | 2
  `[[[1, 2], [3]], [[4, 5]]]`    |      2 | `2, (2, 1), (2, 1, 2)` |
  """

    def __init__(self, partitioned_dim_sizes, inner_dim_sizes, dim_size_dtype=None):
        if False:
            while True:
                i = 10
        'Creates a RaggedTensorDynamicShape.\n\n    Args:\n      partitioned_dim_sizes: A `list` of 0-D or 1-D integer `Tensor`, one for\n        each partitioned dimension.  If dimension `d` is uniform, then\n        `partitioned_dim_sizes[d]` must be an integer scalar, specifying the\n        size of all slices across dimension `d`.  If dimension `d` is ragged,\n        then `partitioned_dim_sizes[d]` must be an integer vector, specifying\n        the size of each slice across dimension `d`.\n      inner_dim_sizes: A 1-D integer `Tensor`, whose length is equal to the\n        number of inner dimensions.  `inner_dim_sizes[n]` is the size of all\n        slices across the `n`th inner dimension (which is the\n        `(len(partitioned_dim_sizes)+n)`th dimension in the overall tensor.\n      dim_size_dtype: dtype for dimension sizes.  If not specified, then it\n        is chosen based on the dtypes of `partitioned_dim_sizes` and\n        `inner_dim_sizes`.\n    '
        assert isinstance(partitioned_dim_sizes, (list, tuple))
        with ops.name_scope(None, 'RaggedTensorDynamicShape', (partitioned_dim_sizes, inner_dim_sizes)):
            partitioned_dim_sizes = tuple((ops.convert_to_tensor(size, name='partitioned_dimension_size_%d' % i) for (i, size) in enumerate(partitioned_dim_sizes)))
            inner_dim_sizes = ops.convert_to_tensor(inner_dim_sizes, name='inner_dim_sizes')
            if partitioned_dim_sizes:
                for (axis, dimension_size) in enumerate(partitioned_dim_sizes):
                    if dimension_size.shape.ndims is None:
                        raise ValueError('rank of partitioned_dim_sizes[%d] is unknown' % axis)
                    dimension_size.shape.with_rank_at_most(1)
                if partitioned_dim_sizes[0].shape.ndims == 1:
                    raise ValueError('outermost partitioned dimension must be uniform')
                if partitioned_dim_sizes[-1].shape.ndims == 0:
                    raise ValueError('innermost partitioned dimension must be ragged')
            inner_dim_sizes.shape.assert_has_rank(1)
            if dim_size_dtype is None:
                dim_size_dtypes = set((p.dtype for p in partitioned_dim_sizes if p.shape.ndims == 1))
                if not dim_size_dtypes:
                    dim_size_dtype = dtypes.int64
                elif len(dim_size_dtypes) == 1:
                    dim_size_dtype = dim_size_dtypes.pop()
                else:
                    if not ragged_config.auto_cast_partition_dtype():
                        raise ValueError('partitioned_dim_sizes must have matching dtypes')
                    dim_size_dtype = dtypes.int64
            partitioned_dim_sizes = tuple((math_ops.cast(p, dim_size_dtype) for p in partitioned_dim_sizes))
            inner_dim_sizes = math_ops.cast(inner_dim_sizes, dim_size_dtype)
            self._partitioned_dim_sizes = partitioned_dim_sizes
            self._inner_dim_sizes = inner_dim_sizes

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'RaggedTensorDynamicShape(partitioned_dim_sizes=%r, inner_dim_sizes=%r)' % (self._partitioned_dim_sizes, self._inner_dim_sizes)

    @staticmethod
    def from_dim_sizes(dim_sizes):
        if False:
            for i in range(10):
                print('nop')
        'Constructs a ragged shape from a list of dimension sizes.\n\n    This list contains a single tensor for each dimension, where the tensor\n    is a scalar if the dimension is uniform, or a vector if the dimension is\n    ragged.\n\n    Args:\n      dim_sizes: List of int32 or int64 scalars or vectors.\n\n    Returns:\n      A RaggedTensorDynamicShape.\n    '
        with ops.name_scope(None, 'RaggedTensorDynamicShapeFromDimensionSizes', [dim_sizes]):
            dim_sizes = tuple((ops.convert_to_tensor(size, preferred_dtype=dtypes.int64, name='dim_sizes') for size in dim_sizes))
            inner_split = 0
            for (dim, dim_size) in enumerate(dim_sizes):
                if dim_size.shape.ndims == 1:
                    inner_split = dim + 1
                elif dim_size.shape.ndims != 0:
                    raise ValueError('Each dim_size must be a scalar or a vector')
            return RaggedTensorDynamicShape(dim_sizes[:inner_split], dim_sizes[inner_split:])

    @classmethod
    def from_tensor(cls, rt_input, dim_size_dtype=None):
        if False:
            i = 10
            return i + 15
        'Constructs a ragged shape for a potentially ragged tensor.'
        with ops.name_scope(None, 'RaggedTensorDynamicShapeFromTensor', [rt_input]):
            rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input)
            if not ragged_tensor.is_ragged(rt_input):
                return cls([], array_ops.shape(rt_input), dim_size_dtype=dim_size_dtype)
            else:
                partitioned_dim_sizes = (rt_input.nrows(),) + rt_input.nested_row_lengths()
                return RaggedTensorDynamicShape(partitioned_dim_sizes, array_ops.shape(rt_input.flat_values)[1:], dim_size_dtype=dim_size_dtype)

    def dimension_size(self, axis):
        if False:
            for i in range(10):
                print('nop')
        'Returns the size of slices across the specified dimension.'
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer')
        partitioned_ndims = len(self._partitioned_dim_sizes)
        if axis < partitioned_ndims:
            return self._partitioned_dim_sizes[axis]
        else:
            return self._inner_dim_sizes[axis - partitioned_ndims]

    def is_ragged(self, axis):
        if False:
            i = 10
            return i + 15
        'Returns true if the indicated dimension is ragged.'
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer')
        rank = self.rank
        if axis < 0:
            raise ValueError('Negative axis values are not supported')
        elif rank is not None and axis >= rank:
            raise ValueError('Expected axis=%s < rank=%s' % (axis, rank))
        else:
            return axis > 0 and axis < len(self._partitioned_dim_sizes) and (self._partitioned_dim_sizes[axis].shape.ndims == 1)

    @property
    def rank(self):
        if False:
            for i in range(10):
                print('nop')
        'The number of dimensions in this shape, or None if unknown.'
        inner_ndims = tensor_shape.dimension_value(self._inner_dim_sizes.shape[0])
        if inner_ndims is None:
            return None
        else:
            return len(self._partitioned_dim_sizes) + inner_ndims

    @property
    def partitioned_dim_sizes(self):
        if False:
            for i in range(10):
                print('nop')
        'The partitioned dimension sizes for this shape.\n\n    Returns:\n      A `list` of 0-D or 1-D integer `Tensor`.\n    '
        return self._partitioned_dim_sizes

    @property
    def inner_dim_sizes(self):
        if False:
            for i in range(10):
                print('nop')
        'The inner dimension sizes for this shape.\n\n    Returns:\n      A 1-D integer `Tensor`.\n    '
        return self._inner_dim_sizes

    @property
    def num_partitioned_dimensions(self):
        if False:
            return 10
        'The number of partitioned dimensions in this shape.'
        return len(self._partitioned_dim_sizes)

    @property
    def num_inner_dimensions(self):
        if False:
            print('Hello World!')
        'The number of inner dimensions, or `None` if not statically known.'
        return tensor_shape.dimension_value(self._inner_dim_sizes.shape[0])

    @property
    def dim_size_dtype(self):
        if False:
            while True:
                i = 10
        'DType used by this shape for dimension sizes.'
        return self._inner_dim_sizes.dtype

    def broadcast_to_rank(self, rank):
        if False:
            i = 10
            return i + 15
        'Adds leading size-1 dimensions to broadcast `self` to the given rank.\n\n    E.g., if `shape1` is `[3, (D2), 4]`, then `shape1.broadcast_to_rank(5)`\n    is `[1, 1, 3, (D2), 4]`.\n\n    Args:\n      rank: The rank for the returned shape.\n\n    Returns:\n      A RaggedTensorDynamicShape with `rank` dimensions, whose inner dimensions\n      have the same size as `self` and whose outer dimensions have size `1`.\n\n    Raises:\n      ValueError: If `self.rank` is unknown or greater than `rank`.\n    '
        if self.rank is None:
            raise ValueError('Unable to broadcast: self.rank is unknown')
        dims_to_add = rank - self.rank
        if dims_to_add < 0:
            raise ValueError('Unable to broadcast: rank=%d must be greater than self.rank=%d.' % (rank, self.rank))
        elif dims_to_add == 0:
            return self
        elif self._partitioned_dim_sizes:
            partitioned_dims = (1,) * dims_to_add + self._partitioned_dim_sizes
            return RaggedTensorDynamicShape(partitioned_dims, self.inner_dim_sizes, self.dim_size_dtype)
        else:
            inner_dims = array_ops.concat([array_ops.ones([dims_to_add], self.dim_size_dtype), self.inner_dim_sizes], axis=0)
            return RaggedTensorDynamicShape([], inner_dims, self.dim_size_dtype)

    def broadcast_dimension(self, axis, lengths):
        if False:
            return 10
        'Returns a shape that is broadcast-compatible with self & lengths.\n\n    * If dimension[axis] is uniform and lengths is a scalar, the check\n      that either lengths==1 or axis==1 or lengths==axis, and tile\n      dimension[axis] with tf.where(lengths==axis, 1, axis) repeats.\n\n    * If dimension[axis] is uniform and lengths is a vector, then check\n      that dimension[axis]==1, and raggedly tile dimension[axis] with\n      lengths repeats.  (we can skip tiling if we statically know that\n      slice_lengths == 1??)\n\n    * If dimension[axis] is ragged and lengths is a scalar, then check\n      that lengths==1.\n\n    * If dimension[axis] is ragged and lengths is a vector, then check\n      that self.dimension_size(axis) == lengths.\n\n    Args:\n      axis: `int`.  The dimension to broadcast.\n      lengths: 0-D or 1-D integer `Tensor`.\n\n    Returns:\n      A `RaggedTensorDynamicShape`.\n    '
        lengths = ragged_util.convert_to_int_tensor(lengths, name='lengths', dtype=self.dim_size_dtype)
        if lengths.shape.ndims is None:
            raise ValueError('lengths must have a known rank.')
        elif lengths.shape.ndims > 1:
            raise ValueError('lengths must be a scalar or vector')
        else:
            lengths_is_scalar = lengths.shape.ndims == 0
        if self.is_ragged(axis):
            if lengths_is_scalar:
                condition = math_ops.equal(lengths, 1)
            else:
                condition = math_ops.reduce_all(math_ops.equal(lengths, self.dimension_size(axis)))
        else:
            axis_dim_size = self.dimension_size(axis)
            if lengths_is_scalar:
                condition = math_ops.equal(lengths, 1) | math_ops.equal(axis_dim_size, 1) | math_ops.equal(axis_dim_size, lengths)
            else:
                condition = math_ops.equal(axis_dim_size, 1)
        broadcast_err = ['Unable to broadcast: dimension size mismatch in dimension', axis, 'lengths=', lengths, 'dim_size=', self.dimension_size(axis)]
        broadcast_check = control_flow_assert.Assert(condition, data=broadcast_err, summarize=10)
        with ops.control_dependencies([broadcast_check]):
            if axis < self.num_partitioned_dimensions:
                if self.is_ragged(axis):
                    return RaggedTensorDynamicShape(self._partitioned_dim_sizes, array_ops.identity(self.inner_dim_sizes), self.dim_size_dtype)
                else:
                    return self._broadcast_uniform_partitioned_dimension(axis, lengths)
            elif lengths_is_scalar:
                return self._broadcast_inner_dimension_to_uniform(axis, lengths)
            else:
                if axis == 0:
                    raise ValueError('Unable to broadcast: outermost dimension must be uniform.')
                return self._broadcast_inner_dimension_to_ragged(axis, lengths)

    def num_slices_in_dimension(self, axis):
        if False:
            i = 10
            return i + 15
        'Returns the total number of slices across the indicated dimension.'
        if axis < 0:
            return constant_op.constant(1, dtype=self.dim_size_dtype)
        elif self.is_ragged(axis):
            return math_ops.reduce_sum(self._partitioned_dim_sizes[axis])
        else:
            return self.dimension_size(axis) * self.num_slices_in_dimension(axis - 1)

    def _broadcast_uniform_partitioned_dimension(self, axis, lengths):
        if False:
            print('Hello World!')
        'Broadcasts the partitioned dimension `axis` to match `lengths`.'
        axis_dim_size = self.dimension_size(axis)
        partitioned_sizes = list(self._partitioned_dim_sizes[:axis])
        if lengths.shape.ndims == 0:
            lengths = array_ops.where(math_ops.equal(axis_dim_size, 1), lengths, axis_dim_size)
            repeats = array_ops.where(math_ops.equal(axis_dim_size, 1), lengths, 1)
            splits = array_ops_stack.stack([0, self.num_slices_in_dimension(axis)])
        else:
            splits = math_ops.range(array_ops.size(lengths, out_type=self.dim_size_dtype) + 1)
            repeats = lengths
        partitioned_sizes.append(lengths)
        for dim_size in self._partitioned_dim_sizes[axis + 1:]:
            if dim_size.shape.ndims == 0:
                partitioned_sizes.append(dim_size)
                splits *= dim_size
            else:
                partitioned_sizes.append(ragged_util.repeat_ranges(dim_size, splits, repeats))
                splits = array_ops.gather(ragged_util.lengths_to_splits(dim_size), splits)
        inner_sizes = self._inner_dim_sizes
        return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes, self.dim_size_dtype)

    def _broadcast_inner_dimension_to_uniform(self, axis, length):
        if False:
            while True:
                i = 10
        'Broadcasts the inner dimension `axis` to match `lengths`.'
        dim_size = self.dimension_size(axis)
        axis_in_inner_dims = axis - self.num_partitioned_dimensions
        partitioned_sizes = self._partitioned_dim_sizes
        inner_sizes = array_ops.concat([self._inner_dim_sizes[:axis_in_inner_dims], [array_ops.where(math_ops.equal(dim_size, 1), length, dim_size)], self._inner_dim_sizes[axis_in_inner_dims + 1:]], axis=0)
        return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes, self.dim_size_dtype)

    def _broadcast_inner_dimension_to_ragged(self, axis, lengths):
        if False:
            return 10
        axis_in_inner_dims = axis - self.num_partitioned_dimensions
        partitioned_sizes = self._partitioned_dim_sizes + tuple([self._inner_dim_sizes[i] for i in range(axis_in_inner_dims)]) + (lengths,)
        inner_sizes = self._inner_dim_sizes[axis_in_inner_dims + 1:]
        return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes)

    def with_dim_size_dtype(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        if dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('dtype must be int32 or int64')
        if self.dim_size_dtype == dtype:
            return self
        return RaggedTensorDynamicShape([math_ops.cast(p, dtype) for p in self._partitioned_dim_sizes], math_ops.cast(self._inner_dim_sizes, dtype))

def broadcast_dynamic_shape(shape_x, shape_y):
    if False:
        return 10
    'Returns the shape formed by broadcasting two shapes to be compatible.\n\n  Args:\n    shape_x: A `RaggedTensorDynamicShape`\n    shape_y: A `RaggedTensorDynamicShape`\n\n  Returns:\n    A `RaggedTensorDynamicShape`.\n  Raises:\n    ValueError: If `shape_x` and `shape_y` are not broadcast-compatible.\n  '
    if not isinstance(shape_x, RaggedTensorDynamicShape):
        raise TypeError('shape_x must be a RaggedTensorDynamicShape')
    if not isinstance(shape_y, RaggedTensorDynamicShape):
        raise TypeError('shape_y must be a RaggedTensorDynamicShape')
    if shape_x.rank is None or shape_y.rank is None:
        raise ValueError('Unable to broadcast: unknown rank')
    broadcast_rank = max(shape_x.rank, shape_y.rank)
    shape_x = shape_x.broadcast_to_rank(broadcast_rank)
    shape_y = shape_y.broadcast_to_rank(broadcast_rank)
    for axis in range(broadcast_rank):
        shape_x = shape_x.broadcast_dimension(axis, shape_y.dimension_size(axis))
        shape_y = shape_y.broadcast_dimension(axis, shape_x.dimension_size(axis))
    return shape_x

def broadcast_to(rt_input, shape, broadcast_inner_dimensions=True):
    if False:
        while True:
            i = 10
    'Broadcasts a potentially ragged tensor to a ragged shape.\n\n  Tiles `rt_input` as necessary to match the given shape.\n\n  Behavior is undefined if `rt_input` is not broadcast-compatible with `shape`.\n\n  Args:\n    rt_input: The potentially ragged tensor to broadcast.\n    shape: A `RaggedTensorDynamicShape`\n    broadcast_inner_dimensions: If false, then inner dimensions will not be\n      tiled.\n\n  Returns:\n    A potentially ragged tensor whose values are taken from\n    `rt_input`, and whose shape matches `shape`.\n  '
    if not isinstance(shape, RaggedTensorDynamicShape):
        raise TypeError('shape must be a RaggedTensorDynamicShape')
    rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input)
    if shape.num_partitioned_dimensions == 0:
        return _broadcast_to_uniform_shape(rt_input, shape, broadcast_inner_dimensions)
    else:
        return _broadcast_to_ragged_shape(rt_input, shape, broadcast_inner_dimensions)

def _broadcast_to_uniform_shape(rt_input, shape, broadcast_inner_dimensions):
    if False:
        return 10
    'Broadcasts rt_input to the uniform shape `shape`.'
    if isinstance(rt_input, ragged_tensor.RaggedTensor):
        raise ValueError('Incompatible with shape: ragged rank mismatch')
    if broadcast_inner_dimensions:
        return array_ops.broadcast_to(rt_input, shape.inner_dim_sizes)
    else:
        return rt_input

def _broadcast_to_ragged_shape(rt_input, dst_shape, broadcast_inner_dimensions):
    if False:
        for i in range(10):
            print('nop')
    'Broadcasts rt_input to the ragged shape `dst_shape`.'
    if isinstance(rt_input, ragged_tensor.RaggedTensor) and rt_input.row_splits.dtype != dst_shape.dim_size_dtype:
        if not ragged_config.auto_cast_partition_dtype():
            raise ValueError('rt_input and dst_shape have different row_split dtypes; use RaggedTensor.with_row_splits_dtype() or RaggedTensorDynamicShape.with_dim_size_dtype() to convert to a compatible dtype.')
        rt_input = rt_input.with_row_splits_dtype(dtypes.int64)
        dst_shape = dst_shape.with_dim_size_dtype(dtypes.int64)
    if rt_input.shape.ndims is None or dst_shape.rank is None:
        raise ValueError('Unable to broadcast: unknown rank')
    if rt_input.shape.ndims > dst_shape.rank:
        raise ValueError('Incompatible with shape: rank mismatch')
    if isinstance(rt_input, ragged_tensor.RaggedTensor) and rt_input.ragged_rank >= dst_shape.num_partitioned_dimensions:
        raise ValueError('Incompatible with shape: ragged rank mismatch')
    src_shape = RaggedTensorDynamicShape.from_tensor(rt_input)
    src_shape = src_shape.broadcast_to_rank(dst_shape.rank)
    if dst_shape.rank > rt_input.shape.ndims:
        if rt_input.shape.ndims < dst_shape.num_inner_dimensions + 1:
            rt_input = array_ops.reshape(rt_input, array_ops.concat([[-1], dst_shape.inner_dim_sizes], axis=0))
        for _ in range(dst_shape.rank - rt_input.shape.ndims):
            if ragged_tensor.is_ragged(rt_input):
                nrows = rt_input.nrows()
            else:
                nrows = array_ops.shape(rt_input, out_type=dst_shape.dim_size_dtype)[0]
            rt_input = ragged_tensor.RaggedTensor.from_row_lengths(rt_input, [nrows], validate=False)
    if ragged_tensor.is_ragged(rt_input):
        inner_rank_diff = rt_input.flat_values.shape.ndims - 1 - dst_shape.num_inner_dimensions
        if inner_rank_diff > 0:
            rt_input = rt_input.with_flat_values(ragged_tensor.RaggedTensor.from_tensor(rt_input.flat_values, ragged_rank=inner_rank_diff, row_splits_dtype=dst_shape.dim_size_dtype))
    else:
        rt_input = ragged_tensor.RaggedTensor.from_tensor(rt_input, ragged_rank=dst_shape.num_partitioned_dimensions - 1, row_splits_dtype=dst_shape.dim_size_dtype)
    multiples = [1] * dst_shape.rank
    for axis in range(dst_shape.num_partitioned_dimensions):
        if not src_shape.is_ragged(axis) and (not dst_shape.is_ragged(axis)):
            src_size = src_shape.dimension_size(axis)
            dst_size = dst_shape.dimension_size(axis)
            if tensor_util.constant_value(src_size) in (1, None) and tensor_util.constant_value(dst_size) != 1:
                multiples[axis] = array_ops.where(math_ops.equal(src_size, 1), dst_size, 1)
    if not all((isinstance(v, int) and v == 1 for v in multiples)):
        multiples = array_ops_stack.stack(multiples, axis=0)
        rt_input = ragged_array_ops.tile(rt_input, multiples)
    if broadcast_inner_dimensions:
        new_shape = array_ops.broadcast_dynamic_shape(array_ops.shape(rt_input.flat_values, out_type=dst_shape.dim_size_dtype), array_ops.concat([[1], dst_shape.inner_dim_sizes], axis=0))
        rt_input = rt_input.with_flat_values(array_ops.broadcast_to(rt_input.flat_values, new_shape))
    for axis in range(dst_shape.num_partitioned_dimensions):
        if not src_shape.is_ragged(axis) and dst_shape.is_ragged(axis):
            dst_size = dst_shape.dimension_size(axis)
            rt_input = _ragged_tile_axis(rt_input, axis, dst_size, dst_shape.dim_size_dtype)
    return rt_input

def _ragged_tile_axis(rt_input, axis, repeats, row_splits_dtype):
    if False:
        return 10
    'Tile a dimension of a RaggedTensor to match a ragged shape.'
    assert axis > 0
    if not ragged_tensor.is_ragged(rt_input):
        rt_input = ragged_tensor.RaggedTensor.from_tensor(rt_input, ragged_rank=1, row_splits_dtype=row_splits_dtype)
    if axis > 1:
        return rt_input.with_values(_ragged_tile_axis(rt_input.values, axis - 1, repeats, row_splits_dtype))
    else:
        src_row_splits = rt_input.nested_row_splits
        src_row_lengths = rt_input.nested_row_lengths()
        splits = src_row_splits[0]
        dst_row_lengths = [repeats]
        for i in range(1, len(src_row_lengths)):
            dst_row_lengths.append(ragged_util.repeat_ranges(src_row_lengths[i], splits, repeats))
            splits = array_ops.gather(src_row_splits[i], splits)
        dst_values = ragged_util.repeat_ranges(rt_input.flat_values, splits, repeats)
        return ragged_tensor.RaggedTensor.from_nested_row_lengths(dst_values, dst_row_lengths, validate=False)