"""Operations for constructing RaggedTensors."""
from typing import Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('ragged.constant')
@dispatch.add_dispatch_support
def constant(pylist, dtype=None, ragged_rank=None, inner_shape=None, name=None, row_splits_dtype=dtypes.int64) -> Union[ragged_tensor.RaggedTensor, ops._EagerTensorBase, ops.Operation]:
    if False:
        for i in range(10):
            print('nop')
    "Constructs a constant RaggedTensor from a nested Python list.\n\n  Example:\n\n  >>> tf.ragged.constant([[1, 2], [3], [4, 5, 6]])\n  <tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>\n\n  All scalar values in `pylist` must have the same nesting depth `K`, and the\n  returned `RaggedTensor` will have rank `K`.  If `pylist` contains no scalar\n  values, then `K` is one greater than the maximum depth of empty lists in\n  `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.\n\n  Args:\n    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that\n      is not a `list`, `tuple` or `np.ndarray` must be a scalar value\n      compatible with `dtype`.\n    dtype: The type of elements for the returned `RaggedTensor`.  If not\n      specified, then a default is chosen based on the scalar values in\n      `pylist`.\n    ragged_rank: An integer specifying the ragged rank of the returned\n      `RaggedTensor`.  Must be nonnegative and less than `K`. Defaults to\n      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to\n      `max(0, K - 1 - len(inner_shape))` if `inner_shape` is specified.\n    inner_shape: A tuple of integers specifying the shape for individual inner\n      values in the returned `RaggedTensor`.  Defaults to `()` if `ragged_rank`\n      is not specified.  If `ragged_rank` is specified, then a default is chosen\n      based on the contents of `pylist`.\n    name: A name prefix for the returned tensor (optional).\n    row_splits_dtype: data type for the constructed `RaggedTensor`'s row_splits.\n      One of `tf.int32` or `tf.int64`.\n\n  Returns:\n    A potentially ragged tensor with rank `K` and the specified `ragged_rank`,\n    containing the values from `pylist`.\n\n  Raises:\n    ValueError: If the scalar values in `pylist` have inconsistent nesting\n      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.\n  "

    def ragged_factory(values, row_splits):
        if False:
            while True:
                i = 10
        row_splits = constant_op.constant(row_splits, dtype=row_splits_dtype)
        return ragged_tensor.RaggedTensor.from_row_splits(values, row_splits, validate=False)
    with ops.name_scope(name, 'RaggedConstant'):
        return _constant_value(ragged_factory, constant_op.constant, pylist, dtype, ragged_rank, inner_shape)

@tf_export(v1=['ragged.constant_value'])
@dispatch.add_dispatch_support
def constant_value(pylist, dtype=None, ragged_rank=None, inner_shape=None, row_splits_dtype='int64') -> Union[ragged_tensor_value.RaggedTensorValue, np.ndarray]:
    if False:
        i = 10
        return i + 15
    "Constructs a RaggedTensorValue from a nested Python list.\n\n  Warning: This function returns a `RaggedTensorValue`, not a `RaggedTensor`.\n  If you wish to construct a constant `RaggedTensor`, use\n  [`ragged.constant(...)`](constant.md) instead.\n\n  Example:\n\n  >>> tf.compat.v1.ragged.constant_value([[1, 2], [3], [4, 5, 6]])\n  tf.RaggedTensorValue(values=array([1, 2, 3, 4, 5, 6]),\n                       row_splits=array([0, 2, 3, 6]))\n\n  All scalar values in `pylist` must have the same nesting depth `K`, and the\n  returned `RaggedTensorValue` will have rank `K`.  If `pylist` contains no\n  scalar values, then `K` is one greater than the maximum depth of empty lists\n  in `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.\n\n  Args:\n    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that\n      is not a `list` or `tuple` must be a scalar value compatible with `dtype`.\n    dtype: `numpy.dtype`.  The type of elements for the returned `RaggedTensor`.\n      If not specified, then a default is chosen based on the scalar values in\n      `pylist`.\n    ragged_rank: An integer specifying the ragged rank of the returned\n      `RaggedTensorValue`.  Must be nonnegative and less than `K`. Defaults to\n      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K\n      - 1 - len(inner_shape))` if `inner_shape` is specified.\n    inner_shape: A tuple of integers specifying the shape for individual inner\n      values in the returned `RaggedTensorValue`.  Defaults to `()` if\n      `ragged_rank` is not specified.  If `ragged_rank` is specified, then a\n      default is chosen based on the contents of `pylist`.\n    row_splits_dtype: data type for the constructed `RaggedTensorValue`'s\n      row_splits.  One of `numpy.int32` or `numpy.int64`.\n\n  Returns:\n    A `tf.RaggedTensorValue` or `numpy.array` with rank `K` and the specified\n    `ragged_rank`, containing the values from `pylist`.\n\n  Raises:\n    ValueError: If the scalar values in `pylist` have inconsistent nesting\n      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.\n  "
    if dtype is not None and isinstance(dtype, dtypes.DType):
        dtype = dtype.as_numpy_dtype
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype).as_numpy_dtype

    def _ragged_factory(values, row_splits):
        if False:
            while True:
                i = 10
        row_splits = np.array(row_splits, dtype=row_splits_dtype)
        return ragged_tensor_value.RaggedTensorValue(values, row_splits)

    def _inner_factory(pylist, dtype, shape, name=None):
        if False:
            i = 10
            return i + 15
        return np.reshape(np.array(pylist, dtype=dtype), shape)
    return _constant_value(_ragged_factory, _inner_factory, pylist, dtype, ragged_rank, inner_shape)

def _constant_value(ragged_factory, inner_factory, pylist, dtype, ragged_rank, inner_shape):
    if False:
        i = 10
        return i + 15
    'Constructs a constant RaggedTensor or RaggedTensorValue.\n\n  Args:\n    ragged_factory: A factory function with the signature:\n      `ragged_factory(values, row_splits)`\n    inner_factory: A factory function with the signature: `inner_factory(pylist,\n      dtype, shape, name)`\n    pylist: A nested `list`, `tuple` or `np.ndarray`.\n    dtype: Data type for returned value.\n    ragged_rank: Ragged rank for returned value.\n    inner_shape: Inner value shape for returned value.\n\n  Returns:\n    A value returned by `ragged_factory` or `inner_factory`.\n\n  Raises:\n    ValueError: If the scalar values in `pylist` have inconsistent nesting\n      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.\n  '
    if ragged_tensor.is_ragged(pylist):
        raise TypeError('pylist may not be a RaggedTensor or RaggedTensorValue.')
    if not isinstance(pylist, (list, tuple)) and np.ndim(pylist) == 0:
        if ragged_rank is not None and ragged_rank != 0:
            raise ValueError('Invalid pylist=%r: incompatible with ragged_rank=%d' % (pylist, ragged_rank))
        if inner_shape is not None and inner_shape:
            raise ValueError('Invalid pylist=%r: incompatible with dim(inner_shape)=%d' % (pylist, len(inner_shape)))
        return inner_factory(pylist, dtype, ())
    if ragged_rank is not None and ragged_rank < 0:
        raise ValueError('Invalid ragged_rank=%r: must be nonnegative' % ragged_rank)
    (scalar_depth, max_depth) = _find_scalar_and_max_depth(pylist)
    if scalar_depth is not None:
        if max_depth > scalar_depth:
            raise ValueError('Invalid pylist=%r: empty list nesting is greater than scalar value nesting' % pylist)
        if ragged_rank is not None and max_depth < ragged_rank:
            raise ValueError(f'Invalid pylist={pylist}, max depth smaller than ragged_rank={ragged_rank}')
    if inner_shape is not None and ragged_rank is not None:
        expected_depth = ragged_rank + len(inner_shape) + 1
        if scalar_depth is not None and expected_depth != scalar_depth or (scalar_depth is None and expected_depth < max_depth):
            raise ValueError('Invalid pylist=%r: incompatible with ragged_rank=%d and dim(inner_shape)=%d' % (pylist, ragged_rank, len(inner_shape)))
    if ragged_rank == 0 or (ragged_rank is None and (max_depth < 2 or (inner_shape is not None and max_depth - len(inner_shape) < 2))):
        return inner_factory(pylist, dtype, inner_shape)
    if inner_shape is None:
        if ragged_rank is None:
            inner_shape = ()
        else:
            inner_shape = _default_inner_shape_for_pylist(pylist, ragged_rank)
    if ragged_rank is None:
        if scalar_depth is None:
            ragged_rank = max(1, max_depth - 1)
        else:
            ragged_rank = max(1, scalar_depth - 1 - len(inner_shape))
    nested_splits = []
    values = pylist
    for dim in range(ragged_rank):
        nested_splits.append([0])
        concatenated_values = []
        for row in values:
            nested_splits[dim].append(nested_splits[dim][-1] + len(row))
            concatenated_values.extend(row)
        values = concatenated_values
    values = inner_factory(values, dtype=dtype, shape=(len(values),) + inner_shape, name='values')
    for row_splits in reversed(nested_splits):
        values = ragged_factory(values, row_splits)
    return values

def _find_scalar_and_max_depth(pylist):
    if False:
        return 10
    'Finds nesting depth of scalar values in pylist.\n\n  Args:\n    pylist: A nested python `list` or `tuple`.\n\n  Returns:\n    A tuple `(scalar_depth, max_depth)`.  `scalar_depth` is the nesting\n    depth of scalar values in `pylist`, or `None` if `pylist` contains no\n    scalars.  `max_depth` is the maximum depth of `pylist` (including\n    empty lists).\n\n  Raises:\n    ValueError: If pylist has inconsistent nesting depths for scalars.\n  '
    if isinstance(pylist, (list, tuple)) or np.ndim(pylist) != 0:
        scalar_depth = None
        max_depth = 1
        for child in pylist:
            (child_scalar_depth, child_max_depth) = _find_scalar_and_max_depth(child)
            if child_scalar_depth is not None:
                if scalar_depth is not None and scalar_depth != child_scalar_depth + 1:
                    raise ValueError('all scalar values must have the same nesting depth')
                scalar_depth = child_scalar_depth + 1
            max_depth = max(max_depth, child_max_depth + 1)
        return (scalar_depth, max_depth)
    return (0, 0)

def _default_inner_shape_for_pylist(pylist, ragged_rank):
    if False:
        while True:
            i = 10
    'Computes a default inner shape for the given python list.'

    def get_inner_shape(item):
        if False:
            print('Hello World!')
        'Returns the inner shape for a python list `item`.'
        if not isinstance(item, (list, tuple)) and np.ndim(item) == 0:
            return ()
        elif len(item) > 0:
            return (len(item),) + get_inner_shape(item[0])
        return (0,)

    def check_inner_shape(item, shape):
        if False:
            for i in range(10):
                print('nop')
        'Checks that `item` has a consistent shape matching `shape`.'
        is_nested = isinstance(item, (list, tuple)) or np.ndim(item) != 0
        if is_nested != bool(shape):
            raise ValueError('inner values have inconsistent shape')
        if is_nested:
            if shape[0] != len(item):
                raise ValueError('inner values have inconsistent shape')
            for child in item:
                check_inner_shape(child, shape[1:])
    flat_values = pylist
    for dim in range(ragged_rank):
        if not all((isinstance(v, (list, tuple)) or np.ndim(v) != 0 for v in flat_values)):
            raise ValueError('pylist has scalar values depth %d, but ragged_rank=%d requires scalar value depth greater than %d' % (dim + 1, ragged_rank, ragged_rank))
        flat_values = sum((list(v) for v in flat_values), [])
    inner_shape = get_inner_shape(flat_values)
    check_inner_shape(flat_values, inner_shape)
    return inner_shape[1:]

@tf_export(v1=['ragged.placeholder'])
@dispatch.add_dispatch_support
def placeholder(dtype, ragged_rank, value_shape=None, name=None):
    if False:
        print('Hello World!')
    'Creates a placeholder for a `tf.RaggedTensor` that will always be fed.\n\n  **Important**: This ragged tensor will produce an error if evaluated.\n  Its value must be fed using the `feed_dict` optional argument to\n  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.\n\n\n  Args:\n    dtype: The data type for the `RaggedTensor`.\n    ragged_rank: The ragged rank for the `RaggedTensor`\n    value_shape: The shape for individual flat values in the `RaggedTensor`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `RaggedTensor` that may be used as a handle for feeding a value, but\n    not evaluated directly.\n\n  Raises:\n    RuntimeError: if eager execution is enabled\n\n  @compatibility(TF2)\n  This API is not compatible with eager execution and `tf.function`. To migrate\n  to TF2, rewrite the code to be compatible with eager execution. Check the\n  [migration\n  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)\n  on replacing `Session.run` calls. In TF2, you can just pass tensors directly\n  into ops and layers. If you want to explicitly set up your inputs, also see\n  [Keras functional API](https://www.tensorflow.org/guide/keras/functional) on\n  how to use `tf.keras.Input` to replace `tf.compat.v1.ragged.placeholder`.\n  `tf.function` arguments also do the job of `tf.compat.v1.ragged.placeholder`.\n  For more details please read [Better\n  performance with tf.function](https://www.tensorflow.org/guide/function).\n  @end_compatibility\n  '
    if ragged_rank == 0:
        return array_ops.placeholder(dtype, value_shape, name)
    with ops.name_scope(name, 'RaggedPlaceholder', []):
        flat_shape = tensor_shape.TensorShape([None]).concatenate(value_shape)
        result = array_ops.placeholder(dtype, flat_shape, 'flat_values')
        for i in reversed(range(ragged_rank)):
            row_splits = array_ops.placeholder(dtypes.int64, [None], 'row_splits_%d' % i)
            result = ragged_tensor.RaggedTensor.from_row_splits(result, row_splits, validate=False)
        return result