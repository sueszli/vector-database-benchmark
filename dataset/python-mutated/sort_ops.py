"""Support for sorting tensors."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('sort')
@dispatch.add_dispatch_support
def sort(values, axis=-1, direction='ASCENDING', name=None):
    if False:
        print('Hello World!')
    "Sorts a tensor.\n\n  Usage:\n\n  >>> a = [1, 10, 26.9, 2.8, 166.32, 62.3]\n  >>> tf.sort(a).numpy()\n  array([  1.  ,   2.8 ,  10.  ,  26.9 ,  62.3 , 166.32], dtype=float32)\n\n  >>> tf.sort(a, direction='DESCENDING').numpy()\n  array([166.32,  62.3 ,  26.9 ,  10.  ,   2.8 ,   1.  ], dtype=float32)\n\n  For multidimensional inputs you can control which axis the sort is applied\n  along. The default `axis=-1` sorts the innermost axis.\n\n  >>> mat = [[3,2,1],\n  ...        [2,1,3],\n  ...        [1,3,2]]\n  >>> tf.sort(mat, axis=-1).numpy()\n  array([[1, 2, 3],\n         [1, 2, 3],\n         [1, 2, 3]], dtype=int32)\n  >>> tf.sort(mat, axis=0).numpy()\n  array([[1, 1, 1],\n         [2, 2, 2],\n         [3, 3, 3]], dtype=int32)\n\n  See also:\n\n    * `tf.argsort`: Like sort, but it returns the sort indices.\n    * `tf.math.top_k`: A partial sort that returns a fixed number of top values\n      and corresponding indices.\n\n\n  Args:\n    values: 1-D or higher **numeric** `Tensor`.\n    axis: The axis along which to sort. The default is -1, which sorts the last\n      axis.\n    direction: The direction in which to sort the values (`'ASCENDING'` or\n      `'DESCENDING'`).\n    name: Optional name for the operation.\n\n  Returns:\n    A `Tensor` with the same dtype and shape as `values`, with the elements\n        sorted along the given `axis`.\n\n  Raises:\n    tf.errors.InvalidArgumentError: If the `values.dtype` is not a `float` or\n        `int` type.\n    ValueError: If axis is not a constant scalar, or the direction is invalid.\n  "
    with framework_ops.name_scope(name, 'sort'):
        return _sort_or_argsort(values, axis, direction, return_argsort=False)

@tf_export('argsort')
@dispatch.add_dispatch_support
def argsort(values, axis=-1, direction='ASCENDING', stable=False, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns the indices of a tensor that give its sorted order along an axis.\n\n  >>> values = [1, 10, 26.9, 2.8, 166.32, 62.3]\n  >>> sort_order = tf.argsort(values)\n  >>> sort_order.numpy()\n  array([0, 3, 1, 2, 5, 4], dtype=int32)\n\n  For a 1D tensor:\n\n  >>> sorted = tf.gather(values, sort_order)\n  >>> assert tf.reduce_all(sorted == tf.sort(values))\n\n  For higher dimensions, the output has the same shape as\n  `values`, but along the given axis, values represent the index of the sorted\n  element in that slice of the tensor at the given position.\n\n  >>> mat = [[30,20,10],\n  ...        [20,10,30],\n  ...        [10,30,20]]\n  >>> indices = tf.argsort(mat)\n  >>> indices.numpy()\n  array([[2, 1, 0],\n         [1, 0, 2],\n         [0, 2, 1]], dtype=int32)\n\n  If `axis=-1` these indices can be used to apply a sort using `tf.gather`:\n\n  >>> tf.gather(mat, indices, batch_dims=-1).numpy()\n  array([[10, 20, 30],\n         [10, 20, 30],\n         [10, 20, 30]], dtype=int32)\n\n  See also:\n\n    * `tf.sort`: Sort along an axis.\n    * `tf.math.top_k`: A partial sort that returns a fixed number of top values\n      and corresponding indices.\n\n  Args:\n    values: 1-D or higher **numeric** `Tensor`.\n    axis: The axis along which to sort. The default is -1, which sorts the last\n      axis.\n    direction: The direction in which to sort the values (`'ASCENDING'` or\n      `'DESCENDING'`).\n    stable: If True, equal elements in the original tensor will not be\n      re-ordered in the returned order. Unstable sort is not yet implemented,\n      but will eventually be the default for performance reasons. If you require\n      a stable order, pass `stable=True` for forwards compatibility.\n    name: Optional name for the operation.\n\n  Returns:\n    An int32 `Tensor` with the same shape as `values`. The indices that would\n        sort each slice of the given `values` along the given `axis`.\n\n  Raises:\n    ValueError: If axis is not a constant scalar, or the direction is invalid.\n    tf.errors.InvalidArgumentError: If the `values.dtype` is not a `float` or\n        `int` type.\n  "
    del stable
    with framework_ops.name_scope(name, 'argsort'):
        return _sort_or_argsort(values, axis, direction, return_argsort=True)

def _sort_or_argsort(values, axis, direction, return_argsort):
    if False:
        print('Hello World!')
    "Internal sort/argsort implementation.\n\n  Args:\n    values: The input values.\n    axis: The axis along which to sort.\n    direction: 'ASCENDING' or 'DESCENDING'.\n    return_argsort: Whether to return the argsort result.\n\n  Returns:\n    Either the sorted values, or the indices of the sorted values in the\n        original tensor. See the `sort` and `argsort` docstrings.\n\n  Raises:\n    ValueError: If axis is not a constant scalar, or the direction is invalid.\n  "
    if direction not in _SORT_IMPL:
        valid_directions = ', '.join(sorted(_SORT_IMPL.keys()))
        raise ValueError(f'Argument `direction` should be one of {valid_directions}. Received: direction={direction}')
    axis = framework_ops.convert_to_tensor(axis, name='axis')
    axis_static = tensor_util.constant_value(axis)
    if axis.shape.ndims not in (None, 0) or axis_static is None:
        raise ValueError(f'Argument `axis` must be a constant scalar. Received: axis={axis}.')
    axis_static = int(axis_static)
    values = framework_ops.convert_to_tensor(values, name='values')
    return _SORT_IMPL[direction](values, axis_static, return_argsort)

def _descending_sort(values, axis, return_argsort=False):
    if False:
        print('Hello World!')
    'Sorts values in reverse using `top_k`.\n\n  Args:\n    values: Tensor of numeric values.\n    axis: Index of the axis which values should be sorted along.\n    return_argsort: If False, return the sorted values. If True, return the\n      indices that would sort the values.\n\n  Returns:\n    The sorted values.\n  '
    k = array_ops.shape(values)[axis]
    rank = array_ops.rank(values)
    static_rank = values.shape.ndims
    if axis == -1 or axis + 1 == values.get_shape().ndims:
        top_k_input = values
        transposition = None
    else:
        if axis < 0:
            axis += static_rank or rank
        if static_rank is not None:
            transposition = constant_op.constant(np.r_[np.arange(axis), [static_rank - 1], np.arange(axis + 1, static_rank - 1), [axis]], name='transposition')
        else:
            transposition = array_ops.tensor_scatter_update(math_ops.range(rank), [[axis], [rank - 1]], [rank - 1, axis])
        top_k_input = array_ops.transpose(values, transposition)
    (values, indices) = nn_ops.top_k(top_k_input, k)
    return_value = indices if return_argsort else values
    if transposition is not None:
        return_value = array_ops.transpose(return_value, transposition)
    return return_value

def _ascending_sort(values, axis, return_argsort=False):
    if False:
        while True:
            i = 10
    'Sorts values in ascending order.\n\n  Args:\n    values: Tensor of numeric values.\n    axis: Index of the axis which values should be sorted along.\n    return_argsort: If False, return the sorted values. If True, return the\n      indices that would sort the values.\n\n  Returns:\n    The sorted values.\n  '
    dtype = values.dtype
    if dtype.is_unsigned:
        offset = dtype.max
        values_or_indices = _descending_sort(offset - values, axis, return_argsort)
        return values_or_indices if return_argsort else offset - values_or_indices
    elif dtype.is_integer:
        values_or_indices = _descending_sort(-values - 1, axis, return_argsort)
        return values_or_indices if return_argsort else -values_or_indices - 1
    else:
        values_or_indices = _descending_sort(-values, axis, return_argsort)
        return values_or_indices if return_argsort else -values_or_indices
_SORT_IMPL = {'ASCENDING': _ascending_sort, 'DESCENDING': _descending_sort}