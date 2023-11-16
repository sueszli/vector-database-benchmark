"""where operation for RaggedTensors."""
import typing
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import dispatch

@dispatch.dispatch_for_api(array_ops.where_v2)
def where_v2(condition: ragged_tensor.RaggedOrDense, x: typing.Optional[ragged_tensor.RaggedOrDense]=None, y: typing.Optional[ragged_tensor.RaggedOrDense]=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Return the elements where `condition` is `True`.\n\n  : If both `x` and `y` are None: Retrieve indices of true elements.\n\n    Returns the coordinates of true elements of `condition`. The coordinates\n    are returned in a 2-D tensor with shape\n    `[num_true_values, dim_size(condition)]`, where `result[i]` is the\n    coordinates of the `i`th true value (in row-major order).\n\n  : If both `x` and `y` are non-`None`: Multiplex between `x` and `y`.\n\n    Choose an output shape  from the shapes of `condition`, `x`, and `y` that\n    all three shapes are broadcastable to; and then use the broadcasted\n    `condition` tensor as a mask that chooses whether the corredsponding element\n    in the output should be taken from `x` (if `condition` is true) or `y` (if\n    `condition` is false).\n\n  >>> # Example: retrieve indices of true elements\n  >>> tf.where(tf.ragged.constant([[True, False], [True]]))\n  <tf.Tensor: shape=(2, 2), dtype=int64, numpy= array([[0, 0], [1, 0]])>\n\n  >>> # Example: multiplex between `x` and `y`\n  >>> tf.where(tf.ragged.constant([[True, False], [True, False, True]]),\n  ...          tf.ragged.constant([['A', 'B'], ['C', 'D', 'E']]),\n  ...          tf.ragged.constant([['a', 'b'], ['c', 'd', 'e']]))\n  <tf.RaggedTensor [[b'A', b'b'], [b'C', b'd', b'E']]>\n\n  Args:\n    condition: A potentially ragged tensor of type `bool`\n    x: A potentially ragged tensor (optional).\n    y: A potentially ragged tensor (optional).  Must be specified if `x` is\n      specified.  Must have the same rank and type as `x`.\n    name: A name of the operation (optional).\n\n  Returns:\n    : If both `x` and `y` are `None`:\n      A `Tensor` with shape `(num_true, rank(condition))`.\n    : Otherwise:\n      A potentially ragged tensor with the same type as `x` and `y`, and whose\n      shape is broadcast-compatible with `x`, `y`, and `condition`.\n\n  Raises:\n    ValueError: When exactly one of `x` or `y` is non-`None`; or when\n      `condition`, `x`, and `y` have incompatible shapes.\n  "
    if (x is None) != (y is None):
        raise ValueError('x and y must be either both None or both non-None')
    with ops.name_scope('RaggedWhere', name, [condition, x, y]):
        condition = ragged_tensor.convert_to_tensor_or_ragged_tensor(condition, name='condition')
        if x is None:
            return _coordinate_where(condition)
        else:
            x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
            y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, name='y')
            (condition, x, y) = ragged_tensor.match_row_splits_dtypes(condition, x, y)
            return _elementwise_where_v2(condition, x, y)

@dispatch.dispatch_for_api(array_ops.where)
def where(condition: ragged_tensor.RaggedOrDense, x: typing.Optional[ragged_tensor.RaggedOrDense]=None, y: typing.Optional[ragged_tensor.RaggedOrDense]=None, name=None):
    if False:
        return 10
    "Return the elements, either from `x` or `y`, depending on the `condition`.\n\n  : If both `x` and `y` are `None`:\n    Returns the coordinates of true elements of `condition`. The coordinates\n    are returned in a 2-D tensor with shape\n    `[num_true_values, dim_size(condition)]`, where `result[i]` is the\n    coordinates of the `i`th true value (in row-major order).\n\n  : If both `x` and `y` are non-`None`:\n    Returns a tensor formed by selecting values from `x` where condition is\n    true, and from `y` when condition is false.  In particular:\n\n    : If `condition`, `x`, and `y` all have the same shape:\n\n      * `result[i1...iN] = x[i1...iN]` if `condition[i1...iN]` is true.\n      * `result[i1...iN] = y[i1...iN]` if `condition[i1...iN]` is false.\n\n    : Otherwise:\n\n      * `condition` must be a vector.\n      * `x` and `y` must have the same number of dimensions.\n      * The outermost dimensions of `condition`, `x`, and `y` must all have the\n        same size.\n      * `result[i] = x[i]` if `condition[i]` is true.\n      * `result[i] = y[i]` if `condition[i]` is false.\n\n  Args:\n    condition: A potentially ragged tensor of type `bool`\n    x: A potentially ragged tensor (optional).\n    y: A potentially ragged tensor (optional).  Must be specified if `x` is\n      specified.  Must have the same rank and type as `x`.\n    name: A name of the operation (optional)\n\n  Returns:\n    : If both `x` and `y` are `None`:\n      A `Tensor` with shape `(num_true, dim_size(condition))`.\n    : Otherwise:\n      A potentially ragged tensor with the same type, rank, and outermost\n      dimension size as `x` and `y`.\n      `result.ragged_rank = max(x.ragged_rank, y.ragged_rank)`.\n\n  Raises:\n    ValueError: When exactly one of `x` or `y` is non-`None`; or when\n      `condition`, `x`, and `y` have incompatible shapes.\n\n  #### Examples:\n\n  >>> # Coordinates where condition is true.\n  >>> condition = tf.ragged.constant([[True, False, True], [False, True]])\n  >>> print(where(condition))\n  tf.Tensor( [[0 0] [0 2] [1 1]], shape=(3, 2), dtype=int64)\n\n  >>> # Elementwise selection between x and y, based on condition.\n  >>> condition = tf.ragged.constant([[True, False, True], [False, True]])\n  >>> x = tf.ragged.constant([['A', 'B', 'C'], ['D', 'E']])\n  >>> y = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e']])\n  >>> print(where(condition, x, y))\n  <tf.RaggedTensor [[b'A', b'b', b'C'], [b'd', b'E']]>\n\n  >>> # Row selection between x and y, based on condition.\n  >>> condition = [True, False]\n  >>> x = tf.ragged.constant([['A', 'B', 'C'], ['D', 'E']])\n  >>> y = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e']])\n  >>> print(where(condition, x, y))\n  <tf.RaggedTensor [[b'A', b'B', b'C'], [b'd', b'e']]>\n  "
    if (x is None) != (y is None):
        raise ValueError('x and y must be either both None or both non-None')
    with ops.name_scope('RaggedWhere', name, [condition, x, y]):
        condition = ragged_tensor.convert_to_tensor_or_ragged_tensor(condition, name='condition')
        if x is None:
            return _coordinate_where(condition)
        else:
            x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
            y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, name='y')
            (condition, x, y) = ragged_tensor.match_row_splits_dtypes(condition, x, y)
            return _elementwise_where(condition, x, y)

def _elementwise_where(condition, x, y):
    if False:
        i = 10
        return i + 15
    'Ragged version of tf.where(condition, x, y).'
    condition_is_ragged = isinstance(condition, ragged_tensor.RaggedTensor)
    x_is_ragged = isinstance(x, ragged_tensor.RaggedTensor)
    y_is_ragged = isinstance(y, ragged_tensor.RaggedTensor)
    if not (condition_is_ragged or x_is_ragged or y_is_ragged):
        return array_ops.where(condition, x, y)
    elif condition_is_ragged and x_is_ragged and y_is_ragged:
        return ragged_functional_ops.map_flat_values(array_ops.where, condition, x, y)
    elif not condition_is_ragged:
        condition.shape.assert_has_rank(1)
        x_and_y = ragged_concat_ops.concat([x, y], axis=0)
        x_nrows = _nrows(x, out_type=x_and_y.row_splits.dtype)
        y_nrows = _nrows(y, out_type=x_and_y.row_splits.dtype)
        indices = array_ops.where(condition, math_ops.range(x_nrows), x_nrows + math_ops.range(y_nrows))
        return ragged_gather_ops.gather(x_and_y, indices)
    else:
        raise ValueError('Input shapes do not match.')

def _elementwise_where_v2(condition, x, y):
    if False:
        return 10
    'Ragged version of tf.where_v2(condition, x, y).'
    if not (condition.shape.is_fully_defined() and x.shape.is_fully_defined() and y.shape.is_fully_defined() and (x.shape == y.shape) and (condition.shape == x.shape)):
        shape_c = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(condition)
        shape_x = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(x)
        shape_y = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(y)
        shape = ragged_tensor_shape.broadcast_dynamic_shape(shape_c, ragged_tensor_shape.broadcast_dynamic_shape(shape_x, shape_y))
        condition = ragged_tensor_shape.broadcast_to(condition, shape)
        x = ragged_tensor_shape.broadcast_to(x, shape)
        y = ragged_tensor_shape.broadcast_to(y, shape)
    condition_is_ragged = isinstance(condition, ragged_tensor.RaggedTensor)
    x_is_ragged = isinstance(x, ragged_tensor.RaggedTensor)
    y_is_ragged = isinstance(y, ragged_tensor.RaggedTensor)
    if not (condition_is_ragged or x_is_ragged or y_is_ragged):
        return array_ops.where_v2(condition, x, y)
    return ragged_functional_ops.map_flat_values(array_ops.where_v2, condition, x, y)

def _coordinate_where(condition):
    if False:
        return 10
    'Ragged version of tf.where(condition).'
    if not isinstance(condition, ragged_tensor.RaggedTensor):
        return array_ops.where(condition)
    selected_coords = _coordinate_where(condition.values)
    condition = condition.with_row_splits_dtype(selected_coords.dtype)
    first_index = selected_coords[:, 0]
    selected_rows = array_ops.gather(condition.value_rowids(), first_index)
    selected_row_starts = array_ops.gather(condition.row_splits, selected_rows)
    selected_cols = first_index - selected_row_starts
    return array_ops.concat([array_ops.expand_dims(selected_rows, 1), array_ops.expand_dims(selected_cols, 1), selected_coords[:, 1:]], axis=1)

def _nrows(rt_input, out_type):
    if False:
        return 10
    if isinstance(rt_input, ragged_tensor.RaggedTensor):
        return rt_input.nrows(out_type=out_type)
    else:
        return array_ops.shape(rt_input, out_type=out_type)[0]