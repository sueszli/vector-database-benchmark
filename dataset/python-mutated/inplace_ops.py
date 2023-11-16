"""Inplace operations.
"""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation

def _inplace_helper(x, i, v, op):
    if False:
        for i in range(10):
            print('nop')
    "Applies an inplace op on (x, i, v).\n\n  op is one of gen_array_ops.alias_inplace_update,\n  gen_array_ops.alias_inplace_add, or gen_array_ops.alias_inplace_sub.\n\n  If i is None, x and v must be the same shape. Computes\n    x op v;\n  If i is a scalar, x has a rank 1 higher than v's. Computes\n    x[i, :] op v;\n  Otherwise, x and v must have the same rank. Computes\n    x[i, :] op v;\n\n  Args:\n    x: A Tensor.\n    i: None, a scalar or a vector.\n    v: A Tensor.\n    op: alias_inplace_update, alias_inplace_add, or alias_inplace_sub.\n\n  Returns:\n    Returns x.\n\n  "
    x = ops.convert_to_tensor(x)
    v = ops.convert_to_tensor(v, x.dtype)
    if i is None:
        return array_ops.reshape(op(array_ops.reshape(x, [1, -1]), [0], array_ops.reshape(v, [1, -1])), array_ops.shape(x))
    i = math_ops.cast(i, dtypes.int32)
    if i.get_shape().ndims == 0:
        return op(x, array_ops.reshape(i, [1]), array_ops.expand_dims(v, 0))
    return op(x, i, v)

@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_update, which offers the same functionality with well-defined read-write semantics.')
def alias_inplace_update(x, i, v):
    if False:
        for i in range(10):
            print('nop')
    "Applies an inplace update on input x at index i with value v. Aliases x.\n\n  If i is None, x and v must be the same shape. Computes\n    x = v;\n  If i is a scalar, x has a rank 1 higher than v's. Computes\n    x[i, :] = v;\n  Otherwise, x and v must have the same rank. Computes\n    x[i, :] = v;\n\n  Args:\n    x: A Tensor.\n    i: None, a scalar or a vector.\n    v: A Tensor.\n\n  Returns:\n    Returns x.\n\n  "
    return _inplace_helper(x, i, v, gen_array_ops.inplace_update)

@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_add, which offers the same functionality with well-defined read-write semantics.')
def alias_inplace_add(x, i, v):
    if False:
        for i in range(10):
            print('nop')
    "Applies an inplace add on input x at index i with value v. Aliases x.\n\n  If i is None, x and v must be the same shape. Computes\n    x += v;\n  If i is a scalar, x has a rank 1 higher than v's. Computes\n    x[i, :] += v;\n  Otherwise, x and v must have the same rank. Computes\n    x[i, :] += v;\n\n  Args:\n    x: A Tensor.\n    i: None, a scalar or a vector.\n    v: A Tensor.\n\n  Returns:\n    Returns x.\n\n  "
    return _inplace_helper(x, i, v, gen_array_ops.inplace_add)

@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_sub, which offers the same functionality with well-defined read-write semantics.')
def alias_inplace_sub(x, i, v):
    if False:
        while True:
            i = 10
    "Applies an inplace sub on input x at index i with value v. Aliases x.\n\n  If i is None, x and v must be the same shape. Computes\n    x -= v;\n  If i is a scalar, x has a rank 1 higher than v's. Computes\n    x[i, :] -= v;\n  Otherwise, x and v must have the same rank. Computes\n    x[i, :] -= v;\n\n  Args:\n    x: A Tensor.\n    i: None, a scalar or a vector.\n    v: A Tensor.\n\n  Returns:\n    Returns x.\n\n  "
    return _inplace_helper(x, i, v, gen_array_ops.inplace_sub)

def empty_like(x, init=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a non-initialized tensor with the same shape and dtype as x.\n\n  Args:\n    x: A Tensor.\n    init: Initialize the returned tensor with the default value of\n      x.dtype(), if True. Otherwise, do not initialize. Defaults to\n      None.\n\n  Returns:\n    A tensor y, whose dtype and shape are the same as those of x.\n    y is guaranteed not to be an alias of x. Upon return, y may contain\n    arbitrary data.\n\n  '
    x = ops.convert_to_tensor(x)
    return gen_array_ops.empty(array_ops.shape(x), x.dtype, init=init)

@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_update, which offers the same functionality with well-defined read-write semantics.')
def inplace_update(x, i, v):
    if False:
        for i in range(10):
            print('nop')
    "Applies an inplace update on input x at index i with value v.\n\n  Note that this function is not actually inplace - it allocates\n  a copy of x.  The utility is not avoiding memory copies but rather\n  specifying a sparse update.\n\n  If i is None, x and v must be the same shape. Computes\n    y = x; y = v;\n  If i is a scalar, x has a rank 1 higher than v's. Computes\n    y = x; y[i, :] = v;\n  Otherwise, x and v must have the same rank. Computes\n    y = x; y[i, :] = v;\n\n  Args:\n    x: A Tensor.\n    i: None, a scalar or a vector.\n    v: A Tensor.\n\n  Returns:\n    Returns y, which is guaranteed not to be an alias of x.\n\n  "
    return alias_inplace_update(gen_array_ops.deep_copy(x), i, v)

@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_add, which offers the same functionality with well-defined read-write semantics.')
def inplace_add(x, i, v):
    if False:
        print('Hello World!')
    "Applies an inplace add on input x at index i with value v.\n\n  Note that this function is not actually inplace - it allocates\n  a copy of x.  The utility is not avoiding memory copies but rather\n  specifying a sparse update.\n\n  If i is None, x and v must be the same shape. Computes\n    y = x; y += v;\n  If i is a scalar, x has a rank 1 higher than v's. Computes\n    y = x; y[i, :] += v;\n  Otherwise, x and v must have the same rank. Computes\n    y = x; y[i, :] += v;\n\n  Args:\n    x: A Tensor.\n    i: None, a scalar or a vector.\n    v: A Tensor.\n\n  Returns:\n    Returns y, which is guaranteed not to be an alias of x.\n\n  "
    return alias_inplace_add(gen_array_ops.deep_copy(x), i, v)

@deprecation.deprecated(None, 'Prefer tf.tensor_scatter_nd_sub, which offers the same functionality with well-defined read-write semantics.')
def inplace_sub(x, i, v):
    if False:
        print('Hello World!')
    "Applies an inplace sub on input x at index i with value v.\n\n  Note that this function is not actually inplace - it allocates\n  a copy of x.  The utility is not avoiding memory copies but rather\n  specifying a sparse update.\n\n  If i is None, x and v must be the same shape. Computes\n    y = x; y -= v;\n  If i is a scalar, x has a rank 1 higher than v's. Computes\n    y = x; y[i, :] -= v;\n  Otherwise, x and v must have the same rank. Computes\n    y = x; y[i, :] -= v;\n\n  Args:\n    x: A Tensor.\n    i: None, a scalar or a vector.\n    v: A Tensor.\n\n  Returns:\n    Returns y, which is guaranteed not to be an alias of x.\n\n  "
    return alias_inplace_sub(gen_array_ops.deep_copy(x), i, v)
empty = gen_array_ops.empty