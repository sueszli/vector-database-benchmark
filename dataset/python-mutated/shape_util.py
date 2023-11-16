"""Tensor shape utilities."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

def shape_tensor(shape):
    if False:
        print('Hello World!')
    'Convert to an int32 or int64 tensor, defaulting to int32 if empty.'
    dtype = None
    if isinstance(shape, (tuple, list)):
        if not shape:
            dtype = dtypes.int32
        else:
            shape = tuple(map(tensor_shape.dimension_value, shape))
    return ops.convert_to_tensor(shape, dtype=dtype, name='shape')
_ENABLE_MAYBE_SET_STATIC_SHAPE = True

def maybe_set_static_shape(tensor, shape):
    if False:
        return 10
    "Sets the shape of `tensor` to the `shape`'s constant value, if inferrable.\n\n  This is a temporary workaround to fix shape inference across functional op\n  boundaries. E.g.\n\n  ```python\n  shape = tf.constant([3])\n  @tf.function\n  def f():\n    u = tf.random_uniform(shape)\n    return u\n  ```\n\n  If we were to rely solely on C++ shape inference, the shape of `u` inside\n  `f` would be unknown because C++ shape inference is not aware of the outer\n  graph and all it sees is a Placeholder node when backtracing the captured\n  tensor for `shape`. `maybe_set_static_shape` computes the static shape value\n  of `shape` by traversing the `FuncGraph` boundaries and sets the correct\n  shape.\n\n  A longer term solution would be to fix C++ shape inference.\n\n  Args:\n    tensor: A tensor.\n    shape: A shape tensor.\n  "
    if _ENABLE_MAYBE_SET_STATIC_SHAPE and (not context.executing_eagerly()) and ops.get_default_graph().building_function and (not tensor.shape.is_fully_defined()) and tensor_util.is_tensor(shape):
        shape = shape_tensor(shape)
        const_shape = tensor_util.constant_value_as_shape(shape)
        tensor.set_shape(const_shape)