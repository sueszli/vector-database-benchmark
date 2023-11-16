"""Array difference ops."""
import tensorflow.compat.v2 as tf

def diff(x, order=1, exclusive=False, axis=-1, dtype=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes the difference between elements of an array at a regular interval.\n\n  For a difference along the final axis, if exclusive is True, then computes:\n\n  ```\n    result[..., i] = x[..., i+order] - x[..., i] for i < size(x) - order\n\n  ```\n\n  This is the same as doing `x[..., order:] - x[..., :-order]`. Note that in\n  this case the result `Tensor` is smaller in size than the input `Tensor`.\n\n  If exclusive is False, then computes:\n\n  ```\n    result[..., i] = x[..., i] - x[..., i-order] for i >= order\n    result[..., i] = x[..., i]  for 0 <= i < order\n\n  ```\n\n  #### Example\n\n  ```python\n    x = tf.constant([1, 2, 3, 4, 5])\n    dx = diff(x, order=1, exclusive=False)  # Returns [1, 1, 1, 1, 1]\n    dx1 = diff(x, order=1, exclusive=True)  # Returns [1, 1, 1, 1]\n    dx2 = diff(x, order=2, exclusive=False)  # Returns [1, 2, 2, 2, 2]\n  ```\n\n  Args:\n    x: A `Tensor` of shape `batch_shape + [n]` and of any dtype for which\n      arithmetic operations are permitted.\n    order: Positive Python int. The order of the difference to compute. `order =\n      1` corresponds to the difference between successive elements.\n      Default value: 1\n    exclusive: Python bool. See description above.\n      Default value: False\n    axis: Python int. The axis of `x` along which to difference.\n      Default value: -1 (the final axis).\n    dtype: Optional `tf.DType`. If supplied, the dtype for `x` to use when\n      converting to `Tensor`.\n      Default value: None which maps to the default dtype inferred by TF.\n    name: Python `str` name prefixed to Ops created by this class.\n      Default value: None which is mapped to the default name 'diff'.\n\n  Returns:\n    diffs: A `Tensor` of the same dtype as `x`. If `exclusive` is True,\n      then the shape is `batch_shape + [n-min(order, n)]`, otherwise it is\n      `batch_shape + [n]`. The final dimension of which contains the differences\n      of the requested order.\n  "
    with tf.name_scope(name or 'diff'):
        x = tf.convert_to_tensor(x, dtype=dtype)
        slices = x.shape.rank * [slice(None)]
        slices[axis] = slice(None, -order)
        x0 = x[slices]
        slices[axis] = slice(order, None)
        x1 = x[slices]
        exclusive_diff = x1 - x0
        if exclusive:
            return exclusive_diff
        slices[axis] = slice(None, order)
        return tf.concat([x[slices], exclusive_diff], axis=axis)