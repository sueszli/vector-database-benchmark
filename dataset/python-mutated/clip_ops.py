"""Operations for clipping (gradient, weight) tensors to min/max values."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export

@tf_export('clip_by_value')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def clip_by_value(t, clip_value_min, clip_value_max, name=None):
    if False:
        return 10
    'Clips tensor values to a specified min and max.\n\n  Given a tensor `t`, this operation returns a tensor of the same type and\n  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.\n  Any values less than `clip_value_min` are set to `clip_value_min`. Any values\n  greater than `clip_value_max` are set to `clip_value_max`.\n\n  Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for\n  correct results.\n\n  For example:\n\n  Basic usage passes a scalar as the min and max value.\n\n  >>> t = tf.constant([[-10., -1., 0.], [0., 2., 10.]])\n  >>> t2 = tf.clip_by_value(t, clip_value_min=-1, clip_value_max=1)\n  >>> t2.numpy()\n  array([[-1., -1.,  0.],\n         [ 0.,  1.,  1.]], dtype=float32)\n\n  The min and max can be the same size as `t`, or broadcastable to that size.\n\n  >>> t = tf.constant([[-1, 0., 10.], [-1, 0, 10]])\n  >>> clip_min = [[2],[1]]\n  >>> t3 = tf.clip_by_value(t, clip_value_min=clip_min, clip_value_max=100)\n  >>> t3.numpy()\n  array([[ 2.,  2., 10.],\n         [ 1.,  1., 10.]], dtype=float32)\n\n  Broadcasting fails, intentionally, if you would expand the dimensions of `t`\n\n  >>> t = tf.constant([[-1, 0., 10.], [-1, 0, 10]])\n  >>> clip_min = [[[2, 1]]] # Has a third axis\n  >>> t4 = tf.clip_by_value(t, clip_value_min=clip_min, clip_value_max=100)\n  Traceback (most recent call last):\n  ...\n  InvalidArgumentError: Incompatible shapes: [2,3] vs. [1,1,2]\n\n  It throws a `TypeError` if you try to clip an `int` to a `float` value\n  (`tf.cast` the input to `float` first).\n\n  >>> t = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)\n  >>> t5 = tf.clip_by_value(t, clip_value_min=-3.1, clip_value_max=3.1)\n  Traceback (most recent call last):\n  ...\n  TypeError: Cannot convert ...\n\n\n  Args:\n    t: A `Tensor` or `IndexedSlices`.\n    clip_value_min: The minimum value to clip to. A scalar `Tensor` or one that\n      is broadcastable to the shape of `t`.\n    clip_value_max: The maximum value to clip to. A scalar `Tensor` or one that\n      is broadcastable to the shape of `t`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A clipped `Tensor` or `IndexedSlices`.\n\n  Raises:\n    `tf.errors.InvalidArgumentError`: If the clip tensors would trigger array\n      broadcasting that would make the returned tensor larger than the input.\n    TypeError: If dtype of the input is `int32` and dtype of\n      the `clip_value_min` or `clip_value_max` is `float32`\n  '
    with ops.name_scope(name, 'clip_by_value', [t, clip_value_min, clip_value_max]) as name:
        values = ops.convert_to_tensor(t.values if isinstance(t, indexed_slices.IndexedSlices) else t, name='t')
        t_min = math_ops.minimum(values, clip_value_max)
        values.shape.assert_is_compatible_with(t_min.shape)
        t_max = math_ops.maximum(t_min, clip_value_min, name=name)
        values.shape.assert_is_compatible_with(t_max.shape)
        if isinstance(t, indexed_slices.IndexedSlices):
            t_max = indexed_slices.IndexedSlices(t_max, t.indices, t.dense_shape)
    return t_max

@ops.RegisterGradient('ClipByValue')
def _clip_by_value_grad(op, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns grad of clip_by_value.'
    x = op.inputs[0]
    y = op.inputs[1]
    z = op.inputs[2]
    gdtype = grad.dtype
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    sz = array_ops.shape(z)
    gradshape = array_ops.shape(grad)
    zeros = array_ops.zeros(gradshape, gdtype)
    xymask = math_ops.less(x, y)
    xzmask = math_ops.greater(x, z)
    (_, ry) = gen_array_ops.broadcast_gradient_args(sx, sy)
    (_, rz) = gen_array_ops.broadcast_gradient_args(sx, sz)
    xgrad = array_ops.where(math_ops.logical_or(xymask, xzmask), zeros, grad)
    ygrad = array_ops.where(xymask, grad, zeros)
    zgrad = array_ops.where(xzmask, grad, zeros)
    gy = array_ops.reshape(math_ops.reduce_sum(ygrad, ry), sy)
    gz = array_ops.reshape(math_ops.reduce_sum(zgrad, rz), sz)
    return (xgrad, gy, gz)

@tf_export('clip_by_norm')
@dispatch.add_dispatch_support
def clip_by_norm(t, clip_norm, axes=None, name=None):
    if False:
        return 10
    'Clips tensor values to a maximum L2-norm.\n\n  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation\n  normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,\n  along the dimensions given in `axes`. Specifically, in the default case\n  where all dimensions are used for calculation, if the L2-norm of `t` is\n  already less than or equal to `clip_norm`, then `t` is not modified. If\n  the L2-norm is greater than `clip_norm`, then this operation returns a\n  tensor of the same type and shape as `t` with its values set to:\n\n  `t * clip_norm / l2norm(t)`\n\n  In this case, the L2-norm of the output tensor is `clip_norm`.\n\n  As another example, if `t` is a matrix and `axes == [1]`, then each row\n  of the output will have L2-norm less than or equal to `clip_norm`. If\n  `axes == [0]` instead, each column of the output will be clipped.\n\n  Code example:\n\n  >>> some_nums = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float32)\n  >>> tf.clip_by_norm(some_nums, 2.0).numpy()\n  array([[0.26967996, 0.5393599 , 0.80903983, 1.0787199 , 1.3483998 ]],\n        dtype=float32)\n\n  This operation is typically used to clip gradients before applying them with\n  an optimizer.  Most gradient data is a collection of different shaped tensors\n  for different parts of the model.  Thus, this is a common usage:\n\n  ```\n  # Get your gradients after training\n  loss_value, grads = grad(model, features, labels)\n\n  # Apply some clipping\n  grads = [tf.clip_by_norm(g, norm)\n               for g in grads]\n\n  # Continue on with training\n  optimizer.apply_gradients(grads)\n  ```\n\n  Args:\n    t: A `Tensor` or `IndexedSlices`.  This must be a floating point type.\n    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value, also\n      floating point.\n      Note: If a negative clip_norm is provided, it will be treated as zero.\n    axes: A 1-D (vector) `Tensor` of type int32 containing the dimensions to use\n      for computing the L2-norm. If `None` (the default), uses all dimensions.\n    name: A name for the operation (optional).\n\n  Returns:\n    A clipped `Tensor` or `IndexedSlices`.\n\n  Raises:\n    ValueError: If the clip_norm tensor is not a 0-D scalar tensor.\n    TypeError: If dtype of the input is not a floating point or\n      complex type.\n  '
    with ops.name_scope(name, 'clip_by_norm', [t, clip_norm]) as name:
        values = ops.convert_to_tensor(t.values if isinstance(t, indexed_slices.IndexedSlices) else t, name='t')
        if np.isscalar(clip_norm):
            if clip_norm < 0:
                clip_norm = 0
        else:
            clip_norm = math_ops.cast(math_ops.maximum(clip_norm, 0), dtype=values.dtype)
        l2sum = math_ops.reduce_sum(values * values, axes, keepdims=True)
        pred = l2sum > 0
        l2sum_safe = array_ops.where(pred, l2sum, array_ops.ones_like(l2sum))
        l2norm = array_ops.where(pred, math_ops.sqrt(l2sum_safe), l2sum)
        intermediate = values * clip_norm
        values.shape.assert_is_compatible_with(intermediate.shape)
        values_clip = array_ops.identity(intermediate / math_ops.maximum(l2norm, clip_norm), name=name)
        if isinstance(t, indexed_slices.IndexedSlices):
            return indexed_slices.IndexedSlices(values_clip, t.indices, t.dense_shape)
        return values_clip

@tf_export('linalg.global_norm', v1=['linalg.global_norm', 'global_norm'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('global_norm')
def global_norm(t_list, name=None):
    if False:
        i = 10
        return i + 15
    'Computes the global norm of multiple tensors.\n\n  Given a tuple or list of tensors `t_list`, this operation returns the\n  global norm of the elements in all tensors in `t_list`. The global norm is\n  computed as:\n\n  `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`\n\n  Any entries in `t_list` that are of type None are ignored.\n\n  Args:\n    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.\n    name: A name for the operation (optional).\n\n  Returns:\n    A 0-D (scalar) `Tensor` of type `float`.\n\n  Raises:\n    TypeError: If `t_list` is not a sequence.\n  '
    if not isinstance(t_list, collections_abc.Sequence) or isinstance(t_list, str):
        raise TypeError(f'`t_list` should be a sequence of tensors. Received {type(t_list)}.')
    t_list = list(t_list)
    with ops.name_scope(name, 'global_norm', t_list) as name:
        values = [ops.convert_to_tensor(t.values if isinstance(t, indexed_slices.IndexedSlices) else t, name='t_%d' % i) if t is not None else t for (i, t) in enumerate(t_list)]
        half_squared_norms = []
        for v in values:
            if v is not None:
                with ops.colocate_with(v):
                    half_squared_norms.append(gen_nn_ops.l2_loss(v))
        half_squared_norm = math_ops.reduce_sum(array_ops_stack.stack(half_squared_norms))
        norm = math_ops.sqrt(half_squared_norm * constant_op.constant(2.0, dtype=half_squared_norm.dtype), name='global_norm')
    return norm

@tf_export('clip_by_global_norm')
@dispatch.add_dispatch_support
def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):
    if False:
        while True:
            i = 10
    "Clips values of multiple tensors by the ratio of the sum of their norms.\n\n  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,\n  this operation returns a list of clipped tensors `list_clipped`\n  and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,\n  if you've already computed the global norm for `t_list`, you can specify\n  the global norm with `use_norm`.\n\n  To perform the clipping, the values `t_list[i]` are set to:\n\n      t_list[i] * clip_norm / max(global_norm, clip_norm)\n\n  where:\n\n      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))\n\n  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,\n  otherwise they're all shrunk by the global ratio.\n\n  If `global_norm == infinity` then the entries in `t_list` are all set to `NaN`\n  to signal that an error occurred.\n\n  Any of the entries of `t_list` that are of type `None` are ignored.\n\n  This is the correct way to perform gradient clipping (Pascanu et al., 2012).\n\n  However, it is slower than `clip_by_norm()` because all the parameters must be\n  ready before the clipping operation can be performed.\n\n  Args:\n    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.\n    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.\n    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global\n      norm to use. If not provided, `global_norm()` is used to compute the norm.\n    name: A name for the operation (optional).\n\n  Returns:\n    list_clipped: A list of `Tensors` of the same type as `list_t`.\n    global_norm: A 0-D (scalar) `Tensor` representing the global norm.\n\n  Raises:\n    TypeError: If `t_list` is not a sequence.\n\n  References:\n    On the difficulty of training Recurrent Neural Networks:\n      [Pascanu et al., 2012](http://proceedings.mlr.press/v28/pascanu13.html)\n      ([pdf](http://proceedings.mlr.press/v28/pascanu13.pdf))\n  "
    if not isinstance(t_list, collections_abc.Sequence) or isinstance(t_list, str):
        raise TypeError(f'`t_list` should be a sequence of tensors. Received {type(t_list)}.')
    t_list = list(t_list)
    if use_norm is None:
        use_norm = global_norm(t_list, name)
    with ops.name_scope(name, 'clip_by_global_norm', t_list + [clip_norm]) as name:
        scale_for_finite = clip_norm * math_ops.minimum(1.0 / use_norm, constant_op.constant(1.0, dtype=use_norm.dtype) / clip_norm)
        scale = scale_for_finite + (use_norm - use_norm)
        values = [ops.convert_to_tensor(t.values if isinstance(t, indexed_slices.IndexedSlices) else t, name='t_%d' % i) if t is not None else t for (i, t) in enumerate(t_list)]
        values_clipped = []
        for (i, v) in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with ops.colocate_with(v):
                    values_clipped.append(array_ops.identity(v * math_ops.cast(scale, v.dtype), name='%s_%d' % (name, i)))
        list_clipped = [indexed_slices.IndexedSlices(c_v, t.indices, t.dense_shape) if isinstance(t, indexed_slices.IndexedSlices) else c_v for (c_v, t) in zip(values_clipped, t_list)]
    return (list_clipped, use_norm)

@deprecation.deprecated(date=None, instructions='clip_by_average_norm is deprecated in TensorFlow 2.0. Please use clip_by_norm(t, clip_norm * tf.cast(tf.size(t), tf.float32), name) instead.')
@tf_export(v1=['clip_by_average_norm'])
@dispatch.add_dispatch_support
def clip_by_average_norm(t, clip_norm, name=None):
    if False:
        print('Hello World!')
    'Clips tensor values to a maximum average L2-norm.\n\n  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation\n  normalizes `t` so that its average L2-norm is less than or equal to\n  `clip_norm`. Specifically, if the average L2-norm is already less than or\n  equal to `clip_norm`, then `t` is not modified. If the average L2-norm is\n  greater than `clip_norm`, then this operation returns a tensor of the same\n  type and shape as `t` with its values set to:\n\n  `t * clip_norm / l2norm_avg(t)`\n\n  In this case, the average L2-norm of the output tensor is `clip_norm`.\n\n  This operation is typically used to clip gradients before applying them with\n  an optimizer.\n\n  Args:\n    t: A `Tensor`.\n    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.\n    name: A name for the operation (optional).\n\n  Returns:\n    A clipped `Tensor`.\n  '
    with ops.name_scope(name, 'clip_by_average_norm', [t, clip_norm]) as name:
        t = ops.convert_to_tensor(t, name='t')
        n_element = math_ops.cast(array_ops.size(t), dtypes.float32)
        l2norm_inv = math_ops.rsqrt(math_ops.reduce_sum(t * t, math_ops.range(array_ops.rank(t))))
        tclip = array_ops.identity(t * clip_norm * math_ops.minimum(l2norm_inv * n_element, constant_op.constant(1.0) / clip_norm), name=name)
    return tclip