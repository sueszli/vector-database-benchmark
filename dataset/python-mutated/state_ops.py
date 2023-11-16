"""Variables.

See the [Variables](https://www.tensorflow.org/guide/variables) guide.
"""
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import state_grad
from tensorflow.python.ops.gen_state_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

def variable_op(shape, dtype, name='Variable', set_shape=True, container='', shared_name=''):
    if False:
        for i in range(10):
            print('nop')
    'Deprecated. Used variable_op_v2 instead.'
    if not set_shape:
        shape = tensor_shape.unknown_shape()
    ret = gen_state_ops.variable(shape=shape, dtype=dtype, name=name, container=container, shared_name=shared_name)
    if set_shape:
        ret.set_shape(shape)
    return ret

def variable_op_v2(shape, dtype, name='Variable', container='', shared_name=''):
    if False:
        print('Hello World!')
    'Create a variable Operation.\n\n  See also variables.Variable.\n\n  Args:\n    shape: The shape of the tensor managed by this variable\n    dtype: The underlying type of the tensor values.\n    name: optional name to use for the variable op.\n    container: An optional string. Defaults to "".\n      If non-empty, this variable is placed in the given container.\n      Otherwise, a default container is used.\n    shared_name: An optional string. Defaults to "".\n      If non-empty, this variable is named in the given bucket\n      with this shared_name. Otherwise, the node name is used instead.\n\n  Returns:\n    A variable tensor.\n  '
    return gen_state_ops.variable_v2(shape=shape, dtype=dtype, name=name, container=container, shared_name=shared_name)

def init_variable(v, init, name='init'):
    if False:
        return 10
    'Initializes variable with "init".\n\n  This op does the following:\n  if init is a Tensor, v = init\n  if callable(init): v = init(VariableShape(v), v.dtype)\n\n  Args:\n    v: Variable to initialize\n    init: Tensor to assign to v,\n      Or an object convertible to Tensor e.g. nparray,\n      Or an Initializer that generates a tensor given the shape and type of v.\n      An "Initializer" is a callable that returns a tensor that "v" should be\n      set to. It will be called as init(shape, dtype).\n    name: Optional name for the op.\n\n  Returns:\n    The operation that initializes v.\n  '
    with ops.name_scope(None, v.op.name + '/', [v, init]):
        with ops.name_scope(name) as scope:
            with ops.colocate_with(v):
                if callable(init):
                    assert v.get_shape().is_fully_defined(), 'Variable shape unknown.'
                    value = init(v.get_shape().as_list(), v.dtype.base_dtype)
                    value = ops.convert_to_tensor(value, name='value')
                    return gen_state_ops.assign(v, value, name=scope)
                else:
                    init = ops.convert_to_tensor(init, name='init')
                    return gen_state_ops.assign(v, init, name=scope)

def is_variable_initialized(ref, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether a tensor has been initialized.\n\n  Outputs boolean scalar indicating whether the tensor has been initialized.\n\n  Args:\n    ref: A mutable `Tensor`.\n      Should be from a `Variable` node. May be uninitialized.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `bool`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.is_variable_initialized(ref=ref, name=name)
    return ref.is_initialized(name=name)

@tf_export(v1=['assign_sub'])
def assign_sub(ref, value, use_locking=None, name=None):
    if False:
        i = 10
        return i + 15
    "Update `ref` by subtracting `value` from it.\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n  Unlike `tf.math.subtract`, this op does not broadcast. `ref` and `value`\n  must have the same shape.\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `float32`,\n      `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`,\n      `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`. Should be\n      from a `Variable` node.\n    value: A `Tensor`. Must have the same shape and dtype as `ref`. The value to\n      be subtracted to the variable.\n    use_locking: An optional `bool`. Defaults to `False`. If True, the\n      subtraction will be protected by a lock; otherwise the behavior is\n      undefined, but may exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    Same as `ref`.  Returned as a convenience for operations that want\n    to use the new value after the variable has been updated.\n\n  @compatibility(TF2)\n  `tf.compat.v1.assign_sub` is mostly compatible with eager\n  execution and `tf.function`.\n\n  To switch to the native TF2 style, one could use method 'assign_sub' of\n  `tf.Variable`:\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name    | Note                       |\n  | :-------------------- | :-------------- | :------------------------- |\n  | `ref`                 | `self`          | In `assign_sub()` method   |\n  | `value`               | `value`         | In `assign_sub()` method   |\n  | `use_locking`         | `use_locking`   | In `assign_sub()` method   |\n  | `name`                | `name`          | In `assign_sub()` method   |\n  | -                     | `read_value`    | Set to True to replicate   |\n  :                       :                 : behavior (True is default) :\n\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> with tf.Graph().as_default():\n  ...   with tf.compat.v1.Session() as sess:\n  ...     a = tf.compat.v1.Variable(1, dtype=tf.int64)\n  ...     sess.run(a.initializer)\n  ...     update_op = tf.compat.v1.assign_sub(a, 1)\n  ...     res_a = sess.run(update_op)\n  ...     res_a\n  0\n\n  After:\n\n  >>> b = tf.Variable(1, dtype=tf.int64)\n  >>> res_b = b.assign_sub(1)\n  >>> res_b.numpy()\n  0\n\n  @end_compatibility\n  "
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.assign_sub(ref, value, use_locking=use_locking, name=name)
    return ref.assign_sub(value)

@tf_export(v1=['assign_add'])
def assign_add(ref, value, use_locking=None, name=None):
    if False:
        while True:
            i = 10
    "Update `ref` by adding `value` to it.\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n  Unlike `tf.math.add`, this op does not broadcast. `ref` and `value` must have\n  the same shape.\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `float32`,\n      `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`,\n      `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`. Should be\n      from a `Variable` node.\n    value: A `Tensor`. Must have the same shape and dtype as `ref`. The value to\n      be added to the variable.\n    use_locking: An optional `bool`. Defaults to `False`. If True, the addition\n      will be protected by a lock; otherwise the behavior is undefined, but may\n      exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    Same as `ref`.  Returned as a convenience for operations that want\n    to use the new value after the variable has been updated.\n\n  @compatibility(TF2)\n  `tf.compat.v1.assign_add` is mostly compatible with eager\n  execution and `tf.function`.\n\n  To switch to the native TF2 style, one could use method 'assign_add' of\n  `tf.Variable`:\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name    | Note                       |\n  | :-------------------- | :-------------- | :------------------------- |\n  | `ref`                 | `self`          | In `assign_add()` method   |\n  | `value`               | `value`         | In `assign_add()` method   |\n  | `use_locking`         | `use_locking`   | In `assign_add()` method   |\n  | `name`                | `name`          | In `assign_add()` method   |\n  | -                     | `read_value`    | Set to True to replicate   |\n  :                       :                 : behavior (True is default) :\n\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> with tf.Graph().as_default():\n  ...   with tf.compat.v1.Session() as sess:\n  ...     a = tf.compat.v1.Variable(0, dtype=tf.int64)\n  ...     sess.run(a.initializer)\n  ...     update_op = tf.compat.v1.assign_add(a, 1)\n  ...     res_a = sess.run(update_op)\n  ...     res_a\n  1\n\n  After:\n\n  >>> b = tf.Variable(0, dtype=tf.int64)\n  >>> res_b = b.assign_add(1)\n  >>> res_b.numpy()\n  1\n\n  @end_compatibility\n  "
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.assign_add(ref, value, use_locking=use_locking, name=name)
    return ref.assign_add(value)

@tf_export(v1=['assign'])
def assign(ref, value, validate_shape=None, use_locking=None, name=None):
    if False:
        print('Hello World!')
    "Update `ref` by assigning `value` to it.\n\n  This operation outputs a Tensor that holds the new value of `ref` after\n  the value has been assigned. This makes it easier to chain operations that\n  need to use the reset value.\n\n  Args:\n    ref: A mutable `Tensor`. Should be from a `Variable` node. May be\n      uninitialized.\n    value: A `Tensor`. Must have the same shape and dtype as `ref`. The value to\n      be assigned to the variable.\n    validate_shape: An optional `bool`. Defaults to `True`. If true, the\n      operation will validate that the shape of 'value' matches the shape of the\n      Tensor being assigned to.  If false, 'ref' will take on the shape of\n      'value'.\n    use_locking: An optional `bool`. Defaults to `True`. If True, the assignment\n      will be protected by a lock; otherwise the behavior is undefined, but may\n      exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` that will hold the new value of `ref` after\n      the assignment has completed.\n\n  @compatibility(TF2)\n  `tf.compat.v1.assign` is mostly compatible with eager\n  execution and `tf.function`. However, argument 'validate_shape' will be\n  ignored. To avoid shape validation, set 'shape' to tf.TensorShape(None) when\n  constructing the variable:\n\n  >>> import tensorflow as tf\n  >>> a = tf.Variable([1], shape=tf.TensorShape(None))\n  >>> tf.compat.v1.assign(a, [2,3])\n\n  To switch to the native TF2 style, one could use method 'assign' of\n  `tf.Variable`:\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name    | Note                       |\n  | :-------------------- | :-------------- | :------------------------- |\n  | `ref`                 | `self`          | In `assign()` method       |\n  | `value`               | `value`         | In `assign()` method       |\n  | `validate_shape`      | Not supported   | Specify `shape` in the     |\n  :                       :                 : constructor to replicate   :\n  :                       :                 : behavior                   :\n  | `use_locking`         | `use_locking`   | In `assign()` method       |\n  | `name`                | `name`          | In `assign()` method       |\n  | -                     | `read_value`    | Set to True to replicate   |\n  :                       :                 : behavior (True is default) :\n  @end_compatibility\n\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> with tf.Graph().as_default():\n  ...   with tf.compat.v1.Session() as sess:\n  ...     a = tf.compat.v1.Variable(0, dtype=tf.int64)\n  ...     sess.run(a.initializer)\n  ...     update_op = tf.compat.v1.assign(a, 2)\n  ...     res_a = sess.run(update_op)\n  ...     res_a\n  2\n\n  After:\n\n  >>> b = tf.Variable(0, dtype=tf.int64)\n  >>> res_b = b.assign(2)\n  >>> res_b.numpy()\n  2\n  "
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.assign(ref, value, use_locking=use_locking, name=name, validate_shape=validate_shape)
    return ref.assign(value, name=name)

@tf_export(v1=['count_up_to'])
@deprecated(None, 'Prefer Dataset.range instead.')
def count_up_to(ref, limit, name=None):
    if False:
        i = 10
        return i + 15
    "Increments 'ref' until it reaches 'limit'.\n\n  Args:\n    ref: A Variable. Must be one of the following types: `int32`, `int64`.\n      Should be from a scalar `Variable` node.\n    limit: An `int`.\n      If incrementing ref would bring it above limit, instead generates an\n      'OutOfRange' error.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor`. Has the same type as `ref`.\n    A copy of the input before increment. If nothing else modifies the\n    input, the values produced will all be distinct.\n  "
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.count_up_to(ref, limit=limit, name=name)
    return gen_state_ops.resource_count_up_to(ref.handle, limit, T=ref.dtype, name=name)

@tf_export(v1=['scatter_update'])
def scatter_update(ref, indices, updates, use_locking=True, name=None):
    if False:
        print('Hello World!')
    'Applies sparse updates to a variable reference.\n\n  This operation computes\n\n  ```python\n      # Scalar indices\n      ref[indices, ...] = updates[...]\n\n      # Vector indices (for each i)\n      ref[indices[i], ...] = updates[i, ...]\n\n      # High rank indices (for each i, ..., j)\n      ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]\n  ```\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n\n  If values in `ref` is to be updated more than once, because there are\n  duplicate entries in `indices`, the order at which the updates happen\n  for each value is undefined.\n\n  Requires `updates.shape = indices.shape + ref.shape[1:]`.\n\n  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">\n  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterUpdate.png" alt>\n  </div>\n\n  Args:\n    ref: A `Variable`.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.\n      A tensor of indices into the first dimension of `ref`.\n    updates: A `Tensor`. Must have the same type as `ref`.\n      A tensor of updated values to store in `ref`.\n    use_locking: An optional `bool`. Defaults to `True`.\n      If True, the assignment will be protected by a lock;\n      otherwise the behavior is undefined, but may exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    Same as `ref`.  Returned as a convenience for operations that want\n    to use the updated values after the update is done.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_update(ref, indices, updates, use_locking=use_locking, name=name)
    return ref._lazy_read(gen_resource_variable_ops.resource_scatter_update(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_nd_update'])
def scatter_nd_update(ref, indices, updates, use_locking=True, name=None):
    if False:
        return 10
    'Applies sparse `updates` to individual values or slices in a Variable.\n\n  `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n  `indices` must be integer tensor, containing indices into `ref`.\n  It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n  The innermost dimension of `indices` (with length `K`) corresponds to\n  indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n  dimension of `ref`.\n\n  `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n  ```\n  [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].\n  ```\n\n  For example, say we want to update 4 scattered elements to a rank-1 tensor to\n  8 elements. In Python, that update would look like this:\n\n  ```python\n      ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n      indices = tf.constant([[4], [3], [1] ,[7]])\n      updates = tf.constant([9, 10, 11, 12])\n      update = tf.compat.v1.scatter_nd_update(ref, indices, updates)\n      with tf.compat.v1.Session() as sess:\n        print sess.run(update)\n  ```\n\n  The resulting update to ref would look like this:\n\n      [1, 11, 3, 10, 9, 6, 7, 12]\n\n  See `tf.scatter_nd` for more details about how to make updates to\n  slices.\n\n  Args:\n    ref: A Variable.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.\n      A tensor of indices into ref.\n    updates: A `Tensor`. Must have the same type as `ref`.\n      A Tensor. Must have the same type as ref. A tensor of updated\n      values to add to ref.\n    use_locking: An optional `bool`. Defaults to `True`.\n      An optional bool. Defaults to True. If True, the assignment will\n      be protected by a lock; otherwise the behavior is undefined,\n      but may exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    The value of the variable after the update.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_nd_update(ref, indices, updates, use_locking, name)
    return ref._lazy_read(gen_state_ops.resource_scatter_nd_update(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_add'])
def scatter_add(ref, indices, updates, use_locking=False, name=None):
    if False:
        i = 10
        return i + 15
    'Adds sparse updates to the variable referenced by `resource`.\n\n  This operation computes\n\n  ```python\n      # Scalar indices\n      ref[indices, ...] += updates[...]\n\n      # Vector indices (for each i)\n      ref[indices[i], ...] += updates[i, ...]\n\n      # High rank indices (for each i, ..., j)\n      ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]\n  ```\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the updated value.\n  Duplicate entries are handled correctly: if multiple `indices` reference\n  the same location, their contributions add.\n\n  Requires `updates.shape = indices.shape + ref.shape[1:]`.\n\n  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">\n  <img style="width:100%" src=\'https://www.tensorflow.org/images/ScatterAdd.png\' alt>\n  </div>\n\n  Args:\n    ref: A `Variable`.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.\n      A tensor of indices into the first dimension of `ref`.\n    updates: A `Tensor`. Must have the same type as `ref`.\n      A tensor of updated values to store in `ref`.\n    use_locking: An optional `bool`. Defaults to `False`.\n      If True, the assignment will be protected by a lock;\n      otherwise the behavior is undefined, but may exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    Same as `ref`.  Returned as a convenience for operations that want\n    to use the updated values after the update is done.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_add(ref, indices, updates, use_locking=use_locking, name=name)
    return ref._lazy_read(gen_resource_variable_ops.resource_scatter_add(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_nd_add'])
def scatter_nd_add(ref, indices, updates, use_locking=False, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Applies sparse addition to individual values or slices in a Variable.\n\n  `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n  `indices` must be integer tensor, containing indices into `ref`.\n  It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n  The innermost dimension of `indices` (with length `K`) corresponds to\n  indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n  dimension of `ref`.\n\n  `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n  ```\n  [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]\n  ```\n\n  For example, say we want to add 4 scattered elements to a rank-1 tensor to\n  8 elements. In Python, that addition would look like this:\n\n  ```python\n  ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n  indices = tf.constant([[4], [3], [1], [7]])\n  updates = tf.constant([9, 10, 11, 12])\n  add = tf.compat.v1.scatter_nd_add(ref, indices, updates)\n  with tf.compat.v1.Session() as sess:\n    print sess.run(add)\n  ```\n\n  The resulting update to ref would look like this:\n\n      [1, 13, 3, 14, 14, 6, 7, 20]\n\n  See `tf.scatter_nd` for more details about how to make updates to\n  slices.\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `float32`,\n      `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`,\n      `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`,\n      `uint32`, `uint64`. A mutable Tensor. Should be from a Variable node.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.\n      A tensor of indices into ref.\n    updates: A `Tensor`. Must have the same type as `ref`.\n      A tensor of updated values to add to ref.\n    use_locking: An optional `bool`. Defaults to `False`.\n      If True, the assignment will be protected by a lock;\n      otherwise the behavior is undefined, but may exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A mutable `Tensor`. Has the same type as `ref`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_nd_add(ref, indices, updates, use_locking, name)
    return ref._lazy_read(gen_state_ops.resource_scatter_nd_add(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_sub'])
def scatter_sub(ref, indices, updates, use_locking=False, name=None):
    if False:
        i = 10
        return i + 15
    'Subtracts sparse updates to a variable reference.\n\n  ```python\n      # Scalar indices\n      ref[indices, ...] -= updates[...]\n\n      # Vector indices (for each i)\n      ref[indices[i], ...] -= updates[i, ...]\n\n      # High rank indices (for each i, ..., j)\n      ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]\n  ```\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n\n  Duplicate entries are handled correctly: if multiple `indices` reference\n  the same location, their (negated) contributions add.\n\n  Requires `updates.shape = indices.shape + ref.shape[1:]` or\n  `updates.shape = []`.\n\n  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">\n  <img style="width:100%"\n       src="https://www.tensorflow.org/images/ScatterSub.png" alt>\n  </div>\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `float32`,\n      `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`,\n      `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`,\n      `uint32`, `uint64`. Should be from a `Variable` node.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.\n      A tensor of indices into the first dimension of `ref`.\n    updates: A `Tensor`. Must have the same type as `ref`.\n      A tensor of updated values to subtract from `ref`.\n    use_locking: An optional `bool`. Defaults to `False`.\n      If True, the subtraction will be protected by a lock;\n      otherwise the behavior is undefined, but may exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A mutable `Tensor`. Has the same type as `ref`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_sub(ref, indices, updates, use_locking=use_locking, name=name)
    return ref._lazy_read(gen_resource_variable_ops.resource_scatter_sub(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_nd_sub'])
def scatter_nd_sub(ref, indices, updates, use_locking=False, name=None):
    if False:
        i = 10
        return i + 15
    'Applies sparse subtraction to individual values or slices in a Variable.\n\n  `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.\n\n  `indices` must be integer tensor, containing indices into `ref`.\n  It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.\n\n  The innermost dimension of `indices` (with length `K`) corresponds to\n  indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th\n  dimension of `ref`.\n\n  `updates` is `Tensor` of rank `Q-1+P-K` with shape:\n\n  ```\n  [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]\n  ```\n\n  For example, say we want to subtract 4 scattered elements from a rank-1 tensor\n  with 8 elements. In Python, that update would look like this:\n\n  ```python\n  ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])\n  indices = tf.constant([[4], [3], [1] ,[7]])\n  updates = tf.constant([9, 10, 11, 12])\n  op = tf.compat.v1.scatter_nd_sub(ref, indices, updates)\n  with tf.compat.v1.Session() as sess:\n    print sess.run(op)\n  ```\n\n  The resulting update to ref would look like this:\n\n      [1, -9, 3, -6, -6, 6, 7, -4]\n\n  See `tf.scatter_nd` for more details about how to make updates to\n  slices.\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `float32`,\n      `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`,\n      `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`,\n      `uint32`, `uint64`. A mutable Tensor. Should be from a Variable node.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.\n      A tensor of indices into ref.\n    updates: A `Tensor`. Must have the same type as `ref`.\n      A tensor of updated values to add to ref.\n    use_locking: An optional `bool`. Defaults to `False`.\n      An optional bool. Defaults to True. If True, the assignment will\n      be protected by a lock; otherwise the behavior is undefined,\n      but may exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A mutable `Tensor`. Has the same type as `ref`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_nd_sub(ref, indices, updates, use_locking, name)
    return ref._lazy_read(gen_state_ops.resource_scatter_nd_sub(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_mul'])
def scatter_mul(ref, indices, updates, use_locking=False, name=None):
    if False:
        i = 10
        return i + 15
    'Multiplies sparse updates into a variable reference.\n\n  This operation computes\n\n  ```python\n      # Scalar indices\n      ref[indices, ...] *= updates[...]\n\n      # Vector indices (for each i)\n      ref[indices[i], ...] *= updates[i, ...]\n\n      # High rank indices (for each i, ..., j)\n      ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]\n  ```\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n\n  Duplicate entries are handled correctly: if multiple `indices` reference\n  the same location, their contributions multiply.\n\n  Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape =\n  []`.\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `float32`,\n      `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`,\n      `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`,\n      `uint32`, `uint64`. Should be from a `Variable` node.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`. A\n      tensor of indices into the first dimension of `ref`.\n    updates: A `Tensor`. Must have the same type as `ref`. A tensor of updated\n      values to multiply to `ref`.\n    use_locking: An optional `bool`. Defaults to `False`. If True, the operation\n      will be protected by a lock; otherwise the behavior is undefined, but may\n      exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A mutable `Tensor`. Has the same type as `ref`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_mul(ref, indices, updates, use_locking=use_locking, name=name)
    return ref._lazy_read(gen_resource_variable_ops.resource_scatter_mul(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_div'])
def scatter_div(ref, indices, updates, use_locking=False, name=None):
    if False:
        i = 10
        return i + 15
    'Divides a variable reference by sparse updates.\n\n  This operation computes\n\n  ```python\n      # Scalar indices\n      ref[indices, ...] /= updates[...]\n\n      # Vector indices (for each i)\n      ref[indices[i], ...] /= updates[i, ...]\n\n      # High rank indices (for each i, ..., j)\n      ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]\n  ```\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n\n  Duplicate entries are handled correctly: if multiple `indices` reference\n  the same location, their contributions divide.\n\n  Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape =\n  []`.\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `float32`,\n      `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`,\n      `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`,\n      `uint32`, `uint64`. Should be from a `Variable` node.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`. A\n      tensor of indices into the first dimension of `ref`.\n    updates: A `Tensor`. Must have the same type as `ref`. A tensor of values\n      that `ref` is divided by.\n    use_locking: An optional `bool`. Defaults to `False`. If True, the operation\n      will be protected by a lock; otherwise the behavior is undefined, but may\n      exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A mutable `Tensor`. Has the same type as `ref`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_div(ref, indices, updates, use_locking=use_locking, name=name)
    return ref._lazy_read(gen_resource_variable_ops.resource_scatter_div(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_max'])
def scatter_max(ref, indices, updates, use_locking=False, name=None):
    if False:
        return 10
    'Reduces sparse updates into a variable reference using the `max` operation.\n\n  This operation computes\n\n      # Scalar indices\n      ref[indices, ...] = max(ref[indices, ...], updates[...])\n\n      # Vector indices (for each i)\n      ref[indices[i], ...] = max(ref[indices[i], ...], updates[i, ...])\n\n      # High rank indices (for each i, ..., j)\n      ref[indices[i, ..., j], ...] = max(ref[indices[i, ..., j], ...],\n      updates[i, ..., j, ...])\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n\n  Duplicate entries are handled correctly: if multiple `indices` reference\n  the same location, their contributions combine.\n\n  Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape =\n  []`.\n\n  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">\n  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png"\n  alt>\n  </div>\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `half`,\n      `bfloat16`, `float32`, `float64`, `int32`, `int64`. Should be from a\n      `Variable` node.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`. A\n      tensor of indices into the first dimension of `ref`.\n    updates: A `Tensor`. Must have the same type as `ref`. A tensor of updated\n      values to reduce into `ref`.\n    use_locking: An optional `bool`. Defaults to `False`. If True, the update\n      will be protected by a lock; otherwise the behavior is undefined, but may\n      exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A mutable `Tensor`. Has the same type as `ref`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_max(ref, indices, updates, use_locking=use_locking, name=name)
    return ref._lazy_read(gen_resource_variable_ops.resource_scatter_max(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['scatter_min'])
def scatter_min(ref, indices, updates, use_locking=False, name=None):
    if False:
        while True:
            i = 10
    'Reduces sparse updates into a variable reference using the `min` operation.\n\n  This operation computes\n\n      # Scalar indices\n      ref[indices, ...] = min(ref[indices, ...], updates[...])\n\n      # Vector indices (for each i)\n      ref[indices[i], ...] = min(ref[indices[i], ...], updates[i, ...])\n\n      # High rank indices (for each i, ..., j)\n      ref[indices[i, ..., j], ...] = min(ref[indices[i, ..., j], ...],\n      updates[i, ..., j, ...])\n\n  This operation outputs `ref` after the update is done.\n  This makes it easier to chain operations that need to use the reset value.\n\n  Duplicate entries are handled correctly: if multiple `indices` reference\n  the same location, their contributions combine.\n\n  Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape =\n  []`.\n\n  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">\n  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png"\n  alt>\n  </div>\n\n  Args:\n    ref: A mutable `Tensor`. Must be one of the following types: `half`,\n      `bfloat16`, `float32`, `float64`, `int32`, `int64`. Should be from a\n      `Variable` node.\n    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`. A\n      tensor of indices into the first dimension of `ref`.\n    updates: A `Tensor`. Must have the same type as `ref`. A tensor of updated\n      values to reduce into `ref`.\n    use_locking: An optional `bool`. Defaults to `False`. If True, the update\n      will be protected by a lock; otherwise the behavior is undefined, but may\n      exhibit less contention.\n    name: A name for the operation (optional).\n\n  Returns:\n    A mutable `Tensor`. Has the same type as `ref`.\n  '
    if ref.dtype._is_ref_dtype:
        return gen_state_ops.scatter_min(ref, indices, updates, use_locking=use_locking, name=name)
    return ref._lazy_read(gen_resource_variable_ops.resource_scatter_min(ref.handle, indices, ops.convert_to_tensor(updates, ref.dtype), name=name))

@tf_export(v1=['batch_scatter_update'])
@deprecation.deprecated('2018-11-29', 'Use the batch_scatter_update method of Variable instead.')
def batch_scatter_update(ref, indices, updates, use_locking=True, name=None):
    if False:
        i = 10
        return i + 15
    'Generalization of `tf.compat.v1.scatter_update` to axis different than 0.\n\n  Analogous to `batch_gather`. This assumes that `ref`, `indices` and `updates`\n  have a series of leading dimensions that are the same for all of them, and the\n  updates are performed on the last dimension of indices. In other words, the\n  dimensions should be the following:\n\n  `num_prefix_dims = indices.ndims - 1`\n  `batch_dim = num_prefix_dims + 1`\n  `updates.shape = indices.shape + var.shape[batch_dim:]`\n\n  where\n\n  `updates.shape[:num_prefix_dims]`\n  `== indices.shape[:num_prefix_dims]`\n  `== var.shape[:num_prefix_dims]`\n\n  And the operation performed can be expressed as:\n\n  `var[i_1, ..., i_n, indices[i_1, ..., i_n, j]] = updates[i_1, ..., i_n, j]`\n\n  When indices is a 1D tensor, this operation is equivalent to\n  `tf.compat.v1.scatter_update`.\n\n  To avoid this operation there would be 2 alternatives:\n  1) Reshaping the variable by merging the first `ndims` dimensions. However,\n     this is not possible because `tf.reshape` returns a Tensor, which we\n     cannot use `tf.compat.v1.scatter_update` on.\n  2) Looping over the first `ndims` of the variable and using\n     `tf.compat.v1.scatter_update` on the subtensors that result of slicing the\n     first\n     dimension. This is a valid option for `ndims = 1`, but less efficient than\n     this implementation.\n\n  See also `tf.compat.v1.scatter_update` and `tf.compat.v1.scatter_nd_update`.\n\n  Args:\n    ref: `Variable` to scatter onto.\n    indices: Tensor containing indices as described above.\n    updates: Tensor of updates to apply to `ref`.\n    use_locking: Boolean indicating whether to lock the writing operation.\n    name: Optional scope name string.\n\n  Returns:\n    Ref to `variable` after it has been modified.\n\n  Raises:\n    ValueError: If the initial `ndims` of `ref`, `indices`, and `updates` are\n        not the same.\n  '
    with ops.name_scope(name):
        indices = ops.convert_to_tensor(indices, name='indices')
        indices_shape = array_ops.shape(indices)
        indices_dimensions = indices.get_shape().ndims
        if indices_dimensions is None:
            raise ValueError('batch_gather does not allow indices with unknown shape.')
        nd_indices = array_ops.expand_dims(indices, axis=-1)
        nd_indices_list = []
        for dimension in range(indices_dimensions - 1):
            dimension_size = indices_shape[dimension]
            shape_to_broadcast = [1] * (indices_dimensions + 1)
            shape_to_broadcast[dimension] = dimension_size
            dimension_range = array_ops.reshape(gen_math_ops._range(0, dimension_size, 1), shape_to_broadcast)
            if dimension_range.dtype.base_dtype != nd_indices.dtype:
                dimension_range = gen_math_ops.cast(dimension_range, nd_indices.dtype)
            nd_indices_list.append(dimension_range * array_ops.ones_like(nd_indices))
        nd_indices_list.append(nd_indices)
        final_indices = array_ops.concat(nd_indices_list, axis=-1)
        return scatter_nd_update(ref, final_indices, updates, use_locking=use_locking)