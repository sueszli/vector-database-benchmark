"""Operator overloads for `RaggedTensor`."""
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator

def ragged_eq(self, other):
    if False:
        while True:
            i = 10
    'Returns result of elementwise `==` or False if not broadcast-compatible.\n\n  Compares two ragged tensors elemewise for equality if they are\n  broadcast-compatible; or returns False if they are not\n  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).\n\n  Note that this behavior differs from `tf.math.equal`, which raises an\n  exception if the two ragged tensors are not broadcast-compatible.\n\n  For example:\n\n  >>> rt1 = tf.ragged.constant([[1, 2], [3]])\n  >>> rt1 == rt1\n  <tf.RaggedTensor [[True, True], [True]]>\n\n  >>> rt2 = tf.ragged.constant([[1, 2], [4]])\n  >>> rt1 == rt2\n  <tf.RaggedTensor [[True, True], [False]]>\n\n  >>> rt3 = tf.ragged.constant([[1, 2], [3, 4]])\n  >>> # rt1 and rt3 are not broadcast-compatible.\n  >>> rt1 == rt3\n  False\n\n  >>> # You can also compare a `tf.RaggedTensor` to a `tf.Tensor`.\n  >>> t = tf.constant([[1, 2], [3, 4]])\n  >>> rt1 == t\n  False\n  >>> t == rt1\n  False\n  >>> rt4 = tf.ragged.constant([[1, 2], [3, 4]])\n  >>> rt4 == t\n  <tf.RaggedTensor [[True, True], [True, True]]>\n  >>> t == rt4\n  <tf.RaggedTensor [[True, True], [True, True]]>\n\n  Args:\n    other: The right-hand side of the `==` operator.\n\n  Returns:\n    The ragged tensor result of the elementwise `==` operation, or `False` if\n    the arguments are not broadcast-compatible.\n  '
    return math_ops.tensor_equals(self, other)

def ragged_ge(self, other):
    if False:
        while True:
            i = 10
    'Elementwise `>=` comparison of two convertible-to-ragged-tensor values.\n\n  Computes the elemewise `>=` comparison of two values that are convertible to\n  ragged tenors, with [broadcasting]\n  (http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) support.\n  Raises an exception if two values are not broadcast-compatible.\n\n  For example:\n\n  >>> rt1 = tf.ragged.constant([[1, 2], [3]])\n  >>> rt1 >= rt1\n  <tf.RaggedTensor [[True, True], [True]]>\n\n  >>> rt2 = tf.ragged.constant([[2, 1], [3]])\n  >>> rt1 >= rt2\n  <tf.RaggedTensor [[False, True], [True]]>\n\n  >>> rt3 = tf.ragged.constant([[1, 2], [3, 4]])\n  >>> # rt1 and rt3 are not broadcast-compatible.\n  >>> rt1 >= rt3\n  Traceback (most recent call last):\n  ...\n  InvalidArgumentError: ...\n\n  >>> # You can also compare a `tf.RaggedTensor` to a `tf.Tensor`.\n  >>> rt4 = tf.ragged.constant([[1, 2],[3, 4]])\n  >>> t1 = tf.constant([[2, 1], [4, 3]])\n  >>> rt4 >= t1\n  <tf.RaggedTensor [[False, True],\n   [False, True]]>\n  >>> t1 >= rt4\n  <tf.RaggedTensor [[True, False],\n   [True, False]]>\n\n  >>> # Compares a `tf.RaggedTensor` to a `tf.Tensor` with broadcasting.\n  >>> t2 = tf.constant([[2]])\n  >>> rt4 >= t2\n  <tf.RaggedTensor [[False, True],\n   [True, True]]>\n  >>> t2 >= rt4\n  <tf.RaggedTensor [[True, True],\n   [False, False]]>\n\n  Args:\n    other: The right-hand side of the `>=` operator.\n\n  Returns:\n    A `tf.RaggedTensor` of dtype `tf.bool` with the shape that `self` and\n    `other` broadcast to.\n\n  Raises:\n    InvalidArgumentError: If `self` and `other` are not broadcast-compatible.\n  '
    return math_ops.greater_equal(self, other)

def ragged_abs(self, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes the absolute value of a ragged tensor.\n\n  Given a ragged tensor of integer or floating-point values, this operation\n  returns a ragged tensor of the same type, where each element contains the\n  absolute value of the corresponding element in the input.\n\n  Given a ragged tensor `x` of complex numbers, this operation returns a tensor\n  of type `float32` or `float64` that is the absolute value of each element in\n  `x`. For a complex number \\\\(a + bj\\\\), its absolute value is computed as\n  \\\\(\\sqrt{a^2 + b^2}\\\\).\n\n  For example:\n\n  >>> # real number\n  >>> x = tf.ragged.constant([[-2.2, 3.2], [-4.2]])\n  >>> tf.abs(x)\n  <tf.RaggedTensor [[2.2, 3.2], [4.2]]>\n\n  >>> # complex number\n  >>> x = tf.ragged.constant([[-2.2 + 4.7j], [-3.2 + 5.7j], [-4.2 + 6.7j]])\n  >>> tf.abs(x)\n  <tf.RaggedTensor [[5.189412298131649],\n   [6.536818798161687],\n   [7.907591289387685]]>\n\n  Args:\n    name: A name for the operation (optional).\n\n  Returns:\n    A `RaggedTensor` of the same size and type as `x`, with absolute values.\n    Note, for `complex64` or `complex128` input, the returned `RaggedTensor`\n    will be of type `float32` or `float64`, respectively.\n  '
    return math_ops.abs(self, name=name)

def ragged_and(self, y, name=None):
    if False:
        print('Hello World!')
    'Returns the truth value of elementwise `x & y`.\n\n  Logical AND function.\n\n  Requires that `x` and `y` have the same shape or have\n  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)\n  shapes. For example, `y` can be:\n\n    - A single Python boolean, where the result will be calculated by applying\n      logical AND with the single element to each element in `x`.\n    - A `tf.Tensor` object of dtype `tf.bool` of the same shape or\n      [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)\n      shape. In this case, the result will be the element-wise logical AND of\n      `x` and `y`.\n    - A `tf.RaggedTensor` object of dtype `tf.bool` of the same shape or\n      [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)\n      shape. In this case, the result will be the element-wise logical AND of\n      `x` and `y`.\n\n  For example:\n\n  >>> # `y` is a Python boolean\n  >>> x = tf.ragged.constant([[True, False], [True]])\n  >>> y = True\n  >>> x & y\n  <tf.RaggedTensor [[True, False], [True]]>\n  >>> tf.math.logical_and(x, y)  # Equivalent of x & y\n  <tf.RaggedTensor [[True, False], [True]]>\n  >>> y & x\n  <tf.RaggedTensor [[True, False], [True]]>\n  >>> tf.math.reduce_all(x & y)  # Reduce to a scalar bool Tensor.\n  <tf.Tensor: shape=(), dtype=bool, numpy=False>\n\n  >>> # `y` is a tf.Tensor of the same shape.\n  >>> x = tf.ragged.constant([[True, False], [True, False]])\n  >>> y = tf.constant([[True, False], [False, True]])\n  >>> x & y\n  <tf.RaggedTensor [[True, False], [False, False]]>\n\n  >>> # `y` is a tf.Tensor of a broadcast-compatible shape.\n  >>> x = tf.ragged.constant([[True, False], [True]])\n  >>> y = tf.constant([[True], [False]])\n  >>> x & y\n  <tf.RaggedTensor [[True, False], [False]]>\n\n  >>> # `y` is a `tf.RaggedTensor` of the same shape.\n  >>> x = tf.ragged.constant([[True, False], [True]])\n  >>> y = tf.ragged.constant([[False, True], [True]])\n  >>> x & y\n  <tf.RaggedTensor [[False, False], [True]]>\n\n  >>> # `y` is a `tf.RaggedTensor` of a broadcast-compatible shape.\n  >>> x = tf.ragged.constant([[[True, True, False]], [[]], [[True, False]]])\n  >>> y = tf.ragged.constant([[[True]], [[True]], [[False]]], ragged_rank=1)\n  >>> x & y\n  <tf.RaggedTensor [[[True, True, False]], [[]], [[False, False]]]>\n\n  Args:\n    y: A Python boolean or a `tf.Tensor` or `tf.RaggedTensor` of dtype\n      `tf.bool`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `tf.RaggedTensor` of dtype `tf.bool` with the shape that `x` and `y`\n    broadcast to.\n  '
    return math_ops.logical_and(self, y, name)

def _right(operator):
    if False:
        print('Hello World!')
    'Right-handed version of an operator: swap args x and y.'
    return tf_decorator.make_decorator(operator, lambda y, x: operator(x, y))

def ragged_hash(self):
    if False:
        print('Hello World!')
    'The operation invoked by the `RaggedTensor.__hash__` operator.'
    g = getattr(self.row_splits, 'graph', None)
    if tensor.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions() and (g is None or g.building_function):
        raise TypeError('RaggedTensor is unhashable.')
    else:
        return id(self)
ragged_tensor.RaggedTensor.__getitem__ = ragged_getitem.ragged_tensor_getitem
ragged_tensor.RaggedTensor.__eq__ = ragged_eq
ragged_tensor.RaggedTensor.__ne__ = math_ops.tensor_not_equals
ragged_tensor.RaggedTensor.__hash__ = ragged_hash
ragged_tensor.RaggedTensor.__ge__ = ragged_ge
ragged_tensor.RaggedTensor.__gt__ = math_ops.greater
ragged_tensor.RaggedTensor.__le__ = math_ops.less_equal
ragged_tensor.RaggedTensor.__lt__ = math_ops.less
ragged_tensor.RaggedTensor.__and__ = ragged_and
ragged_tensor.RaggedTensor.__rand__ = _right(ragged_and)
ragged_tensor.RaggedTensor.__invert__ = math_ops.logical_not
ragged_tensor.RaggedTensor.__ror__ = _right(math_ops.logical_or)
ragged_tensor.RaggedTensor.__or__ = math_ops.logical_or
ragged_tensor.RaggedTensor.__xor__ = math_ops.logical_xor
ragged_tensor.RaggedTensor.__rxor__ = _right(math_ops.logical_xor)
ragged_tensor.RaggedTensor.__abs__ = ragged_abs
ragged_tensor.RaggedTensor.__add__ = math_ops.add
ragged_tensor.RaggedTensor.__radd__ = _right(math_ops.add)
ragged_tensor.RaggedTensor.__div__ = math_ops.div
ragged_tensor.RaggedTensor.__rdiv__ = _right(math_ops.div)
ragged_tensor.RaggedTensor.__floordiv__ = math_ops.floordiv
ragged_tensor.RaggedTensor.__rfloordiv__ = _right(math_ops.floordiv)
ragged_tensor.RaggedTensor.__mod__ = math_ops.floormod
ragged_tensor.RaggedTensor.__rmod__ = _right(math_ops.floormod)
ragged_tensor.RaggedTensor.__mul__ = math_ops.multiply
ragged_tensor.RaggedTensor.__rmul__ = _right(math_ops.multiply)
ragged_tensor.RaggedTensor.__neg__ = math_ops.negative
ragged_tensor.RaggedTensor.__pow__ = math_ops.pow
ragged_tensor.RaggedTensor.__rpow__ = _right(math_ops.pow)
ragged_tensor.RaggedTensor.__sub__ = math_ops.subtract
ragged_tensor.RaggedTensor.__rsub__ = _right(math_ops.subtract)
ragged_tensor.RaggedTensor.__truediv__ = math_ops.truediv
ragged_tensor.RaggedTensor.__rtruediv__ = _right(math_ops.truediv)

def ragged_bool(self):
    if False:
        i = 10
        return i + 15
    'Raises TypeError when a RaggedTensor is used as a Python bool.\n\n  To prevent RaggedTensor from being used as a bool, this function always raise\n  TypeError when being called.\n\n  For example:\n\n  >>> x = tf.ragged.constant([[1, 2], [3]])\n  >>> result = True if x else False  # Evaluate x as a bool value.\n  Traceback (most recent call last):\n  ...\n  TypeError: RaggedTensor may not be used as a boolean.\n\n  >>> x = tf.ragged.constant([[1]])\n  >>> r = (x == 1)  # tf.RaggedTensor [[True]]\n  >>> if r:  # Evaluate r as a bool value.\n  ...   pass\n  Traceback (most recent call last):\n  ...\n  TypeError: RaggedTensor may not be used as a boolean.\n  '
    raise TypeError('RaggedTensor may not be used as a boolean.')
ragged_tensor.RaggedTensor.__bool__ = ragged_bool
ragged_tensor.RaggedTensor.__nonzero__ = ragged_bool