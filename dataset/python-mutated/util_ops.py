"""Utility ops shared across tf.contrib.signal."""
import fractions
import math
import sys
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop

def gcd(a, b, name=None):
    if False:
        return 10
    "Returns the greatest common divisor via Euclid's algorithm.\n\n  Args:\n    a: The dividend. A scalar integer `Tensor`.\n    b: The divisor. A scalar integer `Tensor`.\n    name: An optional name for the operation.\n\n  Returns:\n    A scalar `Tensor` representing the greatest common divisor between `a` and\n    `b`.\n\n  Raises:\n    ValueError: If `a` or `b` are not scalar integers.\n  "
    with ops.name_scope(name, 'gcd', [a, b]):
        a = ops.convert_to_tensor(a)
        b = ops.convert_to_tensor(b)
        a.shape.assert_has_rank(0)
        b.shape.assert_has_rank(0)
        if not a.dtype.is_integer:
            raise ValueError('a must be an integer type. Got: %s' % a.dtype)
        if not b.dtype.is_integer:
            raise ValueError('b must be an integer type. Got: %s' % b.dtype)
        const_a = tensor_util.constant_value(a)
        const_b = tensor_util.constant_value(b)
        if const_a is not None and const_b is not None:
            if sys.version_info.major < 3:
                math_gcd = fractions.gcd
            else:
                math_gcd = math.gcd
            return ops.convert_to_tensor(math_gcd(const_a, const_b))
        cond = lambda _, b: math_ops.greater(b, array_ops.zeros_like(b))
        body = lambda a, b: [b, math_ops.mod(a, b)]
        (a, b) = while_loop.while_loop(cond, body, [a, b], back_prop=False)
        return a