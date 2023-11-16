"""Gradient functions for optional ops."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_optional_ops

@ops.RegisterGradient('OptionalFromValue')
def _OptionalFromValueGrad(op, grad):
    if False:
        return 10
    return gen_optional_ops.optional_get_value(grad, [t.dtype for t in op.inputs], [t.shape for t in op.inputs])

@ops.RegisterGradient('OptionalGetValue')
def _OptionalGetValueGrad(unused_op, *grads):
    if False:
        i = 10
        return i + 15
    return gen_optional_ops.optional_from_value(grads)