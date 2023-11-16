"""The gradient of the tutorial zero_out op."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient('ZeroOut')
def _zero_out_grad(op, grad):
    if False:
        i = 10
        return i + 15
    'The gradients for `zero_out`.\n\n  Args:\n    op: The `zero_out` `Operation` that we are differentiating, which we can use\n      to find the inputs and outputs of the original op.\n    grad: Gradient with respect to the output of the `zero_out` op.\n\n  Returns:\n    Gradients with respect to the input of `zero_out`.\n  '
    to_zero = op.inputs[0]
    shape = array_ops.shape(to_zero)
    index = array_ops.zeros_like(shape)
    first_grad = array_ops.reshape(grad, [-1])[0]
    to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
    return [to_zero_grad]