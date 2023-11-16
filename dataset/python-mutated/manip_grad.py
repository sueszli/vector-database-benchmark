"""Gradients for operators defined in manip_ops.py."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import manip_ops

@ops.RegisterGradient('Roll')
def _RollGrad(op, grad):
    if False:
        for i in range(10):
            print('nop')
    shift = op.inputs[1]
    axis = op.inputs[2]
    roll_grad = manip_ops.roll(grad, -shift, axis)
    return (roll_grad, None, None)