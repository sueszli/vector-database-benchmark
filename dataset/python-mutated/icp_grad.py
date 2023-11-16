"""The gradient of the icp op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops

@ops.RegisterGradient('Icp')
def _icp_grad(op, grad_transform, grad_residual):
    if False:
        i = 10
        return i + 15
    'The gradients for `icp`.\n\n  Args:\n    op: The `icp` `Operation` that we are differentiating, which we can use\n      to find the inputs and outputs of the original op.\n    grad_transform: Gradient with respect to `transform` output of the `icp` op.\n    grad_residual: Gradient with respect to `residual` output of the\n      `icp` op.\n\n  Returns:\n    Gradients with respect to the inputs of `icp`.\n  '
    unused_transform = op.outputs[0]
    unused_residual = op.outputs[1]
    unused_source = op.inputs[0]
    unused_ego_motion = op.inputs[1]
    unused_target = op.inputs[2]
    grad_p = -grad_residual
    grad_ego_motion = -grad_transform
    return [grad_p, grad_ego_motion, None]