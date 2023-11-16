"""Gradients for operators defined in grl_ops.py."""
import tensorflow as tf

@tf.RegisterGradient('GradientReversal')
def _GradientReversalGrad(_, grad):
    if False:
        while True:
            i = 10
    'The gradients for `gradient_reversal`.\n\n  Args:\n    _: The `gradient_reversal` `Operation` that we are differentiating,\n      which we can use to find the inputs and outputs of the original op.\n    grad: Gradient with respect to the output of the `gradient_reversal` op.\n\n  Returns:\n    Gradient with respect to the input of `gradient_reversal`, which is simply\n    the negative of the input gradient.\n\n  '
    return tf.negative(grad)