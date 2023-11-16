"""Gradients for operators defined in random_ops.py."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops

def add_leading_unit_dimensions(x, num_dimensions):
    if False:
        i = 10
        return i + 15
    new_shape = array_ops.concat([array_ops.ones([num_dimensions], dtype=dtypes.int32), array_ops.shape(x)], axis=0)
    return array_ops.reshape(x, new_shape)

@ops.RegisterGradient('RandomGamma')
def _RandomGammaGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns the gradient of a Gamma sample w.r.t. alpha.\n\n  The gradient is computed using implicit differentiation\n  (Figurnov et al., 2018).\n\n  Args:\n    op: A `RandomGamma` operation. We assume that the inputs to the operation\n      are `shape` and `alpha` tensors, and the output is the `sample` tensor.\n    grad: The incoming gradient `dloss / dsample` of the same shape as\n      `op.outputs[0]`.\n\n  Returns:\n    A `Tensor` with derivatives `dloss / dalpha`.\n\n  References:\n    Implicit Reparameterization Gradients:\n      [Figurnov et al., 2018]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)\n      ([pdf]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))\n  '
    shape = op.inputs[0]
    alpha = op.inputs[1]
    sample = op.outputs[0]
    with ops.control_dependencies([grad]):
        num_sample_dimensions = array_ops.shape(shape)[0]
        alpha_broadcastable = add_leading_unit_dimensions(alpha, num_sample_dimensions)
        partial_a = gen_random_ops.random_gamma_grad(alpha_broadcastable, sample)
        return (None, math_ops.reduce_sum(grad * partial_a, axis=math_ops.range(num_sample_dimensions)))

@ops.RegisterGradient('StatelessRandomGammaV2')
def _StatelessRandomGammaV2Grad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns the gradient of a Gamma sample w.r.t. alpha.\n\n  The gradient is computed using implicit differentiation\n  (Figurnov et al., 2018).\n\n  Args:\n    op: A `StatelessRandomGamma` operation. We assume that the inputs to the\n      operation are `shape`, `seed` and `alpha` tensors, and the output is the\n      `sample` tensor.\n    grad: The incoming gradient `dloss / dsample` of the same shape as\n      `op.outputs[0]`.\n\n  Returns:\n    A `Tensor` with derivatives `dloss / dalpha`.\n\n  References:\n    Implicit Reparameterization Gradients:\n      [Figurnov et al., 2018]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)\n      ([pdf]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))\n  '
    shape = op.inputs[0]
    alpha = op.inputs[2]
    sample = op.outputs[0]
    with ops.control_dependencies([grad]):
        return (None, None, _StatelessGammaGradAlpha(shape, alpha, sample, grad))

@ops.RegisterGradient('StatelessRandomGammaV3')
def _StatelessRandomGammaV3Grad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns the gradient of a Gamma sample w.r.t. alpha.\n\n  The gradient is computed using implicit differentiation\n  (Figurnov et al., 2018).\n\n  Args:\n    op: A `StatelessRandomGamma` operation. We assume that the inputs to the\n      operation are `shape`, `key`, `counter`, `alg`, and `alpha` tensors, and\n      the output is the `sample` tensor.\n    grad: The incoming gradient `dloss / dsample` of the same shape as\n      `op.outputs[0]`.\n\n  Returns:\n    A `Tensor` with derivatives `dloss / dalpha`.\n\n  References:\n    Implicit Reparameterization Gradients:\n      [Figurnov et al., 2018]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)\n      ([pdf]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))\n  '
    shape = op.inputs[0]
    alpha = op.inputs[4]
    sample = op.outputs[0]
    with ops.control_dependencies([grad]):
        return (None, None, None, None, _StatelessGammaGradAlpha(shape, alpha, sample, grad))

def _StatelessGammaGradAlpha(shape, alpha, sample, grad):
    if False:
        while True:
            i = 10
    'Returns gradients of a gamma sampler wrt alpha.'
    num_sample_dimensions = array_ops.shape(shape)[0] - array_ops.rank(alpha)
    alpha_broadcastable = add_leading_unit_dimensions(alpha, num_sample_dimensions)
    partial_a = gen_random_ops.random_gamma_grad(alpha_broadcastable, sample)
    return math_ops.reduce_sum(grad * partial_a, axis=math_ops.range(num_sample_dimensions))

def _Ndtr(x):
    if False:
        print('Hello World!')
    'Normal distribution function.'
    half_sqrt_2 = constant_op.constant(0.5 * np.sqrt(2.0), dtype=x.dtype, name='half_sqrt_2')
    w = x * half_sqrt_2
    z = math_ops.abs(w)
    y = array_ops.where(z < half_sqrt_2, 1.0 + math_ops.erf(w), array_ops.where(w > 0.0, 2.0 - math_ops.erfc(z), math_ops.erfc(z)))
    return 0.5 * y

@ops.RegisterGradient('StatelessParameterizedTruncatedNormal')
def _StatelessParameterizedTruncatedNormalGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns the gradient of a TruncatedNormal sample w.r.t. parameters.\n\n  The gradient is computed using implicit differentiation\n  (Figurnov et al., 2018).\n\n  Args:\n    op: A `StatelessParameterizedTruncatedNormal` operation. We assume that the\n      inputs to the operation are `shape`, `seed`, `mean`, `stddev`, `minval`,\n      and `maxval` tensors, and the output is the `sample` tensor.\n    grad: The incoming gradient `dloss / dsample` of the same shape as\n      `op.outputs[0]`.\n\n  Returns:\n    A list of `Tensor` with derivates with respect to each parameter.\n\n  References:\n    Implicit Reparameterization Gradients:\n      [Figurnov et al., 2018]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)\n      ([pdf]\n      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))\n  '
    shape = op.inputs[0]
    mean = op.inputs[2]
    stddev = op.inputs[3]
    minval = op.inputs[4]
    maxval = op.inputs[5]
    sample = op.outputs[0]
    with ops.control_dependencies([grad]):
        minval_std = (minval - mean) / stddev
        maxval_std = (maxval - mean) / stddev
        sample_std = (sample - mean) / stddev
        cdf_sample = (_Ndtr(sample_std) - _Ndtr(minval_std)) / (_Ndtr(maxval_std) - _Ndtr(minval_std))
        tiny = np.finfo(mean.dtype.as_numpy_dtype).tiny
        eps = np.finfo(mean.dtype.as_numpy_dtype).eps
        cdf_sample = clip_ops.clip_by_value(cdf_sample, tiny, 1 - eps)
        dmaxval = math_ops.exp(0.5 * (sample_std ** 2 - maxval_std ** 2) + math_ops.log(cdf_sample))
        dminval = math_ops.exp(0.5 * (sample_std ** 2 - minval_std ** 2) + math_ops.log1p(-cdf_sample))
        dmean = array_ops.ones_like(sample_std)
        dstddev = sample_std
        mean_shape = array_ops.shape(mean)
        stddev_shape = array_ops.shape(stddev)
        minval_shape = array_ops.shape(minval)
        maxval_shape = array_ops.shape(maxval)
        broadcast_shape = array_ops.broadcast_dynamic_shape(mean_shape, stddev_shape)
        broadcast_shape = array_ops.broadcast_dynamic_shape(minval_shape, broadcast_shape)
        broadcast_shape = array_ops.broadcast_dynamic_shape(maxval_shape, broadcast_shape)
        extra_dims = math_ops.range(array_ops.size(shape) - array_ops.size(broadcast_shape))
        grad_mean = math_ops.reduce_sum(grad * dmean, axis=extra_dims)
        grad_stddev = math_ops.reduce_sum(grad * dstddev, axis=extra_dims)
        grad_minval = math_ops.reduce_sum(grad * dminval, axis=extra_dims)
        grad_maxval = math_ops.reduce_sum(grad * dmaxval, axis=extra_dims)
        (_, rmean) = gen_array_ops.broadcast_gradient_args(broadcast_shape, mean_shape)
        (_, rstddev) = gen_array_ops.broadcast_gradient_args(broadcast_shape, stddev_shape)
        (_, rminval) = gen_array_ops.broadcast_gradient_args(broadcast_shape, minval_shape)
        (_, rmaxval) = gen_array_ops.broadcast_gradient_args(broadcast_shape, maxval_shape)
        grad_mean = array_ops.reshape(math_ops.reduce_sum(grad_mean, axis=rmean, keepdims=True), mean_shape)
        grad_stddev = array_ops.reshape(math_ops.reduce_sum(grad_stddev, axis=rstddev, keepdims=True), stddev_shape)
        grad_minval = array_ops.reshape(math_ops.reduce_sum(grad_minval, axis=rminval, keepdims=True), minval_shape)
        grad_maxval = array_ops.reshape(math_ops.reduce_sum(grad_maxval, axis=rmaxval, keepdims=True), maxval_shape)
        return (None, None, grad_mean, grad_stddev, grad_minval, grad_maxval)