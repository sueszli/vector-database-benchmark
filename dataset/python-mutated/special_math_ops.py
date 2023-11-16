"""Arithmetic Operations that don't fit into math_ops due to dependencies.

To avoid circular dependencies, some math_ops should go here.
"""
import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('math.lbeta', v1=['math.lbeta', 'lbeta'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('lbeta')
def lbeta(x, name=None):
    if False:
        print('Hello World!')
    'Computes \\\\(ln(|Beta(x)|)\\\\), reducing along the last dimension.\n\n  Given one-dimensional $z = [z_1,...,z_K]$, we define\n\n  $$Beta(z) = \\frac{\\prod_j \\Gamma(z_j)}{\\Gamma(\\sum_j z_j)},$$\n\n  where $\\Gamma$ is the gamma function.\n\n  And for $n + 1$ dimensional $x$ with shape $[N_1, ..., N_n, K]$, we define\n\n  $$lbeta(x)[i_1, ..., i_n] = \\log{|Beta(x[i_1, ..., i_n, :])|}.$$\n\n  In other words, the last dimension is treated as the $z$ vector.\n\n  Note that if $z = [u, v]$, then\n\n  $$Beta(z) = \\frac{\\Gamma(u)\\Gamma(v)}{\\Gamma(u + v)}\n    = \\int_0^1 t^{u-1} (1 - t)^{v-1} \\mathrm{d}t,$$\n\n  which defines the traditional bivariate beta function.\n\n  If the last dimension is empty, we follow the convention that the sum over\n  the empty set is zero, and the product is one.\n\n  Args:\n    x: A rank `n + 1` `Tensor`, `n >= 0` with type `float`, or `double`.\n    name: A name for the operation (optional).\n\n  Returns:\n    The logarithm of \\\\(|Beta(x)|\\\\) reducing along the last dimension.\n  '
    with ops.name_scope(name, 'lbeta', [x]):
        x = ops.convert_to_tensor(x, name='x')
        log_prod_gamma_x = math_ops.reduce_sum(math_ops.lgamma(x), axis=[-1])
        sum_x = math_ops.reduce_sum(x, axis=[-1])
        log_gamma_sum_x = math_ops.lgamma(sum_x)
        result = log_prod_gamma_x - log_gamma_sum_x
        return result

@tf_export('math.special.dawsn')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def dawsn(x, name=None):
    if False:
        return 10
    "Computes Dawson's integral of `x` element-wise.\n\n  Dawson's integral is defined as `exp(-x**2)` times the integral of\n  `exp(t**2)` from `0` to `x`, with the domain of definition all real numbers.\n\n  Dawson's function is odd.\n  >>> tf.math.special.dawsn([-1., -0.5, 0.5, 1.]).numpy()\n  array([-0.5380795, -0.4244364, 0.4244364,  0.5380795], dtype=float32)\n\n  This implementation is based off of the Cephes math library.\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types:\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.dawsn\n  @end_compatibility\n  "
    with ops.name_scope(name, 'dawsn', [x]):
        return gen_special_math_ops.dawsn(x)

@tf_export('math.special.expint')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def expint(x, name=None):
    if False:
        return 10
    'Computes the Exponential integral of `x` element-wise.\n\n  The Exponential integral is defined as the integral of `exp(t) / t` from\n  `-inf` to `x`, with the domain of definition all positive real numbers.\n\n  >>> tf.math.special.expint([1., 1.1, 2.1, 4.1]).numpy()\n  array([ 1.8951179,  2.1673784,  5.3332353, 21.048464], dtype=float32)\n\n  This implementation is based off of the Cephes math library.\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types:\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.expi\n  @end_compatibility\n  '
    with ops.name_scope(name, 'expint', [x]):
        return gen_special_math_ops.expint(x)

@tf_export('math.special.fresnel_cos')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def fresnel_cos(x, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes Fresnel's cosine integral of `x` element-wise.\n\n  The Fresnel cosine integral is defined as the integral of `cos(t^2)` from\n  `0` to `x`, with the domain of definition all real numbers.\n\n  The Fresnel cosine integral is odd.\n  >>> tf.math.special.fresnel_cos([-1., -0.1, 0.1, 1.]).numpy()\n  array([-0.7798934 , -0.09999753,  0.09999753,  0.7798934 ], dtype=float32)\n\n  This implementation is based off of the Cephes math library.\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types:\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.fresnel second output.\n  @end_compatibility\n  "
    with ops.name_scope(name, 'fresnel_cos', [x]):
        return gen_special_math_ops.fresnel_cos(x)

@tf_export('math.special.fresnel_sin')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def fresnel_sin(x, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes Fresnel's sine integral of `x` element-wise.\n\n  The Fresnel sine integral is defined as the integral of `sin(t^2)` from\n  `0` to `x`, with the domain of definition all real numbers.\n\n  >>> tf.math.special.fresnel_sin([-1., -0.1, 0.1, 1.]).numpy()\n  array([-0.43825912, -0.00052359,  0.00052359,  0.43825912], dtype=float32)\n\n  This implementation is based off of the Cephes math library.\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types:\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.fresnel first output.\n  @end_compatibility\n  "
    with ops.name_scope(name, 'fresnel_sin', [x]):
        return gen_special_math_ops.fresnel_sin(x)

@tf_export('math.special.spence')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def spence(x, name=None):
    if False:
        return 10
    "Computes Spence's integral of `x` element-wise.\n\n  Spence's integral is defined as the integral of `log(t) / (1 - t)` from\n  `1` to `x`, with the domain of definition all non-negative real numbers.\n\n  >>> tf.math.special.spence([0.5, 1., 2., 3.]).numpy()\n  array([ 0.58224034,  0.        , -0.82246685, -1.4367464], dtype=float32)\n\n  This implementation is based off of the Cephes math library.\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types:\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.spence\n  @end_compatibility\n  "
    with ops.name_scope(name, 'spence', [x]):
        return gen_special_math_ops.spence(x)

@tf_export('math.bessel_i0', 'math.special.bessel_i0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i0(x, name=None):
    if False:
        i = 10
        return i + 15
    'Computes the Bessel i0 function of `x` element-wise.\n\n  Modified Bessel function of order 0.\n\n  It is preferable to use the numerically stabler function `i0e(x)` instead.\n\n  >>> tf.math.special.bessel_i0([-1., -0.5, 0.5, 1.]).numpy()\n  array([1.26606588, 1.06348337, 1.06348337, 1.26606588], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.i0\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_i0', [x]):
        return gen_special_math_ops.bessel_i0(x)

@tf_export('math.bessel_i0e', 'math.special.bessel_i0e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i0e(x, name=None):
    if False:
        print('Hello World!')
    'Computes the Bessel i0e function of `x` element-wise.\n\n  Modified Bessel function of order 0.\n\n  >>> tf.math.special.bessel_i0e([-1., -0.5, 0.5, 1.]).numpy()\n  array([0.46575961, 0.64503527, 0.64503527, 0.46575961], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.i0e\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_i0e', [x]):
        return gen_special_math_ops.bessel_i0e(x)

@tf_export('math.bessel_i1', 'math.special.bessel_i1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i1(x, name=None):
    if False:
        print('Hello World!')
    'Computes the Bessel i1 function of `x` element-wise.\n\n  Modified Bessel function of order 1.\n\n  It is preferable to use the numerically stabler function `i1e(x)` instead.\n\n  >>> tf.math.special.bessel_i1([-1., -0.5, 0.5, 1.]).numpy()\n  array([-0.5651591 , -0.25789431,  0.25789431,  0.5651591 ], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.i1\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_i1', [x]):
        return gen_special_math_ops.bessel_i1(x)

@tf_export('math.bessel_i1e', 'math.special.bessel_i1e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_i1e(x, name=None):
    if False:
        while True:
            i = 10
    'Computes the Bessel i1e function of `x` element-wise.\n\n  Modified Bessel function of order 1.\n\n  >>> tf.math.special.bessel_i1e([-1., -0.5, 0.5, 1.]).numpy()\n  array([-0.20791042, -0.15642083,  0.15642083,  0.20791042], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.i1e\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_i1e', [x]):
        return gen_special_math_ops.bessel_i1e(x)

@tf_export('math.special.bessel_k0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k0(x, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes the Bessel k0 function of `x` element-wise.\n\n  Modified Bessel function of order 0.\n\n  It is preferable to use the numerically stabler function `k0e(x)` instead.\n\n  >>> tf.math.special.bessel_k0([0.5, 1., 2., 4.]).numpy()\n  array([0.92441907, 0.42102444, 0.11389387, 0.01115968], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.k0\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_k0', [x]):
        return gen_special_math_ops.bessel_k0(x)

@tf_export('math.special.bessel_k0e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k0e(x, name=None):
    if False:
        print('Hello World!')
    'Computes the Bessel k0e function of `x` element-wise.\n\n  Modified Bessel function of order 0.\n\n  >>> tf.math.special.bessel_k0e([0.5, 1., 2., 4.]).numpy()\n  array([1.52410939, 1.14446308, 0.84156822, 0.60929767], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.k0e\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_k0e', [x]):
        return gen_special_math_ops.bessel_k0e(x)

@tf_export('math.special.bessel_k1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k1(x, name=None):
    if False:
        i = 10
        return i + 15
    'Computes the Bessel k1 function of `x` element-wise.\n\n  Modified Bessel function of order 1.\n\n  It is preferable to use the numerically stabler function `k1e(x)` instead.\n\n  >>> tf.math.special.bessel_k1([0.5, 1., 2., 4.]).numpy()\n  array([1.65644112, 0.60190723, 0.13986588, 0.0124835 ], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.k1\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_k1', [x]):
        return gen_special_math_ops.bessel_k1(x)

@tf_export('math.special.bessel_k1e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k1e(x, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes the Bessel k1e function of `x` element-wise.\n\n  Modified Bessel function of order 1.\n\n  >>> tf.math.special.bessel_k1e([0.5, 1., 2., 4.]).numpy()\n  array([2.73100971, 1.63615349, 1.03347685, 0.68157595], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.k1e\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_k1e', [x]):
        return gen_special_math_ops.bessel_k1e(x)

@tf_export('math.special.bessel_j0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_j0(x, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes the Bessel j0 function of `x` element-wise.\n\n  Modified Bessel function of order 0.\n\n  >>> tf.math.special.bessel_j0([0.5, 1., 2., 4.]).numpy()\n  array([ 0.93846981,  0.76519769,  0.22389078, -0.39714981], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.j0\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_j0', [x]):
        return gen_special_math_ops.bessel_j0(x)

@tf_export('math.special.bessel_j1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_j1(x, name=None):
    if False:
        i = 10
        return i + 15
    'Computes the Bessel j1 function of `x` element-wise.\n\n  Modified Bessel function of order 1.\n\n  >>> tf.math.special.bessel_j1([0.5, 1., 2., 4.]).numpy()\n  array([ 0.24226846,  0.44005059,  0.57672481, -0.06604333], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.j1\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_j1', [x]):
        return gen_special_math_ops.bessel_j1(x)

@tf_export('math.special.bessel_y0')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_y0(x, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes the Bessel y0 function of `x` element-wise.\n\n  Modified Bessel function of order 0.\n\n  >>> tf.math.special.bessel_y0([0.5, 1., 2., 4.]).numpy()\n  array([-0.44451873,  0.08825696,  0.51037567, -0.01694074], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.y0\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_y0', [x]):
        return gen_special_math_ops.bessel_y0(x)

@tf_export('math.special.bessel_y1')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_y1(x, name=None):
    if False:
        i = 10
        return i + 15
    'Computes the Bessel y1 function of `x` element-wise.\n\n  Modified Bessel function of order 1.\n\n  >>> tf.math.special.bessel_y1([0.5, 1., 2., 4.]).numpy()\n  array([-1.47147239, -0.78121282, -0.10703243,  0.39792571], dtype=float32)\n\n  Args:\n    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.\n\n  @compatibility(scipy)\n  Equivalent to scipy.special.y1\n  @end_compatibility\n  '
    with ops.name_scope(name, 'bessel_y1', [x]):
        return gen_special_math_ops.bessel_y1(x)

@ops.RegisterGradient('XlaEinsum')
def _einsum_grad(op, grad):
    if False:
        return 10
    equation = op.get_attr('equation')
    if isinstance(equation, bytes):
        equation = equation.decode()
    (inputs, output) = equation.split('->')
    (left, right) = inputs.split(',')
    return [gen_xla_ops.xla_einsum(grad, op.inputs[1], equation='{},{}->{}'.format(output, right, left), name=None), gen_xla_ops.xla_einsum(grad, op.inputs[0], equation='{},{}->{}'.format(output, left, right), name=None)]

def _enclosing_tpu_context():
    if False:
        print('Hello World!')
    context = ops.get_default_graph()._get_control_flow_context()
    while context is not None and (not isinstance(context, control_flow_ops.XLAControlFlowContext)):
        context = context.outer_context
    return context

@tf_export('einsum', 'linalg.einsum')
@dispatch.add_dispatch_support
def einsum(equation, *inputs, **kwargs):
    if False:
        return 10
    'Tensor contraction over specified indices and outer product.\n\n  Einsum allows defining Tensors by defining their element-wise computation.\n  This computation is defined by `equation`, a shorthand form based on Einstein\n  summation. As an example, consider multiplying two matrices A and B to form a\n  matrix C.  The elements of C are given by:\n\n  $$ C_{i,k} = \\sum_j A_{i,j} B_{j,k} $$\n\n  or\n\n  ```\n  C[i,k] = sum_j A[i,j] * B[j,k]\n  ```\n\n  The corresponding einsum `equation` is:\n\n  ```\n  ij,jk->ik\n  ```\n\n  In general, to convert the element-wise equation into the `equation` string,\n  use the following procedure (intermediate strings for matrix multiplication\n  example provided in parentheses):\n\n  1. remove variable names, brackets, and commas, (`ik = sum_j ij * jk`)\n  2. replace "*" with ",", (`ik = sum_j ij , jk`)\n  3. drop summation signs, and (`ik = ij, jk`)\n  4. move the output to the right, while replacing "=" with "->". (`ij,jk->ik`)\n\n  Note: If the output indices are not specified repeated indices are summed.\n  So `ij,jk->ik` can be simplified to `ij,jk`.\n\n  Many common operations can be expressed in this way.  For example:\n\n  **Matrix multiplication**\n\n  >>> m0 = tf.random.normal(shape=[2, 3])\n  >>> m1 = tf.random.normal(shape=[3, 5])\n  >>> e = tf.einsum(\'ij,jk->ik\', m0, m1)\n  >>> # output[i,k] = sum_j m0[i,j] * m1[j, k]\n  >>> print(e.shape)\n  (2, 5)\n\n  Repeated indices are summed if the output indices are not specified.\n\n  >>> e = tf.einsum(\'ij,jk\', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]\n  >>> print(e.shape)\n  (2, 5)\n\n\n  **Dot product**\n\n  >>> u = tf.random.normal(shape=[5])\n  >>> v = tf.random.normal(shape=[5])\n  >>> e = tf.einsum(\'i,i->\', u, v)  # output = sum_i u[i]*v[i]\n  >>> print(e.shape)\n  ()\n\n  **Outer product**\n\n  >>> u = tf.random.normal(shape=[3])\n  >>> v = tf.random.normal(shape=[5])\n  >>> e = tf.einsum(\'i,j->ij\', u, v)  # output[i,j] = u[i]*v[j]\n  >>> print(e.shape)\n  (3, 5)\n\n  **Transpose**\n\n  >>> m = tf.ones(2,3)\n  >>> e = tf.einsum(\'ij->ji\', m0)  # output[j,i] = m0[i,j]\n  >>> print(e.shape)\n  (3, 2)\n\n  **Diag**\n\n  >>> m = tf.reshape(tf.range(9), [3,3])\n  >>> diag = tf.einsum(\'ii->i\', m)\n  >>> print(diag.shape)\n  (3,)\n\n  **Trace**\n\n  >>> # Repeated indices are summed.\n  >>> trace = tf.einsum(\'ii\', m)  # output[j,i] = trace(m) = sum_i m[i, i]\n  >>> assert trace == sum(diag)\n  >>> print(trace.shape)\n  ()\n\n  **Batch matrix multiplication**\n\n  >>> s = tf.random.normal(shape=[7,5,3])\n  >>> t = tf.random.normal(shape=[7,3,2])\n  >>> e = tf.einsum(\'bij,bjk->bik\', s, t)\n  >>> # output[a,i,k] = sum_j s[a,i,j] * t[a, j, k]\n  >>> print(e.shape)\n  (7, 5, 2)\n\n  This method does not support broadcasting on named-axes. All axes with\n  matching labels should have the same length. If you have length-1 axes,\n  use `tf.squeeze` or `tf.reshape` to eliminate them.\n\n  To write code that is agnostic to the number of indices in the input\n  use an ellipsis. The ellipsis is a placeholder for "whatever other indices\n  fit here".\n\n  For example, to perform a NumPy-style broadcasting-batch-matrix multiplication\n  where the matrix multiply acts on the last two axes of the input, use:\n\n  >>> s = tf.random.normal(shape=[11, 7, 5, 3])\n  >>> t = tf.random.normal(shape=[11, 7, 3, 2])\n  >>> e =  tf.einsum(\'...ij,...jk->...ik\', s, t)\n  >>> print(e.shape)\n  (11, 7, 5, 2)\n\n  Einsum **will** broadcast over axes covered by the ellipsis.\n\n  >>> s = tf.random.normal(shape=[11, 1, 5, 3])\n  >>> t = tf.random.normal(shape=[1, 7, 3, 2])\n  >>> e =  tf.einsum(\'...ij,...jk->...ik\', s, t)\n  >>> print(e.shape)\n  (11, 7, 5, 2)\n\n  Args:\n    equation: a `str` describing the contraction, in the same format as\n      `numpy.einsum`.\n    *inputs: the inputs to contract (each one a `Tensor`), whose shapes should\n      be consistent with `equation`.\n    **kwargs:\n      - optimize: Optimization strategy to use to find contraction path using\n        opt_einsum. Must be \'greedy\', \'optimal\', \'branch-2\', \'branch-all\' or\n          \'auto\'. (optional, default: \'greedy\').\n      - name: A name for the operation (optional).\n\n  Returns:\n    The contracted `Tensor`, with shape determined by `equation`.\n\n  Raises:\n    ValueError: If\n      - the format of `equation` is incorrect,\n      - number of inputs or their shapes are inconsistent with `equation`.\n  '
    return _einsum_v2(equation, *inputs, **kwargs)

def _einsum_v1(equation, *inputs, **kwargs):
    if False:
        while True:
            i = 10
    'Legacy implementation of einsum without using EinsumOp.'
    name = kwargs.pop('name', None)
    if kwargs:
        raise TypeError(f"Invalid keyword arguments for this function: {', '.join([format(key) for key in sorted(list(kwargs.keys()))])}. Expected: name.")
    with ops.name_scope(name, 'einsum', [equation, inputs]) as name:
        inputs = list(inputs)
        input_shapes = [x.shape for x in inputs]
        (input_axis_labels, output_axis_labels) = _einsum_v1_parse_and_resolve_equation(equation, input_shapes)
        axis_labels = set(''.join(input_axis_labels) + output_axis_labels)
        for a in axis_labels:
            for input_labels in input_axis_labels:
                if len(input_axis_labels) == 1 and input_labels.count(a) == 2 and (input_labels == input_labels[::-1]) and ('->' not in equation):
                    return math_ops.trace(inputs[0])
                if input_labels.count(a) > 1:
                    raise ValueError(f'Subscript not supported: the axis {a} appears more than once in {input_labels}.')
        for a in axis_labels:
            input_count = sum((1 for s in input_axis_labels if a in s))
            if input_count > 2 and a not in output_axis_labels:
                logging.warn(f'Falling back to exponential-space implementation of einsum() because index {a} is summed over more than two inputs.')
                return _exponential_space_einsum_v1(equation, *inputs)
        if _enclosing_tpu_context() is not None and len(inputs) == 2:
            return gen_xla_ops.xla_einsum(inputs[0], inputs[1], input_axis_labels[0] + ',' + input_axis_labels[1] + '->' + output_axis_labels)
        temp = inputs[0]
        temp_axis_labels = input_axis_labels[0]
        for i in range(len(inputs) - 1):
            axes_to_sum = set(temp_axis_labels) & set(input_axis_labels[i + 1]) - set(output_axis_labels)
            (temp, temp_axis_labels) = _einsum_v1_reduction(temp, temp_axis_labels, inputs[i + 1], input_axis_labels[i + 1], axes_to_sum)
        missing_indices = set(temp_axis_labels) - set(output_axis_labels)
        if missing_indices:
            axis = [i for (i, a) in enumerate(temp_axis_labels) if a not in output_axis_labels]
            temp = math_ops.reduce_sum(temp, axis=axis)
            temp_axis_labels = ''.join((a for a in temp_axis_labels if a in output_axis_labels))
        if sorted(temp_axis_labels) != sorted(output_axis_labels):
            raise ValueError(f'Invalid equation: {equation}. The computed and specified output labels do not match: {temp_axis_labels} vs {output_axis_labels}.')
        perm = [temp_axis_labels.index(a) for a in output_axis_labels]
        return _transpose_if_necessary(temp, perm)

def _einsum_v1_parse_and_resolve_equation(equation, input_shapes):
    if False:
        print('Hello World!')
    'Helper for einsum() that splits/resolves inputs & outputs.\n\n  Args:\n    equation: Equation string given as argument to einsum().\n    input_shapes: List of the shapes of all inputs given to einsum()\n\n  Returns:\n    input_axis_labels, output_axis_labels where:\n      input_axis_labels: List of length len(input_shapes) of strings\n      representing the character label for each dimension of each given input,\n      resolving any broadcast (...) axes,\n    output_axis_labels: A string of character labels for each axes of output\n      tensor, filling in missing output subscripts and broadcast axes.\n\n  Raises:\n    ValueError: If equation is in the uncorrect format, incorrect number of\n      inputs given or broadcast axes "..." or output axes could not be resolved.\n  '
    equation = equation.replace(' ', '')
    match = re.match('^([a-zA-Z,.]+)(->[a-zA-Z.]*)?$', equation)
    if not match:
        raise ValueError(f'Indices have incorrect format. Received: {equation}.')
    input_axis_labels = match.group(1).split(',')
    output_axis_labels = match.group(2)[2:] if match.group(2) else None
    if len(input_shapes) != len(input_axis_labels):
        raise ValueError(f'Got {len(input_shapes)} arguments for equation "{equation}", expecting {len(input_axis_labels)}.')
    ellipsis_axes = ''
    if '...' in equation:
        unused = ''.join((c for c in string.ascii_letters if c not in ''.join(input_axis_labels)))
        for (i, ax) in enumerate(input_axis_labels):
            if '...' in ax:
                parts = ax.split('...')
                if len(parts) != 2:
                    raise ValueError(f'Unable to resolve ellipsis. Excess number found: {len(parts) - 1} vs 1.')
                if input_shapes[i].ndims is None:
                    raise ValueError('Unable to statically infer ellipsis axes. The input shapes has a dynamic dimensionality.')
                n = input_shapes[i].ndims - len(''.join(parts))
                if n < 0:
                    raise ValueError('Ellipses lengths do not match.')
                if len(unused) < n:
                    raise ValueError('Unable to resolve ellipsis, too many distinct labels.')
                replace_axes = unused[-n:] if n > 0 else ''
                input_axis_labels[i] = input_axis_labels[i].replace('...', replace_axes)
                if len(replace_axes) > len(ellipsis_axes):
                    ellipsis_axes = replace_axes
        if any(('.' in ax for ax in input_axis_labels)):
            raise ValueError(f'Period "." found outside of ellipsis in input {input_axis_labels}.')
        if output_axis_labels is not None:
            output_axis_labels = output_axis_labels.replace('...', ellipsis_axes)
            if '.' in output_axis_labels:
                raise ValueError(f'Period "." found outside of ellipsis in output {output_axis_labels}.')
    if output_axis_labels is None:
        axis_labels = set(''.join(input_axis_labels)) - set(ellipsis_axes)
        indices = ''.join(sorted(axis_labels))
        counts = {ax: 0 for ax in indices}
        for axes_ in input_axis_labels:
            for ax in axes_:
                if ax not in ellipsis_axes:
                    counts[ax] += 1
        output_axis_labels = ellipsis_axes + ''.join(sorted((ax for ax in axis_labels if counts[ax] == 1)))
    return (input_axis_labels, output_axis_labels)

def _einsum_v1_reduction(t0, t0_axis_labels, t1, t1_axis_labels, axes_to_sum):
    if False:
        return 10
    "Helper for einsum() that computes the result of a two-argument einsum().\n\n  Args:\n    t0: a `Tensor`\n    t0_axis_labels: a string of axis labels.  This string's length must equal\n      the rank of t0.\n    t1: a `Tensor`\n    t1_axis_labels: a string to axis labels.  This string's length must equal\n      the rank of t1.\n    axes_to_sum: set of labels of axes to be summed over\n\n  Returns:\n    A `Tensor` whose elements are obtained by summing, over all axes in\n    `axes_to_sum`, the corresponding elements of `t0` and `t1`.\n\n    For example, if t0_axis_labels == 'abijk', t1_axis_labels == 'acjkl', and\n    axes_to_sum == {j,k}, this will return a tensor x where\n\n      out[a,b,c,i,l] = sum_j sum_k t0[a,b,i,j,k] * t1[a,c,j,k,l]\n\n  Raises:\n    ValueError: if the rank of `t0` does not match the length of\n      `t0_axis_labels`, or that of `t1` does not match the length of\n      `t1_axis_labels`.\n  "
    if len(t0_axis_labels) != len(t0.shape):
        raise ValueError(f'Tensor `t0` of rank {len(t0.shape)} does not match einsum reduction of length {len(t0_axis_labels)}.')
    if len(t1_axis_labels) != len(t1.shape):
        raise ValueError(f'Tensor `t1` of rank {len(t1.shape)} does not match einsum reduction of length {len(t1_axis_labels)}')
    assert all((a in t0_axis_labels and a in t1_axis_labels for a in axes_to_sum))
    preserved_axes = (set(t0_axis_labels) & set(t1_axis_labels)) - axes_to_sum
    broadcast_axes = {}
    for (i, sym_list) in enumerate([t0_axis_labels, t1_axis_labels]):
        broadcast_axes[i] = set(sym_list) - preserved_axes - axes_to_sum

    def sort_key(input_index, a):
        if False:
            for i in range(10):
                print('nop')
        if a in preserved_axes:
            return (-1, a)
        elif input_index == 0 and a in broadcast_axes[0] or (input_index == 1 and a in axes_to_sum):
            return (0, a)
        else:
            return (1, a)
    axis_labels = [t0_axis_labels, t1_axis_labels]
    sorted_axes = [sorted(sym_list, key=lambda a: sort_key(i, a)) for (i, sym_list) in enumerate(axis_labels)]
    inputs = [t0, t1]
    for (i, axes_str) in enumerate(axis_labels):
        perm = [axes_str.find(a) for a in sorted_axes[i]]
        inputs[i] = _transpose_if_necessary(inputs[i], perm)
    (t0, t1) = inputs
    if not axes_to_sum:
        for _ in broadcast_axes[1]:
            t0 = array_ops.expand_dims(t0, -1)
        for _ in broadcast_axes[0]:
            t1 = array_ops.expand_dims(t1, len(preserved_axes))
        product = math_ops.multiply(t0, t1)
        product_axes = sorted_axes[0] + sorted_axes[1][len(preserved_axes):]
        return (product, ''.join(product_axes))
    else:
        t0_shape = _get_shape(t0)
        num_broadcast_elements_t0 = _total_size(t0_shape[len(preserved_axes):-len(axes_to_sum)])
        num_summed_elements = _total_size(t0_shape[-len(axes_to_sum):])
        new_shape = t0_shape[:len(preserved_axes)] + [num_broadcast_elements_t0, num_summed_elements]
        t0 = _reshape_if_necessary(t0, new_shape)
        t1_shape = _get_shape(t1)
        num_broadcast_elements_t1 = _total_size(t1_shape[len(preserved_axes) + len(axes_to_sum):])
        new_shape = t1_shape[:len(preserved_axes)] + [num_summed_elements, num_broadcast_elements_t1]
        t1 = _reshape_if_necessary(t1, new_shape)
        product = math_ops.matmul(t0, t1)
        uncompacted_shape = t0_shape[:len(preserved_axes) + len(broadcast_axes[0])] + t1_shape[len(t1_shape) - len(broadcast_axes[1]):]
        product = _reshape_if_necessary(product, uncompacted_shape)
        product_axes = sorted_axes[0][:len(preserved_axes) + len(broadcast_axes[0])] + sorted_axes[1][len(sorted_axes[1]) - len(broadcast_axes[1]):]
        return (product, ''.join(product_axes))

def _transpose_if_necessary(tensor, perm):
    if False:
        for i in range(10):
            print('nop')
    'Like transpose(), but avoids creating a new tensor if possible.'
    if perm != list(range(len(perm))):
        return array_ops.transpose(tensor, perm=perm)
    else:
        return tensor

def _reshape_if_necessary(tensor, new_shape):
    if False:
        print('Hello World!')
    'Like reshape(), but avoids creating a new tensor if possible.'
    new_shape = tuple((-1 if x is None else x for x in new_shape))
    cur_shape = tuple((x.value for x in tensor.shape.dims))
    if len(new_shape) == len(cur_shape) and all((not isinstance(d1, tensor_lib.Tensor) and (d0 == d1 or d1 == -1) for (d0, d1) in zip(cur_shape, new_shape))):
        return tensor
    else:
        return array_ops.reshape(tensor, new_shape)

def _get_shape(tensor):
    if False:
        return 10
    'Like get_shape().as_list(), but explicitly queries the shape of a tensor\n  if necessary to ensure that the returned value contains no unknown value.'
    shape = tensor.shape.as_list()
    none_indices = [i for (i, d) in enumerate(shape) if d is None]
    if none_indices:
        shape_tensor = array_ops.shape(tensor)
        for i in none_indices:
            shape[i] = shape_tensor[i]
    return shape

def _total_size(shape_values):
    if False:
        return 10
    'Given list of tensor shape values, returns total size.\n  If shape_values contains tensor values (which are results of\n  array_ops.shape), then it returns a scalar tensor.\n  If not, it returns an integer.'
    result = 1
    for val in shape_values:
        result *= val
    return result

def _exponential_space_einsum_v1(equation, *inputs):
    if False:
        i = 10
        return i + 15
    'Fallback implementation that supports summing an index over > 2 inputs.'
    inputs = list(inputs)
    input_shapes = [x.shape for x in inputs]
    (idx_in, idx_out) = _einsum_v1_parse_and_resolve_equation(equation, input_shapes)
    idx_all = set(''.join(idx_in) + idx_out)
    indices = ''.join(sorted(idx_all))
    missing_idx = set(idx_out).difference(idx_all)
    if missing_idx:
        raise ValueError(f'Unknown output axes: {missing_idx}.')
    axis_order = {}
    for ax in indices:
        if ax not in idx_out:
            axis_order[ax] = len(axis_order)
    for ax in idx_out:
        axis_order[ax] = len(axis_order)
    for (i, (input_, axes_)) in enumerate(zip(inputs, idx_in)):
        if input_.shape.ndims != len(axes_):
            raise ValueError(f'Input {i} with axes {axes_} has incorrect number of dimensions (expected {len(axes_)}, got {input_.shape.ndims}).')
        sorted_idx = sorted(axes_, key=axis_order.get)
        if len(set(axes_)) != len(axes_):
            raise ValueError(f'Subscript not supported: an axis appears more than once: {axes_}.')
        if list(axes_) != sorted_idx:
            permuted = [axes_.find(ax) for ax in sorted_idx]
            inputs[i] = array_ops.transpose(input_, permuted)
            idx_in[i] = sorted_idx
    reduction_idx = []
    shapes = [[dim if dim else -1 for dim in tensor.shape.as_list()] for tensor in inputs]
    for (j, ax) in enumerate(sorted(idx_all, key=axis_order.get)):
        dims = []
        for (i, idx) in enumerate(idx_in):
            if ax not in idx:
                shapes[i].insert(j, 1)
            else:
                dim = shapes[i][j]
                if isinstance(dim, int) and dim > 1:
                    dims.append(dim)
        if len(set(dims)) > 1:
            raise ValueError(f'Dimension mismatch on axis: {ax}. Found {len(set(dims))}, expected 1.')
        if ax not in idx_out:
            reduction_idx.append(j)
    expanded_inputs = [array_ops.reshape(input_, shape) for (input_, shape) in zip(inputs, shapes)]
    expanded_output = 1
    for input_ in expanded_inputs:
        expanded_output *= input_
    return math_ops.reduce_sum(expanded_output, reduction_idx)

def _einsum_v2(equation, *inputs, **kwargs):
    if False:
        while True:
            i = 10
    'Implementation of einsum utilizing opt_einsum and EinsumOp.'
    name = kwargs.pop('name', None)
    optimize = kwargs.pop('optimize', 'greedy')
    if kwargs:
        raise TypeError(f"Invalid keyword arguments for einsum: {', '.join(kwargs)}. Valid arguments: name, optimize, greedy.")
    with ops.name_scope(name, 'einsum', [equation, inputs]) as name:
        inputs = list(inputs)
        input_shapes = []
        for operand in inputs:
            if isinstance(operand.shape, tensor_shape.TensorShape):
                input_shapes.append(operand.shape.as_list() if operand.shape else None)
            else:
                input_shapes.append(list(operand.shape))
        (resolved_equation, resolved_input_shapes, ellipsis_label) = _einsum_v2_parse_and_resolve_equation(equation, input_shapes)
        if len(inputs) <= 2:
            if ellipsis_label:
                resolved_equation = resolved_equation.replace(ellipsis_label, '...')
            return gen_linalg_ops.einsum(inputs, resolved_equation)
        shaped = collections.namedtuple('shaped', ['shape'])
        shaped_inputs = tuple([shaped(tuple(shape)) for shape in resolved_input_shapes])
        indices_and_equations = _get_opt_einsum_contract_path(resolved_equation, shaped_inputs, optimize)
        for (operand_indices, binary_equation) in indices_and_equations:
            if ellipsis_label:
                binary_equation = binary_equation.replace(ellipsis_label, '...')
            operands = list(map(inputs.pop, operand_indices))
            inputs.append(gen_linalg_ops.einsum(operands, binary_equation))
        return inputs[0]

def _get_opt_einsum_contract_path(equation, shaped_inputs_tuple, optimize):
    if False:
        print('Hello World!')
    'Returns the (memoized) result of opt_einsum.contract_path.'
    (_, contractions) = opt_einsum.contract_path(equation, *shaped_inputs_tuple, optimize=optimize, einsum_call=True, use_blas=True)
    indices_and_equations = tuple([(expr[0], expr[2]) for expr in contractions])
    return indices_and_equations
_get_opt_einsum_contract_path = functools.lru_cache(maxsize=128)(_get_opt_einsum_contract_path)

def _einsum_v2_parse_and_resolve_equation(equation, input_shapes):
    if False:
        print('Hello World!')
    'Helper which validates einsum equation and resolves input shapes.'
    resolved_equation = equation.replace(' ', '')
    ellipsis_label = None
    if '...' in equation:
        ellipsis_label = '0'
        if ellipsis_label in resolved_equation:
            raise ValueError(f'Invalid character "{ellipsis_label}" in equation: {equation}.')
        resolved_equation = resolved_equation.replace('...', ellipsis_label)
    allowed_labels = 'a-zA-Z'
    if ellipsis_label:
        allowed_labels += ellipsis_label
    match = re.match('^([{0},]*)(->[{0}]*)?$'.format(allowed_labels), resolved_equation)
    if not match:
        raise ValueError('Subscripts have incorrect format: {}'.format(resolved_equation))
    input_labels = match.group(1).split(',')
    output_labels = match.group(2)[2:] if match.group(2) else None
    if len(input_shapes) != len(input_labels):
        raise ValueError('Got {} inputs for equation "{}", expecting {}'.format(len(input_shapes), equation, len(input_labels)))
    if '->' not in resolved_equation:
        label_counts = collections.Counter(match.group(1))
        output_labels = ''.join([x for x in sorted(list(label_counts)) if x != ',' and label_counts[x] == 1])
        resolved_equation += '->' + output_labels
    if output_labels and len(set(output_labels)) != len(output_labels):
        raise ValueError('Output subscripts contain a label appearing more than once: {}'.format(equation))
    input_label_set = set(match.group(1))
    for label in output_labels:
        if label != ellipsis_label and label not in input_label_set:
            raise ValueError('Output subscripts contain the label {} not present in the input subscripts.'.format(label))
    if ellipsis_label and output_labels:
        num_output_ellipses = output_labels.count(ellipsis_label)
        if num_output_ellipses > 1:
            raise ValueError('Output subscripts contain multiple ellipsis: {}'.format(equation))
    if len(input_shapes) <= 2:
        return (resolved_equation, None, ellipsis_label)
    label_to_dim = collections.defaultdict(lambda : 1)
    for (i, (labels, shape)) in enumerate(zip(input_labels, input_shapes)):
        if shape is None:
            continue
        ellipsis_start = labels.find(ellipsis_label) if ellipsis_label else -1
        if ellipsis_start != -1:
            if ellipsis_start != labels.rfind(ellipsis_label):
                raise ValueError(f"Too many ellipses in input label {labels.replace(ellipsis_label, '...')}.")
            if len(labels) > len(shape) + 1:
                raise ValueError('Too many named labels in {}th subscript string of equation {} for input shape {} '.format(i, equation, shape))
            ellipsis_end = ellipsis_start + len(shape) + 1 - len(labels)
            shape[ellipsis_start:ellipsis_end] = [np.prod(list(filter(None, shape[ellipsis_start:ellipsis_end])), dtype=np.int64)]
        elif len(labels) != len(shape):
            raise ValueError('Number of named labels in input #{} of equation {} must be equal to the number of dimensions in shape {}'.format(i, equation, shape))
        for (dim, label) in zip(shape, labels):
            if dim is not None:
                label_to_dim[label] = max(label_to_dim[label], dim)
    resolved_shapes = []
    for labels in input_labels:
        resolved_shapes.append([label_to_dim[label] for label in labels])
    return (resolved_equation, resolved_shapes, ellipsis_label)