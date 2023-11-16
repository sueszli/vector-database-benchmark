"""Special Math Ops."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
__all__ = ['erfinv', 'ndtr', 'ndtri', 'log_ndtr', 'log_cdf_laplace']
LOGNDTR_FLOAT64_LOWER = np.array(-20, np.float64)
LOGNDTR_FLOAT32_LOWER = np.array(-10, np.float32)
LOGNDTR_FLOAT64_UPPER = np.array(8, np.float64)
LOGNDTR_FLOAT32_UPPER = np.array(5, np.float32)

def ndtr(x, name='ndtr'):
    if False:
        return 10
    'Normal distribution function.\n\n  Returns the area under the Gaussian probability density function, integrated\n  from minus infinity to x:\n\n  ```\n                    1       / x\n     ndtr(x)  = ----------  |    exp(-0.5 t**2) dt\n                sqrt(2 pi)  /-inf\n\n              = 0.5 (1 + erf(x / sqrt(2)))\n              = 0.5 erfc(x / sqrt(2))\n  ```\n\n  Args:\n    x: `Tensor` of type `float32`, `float64`.\n    name: Python string. A name for the operation (default="ndtr").\n\n  Returns:\n    ndtr: `Tensor` with `dtype=x.dtype`.\n\n  Raises:\n    TypeError: if `x` is not floating-type.\n  '
    with ops.name_scope(name, values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        if x.dtype.as_numpy_dtype not in [np.float32, np.float64]:
            raise TypeError('x.dtype=%s is not handled, see docstring for supported types.' % x.dtype)
        return _ndtr(x)

def _ndtr(x):
    if False:
        for i in range(10):
            print('nop')
    'Implements ndtr core logic.'
    half_sqrt_2 = constant_op.constant(0.5 * np.sqrt(2.0), dtype=x.dtype, name='half_sqrt_2')
    w = x * half_sqrt_2
    z = math_ops.abs(w)
    y = array_ops.where_v2(math_ops.less(z, half_sqrt_2), 1.0 + math_ops.erf(w), array_ops.where_v2(math_ops.greater(w, 0.0), 2.0 - math_ops.erfc(z), math_ops.erfc(z)))
    return 0.5 * y

def ndtri(p, name='ndtri'):
    if False:
        while True:
            i = 10
    'The inverse of the CDF of the Normal distribution function.\n\n  Returns x such that the area under the pdf from minus infinity to x is equal\n  to p.\n\n  A piece-wise rational approximation is done for the function.\n  This is a port of the implementation in netlib.\n\n  Args:\n    p: `Tensor` of type `float32`, `float64`.\n    name: Python string. A name for the operation (default="ndtri").\n\n  Returns:\n    x: `Tensor` with `dtype=p.dtype`.\n\n  Raises:\n    TypeError: if `p` is not floating-type.\n  '
    with ops.name_scope(name, values=[p]):
        p = ops.convert_to_tensor(p, name='p')
        if p.dtype.as_numpy_dtype not in [np.float32, np.float64]:
            raise TypeError('p.dtype=%s is not handled, see docstring for supported types.' % p.dtype)
        return _ndtri(p)

def _ndtri(p):
    if False:
        print('Hello World!')
    'Implements ndtri core logic.'
    p0 = [-1.2391658386738125, 13.931260938727968, -56.67628574690703, 98.00107541859997, -59.96335010141079]
    q0 = [-1.1833162112133, 15.90562251262117, -82.03722561683334, 200.26021238006066, -225.46268785411937, 86.36024213908905, 4.676279128988815, 1.9544885833814176, 1.0]
    p1 = [-0.0008574567851546854, -0.03504246268278482, -0.1402560791713545, 2.1866330685079025, 14.684956192885803, 44.08050738932008, 57.16281922464213, 31.525109459989388, 4.0554489230596245]
    q1 = [-0.0009332594808954574, -0.03808064076915783, -0.14218292285478779, 2.504649462083094, 15.04253856929075, 41.3172038254672, 45.39076351288792, 15.779988325646675, 1.0]
    p2 = [6.239745391849833e-09, 2.6580697468673755e-06, 0.00030158155350823543, 0.012371663481782003, 0.20148538954917908, 1.3330346081580755, 3.9388102529247444, 6.915228890689842, 3.2377489177694603]
    q2 = [6.790194080099813e-09, 2.8924786474538068e-06, 0.00032801446468212774, 0.013420400608854318, 0.21623699359449663, 1.3770209948908132, 3.6798356385616087, 6.02427039364742, 1.0]

    def _create_polynomial(var, coeffs):
        if False:
            i = 10
            return i + 15
        "Compute n_th order polynomial via Horner's method."
        coeffs = np.array(coeffs, var.dtype.as_numpy_dtype)
        if not coeffs.size:
            return array_ops.zeros_like(var)
        return coeffs[0] + _create_polynomial(var, coeffs[1:]) * var
    maybe_complement_p = array_ops.where_v2(p > -np.expm1(-2.0), 1.0 - p, p)
    sanitized_mcp = array_ops.where_v2(maybe_complement_p <= 0.0, array_ops.fill(array_ops.shape(p), np.array(0.5, p.dtype.as_numpy_dtype)), maybe_complement_p)
    w = sanitized_mcp - 0.5
    ww = w ** 2
    x_for_big_p = w + w * ww * (_create_polynomial(ww, p0) / _create_polynomial(ww, q0))
    x_for_big_p *= -np.sqrt(2.0 * np.pi)
    z = math_ops.sqrt(-2.0 * math_ops.log(sanitized_mcp))
    first_term = z - math_ops.log(z) / z
    second_term_small_p = _create_polynomial(1.0 / z, p2) / _create_polynomial(1.0 / z, q2) / z
    second_term_otherwise = _create_polynomial(1.0 / z, p1) / _create_polynomial(1.0 / z, q1) / z
    x_for_small_p = first_term - second_term_small_p
    x_otherwise = first_term - second_term_otherwise
    x = array_ops.where_v2(sanitized_mcp > np.exp(-2.0), x_for_big_p, array_ops.where_v2(z >= 8.0, x_for_small_p, x_otherwise))
    x = array_ops.where_v2(p > 1.0 - np.exp(-2.0), x, -x)
    infinity_scalar = constant_op.constant(np.inf, dtype=p.dtype)
    infinity = array_ops.fill(array_ops.shape(p), infinity_scalar)
    x_nan_replaced = array_ops.where_v2(p <= 0.0, -infinity, array_ops.where_v2(p >= 1.0, infinity, x))
    return x_nan_replaced

def log_ndtr(x, series_order=3, name='log_ndtr'):
    if False:
        print('Hello World!')
    'Log Normal distribution function.\n\n  For details of the Normal distribution function see `ndtr`.\n\n  This function calculates `(log o ndtr)(x)` by either calling `log(ndtr(x))` or\n  using an asymptotic series. Specifically:\n  - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on\n    `log(1-x) ~= -x, x << 1`.\n  - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique\n    and take a log.\n  - For `x <= lower_segment`, we use the series approximation of erf to compute\n    the log CDF directly.\n\n  The `lower_segment` is set based on the precision of the input:\n\n  ```\n  lower_segment = { -20,  x.dtype=float64\n                  { -10,  x.dtype=float32\n  upper_segment = {   8,  x.dtype=float64\n                  {   5,  x.dtype=float32\n  ```\n\n  When `x < lower_segment`, the `ndtr` asymptotic series approximation is:\n\n  ```\n     ndtr(x) = scale * (1 + sum) + R_N\n     scale   = exp(-0.5 x**2) / (-x sqrt(2 pi))\n     sum     = Sum{(-1)^n (2n-1)!! / (x**2)^n, n=1:N}\n     R_N     = O(exp(-0.5 x**2) (2N+1)!! / |x|^{2N+3})\n  ```\n\n  where `(2n-1)!! = (2n-1) (2n-3) (2n-5) ...  (3) (1)` is a\n  [double-factorial](https://en.wikipedia.org/wiki/Double_factorial).\n\n\n  Args:\n    x: `Tensor` of type `float32`, `float64`.\n    series_order: Positive Python `integer`. Maximum depth to\n      evaluate the asymptotic expansion. This is the `N` above.\n    name: Python string. A name for the operation (default="log_ndtr").\n\n  Returns:\n    log_ndtr: `Tensor` with `dtype=x.dtype`.\n\n  Raises:\n    TypeError: if `x.dtype` is not handled.\n    TypeError: if `series_order` is a not Python `integer.`\n    ValueError:  if `series_order` is not in `[0, 30]`.\n  '
    if not isinstance(series_order, int):
        raise TypeError('series_order must be a Python integer.')
    if series_order < 0:
        raise ValueError('series_order must be non-negative.')
    if series_order > 30:
        raise ValueError('series_order must be <= 30.')
    with ops.name_scope(name, values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        if x.dtype.as_numpy_dtype == np.float64:
            lower_segment = LOGNDTR_FLOAT64_LOWER
            upper_segment = LOGNDTR_FLOAT64_UPPER
        elif x.dtype.as_numpy_dtype == np.float32:
            lower_segment = LOGNDTR_FLOAT32_LOWER
            upper_segment = LOGNDTR_FLOAT32_UPPER
        else:
            raise TypeError('x.dtype=%s is not supported.' % x.dtype)
        return array_ops.where_v2(math_ops.greater(x, upper_segment), -_ndtr(-x), array_ops.where_v2(math_ops.greater(x, lower_segment), math_ops.log(_ndtr(math_ops.maximum(x, lower_segment))), _log_ndtr_lower(math_ops.minimum(x, lower_segment), series_order)))

def _log_ndtr_lower(x, series_order):
    if False:
        print('Hello World!')
    'Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`.'
    x_2 = math_ops.square(x)
    log_scale = -0.5 * x_2 - math_ops.log(-x) - 0.5 * np.log(2.0 * np.pi)
    return log_scale + math_ops.log(_log_ndtr_asymptotic_series(x, series_order))

def _log_ndtr_asymptotic_series(x, series_order):
    if False:
        while True:
            i = 10
    'Calculates the asymptotic series used in log_ndtr.'
    dtype = x.dtype.as_numpy_dtype
    if series_order <= 0:
        return np.array(1, dtype)
    x_2 = math_ops.square(x)
    even_sum = array_ops.zeros_like(x)
    odd_sum = array_ops.zeros_like(x)
    x_2n = x_2
    for n in range(1, series_order + 1):
        y = np.array(_double_factorial(2 * n - 1), dtype) / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n *= x_2
    return 1.0 + even_sum - odd_sum

def erfinv(x, name='erfinv'):
    if False:
        print('Hello World!')
    'The inverse function for erf, the error function.\n\n  Args:\n    x: `Tensor` of type `float32`, `float64`.\n    name: Python string. A name for the operation (default="erfinv").\n\n  Returns:\n    x: `Tensor` with `dtype=x.dtype`.\n\n  Raises:\n    TypeError: if `x` is not floating-type.\n  '
    with ops.name_scope(name, values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        if x.dtype.as_numpy_dtype not in [np.float32, np.float64]:
            raise TypeError('x.dtype=%s is not handled, see docstring for supported types.' % x.dtype)
        return ndtri((x + 1.0) / 2.0) / np.sqrt(2)

def _double_factorial(n):
    if False:
        while True:
            i = 10
    'The double factorial function for small Python integer `n`.'
    return np.prod(np.arange(n, 1, -2))

def log_cdf_laplace(x, name='log_cdf_laplace'):
    if False:
        return 10
    'Log Laplace distribution function.\n\n  This function calculates `Log[L(x)]`, where `L(x)` is the cumulative\n  distribution function of the Laplace distribution, i.e.\n\n  ```L(x) := 0.5 * int_{-infty}^x e^{-|t|} dt```\n\n  For numerical accuracy, `L(x)` is computed in different ways depending on `x`,\n\n  ```\n  x <= 0:\n    Log[L(x)] = Log[0.5] + x, which is exact\n\n  0 < x:\n    Log[L(x)] = Log[1 - 0.5 * e^{-x}], which is exact\n  ```\n\n  Args:\n    x: `Tensor` of type `float32`, `float64`.\n    name: Python string. A name for the operation (default="log_ndtr").\n\n  Returns:\n    `Tensor` with `dtype=x.dtype`.\n\n  Raises:\n    TypeError: if `x.dtype` is not handled.\n  '
    with ops.name_scope(name, values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        lower_solution = -np.log(2.0) + x
        safe_exp_neg_x = math_ops.exp(-math_ops.abs(x))
        upper_solution = math_ops.log1p(-0.5 * safe_exp_neg_x)
        return array_ops.where_v2(x < 0.0, lower_solution, upper_solution)