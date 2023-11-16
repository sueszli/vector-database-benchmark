"""
Created on Mar 7, 2011

@author: johnsalvatier
"""
import warnings
from functools import partial
from typing import Iterable
import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.linalg
import scipy.stats
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.scalar import UnaryScalarOp, upgrade_to_float_no_complex
from pytensor.tensor import gammaln
from pytensor.tensor.elemwise import Elemwise
from pymc.distributions.shape_utils import to_tuple
from pymc.logprob.utils import CheckParameterValue
from pymc.pytensorf import floatX
f = floatX
c = -0.5 * np.log(2.0 * np.pi)
_beta_clip_values = {dtype: (np.nextafter(0, 1, dtype=dtype), np.nextafter(1, 0, dtype=dtype)) for dtype in ['float16', 'float32', 'float64']}

def check_parameters(expr: Variable, *conditions: Iterable[Variable], msg: str='', can_be_replaced_by_ninf: bool=True):
    if False:
        i = 10
        return i + 15
    'Wrap an expression in a CheckParameterValue that asserts several conditions are met.\n\n    When conditions are not met a ParameterValueError assertion is raised,\n    with an optional custom message defined by `msg`.\n\n    When the flag `can_be_replaced_by_ninf` is True (default), PyMC is allowed to replace the\n    assertion by a switch(condition, expr, -inf). This is used for logp graphs!\n\n    Note that check_parameter should not be used to enforce the logic of the\n    expression under the normal parameter support as it can be disabled by the user via\n    check_bounds = False in pm.Model()\n    '
    conditions_ = [cond if cond is not True and cond is not False else np.array(cond) for cond in conditions]
    all_true_scalar = pt.all([pt.all(cond) for cond in conditions_])
    return CheckParameterValue(msg, can_be_replaced_by_ninf)(expr, all_true_scalar)
check_icdf_parameters = partial(check_parameters, can_be_replaced_by_ninf=False)

def check_icdf_value(expr: Variable, value: Variable) -> Variable:
    if False:
        for i in range(10):
            print('nop')
    'Wrap icdf expression in nan switch for value.'
    value = pt.as_tensor_variable(value)
    expr = pt.switch(pt.and_(value >= 0, value <= 1), expr, np.nan)
    expr.name = '0 <= value <= 1'
    return expr

def logpow(x, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates log(x**m) since m*log(x) will fail when m, x = 0.\n    '
    return pt.switch(pt.eq(x, 0), pt.switch(pt.eq(m, 0), 0.0, -np.inf), m * pt.log(x))

def factln(n):
    if False:
        while True:
            i = 10
    return gammaln(n + 1)

def binomln(n, k):
    if False:
        return 10
    return factln(n) - factln(k) - factln(n - k)

def betaln(x, y):
    if False:
        for i in range(10):
            print('nop')
    return gammaln(x) + gammaln(y) - gammaln(x + y)

def std_cdf(x):
    if False:
        while True:
            i = 10
    '\n    Calculates the standard normal cumulative distribution function.\n    '
    return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2.0))

def normal_lcdf(mu, sigma, x):
    if False:
        return 10
    'Compute the log of the cumulative density function of the normal.'
    z = (x - mu) / sigma
    return pt.switch(pt.lt(z, -1.0), pt.log(pt.erfcx(-z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0, pt.log1p(-pt.erfc(z / pt.sqrt(2.0)) / 2.0))

def normal_lccdf(mu, sigma, x):
    if False:
        print('Hello World!')
    z = (x - mu) / sigma
    return pt.switch(pt.gt(z, 1.0), pt.log(pt.erfcx(z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0, pt.log1p(-pt.erfc(-z / pt.sqrt(2.0)) / 2.0))

def log_diff_normal_cdf(mu, sigma, x, y):
    if False:
        print('Hello World!')
    '\n    Compute :math:`\\log(\\Phi(\x0crac{x - \\mu}{\\sigma}) - \\Phi(\x0crac{y - \\mu}{\\sigma}))` safely in log space.\n\n    Parameters\n    ----------\n    mu: float\n        mean\n    sigma: float\n        std\n\n    x: float\n\n    y: float\n        must be strictly less than x.\n\n    Returns\n    -------\n    log (\\Phi(x) - \\Phi(y))\n\n    '
    x = (x - mu) / sigma / pt.sqrt(2.0)
    y = (y - mu) / sigma / pt.sqrt(2.0)
    return pt.log(0.5) + pt.switch(pt.gt(y, 0), -pt.square(y) + pt.log(pt.erfcx(y) - pt.exp(pt.square(y) - pt.square(x)) * pt.erfcx(x)), pt.switch(pt.lt(x, 0), -pt.square(x) + pt.log(pt.erfcx(-x) - pt.exp(pt.square(x) - pt.square(y)) * pt.erfcx(-y)), pt.log(pt.erf(x) - pt.erf(y))))

def sigma2rho(sigma):
    if False:
        print('Hello World!')
    '\n    `sigma -> rho` PyTensor converter\n    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`'
    return pt.log(pt.exp(pt.abs(sigma)) - 1.0)

def rho2sigma(rho):
    if False:
        print('Hello World!')
    '\n    `rho -> sigma` PyTensor converter\n    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`'
    return pt.softplus(rho)
rho2sd = rho2sigma
sd2rho = sigma2rho

def log_normal(x, mean, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate logarithm of normal distribution at point `x`\n    with given `mean` and `std`\n\n    Parameters\n    ----------\n    x: Tensor\n        point of evaluation\n    mean: Tensor\n        mean of normal distribution\n    kwargs: one of parameters `{sigma, tau, w, rho}`\n\n    Notes\n    -----\n    There are four variants for density parametrization.\n    They are:\n        1) standard deviation - `std`\n        2) `w`, logarithm of `std` :math:`w = log(std)`\n        3) `rho` that follows this equation :math:`rho = log(exp(std) - 1)`\n        4) `tau` that follows this equation :math:`tau = std^{-1}`\n    ----\n    '
    sigma = kwargs.get('sigma')
    w = kwargs.get('w')
    rho = kwargs.get('rho')
    tau = kwargs.get('tau')
    eps = kwargs.get('eps', 0.0)
    check = sum(map(lambda a: a is not None, [sigma, w, rho, tau]))
    if check > 1:
        raise ValueError('more than one required kwarg is passed')
    if check == 0:
        raise ValueError('none of required kwarg is passed')
    if sigma is not None:
        std = sigma
    elif w is not None:
        std = pt.exp(w)
    elif rho is not None:
        std = rho2sigma(rho)
    else:
        std = tau ** (-1)
    std += f(eps)
    return f(c) - pt.log(pt.abs(std)) - (x - mean) ** 2 / (2.0 * std ** 2)

class SplineWrapper(Op):
    """
    Creates an PyTensor operation from scipy.interpolate.UnivariateSpline
    """
    __props__ = ('spline',)

    def __init__(self, spline):
        if False:
            return 10
        self.spline = spline

    def make_node(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    @property
    def grad_op(self):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, '_grad_op'):
            try:
                self._grad_op = SplineWrapper(self.spline.derivative())
            except ValueError:
                self._grad_op = None
        if self._grad_op is None:
            raise NotImplementedError('Spline of order 0 is not differentiable')
        return self._grad_op

    def perform(self, node, inputs, output_storage):
        if False:
            while True:
                i = 10
        (x,) = inputs
        output_storage[0][0] = np.asarray(self.spline(x), dtype=x.dtype)

    def grad(self, inputs, grads):
        if False:
            return 10
        (x,) = inputs
        (x_grad,) = grads
        return [x_grad * self.grad_op(x)]

class I1e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 1, exponentially scaled.
    """
    nfunc_spec = ('scipy.special.i1e', 1, 1)

    def impl(self, x):
        if False:
            print('Hello World!')
        return scipy.special.i1e(x)
i1e_scalar = I1e(upgrade_to_float_no_complex, name='i1e')
i1e = Elemwise(i1e_scalar, name='Elemwise{i1e,no_inplace}')

class I0e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 0, exponentially scaled.
    """
    nfunc_spec = ('scipy.special.i0e', 1, 1)

    def impl(self, x):
        if False:
            print('Hello World!')
        return scipy.special.i0e(x)

    def grad(self, inp, grads):
        if False:
            i = 10
            return i + 15
        (x,) = inp
        (gz,) = grads
        return (gz * (i1e_scalar(x) - pytensor.scalar.sign(x) * i0e_scalar(x)),)
i0e_scalar = I0e(upgrade_to_float_no_complex, name='i0e')
i0e = Elemwise(i0e_scalar, name='Elemwise{i0e,no_inplace}')

def random_choice(p, size):
    if False:
        while True:
            i = 10
    'Return draws from categorical probability functions\n\n    Parameters\n    ----------\n    p : array\n        Probability of each class. If p.ndim > 1, the last axis is\n        interpreted as the probability of each class, and numpy.random.choice\n        is iterated for every other axis element.\n    size : int or tuple\n        Shape of the desired output array. If p is multidimensional, size\n        should broadcast with p.shape[:-1].\n\n    Returns\n    -------\n    random_sample : array\n\n    '
    k = p.shape[-1]
    if p.ndim > 1:
        size = to_tuple(size) + (1,)
        p = np.broadcast_arrays(p, np.empty(size))[0]
        out_shape = p.shape[:-1]
        p = np.reshape(p, (-1, p.shape[-1]))
        samples = np.array([np.random.choice(k, p=p_) for p_ in p])
        samples = np.reshape(samples, out_shape)
    else:
        samples = np.random.choice(k, p=p, size=size)
    return samples

def zvalue(value, sigma, mu):
    if False:
        print('Hello World!')
    '\n    Calculate the z-value for a normal distribution.\n    '
    return (value - mu) / sigma

def clipped_beta_rvs(a, b, size=None, random_state=None, dtype='float64'):
    if False:
        return 10
    'Draw beta distributed random samples in the open :math:`(0, 1)` interval.\n\n    The samples are generated with ``scipy.stats.beta.rvs``, but any value that\n    is equal to 0 or 1 will be shifted towards the next floating point in the\n    interval :math:`[0, 1]`, depending on the floating point precision that is\n    given by ``dtype``.\n\n    Parameters\n    ----------\n    a : float or array_like of floats\n        Alpha, strictly positive (>0).\n    b : float or array_like of floats\n        Beta, strictly positive (>0).\n    size : int or tuple of ints, optional\n        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then\n        ``m * n * k`` samples are drawn.  If size is ``None`` (default),\n        a single value is returned if ``a`` and ``b`` are both scalars.\n        Otherwise, ``np.broadcast(a, b).size`` samples are drawn.\n    dtype : str or dtype instance\n        The floating point precision that the samples should have. This also\n        determines the value that will be used to shift any samples returned\n        by the numpy random number generator that are zero or one.\n\n    Returns\n    -------\n    out : ndarray or scalar\n        Drawn samples from the parameterized beta distribution. The scipy\n        implementation can yield values that are equal to zero or one. We\n        assume the support of the Beta distribution to be in the open interval\n        :math:`(0, 1)`, so we shift any sample that is equal to 0 to\n        ``np.nextafter(0, 1, dtype=dtype)`` and any sample that is equal to 1\n        is shifted to ``np.nextafter(1, 0, dtype=dtype)``.\n\n    '
    out = scipy.stats.beta.rvs(a, b, size=size, random_state=random_state).astype(dtype)
    (lower, upper) = _beta_clip_values[dtype]
    return np.maximum(np.minimum(out, upper), lower)

def multigammaln(a, p):
    if False:
        return 10
    'Multivariate Log Gamma\n\n    Parameters\n    ----------\n    a: tensor like\n    p: int\n       degrees of freedom. p > 0\n    '
    i = pt.arange(1, p + 1)
    return p * (p - 1) * pt.log(np.pi) / 4.0 + pt.sum(gammaln(a + (1.0 - i) / 2.0), axis=0)

def log_i0(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the logarithm of the 0 order modified Bessel function of the first kind""\n    '
    return pt.switch(pt.lt(x, 5), pt.log1p(x ** 2.0 / 4.0 + x ** 4.0 / 64.0 + x ** 6.0 / 2304.0 + x ** 8.0 / 147456.0 + x ** 10.0 / 14745600.0 + x ** 12.0 / 2123366400.0), x - 0.5 * pt.log(2.0 * np.pi * x) + pt.log1p(1.0 / (8.0 * x) + 9.0 / (128.0 * x ** 2.0) + 225.0 / (3072.0 * x ** 3.0) + 11025.0 / (98304.0 * x ** 4.0)))

def incomplete_beta(a, b, value):
    if False:
        i = 10
        return i + 15
    warnings.warn('incomplete_beta has been deprecated. Use pytensor.tensor.betainc instead.', FutureWarning, stacklevel=2)
    return pt.betainc(a, b, value)