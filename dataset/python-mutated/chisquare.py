import numpy
import chainer
from chainer.backends import cuda
from chainer import distribution
from chainer.functions.math import digamma
from chainer.functions.math import exponential
from chainer.functions.math import lgamma
from chainer.utils import cache

class Chisquare(distribution.Distribution):
    """Chi-Square Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;k) = \\frac{1}{2^{k/2}\\Gamma(k/2)}x^{k/2-1}e^{-x/2}

    Args:
        k(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution.
    """

    def __init__(self, k):
        if False:
            for i in range(10):
                print('nop')
        super(Chisquare, self).__init__()
        self.__k = k

    @cache.cached_property
    def k(self):
        if False:
            for i in range(10):
                print('nop')
        return chainer.as_variable(self.__k)

    @cache.cached_property
    def _half_k(self):
        if False:
            while True:
                i = 10
        return 0.5 * self.k

    @property
    def batch_shape(self):
        if False:
            while True:
                i = 10
        return self.k.shape

    @cache.cached_property
    def entropy(self):
        if False:
            print('Hello World!')
        return self._half_k + numpy.log(2.0) + lgamma.lgamma(self._half_k) + (1 - self._half_k) * digamma.digamma(self._half_k)

    @property
    def event_shape(self):
        if False:
            while True:
                i = 10
        return ()

    def log_prob(self, x):
        if False:
            while True:
                i = 10
        return -lgamma.lgamma(self._half_k) - self._half_k * numpy.log(2.0) + (self._half_k - 1) * exponential.log(x) - 0.5 * x

    @cache.cached_property
    def mean(self):
        if False:
            return 10
        return self.k

    @property
    def params(self):
        if False:
            i = 10
            return i + 15
        return {'k': self.k}

    def sample_n(self, n):
        if False:
            print('Hello World!')
        xp = chainer.backend.get_array_module(self.k)
        if xp is cuda.cupy:
            eps = xp.random.chisquare(self.k.data, (n,) + self.k.shape, dtype=self.k.dtype)
        else:
            eps = xp.random.chisquare(self.k.data, (n,) + self.k.shape).astype(self.k.dtype)
        noise = chainer.Variable(eps)
        return noise

    @property
    def support(self):
        if False:
            print('Hello World!')
        return 'positive'

    @cache.cached_property
    def variance(self):
        if False:
            for i in range(10):
                print('nop')
        return 2 * self.k