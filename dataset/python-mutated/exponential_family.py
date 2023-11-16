import paddle
from paddle.distribution import distribution
from paddle.framework import in_dynamic_mode

class ExponentialFamily(distribution.Distribution):
    """
    ExponentialFamily is the base class for probability distributions belonging
    to exponential family, whose probability mass/density function has the
    form is defined below

    ExponentialFamily is derived from `paddle.distribution.Distribution`.

    .. math::

        f_{F}(x; \\theta) = \\exp(\\langle t(x), \\theta\\rangle - F(\\theta) + k(x))

    where :math:`\\theta` denotes the natural parameters, :math:`t(x)` denotes
    the sufficient statistic, :math:`F(\\theta)` is the log normalizer function
    for a given family and :math:`k(x)` is the carrier measure.

    Distribution belongs to exponential family referring to https://en.wikipedia.org/wiki/Exponential_family
    """

    @property
    def _natural_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _log_normalizer(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def entropy(self):
        if False:
            print('Hello World!')
        'caculate entropy use `bregman divergence`\n        https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf\n        '
        entropy_value = -self._mean_carrier_measure
        natural_parameters = []
        for parameter in self._natural_parameters:
            parameter = parameter.detach()
            parameter.stop_gradient = False
            natural_parameters.append(parameter)
        log_norm = self._log_normalizer(*natural_parameters)
        if in_dynamic_mode():
            grads = paddle.grad(log_norm.sum(), natural_parameters, create_graph=True)
        else:
            grads = paddle.static.gradients(log_norm.sum(), natural_parameters)
        entropy_value += log_norm
        for (p, g) in zip(natural_parameters, grads):
            entropy_value -= p * g
        return entropy_value