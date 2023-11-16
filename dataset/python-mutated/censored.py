import numpy as np
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from pymc.distributions.distribution import Distribution, SymbolicRandomVariable, _moment
from pymc.distributions.shape_utils import _change_dist_size, change_dist_size
from pymc.util import check_dist_not_registered

class CensoredRV(SymbolicRandomVariable):
    """Censored random variable"""
    inline_logprob = True
    _print_name = ('Censored', '\\operatorname{Censored}')

class Censored(Distribution):
    """
    Censored distribution

    The pdf of a censored distribution is

    .. math::

        \\begin{cases}
            0 & \\text{for } x < lower, \\\\
            \\text{CDF}(lower, dist) & \\text{for } x = lower, \\\\
            \\text{PDF}(x, dist) & \\text{for } lower < x < upper, \\\\
            1-\\text{CDF}(upper, dist) & \\text {for} x = upper, \\\\
            0 & \\text{for } x > upper,
        \\end{cases}


    Parameters
    ----------
    dist : unnamed_distribution
        Univariate distribution which will be censored.
        This distribution must have a logcdf method implemented for sampling.

        .. warning:: dist will be cloned, rendering it independent of the one passed as input.

    lower : float or None
        Lower (left) censoring point. If `None` the distribution will not be left censored
    upper : float or None
        Upper (right) censoring point. If `None`, the distribution will not be right censored.

    Warnings
    --------
    Continuous censored distributions should only be used as likelihoods.
    Continuous censored distributions are a form of discrete-continuous mixture
    and as such cannot be sampled properly without a custom step sampler.
    If you wish to sample such a distribution, you can add the latent uncensored
    distribution to the model and then wrap it in a :class:`~pymc.Deterministic`
    :func:`~pymc.math.clip`.


    Examples
    --------
    .. code-block:: python

        with pm.Model():
            normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
            censored_normal = pm.Censored("censored_normal", normal_dist, lower=-1, upper=1)
    """
    rv_type = CensoredRV

    @classmethod
    def dist(cls, dist, lower, upper, **kwargs):
        if False:
            print('Hello World!')
        if not isinstance(dist, TensorVariable) or not isinstance(dist.owner.op, (RandomVariable, SymbolicRandomVariable)):
            raise ValueError(f'Censoring dist must be a distribution created via the `.dist()` API, got {type(dist)}')
        if dist.owner.op.ndim_supp > 0:
            raise NotImplementedError('Censoring of multivariate distributions has not been implemented yet')
        check_dist_not_registered(dist)
        return super().dist([dist, lower, upper], **kwargs)

    @classmethod
    def rv_op(cls, dist, lower=None, upper=None, size=None):
        if False:
            i = 10
            return i + 15
        lower = pt.constant(-np.inf) if lower is None else pt.as_tensor_variable(lower)
        upper = pt.constant(np.inf) if upper is None else pt.as_tensor_variable(upper)
        dist_shape = size if size is not None else pt.broadcast_shape(dist, lower, upper)
        dist = change_dist_size(dist, dist_shape)
        (dist_, lower_, upper_) = (dist.type(), lower.type(), upper.type())
        censored_rv_ = pt.clip(dist_, lower_, upper_)
        return CensoredRV(inputs=[dist_, lower_, upper_], outputs=[censored_rv_], ndim_supp=0)(dist, lower, upper)

@_change_dist_size.register(CensoredRV)
def change_censored_size(cls, dist, new_size, expand=False):
    if False:
        while True:
            i = 10
    (uncensored_dist, lower, upper) = dist.owner.inputs
    if expand:
        new_size = tuple(new_size) + tuple(uncensored_dist.shape)
    return Censored.rv_op(uncensored_dist, lower, upper, size=new_size)

@_moment.register(CensoredRV)
def moment_censored(op, rv, dist, lower, upper):
    if False:
        return 10
    moment = pt.switch(pt.eq(lower, -np.inf), pt.switch(pt.isinf(upper), 0, upper - 1), pt.switch(pt.eq(upper, np.inf), lower + 1, (lower + upper) / 2))
    moment = pt.full_like(dist, moment)
    return moment