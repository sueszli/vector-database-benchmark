import numpy as np

def _make_index(prob, size):
    if False:
        print('Hello World!')
    '\n    Returns a boolean index for given probabilities.\n\n    Notes\n    -----\n    prob = [.75,.25] means that there is a 75% chance of the first column\n    being True and a 25% chance of the second column being True. The\n    columns are mutually exclusive.\n    '
    rv = np.random.uniform(size=(size, 1))
    cumprob = np.cumsum(prob)
    return np.logical_and(np.r_[0, cumprob[:-1]] <= rv, rv < cumprob)

def mixture_rvs(prob, size, dist, kwargs=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sample from a mixture of distributions.\n\n    Parameters\n    ----------\n    prob : array_like\n        Probability of sampling from each distribution in dist\n    size : int\n        The length of the returned sample.\n    dist : array_like\n        An iterable of distributions objects from scipy.stats.\n    kwargs : tuple of dicts, optional\n        A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and\n        args to be passed to the respective distribution in dist.  If not\n        provided, the distribution defaults are used.\n\n    Examples\n    --------\n    Say we want 5000 random variables from mixture of normals with two\n    distributions norm(-1,.5) and norm(1,.5) and we want to sample from the\n    first with probability .75 and the second with probability .25.\n\n    >>> from scipy import stats\n    >>> prob = [.75,.25]\n    >>> Y = mixture_rvs(prob, 5000, dist=[stats.norm, stats.norm],\n    ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))\n    '
    if len(prob) != len(dist):
        raise ValueError('You must provide as many probabilities as distributions')
    if not np.allclose(np.sum(prob), 1):
        raise ValueError('prob does not sum to 1')
    if kwargs is None:
        kwargs = ({},) * len(prob)
    idx = _make_index(prob, size)
    sample = np.empty(size)
    for i in range(len(prob)):
        sample_idx = idx[..., i]
        sample_size = sample_idx.sum()
        loc = kwargs[i].get('loc', 0)
        scale = kwargs[i].get('scale', 1)
        args = kwargs[i].get('args', ())
        sample[sample_idx] = dist[i].rvs(*args, **dict(loc=loc, scale=scale, size=sample_size))
    return sample

class MixtureDistribution:
    """univariate mixture distribution

    for simple case for now (unbound support)
    does not yet inherit from scipy.stats.distributions

    adding pdf to mixture_rvs, some restrictions on broadcasting
    Currently it does not hold any state, all arguments included in each method.
    """

    def rvs(self, prob, size, dist, kwargs=None):
        if False:
            return 10
        return mixture_rvs(prob, size, dist, kwargs=kwargs)

    def pdf(self, x, prob, dist, kwargs=None):
        if False:
            while True:
                i = 10
        '\n        pdf a mixture of distributions.\n\n        Parameters\n        ----------\n        x : array_like\n            Array containing locations where the PDF should be evaluated\n        prob : array_like\n            Probability of sampling from each distribution in dist\n        dist : array_like\n            An iterable of distributions objects from scipy.stats.\n        kwargs : tuple of dicts, optional\n            A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and\n            args to be passed to the respective distribution in dist.  If not\n            provided, the distribution defaults are used.\n\n        Examples\n        --------\n        Say we want 5000 random variables from mixture of normals with two\n        distributions norm(-1,.5) and norm(1,.5) and we want to sample from the\n        first with probability .75 and the second with probability .25.\n\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> from statsmodels.distributions.mixture_rvs import MixtureDistribution\n        >>> x = np.arange(-4.0, 4.0, 0.01)\n        >>> prob = [.75,.25]\n        >>> mixture = MixtureDistribution()\n        >>> Y = mixture.pdf(x, prob, dist=[stats.norm, stats.norm],\n        ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))\n        '
        if len(prob) != len(dist):
            raise ValueError('You must provide as many probabilities as distributions')
        if not np.allclose(np.sum(prob), 1):
            raise ValueError('prob does not sum to 1')
        if kwargs is None:
            kwargs = ({},) * len(prob)
        for i in range(len(prob)):
            loc = kwargs[i].get('loc', 0)
            scale = kwargs[i].get('scale', 1)
            args = kwargs[i].get('args', ())
            if i == 0:
                pdf_ = prob[i] * dist[i].pdf(x, *args, loc=loc, scale=scale)
            else:
                pdf_ += prob[i] * dist[i].pdf(x, *args, loc=loc, scale=scale)
        return pdf_

    def cdf(self, x, prob, dist, kwargs=None):
        if False:
            return 10
        '\n        cdf of a mixture of distributions.\n\n        Parameters\n        ----------\n        x : array_like\n            Array containing locations where the CDF should be evaluated\n        prob : array_like\n            Probability of sampling from each distribution in dist\n        size : int\n            The length of the returned sample.\n        dist : array_like\n            An iterable of distributions objects from scipy.stats.\n        kwargs : tuple of dicts, optional\n            A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and\n            args to be passed to the respective distribution in dist.  If not\n            provided, the distribution defaults are used.\n\n        Examples\n        --------\n        Say we want 5000 random variables from mixture of normals with two\n        distributions norm(-1,.5) and norm(1,.5) and we want to sample from the\n        first with probability .75 and the second with probability .25.\n\n        >>> import numpy as np\n        >>> from scipy import stats\n        >>> from statsmodels.distributions.mixture_rvs import MixtureDistribution\n        >>> x = np.arange(-4.0, 4.0, 0.01)\n        >>> prob = [.75,.25]\n        >>> mixture = MixtureDistribution()\n        >>> Y = mixture.pdf(x, prob, dist=[stats.norm, stats.norm],\n        ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))\n        '
        if len(prob) != len(dist):
            raise ValueError('You must provide as many probabilities as distributions')
        if not np.allclose(np.sum(prob), 1):
            raise ValueError('prob does not sum to 1')
        if kwargs is None:
            kwargs = ({},) * len(prob)
        for i in range(len(prob)):
            loc = kwargs[i].get('loc', 0)
            scale = kwargs[i].get('scale', 1)
            args = kwargs[i].get('args', ())
            if i == 0:
                cdf_ = prob[i] * dist[i].cdf(x, *args, loc=loc, scale=scale)
            else:
                cdf_ += prob[i] * dist[i].cdf(x, *args, loc=loc, scale=scale)
        return cdf_

def mv_mixture_rvs(prob, size, dist, nvars, **kwargs):
    if False:
        print('Hello World!')
    '\n    Sample from a mixture of multivariate distributions.\n\n    Parameters\n    ----------\n    prob : array_like\n        Probability of sampling from each distribution in dist\n    size : int\n        The length of the returned sample.\n    dist : array_like\n        An iterable of distributions instances with callable method rvs.\n    nvargs : int\n        dimension of the multivariate distribution, could be inferred instead\n    kwargs : tuple of dicts, optional\n        ignored\n\n    Examples\n    --------\n    Say we want 2000 random variables from mixture of normals with two\n    multivariate normal distributions, and we want to sample from the\n    first with probability .4 and the second with probability .6.\n\n    import statsmodels.sandbox.distributions.mv_normal as mvd\n\n    cov3 = np.array([[ 1.  ,  0.5 ,  0.75],\n                       [ 0.5 ,  1.5 ,  0.6 ],\n                       [ 0.75,  0.6 ,  2.  ]])\n\n    mu = np.array([-1, 0.0, 2.0])\n    mu2 = np.array([4, 2.0, 2.0])\n    mvn3 = mvd.MVNormal(mu, cov3)\n    mvn32 = mvd.MVNormal(mu2, cov3/2., 4)\n    rvs = mix.mv_mixture_rvs([0.4, 0.6], 2000, [mvn3, mvn32], 3)\n    '
    if len(prob) != len(dist):
        raise ValueError('You must provide as many probabilities as distributions')
    if not np.allclose(np.sum(prob), 1):
        raise ValueError('prob does not sum to 1')
    if kwargs is None:
        kwargs = ({},) * len(prob)
    idx = _make_index(prob, size)
    sample = np.empty((size, nvars))
    for i in range(len(prob)):
        sample_idx = idx[..., i]
        sample_size = sample_idx.sum()
        sample[sample_idx] = dist[i].rvs(size=int(sample_size))
    return sample
if __name__ == '__main__':
    from scipy import stats
    obs_dist = mixture_rvs([0.25, 0.75], size=10000, dist=[stats.norm, stats.beta], kwargs=(dict(loc=-1, scale=0.5), dict(loc=1, scale=1, args=(1, 0.5))))
    nobs = 10000
    mix = MixtureDistribution()
    mix_kwds = (dict(loc=-1, scale=0.25), dict(loc=1, scale=0.75))
    mrvs = mix.rvs([1 / 3.0, 2 / 3.0], size=nobs, dist=[stats.norm, stats.norm], kwargs=mix_kwds)
    grid = np.linspace(-4, 4, 100)
    mpdf = mix.pdf(grid, [1 / 3.0, 2 / 3.0], dist=[stats.norm, stats.norm], kwargs=mix_kwds)
    mcdf = mix.cdf(grid, [1 / 3.0, 2 / 3.0], dist=[stats.norm, stats.norm], kwargs=mix_kwds)
    doplot = 1
    if doplot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(mrvs, bins=50, normed=True, color='red')
        plt.title('histogram of sample and pdf')
        plt.plot(grid, mpdf, lw=2, color='black')
        plt.figure()
        plt.hist(mrvs, bins=50, normed=True, cumulative=True, color='red')
        plt.title('histogram of sample and pdf')
        plt.plot(grid, mcdf, lw=2, color='black')
        plt.show()