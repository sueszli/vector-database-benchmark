"""Bayesian Blocks for Time Series Analysis.

Bayesian Blocks for Time Series Analysis
========================================

Dynamic programming algorithm for solving a piecewise-constant model for
various datasets. This is based on the algorithm presented in Scargle
et al 2013 [1]_. This code was ported from the astroML project [2]_.

Applications include:

- finding an optimal histogram with adaptive bin widths
- finding optimal segmentation of time series data
- detecting inflection points in the rate of event data

The primary interface to these routines is the :func:`bayesian_blocks`
function. This module provides fitness functions suitable for three types
of data:

- Irregularly-spaced event data via the :class:`Events` class
- Regularly-spaced event data via the :class:`RegularEvents` class
- Irregularly-spaced point measurements via the :class:`PointMeasures` class

For more fine-tuned control over the fitness functions used, it is possible
to define custom :class:`FitnessFunc` classes directly and use them with
the :func:`bayesian_blocks` routine.

One common application of the Bayesian Blocks algorithm is the determination
of optimal adaptive-width histogram bins. This uses the same fitness function
as for irregularly-spaced time series events. The easiest interface for
creating Bayesian Blocks histograms is the :func:`astropy.stats.histogram`
function.

References
----------
.. [1] https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S
.. [2] https://www.astroml.org/ https://github.com//astroML/astroML/
.. [3] Bellman, R.E., Dreyfus, S.E., 1962. Applied Dynamic
   Programming. Princeton University Press, Princeton.
   https://press.princeton.edu/books/hardcover/9780691651873/applied-dynamic-programming
.. [4] Bellman, R., Roth, R., 1969. Curve fitting by segmented
   straight lines. J. Amer. Statist. Assoc. 64, 1079–1084.
   https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501038
"""
import warnings
from inspect import signature
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
__all__ = ['FitnessFunc', 'Events', 'RegularEvents', 'PointMeasures', 'bayesian_blocks']

def bayesian_blocks(t, x=None, sigma=None, fitness='events', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Compute optimal segmentation of data with Scargle's Bayesian Blocks.\n\n    This is a flexible implementation of the Bayesian Blocks algorithm\n    described in Scargle 2013 [1]_.\n\n    Parameters\n    ----------\n    t : array-like\n        data times (one dimensional, length N)\n    x : array-like, optional\n        data values\n    sigma : array-like or float, optional\n        data errors\n    fitness : str or object\n        the fitness function to use for the model.\n        If a string, the following options are supported:\n\n        - 'events' : binned or unbinned event data.  Arguments are ``gamma``,\n          which gives the slope of the prior on the number of bins, or\n          ``ncp_prior``, which is :math:`-\\ln({\\tt gamma})`.\n        - 'regular_events' : non-overlapping events measured at multiples of a\n          fundamental tick rate, ``dt``, which must be specified as an\n          additional argument.  Extra arguments are ``p0``, which gives the\n          false alarm probability to compute the prior, or ``gamma``, which\n          gives the slope of the prior on the number of bins, or ``ncp_prior``,\n          which is :math:`-\\ln({\\tt gamma})`.\n        - 'measures' : fitness for a measured sequence with Gaussian errors.\n          Extra arguments are ``p0``, which gives the false alarm probability\n          to compute the prior, or ``gamma``, which gives the slope of the\n          prior on the number of bins, or ``ncp_prior``, which is\n          :math:`-\\ln({\\tt gamma})`.\n\n        In all three cases, if more than one of ``p0``, ``gamma``, and\n        ``ncp_prior`` is chosen, ``ncp_prior`` takes precedence over ``gamma``\n        which takes precedence over ``p0``.\n\n        Alternatively, the fitness parameter can be an instance of\n        :class:`FitnessFunc` or a subclass thereof.\n\n    **kwargs :\n        any additional keyword arguments will be passed to the specified\n        :class:`FitnessFunc` derived class.\n\n    Returns\n    -------\n    edges : ndarray\n        array containing the (N+1) edges defining the N bins\n\n    Examples\n    --------\n    .. testsetup::\n\n        >>> np.random.seed(12345)\n\n    Event data:\n\n    >>> t = np.random.normal(size=100)\n    >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)\n\n    Event data with repeats:\n\n    >>> t = np.random.normal(size=100)\n    >>> t[80:] = t[:20]\n    >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)\n\n    Regular event data:\n\n    >>> dt = 0.05\n    >>> t = dt * np.arange(1000)\n    >>> x = np.zeros(len(t))\n    >>> x[np.random.randint(0, len(t), len(t) // 10)] = 1\n    >>> edges = bayesian_blocks(t, x, fitness='regular_events', dt=dt)\n\n    Measured point data with errors:\n\n    >>> t = 100 * np.random.random(100)\n    >>> x = np.exp(-0.5 * (t - 50) ** 2)\n    >>> sigma = 0.1\n    >>> x_obs = np.random.normal(x, sigma)\n    >>> edges = bayesian_blocks(t, x_obs, sigma, fitness='measures')\n\n    References\n    ----------\n    .. [1] Scargle, J et al. (2013)\n       https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S\n\n    .. [2] Bellman, R.E., Dreyfus, S.E., 1962. Applied Dynamic\n       Programming. Princeton University Press, Princeton.\n       https://press.princeton.edu/books/hardcover/9780691651873/applied-dynamic-programming\n\n    .. [3] Bellman, R., Roth, R., 1969. Curve fitting by segmented\n       straight lines. J. Amer. Statist. Assoc. 64, 1079–1084.\n       https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501038\n\n    See Also\n    --------\n    astropy.stats.histogram : compute a histogram using bayesian blocks\n    "
    FITNESS_DICT = {'events': Events, 'regular_events': RegularEvents, 'measures': PointMeasures}
    fitness = FITNESS_DICT.get(fitness, fitness)
    if type(fitness) is type and issubclass(fitness, FitnessFunc):
        fitfunc = fitness(**kwargs)
    elif isinstance(fitness, FitnessFunc):
        fitfunc = fitness
    else:
        raise ValueError('fitness parameter not understood')
    return fitfunc.fit(t, x, sigma)

class FitnessFunc:
    """Base class for bayesian blocks fitness functions.

    Derived classes should overload the following method:

    ``fitness(self, **kwargs)``:
      Compute the fitness given a set of named arguments.
      Arguments accepted by fitness must be among ``[T_k, N_k, a_k, b_k, c_k]``
      (See [1]_ for details on the meaning of these parameters).

    Additionally, other methods may be overloaded as well:

    ``__init__(self, **kwargs)``:
      Initialize the fitness function with any parameters beyond the normal
      ``p0`` and ``gamma``.

    ``validate_input(self, t, x, sigma)``:
      Enable specific checks of the input data (``t``, ``x``, ``sigma``)
      to be performed prior to the fit.

    ``compute_ncp_prior(self, N)``: If ``ncp_prior`` is not defined explicitly,
      this function is called in order to define it before fitting. This may be
      calculated from ``gamma``, ``p0``, or whatever method you choose.

    ``p0_prior(self, N)``:
      Specify the form of the prior given the false-alarm probability ``p0``
      (See [1]_ for details).

    For examples of implemented fitness functions, see :class:`Events`,
    :class:`RegularEvents`, and :class:`PointMeasures`.

    References
    ----------
    .. [1] Scargle, J et al. (2013)
       https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        if False:
            while True:
                i = 10
        self.p0 = p0
        self.gamma = gamma
        self.ncp_prior = ncp_prior

    def validate_input(self, t, x=None, sigma=None):
        if False:
            for i in range(10):
                print('nop')
        'Validate inputs to the model.\n\n        Parameters\n        ----------\n        t : array-like\n            times of observations\n        x : array-like, optional\n            values observed at each time\n        sigma : float or array-like, optional\n            errors in values x\n\n        Returns\n        -------\n        t, x, sigma : array-like, float or None\n            validated and perhaps modified versions of inputs\n        '
        t = np.asarray(t, dtype=float)
        t = np.array(t)
        if t.ndim != 1:
            raise ValueError('t must be a one-dimensional array')
        (unq_t, unq_ind, unq_inv) = np.unique(t, return_index=True, return_inverse=True)
        if x is None:
            if sigma is not None:
                raise ValueError('If sigma is specified, x must be specified')
            else:
                sigma = 1
            if len(unq_t) == len(t):
                x = np.ones_like(t)
            else:
                x = np.bincount(unq_inv)
            t = unq_t
        else:
            x = np.asarray(x, dtype=float)
            if x.shape not in [(), (1,), (t.size,)]:
                raise ValueError('x does not match shape of t')
            x += np.zeros_like(t)
            if len(unq_t) != len(t):
                raise ValueError('Repeated values in t not supported when x is specified')
            t = unq_t
            x = x[unq_ind]
        if sigma is None:
            sigma = 1
        else:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.shape not in [(), (1,), (t.size,)]:
                raise ValueError('sigma does not match the shape of x')
        return (t, x, sigma)

    def fitness(self, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def p0_prior(self, N):
        if False:
            i = 10
            return i + 15
        'Empirical prior, parametrized by the false alarm probability ``p0``.\n\n        See eq. 21 in Scargle (2013).\n\n        Note that there was an error in this equation in the original Scargle\n        paper (the "log" was missing). The following corrected form is taken\n        from https://arxiv.org/abs/1304.2818\n        '
        return 4 - np.log(73.53 * self.p0 * N ** (-0.478))

    @property
    def _fitness_args(self):
        if False:
            for i in range(10):
                print('nop')
        return signature(self.fitness).parameters.keys()

    def compute_ncp_prior(self, N):
        if False:
            print('Hello World!')
        '\n        If ``ncp_prior`` is not explicitly defined, compute it from ``gamma``\n        or ``p0``.\n        '
        if self.gamma is not None:
            return -np.log(self.gamma)
        elif self.p0 is not None:
            return self.p0_prior(N)
        else:
            raise ValueError('``ncp_prior`` cannot be computed as neither ``gamma`` nor ``p0`` is defined.')

    def fit(self, t, x=None, sigma=None):
        if False:
            i = 10
            return i + 15
        'Fit the Bayesian Blocks model given the specified fitness function.\n\n        Parameters\n        ----------\n        t : array-like\n            data times (one dimensional, length N)\n        x : array-like, optional\n            data values\n        sigma : array-like or float, optional\n            data errors\n\n        Returns\n        -------\n        edges : ndarray\n            array containing the (M+1) edges defining the M optimal bins\n        '
        (t, x, sigma) = self.validate_input(t, x, sigma)
        if 'a_k' in self._fitness_args:
            ak_raw = np.ones_like(x) / sigma ** 2
        if 'b_k' in self._fitness_args:
            bk_raw = x / sigma ** 2
        if 'c_k' in self._fitness_args:
            ck_raw = x * x / sigma ** 2
        edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
        block_length = t[-1] - edges
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)
        if self.ncp_prior is None:
            ncp_prior = self.compute_ncp_prior(N)
        else:
            ncp_prior = self.ncp_prior
        for R in range(N):
            kwds = {}
            if 'T_k' in self._fitness_args:
                kwds['T_k'] = block_length[:R + 1] - block_length[R + 1]
            if 'N_k' in self._fitness_args:
                kwds['N_k'] = np.cumsum(x[:R + 1][::-1])[::-1]
            if 'a_k' in self._fitness_args:
                kwds['a_k'] = 0.5 * np.cumsum(ak_raw[:R + 1][::-1])[::-1]
            if 'b_k' in self._fitness_args:
                kwds['b_k'] = -np.cumsum(bk_raw[:R + 1][::-1])[::-1]
            if 'c_k' in self._fitness_args:
                kwds['c_k'] = 0.5 * np.cumsum(ck_raw[:R + 1][::-1])[::-1]
            fit_vec = self.fitness(**kwds)
            A_R = fit_vec - ncp_prior
            A_R[1:] += best[:R]
            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]
        change_points = np.zeros(N, dtype=int)
        i_cp = N
        ind = N
        while i_cp > 0:
            i_cp -= 1
            change_points[i_cp] = ind
            if ind == 0:
                break
            ind = last[ind - 1]
        if i_cp == 0:
            change_points[i_cp] = 0
        change_points = change_points[i_cp:]
        return edges[change_points]

class Events(FitnessFunc):
    """Bayesian blocks fitness for binned or unbinned events.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on
        :math:`N_{\\rm blocks}` (see eq. 21 of Scargle 2013). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this fitness function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \\sim {\\tt gamma}^{N_{\\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\\tt ncp\\_prior} = -\\ln({\\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """

    def fitness(self, N_k, T_k):
        if False:
            i = 10
            return i + 15
        return N_k * np.log(N_k / T_k)

    def validate_input(self, t, x, sigma):
        if False:
            while True:
                i = 10
        (t, x, sigma) = super().validate_input(t, x, sigma)
        if x is not None and np.any(x % 1 > 0):
            raise ValueError("x must be integer counts for fitness='events'")
        return (t, x, sigma)

class RegularEvents(FitnessFunc):
    """Bayesian blocks fitness for regular events.

    This is for data which has a fundamental "tick" length, so that all
    measured values are multiples of this tick length.  In each tick, there
    are either zero or one counts.

    Parameters
    ----------
    dt : float
        tick rate for data
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\\rm
        blocks}` (see eq. 21 of Scargle 2013). If gamma is specified, p0 is
        ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\\tt ncp\\_prior} = -\\ln({\\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """

    def __init__(self, dt, p0=0.05, gamma=None, ncp_prior=None):
        if False:
            print('Hello World!')
        self.dt = dt
        super().__init__(p0, gamma, ncp_prior)

    def validate_input(self, t, x, sigma):
        if False:
            return 10
        (t, x, sigma) = super().validate_input(t, x, sigma)
        if not np.all((x == 0) | (x == 1)):
            raise ValueError('Regular events must have only 0 and 1 in x')
        return (t, x, sigma)

    def fitness(self, T_k, N_k):
        if False:
            for i in range(10):
                print('nop')
        M_k = T_k / self.dt
        N_over_M = N_k / M_k
        eps = 1e-08
        if np.any(N_over_M > 1 + eps):
            warnings.warn('regular events: N/M > 1.  Is the time step correct?', AstropyUserWarning)
        one_m_NM = 1 - N_over_M
        N_over_M[N_over_M <= 0] = 1
        one_m_NM[one_m_NM <= 0] = 1
        return N_k * np.log(N_over_M) + (M_k - N_k) * np.log(one_m_NM)

class PointMeasures(FitnessFunc):
    """Bayesian blocks fitness for point measures.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\\rm
        blocks}` (see eq. 21 of Scargle 2013). If gamma is specified, p0 is
        ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\\tt ncp\\_prior} = -\\ln({\\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        if False:
            i = 10
            return i + 15
        super().__init__(p0, gamma, ncp_prior)

    def fitness(self, a_k, b_k):
        if False:
            return 10
        return b_k * b_k / (4 * a_k)

    def validate_input(self, t, x, sigma):
        if False:
            return 10
        if x is None:
            raise ValueError('x must be specified for point measures')
        return super().validate_input(t, x, sigma)