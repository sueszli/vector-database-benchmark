from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from .stat import stat
if typing.TYPE_CHECKING:
    from plotnine.typing import FloatArrayLike

@document
class stat_density(stat):
    """
    Compute density estimate

    {usage}

    Parameters
    ----------
    {common_parameters}
    kernel : str, optional (default: 'gaussian')
        Kernel used for density estimation. One of::

            'biweight'
            'cosine'
            'cosine2'
            'epanechnikov'
            'gaussian'
            'triangular'
            'triweight'
            'uniform'

    adjust : float, optional (default: 1)
        An adjustment factor for the ``bw``. Bandwidth becomes
        :py:`bw * adjust`.
        Adjustment of the bandwidth.
    trim : bool, optional (default: False)
        This parameter only matters if you are displaying multiple
        densities in one plot. If :py:`False`, the default, each
        density is computed on the full range of the data. If
        :py:`True`, each density is computed over the range of that
        group; this typically means the estimated x values will not
        line-up, and hence you won't be able to stack density values.
    n : int, optional(default: 1024)
        Number of equally spaced points at which the density is to
        be estimated. For efficient computation, it should be a power
        of two.
    gridsize : int, optional (default: None)
        If gridsize is :py:`None`, :py:`max(len(x), 50)` is used.
    bw : str or float, optional (default: 'nrd0')
        The bandwidth to use, If a float is given, it is the bandwidth.
        The :py:`str` choices are::

            'nrd0'
            'normal_reference'
            'scott'
            'silverman'

        ``nrd0`` is a port of ``stats::bw.nrd0`` in R; it is eqiuvalent
        to ``silverman`` when there is more than 1 value in a group.
    cut : float, optional (default: 3)
        Defines the length of the grid past the lowest and highest
        values of ``x`` so that the kernel goes to zero. The end points
        are ``-/+ cut*bw*{min(x) or max(x)}``.
    clip : tuple, optional (default: (-np.inf, np.inf))
        Values in ``x`` that are outside of the range given by clip are
        dropped. The number of values in ``x`` is then shortened.

    See Also
    --------
    plotnine.geoms.geom_density
    statsmodels.nonparametric.kde.KDEUnivariate
    statsmodels.nonparametric.kde.KDEUnivariate.fit
    """
    _aesthetics_doc = "\n    {aesthetics_table}\n\n    .. rubric:: Options for computed aesthetics\n\n    ::\n\n        'density'   # density estimate\n\n        'count'     # density * number of points,\n                    # useful for stacked density plots\n\n        'scaled'    # density estimate, scaled to maximum of 1\n\n    "
    REQUIRED_AES = {'x'}
    DEFAULT_PARAMS = {'geom': 'density', 'position': 'stack', 'na_rm': False, 'kernel': 'gaussian', 'adjust': 1, 'trim': False, 'n': 1024, 'gridsize': None, 'bw': 'nrd0', 'cut': 3, 'clip': (-np.inf, np.inf)}
    DEFAULT_AES = {'y': after_stat('density')}
    CREATES = {'density', 'count', 'scaled', 'n'}

    def setup_params(self, data):
        if False:
            for i in range(10):
                print('nop')
        params = self.params.copy()
        lookup = {'biweight': 'biw', 'cosine': 'cos', 'cosine2': 'cos2', 'epanechnikov': 'epa', 'gaussian': 'gau', 'triangular': 'tri', 'triweight': 'triw', 'uniform': 'uni'}
        with suppress(KeyError):
            params['kernel'] = lookup[params['kernel'].lower()]
        if params['kernel'] not in lookup.values():
            msg = 'kernel should be one of {}. You may use the abbreviations {}'
            raise PlotnineError(msg.format(lookup.keys(), lookup.values()))
        return params

    @classmethod
    def compute_group(cls, data, scales, **params):
        if False:
            for i in range(10):
                print('nop')
        weight = data.get('weight')
        if params['trim']:
            range_x = (data['x'].min(), data['x'].max())
        else:
            range_x = scales.x.dimension()
        return compute_density(data['x'], weight, range_x, **params)

def compute_density(x, weight, range, **params):
    if False:
        while True:
            i = 10
    '\n    Compute density\n    '
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    not_nan = ~np.isnan(x)
    x = x[not_nan]
    bw = params['bw']
    kernel = params['kernel']
    n = len(x)
    assert isinstance(bw, (str, float))
    if n == 0 or (n == 1 and isinstance(bw, str)):
        if n == 1:
            warn('To compute the density of a group with only one value set the bandwidth manually. e.g `bw=0.1`', PlotnineWarning)
        warn('Groups with fewer than 2 data points have been removed.', PlotnineWarning)
        return pd.DataFrame()
    if weight is None:
        if kernel != 'gau':
            weight = np.ones(n) / n
    else:
        weight = np.asarray(weight, dtype=float)
    if kernel == 'gau' and weight is None:
        fft = True
    else:
        fft = False
    if bw == 'nrd0':
        bw = nrd0(x)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(kernel=kernel, bw=bw, fft=fft, weights=weight, adjust=params['adjust'], cut=params['cut'], gridsize=params['gridsize'], clip=params['clip'])
    x2 = np.linspace(range[0], range[1], params['n'])
    try:
        y = kde.evaluate(x2)
        if np.isscalar(y) and np.isnan(y):
            raise ValueError('kde.evaluate returned nan')
    except ValueError:
        y = []
        for _x in x2:
            result = kde.evaluate(_x)
            if isinstance(result, float):
                y.append(result)
            else:
                y.append(result[0])
    y = np.asarray(y)
    not_nan = ~np.isnan(y)
    x2 = x2[not_nan]
    y = y[not_nan]
    return pd.DataFrame({'x': x2, 'density': y, 'scaled': y / np.max(y) if len(y) else [], 'count': y * n, 'n': n})

def nrd0(x: FloatArrayLike) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Port of R stats::bw.nrd0\n\n    This is equivalent to statsmodels silverman when x has more than\n    1 unique value. It can never give a zero bandwidth.\n\n    Parameters\n    ----------\n    x : array_like\n        Values whose density is to be estimated\n\n    Returns\n    -------\n    out : float\n        Bandwidth of x\n    '
    from scipy.stats import iqr
    n = len(x)
    if n < 1:
        raise ValueError('Need at leat 2 data points to compute the nrd0 bandwidth.')
    std: float = np.std(x, ddof=1)
    std_estimate: float = iqr(x) / 1.349
    low_std = min(std, std_estimate)
    if low_std == 0:
        low_std = std_estimate or np.abs(np.asarray(x)[0]) or 1
    return 0.9 * low_std * n ** (-0.2)