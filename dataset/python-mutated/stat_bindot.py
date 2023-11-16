from __future__ import annotations
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from ..utils import groupby_apply
from .binning import assign_bins, breaks_from_bins, breaks_from_binwidth, freedman_diaconis_bins
from .stat import stat

@document
class stat_bindot(stat):
    """
    Binning for a dot plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    bins : int, optional (default: None)
        Number of bins. Overridden by binwidth. If :py:`None`,
        a number is computed using the freedman-diaconis method.
    binwidth : float, optional (default: None)
        When :py:`method='dotdensity'`, this specifies the maximum
        binwidth. When :py:`method='histodot'`, this specifies the
        binwidth. This supercedes the ``bins``.
    origin : float, optional (default: None)
        When :py:`method='histodot'`, origin of the first bin.
    width : float, optional (default: 0.9)
        When :py:`binaxis='y'`, the spacing of the dotstacks for
        dodging.
    binaxis : str, optional (default: x)
        Axis to bin along. Either :py:`'x'` or :py:`'y'`
    method : str, optional (default: dotdensity)
        One of *dotdensity* or *histodot*. These provide either of
        dot-density binning or fixed bin widths.
    binpositions : str, optional (default: bygroup)
        Position of the bins when :py:`method='dotdensity'`. The value
        is one of::

            'bygroup'  # positions of the bins for each group are
                       # determined separately.
            'all'      # positions of the bins are determined with all
                       # data taken together. This aligns the dots
                       # stacks across multiple groups.

    drop : bool, optional (default: False)
        If :py:`True`, remove all bins with zero counts.
    right : bool, optional (default: True)
        When :py:`method='histodot'`, :py:`True` means include right
        edge of the bins and if :py:`False` the left edge is included.
    breaks : array-like, optional (default: None)
        Bin boundaries for :py:`method='histodot'`. This supercedes the
        ``binwidth`` and ``bins``.

    See Also
    --------
    plotnine.stats.stat_bin
    """
    _aesthetics_doc = "\n    {aesthetics_table}\n\n    .. rubric:: Options for computed aesthetics\n\n    ::\n\n         'count'    # number of points in bin\n         'density'  # density of points in bin, scaled to integrate to 1\n         'ncount'   # count, scaled to maximum of 1\n         'ndensity' # density, scaled to maximum of 1\n\n    "
    REQUIRED_AES = {'x'}
    NON_MISSING_AES = {'weight'}
    DEFAULT_PARAMS = {'geom': 'dotplot', 'position': 'identity', 'na_rm': False, 'bins': None, 'binwidth': None, 'origin': None, 'width': 0.9, 'binaxis': 'x', 'method': 'dotdensity', 'binpositions': 'bygroup', 'drop': False, 'right': True, 'breaks': None}
    DEFAULT_AES = {'y': after_stat('count')}
    CREATES = {'width', 'count', 'density', 'ncount', 'ndensity'}

    def setup_params(self, data):
        if False:
            print('Hello World!')
        params = self.params
        if params['breaks'] is None and params['binwidth'] is None and (params['bins'] is None):
            params = params.copy()
            params['bins'] = freedman_diaconis_bins(data['x'])
            msg = "'stat_bin()' using 'bins = {}'. Pick better value with 'binwidth'."
            warn(msg.format(params['bins']), PlotnineWarning)
        return params

    @classmethod
    def compute_panel(cls, data, scales, **params):
        if False:
            while True:
                i = 10
        if params['method'] == 'dotdensity' and params['binpositions'] == 'all':
            binaxis = params['binaxis']
            if binaxis == 'x':
                newdata = densitybin(x=data['x'], weight=data.get('weight'), binwidth=params['binwidth'], bins=params['bins'])
                data = data.sort_values('x')
                data.reset_index(inplace=True, drop=True)
                newdata = newdata.sort_values('x')
                newdata.reset_index(inplace=True, drop=True)
            elif binaxis == 'y':
                newdata = densitybin(x=data['y'], weight=data.get('weight'), binwidth=params['binwidth'], bins=params['bins'])
                data = data.sort_values('y')
                data.reset_index(inplace=True, drop=True)
                newdata = newdata.sort_values('x')
                newdata.reset_index(inplace=True, drop=True)
            else:
                raise ValueError(f'Unknown value binaxis={binaxis!r}')
            data['bin'] = newdata['bin']
            data['binwidth'] = newdata['binwidth']
            data['weight'] = newdata['weight']
            data['bincenter'] = newdata['bincenter']
        return super(cls, stat_bindot).compute_panel(data, scales, **params)

    @classmethod
    def compute_group(cls, data, scales, **params):
        if False:
            return 10
        weight: pd.Series | None = data.get('weight')
        if weight is not None:
            int_status = [(w * 1.0).is_integer() for w in weight]
            if not all(int_status):
                raise PlotnineError('Weights for stat_bindot must be nonnegative integers.')
        if params['binaxis'] == 'x':
            rangee = scales.x.dimension((0, 0))
            values = data['x'].to_numpy()
            midline = 0
        else:
            rangee = scales.y.dimension((0, 0))
            values = data['y'].to_numpy()
            midline = np.mean([data['x'].min(), data['x'].max()])
        if params['method'] == 'histodot':
            if params['binwidth'] is not None:
                breaks = breaks_from_binwidth(rangee, params['binwidth'], boundary=params['origin'])
            else:
                breaks = breaks_from_bins(rangee, params['bins'], boundary=params['origin'])
            closed = 'right' if params['right'] else 'left'
            data = assign_bins(values, breaks, data.get('weight'), pad=False, closed=closed)
            data.rename(columns={'width': 'binwidth', 'x': 'bincenter'}, inplace=True)
        elif params['method'] == 'dotdensity':
            if params['binpositions'] == 'bygroup':
                data = densitybin(x=values, weight=weight, binwidth=params['binwidth'], bins=params['bins'], rangee=rangee)

            def func(df):
                if False:
                    return 10
                return pd.DataFrame({'binwidth': [df['binwidth'].iloc[0]], 'bincenter': [df['bincenter'].iloc[0]], 'count': [int(df['weight'].sum())]})
            data = groupby_apply(data, 'bincenter', func)
            if data['count'].sum() != 0:
                data.loc[np.isnan(data['count']), 'count'] = 0
                data['ncount'] = data['count'] / data['count'].abs().max()
                if params['drop']:
                    data = data[data['count'] > 0]
                    data.reset_index(inplace=True, drop=True)
        if params['binaxis'] == 'x':
            data['x'] = data.pop('bincenter')
            data['width'] = data['binwidth']
        else:
            data['y'] = data.pop('bincenter')
            data['x'] = midline
        return data

def densitybin(x, weight=None, binwidth=None, bins=None, rangee=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do density binning\n\n    It does not collapse each bin with a count.\n\n    Parameters\n    ----------\n    x : array-like\n        Numbers to bin\n    weight : array-like\n        Weights\n    binwidth : numeric\n        Size of the bins\n    bins : int\n        Number of bins\n    rangee : tuple\n        Range of x\n\n    Returns\n    -------\n    data : DataFrame\n    '
    if all(pd.isna(x)):
        return pd.DataFrame()
    if weight is None:
        weight = np.ones(len(x))
    weight = np.asarray(weight)
    weight[np.isnan(weight)] = 0
    if rangee is None:
        rangee = (np.min(x), np.max(x))
    if bins is None:
        bins = 30
    if binwidth is None:
        binwidth = np.ptp(rangee) / bins
    order = np.argsort(x)
    weight = weight[order]
    x = x[order]
    cbin = 0
    bin_ids = []
    binend = -np.inf
    for value in x:
        if value >= binend:
            binend = value + binwidth
            cbin = cbin + 1
        bin_ids.append(cbin)

    def func(series):
        if False:
            print('Hello World!')
        return (series.min() + series.max()) / 2
    results = pd.DataFrame({'x': x, 'bin': bin_ids, 'binwidth': binwidth, 'weight': weight})
    results['bincenter'] = results.groupby('bin')['x'].transform(func)
    return results