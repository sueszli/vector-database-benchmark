import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ..doctools import document
from ..utils import resolution
from .stat import stat

@document
class stat_boxplot(stat):
    """
    Compute boxplot statistics

    {usage}

    Parameters
    ----------
    {common_parameters}
    coef : float, optional (default: 1.5)
        Length of the whiskers as a multiple of the Interquartile
        Range.

    See Also
    --------
    plotnine.geoms.geom_boxplot
    """
    _aesthetics_doc = "\n    {aesthetics_table}\n\n    .. rubric:: Options for computed aesthetics\n\n    ::\n\n        'width'  # width of boxplot\n        'lower'  # lower hinge, 25% quantile\n        'middle' # median, 50% quantile\n        'upper'  # upper hinge, 75% quantile\n\n        'notchlower' #  lower edge of notch, computed as;\n                     # median - 1.58 * IQR / sqrt(n)\n\n        'notchupper' # upper edge of notch, computed as;\n                     # median + 1.58 * IQR / sqrt(n)\n\n        'ymin'  # lower whisker, computed as; smallest observation\n                # greater than or equal to lower hinge - 1.5 * IQR\n\n        'ymax'  # upper whisker, computed as; largest observation\n                # less than or equal to upper hinge + 1.5 * IQR\n\n    Calculated aesthetics are accessed using the `after_stat` function.\n    e.g. :py:`after_stat('width')`.\n    "
    REQUIRED_AES = {'x', 'y'}
    NON_MISSING_AES = {'weight'}
    DEFAULT_PARAMS = {'geom': 'boxplot', 'position': 'dodge', 'na_rm': False, 'coef': 1.5, 'width': None}
    CREATES = {'lower', 'upper', 'middle', 'ymin', 'ymax', 'outliers', 'notchupper', 'notchlower', 'width', 'relvarwidth'}

    def setup_params(self, data):
        if False:
            i = 10
            return i + 15
        if self.params['width'] is None:
            self.params['width'] = resolution(data['x'], False) * 0.75
        return self.params

    @classmethod
    def compute_group(cls, data, scales, **params):
        if False:
            i = 10
            return i + 15
        y = data['y'].to_numpy()
        if 'weight' in data:
            weights = data['weight']
            total_weight = np.sum(weights)
        else:
            weights = None
            total_weight = len(y)
        res = weighted_boxplot_stats(y, weights=weights, whis=params['coef'])
        if len(np.unique(data['x'])) > 1:
            width = np.ptp(data['x']) * 0.9
        else:
            width = params['width']
        if pdtypes.is_categorical_dtype(data['x']):
            x = data['x'].iloc[0]
        else:
            x = np.mean([data['x'].min(), data['x'].max()])
        d = {'ymin': res['whislo'], 'lower': res['q1'], 'middle': [res['med']], 'upper': res['q3'], 'ymax': res['whishi'], 'outliers': [res['fliers']], 'notchupper': res['cihi'], 'notchlower': res['cilo'], 'x': x, 'width': width, 'relvarwidth': np.sqrt(total_weight)}
        return pd.DataFrame(d)

def weighted_percentile(a, q, weights=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the weighted q-th percentile of data\n\n    Parameters\n    ----------\n    a : array_like\n        Input that can be converted into an array.\n    q : array_like[float]\n        Percentile or sequence of percentiles to compute. Must be int\n        the range [0, 100]\n    weights : array_like\n        Weights associated with the input values.\n    '
    if weights is None:
        weights = np.ones(len(a))
    weights = np.asarray(weights)
    q = np.asarray(q)
    C = 1
    idx_s = np.argsort(a)
    a_s = a[idx_s]
    w_n = weights[idx_s]
    S_N = np.sum(weights)
    S_n = np.cumsum(w_n)
    p_n = (S_n - C * w_n) / (S_N + (1 - 2 * C) * w_n)
    pcts = np.interp(q / 100.0, p_n, a_s)
    return pcts

def weighted_boxplot_stats(x, weights=None, whis=1.5):
    if False:
        print('Hello World!')
    "\n    Calculate weighted boxplot plot statistics\n\n    Parameters\n    ----------\n    x : array_like\n        Data\n    weights : array_like, optional\n        Weights associated with the data.\n    whis : float, optional (default: 1.5)\n        Position of the whiskers beyond the interquartile range.\n        The data beyond the whisker are considered outliers.\n\n        If a float, the lower whisker is at the lowest datum above\n        ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest\n        datum below ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the\n        first and third quartiles.  The default value of\n        ``whis = 1.5`` corresponds to Tukey's original definition of\n        boxplots.\n\n    Notes\n    -----\n    This method adapted from Matplotlibs boxplot_stats. The key difference\n    is the use of a weighted percentile calculation and then using linear\n    interpolation to map weight percentiles back to data.\n    "
    if weights is None:
        (q1, med, q3) = np.percentile(x, (25, 50, 75))
        n = len(x)
    else:
        (q1, med, q3) = weighted_percentile(x, (25, 50, 75), weights)
        n = np.sum(weights)
    iqr = q3 - q1
    mean = np.average(x, weights=weights)
    cilo = med - 1.58 * iqr / np.sqrt(n)
    cihi = med + 1.58 * iqr / np.sqrt(n)
    loval = q1 - whis * iqr
    lox = x[x >= loval]
    if len(lox) == 0 or np.min(lox) > q1:
        whislo = q1
    else:
        whislo = np.min(lox)
    hival = q3 + whis * iqr
    hix = x[x <= hival]
    if len(hix) == 0 or np.max(hix) < q3:
        whishi = q3
    else:
        whishi = np.max(hix)
    bpstats = {'fliers': x[(x < whislo) | (x > whishi)], 'mean': mean, 'med': med, 'q1': q1, 'q3': q3, 'iqr': iqr, 'whislo': whislo, 'whishi': whishi, 'cilo': cilo, 'cihi': cihi}
    return bpstats