import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError
from ..utils import get_valid_kwargs, uniquecols
from .stat import stat

def bootstrap_statistics(series, statistic, n_samples=1000, confidence_interval=0.95, random_state=None):
    if False:
        return 10
    "\n    Default parameters taken from\n    R's Hmisc smean.cl.boot\n    "
    if random_state is None:
        random_state = np.random
    alpha = 1 - confidence_interval
    size = (n_samples, len(series))
    inds = random_state.randint(0, len(series), size=size)
    samples = series.to_numpy()[inds]
    means = np.sort(statistic(samples, axis=1))
    return pd.DataFrame({'ymin': means[int(alpha / 2 * n_samples)], 'ymax': means[int((1 - alpha / 2) * n_samples)], 'y': [statistic(series)]})

def mean_cl_boot(series, n_samples=1000, confidence_interval=0.95, random_state=None):
    if False:
        return 10
    '\n    Bootstrapped mean with confidence interval\n\n    Parameters\n    ----------\n    series : pandas.Series\n        Values\n    n_samples : int (default: 1000)\n        Number of sample to draw.\n    confidence_interval : float\n        Confidence interval in the range (0, 1).\n    random_state : int or ~numpy.random.RandomState, optional\n        Seed or Random number generator to use. If ``None``, then\n        numpy global generator :class:`numpy.random` is used.\n    '
    return bootstrap_statistics(series, np.mean, n_samples=n_samples, confidence_interval=confidence_interval, random_state=random_state)

def mean_cl_normal(series, confidence_interval=0.95):
    if False:
        i = 10
        return i + 15
    '\n    Mean with confidence interval assuming normal distribution\n\n    Credit: from http://stackoverflow.com/a/15034143\n\n    Parameters\n    ----------\n    series : pandas.Series\n        Values\n    confidence_interval : float\n        Confidence interval in the range (0, 1).\n    '
    import scipy.stats as stats
    a = np.asarray(series)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1 + confidence_interval) / 2, len(a) - 1)
    return pd.DataFrame({'y': [m], 'ymin': m - h, 'ymax': m + h})

def mean_sdl(series, mult=2):
    if False:
        return 10
    '\n    Mean +/- a constant times the standard deviation\n\n    Parameters\n    ----------\n    series : pandas.Series\n        Values\n    mult : float\n        Multiplication factor.\n    '
    m = series.mean()
    s = series.std()
    return pd.DataFrame({'y': [m], 'ymin': m - mult * s, 'ymax': m + mult * s})

def median_hilow(series, confidence_interval=0.95):
    if False:
        for i in range(10):
            print('nop')
    '\n    Median and a selected pair of outer quantiles having equal tail areas\n\n    Parameters\n    ----------\n    series : pandas.Series\n        Values\n    confidence_interval : float\n        Confidence interval in the range (0, 1).\n    '
    tail = (1 - confidence_interval) / 2
    return pd.DataFrame({'y': [np.median(series)], 'ymin': np.percentile(series, 100 * tail), 'ymax': np.percentile(series, 100 * (1 - tail))})

def mean_se(series, mult=1):
    if False:
        print('Hello World!')
    '\n    Calculate mean and standard errors on either side\n\n    Parameters\n    ----------\n    series : pandas.Series\n        Values\n    mult : float\n        Multiplication factor.\n    '
    m = np.mean(series)
    se = mult * np.sqrt(np.var(series) / len(series))
    return pd.DataFrame({'y': [m], 'ymin': m - se, 'ymax': m + se})
function_dict = {'mean_cl_boot': mean_cl_boot, 'mean_cl_normal': mean_cl_normal, 'mean_sdl': mean_sdl, 'median_hilow': median_hilow, 'mean_se': mean_se}

def make_summary_fun(fun_data, fun_y, fun_ymin, fun_ymax, fun_args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make summary function\n    '
    if isinstance(fun_data, str):
        fun_data = function_dict[fun_data]
    if any([fun_y, fun_ymin, fun_ymax]):

        def func(df) -> pd.DataFrame:
            if False:
                while True:
                    i = 10
            d = {}
            if fun_y:
                kwargs = get_valid_kwargs(fun_y, fun_args)
                d['y'] = [fun_y(df['y'], **kwargs)]
            if fun_ymin:
                kwargs = get_valid_kwargs(fun_ymin, fun_args)
                d['ymin'] = [fun_ymin(df['y'], **kwargs)]
            if fun_ymax:
                kwargs = get_valid_kwargs(fun_ymax, fun_args)
                d['ymax'] = [fun_ymax(df['y'], **kwargs)]
            return pd.DataFrame(d)
    elif fun_data:
        kwargs = get_valid_kwargs(fun_data, fun_args)

        def func(df) -> pd.DataFrame:
            if False:
                print('Hello World!')
            return fun_data(df['y'], **kwargs)
    else:
        raise ValueError(f'Bad value for function fun_data={fun_data}')
    return func

@document
class stat_summary(stat):
    """
    Calculate summary statistics depending on x

    {usage}

    Parameters
    ----------
    {common_parameters}
    fun_data : str or function, optional
        One of::

            # Bootstrapped mean, confidence interval
            # Arguments:
            #     n_samples - No. of samples to draw
            #     confidence_interval
            #     random_state
            'mean_cl_boot'

            # Mean, C.I. assuming normal distribution
            # Arguments:
            #     confidence_interval
            'mean_cl_normal'

            # Mean, standard deviation * constant
            # Arguments:
            #     mult - multiplication factor
            'mean_sdl'

            # Median, outlier quantiles with equal tail areas
            # Arguments:
            #     confidence_interval
            'median_hilow'

            # Mean, Standard Errors * constant
            # Arguments:
            #     mult - multiplication factor
            'mean_se'

        or any function that takes a array and returns a dataframe
        with three columns named ``y``, ``ymin`` and ``ymax``.
        Defaults to :py:`'mean_cl_boot'`.
    fun_y : function, optional (default: None)
        Any function that takes a array-like and returns a value
        fun_ymin : function (default:None)
        Any function that takes an array-like and returns a value
    fun_ymax : function, optional (default: None)
        Any function that takes an array-like and returns a value
    fun_args : dict, optional (default: None)
        Arguments to any of the functions. Provided the names of the
        arguments of the different functions are in not conflict, the
        arguments will be assigned to the right functions. If there is
        a conflict, create a wrapper function that resolves the
        ambiguity in the argument names.
    random_state : int or ~numpy.random.RandomState, optional
        Seed or Random number generator to use. If ``None``, then
        numpy global generator :class:`numpy.random` is used.

    Notes
    -----
    If any of ``fun_y``, ``fun_ymin`` or ``fun_ymax`` are provided, the
    value of ``fun_data`` will be ignored.

    See Also
    --------
    plotnine.geoms.geom_pointrange
    """
    _aesthetics_doc = "\n    {aesthetics_table}\n\n    .. rubric:: Options for computed aesthetics\n\n    ::\n\n        'ymin'  # ymin computed by the summary function\n        'ymax'  # ymax computed by the summary function\n\n    Calculated aesthetics are accessed using the `after_stat` function.\n    e.g. :py:`after_stat('ymin')`.\n    "
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'geom': 'pointrange', 'position': 'identity', 'na_rm': False, 'fun_data': 'mean_cl_boot', 'fun_y': None, 'fun_ymin': None, 'fun_ymax': None, 'fun_args': None, 'random_state': None}
    CREATES = {'ymin', 'ymax'}

    def setup_params(self, data):
        if False:
            i = 10
            return i + 15
        keys = ('fun_data', 'fun_y', 'fun_ymin', 'fun_ymax')
        if not any((self.params[k] for k in keys)):
            raise PlotnineError('No summary function')
        if self.params['fun_args'] is None:
            self.params['fun_args'] = {}
        if 'random_state' not in self.params['fun_args']:
            if self.params['random_state']:
                random_state = self.params['random_state']
                if random_state is None:
                    random_state = np.random
                elif isinstance(random_state, int):
                    random_state = np.random.RandomState(random_state)
                self.params['fun_args']['random_state'] = random_state
        return self.params

    @classmethod
    def compute_panel(cls, data, scales, **params):
        if False:
            return 10
        func = make_summary_fun(params['fun_data'], params['fun_y'], params['fun_ymin'], params['fun_ymax'], params['fun_args'])
        summaries = []
        for ((group, x), df) in data.groupby(['group', 'x']):
            summary = func(df)
            summary['x'] = x
            summary['group'] = group
            unique = uniquecols(df)
            if 'y' in unique:
                unique = unique.drop('y', axis=1)
            merged = summary.merge(unique, on=['group', 'x'])
            summaries.append(merged)
        new_data = pd.concat(summaries, axis=0, ignore_index=True)
        return new_data