import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
from .stat_qq import stat_qq

@document
class stat_qq_line(stat):
    """
    Calculate line through quantile-quantile plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    distribution : str (default: norm)
        Distribution or distribution function name. The default is
        *norm* for a normal probability plot. Objects that look enough
        like a stats.distributions instance (i.e. they have a ppf
        method) are also accepted. See :mod:`scipy stats <scipy.stats>`
        for available distributions.
    dparams : dict, optional
        Distribution-specific shape parameters (shape parameters plus
        location and scale).
    quantiles : array_like, optional
        Probability points at which to calculate the theoretical
        quantile values. If provided, must be the same number as
        as the sample data points. The default is to use calculated
        theoretical points, use to ``alpha_beta`` control how
        these points are generated.
    alpha_beta : tuple
        Parameter values to use when calculating the quantiles.
        Default is :py:`(3/8, 3/8)`.
    line_p : tuple, optional
        Quantiles to use when fitting a Q-Q line. Must be 2 values.
        Default is :py:`(0.25, 0.75)`.
    fullrange : bool
        If :py:`True` the fit will span the full range of the plot.

    See Also
    --------
    scipy.stats.mstats.plotting_positions : Uses ``alpha_beta``
        to calculate the quantiles.
    """
    REQUIRED_AES = {'sample'}
    DEFAULT_PARAMS = {'geom': 'qq_line', 'position': 'identity', 'na_rm': False, 'distribution': 'norm', 'dparams': {}, 'quantiles': None, 'alpha_beta': (3 / 8, 3 / 8), 'line_p': (0.25, 0.75), 'fullrange': False}
    CREATES = {'x', 'y'}

    def setup_params(self, data):
        if False:
            print('Hello World!')
        if len(self.params['line_p']) != 2:
            raise PlotnineError("Cannot fit line quantiles. 'line_p' must be of length 2")
        return self.params

    @classmethod
    def compute_group(cls, data, scales, **params):
        if False:
            for i in range(10):
                print('nop')
        from scipy.stats.mstats import mquantiles
        from .distributions import get_continuous_distribution
        line_p = params['line_p']
        dparams = params['dparams']
        qq_gdata = stat_qq.compute_group(data, scales, **params)
        sample = qq_gdata['sample'].to_numpy()
        theoretical = qq_gdata['theoretical'].to_numpy()
        cdist = get_continuous_distribution(params['distribution'])
        x_coords = cdist.ppf(line_p, **dparams)
        y_coords = mquantiles(sample, line_p)
        slope = (np.diff(y_coords) / np.diff(x_coords))[0]
        intercept = y_coords[0] - slope * x_coords[0]
        if params['fullrange'] and scales.x:
            x = scales.x.dimension()
        else:
            x = (theoretical.min(), theoretical.max())
        x = np.asarray(x)
        y = slope * x + intercept
        data = pd.DataFrame({'x': x, 'y': y})
        return data