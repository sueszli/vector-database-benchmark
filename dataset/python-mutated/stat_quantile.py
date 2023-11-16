from warnings import warn
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineWarning
from .stat import stat

@document
class stat_quantile(stat):
    """
    Compute quantile regression lines

    {usage}

    Parameters
    ----------
    {common_parameters}
    quantiles : tuple, optional (default: (0.25, 0.5, 0.75))
        Quantiles of y to compute
    formula : str, optional (default: 'y ~ x')
        Formula relating y variables to x variables
    method_args : dict, optional
        Extra arguments passed on to the model fitting method,
        :meth:`statsmodels.regression.quantile_regression.QuantReg.fit`.

    See Also
    --------
    statsmodels.regression.quantile_regression.QuantReg
    plotnine.geoms.geom_quantile
    """
    _aesthetics_doc = "\n    {aesthetics_table}\n\n    .. rubric:: Options for computed aesthetics\n\n    ::\n\n         'quantile'  # quantile\n         'group'     # group identifier\n\n    Calculated aesthetics are accessed using the `after_stat` function.\n    e.g. :py:`after_stat('quantile')`.\n    "
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'geom': 'quantile', 'position': 'identity', 'na_rm': False, 'quantiles': (0.25, 0.5, 0.75), 'formula': 'y ~ x', 'method_args': {}}
    CREATES = {'quantile', 'group'}

    def setup_params(self, data):
        if False:
            for i in range(10):
                print('nop')
        params = self.params.copy()
        if params['formula'] is None:
            params['formula'] = 'y ~ x'
            warn("Formula not specified, using '{}'", PlotnineWarning)
        try:
            iter(params['quantiles'])
        except TypeError:
            params['quantiles'] = (params['quantiles'],)
        return params

    @classmethod
    def compute_group(cls, data, scales, **params):
        if False:
            for i in range(10):
                print('nop')
        res = [quant_pred(q, data, **params) for q in params['quantiles']]
        return pd.concat(res, axis=0, ignore_index=True)

def quant_pred(q, data, **params):
    if False:
        return 10
    '\n    Quantile precitions\n    '
    import statsmodels.formula.api as smf
    mod = smf.quantreg(params['formula'], data)
    reg_res = mod.fit(q=q, **params['method_args'])
    out = pd.DataFrame({'x': [data['x'].min(), data['x'].max()], 'quantile': q, 'group': '{}-{}'.format(data['group'].iloc[0], q)})
    out['y'] = reg_res.predict(out)
    return out