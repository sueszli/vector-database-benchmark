"""
Created on Mon Aug 04 08:00:16 2014

Author: Josef Perktold
License: BSD-3

"""
from statsmodels.compat.python import lzip
import numpy as np
descriptions = {'HC0': 'Standard Errors are heteroscedasticity robust (HC0)', 'HC1': 'Standard Errors are heteroscedasticity robust (HC1)', 'HC2': 'Standard Errors are heteroscedasticity robust (HC2)', 'HC3': 'Standard Errors are heteroscedasticity robust (HC3)', 'HAC': 'Standard Errors are heteroscedasticity and autocorrelation robust (HAC) using {maxlags} lags and {correction} small sample correction', 'fixed_scale': 'Standard Errors are based on fixed scale', 'cluster': 'Standard Errors are robust to cluster correlation (cluster)', 'HAC-Panel': 'Standard Errors are robust to cluster correlation (HAC-Panel)', 'HAC-Groupsum': 'Driscoll and Kraay Standard Errors are robust to cluster correlation (HAC-Groupsum)', 'none': 'Covariance matrix not calculated.', 'approx': 'Covariance matrix calculated using numerical ({approx_type}) differentiation.', 'OPG': 'Covariance matrix calculated using the outer product of gradients ({approx_type}).', 'OIM': 'Covariance matrix calculated using the observed information matrix ({approx_type}) described in Harvey (1989).', 'robust': 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using numerical ({approx_type}) differentiation.', 'robust-OIM': 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using the observed information matrix ({approx_type}) described in Harvey (1989).', 'robust-approx': 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using numerical ({approx_type}) differentiation.'}

def normalize_cov_type(cov_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    Normalize the cov_type string to a canonical version\n\n    Parameters\n    ----------\n    cov_type : str\n\n    Returns\n    -------\n    normalized_cov_type : str\n    '
    if cov_type == 'nw-panel':
        cov_type = 'hac-panel'
    if cov_type == 'nw-groupsum':
        cov_type = 'hac-groupsum'
    return cov_type

def get_robustcov_results(self, cov_type='HC1', use_t=None, **kwds):
    if False:
        while True:
            i = 10
    'create new results instance with robust covariance as default\n\n    Parameters\n    ----------\n    cov_type : str\n        the type of robust sandwich estimator to use. see Notes below\n    use_t : bool\n        If true, then the t distribution is used for inference.\n        If false, then the normal distribution is used.\n    kwds : depends on cov_type\n        Required or optional arguments for robust covariance calculation.\n        see Notes below\n\n    Returns\n    -------\n    results : results instance\n        This method creates a new results instance with the requested\n        robust covariance as the default covariance of the parameters.\n        Inferential statistics like p-values and hypothesis tests will be\n        based on this covariance matrix.\n\n    Notes\n    -----\n    Warning: Some of the options and defaults in cov_kwds may be changed in a\n    future version.\n\n    The covariance keywords provide an option \'scaling_factor\' to adjust the\n    scaling of the covariance matrix, that is the covariance is multiplied by\n    this factor if it is given and is not `None`. This allows the user to\n    adjust the scaling of the covariance matrix to match other statistical\n    packages.\n    For example, `scaling_factor=(nobs - 1.) / (nobs - k_params)` provides a\n    correction so that the robust covariance matrices match those of Stata in\n    some models like GLM and discrete Models.\n\n    The following covariance types and required or optional arguments are\n    currently available:\n\n    - \'HC0\', \'HC1\', \'HC2\', \'HC3\': heteroscedasticity robust covariance\n\n      - no keyword arguments\n\n    - \'HAC\': heteroskedasticity-autocorrelation robust covariance\n\n      ``maxlags`` :  integer, required\n        number of lags to use\n\n      ``kernel`` : {callable, str}, optional\n        kernels currently available kernels are [\'bartlett\', \'uniform\'],\n        default is Bartlett\n\n      ``use_correction``: bool, optional\n        If true, use small sample correction\n\n    - \'cluster\': clustered covariance estimator\n\n      ``groups`` : array_like[int], required :\n        Integer-valued index of clusters or groups.\n\n      ``use_correction``: bool, optional\n        If True the sandwich covariance is calculated with a small\n        sample correction.\n        If False the sandwich covariance is calculated without\n        small sample correction.\n\n      ``df_correction``: bool, optional\n        If True (default), then the degrees of freedom for the\n        inferential statistics and hypothesis tests, such as\n        pvalues, f_pvalue, conf_int, and t_test and f_test, are\n        based on the number of groups minus one instead of the\n        total number of observations minus the number of explanatory\n        variables. `df_resid` of the results instance is also\n        adjusted. When `use_t` is also True, then pvalues are\n        computed using the Student\'s t distribution using the\n        corrected values. These may differ substantially from\n        p-values based on the normal is the number of groups is\n        small.\n        If False, then `df_resid` of the results instance is not\n        adjusted.\n\n\n    - \'hac-groupsum\': Driscoll and Kraay, heteroscedasticity and\n      autocorrelation robust covariance for panel data\n      # TODO: more options needed here\n\n      ``time`` : array_like, required\n        index of time periods\n      ``maxlags`` : integer, required\n        number of lags to use\n      ``kernel`` : {callable, str}, optional\n        The available kernels are [\'bartlett\', \'uniform\']. The default is\n        Bartlett.\n      ``use_correction`` : {False, \'hac\', \'cluster\'}, optional\n        If False the the sandwich covariance is calculated without small\n        sample correction. If `use_correction = \'cluster\'` (default),\n        then the same small sample correction as in the case of\n        `covtype=\'cluster\'` is used.\n      ``df_correction`` : bool, optional\n        The adjustment to df_resid, see cov_type \'cluster\' above\n\n    - \'hac-panel\': heteroscedasticity and autocorrelation robust standard\n      errors in panel data. The data needs to be sorted in this case, the\n      time series for each panel unit or cluster need to be stacked. The\n      membership to a time series of an individual or group can be either\n      specified by group indicators or by increasing time periods. One of\n      ``groups`` or ``time`` is required. # TODO: we need more options here\n\n      ``groups`` : array_like[int]\n        indicator for groups\n      ``time`` : array_like[int]\n        index of time periods\n      ``maxlags`` : int, required\n        number of lags to use\n      ``kernel`` : {callable, str}, optional\n        Available kernels are [\'bartlett\', \'uniform\'], default\n        is Bartlett\n      ``use_correction`` : {False, \'hac\', \'cluster\'}, optional\n        If False the sandwich covariance is calculated without\n        small sample correction.\n      ``df_correction`` : bool, optional\n        Adjustment to df_resid, see cov_type \'cluster\' above\n\n    **Reminder**: ``use_correction`` in "hac-groupsum" and "hac-panel" is\n    not bool, needs to be in {False, \'hac\', \'cluster\'}.\n\n    .. todo:: Currently there is no check for extra or misspelled keywords,\n         except in the case of cov_type `HCx`\n    '
    import statsmodels.stats.sandwich_covariance as sw
    cov_type = normalize_cov_type(cov_type)
    if 'kernel' in kwds:
        kwds['weights_func'] = kwds.pop('kernel')
    if 'weights_func' in kwds and (not callable(kwds['weights_func'])):
        kwds['weights_func'] = sw.kernel_dict[kwds['weights_func']]
    sc_factor = kwds.pop('scaling_factor', None)
    use_self = kwds.pop('use_self', False)
    if use_self:
        res = self
    else:
        res = self.__class__(self.model, self.params, normalized_cov_params=self.normalized_cov_params, scale=self.scale)
    res.cov_type = cov_type
    if use_t is None:
        use_t = self.use_t
    res.cov_kwds = {'use_t': use_t}
    res.use_t = use_t
    adjust_df = False
    if cov_type in ['cluster', 'hac-panel', 'hac-groupsum']:
        df_correction = kwds.get('df_correction', None)
        if df_correction is not False:
            adjust_df = True
    res.cov_kwds['adjust_df'] = adjust_df
    if cov_type.upper() in ('HC0', 'HC1', 'HC2', 'HC3'):
        if kwds:
            raise ValueError('heteroscedasticity robust covariance does not use keywords')
        res.cov_kwds['description'] = descriptions[cov_type.upper()]
        res.cov_params_default = getattr(self, 'cov_' + cov_type.upper(), None)
        if res.cov_params_default is None:
            res.cov_params_default = sw.cov_white_simple(self, use_correction=False)
    elif cov_type.lower() == 'hac':
        maxlags = kwds['maxlags']
        res.cov_kwds['maxlags'] = maxlags
        weights_func = kwds.get('weights_func', sw.weights_bartlett)
        res.cov_kwds['weights_func'] = weights_func
        use_correction = kwds.get('use_correction', False)
        res.cov_kwds['use_correction'] = use_correction
        res.cov_kwds['description'] = descriptions['HAC'].format(maxlags=maxlags, correction=['without', 'with'][use_correction])
        res.cov_params_default = sw.cov_hac_simple(self, nlags=maxlags, weights_func=weights_func, use_correction=use_correction)
    elif cov_type.lower() == 'cluster':
        groups = kwds['groups']
        if not hasattr(groups, 'shape'):
            groups = np.asarray(groups).T
        if groups.ndim >= 2:
            groups = groups.squeeze()
        res.cov_kwds['groups'] = groups
        use_correction = kwds.get('use_correction', True)
        res.cov_kwds['use_correction'] = use_correction
        if groups.ndim == 1:
            if adjust_df:
                self.n_groups = n_groups = len(np.unique(groups))
            res.cov_params_default = sw.cov_cluster(self, groups, use_correction=use_correction)
        elif groups.ndim == 2:
            if hasattr(groups, 'values'):
                groups = groups.values
            if adjust_df:
                n_groups0 = len(np.unique(groups[:, 0]))
                n_groups1 = len(np.unique(groups[:, 1]))
                self.n_groups = (n_groups0, n_groups1)
                n_groups = min(n_groups0, n_groups1)
            res.cov_params_default = sw.cov_cluster_2groups(self, groups, use_correction=use_correction)[0]
        else:
            raise ValueError('only two groups are supported')
        res.cov_kwds['description'] = descriptions['cluster']
    elif cov_type.lower() == 'hac-panel':
        res.cov_kwds['time'] = time = kwds.get('time', None)
        res.cov_kwds['groups'] = groups = kwds.get('groups', None)
        res.cov_kwds['maxlags'] = maxlags = kwds['maxlags']
        use_correction = kwds.get('use_correction', 'hac')
        res.cov_kwds['use_correction'] = use_correction
        weights_func = kwds.get('weights_func', sw.weights_bartlett)
        res.cov_kwds['weights_func'] = weights_func
        if groups is not None:
            groups = np.asarray(groups)
            tt = (np.nonzero(groups[:-1] != groups[1:])[0] + 1).tolist()
            nobs_ = len(groups)
        elif time is not None:
            time = np.asarray(time)
            tt = (np.nonzero(time[1:] < time[:-1])[0] + 1).tolist()
            nobs_ = len(time)
        else:
            raise ValueError('either time or groups needs to be given')
        groupidx = lzip([0] + tt, tt + [nobs_])
        self.n_groups = n_groups = len(groupidx)
        res.cov_params_default = sw.cov_nw_panel(self, maxlags, groupidx, weights_func=weights_func, use_correction=use_correction)
        res.cov_kwds['description'] = descriptions['HAC-Panel']
    elif cov_type.lower() == 'hac-groupsum':
        res.cov_kwds['time'] = time = kwds['time']
        res.cov_kwds['maxlags'] = maxlags = kwds['maxlags']
        use_correction = kwds.get('use_correction', 'cluster')
        res.cov_kwds['use_correction'] = use_correction
        weights_func = kwds.get('weights_func', sw.weights_bartlett)
        res.cov_kwds['weights_func'] = weights_func
        if adjust_df:
            tt = np.nonzero(time[1:] < time[:-1])[0] + 1
            self.n_groups = n_groups = len(tt) + 1
        res.cov_params_default = sw.cov_nw_groupsum(self, maxlags, time, weights_func=weights_func, use_correction=use_correction)
        res.cov_kwds['description'] = descriptions['HAC-Groupsum']
    else:
        raise ValueError('cov_type not recognized. See docstring for ' + 'available options and spelling')
    res.cov_kwds['scaling_factor'] = sc_factor
    if sc_factor is not None:
        res.cov_params_default *= sc_factor
    if adjust_df:
        res.df_resid_inference = n_groups - 1
    return res