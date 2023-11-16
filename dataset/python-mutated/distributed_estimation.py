from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, _calc_nodewise_weight, _calc_approx_inv_cov
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
'\nDistributed estimation routines. Currently, we support several\nmethods of distribution\n\n- sequential, has no extra dependencies\n- parallel\n    - with joblib\n        A variety of backends are supported through joblib\n        This allows for different types of clusters besides\n        standard local clusters.  Some examples of\n        backends supported by joblib are\n          - dask.distributed\n          - yarn\n          - ipyparallel\n\nThe framework is very general and allows for a variety of\nestimation methods.  Currently, these include\n\n- debiased regularized estimation\n- simple coefficient averaging (naive)\n    - regularized\n    - unregularized\n\nCurrently, the default is regularized estimation with debiasing\nwhich follows the methods outlined in\n\nJason D. Lee, Qiang Liu, Yuekai Sun and Jonathan E. Taylor.\n"Communication-Efficient Sparse Regression: A One-Shot Approach."\narXiv:1503.04337. 2015. https://arxiv.org/abs/1503.04337.\n\nThere are several variables that are taken from the source paper\nfor which the interpretation may not be directly clear from the\ncode, these are mostly used to help form the estimate of the\napproximate inverse covariance matrix as part of the\ndebiasing procedure.\n\n    wexog\n\n    A weighted design matrix used to perform the node-wise\n    regression procedure.\n\n    nodewise_row\n\n    nodewise_row is produced as part of the node-wise regression\n    procedure used to produce the approximate inverse covariance\n    matrix.  One is produced for each variable using the\n    LASSO.\n\n    nodewise_weight\n\n    nodewise_weight is produced using the gamma_hat values for\n    each p to produce weights to reweight the gamma_hat values which\n    are ultimately used to form approx_inv_cov.\n\n    approx_inv_cov\n\n    This is the estimate of the approximate inverse covariance\n    matrix.  This is used to debiase the coefficient average\n    along with the average gradient.  For the OLS case,\n    approx_inv_cov is an approximation for\n\n        n * (X^T X)^{-1}\n\n    formed by node-wise regression.\n'

def _est_regularized_naive(mod, pnum, partitions, fit_kwds=None):
    if False:
        i = 10
        return i + 15
    'estimates the regularized fitted parameters.\n\n    Parameters\n    ----------\n    mod : statsmodels model class instance\n        The model for the current partition.\n    pnum : scalar\n        Index of current partition\n    partitions : scalar\n        Total number of partitions\n    fit_kwds : dict-like or None\n        Keyword arguments to be given to fit_regularized\n\n    Returns\n    -------\n    An array of the parameters for the regularized fit\n    '
    if fit_kwds is None:
        raise ValueError('_est_regularized_naive currently ' + 'requires that fit_kwds not be None.')
    return mod.fit_regularized(**fit_kwds).params

def _est_unregularized_naive(mod, pnum, partitions, fit_kwds=None):
    if False:
        i = 10
        return i + 15
    'estimates the unregularized fitted parameters.\n\n    Parameters\n    ----------\n    mod : statsmodels model class instance\n        The model for the current partition.\n    pnum : scalar\n        Index of current partition\n    partitions : scalar\n        Total number of partitions\n    fit_kwds : dict-like or None\n        Keyword arguments to be given to fit\n\n    Returns\n    -------\n    An array of the parameters for the fit\n    '
    if fit_kwds is None:
        raise ValueError('_est_unregularized_naive currently ' + 'requires that fit_kwds not be None.')
    return mod.fit(**fit_kwds).params

def _join_naive(params_l, threshold=0):
    if False:
        while True:
            i = 10
    'joins the results from each run of _est_<type>_naive\n    and returns the mean estimate of the coefficients\n\n    Parameters\n    ----------\n    params_l : list\n        A list of arrays of coefficients.\n    threshold : scalar\n        The threshold at which the coefficients will be cut.\n    '
    p = len(params_l[0])
    partitions = len(params_l)
    params_mn = np.zeros(p)
    for params in params_l:
        params_mn += params
    params_mn /= partitions
    params_mn[np.abs(params_mn) < threshold] = 0
    return params_mn

def _calc_grad(mod, params, alpha, L1_wt, score_kwds):
    if False:
        return 10
    'calculates the log-likelihood gradient for the debiasing\n\n    Parameters\n    ----------\n    mod : statsmodels model class instance\n        The model for the current partition.\n    params : array_like\n        The estimated coefficients for the current partition.\n    alpha : scalar or array_like\n        The penalty weight.  If a scalar, the same penalty weight\n        applies to all variables in the model.  If a vector, it\n        must have the same length as `params`, and contains a\n        penalty weight for each coefficient.\n    L1_wt : scalar\n        The fraction of the penalty given to the L1 penalty term.\n        Must be between 0 and 1 (inclusive).  If 0, the fit is\n        a ridge fit, if 1 it is a lasso fit.\n    score_kwds : dict-like or None\n        Keyword arguments for the score function.\n\n    Returns\n    -------\n    An array-like object of the same dimension as params\n\n    Notes\n    -----\n    In general:\n\n    gradient l_k(params)\n\n    where k corresponds to the index of the partition\n\n    For OLS:\n\n    X^T(y - X^T params)\n    '
    grad = -mod.score(np.asarray(params), **score_kwds)
    grad += alpha * (1 - L1_wt)
    return grad

def _calc_wdesign_mat(mod, params, hess_kwds):
    if False:
        for i in range(10):
            print('nop')
    'calculates the weighted design matrix necessary to generate\n    the approximate inverse covariance matrix\n\n    Parameters\n    ----------\n    mod : statsmodels model class instance\n        The model for the current partition.\n    params : array_like\n        The estimated coefficients for the current partition.\n    hess_kwds : dict-like or None\n        Keyword arguments for the hessian function.\n\n    Returns\n    -------\n    An array-like object, updated design matrix, same dimension\n    as mod.exog\n    '
    rhess = np.sqrt(mod.hessian_factor(np.asarray(params), **hess_kwds))
    return rhess[:, None] * mod.exog

def _est_regularized_debiased(mod, mnum, partitions, fit_kwds=None, score_kwds=None, hess_kwds=None):
    if False:
        while True:
            i = 10
    'estimates the regularized fitted parameters, is the default\n    estimation_method for class DistributedModel.\n\n    Parameters\n    ----------\n    mod : statsmodels model class instance\n        The model for the current partition.\n    mnum : scalar\n        Index of current partition.\n    partitions : scalar\n        Total number of partitions.\n    fit_kwds : dict-like or None\n        Keyword arguments to be given to fit_regularized\n    score_kwds : dict-like or None\n        Keyword arguments for the score function.\n    hess_kwds : dict-like or None\n        Keyword arguments for the Hessian function.\n\n    Returns\n    -------\n    A tuple of parameters for regularized fit\n        An array-like object of the fitted parameters, params\n        An array-like object for the gradient\n        A list of array like objects for nodewise_row\n        A list of array like objects for nodewise_weight\n    '
    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds
    if fit_kwds is None:
        raise ValueError('_est_regularized_debiased currently ' + 'requires that fit_kwds not be None.')
    else:
        alpha = fit_kwds['alpha']
    if 'L1_wt' in fit_kwds:
        L1_wt = fit_kwds['L1_wt']
    else:
        L1_wt = 1
    (nobs, p) = mod.exog.shape
    p_part = int(np.ceil(1.0 * p / partitions))
    params = mod.fit_regularized(**fit_kwds).params
    grad = _calc_grad(mod, params, alpha, L1_wt, score_kwds) / nobs
    wexog = _calc_wdesign_mat(mod, params, hess_kwds)
    nodewise_row_l = []
    nodewise_weight_l = []
    for idx in range(mnum * p_part, min((mnum + 1) * p_part, p)):
        nodewise_row = _calc_nodewise_row(wexog, idx, alpha)
        nodewise_row_l.append(nodewise_row)
        nodewise_weight = _calc_nodewise_weight(wexog, nodewise_row, idx, alpha)
        nodewise_weight_l.append(nodewise_weight)
    return (params, grad, nodewise_row_l, nodewise_weight_l)

def _join_debiased(results_l, threshold=0):
    if False:
        print('Hello World!')
    'joins the results from each run of _est_regularized_debiased\n    and returns the debiased estimate of the coefficients\n\n    Parameters\n    ----------\n    results_l : list\n        A list of tuples each one containing the params, grad,\n        nodewise_row and nodewise_weight values for each partition.\n    threshold : scalar\n        The threshold at which the coefficients will be cut.\n    '
    p = len(results_l[0][0])
    partitions = len(results_l)
    params_mn = np.zeros(p)
    grad_mn = np.zeros(p)
    nodewise_row_l = []
    nodewise_weight_l = []
    for r in results_l:
        params_mn += r[0]
        grad_mn += r[1]
        nodewise_row_l.extend(r[2])
        nodewise_weight_l.extend(r[3])
    nodewise_row_l = np.array(nodewise_row_l)
    nodewise_weight_l = np.array(nodewise_weight_l)
    params_mn /= partitions
    grad_mn *= -1.0 / partitions
    approx_inv_cov = _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l)
    debiased_params = params_mn + approx_inv_cov.dot(grad_mn)
    debiased_params[np.abs(debiased_params) < threshold] = 0
    return debiased_params

def _helper_fit_partition(self, pnum, endog, exog, fit_kwds, init_kwds_e={}):
    if False:
        print('Hello World!')
    'handles the model fitting for each machine. NOTE: this\n    is primarily handled outside of DistributedModel because\n    joblib cannot handle class methods.\n\n    Parameters\n    ----------\n    self : DistributedModel class instance\n        An instance of DistributedModel.\n    pnum : scalar\n        index of current partition.\n    endog : array_like\n        endogenous data for current partition.\n    exog : array_like\n        exogenous data for current partition.\n    fit_kwds : dict-like\n        Keywords needed for the model fitting.\n    init_kwds_e : dict-like\n        Additional init_kwds to add for each partition.\n\n    Returns\n    -------\n    estimation_method result.  For the default,\n    _est_regularized_debiased, a tuple.\n    '
    temp_init_kwds = self.init_kwds.copy()
    temp_init_kwds.update(init_kwds_e)
    model = self.model_class(endog, exog, **temp_init_kwds)
    results = self.estimation_method(model, pnum, self.partitions, fit_kwds=fit_kwds, **self.estimation_kwds)
    return results

class DistributedModel:
    __doc__ = '\n    Distributed model class\n\n    Parameters\n    ----------\n    partitions : scalar\n        The number of partitions that the data will be split into.\n    model_class : statsmodels model class\n        The model class which will be used for estimation. If None\n        this defaults to OLS.\n    init_kwds : dict-like or None\n        Keywords needed for initializing the model, in addition to\n        endog and exog.\n    init_kwds_generator : generator or None\n        Additional keyword generator that produces model init_kwds\n        that may vary based on data partition.  The current usecase\n        is for WLS and GLS\n    estimation_method : function or None\n        The method that performs the estimation for each partition.\n        If None this defaults to _est_regularized_debiased.\n    estimation_kwds : dict-like or None\n        Keywords to be passed to estimation_method.\n    join_method : function or None\n        The method used to recombine the results from each partition.\n        If None this defaults to _join_debiased.\n    join_kwds : dict-like or None\n        Keywords to be passed to join_method.\n    results_class : results class or None\n        The class of results that should be returned.  If None this\n        defaults to RegularizedResults.\n    results_kwds : dict-like or None\n        Keywords to be passed to results class.\n\n    Attributes\n    ----------\n    partitions : scalar\n        See Parameters.\n    model_class : statsmodels model class\n        See Parameters.\n    init_kwds : dict-like\n        See Parameters.\n    init_kwds_generator : generator or None\n        See Parameters.\n    estimation_method : function\n        See Parameters.\n    estimation_kwds : dict-like\n        See Parameters.\n    join_method : function\n        See Parameters.\n    join_kwds : dict-like\n        See Parameters.\n    results_class : results class\n        See Parameters.\n    results_kwds : dict-like\n        See Parameters.\n\n    Notes\n    -----\n\n    Examples\n    --------\n    '

    def __init__(self, partitions, model_class=None, init_kwds=None, estimation_method=None, estimation_kwds=None, join_method=None, join_kwds=None, results_class=None, results_kwds=None):
        if False:
            print('Hello World!')
        self.partitions = partitions
        if model_class is None:
            self.model_class = OLS
        else:
            self.model_class = model_class
        if init_kwds is None:
            self.init_kwds = {}
        else:
            self.init_kwds = init_kwds
        if estimation_method is None:
            self.estimation_method = _est_regularized_debiased
        else:
            self.estimation_method = estimation_method
        if estimation_kwds is None:
            self.estimation_kwds = {}
        else:
            self.estimation_kwds = estimation_kwds
        if join_method is None:
            self.join_method = _join_debiased
        else:
            self.join_method = join_method
        if join_kwds is None:
            self.join_kwds = {}
        else:
            self.join_kwds = join_kwds
        if results_class is None:
            self.results_class = RegularizedResults
        else:
            self.results_class = results_class
        if results_kwds is None:
            self.results_kwds = {}
        else:
            self.results_kwds = results_kwds

    def fit(self, data_generator, fit_kwds=None, parallel_method='sequential', parallel_backend=None, init_kwds_generator=None):
        if False:
            while True:
                i = 10
        'Performs the distributed estimation using the corresponding\n        DistributedModel\n\n        Parameters\n        ----------\n        data_generator : generator\n            A generator that produces a sequence of tuples where the first\n            element in the tuple corresponds to an endog array and the\n            element corresponds to an exog array.\n        fit_kwds : dict-like or None\n            Keywords needed for the model fitting.\n        parallel_method : str\n            type of distributed estimation to be used, currently\n            "sequential", "joblib" and "dask" are supported.\n        parallel_backend : None or joblib parallel_backend object\n            used to allow support for more complicated backends,\n            ex: dask.distributed\n        init_kwds_generator : generator or None\n            Additional keyword generator that produces model init_kwds\n            that may vary based on data partition.  The current usecase\n            is for WLS and GLS\n\n        Returns\n        -------\n        join_method result.  For the default, _join_debiased, it returns a\n        p length array.\n        '
        if fit_kwds is None:
            fit_kwds = {}
        if parallel_method == 'sequential':
            results_l = self.fit_sequential(data_generator, fit_kwds, init_kwds_generator)
        elif parallel_method == 'joblib':
            results_l = self.fit_joblib(data_generator, fit_kwds, parallel_backend, init_kwds_generator)
        else:
            raise ValueError('parallel_method: %s is currently not supported' % parallel_method)
        params = self.join_method(results_l, **self.join_kwds)
        res_mod = self.model_class([0], [0], **self.init_kwds)
        return self.results_class(res_mod, params, **self.results_kwds)

    def fit_sequential(self, data_generator, fit_kwds, init_kwds_generator=None):
        if False:
            while True:
                i = 10
        'Sequentially performs the distributed estimation using\n        the corresponding DistributedModel\n\n        Parameters\n        ----------\n        data_generator : generator\n            A generator that produces a sequence of tuples where the first\n            element in the tuple corresponds to an endog array and the\n            element corresponds to an exog array.\n        fit_kwds : dict-like\n            Keywords needed for the model fitting.\n        init_kwds_generator : generator or None\n            Additional keyword generator that produces model init_kwds\n            that may vary based on data partition.  The current usecase\n            is for WLS and GLS\n\n        Returns\n        -------\n        join_method result.  For the default, _join_debiased, it returns a\n        p length array.\n        '
        results_l = []
        if init_kwds_generator is None:
            for (pnum, (endog, exog)) in enumerate(data_generator):
                results = _helper_fit_partition(self, pnum, endog, exog, fit_kwds)
                results_l.append(results)
        else:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            for (pnum, ((endog, exog), init_kwds_e)) in tup_gen:
                results = _helper_fit_partition(self, pnum, endog, exog, fit_kwds, init_kwds_e)
                results_l.append(results)
        return results_l

    def fit_joblib(self, data_generator, fit_kwds, parallel_backend, init_kwds_generator=None):
        if False:
            while True:
                i = 10
        'Performs the distributed estimation in parallel using joblib\n\n        Parameters\n        ----------\n        data_generator : generator\n            A generator that produces a sequence of tuples where the first\n            element in the tuple corresponds to an endog array and the\n            element corresponds to an exog array.\n        fit_kwds : dict-like\n            Keywords needed for the model fitting.\n        parallel_backend : None or joblib parallel_backend object\n            used to allow support for more complicated backends,\n            ex: dask.distributed\n        init_kwds_generator : generator or None\n            Additional keyword generator that produces model init_kwds\n            that may vary based on data partition.  The current usecase\n            is for WLS and GLS\n\n        Returns\n        -------\n        join_method result.  For the default, _join_debiased, it returns a\n        p length array.\n        '
        from statsmodels.tools.parallel import parallel_func
        (par, f, n_jobs) = parallel_func(_helper_fit_partition, self.partitions)
        if parallel_backend is None and init_kwds_generator is None:
            results_l = par((f(self, pnum, endog, exog, fit_kwds) for (pnum, (endog, exog)) in enumerate(data_generator)))
        elif parallel_backend is not None and init_kwds_generator is None:
            with parallel_backend:
                results_l = par((f(self, pnum, endog, exog, fit_kwds) for (pnum, (endog, exog)) in enumerate(data_generator)))
        elif parallel_backend is None and init_kwds_generator is not None:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            results_l = par((f(self, pnum, endog, exog, fit_kwds, init_kwds) for (pnum, ((endog, exog), init_kwds)) in tup_gen))
        elif parallel_backend is not None and init_kwds_generator is not None:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            with parallel_backend:
                results_l = par((f(self, pnum, endog, exog, fit_kwds, init_kwds) for (pnum, ((endog, exog), init_kwds)) in tup_gen))
        return results_l

class DistributedResults(LikelihoodModelResults):
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        Class instance for model used for distributed data,
        this particular instance uses fake data and is really
        only to allow use of methods like predict.
    params : ndarray
        Parameter estimates from the fit model.
    """

    def __init__(self, model, params):
        if False:
            i = 10
            return i + 15
        super(DistributedResults, self).__init__(model, params)

    def predict(self, exog, *args, **kwargs):
        if False:
            return 10
        'Calls self.model.predict for the provided exog.  See\n        Results.predict.\n\n        Parameters\n        ----------\n        exog : array_like NOT optional\n            The values for which we want to predict, unlike standard\n            predict this is NOT optional since the data in self.model\n            is fake.\n        *args :\n            Some models can take additional arguments. See the\n            predict method of the model for the details.\n        **kwargs :\n            Some models can take additional keywords arguments. See the\n            predict method of the model for the details.\n\n        Returns\n        -------\n            prediction : ndarray, pandas.Series or pandas.DataFrame\n            See self.model.predict\n        '
        return self.model.predict(self.params, exog, *args, **kwargs)