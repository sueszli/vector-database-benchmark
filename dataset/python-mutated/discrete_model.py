"""
Limited dependent variable and qualitative variables.

Includes binary outcomes, count data, (ordered) ordinal data and limited
dependent variables.

General References
--------------------

A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
    Cambridge, 1998

G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
    Cambridge, 1983.

W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
"""
__all__ = ['Poisson', 'Logit', 'Probit', 'MNLogit', 'NegativeBinomial', 'GeneralizedPoisson', 'NegativeBinomialP', 'CountModel']
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning, SpecificationWarning
try:
    import cvxopt
    have_cvxopt = True
except ImportError:
    have_cvxopt = False
FLOAT_EPS = np.finfo(float).eps
EXP_UPPER_LIMIT = np.log(np.finfo(np.float64).max) - 1.0
_discrete_models_docs = '\n'
_discrete_results_docs = '\n    %(one_line_description)s\n\n    Parameters\n    ----------\n    model : A DiscreteModel instance\n    params : array_like\n        The parameters of a fitted model.\n    hessian : array_like\n        The hessian of the fitted model.\n    scale : float\n        A scale parameter for the covariance matrix.\n\n    Attributes\n    ----------\n    df_resid : float\n        See model definition.\n    df_model : float\n        See model definition.\n    llf : float\n        Value of the loglikelihood\n    %(extra_attr)s'
_l1_results_attr = '    nnz_params : int\n        The number of nonzero parameters in the model.  Train with\n        trim_params == True or else numerical error will distort this.\n    trimmed : bool array\n        trimmed[i] == True if the ith parameter was trimmed from the model.'
_get_start_params_null_docs = '\nCompute one-step moment estimator for null (constant-only) model\n\nThis is a preliminary estimator used as start_params.\n\nReturns\n-------\nparams : ndarray\n    parameter estimate based one one-step moment matching\n\n'
_check_rank_doc = '\n    check_rank : bool\n        Check exog rank to determine model degrees of freedom. Default is\n        True. Setting to False reduces model initialization time when\n        exog.shape[1] is large.\n    '

def _numpy_to_dummies(endog):
    if False:
        for i in range(10):
            print('nop')
    if endog.ndim == 2 and endog.dtype.kind not in ['S', 'O']:
        endog_dummies = endog
        ynames = range(endog.shape[1])
    else:
        dummies = get_dummies(endog, drop_first=False)
        ynames = {i: dummies.columns[i] for i in range(dummies.shape[1])}
        endog_dummies = np.asarray(dummies, dtype=float)
        return (endog_dummies, ynames)
    return (endog_dummies, ynames)

def _pandas_to_dummies(endog):
    if False:
        print('Hello World!')
    if endog.ndim == 2:
        if endog.shape[1] == 1:
            yname = endog.columns[0]
            endog_dummies = get_dummies(endog.iloc[:, 0])
        else:
            yname = 'y'
            endog_dummies = endog
    else:
        yname = endog.name
        if yname is None:
            yname = 'y'
        endog_dummies = get_dummies(endog)
    ynames = endog_dummies.columns.tolist()
    return (endog_dummies, ynames, yname)

def _validate_l1_method(method):
    if False:
        return 10
    '\n    As of 0.10.0, the supported values for `method` in `fit_regularized`\n    are "l1" and "l1_cvxopt_cp".  If an invalid value is passed, raise\n    with a helpful error message\n\n    Parameters\n    ----------\n    method : str\n\n    Raises\n    ------\n    ValueError\n    '
    if method not in ['l1', 'l1_cvxopt_cp']:
        raise ValueError('`method` = {method} is not supported, use either "l1" or "l1_cvxopt_cp"'.format(method=method))

class DiscreteModel(base.LikelihoodModel):
    """
    Abstract class for discrete choice models.

    This class does not do anything itself but lays out the methods and
    call signature expected of child classes in addition to those of
    statsmodels.model.LikelihoodModel.
    """

    def __init__(self, endog, exog, check_rank=True, **kwargs):
        if False:
            while True:
                i = 10
        self._check_rank = check_rank
        super().__init__(endog, exog, **kwargs)
        self.raise_on_perfect_prediction = False
        self.k_extra = 0

    def initialize(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize is called by\n        statsmodels.model.LikelihoodModel.__init__\n        and should contain any preprocessing that needs to be done for a model.\n        '
        if self._check_rank:
            rank = tools.matrix_rank(self.exog, method='qr')
        else:
            rank = self.exog.shape[1]
        self.df_model = float(rank - 1)
        self.df_resid = float(self.exog.shape[0] - rank)

    def cdf(self, X):
        if False:
            print('Hello World!')
        '\n        The cumulative distribution function of the model.\n        '
        raise NotImplementedError

    def pdf(self, X):
        if False:
            print('Hello World!')
        '\n        The probability density (mass) function of the model.\n        '
        raise NotImplementedError

    def _check_perfect_pred(self, params, *args):
        if False:
            print('Hello World!')
        endog = self.endog
        fittedvalues = self.predict(params)
        if np.allclose(fittedvalues - endog, 0):
            if self.raise_on_perfect_prediction:
                msg = 'Perfect separation detected, results not available'
                raise PerfectSeparationError(msg)
            else:
                msg = 'Perfect separation or prediction detected, parameter may not be identified'
                warnings.warn(msg, category=PerfectSeparationWarning)

    @Appender(base.LikelihoodModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit the model using maximum likelihood.\n\n        The rest of the docstring is from\n        statsmodels.base.model.LikelihoodModel.fit\n        '
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass
        mlefit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        return mlefit

    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=True, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, qc_verbose=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Fit the model using a regularized maximum likelihood.\n\n        The regularization method AND the solver used is determined by the\n        argument method.\n\n        Parameters\n        ----------\n        start_params : array_like, optional\n            Initial guess of the solution for the loglikelihood maximization.\n            The default is an array of zeros.\n        method : \'l1\' or \'l1_cvxopt_cp\'\n            See notes for details.\n        maxiter : {int, \'defined_by_method\'}\n            Maximum number of iterations to perform.\n            If \'defined_by_method\', then use method defaults (see notes).\n        full_output : bool\n            Set to True to have all available output in the Results object\'s\n            mle_retvals attribute. The output is dependent on the solver.\n            See LikelihoodModelResults notes section for more information.\n        disp : bool\n            Set to True to print convergence messages.\n        fargs : tuple\n            Extra arguments passed to the likelihood function, i.e.,\n            loglike(x,*args).\n        callback : callable callback(xk)\n            Called after each iteration, as callback(xk), where xk is the\n            current parameter vector.\n        retall : bool\n            Set to True to return list of solutions at each iteration.\n            Available in Results object\'s mle_retvals attribute.\n        alpha : non-negative scalar or numpy array (same size as parameters)\n            The weight multiplying the l1 penalty term.\n        trim_mode : \'auto, \'size\', or \'off\'\n            If not \'off\', trim (set to zero) parameters that would have been\n            zero if the solver reached the theoretical minimum.\n            If \'auto\', trim params using the Theory above.\n            If \'size\', trim params if they have very small absolute value.\n        size_trim_tol : float or \'auto\' (default = \'auto\')\n            Tolerance used when trim_mode == \'size\'.\n        auto_trim_tol : float\n            Tolerance used when trim_mode == \'auto\'.\n        qc_tol : float\n            Print warning and do not allow auto trim when (ii) (above) is\n            violated by this much.\n        qc_verbose : bool\n            If true, print out a full QC report upon failure.\n        **kwargs\n            Additional keyword arguments used when fitting the model.\n\n        Returns\n        -------\n        Results\n            A results instance.\n\n        Notes\n        -----\n        Using \'l1_cvxopt_cp\' requires the cvxopt module.\n\n        Extra parameters are not penalized if alpha is given as a scalar.\n        An example is the shape parameter in NegativeBinomial `nb1` and `nb2`.\n\n        Optional arguments for the solvers (available in Results.mle_settings)::\n\n            \'l1\'\n                acc : float (default 1e-6)\n                    Requested accuracy as used by slsqp\n            \'l1_cvxopt_cp\'\n                abstol : float\n                    absolute accuracy (default: 1e-7).\n                reltol : float\n                    relative accuracy (default: 1e-6).\n                feastol : float\n                    tolerance for feasibility conditions (default: 1e-7).\n                refinement : int\n                    number of iterative refinement steps when solving KKT\n                    equations (default: 1).\n\n        Optimization methodology\n\n        With :math:`L` the negative log likelihood, we solve the convex but\n        non-smooth problem\n\n        .. math:: \\min_\\beta L(\\beta) + \\sum_k\\alpha_k |\\beta_k|\n\n        via the transformation to the smooth, convex, constrained problem\n        in twice as many variables (adding the "added variables" :math:`u_k`)\n\n        .. math:: \\min_{\\beta,u} L(\\beta) + \\sum_k\\alpha_k u_k,\n\n        subject to\n\n        .. math:: -u_k \\leq \\beta_k \\leq u_k.\n\n        With :math:`\\partial_k L` the derivative of :math:`L` in the\n        :math:`k^{th}` parameter direction, theory dictates that, at the\n        minimum, exactly one of two conditions holds:\n\n        (i) :math:`|\\partial_k L| = \\alpha_k`  and  :math:`\\beta_k \\neq 0`\n        (ii) :math:`|\\partial_k L| \\leq \\alpha_k`  and  :math:`\\beta_k = 0`\n        '
        _validate_l1_method(method)
        cov_params_func = self.cov_params_func_l1
        alpha = np.array(alpha)
        assert alpha.min() >= 0
        try:
            kwargs['alpha'] = alpha
        except TypeError:
            kwargs = dict(alpha=alpha)
        kwargs['alpha_rescaled'] = kwargs['alpha'] / float(self.endog.shape[0])
        kwargs['trim_mode'] = trim_mode
        kwargs['size_trim_tol'] = size_trim_tol
        kwargs['auto_trim_tol'] = auto_trim_tol
        kwargs['qc_tol'] = qc_tol
        kwargs['qc_verbose'] = qc_verbose
        if maxiter == 'defined_by_method':
            if method == 'l1':
                maxiter = 1000
            elif method == 'l1_cvxopt_cp':
                maxiter = 70
        extra_fit_funcs = {'l1': fit_l1_slsqp}
        if have_cvxopt and method == 'l1_cvxopt_cp':
            from statsmodels.base.l1_cvxopt import fit_l1_cvxopt_cp
            extra_fit_funcs['l1_cvxopt_cp'] = fit_l1_cvxopt_cp
        elif method.lower() == 'l1_cvxopt_cp':
            raise ValueError("Cannot use l1_cvxopt_cp as cvxopt was not found (install it, or use method='l1' instead)")
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass
        mlefit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, extra_fit_funcs=extra_fit_funcs, cov_params_func=cov_params_func, **kwargs)
        return mlefit

    def cov_params_func_l1(self, likelihood_model, xopt, retvals):
        if False:
            while True:
                i = 10
        "\n        Computes cov_params on a reduced parameter space\n        corresponding to the nonzero parameters resulting from the\n        l1 regularized fit.\n\n        Returns a full cov_params matrix, with entries corresponding\n        to zero'd values set to np.nan.\n        "
        H = likelihood_model.hessian(xopt)
        trimmed = retvals['trimmed']
        nz_idx = np.nonzero(~trimmed)[0]
        nnz_params = (~trimmed).sum()
        if nnz_params > 0:
            H_restricted = H[nz_idx[:, None], nz_idx]
            H_restricted_inv = np.linalg.inv(-H_restricted)
        else:
            H_restricted_inv = np.zeros(0)
        cov_params = np.nan * np.ones(H.shape)
        cov_params[nz_idx[:, None], nz_idx] = H_restricted_inv
        return cov_params

    def predict(self, params, exog=None, which='mean', linear=None):
        if False:
            while True:
                i = 10
        '\n        Predict response variable of a model given exogenous variables.\n        '
        raise NotImplementedError

    def _derivative_exog(self, params, exog=None, dummy_idx=None, count_idx=None):
        if False:
            print('Hello World!')
        '\n        This should implement the derivative of the non-linear function\n        '
        raise NotImplementedError

    def _derivative_exog_helper(self, margeff, params, exog, dummy_idx, count_idx, transform):
        if False:
            print('Hello World!')
        '\n        Helper for _derivative_exog to wrap results appropriately\n        '
        from .discrete_margins import _get_count_effects, _get_dummy_effects
        if count_idx is not None:
            margeff = _get_count_effects(margeff, exog, count_idx, transform, self, params)
        if dummy_idx is not None:
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform, self, params)
        return margeff

class BinaryModel(DiscreteModel):
    _continuous_ok = False

    def __init__(self, endog, exog, offset=None, check_rank=True, **kwargs):
        if False:
            i = 10
            return i + 15
        self._check_kwargs(kwargs)
        super().__init__(endog, exog, offset=offset, check_rank=check_rank, **kwargs)
        if not issubclass(self.__class__, MultinomialModel):
            if not np.all((self.endog >= 0) & (self.endog <= 1)):
                raise ValueError('endog must be in the unit interval.')
        if offset is None:
            delattr(self, 'offset')
            if not self._continuous_ok and np.any(self.endog != np.round(self.endog)):
                raise ValueError('endog must be binary, either 0 or 1')

    def predict(self, params, exog=None, which='mean', linear=None, offset=None):
        if False:
            print('Hello World!')
        "\n        Predict response variable of a model given exogenous variables.\n\n        Parameters\n        ----------\n        params : array_like\n            Fitted parameters of the model.\n        exog : array_like\n            1d or 2d array of exogenous values.  If not supplied, the\n            whole exog attribute of the model is used.\n        which : {'mean', 'linear', 'var', 'prob'}, optional\n            Statistic to predict. Default is 'mean'.\n\n            - 'mean' returns the conditional expectation of endog E(y | x),\n              i.e. exp of linear predictor.\n            - 'linear' returns the linear predictor of the mean function.\n            - 'var' returns the estimated variance of endog implied by the\n              model.\n\n            .. versionadded: 0.14\n\n               ``which`` replaces and extends the deprecated ``linear``\n               argument.\n\n        linear : bool\n            If True, returns the linear predicted values.  If False or None,\n            then the statistic specified by ``which`` will be returned.\n\n            .. deprecated: 0.14\n\n               The ``linear` keyword is deprecated and will be removed,\n               use ``which`` keyword instead.\n\n        Returns\n        -------\n        array\n            Fitted values at exog.\n        "
        if linear is not None:
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.warn(msg, FutureWarning)
            if linear is True:
                which = 'linear'
        if offset is None and exog is None and hasattr(self, 'offset'):
            offset = self.offset
        elif offset is None:
            offset = 0.0
        if exog is None:
            exog = self.exog
        linpred = np.dot(exog, params) + offset
        if which == 'mean':
            return self.cdf(linpred)
        elif which == 'linear':
            return linpred
        if which == 'var':
            mu = self.cdf(linpred)
            var_ = mu * (1 - mu)
            return var_
        else:
            raise ValueError('Only `which` is "mean", "linear" or "var" are available.')

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if False:
            return 10
        _validate_l1_method(method)
        bnryfit = super().fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1BinaryResults(self, bnryfit)
        return L1BinaryResultsWrapper(discretefit)

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        if False:
            i = 10
            return i + 15
        res = fit_constrained_wrap(self, constraints, start_params=None, **fit_kwds)
        return res
    fit_constrained.__doc__ = fit_constrained_wrap.__doc__

    def _derivative_predict(self, params, exog=None, transform='dydx', offset=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        For computing marginal effects standard errors.\n\n        This is used only in the case of discrete and count regressors to\n        get the variance-covariance of the marginal effects. It returns\n        [d F / d params] where F is the predict.\n\n        Transform can be 'dydx' or 'eydx'. Checking is done in margeff\n        computations for appropriate transform.\n        "
        if exog is None:
            exog = self.exog
        linpred = self.predict(params, exog, offset=offset, which='linear')
        dF = self.pdf(linpred)[:, None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog, offset=offset)[:, None]
        return dF

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None, offset=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        For computing marginal effects returns dF(XB) / dX where F(.) is\n        the predicted probabilities\n\n        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.\n\n        Not all of these make sense in the presence of discrete regressors,\n        but checks are done in the results in get_margeff.\n        "
        if exog is None:
            exog = self.exog
        linpred = self.predict(params, exog, offset=offset, which='linear')
        margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]
        return self._derivative_exog_helper(margeff, params, exog, dummy_idx, count_idx, transform)

    def _deriv_mean_dparams(self, params):
        if False:
            while True:
                i = 10
        '\n        Derivative of the expected endog with respect to the parameters.\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        The value of the derivative of the expected endog with respect\n        to the parameter vector.\n        '
        link = self.link
        lin_pred = self.predict(params, which='linear')
        idl = link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        return dmat

    def get_distribution(self, params, exog=None, offset=None):
        if False:
            i = 10
            return i + 15
        'Get frozen instance of distribution based on predicted parameters.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n        exog : ndarray, optional\n            Explanatory variables for the main count model.\n            If ``exog`` is None, then the data from the model will be used.\n        offset : ndarray, optional\n            Offset is added to the linear predictor of the mean function with\n            coefficient equal to 1.\n            Default is zero if exog is not None, and the model offset if exog\n            is None.\n        exposure : ndarray, optional\n            Log(exposure) is added to the linear predictor  of the mean\n            function with coefficient equal to 1. If exposure is specified,\n            then it will be logged by the method. The user does not need to\n            log it first.\n            Default is one if exog is is not None, and it is the model exposure\n            if exog is None.\n\n        Returns\n        -------\n        Instance of frozen scipy distribution.\n        '
        mu = self.predict(params, exog=exog, offset=offset)
        distr = stats.bernoulli(mu)
        return distr

class MultinomialModel(BinaryModel):

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        if False:
            i = 10
            return i + 15
        if data_tools._is_using_ndarray_type(endog, None):
            (endog_dummies, ynames) = _numpy_to_dummies(endog)
            yname = 'y'
        elif data_tools._is_using_pandas(endog, None):
            (endog_dummies, ynames, yname) = _pandas_to_dummies(endog)
        else:
            endog = np.asarray(endog)
            (endog_dummies, ynames) = _numpy_to_dummies(endog)
            yname = 'y'
        if not isinstance(ynames, dict):
            ynames = dict(zip(range(endog_dummies.shape[1]), ynames))
        self._ynames_map = ynames
        data = handle_data(endog_dummies, exog, missing, hasconst, **kwargs)
        data.ynames = yname
        data.orig_endog = endog
        self.wendog = data.endog
        for key in kwargs:
            if key in ['design_info', 'formula']:
                continue
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:
                pass
        return data

    def initialize(self):
        if False:
            while True:
                i = 10
        '\n        Preprocesses the data for MNLogit.\n        '
        super().initialize()
        self.endog = self.endog.argmax(1)
        self.J = self.wendog.shape[1]
        self.K = self.exog.shape[1]
        self.df_model *= self.J - 1
        self.df_resid = self.exog.shape[0] - self.df_model - (self.J - 1)

    def predict(self, params, exog=None, which='mean', linear=None):
        if False:
            return 10
        "\n        Predict response variable of a model given exogenous variables.\n\n        Parameters\n        ----------\n        params : array_like\n            2d array of fitted parameters of the model. Should be in the\n            order returned from the model.\n        exog : array_like\n            1d or 2d array of exogenous values.  If not supplied, the\n            whole exog attribute of the model is used. If a 1d array is given\n            it assumed to be 1 row of exogenous variables. If you only have\n            one regressor and would like to do prediction, you must provide\n            a 2d array with shape[1] == 1.\n        which : {'mean', 'linear', 'var', 'prob'}, optional\n            Statistic to predict. Default is 'mean'.\n\n            - 'mean' returns the conditional expectation of endog E(y | x),\n              i.e. exp of linear predictor.\n            - 'linear' returns the linear predictor of the mean function.\n            - 'var' returns the estimated variance of endog implied by the\n              model.\n\n            .. versionadded: 0.14\n\n               ``which`` replaces and extends the deprecated ``linear``\n               argument.\n\n        linear : bool\n            If True, returns the linear predicted values.  If False or None,\n            then the statistic specified by ``which`` will be returned.\n\n            .. deprecated: 0.14\n\n               The ``linear` keyword is deprecated and will be removed,\n               use ``which`` keyword instead.\n\n        Notes\n        -----\n        Column 0 is the base case, the rest conform to the rows of params\n        shifted up one for the base case.\n        "
        if linear is not None:
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.warn(msg, FutureWarning)
            if linear is True:
                which = 'linear'
        if exog is None:
            exog = self.exog
        if exog.ndim == 1:
            exog = exog[None]
        pred = super().predict(params, exog, which=which)
        if which == 'linear':
            pred = np.column_stack((np.zeros(len(exog)), pred))
        return pred

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if start_params is None:
            start_params = np.zeros(self.K * (self.J - 1))
        else:
            start_params = np.asarray(start_params)
        if callback is None:
            callback = lambda x, *args: None
        mnfit = base.LikelihoodModel.fit(self, start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        mnfit.params = mnfit.params.reshape(self.K, -1, order='F')
        mnfit = MultinomialResults(self, mnfit)
        return MultinomialResultsWrapper(mnfit)

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if False:
            while True:
                i = 10
        if start_params is None:
            start_params = np.zeros(self.K * (self.J - 1))
        else:
            start_params = np.asarray(start_params)
        mnfit = DiscreteModel.fit_regularized(self, start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        mnfit.params = mnfit.params.reshape(self.K, -1, order='F')
        mnfit = L1MultinomialResults(self, mnfit)
        return L1MultinomialResultsWrapper(mnfit)

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        if False:
            for i in range(10):
                print('nop')
        "\n        For computing marginal effects standard errors.\n\n        This is used only in the case of discrete and count regressors to\n        get the variance-covariance of the marginal effects. It returns\n        [d F / d params] where F is the predicted probabilities for each\n        choice. dFdparams is of shape nobs x (J*K) x (J-1)*K.\n        The zero derivatives for the base category are not included.\n\n        Transform can be 'dydx' or 'eydx'. Checking is done in margeff\n        computations for appropriate transform.\n        "
        if exog is None:
            exog = self.exog
        if params.ndim == 1:
            params = params.reshape(self.K, self.J - 1, order='F')
        eXB = np.exp(np.dot(exog, params))
        sum_eXB = (1 + eXB.sum(1))[:, None]
        J = int(self.J)
        K = int(self.K)
        repeat_eXB = np.repeat(eXB, J, axis=1)
        X = np.tile(exog, J - 1)
        F0 = -repeat_eXB * X / sum_eXB ** 2
        F1 = eXB.T[:, :, None] * X * (sum_eXB - repeat_eXB) / sum_eXB ** 2
        F1 = F1.transpose((1, 0, 2))
        other_idx = ~np.kron(np.eye(J - 1), np.ones(K)).astype(bool)
        F1[:, other_idx] = (-eXB.T[:, :, None] * X * repeat_eXB / sum_eXB ** 2).transpose((1, 0, 2))[:, other_idx]
        dFdX = np.concatenate((F0[:, None, :], F1), axis=1)
        if 'ey' in transform:
            dFdX /= self.predict(params, exog)[:, :, None]
        return dFdX

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        For computing marginal effects returns dF(XB) / dX where F(.) is\n        the predicted probabilities\n\n        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.\n\n        Not all of these make sense in the presence of discrete regressors,\n        but checks are done in the results in get_margeff.\n\n        For Multinomial models the marginal effects are\n\n        P[j] * (params[j] - sum_k P[k]*params[k])\n\n        It is returned unshaped, so that each row contains each of the J\n        equations. This makes it easier to take derivatives of this for\n        standard errors. If you want average marginal effects you can do\n        margeff.reshape(nobs, K, J, order='F).mean(0) and the marginal effects\n        for choice J are in column J\n        "
        J = int(self.J)
        K = int(self.K)
        if exog is None:
            exog = self.exog
        if params.ndim == 1:
            params = params.reshape(K, J - 1, order='F')
        zeroparams = np.c_[np.zeros(K), params]
        cdf = self.cdf(np.dot(exog, params))
        iterm = np.array([cdf[:, [i]] * zeroparams[:, i] for i in range(int(J))]).sum(0)
        margeff = np.array([cdf[:, [j]] * (zeroparams[:, j] - iterm) for j in range(J)])
        margeff = np.transpose(margeff, (1, 2, 0))
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None, :]
        margeff = self._derivative_exog_helper(margeff, params, exog, dummy_idx, count_idx, transform)
        return margeff.reshape(len(exog), -1, order='F')

    def get_distribution(self, params, exog=None, offset=None):
        if False:
            for i in range(10):
                print('nop')
        'get frozen instance of distribution\n        '
        raise NotImplementedError

class CountModel(DiscreteModel):

    def __init__(self, endog, exog, offset=None, exposure=None, missing='none', check_rank=True, **kwargs):
        if False:
            i = 10
            return i + 15
        self._check_kwargs(kwargs)
        super().__init__(endog, exog, check_rank, missing=missing, offset=offset, exposure=exposure, **kwargs)
        if exposure is not None:
            self.exposure = np.asarray(self.exposure)
            self.exposure = np.log(self.exposure)
        if offset is not None:
            self.offset = np.asarray(self.offset)
        self._check_inputs(self.offset, self.exposure, self.endog)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')
        dt = np.promote_types(self.endog.dtype, np.float64)
        self.endog = np.asarray(self.endog, dt)
        dt = np.promote_types(self.exog.dtype, np.float64)
        self.exog = np.asarray(self.exog, dt)

    def _check_inputs(self, offset, exposure, endog):
        if False:
            print('Hello World!')
        if offset is not None and offset.shape[0] != endog.shape[0]:
            raise ValueError('offset is not the same length as endog')
        if exposure is not None and exposure.shape[0] != endog.shape[0]:
            raise ValueError('exposure is not the same length as endog')

    def _get_init_kwds(self):
        if False:
            i = 10
            return i + 15
        kwds = super()._get_init_kwds()
        if 'exposure' in kwds and kwds['exposure'] is not None:
            kwds['exposure'] = np.exp(kwds['exposure'])
        return kwds

    def _get_predict_arrays(self, exog=None, offset=None, exposure=None):
        if False:
            for i in range(10):
                print('nop')
        if exposure is not None:
            exposure = np.log(np.asarray(exposure))
        if offset is not None:
            offset = np.asarray(offset)
        if exog is None:
            exog = self.exog
            if exposure is None:
                exposure = getattr(self, 'exposure', 0)
            if offset is None:
                offset = getattr(self, 'offset', 0)
        else:
            exog = np.asarray(exog)
            if exposure is None:
                exposure = 0
            if offset is None:
                offset = 0
        return (exog, offset, exposure)

    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', linear=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Predict response variable of a count model given exogenous variables\n\n        Parameters\n        ----------\n        params : array_like\n            Model parameters\n        exog : array_like, optional\n            Design / exogenous data. Is exog is None, model exog is used.\n        exposure : array_like, optional\n            Log(exposure) is added to the linear prediction with\n            coefficient equal to 1. If exposure is not provided and exog\n            is None, uses the model's exposure if present.  If not, uses\n            0 as the default value.\n        offset : array_like, optional\n            Offset is added to the linear prediction with coefficient\n            equal to 1. If offset is not provided and exog\n            is None, uses the model's offset if present.  If not, uses\n            0 as the default value.\n        which : 'mean', 'linear', 'var', 'prob' (optional)\n            Statitistic to predict. Default is 'mean'.\n\n            - 'mean' returns the conditional expectation of endog E(y | x),\n              i.e. exp of linear predictor.\n            - 'linear' returns the linear predictor of the mean function.\n            - 'var' variance of endog implied by the likelihood model\n            - 'prob' predicted probabilities for counts.\n\n            .. versionadded: 0.14\n\n               ``which`` replaces and extends the deprecated ``linear``\n               argument.\n\n        linear : bool\n            If True, returns the linear predicted values.  If False or None,\n            then the statistic specified by ``which`` will be returned.\n\n            .. deprecated: 0.14\n\n               The ``linear` keyword is deprecated and will be removed,\n               use ``which`` keyword instead.\n\n\n        Notes\n        -----\n        If exposure is specified, then it will be logged by the method.\n        The user does not need to log it first.\n        "
        if linear is not None:
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.warn(msg, FutureWarning)
            if linear is True:
                which = 'linear'
        if offset is None and exog is None and hasattr(self, 'offset'):
            offset = self.offset
        elif offset is None:
            offset = 0.0
        if exposure is None and exog is None and hasattr(self, 'exposure'):
            exposure = self.exposure
        elif exposure is None:
            exposure = 0.0
        else:
            exposure = np.log(exposure)
        if exog is None:
            exog = self.exog
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            return np.exp(linpred)
        elif which.startswith('lin'):
            return linpred
        else:
            raise ValueError('keyword which has to be "mean" and "linear"')

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        if False:
            while True:
                i = 10
        "\n        For computing marginal effects standard errors.\n\n        This is used only in the case of discrete and count regressors to\n        get the variance-covariance of the marginal effects. It returns\n        [d F / d params] where F is the predict.\n\n        Transform can be 'dydx' or 'eydx'. Checking is done in margeff\n        computations for appropriate transform.\n        "
        if exog is None:
            exog = self.exog
        dF = self.predict(params, exog)[:, None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog)[:, None]
        return dF

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        For computing marginal effects. These are the marginal effects\n        d F(XB) / dX\n        For the Poisson model F(XB) is the predicted counts rather than\n        the probabilities.\n\n        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.\n\n        Not all of these make sense in the presence of discrete regressors,\n        but checks are done in the results in get_margeff.\n        "
        if exog is None:
            exog = self.exog
        k_extra = getattr(self, 'k_extra', 0)
        params_exog = params if k_extra == 0 else params[:-k_extra]
        margeff = self.predict(params, exog)[:, None] * params_exog[None, :]
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]
        return self._derivative_exog_helper(margeff, params, exog, dummy_idx, count_idx, transform)

    def _deriv_mean_dparams(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Derivative of the expected endog with respect to the parameters.\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        The value of the derivative of the expected endog with respect\n        to the parameter vector.\n        '
        from statsmodels.genmod.families import links
        link = links.Log()
        lin_pred = self.predict(params, which='linear')
        idl = link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        if self.k_extra > 0:
            dmat_extra = np.zeros((dmat.shape[0], self.k_extra))
            dmat = np.column_stack((dmat, dmat_extra))
        return dmat

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        if False:
            return 10
        cntfit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        discretefit = CountResults(self, cntfit)
        return CountResultsWrapper(discretefit)

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if False:
            print('Hello World!')
        _validate_l1_method(method)
        cntfit = super().fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1CountResults(self, cntfit)
        return L1CountResultsWrapper(discretefit)

class Poisson(CountModel):
    __doc__ = '\n    Poisson Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n        ' + base._missing_param_doc + _check_rank_doc}

    @cache_readonly
    def family(self):
        if False:
            i = 10
            return i + 15
        from statsmodels.genmod import families
        return families.Poisson()

    def cdf(self, X):
        if False:
            i = 10
            return i + 15
        '\n        Poisson model cumulative distribution function\n\n        Parameters\n        ----------\n        X : array_like\n            `X` is the linear predictor of the model.  See notes.\n\n        Returns\n        -------\n        The value of the Poisson CDF at each point.\n\n        Notes\n        -----\n        The CDF is defined as\n\n        .. math:: \\exp\\left(-\\lambda\\right)\\sum_{i=0}^{y}\\frac{\\lambda^{i}}{i!}\n\n        where :math:`\\lambda` assumes the loglinear model. I.e.,\n\n        .. math:: \\ln\\lambda_{i}=X\\beta\n\n        The parameter `X` is :math:`X\\beta` in the above formula.\n        '
        y = self.endog
        return stats.poisson.cdf(y, np.exp(X))

    def pdf(self, X):
        if False:
            return 10
        '\n        Poisson model probability mass function\n\n        Parameters\n        ----------\n        X : array_like\n            `X` is the linear predictor of the model.  See notes.\n\n        Returns\n        -------\n        pdf : ndarray\n            The value of the Poisson probability mass function, PMF, for each\n            point of X.\n\n        Notes\n        -----\n        The PMF is defined as\n\n        .. math:: \\frac{e^{-\\lambda_{i}}\\lambda_{i}^{y_{i}}}{y_{i}!}\n\n        where :math:`\\lambda` assumes the loglinear model. I.e.,\n\n        .. math:: \\ln\\lambda_{i}=x_{i}\\beta\n\n        The parameter `X` is :math:`x_{i}\\beta` in the above formula.\n        '
        y = self.endog
        return np.exp(stats.poisson.logpmf(y, np.exp(X)))

    def loglike(self, params):
        if False:
            return 10
        '\n        Loglikelihood of Poisson model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : float\n            The log-likelihood function of the model evaluated at `params`.\n            See notes.\n\n        Notes\n        -----\n        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]\n        '
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        return np.sum(-np.exp(np.clip(XB, None, EXP_UPPER_LIMIT)) + endog * XB - gammaln(endog + 1))

    def loglikeobs(self, params):
        if False:
            return 10
        '\n        Loglikelihood for observations of Poisson model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : array_like\n            The log likelihood for each observation of the model evaluated\n            at `params`. See Notes\n\n        Notes\n        -----\n        .. math:: \\ln L_{i}=\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]\n\n        for observations :math:`i=1,...,n`\n        '
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        return -np.exp(XB) + endog * XB - gammaln(endog + 1)

    @Appender(_get_start_params_null_docs)
    def _get_start_params_null(self):
        if False:
            for i in range(10):
                print('nop')
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]
        return params

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if start_params is None and self.data.const_idx is not None:
            start_params = 0.001 * np.ones(self.exog.shape[1])
            start_params[self.data.const_idx] = self._get_start_params_null()[0]
        kwds = {}
        if kwargs.get('cov_type') is not None:
            kwds['cov_type'] = kwargs.get('cov_type')
            kwds['cov_kwds'] = kwargs.get('cov_kwds', {})
        cntfit = super(CountModel, self).fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        discretefit = PoissonResults(self, cntfit, **kwds)
        return PoissonResultsWrapper(discretefit)

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if False:
            while True:
                i = 10
        _validate_l1_method(method)
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1PoissonResults(self, cntfit)
        return L1PoissonResultsWrapper(discretefit)

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        if False:
            while True:
                i = 10
        'fit the model subject to linear equality constraints\n\n        The constraints are of the form   `R params = q`\n        where R is the constraint_matrix and q is the vector of\n        constraint_values.\n\n        The estimation creates a new model with transformed design matrix,\n        exog, and converts the results back to the original parameterization.\n\n        Parameters\n        ----------\n        constraints : formula expression or tuple\n            If it is a tuple, then the constraint needs to be given by two\n            arrays (constraint_matrix, constraint_value), i.e. (R, q).\n            Otherwise, the constraints can be given as strings or list of\n            strings.\n            see t_test for details\n        start_params : None or array_like\n            starting values for the optimization. `start_params` needs to be\n            given in the original parameter space and are internally\n            transformed.\n        **fit_kwds : keyword arguments\n            fit_kwds are used in the optimization of the transformed model.\n\n        Returns\n        -------\n        results : Results instance\n        '
        from patsy import DesignInfo
        from statsmodels.base._constraints import fit_constrained, LinearConstraints
        lc = DesignInfo(self.exog_names).linear_constraint(constraints)
        (R, q) = (lc.coefs, lc.constants)
        (params, cov, res_constr) = fit_constrained(self, R, q, start_params=start_params, fit_kwds=fit_kwds)
        res = self.fit(maxiter=0, method='nm', disp=0, warn_convergence=False)
        res.mle_retvals['fcall'] = res_constr.mle_retvals.get('fcall', np.nan)
        res.mle_retvals['iterations'] = res_constr.mle_retvals.get('iterations', np.nan)
        res.mle_retvals['converged'] = res_constr.mle_retvals['converged']
        res._results.params = params
        res._results.cov_params_default = cov
        cov_type = fit_kwds.get('cov_type', 'nonrobust')
        if cov_type != 'nonrobust':
            res._results.normalized_cov_params = cov
        else:
            res._results.normalized_cov_params = None
        k_constr = len(q)
        res._results.df_resid += k_constr
        res._results.df_model -= k_constr
        res._results.constraints = LinearConstraints.from_patsy(lc)
        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr
        return res

    def score(self, params):
        if False:
            return 10
        '\n        Poisson model score (gradient) vector of the log-likelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score : ndarray, 1-D\n            The score vector of the model, i.e. the first derivative of the\n            loglikelihood function, evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\lambda_{i}\\right)x_{i}\n\n        where the loglinear model is assumed\n\n        .. math:: \\ln\\lambda_{i}=x_{i}\\beta\n        '
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)
        return np.dot(self.endog - L, X)

    def score_obs(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Poisson model Jacobian of the log-likelihood for each observation\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score : array_like\n            The score vector (nobs, k_vars) of the model evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\lambda_{i}\\right)x_{i}\n\n        for observations :math:`i=1,...,n`\n\n        where the loglinear model is assumed\n\n        .. math:: \\ln\\lambda_{i}=x_{i}\\beta\n        '
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)
        return (self.endog - L)[:, None] * X

    def score_factor(self, params):
        if False:
            return 10
        '\n        Poisson model score_factor for each observation\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score : array_like\n            The score factor (nobs, ) of the model evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\lambda_{i}\\right)\n\n        for observations :math:`i=1,...,n`\n\n        where the loglinear model is assumed\n\n        .. math:: \\ln\\lambda_{i}=x_{i}\\beta\n        '
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)
        return self.endog - L

    def hessian(self, params):
        if False:
            return 10
        '\n        Poisson model Hessian matrix of the loglikelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (k_vars, k_vars)\n            The Hessian, second derivative of loglikelihood function,\n            evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i=1}^{n}\\lambda_{i}x_{i}x_{i}^{\\prime}\n\n        where the loglinear model is assumed\n\n        .. math:: \\ln\\lambda_{i}=x_{i}\\beta\n        '
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + exposure + offset)
        return -np.dot(L * X.T, X)

    def hessian_factor(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Poisson model Hessian factor\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (nobs,)\n            The Hessian factor, second derivative of loglikelihood function\n            with respect to the linear predictor evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i=1}^{n}\\lambda_{i}\n\n        where the loglinear model is assumed\n\n        .. math:: \\ln\\lambda_{i}=x_{i}\\beta\n        '
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        X = self.exog
        L = np.exp(np.dot(X, params) + exposure + offset)
        return -L

    def _deriv_score_obs_dendog(self, params, scale=None):
        if False:
            while True:
                i = 10
        'derivative of score_obs w.r.t. endog\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n        scale : None or float\n            If scale is None, then the default scale will be calculated.\n            Default scale is defined by `self.scaletype` and set in fit.\n            If scale is not None, then it is used as a fixed scale.\n\n        Returns\n        -------\n        derivative : ndarray_2d\n            The derivative of the score_obs with respect to endog. This\n            can is given by `score_factor0[:, None] * exog` where\n            `score_factor0` is the score_factor without the residual.\n        '
        return self.exog

    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', linear=None, y_values=None):
        if False:
            while True:
                i = 10
        '\n        Predict response variable of a model given exogenous variables.\n\n        Parameters\n        ----------\n        params : array_like\n            2d array of fitted parameters of the model. Should be in the\n            order returned from the model.\n        exog : array_like, optional\n            1d or 2d array of exogenous values.  If not supplied, then the\n            exog attribute of the model is used. If a 1d array is given\n            it assumed to be 1 row of exogenous variables. If you only have\n            one regressor and would like to do prediction, you must provide\n            a 2d array with shape[1] == 1.\n        offset : array_like, optional\n            Offset is added to the linear predictor with coefficient equal\n            to 1.\n            Default is zero if exog is not None, and the model offset if exog\n            is None.\n        exposure : array_like, optional\n            Log(exposure) is added to the linear prediction with coefficient\n            equal to 1.\n            Default is one if exog is is not None, and is the model exposure\n            if exog is None.\n        which : \'mean\', \'linear\', \'var\', \'prob\' (optional)\n            Statitistic to predict. Default is \'mean\'.\n\n            - \'mean\' returns the conditional expectation of endog E(y | x),\n              i.e. exp of linear predictor.\n            - \'linear\' returns the linear predictor of the mean function.\n            - \'var\' returns the estimated variance of endog implied by the\n              model.\n            - \'prob\' return probabilities for counts from 0 to max(endog) or\n              for y_values if those are provided.\n\n            .. versionadded: 0.14\n\n               ``which`` replaces and extends the deprecated ``linear``\n               argument.\n\n        linear : bool\n            The ``linear` keyword is deprecated and will be removed,\n            use ``which`` keyword instead.\n            If True, returns the linear predicted values.  If False or None,\n            then the statistic specified by ``which`` will be returned.\n\n            .. deprecated: 0.14\n\n               The ``linear` keyword is deprecated and will be removed,\n               use ``which`` keyword instead.\n\n        y_values : array_like\n            Values of the random variable endog at which pmf is evaluated.\n            Only used if ``which="prob"``\n        '
        if linear is not None:
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.warn(msg, FutureWarning)
            if linear is True:
                which = 'linear'
        if which.startswith('lin'):
            which = 'linear'
        if which in ['mean', 'linear']:
            return super().predict(params, exog=exog, exposure=exposure, offset=offset, which=which, linear=linear)
        elif which == 'var':
            mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)
            return mu
        elif which == 'prob':
            if y_values is not None:
                y_values = np.atleast_2d(y_values)
            else:
                y_values = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)[:, None]
            return stats.poisson._pmf(y_values, mu)
        else:
            raise ValueError('Value of the `which` option is not recognized')

    def _prob_nonzero(self, mu, params=None):
        if False:
            print('Hello World!')
        'Probability that count is not zero\n\n        internal use in Censored model, will be refactored or removed\n        '
        prob_nz = -np.expm1(-mu)
        return prob_nz

    def _var(self, mu, params=None):
        if False:
            while True:
                i = 10
        'variance implied by the distribution\n\n        internal use, will be refactored or removed\n        '
        return mu

    def get_distribution(self, params, exog=None, exposure=None, offset=None):
        if False:
            while True:
                i = 10
        'Get frozen instance of distribution based on predicted parameters.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n        exog : ndarray, optional\n            Explanatory variables for the main count model.\n            If ``exog`` is None, then the data from the model will be used.\n        offset : ndarray, optional\n            Offset is added to the linear predictor of the mean function with\n            coefficient equal to 1.\n            Default is zero if exog is not None, and the model offset if exog\n            is None.\n        exposure : ndarray, optional\n            Log(exposure) is added to the linear predictor  of the mean\n            function with coefficient equal to 1. If exposure is specified,\n            then it will be logged by the method. The user does not need to\n            log it first.\n            Default is one if exog is is not None, and it is the model exposure\n            if exog is None.\n\n        Returns\n        -------\n        Instance of frozen scipy distribution subclass.\n        '
        mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)
        distr = stats.poisson(mu)
        return distr

class GeneralizedPoisson(CountModel):
    __doc__ = '\n    Generalized Poisson Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': '\n    p : scalar\n        P denotes parameterizations for GP regression. p=1 for GP-1 and\n        p=2 for GP-2. Default is p=1.\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.' + base._missing_param_doc + _check_rank_doc}

    def __init__(self, endog, exog, p=1, offset=None, exposure=None, missing='none', check_rank=True, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, check_rank=check_rank, **kwargs)
        self.parameterization = p - 1
        self.exog_names.append('alpha')
        self.k_extra = 1
        self._transparams = False

    def _get_init_kwds(self):
        if False:
            for i in range(10):
                print('nop')
        kwds = super()._get_init_kwds()
        kwds['p'] = self.parameterization + 1
        return kwds

    def _get_exogs(self):
        if False:
            while True:
                i = 10
        return (self.exog, None)

    def loglike(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Loglikelihood of Generalized Poisson model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : float\n            The log-likelihood function of the model evaluated at `params`.\n            See notes.\n\n        Notes\n        -----\n        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[\\mu_{i}+(y_{i}-1)*ln(\\mu_{i}+\n            \\alpha*\\mu_{i}^{p-1}*y_{i})-y_{i}*ln(1+\\alpha*\\mu_{i}^{p-1})-\n            ln(y_{i}!)-\\frac{\\mu_{i}+\\alpha*\\mu_{i}^{p-1}*y_{i}}{1+\\alpha*\n            \\mu_{i}^{p-1}}\\right]\n        '
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Loglikelihood for observations of Generalized Poisson model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : ndarray\n            The log likelihood for each observation of the model evaluated\n            at `params`. See Notes\n\n        Notes\n        -----\n        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[\\mu_{i}+(y_{i}-1)*ln(\\mu_{i}+\n            \\alpha*\\mu_{i}^{p-1}*y_{i})-y_{i}*ln(1+\\alpha*\\mu_{i}^{p-1})-\n            ln(y_{i}!)-\\frac{\\mu_{i}+\\alpha*\\mu_{i}^{p-1}*y_{i}}{1+\\alpha*\n            \\mu_{i}^{p-1}}\\right]\n\n        for observations :math:`i=1,...,n`\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        endog = self.endog
        mu = self.predict(params)
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + (a1 - 1) * endog
        a1 = np.maximum(1e-20, a1)
        a2 = np.maximum(1e-20, a2)
        return np.log(mu) + (endog - 1) * np.log(a2) - endog * np.log(a1) - gammaln(endog + 1) - a2 / a1

    @Appender(_get_start_params_null_docs)
    def _get_start_params_null(self):
        if False:
            print('Hello World!')
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]
        mu = const * np.exp(offset + exposure)
        resid = self.endog - mu
        a = self._estimate_dispersion(mu, resid, df_resid=resid.shape[0] - 1)
        params.append(a)
        return np.array(params)

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        if False:
            return 10
        q = self.parameterization
        if df_resid is None:
            df_resid = resid.shape[0]
        a = ((np.abs(resid) / np.sqrt(mu) - 1) * mu ** (-q)).sum() / df_resid
        return a

    @Appender('\n        use_transparams : bool\n            This parameter enable internal transformation to impose\n            non-negativity. True to enable. Default is False.\n            use_transparams=True imposes the no underdispersion (alpha > 0)\n            constraint. In case use_transparams=True and method="newton" or\n            "ncg" transformation is ignored.\n        ')
    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, use_transparams=False, cov_type='nonrobust', cov_kwds=None, use_t=None, optim_kwds_prelim=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = True
        else:
            if use_transparams:
                warnings.warn('Parameter "use_transparams" is ignored', RuntimeWarning)
            self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            kwds_prelim = {'disp': 0, 'skip_hessian': True, 'warn_convergence': False}
            if optim_kwds_prelim is not None:
                kwds_prelim.update(optim_kwds_prelim)
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                res_poi = mod_poi.fit(**kwds_prelim)
            start_params = res_poi.params
            a = self._estimate_dispersion(res_poi.predict(), res_poi.resid, df_resid=res_poi.df_resid)
            start_params = np.append(start_params, max(-0.1, a))
        if callback is None:
            callback = lambda *x: x
        mlefit = super().fit(start_params=start_params, maxiter=maxiter, method=method, disp=disp, full_output=full_output, callback=callback, **kwargs)
        if optim_kwds_prelim is not None:
            mlefit.mle_settings['optim_kwds_prelim'] = optim_kwds_prelim
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = False
            mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
        gpfit = GeneralizedPoissonResults(self, mlefit._results)
        result = GeneralizedPoissonResultsWrapper(gpfit)
        if cov_kwds is None:
            cov_kwds = {}
        result._get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        return result

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if False:
            i = 10
            return i + 15
        _validate_l1_method(method)
        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1] + self.k_extra
            alpha = alpha * np.ones(k_params)
            alpha[-1] = 0
        alpha_p = alpha[:-1] if self.k_extra and np.size(alpha) > 1 else alpha
        self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                start_params = mod_poi.fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=0, callback=callback, alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            start_params = np.append(start_params, 0.1)
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1GeneralizedPoissonResults(self, cntfit)
        return L1GeneralizedPoissonResultsWrapper(discretefit)

    def score_obs(self, params):
        if False:
            return 10
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        dmudb = mu * exog
        dalpha = mu_p * (y * ((y - 1) / a2 - 2 / a1) + a2 / a1 ** 2)
        dparams = dmudb * (-a4 / a1 + a3 * a2 / a1 ** 2 + (1 + a4) * ((y - 1) / a2 - 1 / a1) + 1 / mu)
        return np.concatenate((dparams, np.atleast_2d(dalpha)), axis=1)

    def score(self, params):
        if False:
            for i in range(10):
                print('nop')
        score = np.sum(self.score_obs(params), axis=0)
        if self._transparams:
            score[-1] == score[-1] ** 2
            return score
        else:
            return score

    def score_factor(self, params, endog=None):
        if False:
            while True:
                i = 10
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog if endog is None else endog
        mu = self.predict(params)
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        dmudb = mu
        dalpha = mu_p * (y * ((y - 1) / a2 - 2 / a1) + a2 / a1 ** 2)
        dparams = dmudb * (-a4 / a1 + a3 * a2 / a1 ** 2 + (1 + a4) * ((y - 1) / a2 - 1 / a1) + 1 / mu)
        return (dparams, dalpha)

    def _score_p(self, params):
        if False:
            print('Hello World!')
        '\n        Generalized Poisson model derivative of the log-likelihood by p-parameter\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        dldp : float\n            dldp is first derivative of the loglikelihood function,\n        evaluated at `p-parameter`.\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        dp = np.sum(np.log(mu) * ((a2 - mu) * ((y - 1) / a2 - 2 / a1) + (a1 - 1) * a2 / a1 ** 2))
        return dp

    def hessian(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generalized Poisson model Hessian matrix of the loglikelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (k_vars, k_vars)\n            The Hessian, second derivative of loglikelihood function,\n            evaluated at `params`\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        a5 = p * mu ** (p - 1)
        dmudb = mu * exog
        dim = exog.shape[1]
        hess_arr = np.empty((dim + 1, dim + 1))
        for i in range(dim):
            for j in range(i + 1):
                hess_val = np.sum(mu * exog[:, i, None] * exog[:, j, None] * (mu * (a3 * a4 / a1 ** 2 - 2 * a3 ** 2 * a2 / a1 ** 3 + 2 * a3 * (a4 + 1) / a1 ** 2 - a4 * p / (mu * a1) + a3 * p * a2 / (mu * a1 ** 2) + (y - 1) * a4 * (p - 1) / (a2 * mu) - (y - 1) * (1 + a4) ** 2 / a2 ** 2 - a4 * (p - 1) / (a1 * mu)) + ((y - 1) * (1 + a4) / a2 - (1 + a4) / a1)), axis=0)
                hess_arr[i, j] = np.squeeze(hess_val)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        dldpda = np.sum((2 * a4 * mu_p / a1 ** 2 - 2 * a3 * mu_p * a2 / a1 ** 3 - mu_p * y * (y - 1) * (1 + a4) / a2 ** 2 + mu_p * (1 + a4) / a1 ** 2 + a5 * y * (y - 1) / a2 - 2 * a5 * y / a1 + a5 * a2 / a1 ** 2) * dmudb, axis=0)
        hess_arr[-1, :-1] = dldpda
        hess_arr[:-1, -1] = dldpda
        dldada = mu_p ** 2 * (3 * y / a1 ** 2 - (y / a2) ** 2.0 * (y - 1) - 2 * a2 / a1 ** 3)
        hess_arr[-1, -1] = dldada.sum()
        return hess_arr

    def hessian_factor(self, params):
        if False:
            while True:
                i = 10
        '\n        Generalized Poisson model Hessian matrix of the loglikelihood\n\n        Parameters\n        ----------\n        params : array-like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (nobs, 3)\n            The Hessian factor, second derivative of loglikelihood function\n            with respect to linear predictor and dispersion parameter\n            evaluated at `params`\n            The first column contains the second derivative w.r.t. linpred,\n            the second column contains the cross derivative, and the\n            third column contains the second derivative w.r.t. the dispersion\n            parameter.\n\n        '
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        a5 = p * mu ** (p - 1)
        dmudb = mu
        dbb = mu * (mu * (a3 * a4 / a1 ** 2 - 2 * a3 ** 2 * a2 / a1 ** 3 + 2 * a3 * (a4 + 1) / a1 ** 2 - a4 * p / (mu * a1) + a3 * p * a2 / (mu * a1 ** 2) + a4 / (mu * a1) - a3 * a2 / (mu * a1 ** 2) + (y - 1) * a4 * (p - 1) / (a2 * mu) - (y - 1) * (1 + a4) ** 2 / a2 ** 2 - a4 * (p - 1) / (a1 * mu) - 1 / mu ** 2) + (-a4 / a1 + a3 * a2 / a1 ** 2 + (y - 1) * (1 + a4) / a2 - (1 + a4) / a1 + 1 / mu))
        dba = (2 * a4 * mu_p / a1 ** 2 - 2 * a3 * mu_p * a2 / a1 ** 3 - mu_p * y * (y - 1) * (1 + a4) / a2 ** 2 + mu_p * (1 + a4) / a1 ** 2 + a5 * y * (y - 1) / a2 - 2 * a5 * y / a1 + a5 * a2 / a1 ** 2) * dmudb
        daa = mu_p ** 2 * (3 * y / a1 ** 2 - (y / a2) ** 2.0 * (y - 1) - 2 * a2 / a1 ** 3)
        return (dbb, dba, daa)

    @Appender(Poisson.predict.__doc__)
    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', y_values=None):
        if False:
            return 10
        if exog is None:
            exog = self.exog
        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)
        if offset is None:
            offset = getattr(self, 'offset', 0)
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            return np.exp(linpred)
        elif which == 'linear':
            return linpred
        elif which == 'var':
            mean = np.exp(linpred)
            alpha = params[-1]
            pm1 = self.parameterization
            var_ = mean * (1 + alpha * mean ** pm1) ** 2
            return var_
        elif which == 'prob':
            if y_values is None:
                y_values = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)[:, None]
            return genpoisson_p.pmf(y_values, mu, params[-1], self.parameterization + 1)
        else:
            raise ValueError("keyword 'which' not recognized")

    def _deriv_score_obs_dendog(self, params):
        if False:
            return 10
        'derivative of score_obs w.r.t. endog\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        derivative : ndarray_2d\n            The derivative of the score_obs with respect to endog.\n        '
        from statsmodels.tools.numdiff import _approx_fprime_cs_scalar

        def f(y):
            if False:
                for i in range(10):
                    print('nop')
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            sf = self.score_factor(params, endog=y)
            return np.column_stack(sf)
        dsf = _approx_fprime_cs_scalar(self.endog[:, None], f)
        d1 = dsf[:, :1] * self.exog
        d2 = dsf[:, 1:2]
        return np.column_stack((d1, d2))

    def _var(self, mu, params=None):
        if False:
            while True:
                i = 10
        'variance implied by the distribution\n\n        internal use, will be refactored or removed\n        '
        alpha = params[-1]
        pm1 = self.parameterization
        var_ = mu * (1 + alpha * mu ** pm1) ** 2
        return var_

    def _prob_nonzero(self, mu, params):
        if False:
            i = 10
            return i + 15
        'Probability that count is not zero\n\n        internal use in Censored model, will be refactored or removed\n        '
        alpha = params[-1]
        pm1 = self.parameterization
        prob_zero = np.exp(-mu / (1 + alpha * mu ** pm1))
        prob_nz = 1 - prob_zero
        return prob_nz

    @Appender(Poisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exposure=None, offset=None):
        if False:
            while True:
                i = 10
        'get frozen instance of distribution\n        '
        mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)
        p = self.parameterization + 1
        distr = genpoisson_p(mu, params[-1], p)
        return distr

class Logit(BinaryModel):
    __doc__ = '\n    Logit Model\n\n    %(params)s\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': base._missing_param_doc + _check_rank_doc}
    _continuous_ok = True

    @cache_readonly
    def link(self):
        if False:
            return 10
        from statsmodels.genmod.families import links
        link = links.Logit()
        return link

    def cdf(self, X):
        if False:
            while True:
                i = 10
        '\n        The logistic cumulative distribution function\n\n        Parameters\n        ----------\n        X : array_like\n            `X` is the linear predictor of the logit model.  See notes.\n\n        Returns\n        -------\n        1/(1 + exp(-X))\n\n        Notes\n        -----\n        In the logit model,\n\n        .. math:: \\Lambda\\left(x^{\\prime}\\beta\\right)=\n                  \\text{Prob}\\left(Y=1|x\\right)=\n                  \\frac{e^{x^{\\prime}\\beta}}{1+e^{x^{\\prime}\\beta}}\n        '
        X = np.asarray(X)
        return 1 / (1 + np.exp(-X))

    def pdf(self, X):
        if False:
            return 10
        '\n        The logistic probability density function\n\n        Parameters\n        ----------\n        X : array_like\n            `X` is the linear predictor of the logit model.  See notes.\n\n        Returns\n        -------\n        pdf : ndarray\n            The value of the Logit probability mass function, PMF, for each\n            point of X. ``np.exp(-x)/(1+np.exp(-X))**2``\n\n        Notes\n        -----\n        In the logit model,\n\n        .. math:: \\lambda\\left(x^{\\prime}\\beta\\right)=\\frac{e^{-x^{\\prime}\\beta}}{\\left(1+e^{-x^{\\prime}\\beta}\\right)^{2}}\n        '
        X = np.asarray(X)
        return np.exp(-X) / (1 + np.exp(-X)) ** 2

    @cache_readonly
    def family(self):
        if False:
            while True:
                i = 10
        from statsmodels.genmod import families
        return families.Binomial()

    def loglike(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Log-likelihood of logit model.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the logit model.\n\n        Returns\n        -------\n        loglike : float\n            The log-likelihood function of the model evaluated at `params`.\n            See notes.\n\n        Notes\n        -----\n        .. math::\n\n           \\ln L=\\sum_{i}\\ln\\Lambda\n           \\left(q_{i}x_{i}^{\\prime}\\beta\\right)\n\n        Where :math:`q=2y-1`. This simplification comes from the fact that the\n        logistic distribution is symmetric.\n        '
        q = 2 * self.endog - 1
        linpred = self.predict(params, which='linear')
        return np.sum(np.log(self.cdf(q * linpred)))

    def loglikeobs(self, params):
        if False:
            print('Hello World!')
        '\n        Log-likelihood of logit model for each observation.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the logit model.\n\n        Returns\n        -------\n        loglike : ndarray\n            The log likelihood for each observation of the model evaluated\n            at `params`. See Notes\n\n        Notes\n        -----\n        .. math::\n\n           \\ln L=\\sum_{i}\\ln\\Lambda\n           \\left(q_{i}x_{i}^{\\prime}\\beta\\right)\n\n        for observations :math:`i=1,...,n`\n\n        where :math:`q=2y-1`. This simplification comes from the fact that the\n        logistic distribution is symmetric.\n        '
        q = 2 * self.endog - 1
        linpred = self.predict(params, which='linear')
        return np.log(self.cdf(q * linpred))

    def score(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Logit model score (gradient) vector of the log-likelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score : ndarray, 1-D\n            The score vector of the model, i.e. the first derivative of the\n            loglikelihood function, evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}\n        '
        y = self.endog
        X = self.exog
        fitted = self.predict(params)
        return np.dot(y - fitted, X)

    def score_obs(self, params):
        if False:
            while True:
                i = 10
        '\n        Logit model Jacobian of the log-likelihood for each observation\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        jac : array_like\n            The derivative of the loglikelihood for each observation evaluated\n            at `params`.\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\Lambda_{i}\\right)x_{i}\n\n        for observations :math:`i=1,...,n`\n        '
        y = self.endog
        X = self.exog
        fitted = self.predict(params)
        return (y - fitted)[:, None] * X

    def score_factor(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Logit model derivative of the log-likelihood with respect to linpred.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score_factor : array_like\n            The derivative of the loglikelihood for each observation evaluated\n            at `params`.\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\lambda_{i}\\right)\n\n        for observations :math:`i=1,...,n`\n\n        where the loglinear model is assumed\n\n        .. math:: \\ln\\lambda_{i}=x_{i}\\beta\n        '
        y = self.endog
        fitted = self.predict(params)
        return y - fitted

    def hessian(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Logit model Hessian matrix of the log-likelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (k_vars, k_vars)\n            The Hessian, second derivative of loglikelihood function,\n            evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}\n        '
        X = self.exog
        L = self.predict(params)
        return -np.dot(L * (1 - L) * X.T, X)

    def hessian_factor(self, params):
        if False:
            while True:
                i = 10
        '\n        Logit model Hessian factor\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (nobs,)\n            The Hessian factor, second derivative of loglikelihood function\n            with respect to the linear predictor evaluated at `params`\n        '
        L = self.predict(params)
        return -L * (1 - L)

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        if False:
            i = 10
            return i + 15
        bnryfit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        discretefit = LogitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)

    def _deriv_score_obs_dendog(self, params):
        if False:
            i = 10
            return i + 15
        'derivative of score_obs w.r.t. endog\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        derivative : ndarray_2d\n            The derivative of the score_obs with respect to endog. This\n            can is given by `score_factor0[:, None] * exog` where\n            `score_factor0` is the score_factor without the residual.\n        '
        return self.exog

class Probit(BinaryModel):
    __doc__ = '\n    Probit Model\n\n    %(params)s\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': base._missing_param_doc + _check_rank_doc}

    @cache_readonly
    def link(self):
        if False:
            i = 10
            return i + 15
        from statsmodels.genmod.families import links
        link = links.Probit()
        return link

    def cdf(self, X):
        if False:
            return 10
        '\n        Probit (Normal) cumulative distribution function\n\n        Parameters\n        ----------\n        X : array_like\n            The linear predictor of the model (XB).\n\n        Returns\n        -------\n        cdf : ndarray\n            The cdf evaluated at `X`.\n\n        Notes\n        -----\n        This function is just an alias for scipy.stats.norm.cdf\n        '
        return stats.norm._cdf(X)

    def pdf(self, X):
        if False:
            i = 10
            return i + 15
        '\n        Probit (Normal) probability density function\n\n        Parameters\n        ----------\n        X : array_like\n            The linear predictor of the model (XB).\n\n        Returns\n        -------\n        pdf : ndarray\n            The value of the normal density function for each point of X.\n\n        Notes\n        -----\n        This function is just an alias for scipy.stats.norm.pdf\n        '
        X = np.asarray(X)
        return stats.norm._pdf(X)

    def loglike(self, params):
        if False:
            print('Hello World!')
        '\n        Log-likelihood of probit model (i.e., the normal distribution).\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : float\n            The log-likelihood function of the model evaluated at `params`.\n            See notes.\n\n        Notes\n        -----\n        .. math:: \\ln L=\\sum_{i}\\ln\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)\n\n        Where :math:`q=2y-1`. This simplification comes from the fact that the\n        normal distribution is symmetric.\n        '
        q = 2 * self.endog - 1
        linpred = self.predict(params, which='linear')
        return np.sum(np.log(np.clip(self.cdf(q * linpred), FLOAT_EPS, 1)))

    def loglikeobs(self, params):
        if False:
            return 10
        '\n        Log-likelihood of probit model for each observation\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : array_like\n            The log likelihood for each observation of the model evaluated\n            at `params`. See Notes\n\n        Notes\n        -----\n        .. math:: \\ln L_{i}=\\ln\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)\n\n        for observations :math:`i=1,...,n`\n\n        where :math:`q=2y-1`. This simplification comes from the fact that the\n        normal distribution is symmetric.\n        '
        q = 2 * self.endog - 1
        linpred = self.predict(params, which='linear')
        return np.log(np.clip(self.cdf(q * linpred), FLOAT_EPS, 1))

    def score(self, params):
        if False:
            return 10
        '\n        Probit model score (gradient) vector\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score : ndarray, 1-D\n            The score vector of the model, i.e. the first derivative of the\n            loglikelihood function, evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left[\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\\right]x_{i}\n\n        Where :math:`q=2y-1`. This simplification comes from the fact that the\n        normal distribution is symmetric.\n        '
        y = self.endog
        X = self.exog
        XB = self.predict(params, which='linear')
        q = 2 * y - 1
        L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
        return np.dot(L, X)

    def score_obs(self, params):
        if False:
            while True:
                i = 10
        '\n        Probit model Jacobian for each observation\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        jac : array_like\n            The derivative of the loglikelihood for each observation evaluated\n            at `params`.\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left[\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\\right]x_{i}\n\n        for observations :math:`i=1,...,n`\n\n        Where :math:`q=2y-1`. This simplification comes from the fact that the\n        normal distribution is symmetric.\n        '
        y = self.endog
        X = self.exog
        XB = self.predict(params, which='linear')
        q = 2 * y - 1
        L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
        return L[:, None] * X

    def score_factor(self, params):
        if False:
            while True:
                i = 10
        '\n        Probit model Jacobian for each observation\n\n        Parameters\n        ----------\n        params : array-like\n            The parameters of the model\n\n        Returns\n        -------\n        score_factor : array_like (nobs,)\n            The derivative of the loglikelihood function for each observation\n            with respect to linear predictor evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left[\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\\right]x_{i}\n\n        for observations :math:`i=1,...,n`\n\n        Where :math:`q=2y-1`. This simplification comes from the fact that the\n        normal distribution is symmetric.\n        '
        y = self.endog
        XB = self.predict(params, which='linear')
        q = 2 * y - 1
        L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
        return L

    def hessian(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Probit model Hessian matrix of the log-likelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (k_vars, k_vars)\n            The Hessian, second derivative of loglikelihood function,\n            evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\lambda_{i}\\left(\\lambda_{i}+x_{i}^{\\prime}\\beta\\right)x_{i}x_{i}^{\\prime}\n\n        where\n\n        .. math:: \\lambda_{i}=\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\n\n        and :math:`q=2y-1`\n        '
        X = self.exog
        XB = self.predict(params, which='linear')
        q = 2 * self.endog - 1
        L = q * self.pdf(q * XB) / self.cdf(q * XB)
        return np.dot(-L * (L + XB) * X.T, X)

    def hessian_factor(self, params):
        if False:
            print('Hello World!')
        '\n        Probit model Hessian factor of the log-likelihood\n\n        Parameters\n        ----------\n        params : array-like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (nobs,)\n            The Hessian factor, second derivative of loglikelihood function\n            with respect to linear predictor evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\lambda_{i}\\left(\\lambda_{i}+x_{i}^{\\prime}\\beta\\right)x_{i}x_{i}^{\\prime}\n\n        where\n\n        .. math:: \\lambda_{i}=\\frac{q_{i}\\phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}{\\Phi\\left(q_{i}x_{i}^{\\prime}\\beta\\right)}\n\n        and :math:`q=2y-1`\n        '
        XB = self.predict(params, which='linear')
        q = 2 * self.endog - 1
        L = q * self.pdf(q * XB) / self.cdf(q * XB)
        return -L * (L + XB)

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        bnryfit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        discretefit = ProbitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)

    def _deriv_score_obs_dendog(self, params):
        if False:
            for i in range(10):
                print('nop')
        'derivative of score_obs w.r.t. endog\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        derivative : ndarray_2d\n            The derivative of the score_obs with respect to endog. This\n            can is given by `score_factor0[:, None] * exog` where\n            `score_factor0` is the score_factor without the residual.\n        '
        linpred = self.predict(params, which='linear')
        pdf_ = self.pdf(linpred)
        cdf_ = np.clip(self.cdf(linpred), FLOAT_EPS, 1 - FLOAT_EPS)
        deriv = pdf_ / cdf_ / (1 - cdf_)
        return deriv[:, None] * self.exog

class MNLogit(MultinomialModel):
    __doc__ = '\n    Multinomial Logit Model\n\n    Parameters\n    ----------\n    endog : array_like\n        `endog` is an 1-d vector of the endogenous response.  `endog` can\n        contain strings, ints, or floats or may be a pandas Categorical Series.\n        Note that if it contains strings, every distinct string will be a\n        category.  No stripping of whitespace is done.\n    exog : array_like\n        A nobs x k array where `nobs` is the number of observations and `k`\n        is the number of regressors. An intercept is not included by default\n        and should be added by the user. See `statsmodels.tools.add_constant`.\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    J : float\n        The number of choices for the endogenous variable. Note that this\n        is zero-indexed.\n    K : float\n        The actual number of parameters for the exogenous design.  Includes\n        the constant if the design has one.\n    names : dict\n        A dictionary mapping the column number in `wendog` to the variables\n        in `endog`.\n    wendog : ndarray\n        An n x j array where j is the number of unique categories in `endog`.\n        Each column of j is a dummy variable indicating the category of\n        each observation. See `names` for a dictionary mapping each column to\n        its category.\n\n    Notes\n    -----\n    See developer notes for further information on `MNLogit` internals.\n    ' % {'extra_params': base._missing_param_doc + _check_rank_doc}

    def __init__(self, endog, exog, check_rank=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(endog, exog, check_rank=check_rank, **kwargs)
        yname = self.endog_names
        ynames = self._ynames_map
        ynames = MultinomialResults._maybe_convert_ynames_int(ynames)
        ynames = [ynames[key] for key in range(int(self.J))]
        idx = MultiIndex.from_product((ynames[1:], self.data.xnames), names=(yname, None))
        self.data.cov_names = idx

    def pdf(self, eXB):
        if False:
            i = 10
            return i + 15
        '\n        NotImplemented\n        '
        raise NotImplementedError

    def cdf(self, X):
        if False:
            while True:
                i = 10
        '\n        Multinomial logit cumulative distribution function.\n\n        Parameters\n        ----------\n        X : ndarray\n            The linear predictor of the model XB.\n\n        Returns\n        -------\n        cdf : ndarray\n            The cdf evaluated at `X`.\n\n        Notes\n        -----\n        In the multinomial logit model.\n        .. math:: \\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\n        '
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
        return eXB / eXB.sum(1)[:, None]

    def loglike(self, params):
        if False:
            print('Hello World!')
        '\n        Log-likelihood of the multinomial logit model.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the multinomial logit model.\n\n        Returns\n        -------\n        loglike : float\n            The log-likelihood function of the model evaluated at `params`.\n            See notes.\n\n        Notes\n        -----\n        .. math::\n\n           \\ln L=\\sum_{i=1}^{n}\\sum_{j=0}^{J}d_{ij}\\ln\n           \\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}\n           {\\sum_{k=0}^{J}\n           \\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)\n\n        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0\n        if not.\n        '
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog, params)))
        return np.sum(d * logprob)

    def loglikeobs(self, params):
        if False:
            return 10
        '\n        Log-likelihood of the multinomial logit model for each observation.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the multinomial logit model.\n\n        Returns\n        -------\n        loglike : array_like\n            The log likelihood for each observation of the model evaluated\n            at `params`. See Notes\n\n        Notes\n        -----\n        .. math::\n\n           \\ln L_{i}=\\sum_{j=0}^{J}d_{ij}\\ln\n           \\left(\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}\n           {\\sum_{k=0}^{J}\n           \\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)\n\n        for observations :math:`i=1,...,n`\n\n        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0\n        if not.\n        '
        params = params.reshape(self.K, -1, order='F')
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog, params)))
        return d * logprob

    def score(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Score matrix for multinomial logit model log-likelihood\n\n        Parameters\n        ----------\n        params : ndarray\n            The parameters of the multinomial logit model.\n\n        Returns\n        -------\n        score : ndarray, (K * (J-1),)\n            The 2-d score vector, i.e. the first derivative of the\n            loglikelihood function, of the multinomial logit model evaluated at\n            `params`.\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta_{j}}=\\sum_{i}\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}\n\n        for :math:`j=1,...,J`\n\n        In the multinomial model the score matrix is K x J-1 but is returned\n        as a flattened array to work with the solvers.\n        '
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params))[:, 1:]
        return np.dot(firstterm.T, self.exog).flatten()

    def loglike_and_score(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns log likelihood and score, efficiently reusing calculations.\n\n        Note that both of these returned quantities will need to be negated\n        before being minimized by the maximum likelihood fitting machinery.\n        '
        params = params.reshape(self.K, -1, order='F')
        cdf_dot_exog_params = self.cdf(np.dot(self.exog, params))
        loglike_value = np.sum(self.wendog * np.log(cdf_dot_exog_params))
        firstterm = self.wendog[:, 1:] - cdf_dot_exog_params[:, 1:]
        score_array = np.dot(firstterm.T, self.exog).flatten()
        return (loglike_value, score_array)

    def score_obs(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Jacobian matrix for multinomial logit model log-likelihood\n\n        Parameters\n        ----------\n        params : ndarray\n            The parameters of the multinomial logit model.\n\n        Returns\n        -------\n        jac : array_like\n            The derivative of the loglikelihood for each observation evaluated\n            at `params` .\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta_{j}}=\\left(d_{ij}-\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right)x_{i}\n\n        for :math:`j=1,...,J`, for observations :math:`i=1,...,n`\n\n        In the multinomial model the score vector is K x (J-1) but is returned\n        as a flattened array. The Jacobian has the observations in rows and\n        the flattened array of derivatives in columns.\n        '
        params = params.reshape(self.K, -1, order='F')
        firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params))[:, 1:]
        return (firstterm[:, :, None] * self.exog[:, None, :]).reshape(self.exog.shape[0], -1)

    def hessian(self, params):
        if False:
            print('Hello World!')
        '\n        Multinomial logit Hessian matrix of the log-likelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hess : ndarray, (J*K, J*K)\n            The Hessian, second derivative of loglikelihood function with\n            respect to the flattened parameters, evaluated at `params`\n\n        Notes\n        -----\n        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta_{j}\\partial\\beta_{l}}=-\\sum_{i=1}^{n}\\frac{\\exp\\left(\\beta_{j}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\left[\\boldsymbol{1}\\left(j=l\\right)-\\frac{\\exp\\left(\\beta_{l}^{\\prime}x_{i}\\right)}{\\sum_{k=0}^{J}\\exp\\left(\\beta_{k}^{\\prime}x_{i}\\right)}\\right]x_{i}x_{l}^{\\prime}\n\n        where\n        :math:`\\boldsymbol{1}\\left(j=l\\right)` equals 1 if `j` = `l` and 0\n        otherwise.\n\n        The actual Hessian matrix has J**2 * K x K elements. Our Hessian\n        is reshaped to be square (J*K, J*K) so that the solvers can use it.\n\n        This implementation does not take advantage of the symmetry of\n        the Hessian and could probably be refactored for speed.\n        '
        params = params.reshape(self.K, -1, order='F')
        X = self.exog
        pr = self.cdf(np.dot(X, params))
        partials = []
        J = self.J
        K = self.K
        for i in range(J - 1):
            for j in range(J - 1):
                if i == j:
                    partials.append(-np.dot(((pr[:, i + 1] * (1 - pr[:, j + 1]))[:, None] * X).T, X))
                else:
                    partials.append(-np.dot(((pr[:, i + 1] * -pr[:, j + 1])[:, None] * X).T, X))
        H = np.array(partials)
        H = np.transpose(H.reshape(J - 1, J - 1, K, K), (0, 2, 1, 3)).reshape((J - 1) * K, (J - 1) * K)
        return H

class NegativeBinomial(CountModel):
    __doc__ = '\n    Negative Binomial Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n\n    References\n    ----------\n    Greene, W. 2008. "Functional forms for the negative binomial model\n        for count data". Economics Letters. Volume 99, Number 3, pp.585-590.\n    Hilbe, J.M. 2011. "Negative binomial regression". Cambridge University\n        Press.\n    ' % {'params': base._model_params_doc, 'extra_params': "loglike_method : str\n        Log-likelihood type. 'nb2','nb1', or 'geometric'.\n        Fitted value :math:`\\mu`\n        Heterogeneity parameter :math:`\\alpha`\n\n        - nb2: Variance equal to :math:`\\mu + \\alpha\\mu^2` (most common)\n        - nb1: Variance equal to :math:`\\mu + \\alpha\\mu`\n        - geometric: Variance equal to :math:`\\mu + \\mu^2`\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n    " + base._missing_param_doc + _check_rank_doc}

    def __init__(self, endog, exog, loglike_method='nb2', offset=None, exposure=None, missing='none', check_rank=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, check_rank=check_rank, **kwargs)
        self.loglike_method = loglike_method
        self._initialize()
        if loglike_method in ['nb2', 'nb1']:
            self.exog_names.append('alpha')
            self.k_extra = 1
        else:
            self.k_extra = 0
        self._init_keys.append('loglike_method')

    def _initialize(self):
        if False:
            return 10
        if self.loglike_method == 'nb2':
            self.hessian = self._hessian_nb2
            self.score = self._score_nbin
            self.loglikeobs = self._ll_nb2
            self._transparams = True
        elif self.loglike_method == 'nb1':
            self.hessian = self._hessian_nb1
            self.score = self._score_nb1
            self.loglikeobs = self._ll_nb1
            self._transparams = True
        elif self.loglike_method == 'geometric':
            self.hessian = self._hessian_geom
            self.score = self._score_geom
            self.loglikeobs = self._ll_geometric
        else:
            raise ValueError('Likelihood type must "nb1", "nb2" or "geometric"')

    def __getstate__(self):
        if False:
            print('Hello World!')
        odict = self.__dict__.copy()
        del odict['hessian']
        del odict['score']
        del odict['loglikeobs']
        return odict

    def __setstate__(self, indict):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(indict)
        self._initialize()

    def _ll_nbin(self, params, alpha, Q=0):
        if False:
            while True:
                i = 10
        if np.any(np.iscomplex(params)) or np.iscomplex(alpha):
            gamma_ln = loggamma
        else:
            gamma_ln = gammaln
        endog = self.endog
        mu = self.predict(params)
        size = 1 / alpha * mu ** Q
        prob = size / (size + mu)
        coeff = gamma_ln(size + endog) - gamma_ln(endog + 1) - gamma_ln(size)
        llf = coeff + size * np.log(prob) + endog * np.log(1 - prob)
        return llf

    def _ll_nb2(self, params):
        if False:
            while True:
                i = 10
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        return self._ll_nbin(params[:-1], alpha, Q=0)

    def _ll_nb1(self, params):
        if False:
            while True:
                i = 10
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        return self._ll_nbin(params[:-1], alpha, Q=1)

    def _ll_geometric(self, params):
        if False:
            return 10
        return self._ll_nbin(params, 1, 0)

    def loglike(self, params):
        if False:
            return 10
        '\n        Loglikelihood for negative binomial model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model. If `loglike_method` is nb1 or\n            nb2, then the ancillary parameter is expected to be the\n            last element.\n\n        Returns\n        -------\n        llf : float\n            The loglikelihood value at `params`\n\n        Notes\n        -----\n        Following notation in Greene (2008), with negative binomial\n        heterogeneity parameter :math:`\\alpha`:\n\n        .. math::\n\n           \\lambda_i &= exp(X\\beta) \\\\\n           \\theta &= 1 / \\alpha \\\\\n           g_i &= \\theta \\lambda_i^Q \\\\\n           w_i &= g_i/(g_i + \\lambda_i) \\\\\n           r_i &= \\theta / (\\theta+\\lambda_i) \\\\\n           ln \\mathcal{L}_i &= ln \\Gamma(y_i+g_i) - ln \\Gamma(1+y_i) + g_iln (r_i) + y_i ln(1-r_i)\n\n        where :math`Q=0` for NB2 and geometric and :math:`Q=1` for NB1.\n        For the geometric, :math:`\\alpha=0` as well.\n        '
        llf = np.sum(self.loglikeobs(params))
        return llf

    def _score_geom(self, params):
        if False:
            while True:
                i = 10
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        dparams = exog * (y - mu) / (mu + 1)
        return dparams.sum(0)

    def _score_nbin(self, params, Q=0):
        if False:
            i = 10
            return i + 15
        '\n        Score vector for NB2 model\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        a1 = 1 / alpha * mu ** Q
        prob = a1 / (a1 + mu)
        if Q == 1:
            dgpart = digamma(y + a1) - digamma(a1)
            dparams = exog * a1 * (np.log(prob) + dgpart)
            dalpha = ((alpha * (y - mu * np.log(prob) - mu * (dgpart + 1)) - mu * (np.log(prob) + dgpart)) / (alpha ** 2 * (alpha + 1))).sum()
        elif Q == 0:
            dgpart = digamma(y + a1) - digamma(a1)
            dparams = exog * a1 * (y - mu) / (mu + a1)
            da1 = -alpha ** (-2)
            dalpha = (dgpart + np.log(a1) - np.log(a1 + mu) - (y - mu) / (a1 + mu)).sum() * da1
        if self._transparams:
            return np.r_[dparams.sum(0), dalpha * alpha]
        else:
            return np.r_[dparams.sum(0), dalpha]

    def _score_nb1(self, params):
        if False:
            print('Hello World!')
        return self._score_nbin(params, Q=1)

    def _hessian_geom(self, params):
        if False:
            while True:
                i = 10
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        dim = exog.shape[1]
        hess_arr = np.empty((dim, dim))
        const_arr = mu * (1 + y) / (mu + 1) ** 2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i, j] = np.squeeze(np.sum(-exog[:, i, None] * exog[:, j, None] * const_arr, axis=0))
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        return hess_arr

    def _hessian_nb1(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hessian of NB1 model.\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        a1 = mu / alpha
        dgpart = digamma(y + a1) - digamma(a1)
        prob = 1 / (1 + alpha)
        dim = exog.shape[1]
        hess_arr = np.empty((dim + 1, dim + 1))
        dparams = exog / alpha * (np.log(prob) + dgpart)
        dmudb = exog * mu
        xmu_alpha = exog * a1
        trigamma = special.polygamma(1, a1 + y) - special.polygamma(1, a1)
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i, j] = np.squeeze(np.sum(dparams[:, i, None] * dmudb[:, j, None] + xmu_alpha[:, i, None] * xmu_alpha[:, j, None] * trigamma, axis=0))
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        dldpda = np.sum(-a1 * dparams + exog * a1 * (-trigamma * mu / alpha ** 2 - prob), axis=0)
        hess_arr[-1, :-1] = dldpda
        hess_arr[:-1, -1] = dldpda
        log_alpha = np.log(prob)
        alpha3 = alpha ** 3
        alpha2 = alpha ** 2
        mu2 = mu ** 2
        dada = (alpha3 * mu * (2 * log_alpha + 2 * dgpart + 3) - 2 * alpha3 * y + 4 * alpha2 * mu * (log_alpha + dgpart) + alpha2 * (2 * mu - y) + 2 * alpha * mu2 * trigamma + mu2 * trigamma + alpha2 * mu2 * trigamma + 2 * alpha * mu * (log_alpha + dgpart)) / (alpha ** 4 * (alpha2 + 2 * alpha + 1))
        hess_arr[-1, -1] = dada.sum()
        return hess_arr

    def _hessian_nb2(self, params):
        if False:
            print('Hello World!')
        '\n        Hessian of NB2 model.\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        a1 = 1 / alpha
        params = params[:-1]
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        prob = a1 / (a1 + mu)
        dgpart = digamma(a1 + y) - digamma(a1)
        dim = exog.shape[1]
        hess_arr = np.empty((dim + 1, dim + 1))
        const_arr = a1 * mu * (a1 + y) / (mu + a1) ** 2
        for i in range(dim):
            for j in range(dim):
                if j > i:
                    continue
                hess_arr[i, j] = np.sum(-exog[:, i, None] * exog[:, j, None] * const_arr, axis=0).squeeze()
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        da1 = -alpha ** (-2)
        dldpda = -np.sum(mu * exog * (y - mu) * a1 ** 2 / (mu + a1) ** 2, axis=0)
        hess_arr[-1, :-1] = dldpda
        hess_arr[:-1, -1] = dldpda
        da2 = 2 * alpha ** (-3)
        dalpha = da1 * (dgpart + np.log(prob) - (y - mu) / (a1 + mu))
        dada = (da2 * dalpha / da1 + da1 ** 2 * (special.polygamma(1, a1 + y) - special.polygamma(1, a1) + 1 / a1 - 1 / (a1 + mu) + (y - mu) / (mu + a1) ** 2)).sum()
        hess_arr[-1, -1] = dada
        return hess_arr

    def score_obs(self, params):
        if False:
            while True:
                i = 10
        sc = approx_fprime_cs(params, self.loglikeobs)
        return sc

    @Appender(Poisson.predict.__doc__)
    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', linear=None, y_values=None):
        if False:
            print('Hello World!')
        if linear is not None:
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.warn(msg, FutureWarning)
            if linear is True:
                which = 'linear'
        if which == 'prob':
            distr = self.get_distribution(params, exog=exog, exposure=exposure, offset=offset)
            if y_values is None:
                y_values = np.arange(0, np.max(self.endog) + 1)
            else:
                y_values = np.asarray(y_values)
            assert y_values.ndim == 1
            y_values = y_values[..., None]
            return distr.pmf(y_values).T
        (exog, offset, exposure) = self._get_predict_arrays(exog=exog, offset=offset, exposure=exposure)
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            return np.exp(linpred)
        elif which.startswith('lin'):
            return linpred
        elif which == 'var':
            mu = np.exp(linpred)
            if self.loglike_method == 'geometric':
                var_ = mu * (1 + mu)
            else:
                if self.loglike_method == 'nb2':
                    p = 2
                elif self.loglike_method == 'nb1':
                    p = 1
                alpha = params[-1]
                var_ = mu * (1 + alpha * mu ** (p - 1))
            return var_
        else:
            raise ValueError('keyword which has to be "mean" and "linear"')

    @Appender(_get_start_params_null_docs)
    def _get_start_params_null(self):
        if False:
            return 10
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]
        mu = const * np.exp(offset + exposure)
        resid = self.endog - mu
        a = self._estimate_dispersion(mu, resid, df_resid=resid.shape[0] - 1)
        params.append(a)
        return np.array(params)

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        if False:
            while True:
                i = 10
        if df_resid is None:
            df_resid = resid.shape[0]
        if self.loglike_method == 'nb2':
            a = ((resid ** 2 / mu - 1) / mu).sum() / df_resid
        else:
            a = (resid ** 2 / mu - 1).sum() / df_resid
        return a

    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, cov_type='nonrobust', cov_kwds=None, use_t=None, optim_kwds_prelim=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._transparams = False
        if self.loglike_method.startswith('nb') and method not in ['newton', 'ncg']:
            self._transparams = True
        elif self.loglike_method.startswith('nb'):
            self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            kwds_prelim = {'disp': 0, 'skip_hessian': True, 'warn_convergence': False}
            if optim_kwds_prelim is not None:
                kwds_prelim.update(optim_kwds_prelim)
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                res_poi = mod_poi.fit(**kwds_prelim)
            start_params = res_poi.params
            if self.loglike_method.startswith('nb'):
                a = self._estimate_dispersion(res_poi.predict(), res_poi.resid, df_resid=res_poi.df_resid)
                start_params = np.append(start_params, max(0.05, a))
        elif self._transparams is True:
            start_params = np.array(start_params, copy=True)
            start_params[-1] = np.log(start_params[-1])
        if callback is None:
            callback = lambda *x: x
        mlefit = super().fit(start_params=start_params, maxiter=maxiter, method=method, disp=disp, full_output=full_output, callback=callback, **kwargs)
        if optim_kwds_prelim is not None:
            mlefit.mle_settings['optim_kwds_prelim'] = optim_kwds_prelim
        if self.loglike_method.startswith('nb'):
            self._transparams = False
            if method not in ['newton', 'ncg']:
                mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
            nbinfit = NegativeBinomialResults(self, mlefit._results)
            result = NegativeBinomialResultsWrapper(nbinfit)
        else:
            result = mlefit
        if cov_kwds is None:
            cov_kwds = {}
        result._get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        return result

    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if False:
            i = 10
            return i + 15
        _validate_l1_method(method)
        if self.loglike_method.startswith('nb') and (np.size(alpha) == 1 and alpha != 0):
            k_params = self.exog.shape[1] + self.k_extra
            alpha = alpha * np.ones(k_params)
            alpha[-1] = 0
        alpha_p = alpha[:-1] if self.k_extra and np.size(alpha) > 1 else alpha
        self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                start_params = mod_poi.fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=0, callback=callback, alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            if self.loglike_method.startswith('nb'):
                start_params = np.append(start_params, 0.1)
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1NegativeBinomialResults(self, cntfit)
        return L1NegativeBinomialResultsWrapper(discretefit)

    @Appender(Poisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exposure=None, offset=None):
        if False:
            while True:
                i = 10
        'get frozen instance of distribution\n        '
        mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)
        if self.loglike_method == 'geometric':
            distr = stats.geom(1 / (1 + mu), loc=-1)
        else:
            if self.loglike_method == 'nb2':
                p = 2
            elif self.loglike_method == 'nb1':
                p = 1
            alpha = params[-1]
            q = 2 - p
            size = 1.0 / alpha * mu ** q
            prob = size / (size + mu)
            distr = nbinom(size, prob)
        return distr

class NegativeBinomialP(CountModel):
    __doc__ = '\n    Generalized Negative Binomial (NB-P) Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    p : scalar\n        P denotes parameterizations for NB-P regression. p=1 for NB-1 and\n        p=2 for NB-2. Default is p=1.\n    ' % {'params': base._model_params_doc, 'extra_params': 'p : scalar\n        P denotes parameterizations for NB regression. p=1 for NB-1 and\n        p=2 for NB-2. Default is p=2.\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n        ' + base._missing_param_doc + _check_rank_doc}

    def __init__(self, endog, exog, p=2, offset=None, exposure=None, missing='none', check_rank=True, **kwargs):
        if False:
            return 10
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, check_rank=check_rank, **kwargs)
        self.parameterization = p
        self.exog_names.append('alpha')
        self.k_extra = 1
        self._transparams = False

    def _get_init_kwds(self):
        if False:
            return 10
        kwds = super()._get_init_kwds()
        kwds['p'] = self.parameterization
        return kwds

    def _get_exogs(self):
        if False:
            while True:
                i = 10
        return (self.exog, None)

    def loglike(self, params):
        if False:
            return 10
        '\n        Loglikelihood of Generalized Negative Binomial (NB-P) model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : float\n            The log-likelihood function of the model evaluated at `params`.\n            See notes.\n        '
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        if False:
            return 10
        '\n        Loglikelihood for observations of Generalized Negative Binomial (NB-P) model\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model.\n\n        Returns\n        -------\n        loglike : ndarray\n            The log likelihood for each observation of the model evaluated\n            at `params`. See Notes\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = mu ** (2 - p)
        a1 = mu_p / alpha
        a2 = mu + a1
        llf = gammaln(y + a1) - gammaln(y + 1) - gammaln(a1) + a1 * np.log(a1) + y * np.log(mu) - (y + a1) * np.log(a2)
        return llf

    def score_obs(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generalized Negative Binomial (NB-P) model score (gradient) vector of the log-likelihood for each observations.\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score : ndarray, 1-D\n            The score vector of the model, i.e. the first derivative of the\n            loglikelihood function, evaluated at `params`\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        dgpart = digamma(a3) - digamma(a1)
        dgterm = dgpart + np.log(a1 / a2) + 1 - a3 / a2
        dparams = a4 * dgterm - a3 / a2 + y / mu
        dparams = (self.exog.T * mu * dparams).T
        dalpha = -a1 / alpha * dgterm
        return np.concatenate((dparams, np.atleast_2d(dalpha).T), axis=1)

    def score(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generalized Negative Binomial (NB-P) model score (gradient) vector of the log-likelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        score : ndarray, 1-D\n            The score vector of the model, i.e. the first derivative of the\n            loglikelihood function, evaluated at `params`\n        '
        score = np.sum(self.score_obs(params), axis=0)
        if self._transparams:
            score[-1] == score[-1] ** 2
            return score
        else:
            return score

    def score_factor(self, params, endog=None):
        if False:
            return 10
        '\n        Generalized Negative Binomial (NB-P) model score (gradient) vector of the log-likelihood for each observations.\n\n        Parameters\n        ----------\n        params : array-like\n            The parameters of the model\n\n        Returns\n        -------\n        score : ndarray, 1-D\n            The score vector of the model, i.e. the first derivative of the\n            loglikelihood function, evaluated at `params`\n        '
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog if endog is None else endog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        dgpart = digamma(a3) - digamma(a1)
        dparams = a4 * dgpart - a3 / a2 + y / mu + a4 * (1 - a3 / a2 + np.log(a1 / a2))
        dparams = (mu * dparams).T
        dalpha = -a1 / alpha * (dgpart + np.log(a1 / a2) + 1 - a3 / a2)
        return (dparams, dalpha)

    def hessian(self, params):
        if False:
            return 10
        '\n        Generalized Negative Binomial (NB-P) model hessian maxtrix of the log-likelihood\n\n        Parameters\n        ----------\n        params : array_like\n            The parameters of the model\n\n        Returns\n        -------\n        hessian : ndarray, 2-D\n            The hessian matrix of the model.\n        '
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog
        exog = self.exog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        prob = a1 / a2
        lprob = np.log(prob)
        dgpart = digamma(a3) - digamma(a1)
        pgpart = polygamma(1, a3) - polygamma(1, a1)
        dim = exog.shape[1]
        hess_arr = np.zeros((dim + 1, dim + 1))
        coeff = mu ** 2 * ((1 + a4) ** 2 * a3 / a2 ** 2 - a3 / a2 * (p - 1) * a4 / mu - y / mu ** 2 - 2 * a4 * (1 + a4) / a2 + p * a4 / mu * (lprob + dgpart + 2) - a4 / mu * (lprob + dgpart + 1) + a4 ** 2 * pgpart + (-(1 + a4) * a3 / a2 + y / mu + a4 * (lprob + dgpart + 1)) / mu)
        for i in range(dim):
            hess_arr[i, :-1] = np.sum(self.exog[:, :].T * self.exog[:, i] * coeff, axis=1)
        hess_arr[-1, :-1] = (self.exog[:, :].T * mu * a1 * ((1 + a4) * (1 - a3 / a2) / a2 - p * (lprob + dgpart + 2) / mu + p / mu * (a3 + p * a1) / a2 - a4 * pgpart) / alpha).sum(axis=1)
        da2 = a1 * (2 * lprob + 2 * dgpart + 3 - 2 * a3 / a2 + a1 * pgpart - 2 * prob + prob * a3 / a2) / alpha ** 2
        hess_arr[-1, -1] = da2.sum()
        tri_idx = np.triu_indices(dim + 1, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        return hess_arr

    def hessian_factor(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Generalized Negative Binomial (NB-P) model hessian maxtrix of the log-likelihood\n\n        Parameters\n        ----------\n        params : array-like\n            The parameters of the model\n\n        Returns\n        -------\n        hessian : ndarray, 2-D\n            The hessian matrix of the model.\n        '
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = 2 - self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = mu ** p
        a1 = mu_p / alpha
        a2 = mu + a1
        a3 = y + a1
        a4 = p * a1 / mu
        a5 = a4 * p / mu
        dgpart = digamma(a3) - digamma(a1)
        coeff = mu ** 2 * ((1 + a4) ** 2 * a3 / a2 ** 2 - a3 * (a5 - a4 / mu) / a2 - y / mu ** 2 - 2 * a4 * (1 + a4) / a2 + a5 * (np.log(a1) - np.log(a2) + dgpart + 2) - a4 * (np.log(a1) - np.log(a2) + dgpart + 1) / mu - a4 ** 2 * (polygamma(1, a1) - polygamma(1, a3)) + (-(1 + a4) * a3 / a2 + y / mu + a4 * (np.log(a1) - np.log(a2) + dgpart + 1)) / mu)
        hfbb = coeff
        hfba = mu * a1 * ((1 + a4) * (1 - a3 / a2) / a2 - p * (np.log(a1 / a2) + dgpart + 2) / mu + p * (a3 / mu + a4) / a2 + a4 * (polygamma(1, a1) - polygamma(1, a3))) / alpha
        hfaa = a1 * (2 * np.log(a1 / a2) + 2 * dgpart + 3 - 2 * a3 / a2 - a1 * polygamma(1, a1) + a1 * polygamma(1, a3) - 2 * a1 / a2 + a1 * a3 / a2 ** 2) / alpha ** 2
        return (hfbb, hfba, hfaa)

    @Appender(_get_start_params_null_docs)
    def _get_start_params_null(self):
        if False:
            print('Hello World!')
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]
        mu = const * np.exp(offset + exposure)
        resid = self.endog - mu
        a = self._estimate_dispersion(mu, resid, df_resid=resid.shape[0] - 1)
        params.append(a)
        return np.array(params)

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        if False:
            for i in range(10):
                print('nop')
        q = self.parameterization - 1
        if df_resid is None:
            df_resid = resid.shape[0]
        a = ((resid ** 2 / mu - 1) * mu ** (-q)).sum() / df_resid
        return a

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, use_transparams=False, cov_type='nonrobust', cov_kwds=None, use_t=None, optim_kwds_prelim=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        use_transparams : bool\n            This parameter enable internal transformation to impose\n            non-negativity. True to enable. Default is False.\n            use_transparams=True imposes the no underdispersion (alpha > 0)\n            constraint. In case use_transparams=True and method="newton" or\n            "ncg" transformation is ignored.\n        '
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = True
        else:
            if use_transparams:
                warnings.warn('Parameter "use_transparams" is ignored', RuntimeWarning)
            self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            kwds_prelim = {'disp': 0, 'skip_hessian': True, 'warn_convergence': False}
            if optim_kwds_prelim is not None:
                kwds_prelim.update(optim_kwds_prelim)
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                res_poi = mod_poi.fit(**kwds_prelim)
            start_params = res_poi.params
            a = self._estimate_dispersion(res_poi.predict(), res_poi.resid, df_resid=res_poi.df_resid)
            start_params = np.append(start_params, max(0.05, a))
        if callback is None:
            callback = lambda *x: x
        mlefit = super(NegativeBinomialP, self).fit(start_params=start_params, maxiter=maxiter, method=method, disp=disp, full_output=full_output, callback=callback, **kwargs)
        if optim_kwds_prelim is not None:
            mlefit.mle_settings['optim_kwds_prelim'] = optim_kwds_prelim
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = False
            mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
        nbinfit = NegativeBinomialPResults(self, mlefit._results)
        result = NegativeBinomialPResultsWrapper(nbinfit)
        if cov_kwds is None:
            cov_kwds = {}
        result._get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        return result

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if False:
            while True:
                i = 10
        _validate_l1_method(method)
        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1] + self.k_extra
            alpha = alpha * np.ones(k_params)
            alpha[-1] = 0
        alpha_p = alpha[:-1] if self.k_extra and np.size(alpha) > 1 else alpha
        self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                start_params = mod_poi.fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=0, callback=callback, alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            start_params = np.append(start_params, 0.1)
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1NegativeBinomialResults(self, cntfit)
        return L1NegativeBinomialResultsWrapper(discretefit)

    @Appender(Poisson.predict.__doc__)
    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', y_values=None):
        if False:
            return 10
        if exog is None:
            exog = self.exog
        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)
        if offset is None:
            offset = getattr(self, 'offset', 0)
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            return np.exp(linpred)
        elif which == 'linear':
            return linpred
        elif which == 'var':
            mean = np.exp(linpred)
            alpha = params[-1]
            p = self.parameterization
            var_ = mean * (1 + alpha * mean ** (p - 1))
            return var_
        elif which == 'prob':
            if y_values is None:
                y_values = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = self.predict(params, exog, exposure, offset)
            (size, prob) = self.convert_params(params, mu)
            return nbinom.pmf(y_values, size[:, None], prob[:, None])
        else:
            raise ValueError('keyword "which" = %s not recognized' % which)

    def convert_params(self, params, mu):
        if False:
            return 10
        alpha = params[-1]
        p = 2 - self.parameterization
        size = 1.0 / alpha * mu ** p
        prob = size / (size + mu)
        return (size, prob)

    def _deriv_score_obs_dendog(self, params):
        if False:
            for i in range(10):
                print('nop')
        'derivative of score_obs w.r.t. endog\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        derivative : ndarray_2d\n            The derivative of the score_obs with respect to endog.\n        '
        from statsmodels.tools.numdiff import _approx_fprime_cs_scalar

        def f(y):
            if False:
                return 10
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            sf = self.score_factor(params, endog=y)
            return np.column_stack(sf)
        dsf = _approx_fprime_cs_scalar(self.endog[:, None], f)
        d1 = dsf[:, :1] * self.exog
        d2 = dsf[:, 1:2]
        return np.column_stack((d1, d2))

    def _var(self, mu, params=None):
        if False:
            i = 10
            return i + 15
        'variance implied by the distribution\n\n        internal use, will be refactored or removed\n        '
        alpha = params[-1]
        p = self.parameterization
        var_ = mu * (1 + alpha * mu ** (p - 1))
        return var_

    def _prob_nonzero(self, mu, params):
        if False:
            i = 10
            return i + 15
        'Probability that count is not zero\n\n        internal use in Censored model, will be refactored or removed\n        '
        alpha = params[-1]
        p = self.parameterization
        prob_nz = 1 - (1 + alpha * mu ** (p - 1)) ** (-1 / alpha)
        return prob_nz

    @Appender(Poisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exposure=None, offset=None):
        if False:
            print('Hello World!')
        'get frozen instance of distribution\n        '
        mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)
        (size, prob) = self.convert_params(params, mu)
        distr = nbinom(size, prob)
        return distr

class DiscreteResults(base.LikelihoodModelResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for the discrete dependent variable models.', 'extra_attr': ''}

    def __init__(self, model, mlefit, cov_type='nonrobust', cov_kwds=None, use_t=None):
        if False:
            while True:
                i = 10
        self.model = model
        self.method = 'MLE'
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = {}
        self.nobs = model.exog.shape[0]
        self.__dict__.update(mlefit.__dict__)
        self.converged = mlefit.mle_retvals['converged']
        if not hasattr(self, 'cov_type'):
            if use_t is not None:
                self.use_t = use_t
            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description': 'Standard Errors assume that the ' + 'covariance matrix of the errors is correctly ' + 'specified.'}
            else:
                if cov_kwds is None:
                    cov_kwds = {}
                from statsmodels.base.covtype import get_robustcov_results
                get_robustcov_results(self, cov_type=cov_type, use_self=True, **cov_kwds)

    def __getstate__(self):
        if False:
            print('Hello World!')
        mle_settings = getattr(self, 'mle_settings', None)
        if mle_settings is not None:
            if 'callback' in mle_settings:
                mle_settings['callback'] = None
            if 'cov_params_func' in mle_settings:
                mle_settings['cov_params_func'] = None
        return self.__dict__

    @cache_readonly
    def prsquared(self):
        if False:
            i = 10
            return i + 15
        "\n        McFadden's pseudo-R-squared. `1 - (llf / llnull)`\n        "
        return 1 - self.llf / self.llnull

    @cache_readonly
    def llr(self):
        if False:
            while True:
                i = 10
        '\n        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`\n        '
        return -2 * (self.llnull - self.llf)

    @cache_readonly
    def llr_pvalue(self):
        if False:
            while True:
                i = 10
        '\n        The chi-squared probability of getting a log-likelihood ratio\n        statistic greater than llr.  llr has a chi-squared distribution\n        with degrees of freedom `df_model`.\n        '
        return stats.distributions.chi2.sf(self.llr, self.df_model)

    def set_null_options(self, llnull=None, attach_results=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the fit options for the Null (constant-only) model.\n\n        This resets the cache for related attributes which is potentially\n        fragile. This only sets the option, the null model is estimated\n        when llnull is accessed, if llnull is not yet in cache.\n\n        Parameters\n        ----------\n        llnull : {None, float}\n            If llnull is not None, then the value will be directly assigned to\n            the cached attribute "llnull".\n        attach_results : bool\n            Sets an internal flag whether the results instance of the null\n            model should be attached. By default without calling this method,\n            thenull model results are not attached and only the loglikelihood\n            value llnull is stored.\n        **kwargs\n            Additional keyword arguments used as fit keyword arguments for the\n            null model. The override and model default values.\n\n        Notes\n        -----\n        Modifies attributes of this instance, and so has no return.\n        '
        self._cache.pop('llnull', None)
        self._cache.pop('llr', None)
        self._cache.pop('llr_pvalue', None)
        self._cache.pop('prsquared', None)
        if hasattr(self, 'res_null'):
            del self.res_null
        if llnull is not None:
            self._cache['llnull'] = llnull
        self._attach_nullmodel = attach_results
        self._optim_kwds_null = kwargs

    @cache_readonly
    def llnull(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Value of the constant-only loglikelihood\n        '
        model = self.model
        kwds = model._get_init_kwds().copy()
        for key in getattr(model, '_null_drop_keys', []):
            del kwds[key]
        mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
        optim_kwds = getattr(self, '_optim_kwds_null', {}).copy()
        if 'start_params' in optim_kwds:
            sp_null = optim_kwds.pop('start_params')
        elif hasattr(model, '_get_start_params_null'):
            sp_null = model._get_start_params_null()
        else:
            sp_null = None
        opt_kwds = dict(method='bfgs', warn_convergence=False, maxiter=10000, disp=0)
        opt_kwds.update(optim_kwds)
        if optim_kwds:
            res_null = mod_null.fit(start_params=sp_null, **opt_kwds)
        else:
            res_null = mod_null.fit(start_params=sp_null, method='nm', warn_convergence=False, maxiter=10000, disp=0)
            res_null = mod_null.fit(start_params=res_null.params, method='bfgs', warn_convergence=False, maxiter=10000, disp=0)
        if getattr(self, '_attach_nullmodel', False) is not False:
            self.res_null = res_null
        return res_null.llf

    @cache_readonly
    def fittedvalues(self):
        if False:
            while True:
                i = 10
        '\n        Linear predictor XB.\n        '
        return np.dot(self.model.exog, self.params[:self.model.exog.shape[1]])

    @cache_readonly
    def resid_response(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Respnose residuals. The response residuals are defined as\n        `endog - fittedvalues`\n        '
        return self.model.endog - self.predict()

    @cache_readonly
    def resid_pearson(self):
        if False:
            return 10
        '\n        Pearson residuals defined as response residuals divided by standard\n        deviation implied by the model.\n        '
        var_ = self.predict(which='var')
        return self.resid_response / np.sqrt(var_)

    @cache_readonly
    def aic(self):
        if False:
            while True:
                i = 10
        '\n        Akaike information criterion.  `-2*(llf - p)` where `p` is the number\n        of regressors including the intercept.\n        '
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * (self.llf - (self.df_model + 1 + k_extra))

    @cache_readonly
    def bic(self):
        if False:
            print('Hello World!')
        '\n        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the\n        number of regressors including the intercept.\n        '
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + 1 + k_extra)

    @cache_readonly
    def im_ratio(self):
        if False:
            for i in range(10):
                print('nop')
        return pinfer.im_ratio(self)

    def info_criteria(self, crit, dk_params=0):
        if False:
            i = 10
            return i + 15
        "Return an information criterion for the model.\n\n        Parameters\n        ----------\n        crit : string\n            One of 'aic', 'bic', 'tic' or 'gbic'.\n        dk_params : int or float\n            Correction to the number of parameters used in the information\n            criterion.\n\n        Returns\n        -------\n        Value of information criterion.\n\n        Notes\n        -----\n        Tic and gbic\n\n        References\n        ----------\n        Burnham KP, Anderson KR (2002). Model Selection and Multimodel\n        Inference; Springer New York.\n        "
        crit = crit.lower()
        k_extra = getattr(self.model, 'k_extra', 0)
        k_params = self.df_model + 1 + k_extra + dk_params
        if crit == 'aic':
            return -2 * self.llf + 2 * k_params
        elif crit == 'bic':
            nobs = self.df_model + self.df_resid + 1
            bic = -2 * self.llf + k_params * np.log(nobs)
            return bic
        elif crit == 'tic':
            return pinfer.tic(self)
        elif crit == 'gbic':
            return pinfer.gbic(self)
        else:
            raise ValueError('Name of information criterion not recognized.')

    def score_test(self, exog_extra=None, params_constrained=None, hypothesis='joint', cov_type=None, cov_kwds=None, k_constraints=None, observed=True):
        if False:
            return 10
        res = pinfer.score_test(self, exog_extra=exog_extra, params_constrained=params_constrained, hypothesis=hypothesis, cov_type=cov_type, cov_kwds=cov_kwds, k_constraints=k_constraints, observed=observed)
        return res
    score_test.__doc__ = pinfer.score_test.__doc__

    def get_prediction(self, exog=None, transform=True, which='mean', linear=None, row_labels=None, average=False, agg_weights=None, y_values=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Compute prediction results when endpoint transformation is valid.\n\n        Parameters\n        ----------\n        exog : array_like, optional\n            The values for which you want to predict.\n        transform : bool, optional\n            If the model was fit via a formula, do you want to pass\n            exog through the formula. Default is True. E.g., if you fit\n            a model y ~ log(x1) + log(x2), and transform is True, then\n            you can pass a data structure that contains x1 and x2 in\n            their original form. Otherwise, you\'d need to log the data\n            first.\n        which : str\n            Which statistic is to be predicted. Default is "mean".\n            The available statistics and options depend on the model.\n            see the model.predict docstring\n        linear : bool\n            Linear has been replaced by the `which` keyword and will be\n            deprecated.\n            If linear is True, then `which` is ignored and the linear\n            prediction is returned.\n        row_labels : list of str or None\n            If row_lables are provided, then they will replace the generated\n            labels.\n        average : bool\n            If average is True, then the mean prediction is computed, that is,\n            predictions are computed for individual exog and then the average\n            over observation is used.\n            If average is False, then the results are the predictions for all\n            observations, i.e. same length as ``exog``.\n        agg_weights : ndarray, optional\n            Aggregation weights, only used if average is True.\n            The weights are not normalized.\n        y_values : None or nd_array\n            Some predictive statistics like which="prob" are computed at\n            values of the response variable. If y_values is not None, then\n            it will be used instead of the default set of y_values.\n\n            **Warning:** ``which="prob"`` for count models currently computes\n            the pmf for all y=k up to max(endog). This can be a large array if\n            the observed endog values are large.\n            This will likely change so that the set of y_values will be chosen\n            to limit the array size.\n        **kwargs :\n            Some models can take additional keyword arguments, such as offset,\n            exposure or additional exog in multi-part models like zero inflated\n            models.\n            See the predict method of the model for the details.\n\n        Returns\n        -------\n        prediction_results : PredictionResults\n            The prediction results instance contains prediction and prediction\n            variance and can on demand calculate confidence intervals and\n            summary dataframe for the prediction.\n\n        Notes\n        -----\n        Status: new in 0.14, experimental\n        '
        if linear is True:
            which = 'linear'
        pred_kwds = kwargs
        if y_values is not None:
            pred_kwds['y_values'] = y_values
        res = pred.get_prediction(self, exog=exog, which=which, transform=transform, row_labels=row_labels, average=average, agg_weights=agg_weights, pred_kwds=pred_kwds)
        return res

    def get_distribution(self, exog=None, transform=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (exog, _) = self._transform_predict_exog(exog, transform=transform)
        if exog is not None:
            exog = np.asarray(exog)
        distr = self.model.get_distribution(self.params, exog=exog, **kwargs)
        return distr

    def _get_endog_name(self, yname, yname_list):
        if False:
            print('Hello World!')
        if yname is None:
            yname = self.model.endog_names
        if yname_list is None:
            yname_list = self.model.endog_names
        return (yname, yname_list)

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        if False:
            while True:
                i = 10
        "Get marginal effects of the fitted model.\n\n        Parameters\n        ----------\n        at : str, optional\n            Options are:\n\n            - 'overall', The average of the marginal effects at each\n              observation.\n            - 'mean', The marginal effects at the mean of each regressor.\n            - 'median', The marginal effects at the median of each regressor.\n            - 'zero', The marginal effects at zero for each regressor.\n            - 'all', The marginal effects at each observation. If `at` is all\n              only margeff will be available from the returned object.\n\n            Note that if `exog` is specified, then marginal effects for all\n            variables not specified by `exog` are calculated using the `at`\n            option.\n        method : str, optional\n            Options are:\n\n            - 'dydx' - dy/dx - No transformation is made and marginal effects\n              are returned.  This is the default.\n            - 'eyex' - estimate elasticities of variables in `exog` --\n              d(lny)/d(lnx)\n            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)\n            - 'eydx' - estimate semi-elasticity -- d(lny)/dx\n\n            Note that tranformations are done after each observation is\n            calculated.  Semi-elasticities for binary variables are computed\n            using the midpoint method. 'dyex' and 'eyex' do not make sense\n            for discrete variables. For interpretations of these methods\n            see notes below.\n        atexog : array_like, optional\n            Optionally, you can provide the exogenous variables over which to\n            get the marginal effects.  This should be a dictionary with the key\n            as the zero-indexed column number and the value of the dictionary.\n            Default is None for all independent variables less the constant.\n        dummy : bool, optional\n            If False, treats binary variables (if present) as continuous.  This\n            is the default.  Else if True, treats binary variables as\n            changing from 0 to 1.  Note that any variable that is either 0 or 1\n            is treated as binary.  Each binary variable is treated separately\n            for now.\n        count : bool, optional\n            If False, treats count variables (if present) as continuous.  This\n            is the default.  Else if True, the marginal effect is the\n            change in probabilities when each observation is increased by one.\n\n        Returns\n        -------\n        DiscreteMargins : marginal effects instance\n            Returns an object that holds the marginal effects, standard\n            errors, confidence intervals, etc. See\n            `statsmodels.discrete.discrete_margins.DiscreteMargins` for more\n            information.\n\n        Notes\n        -----\n        Interpretations of methods:\n\n        - 'dydx' - change in `endog` for a change in `exog`.\n        - 'eyex' - proportional change in `endog` for a proportional change\n          in `exog`.\n        - 'dyex' - change in `endog` for a proportional change in `exog`.\n        - 'eydx' - proportional change in `endog` for a change in `exog`.\n\n        When using after Poisson, returns the expected number of events per\n        period, assuming that the model is loglinear.\n        "
        if getattr(self.model, 'offset', None) is not None:
            raise NotImplementedError('Margins with offset are not available.')
        from statsmodels.discrete.discrete_margins import DiscreteMargins
        return DiscreteMargins(self, (at, method, atexog, dummy, count))

    def get_influence(self):
        if False:
            while True:
                i = 10
        '\n        Get an instance of MLEInfluence with influence and outlier measures\n\n        Returns\n        -------\n        infl : MLEInfluence instance\n            The instance has methods to calculate the main influence and\n            outlier measures as attributes.\n\n        See Also\n        --------\n        statsmodels.stats.outliers_influence.MLEInfluence\n        '
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Summarize the Regression Results.\n\n        Parameters\n        ----------\n        yname : str, optional\n            The name of the endog variable in the tables. The default is `y`.\n        xname : list[str], optional\n            The names for the exogenous variables, default is "var_xx".\n            Must match the number of parameters in the model.\n        title : str, optional\n            Title for the top table. If not None, then this replaces the\n            default title.\n        alpha : float\n            The significance level for the confidence intervals.\n\n        Returns\n        -------\n        Summary\n            Class that holds the summary tables and text, which can be printed\n            or converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary.Summary : Class that hold summary results.\n        '
        top_left = [('Dep. Variable:', None), ('Model:', [self.model.__class__.__name__]), ('Method:', [self.method]), ('Date:', None), ('Time:', None), ('converged:', ['%s' % self.mle_retvals['converged']])]
        top_right = [('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None), ('Pseudo R-squ.:', ['%#6.4g' % self.prsquared]), ('Log-Likelihood:', None), ('LL-Null:', ['%#8.5g' % self.llnull]), ('LLR p-value:', ['%#6.4g' % self.llr_pvalue])]
        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        (yname, yname_list) = self._get_endog_name(yname, yname_list)
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname_list, xname=xname, alpha=alpha, use_t=self.use_t)
        if hasattr(self, 'constraints'):
            smry.add_extra_txt(['Model has been estimated subject to linear equality constraints.'])
        return smry

    def summary2(self, yname=None, xname=None, title=None, alpha=0.05, float_format='%.4f'):
        if False:
            print('Hello World!')
        '\n        Experimental function to summarize regression results.\n\n        Parameters\n        ----------\n        yname : str\n            Name of the dependent variable (optional).\n        xname : list[str], optional\n            List of strings of length equal to the number of parameters\n            Names of the independent variables (optional).\n        title : str, optional\n            Title for the top table. If not None, then this replaces the\n            default title.\n        alpha : float\n            The significance level for the confidence intervals.\n        float_format : str\n            The print format for floats in parameters summary.\n\n        Returns\n        -------\n        Summary\n            Instance that contains the summary tables and text, which can be\n            printed or converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary2.Summary : Class that holds summary results.\n        '
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format, xname=xname, yname=yname, title=title)
        if hasattr(self, 'constraints'):
            smry.add_text('Model has been estimated subject to linear equality constraints.')
        return smry

class CountResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for count data', 'extra_attr': ''}

    @cache_readonly
    def resid(self):
        if False:
            print('Hello World!')
        '\n        Residuals\n\n        Notes\n        -----\n        The residuals for Count models are defined as\n\n        .. math:: y - p\n\n        where :math:`p = \\exp(X\\beta)`. Any exposure and offset variables\n        are also handled.\n        '
        return self.model.endog - self.predict()

    def get_diagnostic(self, y_max=None):
        if False:
            return 10
        '\n        Get instance of class with specification and diagnostic methods.\n\n        experimental, API of Diagnostic classes will change\n\n        Returns\n        -------\n        CountDiagnostic instance\n            The instance has methods to perform specification and diagnostic\n            tesst and plots\n\n        See Also\n        --------\n        statsmodels.statsmodels.discrete.diagnostic.CountDiagnostic\n        '
        from statsmodels.discrete.diagnostic import CountDiagnostic
        return CountDiagnostic(self, y_max=y_max)

class NegativeBinomialResults(CountResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for NegativeBinomial 1 and 2', 'extra_attr': ''}

    @cache_readonly
    def lnalpha(self):
        if False:
            i = 10
            return i + 15
        'Natural log of alpha'
        return np.log(self.params[-1])

    @cache_readonly
    def lnalpha_std_err(self):
        if False:
            while True:
                i = 10
        'Natural log of standardized error'
        return self.bse[-1] / self.params[-1]

    @cache_readonly
    def aic(self):
        if False:
            while True:
                i = 10
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * (self.llf - (self.df_model + self.k_constant + k_extra))

    @cache_readonly
    def bic(self):
        if False:
            i = 10
            return i + 15
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + self.k_constant + k_extra)

class NegativeBinomialPResults(NegativeBinomialResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for NegativeBinomialP', 'extra_attr': ''}

class GeneralizedPoissonResults(NegativeBinomialResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Generalized Poisson', 'extra_attr': ''}

    @cache_readonly
    def _dispersion_factor(self):
        if False:
            i = 10
            return i + 15
        p = getattr(self.model, 'parameterization', 0)
        mu = self.predict()
        return (1 + self.params[-1] * mu ** p) ** 2

class L1CountResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for count data fit by l1 regularization', 'extra_attr': _l1_results_attr}

    def __init__(self, model, cntfit):
        if False:
            return 10
        super(L1CountResults, self).__init__(model, cntfit)
        self.trimmed = cntfit.mle_retvals['trimmed']
        self.nnz_params = (~self.trimmed).sum()
        k_extra = getattr(self.model, 'k_extra', 0)
        self.df_model = self.nnz_params - 1 - k_extra
        self.df_resid = float(self.model.endog.shape[0] - self.nnz_params) + k_extra

class PoissonResults(CountResults):

    def predict_prob(self, n=None, exog=None, exposure=None, offset=None, transform=True):
        if False:
            while True:
                i = 10
        '\n        Return predicted probability of each count level for each observation\n\n        Parameters\n        ----------\n        n : array_like or int\n            The counts for which you want the probabilities. If n is None\n            then the probabilities for each count from 0 to max(y) are\n            given.\n\n        Returns\n        -------\n        ndarray\n            A nobs x n array where len(`n`) columns are indexed by the count\n            n. If n is None, then column 0 is the probability that each\n            observation is 0, column 1 is the probability that each\n            observation is 1, etc.\n        '
        if n is not None:
            counts = np.atleast_2d(n)
        else:
            counts = np.atleast_2d(np.arange(0, np.max(self.model.endog) + 1))
        mu = self.predict(exog=exog, exposure=exposure, offset=offset, transform=transform, which='mean')[:, None]
        return stats.poisson.pmf(counts, mu)

    @property
    def resid_pearson(self):
        if False:
            i = 10
            return i + 15
        '\n        Pearson residuals\n\n        Notes\n        -----\n        Pearson residuals are defined to be\n\n        .. math:: r_j = \\frac{(y - M_jp_j)}{\\sqrt{M_jp_j(1-p_j)}}\n\n        where :math:`p_j=cdf(X\\beta)` and :math:`M_j` is the total number of\n        observations sharing the covariate pattern :math:`j`.\n\n        For now :math:`M_j` is always set to 1.\n        '
        p = self.predict()
        return (self.model.endog - p) / np.sqrt(p)

    def get_influence(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get an instance of MLEInfluence with influence and outlier measures\n\n        Returns\n        -------\n        infl : MLEInfluence instance\n            The instance has methods to calculate the main influence and\n            outlier measures as attributes.\n\n        See Also\n        --------\n        statsmodels.stats.outliers_influence.MLEInfluence\n        '
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

    def get_diagnostic(self, y_max=None):
        if False:
            return 10
        '\n        Get instance of class with specification and diagnostic methods\n\n        experimental, API of Diagnostic classes will change\n\n        Returns\n        -------\n        PoissonDiagnostic instance\n            The instance has methods to perform specification and diagnostic\n            tesst and plots\n\n        See Also\n        --------\n        statsmodels.statsmodels.discrete.diagnostic.PoissonDiagnostic\n        '
        from statsmodels.discrete.diagnostic import PoissonDiagnostic
        return PoissonDiagnostic(self, y_max=y_max)

class L1PoissonResults(L1CountResults, PoissonResults):
    pass

class L1NegativeBinomialResults(L1CountResults, NegativeBinomialResults):
    pass

class L1GeneralizedPoissonResults(L1CountResults, GeneralizedPoissonResults):
    pass

class OrderedResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for ordered discrete data.', 'extra_attr': ''}
    pass

class BinaryResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for binary data', 'extra_attr': ''}

    def pred_table(self, threshold=0.5):
        if False:
            while True:
                i = 10
        '\n        Prediction table\n\n        Parameters\n        ----------\n        threshold : scalar\n            Number between 0 and 1. Threshold above which a prediction is\n            considered 1 and below which a prediction is considered 0.\n\n        Notes\n        -----\n        pred_table[i,j] refers to the number of times "i" was observed and\n        the model predicted "j". Correct predictions are along the diagonal.\n        '
        model = self.model
        actual = model.endog
        pred = np.array(self.predict() > threshold, dtype=float)
        bins = np.array([0, 0.5, 1])
        return np.histogram2d(actual, pred, bins=bins)[0]

    @Appender(DiscreteResults.summary.__doc__)
    def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
        if False:
            while True:
                i = 10
        smry = super(BinaryResults, self).summary(yname, xname, title, alpha, yname_list)
        fittedvalues = self.model.cdf(self.fittedvalues)
        absprederror = np.abs(self.model.endog - fittedvalues)
        predclose_sum = (absprederror < 0.0001).sum()
        predclose_frac = predclose_sum / len(fittedvalues)
        etext = []
        if predclose_sum == len(fittedvalues):
            wstr = 'Complete Separation: The results show that there is'
            wstr += 'complete separation or perfect prediction.\n'
            wstr += 'In this case the Maximum Likelihood Estimator does '
            wstr += 'not exist and the parameters\n'
            wstr += 'are not identified.'
            etext.append(wstr)
        elif predclose_frac > 0.1:
            wstr = 'Possibly complete quasi-separation: A fraction '
            wstr += '%4.2f of observations can be\n' % predclose_frac
            wstr += 'perfectly predicted. This might indicate that there '
            wstr += 'is complete\nquasi-separation. In this case some '
            wstr += 'parameters will not be identified.'
            etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry

    @cache_readonly
    def resid_dev(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deviance residuals\n\n        Notes\n        -----\n        Deviance residuals are defined\n\n        .. math:: d_j = \\pm\\left(2\\left[Y_j\\ln\\left(\\frac{Y_j}{M_jp_j}\\right) + (M_j - Y_j\\ln\\left(\\frac{M_j-Y_j}{M_j(1-p_j)} \\right) \\right] \\right)^{1/2}\n\n        where\n\n        :math:`p_j = cdf(X\\beta)` and :math:`M_j` is the total number of\n        observations sharing the covariate pattern :math:`j`.\n\n        For now :math:`M_j` is always set to 1.\n        '
        endog = self.model.endog
        M = 1
        p = self.predict()
        res = -(1 - endog) * np.sqrt(2 * M * np.abs(np.log(1 - p))) + endog * np.sqrt(2 * M * np.abs(np.log(p)))
        return res

    @cache_readonly
    def resid_pearson(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pearson residuals\n\n        Notes\n        -----\n        Pearson residuals are defined to be\n\n        .. math:: r_j = \\frac{(y - M_jp_j)}{\\sqrt{M_jp_j(1-p_j)}}\n\n        where :math:`p_j=cdf(X\\beta)` and :math:`M_j` is the total number of\n        observations sharing the covariate pattern :math:`j`.\n\n        For now :math:`M_j` is always set to 1.\n        '
        endog = self.model.endog
        M = 1
        p = self.predict()
        return (endog - M * p) / np.sqrt(M * p * (1 - p))

    @cache_readonly
    def resid_response(self):
        if False:
            return 10
        '\n        The response residuals\n\n        Notes\n        -----\n        Response residuals are defined to be\n\n        .. math:: y - p\n\n        where :math:`p=cdf(X\\beta)`.\n        '
        return self.model.endog - self.predict()

class LogitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Logit Model', 'extra_attr': ''}

    @cache_readonly
    def resid_generalized(self):
        if False:
            print('Hello World!')
        '\n        Generalized residuals\n\n        Notes\n        -----\n        The generalized residuals for the Logit model are defined\n\n        .. math:: y - p\n\n        where :math:`p=cdf(X\\beta)`. This is the same as the `resid_response`\n        for the Logit model.\n        '
        return self.model.endog - self.predict()

    def get_influence(self):
        if False:
            i = 10
            return i + 15
        '\n        Get an instance of MLEInfluence with influence and outlier measures\n\n        Returns\n        -------\n        infl : MLEInfluence instance\n            The instance has methods to calculate the main influence and\n            outlier measures as attributes.\n\n        See Also\n        --------\n        statsmodels.stats.outliers_influence.MLEInfluence\n        '
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

class ProbitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Probit Model', 'extra_attr': ''}

    @cache_readonly
    def resid_generalized(self):
        if False:
            while True:
                i = 10
        '\n        Generalized residuals\n\n        Notes\n        -----\n        The generalized residuals for the Probit model are defined\n\n        .. math:: y\\frac{\\phi(X\\beta)}{\\Phi(X\\beta)}-(1-y)\\frac{\\phi(X\\beta)}{1-\\Phi(X\\beta)}\n        '
        model = self.model
        endog = model.endog
        XB = self.predict(which='linear')
        pdf = model.pdf(XB)
        cdf = model.cdf(XB)
        return endog * pdf / cdf - (1 - endog) * pdf / (1 - cdf)

class L1BinaryResults(BinaryResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'Results instance for binary data fit by l1 regularization', 'extra_attr': _l1_results_attr}

    def __init__(self, model, bnryfit):
        if False:
            while True:
                i = 10
        super(L1BinaryResults, self).__init__(model, bnryfit)
        self.trimmed = bnryfit.mle_retvals['trimmed']
        self.nnz_params = (~self.trimmed).sum()
        self.df_model = self.nnz_params - 1
        self.df_resid = float(self.model.endog.shape[0] - self.nnz_params)

class MultinomialResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for multinomial data', 'extra_attr': ''}

    def __init__(self, model, mlefit):
        if False:
            for i in range(10):
                print('nop')
        super(MultinomialResults, self).__init__(model, mlefit)
        self.J = model.J
        self.K = model.K

    @staticmethod
    def _maybe_convert_ynames_int(ynames):
        if False:
            print('Hello World!')
        issue_warning = False
        msg = 'endog contains values are that not int-like. Uses string representation of value. Use integer-valued endog to suppress this warning.'
        for i in ynames:
            try:
                if ynames[i] % 1 == 0:
                    ynames[i] = str(int(ynames[i]))
                else:
                    issue_warning = True
                    ynames[i] = str(ynames[i])
            except TypeError:
                ynames[i] = str(ynames[i])
        if issue_warning:
            warnings.warn(msg, SpecificationWarning)
        return ynames

    def _get_endog_name(self, yname, yname_list, all=False):
        if False:
            print('Hello World!')
        '\n        If all is False, the first variable name is dropped\n        '
        model = self.model
        if yname is None:
            yname = model.endog_names
        if yname_list is None:
            ynames = model._ynames_map
            ynames = self._maybe_convert_ynames_int(ynames)
            ynames = [ynames[key] for key in range(int(model.J))]
            ynames = ['='.join([yname, name]) for name in ynames]
            if not all:
                yname_list = ynames[1:]
            else:
                yname_list = ynames
        return (yname, yname_list)

    def pred_table(self):
        if False:
            while True:
                i = 10
        '\n        Returns the J x J prediction table.\n\n        Notes\n        -----\n        pred_table[i,j] refers to the number of times "i" was observed and\n        the model predicted "j". Correct predictions are along the diagonal.\n        '
        ju = self.model.J - 1
        bins = np.concatenate(([0], np.linspace(0.5, ju - 0.5, ju), [ju]))
        return np.histogram2d(self.model.endog, self.predict().argmax(1), bins=bins)[0]

    @cache_readonly
    def bse(self):
        if False:
            for i in range(10):
                print('nop')
        bse = np.sqrt(np.diag(self.cov_params()))
        return bse.reshape(self.params.shape, order='F')

    @cache_readonly
    def aic(self):
        if False:
            print('Hello World!')
        return -2 * (self.llf - (self.df_model + self.model.J - 1))

    @cache_readonly
    def bic(self):
        if False:
            i = 10
            return i + 15
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + self.model.J - 1)

    def conf_int(self, alpha=0.05, cols=None):
        if False:
            print('Hello World!')
        confint = super(DiscreteResults, self).conf_int(alpha=alpha, cols=cols)
        return confint.transpose(2, 0, 1)

    def get_prediction(self):
        if False:
            print('Hello World!')
        'Not implemented for Multinomial\n        '
        raise NotImplementedError

    def margeff(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Use get_margeff instead')

    @cache_readonly
    def resid_misclassified(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Residuals indicating which observations are misclassified.\n\n        Notes\n        -----\n        The residuals for the multinomial model are defined as\n\n        .. math:: argmax(y_i) \\neq argmax(p_i)\n\n        where :math:`argmax(y_i)` is the index of the category for the\n        endogenous variable and :math:`argmax(p_i)` is the index of the\n        predicted probabilities for each category. That is, the residual\n        is a binary indicator that is 0 if the category with the highest\n        predicted probability is the same as that of the observed variable\n        and 1 otherwise.\n        '
        return (self.model.wendog.argmax(1) != self.predict().argmax(1)).astype(float)

    def summary2(self, alpha=0.05, float_format='%.4f'):
        if False:
            return 10
        'Experimental function to summarize regression results\n\n        Parameters\n        ----------\n        alpha : float\n            significance level for the confidence intervals\n        float_format : str\n            print format for floats in parameters summary\n\n        Returns\n        -------\n        smry : Summary instance\n            this holds the summary tables and text, which can be printed or\n            converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary2.Summary : class to hold summary results\n        '
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_dict(summary2.summary_model(self))
        eqn = self.params.shape[1]
        confint = self.conf_int(alpha)
        for i in range(eqn):
            coefs = summary2.summary_params((self, self.params[:, i], self.bse[:, i], self.tvalues[:, i], self.pvalues[:, i], confint[i]), alpha=alpha)
            level_str = self.model.endog_names + ' = ' + str(i)
            coefs[level_str] = coefs.index
            coefs = coefs.iloc[:, [-1, 0, 1, 2, 3, 4, 5]]
            smry.add_df(coefs, index=False, header=True, float_format=float_format)
            smry.add_title(results=self)
        return smry

class L1MultinomialResults(MultinomialResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for multinomial data fit by l1 regularization', 'extra_attr': _l1_results_attr}

    def __init__(self, model, mlefit):
        if False:
            print('Hello World!')
        super(L1MultinomialResults, self).__init__(model, mlefit)
        self.trimmed = mlefit.mle_retvals['trimmed']
        self.nnz_params = (~self.trimmed).sum()
        self.df_model = self.nnz_params - (self.model.J - 1)
        self.df_resid = float(self.model.endog.shape[0] - self.nnz_params)

class OrderedResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(OrderedResultsWrapper, OrderedResults)

class CountResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(CountResultsWrapper, CountResults)

class NegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(NegativeBinomialResultsWrapper, NegativeBinomialResults)

class NegativeBinomialPResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(NegativeBinomialPResultsWrapper, NegativeBinomialPResults)

class GeneralizedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(GeneralizedPoissonResultsWrapper, GeneralizedPoissonResults)

class PoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(PoissonResultsWrapper, PoissonResults)

class L1CountResultsWrapper(lm.RegressionResultsWrapper):
    pass

class L1PoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1PoissonResultsWrapper, L1PoissonResults)

class L1NegativeBinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1NegativeBinomialResultsWrapper, L1NegativeBinomialResults)

class L1GeneralizedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1GeneralizedPoissonResultsWrapper, L1GeneralizedPoissonResults)

class BinaryResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {'resid_dev': 'rows', 'resid_generalized': 'rows', 'resid_pearson': 'rows', 'resid_response': 'rows'}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs, _attrs)
wrap.populate_wrapper(BinaryResultsWrapper, BinaryResults)

class L1BinaryResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1BinaryResultsWrapper, L1BinaryResults)

class MultinomialResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {'resid_misclassified': 'rows'}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs, _attrs)
    _methods = {'conf_int': 'multivariate_confint'}
    _wrap_methods = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(MultinomialResultsWrapper, MultinomialResults)

class L1MultinomialResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1MultinomialResultsWrapper, L1MultinomialResults)