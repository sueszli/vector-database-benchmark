"""
Procedures for fitting marginal regression models to dependent data
using Generalized Estimating Equations.

References
----------
KY Liang and S Zeger. "Longitudinal data analysis using
generalized linear models". Biometrika (1986) 73 (1): 13-22.

S Zeger and KY Liang. "Longitudinal Data Analysis for Discrete and
Continuous Outcomes". Biometrics Vol. 42, No. 1 (Mar., 1986),
pp. 121-130

A Rotnitzky and NP Jewell (1990). "Hypothesis testing of regression
parameters in semiparametric generalized linear models for cluster
correlated data", Biometrika, 77, 485-497.

Xu Guo and Wei Pan (2002). "Small sample performance of the score
test in GEE".
http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf

LA Mancl LA, TA DeRouen (2001). A covariance estimator for GEE with
improved small-sample properties.  Biometrics. 2001 Mar;57(1):126-34.
"""
from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import ConvergenceWarning, DomainWarning, IterationLimitWarning, ValueWarning
import warnings
from statsmodels.graphics._regressionplots_doc import _plot_added_variable_doc, _plot_partial_residuals_doc, _plot_ceres_residuals_doc
from statsmodels.discrete.discrete_margins import _get_margeff_exog, _check_margeff_args, _effects_at, margeff_cov_with_se, _check_at_is_all, _transform_names, _check_discrete_args, _get_dummy_index, _get_count_index

class ParameterConstraint:
    """
    A class for managing linear equality constraints for a parameter
    vector.
    """

    def __init__(self, lhs, rhs, exog):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        lhs : ndarray\n           A q x p matrix which is the left hand side of the\n           constraint lhs * param = rhs.  The number of constraints is\n           q >= 1 and p is the dimension of the parameter vector.\n        rhs : ndarray\n          A 1-dimensional vector of length q which is the right hand\n          side of the constraint equation.\n        exog : ndarray\n          The n x p exognenous data for the full model.\n        '
        rhs = np.atleast_1d(rhs.squeeze())
        if rhs.ndim > 1:
            raise ValueError('The right hand side of the constraint must be a vector.')
        if len(rhs) != lhs.shape[0]:
            raise ValueError('The number of rows of the left hand side constraint matrix L must equal the length of the right hand side constraint vector R.')
        self.lhs = lhs
        self.rhs = rhs
        (lhs_u, lhs_s, lhs_vt) = np.linalg.svd(lhs.T, full_matrices=1)
        self.lhs0 = lhs_u[:, len(lhs_s):]
        self.lhs1 = lhs_u[:, 0:len(lhs_s)]
        self.lhsf = np.hstack((self.lhs0, self.lhs1))
        self.param0 = np.dot(self.lhs1, np.dot(lhs_vt, self.rhs) / lhs_s)
        self._offset_increment = np.dot(exog, self.param0)
        self.orig_exog = exog
        self.exog_fulltrans = np.dot(exog, self.lhsf)

    def offset_increment(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a vector that should be added to the offset vector to\n        accommodate the constraint.\n\n        Parameters\n        ----------\n        exog : array_like\n           The exogeneous data for the model.\n        '
        return self._offset_increment

    def reduced_exog(self):
        if False:
            print('Hello World!')
        '\n        Returns a linearly transformed exog matrix whose columns span\n        the constrained model space.\n\n        Parameters\n        ----------\n        exog : array_like\n           The exogeneous data for the model.\n        '
        return self.exog_fulltrans[:, 0:self.lhs0.shape[1]]

    def restore_exog(self):
        if False:
            while True:
                i = 10
        '\n        Returns the full exog matrix before it was reduced to\n        satisfy the constraint.\n        '
        return self.orig_exog

    def unpack_param(self, params):
        if False:
            print('Hello World!')
        '\n        Converts the parameter vector `params` from reduced to full\n        coordinates.\n        '
        return self.param0 + np.dot(self.lhs0, params)

    def unpack_cov(self, bcov):
        if False:
            while True:
                i = 10
        '\n        Converts the covariance matrix `bcov` from reduced to full\n        coordinates.\n        '
        return np.dot(self.lhs0, np.dot(bcov, self.lhs0.T))
_gee_init_doc = '\n    Marginal regression model fit using Generalized Estimating Equations.\n\n    GEE can be used to fit Generalized Linear Models (GLMs) when the\n    data have a grouped structure, and the observations are possibly\n    correlated within groups but not between groups.\n\n    Parameters\n    ----------\n    endog : array_like\n        1d array of endogenous values (i.e. responses, outcomes,\n        dependent variables, or \'Y\' values).\n    exog : array_like\n        2d array of exogeneous values (i.e. covariates, predictors,\n        independent variables, regressors, or \'X\' values). A `nobs x\n        k` array where `nobs` is the number of observations and `k` is\n        the number of regressors. An intercept is not included by\n        default and should be added by the user. See\n        `statsmodels.tools.add_constant`.\n    groups : array_like\n        A 1d array of length `nobs` containing the group labels.\n    time : array_like\n        A 2d array of time (or other index) values, used by some\n        dependence structures to define similarity relationships among\n        observations within a cluster.\n    family : family class instance\n%(family_doc)s\n    cov_struct : CovStruct class instance\n        The default is Independence.  To specify an exchangeable\n        structure use cov_struct = Exchangeable().  See\n        statsmodels.genmod.cov_struct.CovStruct for more\n        information.\n    offset : array_like\n        An offset to be included in the fit.  If provided, must be\n        an array whose length is the number of rows in exog.\n    dep_data : array_like\n        Additional data passed to the dependence structure.\n    constraint : (ndarray, ndarray)\n        If provided, the constraint is a tuple (L, R) such that the\n        model parameters are estimated under the constraint L *\n        param = R, where L is a q x p matrix and R is a\n        q-dimensional vector.  If constraint is provided, a score\n        test is performed to compare the constrained model to the\n        unconstrained model.\n    update_dep : bool\n        If true, the dependence parameters are optimized, otherwise\n        they are held fixed at their starting values.\n    weights : array_like\n        An array of case weights to use in the analysis.\n    %(extra_params)s\n\n    See Also\n    --------\n    statsmodels.genmod.families.family\n    :ref:`families`\n    :ref:`links`\n\n    Notes\n    -----\n    Only the following combinations make sense for family and link ::\n\n                   + ident log logit probit cloglog pow opow nbinom loglog logc\n      Gaussian     |   x    x                        x\n      inv Gaussian |   x    x                        x\n      binomial     |   x    x    x     x       x     x    x           x      x\n      Poisson      |   x    x                        x\n      neg binomial |   x    x                        x          x\n      gamma        |   x    x                        x\n\n    Not all of these link functions are currently available.\n\n    Endog and exog are references so that if the data they refer\n    to are already arrays and these arrays are changed, endog and\n    exog will change.\n\n    The "robust" covariance type is the standard "sandwich estimator"\n    (e.g. Liang and Zeger (1986)).  It is the default here and in most\n    other packages.  The "naive" estimator gives smaller standard\n    errors, but is only correct if the working correlation structure\n    is correctly specified.  The "bias reduced" estimator of Mancl and\n    DeRouen (Biometrics, 2001) reduces the downward bias of the robust\n    estimator.\n\n    The robust covariance provided here follows Liang and Zeger (1986)\n    and agrees with R\'s gee implementation.  To obtain the robust\n    standard errors reported in Stata, multiply by sqrt(N / (N - g)),\n    where N is the total sample size, and g is the average group size.\n    %(notes)s\n    Examples\n    --------\n    %(example)s\n'
_gee_nointercept = '\n    The nominal and ordinal GEE models should not have an intercept\n    (either implicit or explicit).  Use "0 + " in a formula to\n    suppress the intercept.\n'
_gee_family_doc = '        The default is Gaussian.  To specify the binomial\n        distribution use `family=sm.families.Binomial()`. Each family\n        can take a link instance as an argument.  See\n        statsmodels.genmod.families.family for more information.'
_gee_ordinal_family_doc = '        The only family supported is `Binomial`.  The default `Logit`\n        link may be replaced with `probit` if desired.'
_gee_nominal_family_doc = '        The default value `None` uses a multinomial logit family\n        specifically designed for use with GEE.  Setting this\n        argument to a non-default value is not currently supported.'
_gee_fit_doc = '\n    Fits a marginal regression model using generalized estimating\n    equations (GEE).\n\n    Parameters\n    ----------\n    maxiter : int\n        The maximum number of iterations\n    ctol : float\n        The convergence criterion for stopping the Gauss-Seidel\n        iterations\n    start_params : array_like\n        A vector of starting values for the regression\n        coefficients.  If None, a default is chosen.\n    params_niter : int\n        The number of Gauss-Seidel updates of the mean structure\n        parameters that take place prior to each update of the\n        dependence structure.\n    first_dep_update : int\n        No dependence structure updates occur before this\n        iteration number.\n    cov_type : str\n        One of "robust", "naive", or "bias_reduced".\n    ddof_scale : scalar or None\n        The scale parameter is estimated as the sum of squared\n        Pearson residuals divided by `N - ddof_scale`, where N\n        is the total sample size.  If `ddof_scale` is None, the\n        number of covariates (including an intercept if present)\n        is used.\n    scaling_factor : scalar\n        The estimated covariance of the parameter estimates is\n        scaled by this value.  Default is 1, Stata uses N / (N - g),\n        where N is the total sample size and g is the average group\n        size.\n    scale : str or float, optional\n        `scale` can be None, \'X2\', or a float\n        If a float, its value is used as the scale parameter.\n        The default value is None, which uses `X2` (Pearson\'s\n        chi-square) for Gamma, Gaussian, and Inverse Gaussian.\n        The default is 1 for the Binomial and Poisson families.\n\n    Returns\n    -------\n    An instance of the GEEResults class or subclass\n\n    Notes\n    -----\n    If convergence difficulties occur, increase the values of\n    `first_dep_update` and/or `params_niter`.  Setting\n    `first_dep_update` to a greater value (e.g. ~10-20) causes the\n    algorithm to move close to the GLM solution before attempting\n    to identify the dependence structure.\n\n    For the Gaussian family, there is no benefit to setting\n    `params_niter` to a value greater than 1, since the mean\n    structure parameters converge in one step.\n'
_gee_results_doc = '\n    Attributes\n    ----------\n\n    cov_params_default : ndarray\n        default covariance of the parameter estimates. Is chosen among one\n        of the following three based on `cov_type`\n    cov_robust : ndarray\n        covariance of the parameter estimates that is robust\n    cov_naive : ndarray\n        covariance of the parameter estimates that is not robust to\n        correlation or variance misspecification\n    cov_robust_bc : ndarray\n        covariance of the parameter estimates that is robust and bias\n        reduced\n    converged : bool\n        indicator for convergence of the optimization.\n        True if the norm of the score is smaller than a threshold\n    cov_type : str\n        string indicating whether a "robust", "naive" or "bias_reduced"\n        covariance is used as default\n    fit_history : dict\n        Contains information about the iterations.\n    fittedvalues : ndarray\n        Linear predicted values for the fitted model.\n        dot(exog, params)\n    model : class instance\n        Pointer to GEE model instance that called `fit`.\n    normalized_cov_params : ndarray\n        See GEE docstring\n    params : ndarray\n        The coefficients of the fitted model.  Note that\n        interpretation of the coefficients often depends on the\n        distribution family and the data.\n    scale : float\n        The estimate of the scale / dispersion for the model fit.\n        See GEE.fit for more information.\n    score_norm : float\n        norm of the score at the end of the iterative estimation.\n    bse : ndarray\n        The standard errors of the fitted GEE parameters.\n'
_gee_example = '\n    Logistic regression with autoregressive working dependence:\n\n    >>> import statsmodels.api as sm\n    >>> family = sm.families.Binomial()\n    >>> va = sm.cov_struct.Autoregressive()\n    >>> model = sm.GEE(endog, exog, group, family=family, cov_struct=va)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Use formulas to fit a Poisson GLM with independent working\n    dependence:\n\n    >>> import statsmodels.api as sm\n    >>> fam = sm.families.Poisson()\n    >>> ind = sm.cov_struct.Independence()\n    >>> model = sm.GEE.from_formula("y ~ age + trt + base", "subject",\n                                 data, cov_struct=ind, family=fam)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Equivalent, using the formula API:\n\n    >>> import statsmodels.api as sm\n    >>> import statsmodels.formula.api as smf\n    >>> fam = sm.families.Poisson()\n    >>> ind = sm.cov_struct.Independence()\n    >>> model = smf.gee("y ~ age + trt + base", "subject",\n                    data, cov_struct=ind, family=fam)\n    >>> result = model.fit()\n    >>> print(result.summary())\n'
_gee_ordinal_example = '\n    Fit an ordinal regression model using GEE, with "global\n    odds ratio" dependence:\n\n    >>> import statsmodels.api as sm\n    >>> gor = sm.cov_struct.GlobalOddsRatio("ordinal")\n    >>> model = sm.OrdinalGEE(endog, exog, groups, cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Using formulas:\n\n    >>> import statsmodels.formula.api as smf\n    >>> model = smf.ordinal_gee("y ~ 0 + x1 + x2", groups, data,\n                                    cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n'
_gee_nominal_example = '\n    Fit a nominal regression model using GEE:\n\n    >>> import statsmodels.api as sm\n    >>> import statsmodels.formula.api as smf\n    >>> gor = sm.cov_struct.GlobalOddsRatio("nominal")\n    >>> model = sm.NominalGEE(endog, exog, groups, cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Using formulas:\n\n    >>> import statsmodels.api as sm\n    >>> model = sm.NominalGEE.from_formula("y ~ 0 + x1 + x2", groups,\n                     data, cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Using the formula API:\n\n    >>> import statsmodels.formula.api as smf\n    >>> model = smf.nominal_gee("y ~ 0 + x1 + x2", groups, data,\n                                cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n'

def _check_args(endog, exog, groups, time, offset, exposure):
    if False:
        i = 10
        return i + 15
    if endog.size != exog.shape[0]:
        raise ValueError("Leading dimension of 'exog' should match length of 'endog'")
    if groups.size != endog.size:
        raise ValueError("'groups' and 'endog' should have the same size")
    if time is not None and time.size != endog.size:
        raise ValueError("'time' and 'endog' should have the same size")
    if offset is not None and offset.size != endog.size:
        raise ValueError("'offset and 'endog' should have the same size")
    if exposure is not None and exposure.size != endog.size:
        raise ValueError("'exposure' and 'endog' should have the same size")

class GEE(GLM):
    __doc__ = '    Marginal Regression Model using Generalized Estimating Equations.\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_family_doc, 'example': _gee_example, 'notes': ''}
    cached_means = None

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, exposure=None, dep_data=None, constraint=None, update_dep=True, weights=None, **kwargs):
        if False:
            while True:
                i = 10
        if type(self) is GEE:
            self._check_kwargs(kwargs)
        if family is not None:
            if not isinstance(family.link, tuple(family.safe_links)):
                msg = 'The {0} link function does not respect the domain of the {1} family.'
                warnings.warn(msg.format(family.link.__class__.__name__, family.__class__.__name__), DomainWarning)
        groups = np.asarray(groups)
        if 'missing_idx' in kwargs and kwargs['missing_idx'] is not None:
            ii = ~kwargs['missing_idx']
            groups = groups[ii]
            if time is not None:
                time = time[ii]
            if offset is not None:
                offset = offset[ii]
            if exposure is not None:
                exposure = exposure[ii]
            del kwargs['missing_idx']
        self.missing = missing
        self.dep_data = dep_data
        self.constraint = constraint
        self.update_dep = update_dep
        self._fit_history = defaultdict(list)
        super(GEE, self).__init__(endog, exog, groups=groups, time=time, offset=offset, exposure=exposure, weights=weights, dep_data=dep_data, missing=missing, family=family, **kwargs)
        _check_args(self.endog, self.exog, self.groups, self.time, getattr(self, 'offset', None), getattr(self, 'exposure', None))
        self._init_keys.extend(['update_dep', 'constraint', 'family', 'cov_struct'])
        try:
            self._init_keys.remove('freq_weights')
            self._init_keys.remove('var_weights')
        except ValueError:
            pass
        if family is None:
            family = families.Gaussian()
        elif not issubclass(family.__class__, families.Family):
            raise ValueError('GEE: `family` must be a genmod family instance')
        self.family = family
        if cov_struct is None:
            cov_struct = cov_structs.Independence()
        elif not issubclass(cov_struct.__class__, cov_structs.CovStruct):
            raise ValueError('GEE: `cov_struct` must be a genmod cov_struct instance')
        self.cov_struct = cov_struct
        self.constraint = None
        if constraint is not None:
            if len(constraint) != 2:
                raise ValueError('GEE: `constraint` must be a 2-tuple.')
            if constraint[0].shape[1] != self.exog.shape[1]:
                raise ValueError('GEE: the left hand side of the constraint must have the same number of columns as the exog matrix.')
            self.constraint = ParameterConstraint(constraint[0], constraint[1], self.exog)
            if self._offset_exposure is not None:
                self._offset_exposure += self.constraint.offset_increment()
            else:
                self._offset_exposure = self.constraint.offset_increment().copy()
            self.exog = self.constraint.reduced_exog()
        (group_labels, ix) = np.unique(self.groups, return_inverse=True)
        se = pd.Series(index=np.arange(len(ix)), dtype='int')
        gb = se.groupby(ix).groups
        dk = [(lb, np.asarray(gb[k])) for (k, lb) in enumerate(group_labels)]
        self.group_indices = dict(dk)
        self.group_labels = group_labels
        self.endog_li = self.cluster_list(self.endog)
        self.exog_li = self.cluster_list(self.exog)
        if self.weights is not None:
            self.weights_li = self.cluster_list(self.weights)
        self.num_group = len(self.endog_li)
        if self.time is not None:
            if self.time.ndim == 1:
                self.time = self.time[:, None]
            self.time_li = self.cluster_list(self.time)
        else:
            self.time_li = [np.arange(len(y), dtype=np.float64)[:, None] for y in self.endog_li]
            self.time = np.concatenate(self.time_li)
        if self._offset_exposure is None or (np.isscalar(self._offset_exposure) and self._offset_exposure == 0.0):
            self.offset_li = None
        else:
            self.offset_li = self.cluster_list(self._offset_exposure)
        if constraint is not None:
            self.constraint.exog_fulltrans_li = self.cluster_list(self.constraint.exog_fulltrans)
        self.family = family
        self.cov_struct.initialize(self)
        group_ns = [len(y) for y in self.endog_li]
        self.nobs = sum(group_ns)
        self.df_model = self.exog.shape[1] - 1
        self.df_resid = self.nobs - self.exog.shape[1]
        maxgroup = max([len(x) for x in self.endog_li])
        if maxgroup == 1:
            self.update_dep = False

    @classmethod
    def from_formula(cls, formula, groups, data, subset=None, time=None, offset=None, exposure=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a GEE model instance from a formula and dataframe.\n\n        Parameters\n        ----------\n        formula : str or generic Formula object\n            The formula specifying the model\n        groups : array_like or string\n            Array of grouping labels.  If a string, this is the name\n            of a variable in `data` that contains the grouping labels.\n        data : array_like\n            The data for the model.\n        subset : array_like\n            An array-like object of booleans, integers, or index\n            values that indicate the subset of the data to used when\n            fitting the model.\n        time : array_like or string\n            The time values, used for dependence structures involving\n            distances between observations.  If a string, this is the\n            name of a variable in `data` that contains the time\n            values.\n        offset : array_like or string\n            The offset values, added to the linear predictor.  If a\n            string, this is the name of a variable in `data` that\n            contains the offset values.\n        exposure : array_like or string\n            The exposure values, only used if the link function is the\n            logarithm function, in which case the log of `exposure`\n            is added to the offset (if any).  If a string, this is the\n            name of a variable in `data` that contains the offset\n            values.\n        %(missing_param_doc)s\n        args : extra arguments\n            These are passed to the model\n        kwargs : extra keyword arguments\n            These are passed to the model with two exceptions. `dep_data`\n            is processed as described below.  The ``eval_env`` keyword is\n            passed to patsy. It can be either a\n            :class:`patsy:patsy.EvalEnvironment` object or an integer\n            indicating the depth of the namespace to use. For example, the\n            default ``eval_env=0`` uses the calling namespace.\n            If you wish to use a "clean" environment set ``eval_env=-1``.\n\n        Optional arguments\n        ------------------\n        dep_data : str or array_like\n            Data used for estimating the dependence structure.  See\n            specific dependence structure classes (e.g. Nested) for\n            details.  If `dep_data` is a string, it is interpreted as\n            a formula that is applied to `data`. If it is an array, it\n            must be an array of strings corresponding to column names in\n            `data`.  Otherwise it must be an array-like with the same\n            number of rows as data.\n\n        Returns\n        -------\n        model : GEE model instance\n\n        Notes\n        -----\n        `data` must define __getitem__ with the keys in the formula\n        terms args and kwargs are passed on to the model\n        instantiation. E.g., a numpy structured or rec array, a\n        dictionary, or a pandas DataFrame.\n        ' % {'missing_param_doc': base._missing_param_doc}
        groups_name = 'Groups'
        if isinstance(groups, str):
            groups_name = groups
            groups = data[groups]
        if isinstance(time, str):
            time = data[time]
        if isinstance(offset, str):
            offset = data[offset]
        if isinstance(exposure, str):
            exposure = data[exposure]
        dep_data = kwargs.get('dep_data')
        dep_data_names = None
        if dep_data is not None:
            if isinstance(dep_data, str):
                dep_data = patsy.dmatrix(dep_data, data, return_type='dataframe')
                dep_data_names = dep_data.columns.tolist()
            else:
                dep_data_names = list(dep_data)
                dep_data = data[dep_data]
            kwargs['dep_data'] = np.asarray(dep_data)
        family = None
        if 'family' in kwargs:
            family = kwargs['family']
            del kwargs['family']
        model = super(GEE, cls).from_formula(formula, *args, data=data, subset=subset, groups=groups, time=time, offset=offset, exposure=exposure, family=family, **kwargs)
        if dep_data_names is not None:
            model._dep_data_names = dep_data_names
        model._groups_name = groups_name
        return model

    def cluster_list(self, array):
        if False:
            return 10
        '\n        Returns `array` split into subarrays corresponding to the\n        cluster structure.\n        '
        if array.ndim == 1:
            return [np.array(array[self.group_indices[k]]) for k in self.group_labels]
        else:
            return [np.array(array[self.group_indices[k], :]) for k in self.group_labels]

    def compare_score_test(self, submodel):
        if False:
            print('Hello World!')
        '\n        Perform a score test for the given submodel against this model.\n\n        Parameters\n        ----------\n        submodel : GEEResults instance\n            A fitted GEE model that is a submodel of this model.\n\n        Returns\n        -------\n        A dictionary with keys "statistic", "p-value", and "df",\n        containing the score test statistic, its chi^2 p-value,\n        and the degrees of freedom used to compute the p-value.\n\n        Notes\n        -----\n        The score test can be performed without calling \'fit\' on the\n        larger model.  The provided submodel must be obtained from a\n        fitted GEE.\n\n        This method performs the same score test as can be obtained by\n        fitting the GEE with a linear constraint and calling `score_test`\n        on the results.\n\n        References\n        ----------\n        Xu Guo and Wei Pan (2002). "Small sample performance of the score\n        test in GEE".\n        http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf\n        '
        self.scaletype = submodel.model.scaletype
        submod = submodel.model
        if self.exog.shape[0] != submod.exog.shape[0]:
            msg = 'Model and submodel have different numbers of cases.'
            raise ValueError(msg)
        if self.exog.shape[1] == submod.exog.shape[1]:
            msg = 'Model and submodel have the same number of variables'
            warnings.warn(msg)
        if not isinstance(self.family, type(submod.family)):
            msg = 'Model and submodel have different GLM families.'
            warnings.warn(msg)
        if not isinstance(self.cov_struct, type(submod.cov_struct)):
            warnings.warn('Model and submodel have different GEE covariance structures.')
        if not np.equal(self.weights, submod.weights).all():
            msg = 'Model and submodel should have the same weights.'
            warnings.warn(msg)
        (qm, qc) = _score_test_submodel(self, submodel.model)
        if qm is None:
            msg = 'The provided model is not a submodel.'
            raise ValueError(msg)
        params_ex = np.dot(qm, submodel.params)
        cov_struct_save = self.cov_struct
        import copy
        cached_means_save = copy.deepcopy(self.cached_means)
        self.cov_struct = submodel.cov_struct
        self.update_cached_means(params_ex)
        (_, score) = self._update_mean_params()
        if score is None:
            msg = 'Singular matrix encountered in GEE score test'
            warnings.warn(msg, ConvergenceWarning)
            return None
        if not hasattr(self, 'ddof_scale'):
            self.ddof_scale = self.exog.shape[1]
        if not hasattr(self, 'scaling_factor'):
            self.scaling_factor = 1
        (_, ncov1, cmat) = self._covmat()
        score2 = np.dot(qc.T, score)
        try:
            amat = np.linalg.inv(ncov1)
        except np.linalg.LinAlgError:
            amat = np.linalg.pinv(ncov1)
        bmat_11 = np.dot(qm.T, np.dot(cmat, qm))
        bmat_22 = np.dot(qc.T, np.dot(cmat, qc))
        bmat_12 = np.dot(qm.T, np.dot(cmat, qc))
        amat_11 = np.dot(qm.T, np.dot(amat, qm))
        amat_12 = np.dot(qm.T, np.dot(amat, qc))
        try:
            ab = np.linalg.solve(amat_11, bmat_12)
        except np.linalg.LinAlgError:
            ab = np.dot(np.linalg.pinv(amat_11), bmat_12)
        score_cov = bmat_22 - np.dot(amat_12.T, ab)
        try:
            aa = np.linalg.solve(amat_11, amat_12)
        except np.linalg.LinAlgError:
            aa = np.dot(np.linalg.pinv(amat_11), amat_12)
        score_cov -= np.dot(bmat_12.T, aa)
        try:
            ab = np.linalg.solve(amat_11, bmat_11)
        except np.linalg.LinAlgError:
            ab = np.dot(np.linalg.pinv(amat_11), bmat_11)
        try:
            aa = np.linalg.solve(amat_11, amat_12)
        except np.linalg.LinAlgError:
            aa = np.dot(np.linalg.pinv(amat_11), amat_12)
        score_cov += np.dot(amat_12.T, np.dot(ab, aa))
        self.cov_struct = cov_struct_save
        self.cached_means = cached_means_save
        from scipy.stats.distributions import chi2
        try:
            sc2 = np.linalg.solve(score_cov, score2)
        except np.linalg.LinAlgError:
            sc2 = np.dot(np.linalg.pinv(score_cov), score2)
        score_statistic = np.dot(score2, sc2)
        score_df = len(score2)
        score_pvalue = 1 - chi2.cdf(score_statistic, score_df)
        return {'statistic': score_statistic, 'df': score_df, 'p-value': score_pvalue}

    def estimate_scale(self):
        if False:
            i = 10
            return i + 15
        '\n        Estimate the dispersion/scale.\n        '
        if self.scaletype is None:
            if isinstance(self.family, (families.Binomial, families.Poisson, families.NegativeBinomial, _Multinomial)):
                return 1.0
        elif isinstance(self.scaletype, float):
            return np.array(self.scaletype)
        endog = self.endog_li
        cached_means = self.cached_means
        nobs = self.nobs
        varfunc = self.family.variance
        scale = 0.0
        fsum = 0.0
        for i in range(self.num_group):
            if len(endog[i]) == 0:
                continue
            (expval, _) = cached_means[i]
            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / sdev
            if self.weights is not None:
                f = self.weights_li[i]
                scale += np.sum(f * resid ** 2)
                fsum += f.sum()
            else:
                scale += np.sum(resid ** 2)
                fsum += len(resid)
        scale /= fsum * (nobs - self.ddof_scale) / float(nobs)
        return scale

    def mean_deriv(self, exog, lin_pred):
        if False:
            for i in range(10):
                print('nop')
        '\n        Derivative of the expected endog with respect to the parameters.\n\n        Parameters\n        ----------\n        exog : array_like\n           The exogeneous data at which the derivative is computed.\n        lin_pred : array_like\n           The values of the linear predictor.\n\n        Returns\n        -------\n        The value of the derivative of the expected endog with respect\n        to the parameter vector.\n\n        Notes\n        -----\n        If there is an offset or exposure, it should be added to\n        `lin_pred` prior to calling this function.\n        '
        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = exog * idl[:, None]
        return dmat

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        if False:
            return 10
        '\n        Derivative of the expected endog with respect to exog.\n\n        Parameters\n        ----------\n        exog : array_like\n            Values of the independent variables at which the derivative\n            is calculated.\n        params : array_like\n            Parameter values at which the derivative is calculated.\n        offset_exposure : array_like, optional\n            Combined offset and exposure.\n\n        Returns\n        -------\n        The derivative of the expected endog with respect to exog.\n        '
        lin_pred = np.dot(exog, params)
        if offset_exposure is not None:
            lin_pred += offset_exposure
        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = np.outer(idl, params)
        return dmat

    def _update_mean_params(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        update : array_like\n            The update vector such that params + update is the next\n            iterate when solving the score equations.\n        score : array_like\n            The current value of the score equations, not\n            incorporating the scale parameter.  If desired,\n            multiply this vector by the scale parameter to\n            incorporate the scale.\n        '
        endog = self.endog_li
        exog = self.exog_li
        weights = getattr(self, 'weights_li', None)
        cached_means = self.cached_means
        varfunc = self.family.variance
        (bmat, score) = (0, 0)
        for i in range(self.num_group):
            (expval, lpr) = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))
            if weights is not None:
                w = weights[i]
                wresid = resid * w
                wdmat = dmat * w[:, None]
            else:
                wresid = resid
                wdmat = dmat
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (wdmat, wresid))
            if rslt is None:
                return (None, None)
            (vinv_d, vinv_resid) = tuple(rslt)
            bmat += np.dot(dmat.T, vinv_d)
            score += np.dot(dmat.T, vinv_resid)
        try:
            update = np.linalg.solve(bmat, score)
        except np.linalg.LinAlgError:
            update = np.dot(np.linalg.pinv(bmat), score)
        self._fit_history['cov_adjust'].append(self.cov_struct.cov_adjust)
        return (update, score)

    def update_cached_means(self, mean_params):
        if False:
            print('Hello World!')
        '\n        cached_means should always contain the most recent calculation\n        of the group-wise mean vectors.  This function should be\n        called every time the regression parameters are changed, to\n        keep the cached means up to date.\n        '
        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li
        linkinv = self.family.link.inverse
        self.cached_means = []
        for i in range(self.num_group):
            if len(endog[i]) == 0:
                continue
            lpr = np.dot(exog[i], mean_params)
            if offset is not None:
                lpr += offset[i]
            expval = linkinv(lpr)
            self.cached_means.append((expval, lpr))

    def _covmat(self):
        if False:
            while True:
                i = 10
        '\n        Returns the sampling covariance matrix of the regression\n        parameters and related quantities.\n\n        Returns\n        -------\n        cov_robust : array_like\n           The robust, or sandwich estimate of the covariance, which\n           is meaningful even if the working covariance structure is\n           incorrectly specified.\n        cov_naive : array_like\n           The model-based estimate of the covariance, which is\n           meaningful if the covariance structure is correctly\n           specified.\n        cmat : array_like\n           The center matrix of the sandwich expression, used in\n           obtaining score test results.\n        '
        endog = self.endog_li
        exog = self.exog_li
        weights = getattr(self, 'weights_li', None)
        varfunc = self.family.variance
        cached_means = self.cached_means
        (bmat, cmat) = (0, 0)
        for i in range(self.num_group):
            (expval, lpr) = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))
            if weights is not None:
                w = weights[i]
                wresid = resid * w
                wdmat = dmat * w[:, None]
            else:
                wresid = resid
                wdmat = dmat
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (wdmat, wresid))
            if rslt is None:
                return (None, None, None, None)
            (vinv_d, vinv_resid) = tuple(rslt)
            bmat += np.dot(dmat.T, vinv_d)
            dvinv_resid = np.dot(dmat.T, vinv_resid)
            cmat += np.outer(dvinv_resid, dvinv_resid)
        scale = self.estimate_scale()
        try:
            bmati = np.linalg.inv(bmat)
        except np.linalg.LinAlgError:
            bmati = np.linalg.pinv(bmat)
        cov_naive = bmati * scale
        cov_robust = np.dot(bmati, np.dot(cmat, bmati))
        cov_naive *= self.scaling_factor
        cov_robust *= self.scaling_factor
        return (cov_robust, cov_naive, cmat)

    def _bc_covmat(self, cov_naive):
        if False:
            for i in range(10):
                print('nop')
        cov_naive = cov_naive / self.scaling_factor
        endog = self.endog_li
        exog = self.exog_li
        varfunc = self.family.variance
        cached_means = self.cached_means
        scale = self.estimate_scale()
        bcm = 0
        for i in range(self.num_group):
            (expval, lpr) = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (dmat,))
            if rslt is None:
                return None
            vinv_d = rslt[0]
            vinv_d /= scale
            hmat = np.dot(vinv_d, cov_naive)
            hmat = np.dot(hmat, dmat.T).T
            f = self.weights_li[i] if self.weights is not None else 1.0
            aresid = np.linalg.solve(np.eye(len(resid)) - hmat, resid)
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (aresid,))
            if rslt is None:
                return None
            srt = rslt[0]
            srt = f * np.dot(dmat.T, srt) / scale
            bcm += np.outer(srt, srt)
        cov_robust_bc = np.dot(cov_naive, np.dot(bcm, cov_naive))
        cov_robust_bc *= self.scaling_factor
        return cov_robust_bc

    def _starting_params(self):
        if False:
            for i in range(10):
                print('nop')
        if np.isscalar(self._offset_exposure):
            offset = None
        else:
            offset = self._offset_exposure
        model = GLM(self.endog, self.exog, family=self.family, offset=offset, freq_weights=self.weights)
        result = model.fit()
        return result.params

    @Appender(_gee_fit_doc)
    def fit(self, maxiter=60, ctol=1e-06, start_params=None, params_niter=1, first_dep_update=0, cov_type='robust', ddof_scale=None, scaling_factor=1.0, scale=None):
        if False:
            i = 10
            return i + 15
        self.scaletype = scale
        if ddof_scale is None:
            self.ddof_scale = self.exog.shape[1]
        else:
            if not ddof_scale >= 0:
                raise ValueError('ddof_scale must be a non-negative number or None')
            self.ddof_scale = ddof_scale
        self.scaling_factor = scaling_factor
        self._fit_history = defaultdict(list)
        if self.weights is not None and cov_type == 'naive':
            raise ValueError('when using weights, cov_type may not be naive')
        if start_params is None:
            mean_params = self._starting_params()
        else:
            start_params = np.asarray(start_params)
            mean_params = start_params.copy()
        self.update_cached_means(mean_params)
        del_params = -1.0
        num_assoc_updates = 0
        for itr in range(maxiter):
            (update, score) = self._update_mean_params()
            if update is None:
                warnings.warn('Singular matrix encountered in GEE update', ConvergenceWarning)
                break
            mean_params += update
            self.update_cached_means(mean_params)
            del_params = np.sqrt(np.sum(score ** 2))
            self._fit_history['params'].append(mean_params.copy())
            self._fit_history['score'].append(score)
            self._fit_history['dep_params'].append(self.cov_struct.dep_params)
            if del_params < ctol and (num_assoc_updates > 0 or self.update_dep is False):
                break
            if self.update_dep and itr % params_niter == 0 and (itr >= first_dep_update):
                self._update_assoc(mean_params)
                num_assoc_updates += 1
        if del_params >= ctol:
            warnings.warn('Iteration limit reached prior to convergence', IterationLimitWarning)
        if mean_params is None:
            warnings.warn('Unable to estimate GEE parameters.', ConvergenceWarning)
            return None
        (bcov, ncov, _) = self._covmat()
        if bcov is None:
            warnings.warn('Estimated covariance structure for GEE estimates is singular', ConvergenceWarning)
            return None
        bc_cov = None
        if cov_type == 'bias_reduced':
            bc_cov = self._bc_covmat(ncov)
        if self.constraint is not None:
            x = mean_params.copy()
            (mean_params, bcov) = self._handle_constraint(mean_params, bcov)
            if mean_params is None:
                warnings.warn('Unable to estimate constrained GEE parameters.', ConvergenceWarning)
                return None
            (y, ncov) = self._handle_constraint(x, ncov)
            if y is None:
                warnings.warn('Unable to estimate constrained GEE parameters.', ConvergenceWarning)
                return None
            if bc_cov is not None:
                (y, bc_cov) = self._handle_constraint(x, bc_cov)
                if x is None:
                    warnings.warn('Unable to estimate constrained GEE parameters.', ConvergenceWarning)
                    return None
        scale = self.estimate_scale()
        res_kwds = dict(cov_type=cov_type, cov_robust=bcov, cov_naive=ncov, cov_robust_bc=bc_cov)
        results = GEEResults(self, mean_params, bcov / scale, scale, cov_type=cov_type, use_t=False, attr_kwds=res_kwds)
        results.fit_history = self._fit_history
        self.fit_history = defaultdict(list)
        results.score_norm = del_params
        results.converged = del_params < ctol
        results.cov_struct = self.cov_struct
        results.params_niter = params_niter
        results.first_dep_update = first_dep_update
        results.ctol = ctol
        results.maxiter = maxiter
        results._props = ['cov_type', 'use_t', 'cov_params_default', 'cov_robust', 'cov_naive', 'cov_robust_bc', 'fit_history', 'score_norm', 'converged', 'cov_struct', 'params_niter', 'first_dep_update', 'ctol', 'maxiter']
        return GEEResultsWrapper(results)

    def _update_regularized(self, params, pen_wt, scad_param, eps):
        if False:
            for i in range(10):
                print('nop')
        (sn, hm) = (0, 0)
        for i in range(self.num_group):
            (expval, _) = self.cached_means[i]
            resid = self.endog_li[i] - expval
            sdev = np.sqrt(self.family.variance(expval))
            ex = self.exog_li[i] * sdev[:, None] ** 2
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (resid, ex))
            sn0 = rslt[0]
            sn += np.dot(ex.T, sn0)
            hm0 = rslt[1]
            hm += np.dot(ex.T, hm0)
        ap = np.abs(params)
        clipped = np.clip(scad_param * pen_wt - ap, 0, np.inf)
        en = pen_wt * clipped * (ap > pen_wt)
        en /= (scad_param - 1) * pen_wt
        en += pen_wt * (ap <= pen_wt)
        en /= eps + ap
        hm.flat[::hm.shape[0] + 1] += self.num_group * en
        sn -= self.num_group * en * params
        try:
            update = np.linalg.solve(hm, sn)
        except np.linalg.LinAlgError:
            update = np.dot(np.linalg.pinv(hm), sn)
            msg = 'Encountered singularity in regularized GEE update'
            warnings.warn(msg)
        hm *= self.estimate_scale()
        return (update, hm)

    def _regularized_covmat(self, mean_params):
        if False:
            while True:
                i = 10
        self.update_cached_means(mean_params)
        ma = 0
        for i in range(self.num_group):
            (expval, _) = self.cached_means[i]
            resid = self.endog_li[i] - expval
            sdev = np.sqrt(self.family.variance(expval))
            ex = self.exog_li[i] * sdev[:, None] ** 2
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (resid,))
            ma0 = np.dot(ex.T, rslt[0])
            ma += np.outer(ma0, ma0)
        return ma

    def fit_regularized(self, pen_wt, scad_param=3.7, maxiter=100, ddof_scale=None, update_assoc=5, ctol=1e-05, ztol=0.001, eps=1e-06, scale=None):
        if False:
            i = 10
            return i + 15
        '\n        Regularized estimation for GEE.\n\n        Parameters\n        ----------\n        pen_wt : float\n            The penalty weight (a non-negative scalar).\n        scad_param : float\n            Non-negative scalar determining the shape of the Scad\n            penalty.\n        maxiter : int\n            The maximum number of iterations.\n        ddof_scale : int\n            Value to subtract from `nobs` when calculating the\n            denominator degrees of freedom for t-statistics, defaults\n            to the number of columns in `exog`.\n        update_assoc : int\n            The dependence parameters are updated every `update_assoc`\n            iterations of the mean structure parameter updates.\n        ctol : float\n            Convergence criterion, default is one order of magnitude\n            smaller than proposed in section 3.1 of Wang et al.\n        ztol : float\n            Coefficients smaller than this value are treated as\n            being zero, default is based on section 5 of Wang et al.\n        eps : non-negative scalar\n            Numerical constant, see section 3.2 of Wang et al.\n        scale : float or string\n            If a float, this value is used as the scale parameter.\n            If "X2", the scale parameter is always estimated using\n            Pearson\'s chi-square method (e.g. as in a quasi-Poisson\n            analysis).  If None, the default approach for the family\n            is used to estimate the scale parameter.\n\n        Returns\n        -------\n        GEEResults instance.  Note that not all methods of the results\n        class make sense when the model has been fit with regularization.\n\n        Notes\n        -----\n        This implementation assumes that the link is canonical.\n\n        References\n        ----------\n        Wang L, Zhou J, Qu A. (2012). Penalized generalized estimating\n        equations for high-dimensional longitudinal data analysis.\n        Biometrics. 2012 Jun;68(2):353-60.\n        doi: 10.1111/j.1541-0420.2011.01678.x.\n        https://www.ncbi.nlm.nih.gov/pubmed/21955051\n        http://users.stat.umn.edu/~wangx346/research/GEE_selection.pdf\n        '
        self.scaletype = scale
        mean_params = np.zeros(self.exog.shape[1])
        self.update_cached_means(mean_params)
        converged = False
        fit_history = defaultdict(list)
        if ddof_scale is None:
            self.ddof_scale = self.exog.shape[1]
        else:
            if not ddof_scale >= 0:
                raise ValueError('ddof_scale must be a non-negative number or None')
            self.ddof_scale = ddof_scale
        miniter = 20
        for itr in range(maxiter):
            (update, hm) = self._update_regularized(mean_params, pen_wt, scad_param, eps)
            if update is None:
                msg = 'Singular matrix encountered in regularized GEE update'
                warnings.warn(msg, ConvergenceWarning)
                break
            if itr > miniter and np.sqrt(np.sum(update ** 2)) < ctol:
                converged = True
                break
            mean_params += update
            fit_history['params'].append(mean_params.copy())
            self.update_cached_means(mean_params)
            if itr != 0 and itr % update_assoc == 0:
                self._update_assoc(mean_params)
        if not converged:
            msg = 'GEE.fit_regularized did not converge'
            warnings.warn(msg)
        mean_params[np.abs(mean_params) < ztol] = 0
        self._update_assoc(mean_params)
        ma = self._regularized_covmat(mean_params)
        cov = np.linalg.solve(hm, ma)
        cov = np.linalg.solve(hm, cov.T)
        res_kwds = dict(cov_type='robust', cov_robust=cov)
        scale = self.estimate_scale()
        rslt = GEEResults(self, mean_params, cov, scale, regularized=True, attr_kwds=res_kwds)
        rslt.fit_history = fit_history
        return GEEResultsWrapper(rslt)

    def _handle_constraint(self, mean_params, bcov):
        if False:
            return 10
        '\n        Expand the parameter estimate `mean_params` and covariance matrix\n        `bcov` to the coordinate system of the unconstrained model.\n\n        Parameters\n        ----------\n        mean_params : array_like\n            A parameter vector estimate for the reduced model.\n        bcov : array_like\n            The covariance matrix of mean_params.\n\n        Returns\n        -------\n        mean_params : array_like\n            The input parameter vector mean_params, expanded to the\n            coordinate system of the full model\n        bcov : array_like\n            The input covariance matrix bcov, expanded to the\n            coordinate system of the full model\n        '
        red_p = len(mean_params)
        full_p = self.constraint.lhs.shape[1]
        mean_params0 = np.r_[mean_params, np.zeros(full_p - red_p)]
        save_exog_li = self.exog_li
        self.exog_li = self.constraint.exog_fulltrans_li
        import copy
        save_cached_means = copy.deepcopy(self.cached_means)
        self.update_cached_means(mean_params0)
        (_, score) = self._update_mean_params()
        if score is None:
            warnings.warn('Singular matrix encountered in GEE score test', ConvergenceWarning)
            return (None, None)
        (_, ncov1, cmat) = self._covmat()
        scale = self.estimate_scale()
        cmat = cmat / scale ** 2
        score2 = score[red_p:] / scale
        amat = np.linalg.inv(ncov1)
        bmat_11 = cmat[0:red_p, 0:red_p]
        bmat_22 = cmat[red_p:, red_p:]
        bmat_12 = cmat[0:red_p, red_p:]
        amat_11 = amat[0:red_p, 0:red_p]
        amat_12 = amat[0:red_p, red_p:]
        score_cov = bmat_22 - np.dot(amat_12.T, np.linalg.solve(amat_11, bmat_12))
        score_cov -= np.dot(bmat_12.T, np.linalg.solve(amat_11, amat_12))
        score_cov += np.dot(amat_12.T, np.dot(np.linalg.solve(amat_11, bmat_11), np.linalg.solve(amat_11, amat_12)))
        from scipy.stats.distributions import chi2
        score_statistic = np.dot(score2, np.linalg.solve(score_cov, score2))
        score_df = len(score2)
        score_pvalue = 1 - chi2.cdf(score_statistic, score_df)
        self.score_test_results = {'statistic': score_statistic, 'df': score_df, 'p-value': score_pvalue}
        mean_params = self.constraint.unpack_param(mean_params)
        bcov = self.constraint.unpack_cov(bcov)
        self.exog_li = save_exog_li
        self.cached_means = save_cached_means
        self.exog = self.constraint.restore_exog()
        return (mean_params, bcov)

    def _update_assoc(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Update the association parameters\n        '
        self.cov_struct.update(params)

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        if False:
            return 10
        "\n        For computing marginal effects, returns dF(XB) / dX where F(.)\n        is the fitted mean.\n\n        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.\n\n        Not all of these make sense in the presence of discrete regressors,\n        but checks are done in the results in get_margeff.\n        "
        offset_exposure = None
        if exog is None:
            exog = self.exog
            offset_exposure = self._offset_exposure
        margeff = self.mean_deriv_exog(exog, params, offset_exposure)
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]
        if count_idx is not None:
            from statsmodels.discrete.discrete_margins import _get_count_effects
            margeff = _get_count_effects(margeff, exog, count_idx, transform, self, params)
        if dummy_idx is not None:
            from statsmodels.discrete.discrete_margins import _get_dummy_effects
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform, self, params)
        return margeff

    def qic(self, params, scale, cov_params, n_step=1000):
        if False:
            i = 10
            return i + 15
        "\n        Returns quasi-information criteria and quasi-likelihood values.\n\n        Parameters\n        ----------\n        params : array_like\n            The GEE estimates of the regression parameters.\n        scale : scalar\n            Estimated scale parameter\n        cov_params : array_like\n            An estimate of the covariance matrix for the\n            model parameters.  Conventionally this is the robust\n            covariance matrix.\n        n_step : integer\n            The number of points in the trapezoidal approximation\n            to the quasi-likelihood function.\n\n        Returns\n        -------\n        ql : scalar\n            The quasi-likelihood value\n        qic : scalar\n            A QIC that can be used to compare the mean and covariance\n            structures of the model.\n        qicu : scalar\n            A simplified QIC that can be used to compare mean structures\n            but not covariance structures\n\n        Notes\n        -----\n        The quasi-likelihood used here is obtained by numerically evaluating\n        Wedderburn's integral representation of the quasi-likelihood function.\n        This approach is valid for all families and  links.  Many other\n        packages use analytical expressions for quasi-likelihoods that are\n        valid in special cases where the link function is canonical.  These\n        analytical expressions may omit additive constants that only depend\n        on the data.  Therefore, the numerical values of our QL and QIC values\n        will differ from the values reported by other packages.  However only\n        the differences between two QIC values calculated for different models\n        using the same data are meaningful.  Our QIC should produce the same\n        QIC differences as other software.\n\n        When using the QIC for models with unknown scale parameter, use a\n        common estimate of the scale parameter for all models being compared.\n\n        References\n        ----------\n        .. [*] W. Pan (2001).  Akaike's information criterion in generalized\n               estimating equations.  Biometrics (57) 1.\n        "
        varfunc = self.family.variance
        means = []
        omega = 0.0
        for i in range(self.num_group):
            (expval, lpr) = self.cached_means[i]
            means.append(expval)
            dmat = self.mean_deriv(self.exog_li[i], lpr)
            omega += np.dot(dmat.T, dmat) / scale
        means = np.concatenate(means)
        endog_li = np.concatenate(self.endog_li)
        du = means - endog_li
        qv = np.empty(n_step)
        xv = np.linspace(-0.99999, 1, n_step)
        for (i, g) in enumerate(xv):
            u = endog_li + (g + 1) * du / 2.0
            vu = varfunc(u)
            qv[i] = -np.sum(du ** 2 * (g + 1) / vu)
        qv /= 4 * scale
        try:
            from scipy.integrate import trapezoid
        except ImportError:
            from scipy.integrate import trapz as trapezoid
        ql = trapezoid(qv, dx=xv[1] - xv[0])
        qicu = -2 * ql + 2 * self.exog.shape[1]
        qic = -2 * ql + 2 * np.trace(np.dot(omega, cov_params))
        return (ql, qic, qicu)

class GEEResults(GLMResults):
    __doc__ = 'This class summarizes the fit of a marginal regression model using GEE.\n' + _gee_results_doc

    def __init__(self, model, params, cov_params, scale, cov_type='robust', use_t=False, regularized=False, **kwds):
        if False:
            print('Hello World!')
        super(GEEResults, self).__init__(model, params, normalized_cov_params=cov_params, scale=scale)
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.family = model.family
        attr_kwds = kwds.pop('attr_kwds', {})
        self.__dict__.update(attr_kwds)
        if not (hasattr(self, 'cov_type') and hasattr(self, 'cov_params_default')):
            self.cov_type = cov_type
            covariance_type = self.cov_type.lower()
            allowed_covariances = ['robust', 'naive', 'bias_reduced']
            if covariance_type not in allowed_covariances:
                msg = 'GEE: `cov_type` must be one of ' + ', '.join(allowed_covariances)
                raise ValueError(msg)
            if cov_type == 'robust':
                cov = self.cov_robust
            elif cov_type == 'naive':
                cov = self.cov_naive
            elif cov_type == 'bias_reduced':
                cov = self.cov_robust_bc
            self.cov_params_default = cov
        elif self.cov_type != cov_type:
            raise ValueError('cov_type in argument is different from already attached cov_type')

    @cache_readonly
    def resid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The response residuals.\n        '
        return self.resid_response

    def standard_errors(self, cov_type='robust'):
        if False:
            return 10
        '\n        This is a convenience function that returns the standard\n        errors for any covariance type.  The value of `bse` is the\n        standard errors for whichever covariance type is specified as\n        an argument to `fit` (defaults to "robust").\n\n        Parameters\n        ----------\n        cov_type : str\n            One of "robust", "naive", or "bias_reduced".  Determines\n            the covariance used to compute standard errors.  Defaults\n            to "robust".\n        '
        covariance_type = cov_type.lower()
        allowed_covariances = ['robust', 'naive', 'bias_reduced']
        if covariance_type not in allowed_covariances:
            msg = 'GEE: `covariance_type` must be one of ' + ', '.join(allowed_covariances)
            raise ValueError(msg)
        if covariance_type == 'robust':
            return np.sqrt(np.diag(self.cov_robust))
        elif covariance_type == 'naive':
            return np.sqrt(np.diag(self.cov_naive))
        elif covariance_type == 'bias_reduced':
            if self.cov_robust_bc is None:
                raise ValueError('GEE: `bias_reduced` covariance not available')
            return np.sqrt(np.diag(self.cov_robust_bc))

    @cache_readonly
    def bse(self):
        if False:
            for i in range(10):
                print('nop')
        return self.standard_errors(self.cov_type)

    def score_test(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the results of a score test for a linear constraint.\n\n        Returns\n        -------\n        A\x7fdictionary containing the p-value, the test statistic,\n        and the degrees of freedom for the score test.\n\n        Notes\n        -----\n        See also GEE.compare_score_test for an alternative way to perform\n        a score test.  GEEResults.score_test is more general, in that it\n        supports testing arbitrary linear equality constraints.   However\n        GEE.compare_score_test might be easier to use when comparing\n        two explicit models.\n\n        References\n        ----------\n        Xu Guo and Wei Pan (2002). "Small sample performance of the score\n        test in GEE".\n        http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf\n        '
        if not hasattr(self.model, 'score_test_results'):
            msg = 'score_test on results instance only available when '
            msg += ' model was fit with constraints'
            raise ValueError(msg)
        return self.model.score_test_results

    @cache_readonly
    def resid_split(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the residuals, the endogeneous data minus the fitted\n        values from the model.  The residuals are returned as a list\n        of arrays containing the residuals for each cluster.\n        '
        sresid = []
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            sresid.append(self.resid[ii])
        return sresid

    @cache_readonly
    def resid_centered(self):
        if False:
            print('Hello World!')
        '\n        Returns the residuals centered within each group.\n        '
        cresid = self.resid.copy()
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            cresid[ii] -= cresid[ii].mean()
        return cresid

    @cache_readonly
    def resid_centered_split(self):
        if False:
            while True:
                i = 10
        '\n        Returns the residuals centered within each group.  The\n        residuals are returned as a list of arrays containing the\n        centered residuals for each cluster.\n        '
        sresid = []
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            sresid.append(self.centered_resid[ii])
        return sresid

    def qic(self, scale=None, n_step=1000):
        if False:
            while True:
                i = 10
        '\n        Returns the QIC and QICu information criteria.\n\n        See GEE.qic for documentation.\n        '
        if scale is None:
            warnings.warn('QIC values obtained using scale=None are not appropriate for comparing models')
        if scale is None:
            scale = self.scale
        (_, qic, qicu) = self.model.qic(self.params, scale, self.cov_params(), n_step=n_step)
        return (qic, qicu)
    split_resid = resid_split
    centered_resid = resid_centered
    split_centered_resid = resid_centered_split

    @Appender(_plot_added_variable_doc % {'extra_params_doc': ''})
    def plot_added_variable(self, focus_exog, resid_type=None, use_glm_weights=True, fit_kwargs=None, ax=None):
        if False:
            for i in range(10):
                print('nop')
        from statsmodels.graphics.regressionplots import plot_added_variable
        fig = plot_added_variable(self, focus_exog, resid_type=resid_type, use_glm_weights=use_glm_weights, fit_kwargs=fit_kwargs, ax=ax)
        return fig

    @Appender(_plot_partial_residuals_doc % {'extra_params_doc': ''})
    def plot_partial_residuals(self, focus_exog, ax=None):
        if False:
            print('Hello World!')
        from statsmodels.graphics.regressionplots import plot_partial_residuals
        return plot_partial_residuals(self, focus_exog, ax=ax)

    @Appender(_plot_ceres_residuals_doc % {'extra_params_doc': ''})
    def plot_ceres_residuals(self, focus_exog, frac=0.66, cond_means=None, ax=None):
        if False:
            for i in range(10):
                print('nop')
        from statsmodels.graphics.regressionplots import plot_ceres_residuals
        return plot_ceres_residuals(self, focus_exog, frac, cond_means=cond_means, ax=ax)

    def conf_int(self, alpha=0.05, cols=None, cov_type=None):
        if False:
            while True:
                i = 10
        "\n        Returns confidence intervals for the fitted parameters.\n\n        Parameters\n        ----------\n        alpha : float, optional\n             The `alpha` level for the confidence interval.  i.e., The\n             default `alpha` = .05 returns a 95% confidence interval.\n        cols : array_like, optional\n             `cols` specifies which confidence intervals to return\n        cov_type : str\n             The covariance type used for computing standard errors;\n             must be one of 'robust', 'naive', and 'bias reduced'.\n             See `GEE` for details.\n\n        Notes\n        -----\n        The confidence interval is based on the Gaussian distribution.\n        "
        if cov_type is None:
            bse = self.bse
        else:
            bse = self.standard_errors(cov_type=cov_type)
        params = self.params
        dist = stats.norm
        q = dist.ppf(1 - alpha / 2)
        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = params[cols] - q * bse[cols]
            upper = params[cols] + q * bse[cols]
        return np.asarray(lzip(lower, upper))

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        if False:
            for i in range(10):
                print('nop')
        "\n        Summarize the GEE regression results\n\n        Parameters\n        ----------\n        yname : str, optional\n            Default is `y`\n        xname : list[str], optional\n            Names for the exogenous variables, default is `var_#` for ## in\n            the number of regressors. Must match the number of parameters in\n            the model\n        title : str, optional\n            Title for the top table. If not None, then this replaces\n            the default title\n        alpha : float\n            significance level for the confidence intervals\n        cov_type : str\n            The covariance type used to compute the standard errors;\n            one of 'robust' (the usual robust sandwich-type covariance\n            estimate), 'naive' (ignores dependence), and 'bias\n            reduced' (the Mancl/DeRouen estimate).\n\n        Returns\n        -------\n        smry : Summary instance\n            this holds the summary tables and text, which can be\n            printed or converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary.Summary : class to hold summary results\n        "
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['Generalized']), ('', ['Estimating Equations']), ('Family:', [self.model.family.__class__.__name__]), ('Dependence structure:', [self.model.cov_struct.__class__.__name__]), ('Date:', None), ('Covariance type: ', [self.cov_type])]
        NY = [len(y) for y in self.model.endog_li]
        top_right = [('No. Observations:', [sum(NY)]), ('No. clusters:', [len(self.model.endog_li)]), ('Min. cluster size:', [min(NY)]), ('Max. cluster size:', [max(NY)]), ('Mean cluster size:', ['%.1f' % np.mean(NY)]), ('Num. iterations:', ['%d' % len(self.fit_history['params'])]), ('Scale:', ['%.3f' % self.scale]), ('Time:', None)]
        skew1 = stats.skew(self.resid)
        kurt1 = stats.kurtosis(self.resid)
        skew2 = stats.skew(self.centered_resid)
        kurt2 = stats.kurtosis(self.centered_resid)
        diagn_left = [('Skew:', ['%12.4f' % skew1]), ('Centered skew:', ['%12.4f' % skew2])]
        diagn_right = [('Kurtosis:', ['%12.4f' % kurt1]), ('Centered kurtosis:', ['%12.4f' % kurt2])]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        if xname is None:
            xname = self.model.exog_names
        if yname is None:
            yname = self.model.endog_names
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=False)
        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right, yname=yname, xname=xname, title='')
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        if False:
            while True:
                i = 10
        "Get marginal effects of the fitted model.\n\n        Parameters\n        ----------\n        at : str, optional\n            Options are:\n\n            - 'overall', The average of the marginal effects at each\n              observation.\n            - 'mean', The marginal effects at the mean of each regressor.\n            - 'median', The marginal effects at the median of each regressor.\n            - 'zero', The marginal effects at zero for each regressor.\n            - 'all', The marginal effects at each observation. If `at` is 'all'\n              only margeff will be available.\n\n            Note that if `exog` is specified, then marginal effects for all\n            variables not specified by `exog` are calculated using the `at`\n            option.\n        method : str, optional\n            Options are:\n\n            - 'dydx' - dy/dx - No transformation is made and marginal effects\n              are returned.  This is the default.\n            - 'eyex' - estimate elasticities of variables in `exog` --\n              d(lny)/d(lnx)\n            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)\n            - 'eydx' - estimate semi-elasticity -- d(lny)/dx\n\n            Note that tranformations are done after each observation is\n            calculated.  Semi-elasticities for binary variables are computed\n            using the midpoint method. 'dyex' and 'eyex' do not make sense\n            for discrete variables.\n        atexog : array_like, optional\n            Optionally, you can provide the exogenous variables over which to\n            get the marginal effects.  This should be a dictionary with the key\n            as the zero-indexed column number and the value of the dictionary.\n            Default is None for all independent variables less the constant.\n        dummy : bool, optional\n            If False, treats binary variables (if present) as continuous.  This\n            is the default.  Else if True, treats binary variables as\n            changing from 0 to 1.  Note that any variable that is either 0 or 1\n            is treated as binary.  Each binary variable is treated separately\n            for now.\n        count : bool, optional\n            If False, treats count variables (if present) as continuous.  This\n            is the default.  Else if True, the marginal effect is the\n            change in probabilities when each observation is increased by one.\n\n        Returns\n        -------\n        effects : ndarray\n            the marginal effect corresponding to the input options\n\n        Notes\n        -----\n        When using after Poisson, returns the expected number of events\n        per period, assuming that the model is loglinear.\n        "
        if self.model.constraint is not None:
            warnings.warn('marginal effects ignore constraints', ValueWarning)
        return GEEMargins(self, (at, method, atexog, dummy, count))

    def plot_isotropic_dependence(self, ax=None, xpoints=10, min_n=50):
        if False:
            i = 10
            return i + 15
        '\n        Create a plot of the pairwise products of within-group\n        residuals against the corresponding time differences.  This\n        plot can be used to assess the possible form of an isotropic\n        covariance structure.\n\n        Parameters\n        ----------\n        ax : AxesSubplot\n            An axes on which to draw the graph.  If None, new\n            figure and axes objects are created\n        xpoints : scalar or array_like\n            If scalar, the number of points equally spaced points on\n            the time difference axis used to define bins for\n            calculating local means.  If an array, the specific points\n            that define the bins.\n        min_n : int\n            The minimum sample size in a bin for the mean residual\n            product to be included on the plot.\n        '
        from statsmodels.graphics import utils as gutils
        resid = self.model.cluster_list(self.resid)
        time = self.model.cluster_list(self.model.time)
        (xre, xdt) = ([], [])
        for (re, ti) in zip(resid, time):
            ix = np.tril_indices(re.shape[0], 0)
            re = re[ix[0]] * re[ix[1]] / self.scale ** 2
            xre.append(re)
            dists = np.sqrt(((ti[ix[0], :] - ti[ix[1], :]) ** 2).sum(1))
            xdt.append(dists)
        xre = np.concatenate(xre)
        xdt = np.concatenate(xdt)
        if ax is None:
            (fig, ax) = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        ii = np.flatnonzero(xdt == 0)
        v0 = np.mean(xre[ii])
        xre /= v0
        if np.isscalar(xpoints):
            xpoints = np.linspace(0, max(xdt), xpoints)
        dg = np.digitize(xdt, xpoints)
        dgu = np.unique(dg)
        hist = np.asarray([np.sum(dg == k) for k in dgu])
        ii = np.flatnonzero(hist >= min_n)
        dgu = dgu[ii]
        dgy = np.asarray([np.mean(xre[dg == k]) for k in dgu])
        dgx = np.asarray([np.mean(xdt[dg == k]) for k in dgu])
        ax.plot(dgx, dgy, '-', color='orange', lw=5)
        ax.set_xlabel('Time difference')
        ax.set_ylabel('Product of scaled residuals')
        return fig

    def sensitivity_params(self, dep_params_first, dep_params_last, num_steps):
        if False:
            return 10
        '\n        Refits the GEE model using a sequence of values for the\n        dependence parameters.\n\n        Parameters\n        ----------\n        dep_params_first : array_like\n            The first dep_params in the sequence\n        dep_params_last : array_like\n            The last dep_params in the sequence\n        num_steps : int\n            The number of dep_params in the sequence\n\n        Returns\n        -------\n        results : array_like\n            The GEEResults objects resulting from the fits.\n        '
        model = self.model
        import copy
        cov_struct = copy.deepcopy(self.model.cov_struct)
        update_dep = model.update_dep
        model.update_dep = False
        dep_params = []
        results = []
        for x in np.linspace(0, 1, num_steps):
            dp = x * dep_params_last + (1 - x) * dep_params_first
            dep_params.append(dp)
            model.cov_struct = copy.deepcopy(cov_struct)
            model.cov_struct.dep_params = dp
            rslt = model.fit(start_params=self.params, ctol=self.ctol, params_niter=self.params_niter, first_dep_update=self.first_dep_update, cov_type=self.cov_type)
            results.append(rslt)
        model.update_dep = update_dep
        return results
    params_sensitivity = sensitivity_params

class GEEResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {'centered_resid': 'rows'}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs, _attrs)
wrap.populate_wrapper(GEEResultsWrapper, GEEResults)

class OrdinalGEE(GEE):
    __doc__ = '    Ordinal Response Marginal Regression Model using GEE\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_ordinal_family_doc, 'example': _gee_ordinal_example, 'notes': _gee_nointercept}

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, dep_data=None, constraint=None, **kwargs):
        if False:
            while True:
                i = 10
        if family is None:
            family = families.Binomial()
        elif not isinstance(family, families.Binomial):
            raise ValueError('ordinal GEE must use a Binomial family')
        if cov_struct is None:
            cov_struct = cov_structs.OrdinalIndependence()
        (endog, exog, groups, time, offset) = self.setup_ordinal(endog, exog, groups, time, offset)
        super(OrdinalGEE, self).__init__(endog, exog, groups, time, family, cov_struct, missing, offset, dep_data, constraint)

    def setup_ordinal(self, endog, exog, groups, time, offset):
        if False:
            while True:
                i = 10
        '\n        Restructure ordinal data as binary indicators so that they can\n        be analyzed using Generalized Estimating Equations.\n        '
        self.endog_orig = endog.copy()
        self.exog_orig = exog.copy()
        self.groups_orig = groups.copy()
        if offset is not None:
            self.offset_orig = offset.copy()
        else:
            self.offset_orig = None
            offset = np.zeros(len(endog))
        if time is not None:
            self.time_orig = time.copy()
        else:
            self.time_orig = None
            time = np.zeros((len(endog), 1))
        exog = np.asarray(exog)
        endog = np.asarray(endog)
        groups = np.asarray(groups)
        time = np.asarray(time)
        offset = np.asarray(offset)
        self.endog_values = np.unique(endog)
        endog_cuts = self.endog_values[0:-1]
        ncut = len(endog_cuts)
        nrows = ncut * len(endog)
        exog_out = np.zeros((nrows, exog.shape[1]), dtype=np.float64)
        endog_out = np.zeros(nrows, dtype=np.float64)
        intercepts = np.zeros((nrows, ncut), dtype=np.float64)
        groups_out = np.zeros(nrows, dtype=groups.dtype)
        time_out = np.zeros((nrows, time.shape[1]), dtype=np.float64)
        offset_out = np.zeros(nrows, dtype=np.float64)
        jrow = 0
        zipper = zip(exog, endog, groups, time, offset)
        for (exog_row, endog_value, group_value, time_value, offset_value) in zipper:
            for (thresh_ix, thresh) in enumerate(endog_cuts):
                exog_out[jrow, :] = exog_row
                endog_out[jrow] = int(np.squeeze(endog_value > thresh))
                intercepts[jrow, thresh_ix] = 1
                groups_out[jrow] = group_value
                time_out[jrow] = time_value
                offset_out[jrow] = offset_value
                jrow += 1
        exog_out = np.concatenate((intercepts, exog_out), axis=1)
        xnames = ['I(y>%.1f)' % v for v in endog_cuts]
        if type(self.exog_orig) is pd.DataFrame:
            xnames.extend(self.exog_orig.columns)
        else:
            xnames.extend(['x%d' % k for k in range(1, exog.shape[1] + 1)])
        exog_out = pd.DataFrame(exog_out, columns=xnames)
        if type(self.endog_orig) is pd.Series:
            endog_out = pd.Series(endog_out, name=self.endog_orig.name)
        return (endog_out, exog_out, groups_out, time_out, offset_out)

    def _starting_params(self):
        if False:
            while True:
                i = 10
        exposure = getattr(self, 'exposure', None)
        model = GEE(self.endog, self.exog, self.groups, time=self.time, family=families.Binomial(), offset=self.offset, exposure=exposure)
        result = model.fit()
        return result.params

    @Appender(_gee_fit_doc)
    def fit(self, maxiter=60, ctol=1e-06, start_params=None, params_niter=1, first_dep_update=0, cov_type='robust'):
        if False:
            return 10
        rslt = super(OrdinalGEE, self).fit(maxiter, ctol, start_params, params_niter, first_dep_update, cov_type=cov_type)
        rslt = rslt._results
        res_kwds = dict(((k, getattr(rslt, k)) for k in rslt._props))
        ord_rslt = OrdinalGEEResults(self, rslt.params, rslt.cov_params() / rslt.scale, rslt.scale, cov_type=cov_type, attr_kwds=res_kwds)
        return OrdinalGEEResultsWrapper(ord_rslt)

class OrdinalGEEResults(GEEResults):
    __doc__ = 'This class summarizes the fit of a marginal regression modelfor an ordinal response using GEE.\n' + _gee_results_doc

    def plot_distribution(self, ax=None, exog_values=None):
        if False:
            return 10
        '\n        Plot the fitted probabilities of endog in an ordinal model,\n        for specified values of the predictors.\n\n        Parameters\n        ----------\n        ax : AxesSubplot\n            An axes on which to draw the graph.  If None, new\n            figure and axes objects are created\n        exog_values : array_like\n            A list of dictionaries, with each dictionary mapping\n            variable names to values at which the variable is held\n            fixed.  The values P(endog=y | exog) are plotted for all\n            possible values of y, at the given exog value.  Variables\n            not included in a dictionary are held fixed at the mean\n            value.\n\n        Example:\n        --------\n        We have a model with covariates \'age\' and \'sex\', and wish to\n        plot the probabilities P(endog=y | exog) for males (sex=0) and\n        for females (sex=1), as separate paths on the plot.  Since\n        \'age\' is not included below in the map, it is held fixed at\n        its mean value.\n\n        >>> ev = [{"sex": 1}, {"sex": 0}]\n        >>> rslt.distribution_plot(exog_values=ev)\n        '
        from statsmodels.graphics import utils as gutils
        if ax is None:
            (fig, ax) = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        if exog_values is None:
            exog_values = [{}]
        exog_means = self.model.exog.mean(0)
        ix_icept = [i for (i, x) in enumerate(self.model.exog_names) if x.startswith('I(')]
        for ev in exog_values:
            for k in ev.keys():
                if k not in self.model.exog_names:
                    raise ValueError('%s is not a variable in the model' % k)
            pr = []
            for j in ix_icept:
                xp = np.zeros_like(self.params)
                xp[j] = 1.0
                for (i, vn) in enumerate(self.model.exog_names):
                    if i in ix_icept:
                        continue
                    if vn in ev:
                        xp[i] = ev[vn]
                    else:
                        xp[i] = exog_means[i]
                p = 1 / (1 + np.exp(-np.dot(xp, self.params)))
                pr.append(p)
            pr.insert(0, 1)
            pr.append(0)
            pr = np.asarray(pr)
            prd = -np.diff(pr)
            ax.plot(self.model.endog_values, prd, 'o-')
        ax.set_xlabel('Response value')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        return fig

def _score_test_submodel(par, sub):
    if False:
        while True:
            i = 10
    '\n    Return transformation matrices for design matrices.\n\n    Parameters\n    ----------\n    par : instance\n        The parent model\n    sub : instance\n        The sub-model\n\n    Returns\n    -------\n    qm : array_like\n        Matrix mapping the design matrix of the parent to the design matrix\n        for the sub-model.\n    qc : array_like\n        Matrix mapping the design matrix of the parent to the orthogonal\n        complement of the columnspace of the submodel in the columnspace\n        of the parent.\n\n    Notes\n    -----\n    Returns None, None if the provided submodel is not actually a submodel.\n    '
    x1 = par.exog
    x2 = sub.exog
    (u, s, vt) = np.linalg.svd(x1, 0)
    v = vt.T
    (a, _) = np.linalg.qr(x2)
    a = u - np.dot(a, np.dot(a.T, u))
    (x2c, sb, _) = np.linalg.svd(a, 0)
    x2c = x2c[:, sb > 1e-12]
    ii = np.flatnonzero(np.abs(s) > 1e-12)
    qm = np.dot(v[:, ii], np.dot(u[:, ii].T, x2) / s[ii, None])
    e = np.max(np.abs(x2 - np.dot(x1, qm)))
    if e > 1e-08:
        return (None, None)
    qc = np.dot(v[:, ii], np.dot(u[:, ii].T, x2c) / s[ii, None])
    return (qm, qc)

class OrdinalGEEResultsWrapper(GEEResultsWrapper):
    pass
wrap.populate_wrapper(OrdinalGEEResultsWrapper, OrdinalGEEResults)

class NominalGEE(GEE):
    __doc__ = '    Nominal Response Marginal Regression Model using GEE.\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_nominal_family_doc, 'example': _gee_nominal_example, 'notes': _gee_nointercept}

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, dep_data=None, constraint=None, **kwargs):
        if False:
            while True:
                i = 10
        (endog, exog, groups, time, offset) = self.setup_nominal(endog, exog, groups, time, offset)
        if family is None:
            family = _Multinomial(self.ncut + 1)
        if cov_struct is None:
            cov_struct = cov_structs.NominalIndependence()
        super(NominalGEE, self).__init__(endog, exog, groups, time, family, cov_struct, missing, offset, dep_data, constraint)

    def _starting_params(self):
        if False:
            for i in range(10):
                print('nop')
        exposure = getattr(self, 'exposure', None)
        model = GEE(self.endog, self.exog, self.groups, time=self.time, family=families.Binomial(), offset=self.offset, exposure=exposure)
        result = model.fit()
        return result.params

    def setup_nominal(self, endog, exog, groups, time, offset):
        if False:
            print('Hello World!')
        '\n        Restructure nominal data as binary indicators so that they can\n        be analyzed using Generalized Estimating Equations.\n        '
        self.endog_orig = endog.copy()
        self.exog_orig = exog.copy()
        self.groups_orig = groups.copy()
        if offset is not None:
            self.offset_orig = offset.copy()
        else:
            self.offset_orig = None
            offset = np.zeros(len(endog))
        if time is not None:
            self.time_orig = time.copy()
        else:
            self.time_orig = None
            time = np.zeros((len(endog), 1))
        exog = np.asarray(exog)
        endog = np.asarray(endog)
        groups = np.asarray(groups)
        time = np.asarray(time)
        offset = np.asarray(offset)
        self.endog_values = np.unique(endog)
        endog_cuts = self.endog_values[0:-1]
        ncut = len(endog_cuts)
        self.ncut = ncut
        nrows = len(endog_cuts) * exog.shape[0]
        ncols = len(endog_cuts) * exog.shape[1]
        exog_out = np.zeros((nrows, ncols), dtype=np.float64)
        endog_out = np.zeros(nrows, dtype=np.float64)
        groups_out = np.zeros(nrows, dtype=np.float64)
        time_out = np.zeros((nrows, time.shape[1]), dtype=np.float64)
        offset_out = np.zeros(nrows, dtype=np.float64)
        jrow = 0
        zipper = zip(exog, endog, groups, time, offset)
        for (exog_row, endog_value, group_value, time_value, offset_value) in zipper:
            for (thresh_ix, thresh) in enumerate(endog_cuts):
                u = np.zeros(len(endog_cuts), dtype=np.float64)
                u[thresh_ix] = 1
                exog_out[jrow, :] = np.kron(u, exog_row)
                endog_out[jrow] = int(endog_value == thresh)
                groups_out[jrow] = group_value
                time_out[jrow] = time_value
                offset_out[jrow] = offset_value
                jrow += 1
        if isinstance(self.exog_orig, pd.DataFrame):
            xnames_in = self.exog_orig.columns
        else:
            xnames_in = ['x%d' % k for k in range(1, exog.shape[1] + 1)]
        xnames = []
        for tr in endog_cuts:
            xnames.extend(['%s[%.1f]' % (v, tr) for v in xnames_in])
        exog_out = pd.DataFrame(exog_out, columns=xnames)
        exog_out = pd.DataFrame(exog_out, columns=xnames)
        if isinstance(self.endog_orig, pd.Series):
            endog_out = pd.Series(endog_out, name=self.endog_orig.name)
        return (endog_out, exog_out, groups_out, time_out, offset_out)

    def mean_deriv(self, exog, lin_pred):
        if False:
            return 10
        '\n        Derivative of the expected endog with respect to the parameters.\n\n        Parameters\n        ----------\n        exog : array_like\n           The exogeneous data at which the derivative is computed,\n           number of rows must be a multiple of `ncut`.\n        lin_pred : array_like\n           The values of the linear predictor, length must be multiple\n           of `ncut`.\n\n        Returns\n        -------\n        The derivative of the expected endog with respect to the\n        parameters.\n        '
        expval = np.exp(lin_pred)
        expval_m = np.reshape(expval, (len(expval) // self.ncut, self.ncut))
        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))
        mprob = expval / denom
        dmat = mprob[:, None] * exog
        ddenom = expval[:, None] * exog
        dmat -= mprob[:, None] * ddenom / denom[:, None]
        return dmat

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Derivative of the expected endog with respect to exog for the\n        multinomial model, used in analyzing marginal effects.\n\n        Parameters\n        ----------\n        exog : array_like\n           The exogeneous data at which the derivative is computed,\n           number of rows must be a multiple of `ncut`.\n        lpr : array_like\n           The linear predictor values, length must be multiple of\n           `ncut`.\n\n        Returns\n        -------\n        The value of the derivative of the expected endog with respect\n        to exog.\n\n        Notes\n        -----\n        offset_exposure must be set at None for the multinomial family.\n        '
        if offset_exposure is not None:
            warnings.warn('Offset/exposure ignored for the multinomial family', ValueWarning)
        lpr = np.dot(exog, params)
        expval = np.exp(lpr)
        expval_m = np.reshape(expval, (len(expval) // self.ncut, self.ncut))
        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))
        bmat0 = np.outer(np.ones(exog.shape[0]), params)
        qmat = []
        for j in range(self.ncut):
            ee = np.zeros(self.ncut, dtype=np.float64)
            ee[j] = 1
            qmat.append(np.kron(ee, np.ones(len(params) // self.ncut)))
        qmat = np.array(qmat)
        qmat = np.kron(np.ones((exog.shape[0] // self.ncut, 1)), qmat)
        bmat = bmat0 * qmat
        dmat = expval[:, None] * bmat / denom[:, None]
        expval_mb = np.kron(expval_m, np.ones((self.ncut, 1)))
        expval_mb = np.kron(expval_mb, np.ones((1, self.ncut)))
        dmat -= expval[:, None] * (bmat * expval_mb) / denom[:, None] ** 2
        return dmat

    @Appender(_gee_fit_doc)
    def fit(self, maxiter=60, ctol=1e-06, start_params=None, params_niter=1, first_dep_update=0, cov_type='robust'):
        if False:
            return 10
        rslt = super(NominalGEE, self).fit(maxiter, ctol, start_params, params_niter, first_dep_update, cov_type=cov_type)
        if rslt is None:
            warnings.warn('GEE updates did not converge', ConvergenceWarning)
            return None
        rslt = rslt._results
        res_kwds = dict(((k, getattr(rslt, k)) for k in rslt._props))
        nom_rslt = NominalGEEResults(self, rslt.params, rslt.cov_params() / rslt.scale, rslt.scale, cov_type=cov_type, attr_kwds=res_kwds)
        return NominalGEEResultsWrapper(nom_rslt)

class NominalGEEResults(GEEResults):
    __doc__ = 'This class summarizes the fit of a marginal regression modelfor a nominal response using GEE.\n' + _gee_results_doc

    def plot_distribution(self, ax=None, exog_values=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Plot the fitted probabilities of endog in an nominal model,\n        for specified values of the predictors.\n\n        Parameters\n        ----------\n        ax : AxesSubplot\n            An axes on which to draw the graph.  If None, new\n            figure and axes objects are created\n        exog_values : array_like\n            A list of dictionaries, with each dictionary mapping\n            variable names to values at which the variable is held\n            fixed.  The values P(endog=y | exog) are plotted for all\n            possible values of y, at the given exog value.  Variables\n            not included in a dictionary are held fixed at the mean\n            value.\n\n        Example:\n        --------\n        We have a model with covariates \'age\' and \'sex\', and wish to\n        plot the probabilities P(endog=y | exog) for males (sex=0) and\n        for females (sex=1), as separate paths on the plot.  Since\n        \'age\' is not included below in the map, it is held fixed at\n        its mean value.\n\n        >>> ex = [{"sex": 1}, {"sex": 0}]\n        >>> rslt.distribution_plot(exog_values=ex)\n        '
        from statsmodels.graphics import utils as gutils
        if ax is None:
            (fig, ax) = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        if exog_values is None:
            exog_values = [{}]
        link = self.model.family.link.inverse
        ncut = self.model.family.ncut
        k = int(self.model.exog.shape[1] / ncut)
        exog_means = self.model.exog.mean(0)[0:k]
        exog_names = self.model.exog_names[0:k]
        exog_names = [x.split('[')[0] for x in exog_names]
        params = np.reshape(self.params, (ncut, len(self.params) // ncut))
        for ev in exog_values:
            exog = exog_means.copy()
            for k in ev.keys():
                if k not in exog_names:
                    raise ValueError('%s is not a variable in the model' % k)
                ii = exog_names.index(k)
                exog[ii] = ev[k]
            lpr = np.dot(params, exog)
            pr = link(lpr)
            pr = np.r_[pr, 1 - pr.sum()]
            ax.plot(self.model.endog_values, pr, 'o-')
        ax.set_xlabel('Response value')
        ax.set_ylabel('Probability')
        ax.set_xticks(self.model.endog_values)
        ax.set_xticklabels(self.model.endog_values)
        ax.set_ylim(0, 1)
        return fig

class NominalGEEResultsWrapper(GEEResultsWrapper):
    pass
wrap.populate_wrapper(NominalGEEResultsWrapper, NominalGEEResults)

class _MultinomialLogit(Link):
    """
    The multinomial logit transform, only for use with GEE.

    Notes
    -----
    The data are assumed coded as binary indicators, where each
    observed multinomial value y is coded as I(y == S[0]), ..., I(y ==
    S[-1]), where S is the set of possible response labels, excluding
    the largest one.  Thererefore functions in this class should only
    be called using vector argument whose length is a multiple of |S|
    = ncut, which is an argument to be provided when initializing the
    class.

    call and derivative use a private method _clean to trim p by 1e-10
    so that p is in (0, 1)
    """

    def __init__(self, ncut):
        if False:
            while True:
                i = 10
        self.ncut = ncut

    def inverse(self, lpr):
        if False:
            while True:
                i = 10
        '\n        Inverse of the multinomial logit transform, which gives the\n        expected values of the data as a function of the linear\n        predictors.\n\n        Parameters\n        ----------\n        lpr : array_like (length must be divisible by `ncut`)\n            The linear predictors\n\n        Returns\n        -------\n        prob : ndarray\n            Probabilities, or expected values\n        '
        expval = np.exp(lpr)
        denom = 1 + np.reshape(expval, (len(expval) // self.ncut, self.ncut)).sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))
        prob = expval / denom
        return prob

class _Multinomial(families.Family):
    """
    Pseudo-link function for fitting nominal multinomial models with
    GEE.  Not for use outside the GEE class.
    """
    links = [_MultinomialLogit]
    variance = varfuncs.binary
    safe_links = [_MultinomialLogit]

    def __init__(self, nlevels, check_link=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        nlevels : int\n            The number of distinct categories for the multinomial\n            distribution.\n        '
        self._check_link = check_link
        self.initialize(nlevels)

    def initialize(self, nlevels):
        if False:
            return 10
        self.ncut = nlevels - 1
        self.link = _MultinomialLogit(self.ncut)

class GEEMargins:
    """
    Estimated marginal effects for a regression model fit with GEE.

    Parameters
    ----------
    results : GEEResults instance
        The results instance of a fitted discrete choice model
    args : tuple
        Args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    kwargs : dict
        Keyword args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    """

    def __init__(self, results, args, kwargs={}):
        if False:
            print('Hello World!')
        self._cache = {}
        self.results = results
        self.get_margeff(*args, **kwargs)

    def _reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._cache = {}

    @cache_readonly
    def tvalues(self):
        if False:
            for i in range(10):
                print('nop')
        _check_at_is_all(self.margeff_options)
        return self.margeff / self.margeff_se

    def summary_frame(self, alpha=0.05):
        if False:
            while True:
                i = 10
        '\n        Returns a DataFrame summarizing the marginal effects.\n\n        Parameters\n        ----------\n        alpha : float\n            Number between 0 and 1. The confidence intervals have the\n            probability 1-alpha.\n\n        Returns\n        -------\n        frame : DataFrames\n            A DataFrame summarizing the marginal effects.\n        '
        _check_at_is_all(self.margeff_options)
        from pandas import DataFrame
        names = [_transform_names[self.margeff_options['method']], 'Std. Err.', 'z', 'Pr(>|z|)', 'Conf. Int. Low', 'Cont. Int. Hi.']
        ind = self.results.model.exog.var(0) != 0
        exog_names = self.results.model.exog_names
        var_names = [name for (i, name) in enumerate(exog_names) if ind[i]]
        table = np.column_stack((self.margeff, self.margeff_se, self.tvalues, self.pvalues, self.conf_int(alpha)))
        return DataFrame(table, columns=names, index=var_names)

    @cache_readonly
    def pvalues(self):
        if False:
            i = 10
            return i + 15
        _check_at_is_all(self.margeff_options)
        return stats.norm.sf(np.abs(self.tvalues)) * 2

    def conf_int(self, alpha=0.05):
        if False:
            while True:
                i = 10
        '\n        Returns the confidence intervals of the marginal effects\n\n        Parameters\n        ----------\n        alpha : float\n            Number between 0 and 1. The confidence intervals have the\n            probability 1-alpha.\n\n        Returns\n        -------\n        conf_int : ndarray\n            An array with lower, upper confidence intervals for the marginal\n            effects.\n        '
        _check_at_is_all(self.margeff_options)
        me_se = self.margeff_se
        q = stats.norm.ppf(1 - alpha / 2)
        lower = self.margeff - q * me_se
        upper = self.margeff + q * me_se
        return np.asarray(lzip(lower, upper))

    def summary(self, alpha=0.05):
        if False:
            print('Hello World!')
        '\n        Returns a summary table for marginal effects\n\n        Parameters\n        ----------\n        alpha : float\n            Number between 0 and 1. The confidence intervals have the\n            probability 1-alpha.\n\n        Returns\n        -------\n        Summary : SummaryTable\n            A SummaryTable instance\n        '
        _check_at_is_all(self.margeff_options)
        results = self.results
        model = results.model
        title = model.__class__.__name__ + ' Marginal Effects'
        method = self.margeff_options['method']
        top_left = [('Dep. Variable:', [model.endog_names]), ('Method:', [method]), ('At:', [self.margeff_options['at']])]
        from statsmodels.iolib.summary import Summary, summary_params, table_extend
        exog_names = model.exog_names[:]
        smry = Summary()
        const_idx = model.data.const_idx
        if const_idx is not None:
            exog_names.pop(const_idx)
        J = int(getattr(model, 'J', 1))
        if J > 1:
            (yname, yname_list) = results._get_endog_name(model.endog_names, None, all=True)
        else:
            yname = model.endog_names
            yname_list = [yname]
        smry.add_table_2cols(self, gleft=top_left, gright=[], yname=yname, xname=exog_names, title=title)
        table = []
        conf_int = self.conf_int(alpha)
        margeff = self.margeff
        margeff_se = self.margeff_se
        tvalues = self.tvalues
        pvalues = self.pvalues
        if J > 1:
            for eq in range(J):
                restup = (results, margeff[:, eq], margeff_se[:, eq], tvalues[:, eq], pvalues[:, eq], conf_int[:, :, eq])
                tble = summary_params(restup, yname=yname_list[eq], xname=exog_names, alpha=alpha, use_t=False, skip_header=True)
                tble.title = yname_list[eq]
                header = ['', _transform_names[method], 'std err', 'z', 'P>|z|', '[%3.1f%% Conf. Int.]' % (100 - alpha * 100)]
                tble.insert_header_row(0, header)
                table.append(tble)
            table = table_extend(table, keep_headers=True)
        else:
            restup = (results, margeff, margeff_se, tvalues, pvalues, conf_int)
            table = summary_params(restup, yname=yname, xname=exog_names, alpha=alpha, use_t=False, skip_header=True)
            header = ['', _transform_names[method], 'std err', 'z', 'P>|z|', '[%3.1f%% Conf. Int.]' % (100 - alpha * 100)]
            table.insert_header_row(0, header)
        smry.tables.append(table)
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        if False:
            while True:
                i = 10
        self._reset()
        method = method.lower()
        at = at.lower()
        _check_margeff_args(at, method)
        self.margeff_options = dict(method=method, at=at)
        results = self.results
        model = results.model
        params = results.params
        exog = model.exog.copy()
        effects_idx = exog.var(0) != 0
        const_idx = model.data.const_idx
        if dummy:
            _check_discrete_args(at, method)
            (dummy_idx, dummy) = _get_dummy_index(exog, const_idx)
        else:
            dummy_idx = None
        if count:
            _check_discrete_args(at, method)
            (count_idx, count) = _get_count_index(exog, const_idx)
        else:
            count_idx = None
        exog = _get_margeff_exog(exog, at, atexog, effects_idx)
        effects = model._derivative_exog(params, exog, method, dummy_idx, count_idx)
        effects = _effects_at(effects, at)
        if at == 'all':
            self.margeff = effects[:, effects_idx]
        else:
            (margeff_cov, margeff_se) = margeff_cov_with_se(model, params, exog, results.cov_params(), at, model._derivative_exog, dummy_idx, count_idx, method, 1)
            self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
            self.margeff_se = margeff_se[effects_idx]
            self.margeff = effects[effects_idx]