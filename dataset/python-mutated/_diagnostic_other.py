"""Score, lagrange multiplier and conditional moment tests
robust to misspecification or without specification of higher moments

Created on Thu Oct 30 00:42:38 2014

Author: Josef Perktold
License: BSD-3

Notes
-----

This module is a mixture of very general and very specific functions for
hypothesis testing in general models, targeted mainly to non-normal models.

Some of the options or versions of these tests are mainly intented for
cross-checking and to replicate different examples in references.

We need clean versions with good defaults for those functions that are
intended for the user.



References
----------

The following references are collected after my intitial implementation and is
most likely not exactly what I used.

The main articles on which the functions are directly based upon, are Boos 1992,
Tauchen 1985 and Whitney 1985a. Wooldrige artificial regression is
based on several articles and his text book.
Background reading are the textbooks by Cameron and Trivedi, Wooldridge and
Davidson and MacKinnon.
Newey and MacFadden 1994 provide some of the theoretical background.

Poisson dispersion tests are based on Dean 1992 and articles and text books by
Cameron and Trivedi.

The references currently do not include the literature on LM-test for
specification and diagnostic testing, like Pagan, Bera, Bera and Yoon and
many others, except those for the Poisson excess dispersion case and Pagan
and Vella.


Boos, Dennis D. 1992. “On Generalized Score Tests.” The American Statistician 46
(4): 327–33. https://doi.org/10.2307/2685328.

Breslow, Norman. 1989. “Score Tests in Overdispersed GLM’s.” In Statistical
Modelling, edited by Adriano Decarli, Brian J. Francis, Robert Gilchrist, and
Gilg U. H. Seeber, 64–74. Lecture Notes in Statistics 57. Springer New York.
http://link.springer.com/chapter/10.1007/978-1-4612-3680-1_8.

Breslow, Norman. 1990. “Tests of Hypotheses in Overdispersed Poisson Regression
and Other Quasi- Likelihood Models.” Journal of the American Statistical
Association 85 (410): 565–71. https://doi.org/10.2307/2289799.

Cameron, A. Colin, and Pravin K. Trivedi. 1986. “Econometric Models Based on
Count Data. Comparisons and Applications of Some Estimators and Tests.” Journal
of Applied Econometrics 1 (1): 29–53. https://doi.org/10.1002/jae.3950010104.

Cameron, A. Colin, and Pravin K. Trivedi. 1990a. “Conditional Moment Tests and
Orthogonal Polynomials.” Indiana University, Department of Economics, Working
Paper, 90–051.

Cameron, A. Colin, and Pravin K. Trivedi. 1990b. “Regression-Based Tests for
Overdispersion in the Poisson Model.” Journal of Econometrics 46 (3): 347–64.
https://doi.org/10.1016/0304-4076(90)90014-K.

Cameron, A. Colin, and Pravin K. Trivedi. Microeconometrics: methods and
applications. Cambridge university press, 2005.

Cameron, A. Colin, and Pravin K. Trivedi. Regression analysis of count data.
Vol. 53. Cambridge university press, 2013.

Davidson, Russell, and James G. MacKinnon. 1981. “Several Tests for Model
Specification in the Presence of Alternative Hypotheses.” Econometrica 49 (3):
781–93. https://doi.org/10.2307/1911522.

Davidson, Russell, and James G. MacKinnon. 1990. “Specification Tests Based on
Artificial Regressions.” Journal of the American Statistical Association 85
(409): 220–27. https://doi.org/10.2307/2289548.

Davidson, Russell, and James G. MacKinnon. 1991. “Artificial Regressions and C
(α) Tests.” Economics Letters 35 (2): 149–53.
https://doi.org/10.1016/0165-1765(91)90162-E.

Davidson, Russell, and James G. MacKinnon. Econometric theory and methods. Vol.
5. New York: Oxford University Press, 2004.

Dean, C. B. 1992. “Testing for Overdispersion in Poisson and Binomial Regression
Models.” Journal of the American Statistical Association 87 (418): 451–57.
https://doi.org/10.2307/2290276.

Dean, C., and J. F. Lawless. 1989. “Tests for Detecting Overdispersion in
Poisson Regression Models.” Journal of the American Statistical Association 84
(406): 467–72. https://doi.org/10.2307/2289931.

Newey, Whitney K. 1985a. “Generalized Method of Moments Specification Testing.”
Journal of Econometrics 29 (3): 229–56.
https://doi.org/10.1016/0304-4076(85)90154-X.

Newey, Whitney K. 1985b. “Maximum Likelihood Specification Testing and
Conditional Moment Tests.” Econometrica 53 (5): 1047–70.
https://doi.org/10.2307/1911011.

Newey, Whitney K. and Kenneth D. West. 1987. “Hypothesis Testing with Efficient
Method of Moments Estimation.” International Economic Review 28 (3): 777–87.
https://doi.org/10.2307/2526578.

Newey, Whitney K. and Daniel McFadden. 1994 "Large sample estimation and
hypothesis testing." Handbook of econometrics 4: 2111-2245.

Pagan, Adrian, and Frank Vella. 1989. “Diagnostic Tests for Models Based on
Individual Data: A Survey.” Journal of Applied Econometrics 4 (S1): S29–59.
https://doi.org/10.1002/jae.3950040504.

Tauchen, George. 1985. “Diagnostic Testing and Evaluation of Maximum Likelihood
Models.” Journal of Econometrics 30 (1–2): 415–43.
https://doi.org/10.1016/0304-4076(85)90149-6.

White, Halbert. 1981. “Consequences and Detection of Misspecified Nonlinear
Regression Models.” Journal of the American Statistical Association 76 (374):
419–33. https://doi.org/10.2307/2287845.

White, Halbert. 1983. “Maximum Likelihood Estimation of Misspecified Models.”
Econometrica 51 (2): 513. https://doi.org/10.2307/1912004.

White, Halbert. 1994. Estimation, Inference and Specification Analysis.
Cambridge: Cambridge University Press. https://doi.org/10.1017/CCOL0521252806.

Wooldridge, Jeffrey M. 1991. “Specification Testing and Quasi-Maximum-
Likelihood Estimation.” Journal of Econometrics 48 (1–2): 29–55.
https://doi.org/10.1016/0304-4076(91)90031-8.

Wooldridge, Jeffrey M. 1990. “A Unified Approach to Robust, Regression-Based
Specification Tests.” Econometric Theory 6 (1): 17–43.

Wooldridge, Jeffrey M. 1991a. “On the Application of Robust, Regression- Based
Diagnostics to Models of Conditional Means and Conditional Variances.” Journal
of Econometrics 47 (1): 5–46. https://doi.org/10.1016/0304-4076(91)90076-P.

Wooldridge, Jeffrey M. 1991b. “On the Application of Robust, Regression- Based
Diagnostics to Models of Conditional Means and Conditional Variances.” Journal
of Econometrics 47 (1): 5–46. https://doi.org/10.1016/0304-4076(91)90076-P.

Wooldridge, Jeffrey M. 1991c. “Specification Testing and Quasi-Maximum-
Likelihood Estimation.” Journal of Econometrics 48 (1–2): 29–55.
https://doi.org/10.1016/0304-4076(91)90031-8.

Wooldridge, Jeffrey M. 1994. “On the Limits of GLM for Specification Testing: A
Comment on Gurmu and Trivedi.” Econometric Theory 10 (2): 409–18.
https://doi.org/10.2307/3532875.

Wooldridge, Jeffrey M. 1997. “Quasi-Likelihood Methods for Count Data.” Handbook
of Applied Econometrics 2: 352–406.

Wooldridge, Jeffrey M. Econometric analysis of cross section and panel data. MIT
press, 2010.

"""
import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS

def dispersion_poisson(results):
    if False:
        i = 10
        return i + 15
    'Score/LM type tests for Poisson variance assumptions\n\n    .. deprecated:: 0.14\n\n       dispersion_poisson moved to discrete._diagnostic_count\n\n    Null Hypothesis is\n\n    H0: var(y) = E(y) and assuming E(y) is correctly specified\n    H1: var(y) ~= E(y)\n\n    The tests are based on the constrained model, i.e. the Poisson model.\n    The tests differ in their assumed alternatives, and in their maintained\n    assumptions.\n\n    Parameters\n    ----------\n    results : Poisson results instance\n        This can be a results instance for either a discrete Poisson or a GLM\n        with family Poisson.\n\n    Returns\n    -------\n    res : ndarray, shape (7, 2)\n       each row contains the test statistic and p-value for one of the 7 tests\n       computed here.\n    description : 2-D list of strings\n       Each test has two strings a descriptive name and a string for the\n       alternative hypothesis.\n    '
    msg = 'dispersion_poisson here is deprecated, use the version in discrete._diagnostic_count'
    warnings.warn(msg, FutureWarning)
    from statsmodels.discrete._diagnostics_count import test_poisson_dispersion
    return test_poisson_dispersion(results, _old=True)

def dispersion_poisson_generic(results, exog_new_test, exog_new_control=None, include_score=False, use_endog=True, cov_type='HC3', cov_kwds=None, use_t=False):
    if False:
        while True:
            i = 10
    'A variable addition test for the variance function\n\n    .. deprecated:: 0.14\n\n       dispersion_poisson_generic moved to discrete._diagnostic_count\n\n    This uses an artificial regression to calculate a variant of an LM or\n    generalized score test for the specification of the variance assumption\n    in a Poisson model. The performed test is a Wald test on the coefficients\n    of the `exog_new_test`.\n\n    Warning: insufficiently tested, especially for options\n    '
    msg = 'dispersion_poisson_generic here is deprecated, use the version in discrete._diagnostic_count'
    warnings.warn(msg, FutureWarning)
    from statsmodels.discrete._diagnostics_count import _test_poisson_dispersion_generic
    res_test = _test_poisson_dispersion_generic(results, exog_new_test, exog_new_control=exog_new_control, include_score=include_score, use_endog=use_endog, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    return res_test

class ResultsGeneric:

    def __init__(self, **kwds):
        if False:
            return 10
        self.__dict__.update(kwds)

class TestResults(ResultsGeneric):

    def summary(self):
        if False:
            while True:
                i = 10
        txt = 'Specification Test (LM, score)\n'
        stat = [self.c1, self.c2, self.c3]
        pval = [self.pval1, self.pval2, self.pval3]
        description = ['nonrobust', 'dispersed', 'HC']
        for row in zip(description, stat, pval):
            txt += '%-12s  statistic = %6.4f  pvalue = %6.5f\n' % row
        txt += '\nAssumptions:\n'
        txt += 'nonrobust: variance is correctly specified\n'
        txt += 'dispersed: variance correctly specified up to scale factor\n'
        txt += 'HC       : robust to any heteroscedasticity\n'
        txt += 'test is not robust to correlation across observations'
        return txt

def lm_test_glm(result, exog_extra, mean_deriv=None):
    if False:
        print('Hello World!')
    'score/lagrange multiplier test for GLM\n\n    Wooldridge procedure for test of mean function in GLM\n\n    Parameters\n    ----------\n    results : GLMResults instance\n        results instance with the constrained model\n    exog_extra : ndarray or None\n        additional exogenous variables for variable addition test\n        This can be set to None if mean_deriv is provided.\n    mean_deriv : None or ndarray\n        Extra moment condition that correspond to the partial derivative of\n        a mean function with respect to some parameters.\n\n    Returns\n    -------\n    test_results : Results instance\n        The results instance has the following attributes which are score\n        statistic and p-value for 3 versions of the score test.\n\n        c1, pval1 : nonrobust score_test results\n        c2, pval2 : score test results robust to over or under dispersion\n        c3, pval3 : score test results fully robust to any heteroscedasticity\n\n        The test results instance also has a simple summary method.\n\n    Notes\n    -----\n    TODO: add `df` to results and make df detection more robust\n\n    This implements the auxiliary regression procedure of Wooldridge,\n    implemented based on the presentation in chapter 8 in Handbook of\n    Applied Econometrics 2.\n\n    References\n    ----------\n    Wooldridge, Jeffrey M. 1997. “Quasi-Likelihood Methods for Count Data.”\n    Handbook of Applied Econometrics 2: 352–406.\n\n    and other articles and text book by Wooldridge\n\n    '
    if hasattr(result, '_result'):
        res = result._result
    else:
        res = result
    mod = result.model
    nobs = mod.endog.shape[0]
    dlinkinv = mod.family.link.inverse_deriv
    dm = lambda x, linpred: dlinkinv(linpred)[:, None] * x
    var_func = mod.family.variance
    x = result.model.exog
    x2 = exog_extra
    try:
        lin_pred = res.predict(which='linear')
    except TypeError:
        lin_pred = res.predict(linear=True)
    dm_incl = dm(x, lin_pred)
    if x2 is not None:
        dm_excl = dm(x2, lin_pred)
        if mean_deriv is not None:
            dm_excl = np.column_stack((dm_excl, mean_deriv))
    elif mean_deriv is not None:
        dm_excl = mean_deriv
    else:
        raise ValueError('either exog_extra or mean_deriv have to be provided')
    k_constraint = dm_excl.shape[1]
    fittedvalues = res.predict()
    v = var_func(fittedvalues)
    std = np.sqrt(v)
    res_ols1 = OLS(res.resid_response / std, np.column_stack((dm_incl, dm_excl)) / std[:, None]).fit()
    c1 = res_ols1.ess
    pval1 = stats.chi2.sf(c1, k_constraint)
    c2 = nobs * res_ols1.rsquared
    pval2 = stats.chi2.sf(c2, k_constraint)
    from statsmodels.stats.multivariate_tools import partial_project
    pp = partial_project(dm_excl / std[:, None], dm_incl / std[:, None])
    resid_p = res.resid_response / std
    res_ols3 = OLS(np.ones(nobs), pp.resid * resid_p[:, None]).fit()
    c3b = res_ols3.ess
    pval3 = stats.chi2.sf(c3b, k_constraint)
    tres = TestResults(c1=c1, pval1=pval1, c2=c2, pval2=pval2, c3=c3b, pval3=pval3)
    return tres

def cm_test_robust(resid, resid_deriv, instruments, weights=1):
    if False:
        for i in range(10):
            print('nop')
    'score/lagrange multiplier of Wooldridge\n\n    generic version of Wooldridge procedure for test of conditional moments\n\n    Limitation: This version allows only for one unconditional moment\n    restriction, i.e. resid is scalar for each observation.\n    Another limitation is that it assumes independent observations, no\n    correlation in residuals and weights cannot be replaced by cross-observation\n    whitening.\n\n    Parameters\n    ----------\n    resid : ndarray, (nobs, )\n        conditional moment restriction, E(r | x, params) = 0\n    resid_deriv : ndarray, (nobs, k_params)\n        derivative of conditional moment restriction with respect to parameters\n    instruments : ndarray, (nobs, k_instruments)\n        indicator variables of Wooldridge, multiplies the conditional momen\n        restriction\n    weights : ndarray\n        This is a weights function as used in WLS. The moment\n        restrictions are multiplied by weights. This corresponds to the\n        inverse of the variance in a heteroskedastic model.\n\n    Returns\n    -------\n    test_results : Results instance\n        ???  TODO\n\n    Notes\n    -----\n    This implements the auxiliary regression procedure of Wooldridge,\n    implemented based on procedure 2.1 in Wooldridge 1990.\n\n    Wooldridge allows for multivariate conditional moments (`resid`)\n    TODO: check dimensions for multivariate case for extension\n\n    References\n    ----------\n    Wooldridge\n    Wooldridge\n    and more Wooldridge\n\n    '
    nobs = resid.shape[0]
    from statsmodels.stats.multivariate_tools import partial_project
    w_sqrt = np.sqrt(weights)
    if np.size(weights) > 1:
        w_sqrt = w_sqrt[:, None]
    pp = partial_project(instruments * w_sqrt, resid_deriv * w_sqrt)
    mom_resid = pp.resid
    moms_test = mom_resid * resid[:, None] * w_sqrt
    k_constraint = moms_test.shape[1]
    cov = moms_test.T.dot(moms_test)
    diff = moms_test.sum(0)
    stat = diff.dot(np.linalg.solve(cov, diff))
    stat2 = OLS(np.ones(nobs), moms_test).fit().ess
    pval = stats.chi2.sf(stat, k_constraint)
    return (stat, pval, stat2)

def lm_robust(score, constraint_matrix, score_deriv_inv, cov_score, cov_params=None):
    if False:
        return 10
    'general formula for score/LM test\n\n    generalized score or lagrange multiplier test for implicit constraints\n\n    `r(params) = 0`, with gradient `R = d r / d params`\n\n    linear constraints are given by `R params - q = 0`\n\n    It is assumed that all arrays are evaluated at the constrained estimates.\n\n    Parameters\n    ----------\n    score : ndarray, 1-D\n        derivative of objective function at estimated parameters\n        of constrained model\n    constraint_matrix R : ndarray\n        Linear restriction matrix or Jacobian of nonlinear constraints\n    hessian_inv, Ainv : ndarray, symmetric, square\n        inverse of second derivative of objective function\n        TODO: could be OPG or any other estimator if information matrix\n        equality holds\n    cov_score B :  ndarray, symmetric, square\n        covariance matrix of the score. This is the inner part of a sandwich\n        estimator.\n    cov_params V :  ndarray, symmetric, square\n        covariance of full parameter vector evaluated at constrained parameter\n        estimate. This can be specified instead of cov_score B.\n\n    Returns\n    -------\n    lm_stat : float\n        score/lagrange multiplier statistic\n\n    Notes\n    -----\n\n    '
    (R, Ainv, B, V) = (constraint_matrix, score_deriv_inv, cov_score, cov_params)
    tmp = R.dot(Ainv)
    wscore = tmp.dot(score)
    if B is None and V is None:
        lm_stat = score.dot(Ainv.dot(score))
    else:
        if V is None:
            inner = tmp.dot(B).dot(tmp.T)
        else:
            inner = R.dot(V).dot(R.T)
        lm_stat = wscore.dot(np.linalg.solve(inner, wscore))
    return lm_stat

def lm_robust_subset(score, k_constraints, score_deriv_inv, cov_score):
    if False:
        while True:
            i = 10
    'general formula for score/LM test\n\n    generalized score or lagrange multiplier test for constraints on a subset\n    of parameters\n\n    `params_1 = value`, where params_1 is a subset of the unconstrained\n    parameter vector.\n\n    It is assumed that all arrays are evaluated at the constrained estimates.\n\n    Parameters\n    ----------\n    score : ndarray, 1-D\n        derivative of objective function at estimated parameters\n        of constrained model\n    k_constraint : int\n        number of constraints\n    score_deriv_inv : ndarray, symmetric, square\n        inverse of second derivative of objective function\n        TODO: could be OPG or any other estimator if information matrix\n        equality holds\n    cov_score B :  ndarray, symmetric, square\n        covariance matrix of the score. This is the inner part of a sandwich\n        estimator.\n    not cov_params V :  ndarray, symmetric, square\n        covariance of full parameter vector evaluated at constrained parameter\n        estimate. This can be specified instead of cov_score B.\n\n    Returns\n    -------\n    lm_stat : float\n        score/lagrange multiplier statistic\n    p-value : float\n        p-value of the LM test based on chisquare distribution\n\n    Notes\n    -----\n    The implementation is based on Boos 1992 section 4.1. The same derivation\n    is also in other articles and in text books.\n\n    '
    h_uu = score_deriv_inv[:-k_constraints, :-k_constraints]
    h_cu = score_deriv_inv[-k_constraints:, :-k_constraints]
    tmp_proj = h_cu.dot(np.linalg.inv(h_uu))
    tmp = np.column_stack((-tmp_proj, np.eye(k_constraints)))
    cov_score_constraints = tmp.dot(cov_score.dot(tmp.T))
    lm_stat = score.dot(np.linalg.solve(cov_score_constraints, score))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return (lm_stat, pval)

def lm_robust_subset_parts(score, k_constraints, score_deriv_uu, score_deriv_cu, cov_score_cc, cov_score_cu, cov_score_uu):
    if False:
        print('Hello World!')
    'robust generalized score tests on subset of parameters\n\n    This is the same as lm_robust_subset with arguments in parts of\n    partitioned matrices.\n    This can be useful, when we have the parts based on different estimation\n    procedures, i.e. when we do not have the full unconstrained model.\n\n    Calculates mainly the covariance of the constraint part of the score.\n\n    Parameters\n    ----------\n    score : ndarray, 1-D\n        derivative of objective function at estimated parameters\n        of constrained model. These is the score component for the restricted\n        part under hypothesis. The unconstrained part of the score is assumed\n        to be zero.\n    k_constraint : int\n        number of constraints\n    score_deriv_uu : ndarray, symmetric, square\n        first derivative of moment equation or second derivative of objective\n        function for the unconstrained part\n        TODO: could be OPG or any other estimator if information matrix\n        equality holds\n    score_deriv_cu : ndarray\n        first cross derivative of moment equation or second cross\n        derivative of objective function between.\n    cov_score_cc :  ndarray\n        covariance matrix of the score for the unconstrained part.\n        This is the inner part of a sandwich estimator.\n    cov_score_cu :  ndarray\n        covariance matrix of the score for the off-diagonal block, i.e.\n        covariance between constrained and unconstrained part.\n    cov_score_uu :  ndarray\n        covariance matrix of the score for the unconstrained part.\n\n    Returns\n    -------\n    lm_stat : float\n        score/lagrange multiplier statistic\n    p-value : float\n        p-value of the LM test based on chisquare distribution\n\n    Notes\n    -----\n    TODO: these function should just return the covariance of the score\n    instead of calculating the score/lm test.\n\n    Implementation similar to lm_robust_subset and is based on Boos 1992,\n    section 4.1 in the form attributed to Breslow (1990). It does not use the\n    computation attributed to Kent (1982) and Engle (1984).\n    '
    tmp_proj = np.linalg.solve(score_deriv_uu, score_deriv_cu.T).T
    tmp = tmp_proj.dot(cov_score_cu.T)
    cov = cov_score_cc - tmp
    cov -= tmp.T
    cov += tmp_proj.dot(cov_score_uu).dot(tmp_proj.T)
    lm_stat = score.dot(np.linalg.solve(cov, score))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return (lm_stat, pval)

def lm_robust_reparameterized(score, params_deriv, score_deriv, cov_score):
    if False:
        for i in range(10):
            print('nop')
    'robust generalized score test for transformed parameters\n\n    The parameters are given by a nonlinear transformation of the estimated\n    reduced parameters\n\n    `params = g(params_reduced)`  with jacobian `G = d g / d params_reduced`\n\n    score and other arrays are for full parameter space `params`\n\n    Parameters\n    ----------\n    score : ndarray, 1-D\n        derivative of objective function at estimated parameters\n        of constrained model\n    params_deriv : ndarray\n        Jacobian G of the parameter trasnformation\n    score_deriv : ndarray, symmetric, square\n        second derivative of objective function\n        TODO: could be OPG or any other estimator if information matrix\n        equality holds\n    cov_score B :  ndarray, symmetric, square\n        covariance matrix of the score. This is the inner part of a sandwich\n        estimator.\n\n    Returns\n    -------\n    lm_stat : float\n        score/lagrange multiplier statistic\n    p-value : float\n        p-value of the LM test based on chisquare distribution\n\n    Notes\n    -----\n    Boos 1992, section 4.3, expression for T_{GS} just before example 6\n    '
    (k_params, k_reduced) = params_deriv.shape
    k_constraints = k_params - k_reduced
    G = params_deriv
    tmp_c0 = np.linalg.pinv(G.T.dot(score_deriv.dot(G)))
    tmp_c1 = score_deriv.dot(G.dot(tmp_c0.dot(G.T)))
    tmp_c = np.eye(k_params) - tmp_c1
    cov = tmp_c.dot(cov_score.dot(tmp_c.T))
    lm_stat = score.dot(np.linalg.pinv(cov).dot(score))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return (lm_stat, pval)

def conditional_moment_test_generic(mom_test, mom_test_deriv, mom_incl, mom_incl_deriv, var_mom_all=None, cov_type='OPG', cov_kwds=None):
    if False:
        return 10
    'generic conditional moment test\n\n    This is mainly intended as internal function in support of diagnostic\n    and specification tests. It has no conversion and checking of correct\n    arguments.\n\n    Parameters\n    ----------\n    mom_test : ndarray, 2-D (nobs, k_constraints)\n        moment conditions that will be tested to be zero\n    mom_test_deriv : ndarray, 2-D, square (k_constraints, k_constraints)\n        derivative of moment conditions under test with respect to the\n        parameters of the model summed over observations.\n    mom_incl : ndarray, 2-D (nobs, k_params)\n        moment conditions that where use in estimation, assumed to be zero\n        This is score_obs in the case of (Q)MLE\n    mom_incl_deriv : ndarray, 2-D, square (k_params, k_params)\n        derivative of moment conditions of estimator summed over observations\n        This is the information matrix or Hessian in the case of (Q)MLE.\n    var_mom_all : None, or ndarray, 2-D, (k, k) with k = k_constraints + k_params\n        Expected product or variance of the joint (column_stacked) moment\n        conditions. The stacking should have the variance of the moment\n        conditions under test in the first k_constraint rows and columns.\n        If it is not None, then it will be estimated based on cov_type.\n        I think: This is the Hessian of the extended or alternative model\n        under full MLE and score test assuming information matrix identity\n        holds.\n\n    Returns\n    -------\n    results\n\n    Notes\n    -----\n    TODO: cov_type other than OPG is missing\n    initial implementation based on Cameron Trived countbook 1998 p.48, p.56\n\n    also included: mom_incl can be None if expected mom_test_deriv is zero.\n\n    References\n    ----------\n    Cameron and Trivedi 1998 count book\n    Wooldridge ???\n    Pagan and Vella 1989\n    '
    if cov_type != 'OPG':
        raise NotImplementedError
    k_constraints = mom_test.shape[1]
    if mom_incl is None:
        if var_mom_all is None:
            var_cm = mom_test.T.dot(mom_test)
        else:
            var_cm = var_mom_all
    else:
        if var_mom_all is None:
            mom_all = np.column_stack((mom_test, mom_incl))
            var_mom_all = mom_all.T.dot(mom_all)
        tmp = mom_test_deriv.dot(np.linalg.pinv(mom_incl_deriv))
        h = np.column_stack((np.eye(k_constraints), -tmp))
        var_cm = h.dot(var_mom_all.dot(h.T))
    var_cm_inv = np.linalg.pinv(var_cm)
    mom_test_sum = mom_test.sum(0)
    statistic = mom_test_sum.dot(var_cm_inv.dot(mom_test_sum))
    pval = stats.chi2.sf(statistic, k_constraints)
    se = np.sqrt(np.diag(var_cm))
    tvalues = mom_test_sum / se
    pvalues = stats.norm.sf(np.abs(tvalues))
    res = ResultsGeneric(var_cm=var_cm, stat_cmt=statistic, pval_cmt=pval, tvalues=tvalues, pvalues=pvalues)
    return res

def conditional_moment_test_regression(mom_test, mom_test_deriv=None, mom_incl=None, mom_incl_deriv=None, var_mom_all=None, demean=False, cov_type='OPG', cov_kwds=None):
    if False:
        for i in range(10):
            print('nop')
    'generic conditional moment test based artificial regression\n\n    this is very experimental, no options implemented yet\n\n    so far\n    OPG regression, or\n    artificial regression with Robust Wald test\n\n    The latter is (as far as I can see) the same as an overidentifying test\n    in GMM where the test statistic is the value of the GMM objective function\n    and it is assumed that parameters were estimated with optimial GMM, i.e.\n    the weight matrix equal to the expectation of the score variance.\n    '
    (nobs, k_constraints) = mom_test.shape
    endog = np.ones(nobs)
    if mom_incl is not None:
        ex = np.column_stack((mom_test, mom_incl))
    else:
        ex = mom_test
    if demean:
        ex -= ex.mean(0)
    if cov_type == 'OPG':
        res = OLS(endog, ex).fit()
        statistic = nobs * res.rsquared
        pval = stats.chi2.sf(statistic, k_constraints)
    else:
        res = OLS(endog, ex).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        tres = res.wald_test(np.eye(ex.shape[1]))
        statistic = tres.statistic
        pval = tres.pvalue
    return (statistic, pval)

class CMTNewey:
    """generic moment test for GMM

    This is a class to calculate and hold the various results

    This is based on Newey 1985 on GMM.
    Lemma 1:
    Theorem 1

    The main method is `chisquare` which returns the result of the
    conditional moment test.

    Warning: name of class and methods will likely be changed

    Parameters
    ----------
    moments : ndarray, 1-D
        moments that are tested to be zero. They do not need to be derived
        from a likelihood function.
    moments_deriv : ndarray
        derivative of the moment function with respect to the parameters that
        are estimated
    cov_moments : ndarray
        An estimate for the joint (expected) covariance of all moments. This
        can be a heteroscedasticity or correlation robust covariance estimate,
        i.e. the inner part of a sandwich covariance.
    weights : ndarray
        Weights used in the GMM estimation.
    transf_mt : ndarray
        This defines the test moments where `transf_mt` is the matrix that
        defines a Linear combination of moments that have expected value equal
        to zero under the Null hypothesis.

    Notes
    -----
    The one letter names in Newey 1985 are

    moments, g :
    cov_moments, V :
    moments_deriv, H :
    weights, W :
    transf_mt, L :
        linear transformation to get the test condition from the moments

    not used, add as argument to methods or __init__?
    K cov for misspecification
    or mispecification_deriv

    This follows the GMM version in Newey 1985a, not the MLE version in
    Newey 1985b. Newey uses the generalized information matrix equality in the
    MLE version Newey (1985b).

    Newey 1985b Lemma 1 does not impose correctly specified likelihood, but
    assumes it in the following. Lemma 1 in both articles are essentially the
    same assuming D = H' W.

    References
    ----------
    - Newey 1985a, Generalized Method of Moment specification testing,
      Journal of Econometrics
    - Newey 1985b, Maximum Likelihood Specification Testing and Conditional
      Moment Tests, Econometrica
    """

    def __init__(self, moments, cov_moments, moments_deriv, weights, transf_mt):
        if False:
            i = 10
            return i + 15
        self.moments = moments
        self.cov_moments = cov_moments
        self.moments_deriv = moments_deriv
        self.weights = weights
        self.transf_mt = transf_mt
        self.moments_constraint = transf_mt.dot(moments)
        self.htw = moments_deriv.T.dot(weights)
        self.k_moments = self.moments.shape[-1]
        self.k_constraints = self.transf_mt.shape[0]

    @cache_readonly
    def asy_transf_params(self):
        if False:
            print('Hello World!')
        moments_deriv = self.moments_deriv
        htw = self.htw
        res = np.linalg.solve(htw.dot(moments_deriv), htw)
        return -res

    @cache_readonly
    def project_w(self):
        if False:
            while True:
                i = 10
        moments_deriv = self.moments_deriv
        res = moments_deriv.dot(self.asy_transf_params)
        res += np.eye(res.shape[0])
        return res

    @cache_readonly
    def asy_transform_mom_constraints(self):
        if False:
            print('Hello World!')
        res = self.transf_mt.dot(self.project_w)
        return res

    @cache_readonly
    def asy_cov_moments(self):
        if False:
            while True:
                i = 10
        '\n\n        `sqrt(T) * g_T(b_0) asy N(K delta, V)`\n\n        mean is not implemented,\n        V is the same as cov_moments in __init__ argument\n        '
        return self.cov_moments

    @cache_readonly
    def cov_mom_constraints(self):
        if False:
            while True:
                i = 10
        transf = self.asy_transform_mom_constraints
        return transf.dot(self.asy_cov_moments).dot(transf.T)

    @cache_readonly
    def rank_cov_mom_constraints(self):
        if False:
            i = 10
            return i + 15
        return np.linalg.matrix_rank(self.cov_mom_constraints)

    def ztest(self):
        if False:
            while True:
                i = 10
        'statistic, p-value and degrees of freedom of separate moment test\n\n        currently two sided test only\n\n        TODO: This can use generic ztest/ttest features and return\n        ContrastResults\n        '
        diff = self.moments_constraint
        bse = np.sqrt(np.diag(self.cov_mom_constraints))
        stat = diff / bse
        pval = stats.norm.sf(np.abs(stat)) * 2
        return (stat, pval)

    @cache_readonly
    def chisquare(self):
        if False:
            i = 10
            return i + 15
        'statistic, p-value and degrees of freedom of joint moment test\n        '
        diff = self.moments_constraint
        cov = self.cov_mom_constraints
        stat = diff.T.dot(np.linalg.pinv(cov).dot(diff))
        df = self.rank_cov_mom_constraints
        pval = stats.chi2.sf(stat, df)
        return (stat, pval, df)

class CMTTauchen:
    """generic moment tests or conditional moment tests for Quasi-MLE

    This is a generic class based on Tauchen 1985

    The main method is `chisquare` which returns the result of the
    conditional moment test.

    Warning: name of class and of methods will likely be changed

    Parameters
    ----------
    score : ndarray, 1-D
        moment condition used in estimation, score of log-likelihood function
    score_deriv : ndarray
        derivative of score function with respect to the parameters that are
        estimated. This is the Hessian in quasi-maximum likelihood
    moments : ndarray, 1-D
        moments that are tested to be zero. They do not need to be derived
        from a likelihood function.
    moments_deriv : ndarray
        derivative of the moment function with respect to the parameters that
        are estimated
    cov_moments : ndarray
        An estimate for the joint (expected) covariance of score and test
        moments. This can be a heteroscedasticity or correlation robust
        covariance estimate, i.e. the inner part of a sandwich covariance.
    """

    def __init__(self, score, score_deriv, moments, moments_deriv, cov_moments):
        if False:
            i = 10
            return i + 15
        self.score = score
        self.score_deriv = score_deriv
        self.moments = moments
        self.moments_deriv = moments_deriv
        self.cov_moments_all = cov_moments
        self.k_moments_test = moments.shape[-1]
        self.k_params = score.shape[-1]
        self.k_moments_all = self.k_params + self.k_moments_test

    @cache_readonly
    def cov_params_all(self):
        if False:
            while True:
                i = 10
        m_deriv = np.zeros((self.k_moments_all, self.k_moments_all))
        m_deriv[:self.k_params, :self.k_params] = self.score_deriv
        m_deriv[self.k_params:, :self.k_params] = self.moments_deriv
        m_deriv[self.k_params:, self.k_params:] = np.eye(self.k_moments_test)
        m_deriv_inv = np.linalg.inv(m_deriv)
        cov = m_deriv_inv.dot(self.cov_moments_all.dot(m_deriv_inv.T))
        return cov

    @cache_readonly
    def cov_mom_constraints(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cov_params_all[self.k_params:, self.k_params:]

    @cache_readonly
    def rank_cov_mom_constraints(self):
        if False:
            i = 10
            return i + 15
        return np.linalg.matrix_rank(self.cov_mom_constraints)

    def ztest(self):
        if False:
            for i in range(10):
                print('nop')
        'statistic, p-value and degrees of freedom of separate moment test\n\n        currently two sided test only\n\n        TODO: This can use generic ztest/ttest features and return\n        ContrastResults\n        '
        diff = self.moments_constraint
        bse = np.sqrt(np.diag(self.cov_mom_constraints))
        stat = diff / bse
        pval = stats.norm.sf(np.abs(stat)) * 2
        return (stat, pval)

    @cache_readonly
    def chisquare(self):
        if False:
            return 10
        'statistic, p-value and degrees of freedom of joint moment test\n        '
        diff = self.moments
        cov = self.cov_mom_constraints
        stat = diff.T.dot(np.linalg.pinv(cov).dot(diff))
        df = self.rank_cov_mom_constraints
        pval = stats.chi2.sf(stat, df)
        return (stat, pval, df)