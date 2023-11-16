"""
Created on Wed May 30 15:11:09 2018

@author: josef
"""
import numpy as np
from scipy import stats

def _lm_robust(score, constraint_matrix, score_deriv_inv, cov_score, cov_params=None):
    if False:
        i = 10
        return i + 15
    'general formula for score/LM test\n\n    generalized score or lagrange multiplier test for implicit constraints\n\n    `r(params) = 0`, with gradient `R = d r / d params`\n\n    linear constraints are given by `R params - q = 0`\n\n    It is assumed that all arrays are evaluated at the constrained estimates.\n\n\n    Parameters\n    ----------\n    score : ndarray, 1-D\n        derivative of objective function at estimated parameters\n        of constrained model\n    constraint_matrix R : ndarray\n        Linear restriction matrix or Jacobian of nonlinear constraints\n    score_deriv_inv, Ainv : ndarray, symmetric, square\n        inverse of second derivative of objective function\n        TODO: could be inverse of OPG or any other estimator if information\n        matrix equality holds\n    cov_score B :  ndarray, symmetric, square\n        covariance matrix of the score. This is the inner part of a sandwich\n        estimator.\n    cov_params V :  ndarray, symmetric, square\n        covariance of full parameter vector evaluated at constrained parameter\n        estimate. This can be specified instead of cov_score B.\n\n    Returns\n    -------\n    lm_stat : float\n        score/lagrange multiplier statistic\n    p-value : float\n        p-value of the LM test based on chisquare distribution\n\n    Notes\n    -----\n\n    '
    (R, Ainv, B, V) = (constraint_matrix, score_deriv_inv, cov_score, cov_params)
    k_constraints = np.linalg.matrix_rank(R)
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
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return (lm_stat, pval, k_constraints)

def score_test(self, exog_extra=None, params_constrained=None, hypothesis='joint', cov_type=None, cov_kwds=None, k_constraints=None, r_matrix=None, scale=None, observed=True):
    if False:
        i = 10
        return i + 15
    'score test for restrictions or for omitted variables\n\n    Null Hypothesis : constraints are satisfied\n\n    Alternative Hypothesis : at least one of the constraints does not hold\n\n    This allows to specify restricted and unrestricted model properties in\n    three different ways\n\n    - fit_constrained result: model contains score and hessian function for\n      the full, unrestricted model, but the parameter estimate in the results\n      instance is for the restricted model. This is the case if the model\n      was estimated with fit_constrained.\n    - restricted model with variable addition: If exog_extra is not None, then\n      it is assumed that the current model is a model with zero restrictions\n      and the unrestricted model is given by adding exog_extra as additional\n      explanatory variables.\n    - unrestricted model with restricted parameters explicitly provided. If\n      params_constrained is not None, then the model is assumed to be for the\n      unrestricted model, but the provided parameters are for the restricted\n      model.\n      TODO: This case will currently only work for `nonrobust` cov_type,\n      otherwise we will also need the restriction matrix provided by the user.\n\n\n    Parameters\n    ----------\n    exog_extra : None or array_like\n        Explanatory variables that are jointly tested for inclusion in the\n        model, i.e. omitted variables.\n    params_constrained : array_like\n        estimated parameter of the restricted model. This can be the\n        parameter estimate for the current when testing for omitted\n        variables.\n    hypothesis : str, \'joint\' (default) or \'separate\'\n        If hypothesis is \'joint\', then the chisquare test results for the\n        joint hypothesis that all constraints hold is returned.\n        If hypothesis is \'joint\', then z-test results for each constraint\n        is returned.\n        This is currently only implemented for cov_type="nonrobust".\n    cov_type : str\n        Warning: only partially implemented so far, currently only "nonrobust"\n        and "HC0" are supported.\n        If cov_type is None, then the cov_type specified in fit for the Wald\n        tests is used.\n        If the cov_type argument is not None, then it will be used instead of\n        the Wald cov_type given in fit.\n    k_constraints : int or None\n        Number of constraints that were used in the estimation of params\n        restricted relative to the number of exog in the model.\n        This must be provided if no exog_extra are given. If exog_extra is\n        not None, then k_constraints is assumed to be zero if it is None.\n    observed : bool\n        If True, then the observed Hessian is used in calculating the\n        covariance matrix of the score. If false then the expected\n        information matrix is used. This currently only applies to GLM where\n        EIM is available.\n        Warning: This option might still change.\n\n    Returns\n    -------\n    chi2_stat : float\n        chisquare statistic for the score test\n    p-value : float\n        P-value of the score test based on the chisquare distribution.\n    df : int\n        Degrees of freedom used in the p-value calculation. This is equal\n        to the number of constraints.\n\n    Notes\n    -----\n    Status: experimental, several options are not implemented yet or are not\n    verified yet. Currently available ptions might also still change.\n\n    cov_type is \'nonrobust\':\n\n    The covariance matrix for the score is based on the Hessian, i.e.\n    observed information matrix or optionally on the expected information\n    matrix.\n\n    cov_type is \'HC0\'\n\n    The covariance matrix of the score is the simple empirical covariance of\n    score_obs without degrees of freedom correction.\n    '
    if hasattr(self, '_results'):
        self = self._results
    model = self.model
    nobs = model.endog.shape[0]
    if params_constrained is None:
        params_constrained = self.params
    cov_type = cov_type if cov_type is not None else self.cov_type
    if observed is False:
        hess_kwd = {'observed': False}
    else:
        hess_kwd = {}
    if exog_extra is None:
        if hasattr(self, 'constraints'):
            if isinstance(self.constraints, tuple):
                r_matrix = self.constraints[0]
            else:
                r_matrix = self.constraints.coefs
            k_constraints = r_matrix.shape[0]
        elif k_constraints is None:
            raise ValueError('if exog_extra is None, then k_constraintsneeds to be given')
        if scale is not None:
            score_kwd = {'scale': scale}
            hess_kwd['scale'] = scale
        else:
            score_kwd = {}
        score = model.score(params_constrained, **score_kwd)
        score_obs = model.score_obs(params_constrained, **score_kwd)
        hessian = model.hessian(params_constrained, **hess_kwd)
    else:
        if cov_type == 'V':
            raise ValueError('if exog_extra is not None, then cov_type cannot be V')
        if hasattr(self, 'constraints'):
            raise NotImplementedError('if exog_extra is not None, then selfshould not be a constrained fit result')
        if isinstance(exog_extra, tuple):
            sh = _scorehess_extra(self, params_constrained, *exog_extra, hess_kwds=hess_kwd)
            (score_obs, hessian, k_constraints, r_matrix) = sh
            score = score_obs.sum(0)
        else:
            exog_extra = np.asarray(exog_extra)
            k_constraints = 0
            ex = np.column_stack((model.exog, exog_extra))
            k_constraints += ex.shape[1] - model.exog.shape[1]
            r_matrix = np.eye(len(self.params) + k_constraints)[-k_constraints:]
            score_factor = model.score_factor(params_constrained)
            if score_factor.ndim == 1:
                score_obs = score_factor[:, None] * ex
            else:
                sf = score_factor
                score_obs = np.column_stack((sf[:, :1] * ex, sf[:, 1:]))
            score = score_obs.sum(0)
            hessian_factor = model.hessian_factor(params_constrained, **hess_kwd)
            from statsmodels.genmod.generalized_linear_model import GLM
            if isinstance(model, GLM):
                hessian_factor *= -1
            hessian = np.dot(ex.T * hessian_factor, ex)
    if cov_type == 'nonrobust':
        cov_score_test = -hessian
    elif cov_type.upper() == 'HC0':
        hinv = -np.linalg.inv(hessian)
        cov_score = nobs * np.cov(score_obs.T)
        lm = _lm_robust(score, r_matrix, hinv, cov_score, cov_params=None)
        return lm
    elif cov_type.upper() == 'V':
        hinv = -np.linalg.inv(hessian)
        cov_score = nobs * np.cov(score_obs.T)
        V = self.cov_params_default
        chi2stat = _lm_robust(score, r_matrix, hinv, cov_score, cov_params=V)
        pval = stats.chi2.sf(chi2stat, k_constraints)
        return (chi2stat, pval)
    else:
        msg = 'Only cov_type "nonrobust" and "HC0" are available.'
        raise NotImplementedError(msg)
    if hypothesis == 'joint':
        chi2stat = score.dot(np.linalg.solve(cov_score_test, score[:, None]))
        pval = stats.chi2.sf(chi2stat, k_constraints)
        return (chi2stat, pval, k_constraints)
    elif hypothesis == 'separate':
        diff = score
        bse = np.sqrt(np.diag(cov_score_test))
        stat = diff / bse
        pval = stats.norm.sf(np.abs(stat)) * 2
        return (stat, pval)
    else:
        raise NotImplementedError('only hypothesis "joint" is available')

def _scorehess_extra(self, params=None, exog_extra=None, exog2_extra=None, hess_kwds=None):
    if False:
        while True:
            i = 10
    'Experimental helper function for variable addition score test.\n\n    This uses score and hessian factor at the params which should be the\n    params of the restricted model.\n\n    '
    if hess_kwds is None:
        hess_kwds = {}
    model = self.model
    if params is None:
        params = self.params
    (exog_o1, exog_o2) = model._get_exogs()
    if exog_o2 is None:
        exog_o2 = np.ones((exog_o1.shape[0], 1))
    k_mean = exog_o1.shape[1]
    k_prec = exog_o2.shape[1]
    if exog_extra is not None:
        exog = np.column_stack((exog_o1, exog_extra))
    else:
        exog = exog_o1
    if exog2_extra is not None:
        exog2 = np.column_stack((exog_o2, exog2_extra))
    else:
        exog2 = exog_o2
    k_mean_new = exog.shape[1]
    k_prec_new = exog2.shape[1]
    k_cm = k_mean_new - k_mean
    k_cp = k_prec_new - k_prec
    k_constraints = k_cm + k_cp
    index_mean = np.arange(k_mean, k_mean_new)
    index_prec = np.arange(k_mean_new + k_prec, k_mean_new + k_prec_new)
    r_matrix = np.zeros((k_constraints, len(params) + k_constraints))
    r_matrix[:k_cm, index_mean] = np.eye(k_cm)
    r_matrix[k_cm:k_cm + k_cp, index_prec] = np.eye(k_cp)
    if hasattr(model, 'score_hessian_factor'):
        (sf, hf) = model.score_hessian_factor(params, return_hessian=True, **hess_kwds)
    else:
        sf = model.score_factor(params)
        hf = model.hessian_factor(params, **hess_kwds)
    (sf1, sf2) = sf
    (hf11, hf12, hf22) = hf
    d1 = sf1[:, None] * exog
    d2 = sf2[:, None] * exog2
    score_obs = np.column_stack((d1, d2))
    d11 = (exog.T * hf11).dot(exog)
    d12 = (exog.T * hf12).dot(exog2)
    d22 = (exog2.T * hf22).dot(exog2)
    hessian = np.block([[d11, d12], [d12.T, d22]])
    return (score_obs, hessian, k_constraints, r_matrix)

def im_ratio(results):
    if False:
        for i in range(10):
            print('nop')
    res = getattr(results, '_results', results)
    hess = res.model.hessian(res.params)
    if res.cov_type == 'nonrobust':
        score_obs = res.model.score_obs(res.params)
        cov_score = score_obs.T @ score_obs
        hessneg_inv = np.linalg.inv(-hess)
        im_ratio = hessneg_inv @ cov_score
    else:
        im_ratio = res.cov_params() @ -hess
    return im_ratio

def tic(results):
    if False:
        i = 10
        return i + 15
    'Takeuchi information criterion for misspecified models\n\n    '
    imr = getattr(results, 'im_ratio', im_ratio(results))
    tic = -2 * results.llf + 2 * np.trace(imr)
    return tic

def gbic(results, gbicp=False):
    if False:
        print('Hello World!')
    'generalized BIC for misspecified models\n\n    References\n    ----------\n    Lv, Jinchi, and Jun S. Liu. 2014. "Model Selection Principles in\n    Misspecified Models." Journal of the Royal Statistical Society.\n    Series B (Statistical Methodology) 76 (1): 141–67.\n\n    '
    self = getattr(results, '_results', results)
    k_params = self.df_model + 1
    nobs = k_params + self.df_resid
    imr = getattr(results, 'im_ratio', im_ratio(results))
    imr_logdet = np.linalg.slogdet(imr)[1]
    gbic = -2 * self.llf + k_params * np.log(nobs) - imr_logdet
    gbicp = gbic + np.trace(imr)
    return (gbic, gbicp)