"""

Accelerated Failure Time (AFT) Model with empirical likelihood inference.

AFT regression analysis is applicable when the researcher has access
to a randomly right censored dependent variable, a matrix of exogenous
variables and an indicatior variable (delta) that takes a value of 0 if the
observation is censored and 1 otherwise.

AFT References
--------------

Stute, W. (1993). "Consistent Estimation Under Random Censorship when
Covariables are Present." Journal of Multivariate Analysis.
Vol. 45. Iss. 1. 89-103

EL and AFT References
---------------------

Zhou, Kim And Bathke. "Empirical Likelihood Analysis for the Heteroskedastic
Accelerated Failure Time Model." Manuscript:
URL: www.ms.uky.edu/~mai/research/CasewiseEL20080724.pdf

Zhou, M. (2005). Empirical Likelihood Ratio with Arbitrarily Censored/
Truncated Data by EM Algorithm.  Journal of Computational and Graphical
Statistics. 14:3, 643-656.


"""
import warnings
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from .descriptive import _OptFuncts

class OptAFT(_OptFuncts):
    """
    Provides optimization functions used in estimating and conducting
    inference in an AFT model.

    Methods
    ------

    _opt_wtd_nuis_regress:
        Function optimized over nuisance parameters to compute
        the profile likelihood

    _EM_test:
        Uses the modified Em algorithm of Zhou 2005 to maximize the
        likelihood of a parameter vector.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def _opt_wtd_nuis_regress(self, test_vals):
        if False:
            for i in range(10):
                print('nop')
        '\n        A function that is optimized over nuisance parameters to conduct a\n        hypothesis test for the parameters of interest\n\n        Parameters\n        ----------\n\n        params: 1d array\n            The regression coefficients of the model.  This includes the\n            nuisance and parameters of interests.\n\n        Returns\n        -------\n        llr : float\n            -2 times the log likelihood of the nuisance parameters and the\n            hypothesized value of the parameter(s) of interest.\n        '
        test_params = test_vals.reshape(self.model.nvar, 1)
        est_vect = self.model.uncens_exog * (self.model.uncens_endog - np.dot(self.model.uncens_exog, test_params))
        eta_star = self._modif_newton(np.zeros(self.model.nvar), est_vect, self.model._fit_weights)
        denom = np.sum(self.model._fit_weights) + np.dot(eta_star, est_vect.T)
        self.new_weights = self.model._fit_weights / denom
        return -1 * np.sum(np.log(self.new_weights))

    def _EM_test(self, nuisance_params, params=None, param_nums=None, b0_vals=None, F=None, survidx=None, uncens_nobs=None, numcensbelow=None, km=None, uncensored=None, censored=None, maxiter=None, ftol=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Uses EM algorithm to compute the maximum likelihood of a test\n\n        Parameters\n        ----------\n\n        nuisance_params : ndarray\n            Vector of values to be used as nuisance params.\n\n        maxiter : int\n            Number of iterations in the EM algorithm for a parameter vector\n\n        Returns\n        -------\n        -2 ''*'' log likelihood ratio at hypothesized values and\n        nuisance params\n\n        Notes\n        -----\n        Optional parameters are provided by the test_beta function.\n        "
        iters = 0
        params[param_nums] = b0_vals
        nuis_param_index = np.int_(np.delete(np.arange(self.model.nvar), param_nums))
        params[nuis_param_index] = nuisance_params
        to_test = params.reshape(self.model.nvar, 1)
        opt_res = np.inf
        diff = np.inf
        while iters < maxiter and diff > ftol:
            F = F.flatten()
            death = np.cumsum(F[::-1])
            survivalprob = death[::-1]
            surv_point_mat = np.dot(F.reshape(-1, 1), 1.0 / survivalprob[survidx].reshape(1, -1))
            surv_point_mat = add_constant(surv_point_mat)
            summed_wts = np.cumsum(surv_point_mat, axis=1)
            wts = summed_wts[np.int_(np.arange(uncens_nobs)), numcensbelow[uncensored]]
            self.model._fit_weights = wts
            new_opt_res = self._opt_wtd_nuis_regress(to_test)
            F = self.new_weights
            diff = np.abs(new_opt_res - opt_res)
            opt_res = new_opt_res
            iters = iters + 1
        death = np.cumsum(F.flatten()[::-1])
        survivalprob = death[::-1]
        llike = -opt_res + np.sum(np.log(survivalprob[survidx]))
        wtd_km = km.flatten() / np.sum(km)
        survivalmax = np.cumsum(wtd_km[::-1])[::-1]
        llikemax = np.sum(np.log(wtd_km[uncensored])) + np.sum(np.log(survivalmax[censored]))
        if iters == maxiter:
            warnings.warn('The EM reached the maximum number of iterations', IterationLimitWarning)
        return -2 * (llike - llikemax)

    def _ci_limits_beta(self, b0, param_num=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns the difference between the log likelihood for a\n        parameter and some critical value.\n\n        Parameters\n        ----------\n        b0: float\n            Value of a regression parameter\n        param_num : int\n            Parameter index of b0\n        '
        return self.test_beta([b0], [param_num])[0] - self.r0

class emplikeAFT:
    """

    Class for estimating and conducting inference in an AFT model.

    Parameters
    ----------

    endog: nx1 array
        Response variables that are subject to random censoring

    exog: nxk array
        Matrix of covariates

    censors: nx1 array
        array with entries 0 or 1.  0 indicates a response was
        censored.

    Attributes
    ----------
    nobs : float
        Number of observations
    endog : ndarray
        Endog attay
    exog : ndarray
        Exogenous variable matrix
    censors
        Censors array but sets the max(endog) to uncensored
    nvar : float
        Number of exogenous variables
    uncens_nobs : float
        Number of uncensored observations
    uncens_endog : ndarray
        Uncensored response variables
    uncens_exog : ndarray
        Exogenous variables of the uncensored observations

    Methods
    -------

    params:
        Fits model parameters

    test_beta:
        Tests if beta = b0 for any vector b0.

    Notes
    -----

    The data is immediately sorted in order of increasing endogenous
    variables

    The last observation is assumed to be uncensored which makes
    estimation and inference possible.
    """

    def __init__(self, endog, exog, censors):
        if False:
            print('Hello World!')
        self.nobs = np.shape(exog)[0]
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog.reshape(self.nobs, -1)
        self.censors = np.asarray(censors).reshape(self.nobs, 1)
        self.nvar = self.exog.shape[1]
        idx = np.lexsort((-self.censors[:, 0], self.endog[:, 0]))
        self.endog = self.endog[idx]
        self.exog = self.exog[idx]
        self.censors = self.censors[idx]
        self.censors[-1] = 1
        self.uncens_nobs = int(np.sum(self.censors))
        mask = self.censors.ravel().astype(bool)
        self.uncens_endog = self.endog[mask, :].reshape(-1, 1)
        self.uncens_exog = self.exog[mask, :]

    def _is_tied(self, endog, censors):
        if False:
            while True:
                i = 10
        '\n        Indicated if an observation takes the same value as the next\n        ordered observation.\n\n        Parameters\n        ----------\n        endog : ndarray\n            Models endogenous variable\n        censors : ndarray\n            arrat indicating a censored array\n\n        Returns\n        -------\n        indic_ties : ndarray\n            ties[i]=1 if endog[i]==endog[i+1] and\n            censors[i]=censors[i+1]\n        '
        nobs = int(self.nobs)
        endog_idx = endog[np.arange(nobs - 1)] == endog[np.arange(nobs - 1) + 1]
        censors_idx = censors[np.arange(nobs - 1)] == censors[np.arange(nobs - 1) + 1]
        indic_ties = endog_idx * censors_idx
        return np.int_(indic_ties)

    def _km_w_ties(self, tie_indic, untied_km):
        if False:
            for i in range(10):
                print('nop')
        "\n        Computes KM estimator value at each observation, taking into acocunt\n        ties in the data.\n\n        Parameters\n        ----------\n        tie_indic: 1d array\n            Indicates if the i'th observation is the same as the ith +1\n        untied_km: 1d array\n            Km estimates at each observation assuming no ties.\n        "
        num_same = 1
        idx_nums = []
        for obs_num in np.arange(int(self.nobs - 1))[::-1]:
            if tie_indic[obs_num] == 1:
                idx_nums.append(obs_num)
                num_same = num_same + 1
                untied_km[obs_num] = untied_km[obs_num + 1]
            elif tie_indic[obs_num] == 0 and num_same > 1:
                idx_nums.append(max(idx_nums) + 1)
                idx_nums = np.asarray(idx_nums)
                untied_km[idx_nums] = untied_km[idx_nums]
                num_same = 1
                idx_nums = []
        return untied_km.reshape(self.nobs, 1)

    def _make_km(self, endog, censors):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Computes the Kaplan-Meier estimate for the weights in the AFT model\n\n        Parameters\n        ----------\n        endog: nx1 array\n            Array of response variables\n        censors: nx1 array\n            Censor-indicating variable\n\n        Returns\n        -------\n        Kaplan Meier estimate for each observation\n\n        Notes\n        -----\n\n        This function makes calls to _is_tied and km_w_ties to handle ties in\n        the data.If a censored observation and an uncensored observation has\n        the same value, it is assumed that the uncensored happened first.\n        '
        nobs = self.nobs
        num = nobs - (np.arange(nobs) + 1.0)
        denom = nobs - (np.arange(nobs) + 1.0) + 1.0
        km = (num / denom).reshape(nobs, 1)
        km = km ** np.abs(censors - 1.0)
        km = np.cumprod(km)
        tied = self._is_tied(endog, censors)
        wtd_km = self._km_w_ties(tied, km)
        return (censors / wtd_km).reshape(nobs, 1)

    def fit(self):
        if False:
            while True:
                i = 10
        '\n\n        Fits an AFT model and returns results instance\n\n        Parameters\n        ----------\n        None\n\n\n        Returns\n        -------\n        Results instance.\n\n        Notes\n        -----\n        To avoid dividing by zero, max(endog) is assumed to be uncensored.\n        '
        return AFTResults(self)

    def predict(self, params, endog=None):
        if False:
            return 10
        if endog is None:
            endog = self.endog
        return np.dot(endog, params)

class AFTResults(OptAFT):

    def __init__(self, model):
        if False:
            i = 10
            return i + 15
        self.model = model

    def params(self):
        if False:
            print('Hello World!')
        '\n\n        Fits an AFT model and returns parameters.\n\n        Parameters\n        ----------\n        None\n\n\n        Returns\n        -------\n        Fitted params\n\n        Notes\n        -----\n        To avoid dividing by zero, max(endog) is assumed to be uncensored.\n        '
        self.model.modif_censors = np.copy(self.model.censors)
        self.model.modif_censors[-1] = 1
        wts = self.model._make_km(self.model.endog, self.model.modif_censors)
        res = WLS(self.model.endog, self.model.exog, wts).fit()
        params = res.params
        return params

    def test_beta(self, b0_vals, param_nums, ftol=10 ** (-5), maxiter=30, print_weights=1):
        if False:
            return 10
        "\n        Returns the profile log likelihood for regression parameters\n        'param_num' at 'b0_vals.'\n\n        Parameters\n        ----------\n        b0_vals : list\n            The value of parameters to be tested\n        param_num : list\n            Which parameters to be tested\n        maxiter : int, optional\n            How many iterations to use in the EM algorithm.  Default is 30\n        ftol : float, optional\n            The function tolerance for the EM optimization.\n            Default is 10''**''-5\n        print_weights : bool\n            If true, returns the weights tate maximize the profile\n            log likelihood. Default is False\n\n        Returns\n        -------\n\n        test_results : tuple\n            The log-likelihood and p-pvalue of the test.\n\n        Notes\n        -----\n\n        The function will warn if the EM reaches the maxiter.  However, when\n        optimizing over nuisance parameters, it is possible to reach a\n        maximum number of inner iterations for a specific value for the\n        nuisance parameters while the resultsof the function are still valid.\n        This usually occurs when the optimization over the nuisance parameters\n        selects parameter values that yield a log-likihood ratio close to\n        infinity.\n\n        Examples\n        --------\n\n        >>> import statsmodels.api as sm\n        >>> import numpy as np\n\n        # Test parameter is .05 in one regressor no intercept model\n        >>> data=sm.datasets.heart.load()\n        >>> y = np.log10(data.endog)\n        >>> x = data.exog\n        >>> cens = data.censors\n        >>> model = sm.emplike.emplikeAFT(y, x, cens)\n        >>> res=model.test_beta([0], [0])\n        >>> res\n        (1.4657739632606308, 0.22601365256959183)\n\n        #Test slope is 0 in  model with intercept\n\n        >>> data=sm.datasets.heart.load()\n        >>> y = np.log10(data.endog)\n        >>> x = data.exog\n        >>> cens = data.censors\n        >>> model = sm.emplike.emplikeAFT(y, sm.add_constant(x), cens)\n        >>> res = model.test_beta([0], [1])\n        >>> res\n        (4.623487775078047, 0.031537049752572731)\n        "
        censors = self.model.censors
        endog = self.model.endog
        exog = self.model.exog
        uncensored = (censors == 1).flatten()
        censored = (censors == 0).flatten()
        uncens_endog = endog[uncensored]
        uncens_exog = exog[uncensored, :]
        reg_model = OLS(uncens_endog, uncens_exog).fit()
        (llr, pval, new_weights) = reg_model.el_test(b0_vals, param_nums, return_weights=True)
        km = self.model._make_km(endog, censors).flatten()
        uncens_nobs = self.model.uncens_nobs
        F = np.asarray(new_weights).reshape(uncens_nobs)
        params = self.params()
        survidx = np.where(censors == 0)
        survidx = survidx[0] - np.arange(len(survidx[0]))
        numcensbelow = np.int_(np.cumsum(1 - censors))
        if len(param_nums) == len(params):
            llr = self._EM_test([], F=F, params=params, param_nums=param_nums, b0_vals=b0_vals, survidx=survidx, uncens_nobs=uncens_nobs, numcensbelow=numcensbelow, km=km, uncensored=uncensored, censored=censored, ftol=ftol, maxiter=25)
            return (llr, chi2.sf(llr, self.model.nvar))
        else:
            x0 = np.delete(params, param_nums)
            try:
                res = optimize.fmin(self._EM_test, x0, (params, param_nums, b0_vals, F, survidx, uncens_nobs, numcensbelow, km, uncensored, censored, maxiter, ftol), full_output=1, disp=0)
                llr = res[1]
                return (llr, chi2.sf(llr, len(param_nums)))
            except np.linalg.linalg.LinAlgError:
                return (np.inf, 0)

    def ci_beta(self, param_num, beta_high, beta_low, sig=0.05):
        if False:
            i = 10
            return i + 15
        '\n        Returns the confidence interval for a regression\n        parameter in the AFT model.\n\n        Parameters\n        ----------\n        param_num : int\n            Parameter number of interest\n        beta_high : float\n            Upper bound for the confidence interval\n        beta_low : float\n            Lower bound for the confidence interval\n        sig : float, optional\n            Significance level.  Default is .05\n\n        Notes\n        -----\n        If the function returns f(a) and f(b) must have different signs,\n        consider widening the search area by adjusting beta_low and\n        beta_high.\n\n        Also note that this process is computational intensive.  There\n        are 4 levels of optimization/solving.  From outer to inner:\n\n        1) Solving so that llr-critical value = 0\n        2) maximizing over nuisance parameters\n        3) Using  EM at each value of nuisamce parameters\n        4) Using the _modified_Newton optimizer at each iteration\n           of the EM algorithm.\n\n        Also, for very unlikely nuisance parameters, it is possible for\n        the EM algorithm to not converge.  This is not an indicator\n        that the solver did not find the correct solution.  It just means\n        for a specific iteration of the nuisance parameters, the optimizer\n        was unable to converge.\n\n        If the user desires to verify the success of the optimization,\n        it is recommended to test the limits using test_beta.\n        '
        params = self.params()
        self.r0 = chi2.ppf(1 - sig, 1)
        ll = optimize.brentq(self._ci_limits_beta, beta_low, params[param_num], param_num)
        ul = optimize.brentq(self._ci_limits_beta, params[param_num], beta_high, param_num)
        return (ll, ul)