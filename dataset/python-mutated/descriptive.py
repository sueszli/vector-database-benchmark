"""
Empirical likelihood inference on descriptive statistics

This module conducts hypothesis tests and constructs confidence
intervals for the mean, variance, skewness, kurtosis and correlation.

If matplotlib is installed, this module can also generate multivariate
confidence region plots as well as mean-variance contour plots.

See _OptFuncts docstring for technical details and optimization variable
definitions.

General References:
------------------
Owen, A. (2001). "Empirical Likelihood." Chapman and Hall

"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils

def DescStat(endog):
    if False:
        while True:
            i = 10
    '\n    Returns an instance to conduct inference on descriptive statistics\n    via empirical likelihood.  See DescStatUV and DescStatMV for more\n    information.\n\n    Parameters\n    ----------\n    endog : ndarray\n         Array of data\n\n    Returns : DescStat instance\n        If k=1, the function returns a univariate instance, DescStatUV.\n        If k>1, the function returns a multivariate instance, DescStatMV.\n    '
    if endog.ndim == 1:
        endog = endog.reshape(len(endog), 1)
    if endog.shape[1] == 1:
        return DescStatUV(endog)
    if endog.shape[1] > 1:
        return DescStatMV(endog)

class _OptFuncts:
    """
    A class that holds functions that are optimized/solved.

    The general setup of the class is simple.  Any method that starts with
    _opt_ creates a vector of estimating equations named est_vect such that
    np.dot(p, (est_vect))=0 where p is the weight on each
    observation as a 1 x n array and est_vect is n x k.  Then _modif_Newton is
    called to determine the optimal p by solving for the Lagrange multiplier
    (eta) in the profile likelihood maximization problem.  In the presence
    of nuisance parameters, _opt_ functions are  optimized over to profile
    out the nuisance parameters.

    Any method starting with _ci_limits calculates the log likelihood
    ratio for a specific value of a parameter and then subtracts a
    pre-specified critical value.  This is solved so that llr - crit = 0.
    """

    def __init__(self, endog):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _log_star(self, eta, est_vect, weights, nobs):
        if False:
            i = 10
            return i + 15
        "\n        Transforms the log of observation probabilities in terms of the\n        Lagrange multiplier to the log 'star' of the probabilities.\n\n        Parameters\n        ----------\n        eta : float\n            Lagrange multiplier\n\n        est_vect : ndarray (n,k)\n            Estimating equations vector\n\n        wts : nx1 array\n            Observation weights\n\n        Returns\n        ------\n        data_star : ndarray\n            The weighted logstar of the estimting equations\n\n        Notes\n        -----\n        This function is only a placeholder for the _fit_Newton.\n        The function value is not used in optimization and the optimal value\n        is disregarded when computing the log likelihood ratio.\n        "
        data_star = np.log(weights) + (np.sum(weights) + np.dot(est_vect, eta))
        idx = data_star < 1.0 / nobs
        not_idx = ~idx
        nx = nobs * data_star[idx]
        data_star[idx] = np.log(1.0 / nobs) - 1.5 + nx * (2.0 - nx / 2)
        data_star[not_idx] = np.log(data_star[not_idx])
        return data_star

    def _hess(self, eta, est_vect, weights, nobs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the hessian of a weighted empirical likelihood\n        problem.\n\n        Parameters\n        ----------\n        eta : ndarray, (1,m)\n            Lagrange multiplier in the profile likelihood maximization\n\n        est_vect : ndarray (n,k)\n            Estimating equations vector\n\n        weights : 1darray\n            Observation weights\n\n        Returns\n        -------\n        hess : m x m array\n            Weighted hessian used in _wtd_modif_newton\n        '
        data_star_doub_prime = np.sum(weights) + np.dot(est_vect, eta)
        idx = data_star_doub_prime < 1.0 / nobs
        not_idx = ~idx
        data_star_doub_prime[idx] = -nobs ** 2
        data_star_doub_prime[not_idx] = -data_star_doub_prime[not_idx] ** (-2)
        wtd_dsdp = weights * data_star_doub_prime
        return np.dot(est_vect.T, wtd_dsdp[:, None] * est_vect)

    def _grad(self, eta, est_vect, weights, nobs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the gradient of a weighted empirical likelihood\n        problem\n\n        Parameters\n        ----------\n        eta : ndarray, (1,m)\n            Lagrange multiplier in the profile likelihood maximization\n\n        est_vect : ndarray, (n,k)\n            Estimating equations vector\n\n        weights : 1darray\n            Observation weights\n\n        Returns\n        -------\n        gradient : ndarray (m,1)\n            The gradient used in _wtd_modif_newton\n        '
        data_star_prime = np.sum(weights) + np.dot(est_vect, eta)
        idx = data_star_prime < 1.0 / nobs
        not_idx = ~idx
        data_star_prime[idx] = nobs * (2 - nobs * data_star_prime[idx])
        data_star_prime[not_idx] = 1.0 / data_star_prime[not_idx]
        return np.dot(weights * data_star_prime, est_vect)

    def _modif_newton(self, eta, est_vect, weights):
        if False:
            print('Hello World!')
        "\n        Modified Newton's method for maximizing the log 'star' equation.  This\n        function calls _fit_newton to find the optimal values of eta.\n\n        Parameters\n        ----------\n        eta : ndarray, (1,m)\n            Lagrange multiplier in the profile likelihood maximization\n\n        est_vect : ndarray, (n,k)\n            Estimating equations vector\n\n        weights : 1darray\n            Observation weights\n\n        Returns\n        -------\n        params : 1xm array\n            Lagrange multiplier that maximizes the log-likelihood\n        "
        nobs = len(est_vect)
        f = lambda x0: -np.sum(self._log_star(x0, est_vect, weights, nobs))
        grad = lambda x0: -self._grad(x0, est_vect, weights, nobs)
        hess = lambda x0: -self._hess(x0, est_vect, weights, nobs)
        kwds = {'tol': 1e-08}
        eta = eta.squeeze()
        res = _fit_newton(f, grad, eta, (), kwds, hess=hess, maxiter=50, disp=0)
        return res[0]

    def _find_eta(self, eta):
        if False:
            return 10
        '\n        Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for\n        eta when computing ELR for univariate mean.\n\n        Parameters\n        ----------\n        eta : float\n            Lagrange multiplier in the empirical likelihood maximization\n\n        Returns\n        -------\n        llr : float\n            n times the log likelihood value for a given value of eta\n        '
        return np.sum((self.endog - self.mu0) / (1.0 + eta * (self.endog - self.mu0)))

    def _ci_limits_mu(self, mu):
        if False:
            print('Hello World!')
        '\n        Calculates the difference between the log likelihood of mu_test and a\n        specified critical value.\n\n        Parameters\n        ----------\n        mu : float\n           Hypothesized value of the mean.\n\n        Returns\n        -------\n        diff : float\n            The difference between the log likelihood value of mu0 and\n            a specified value.\n        '
        return self.test_mean(mu)[0] - self.r0

    def _find_gamma(self, gamma):
        if False:
            i = 10
            return i + 15
        '\n        Finds gamma that satisfies\n        sum(log(n * w(gamma))) - log(r0) = 0\n\n        Used for confidence intervals for the mean\n\n        Parameters\n        ----------\n        gamma : float\n            Lagrange multiplier when computing confidence interval\n\n        Returns\n        -------\n        diff : float\n            The difference between the log-liklihood when the Lagrange\n            multiplier is gamma and a pre-specified value\n        '
        denom = np.sum((self.endog - gamma) ** (-1))
        new_weights = (self.endog - gamma) ** (-1) / denom
        return -2 * np.sum(np.log(self.nobs * new_weights)) - self.r0

    def _opt_var(self, nuisance_mu, pval=False):
        if False:
            print('Hello World!')
        '\n        This is the function to be optimized over a nuisance mean parameter\n        to determine the likelihood ratio for the variance\n\n        Parameters\n        ----------\n        nuisance_mu : float\n            Value of a nuisance mean parameter\n\n        Returns\n        -------\n        llr : float\n            Log likelihood of a pre-specified variance holding the nuisance\n            parameter constant\n        '
        endog = self.endog
        nobs = self.nobs
        sig_data = (endog - nuisance_mu) ** 2 - self.sig2_0
        mu_data = endog - nuisance_mu
        est_vect = np.column_stack((mu_data, sig_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        if pval:
            return chi2.sf(-2 * llr, 1)
        return -2 * llr

    def _ci_limits_var(self, var):
        if False:
            for i in range(10):
                print('nop')
        '\n        Used to determine the confidence intervals for the variance.\n        It calls test_var and when called by an optimizer,\n        finds the value of sig2_0 that is chi2.ppf(significance-level)\n\n        Parameters\n        ----------\n        var_test : float\n            Hypothesized value of the variance\n\n        Returns\n        -------\n        diff : float\n            The difference between the log likelihood ratio at var_test and a\n            pre-specified value.\n        '
        return self.test_var(var)[0] - self.r0

    def _opt_skew(self, nuis_params):
        if False:
            print('Hello World!')
        '\n        Called by test_skew.  This function is optimized over\n        nuisance parameters mu and sigma\n\n        Parameters\n        ----------\n        nuis_params : 1darray\n            An array with a  nuisance mean and variance parameter\n\n        Returns\n        -------\n        llr : float\n            The log likelihood ratio of a pre-specified skewness holding\n            the nuisance parameters constant.\n        '
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = (endog - nuis_params[0]) ** 2 - nuis_params[1]
        skew_data = (endog - nuis_params[0]) ** 3 / nuis_params[1] ** 1.5 - self.skew0
        est_vect = np.column_stack((mu_data, sig_data, skew_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1.0 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _opt_kurt(self, nuis_params):
        if False:
            while True:
                i = 10
        '\n        Called by test_kurt.  This function is optimized over\n        nuisance parameters mu and sigma\n\n        Parameters\n        ----------\n        nuis_params : 1darray\n            An array with a nuisance mean and variance parameter\n\n        Returns\n        -------\n        llr : float\n            The log likelihood ratio of a pre-speified kurtosis holding the\n            nuisance parameters constant\n        '
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = (endog - nuis_params[0]) ** 2 - nuis_params[1]
        kurt_data = (endog - nuis_params[0]) ** 4 / nuis_params[1] ** 2 - 3 - self.kurt0
        est_vect = np.column_stack((mu_data, sig_data, kurt_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _opt_skew_kurt(self, nuis_params):
        if False:
            i = 10
            return i + 15
        '\n        Called by test_joint_skew_kurt.  This function is optimized over\n        nuisance parameters mu and sigma\n\n        Parameters\n        ----------\n        nuis_params : 1darray\n            An array with a nuisance mean and variance parameter\n\n        Returns\n        ------\n        llr : float\n            The log likelihood ratio of a pre-speified skewness and\n            kurtosis holding the nuisance parameters constant.\n        '
        endog = self.endog
        nobs = self.nobs
        mu_data = endog - nuis_params[0]
        sig_data = (endog - nuis_params[0]) ** 2 - nuis_params[1]
        skew_data = (endog - nuis_params[0]) ** 3 / nuis_params[1] ** 1.5 - self.skew0
        kurt_data = (endog - nuis_params[0]) ** 4 / nuis_params[1] ** 2 - 3 - self.kurt0
        est_vect = np.column_stack((mu_data, sig_data, skew_data, kurt_data))
        eta_star = self._modif_newton(np.array([1.0 / nobs, 1.0 / nobs, 1.0 / nobs, 1.0 / nobs]), est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1.0 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_skew(self, skew):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        skew0 : float\n            Hypothesized value of skewness\n\n        Returns\n        -------\n        diff : float\n            The difference between the log likelihood ratio at skew and a\n            pre-specified value.\n        '
        return self.test_skew(skew)[0] - self.r0

    def _ci_limits_kurt(self, kurt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        skew0 : float\n            Hypothesized value of kurtosis\n\n        Returns\n        -------\n        diff : float\n            The difference between the log likelihood ratio at kurt and a\n            pre-specified value.\n        '
        return self.test_kurt(kurt)[0] - self.r0

    def _opt_correl(self, nuis_params, corr0, endog, nobs, x0, weights0):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        nuis_params : 1darray\n            Array containing two nuisance means and two nuisance variances\n\n        Returns\n        -------\n        llr : float\n            The log-likelihood of the correlation coefficient holding nuisance\n            parameters constant\n        '
        (mu1_data, mu2_data) = (endog - nuis_params[::2]).T
        sig1_data = mu1_data ** 2 - nuis_params[1]
        sig2_data = mu2_data ** 2 - nuis_params[3]
        correl_data = mu1_data * mu2_data - corr0 * (nuis_params[1] * nuis_params[3]) ** 0.5
        est_vect = np.column_stack((mu1_data, sig1_data, mu2_data, sig2_data, correl_data))
        eta_star = self._modif_newton(x0, est_vect, weights0)
        denom = 1.0 + np.dot(est_vect, eta_star)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_corr(self, corr):
        if False:
            i = 10
            return i + 15
        return self.test_corr(corr)[0] - self.r0

class DescStatUV(_OptFuncts):
    """
    A class to compute confidence intervals and hypothesis tests involving
    mean, variance, kurtosis and skewness of a univariate random variable.

    Parameters
    ----------
    endog : 1darray
        Data to be analyzed

    Attributes
    ----------
    endog : 1darray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        if False:
            print('Hello World!')
        self.endog = np.squeeze(endog)
        self.nobs = endog.shape[0]

    def test_mean(self, mu0, return_weights=False):
        if False:
            while True:
                i = 10
        '\n        Returns - 2 x log-likelihood ratio, p-value and weights\n        for a hypothesis test of the mean.\n\n        Parameters\n        ----------\n        mu0 : float\n            Mean value to be tested\n\n        return_weights : bool\n            If return_weights is True the function returns\n            the weights of the observations under the null hypothesis.\n            Default is False\n\n        Returns\n        -------\n        test_results : tuple\n            The log-likelihood ratio and p-value of mu0\n        '
        self.mu0 = mu0
        endog = self.endog
        nobs = self.nobs
        eta_min = (1.0 - 1.0 / nobs) / (self.mu0 - max(endog))
        eta_max = (1.0 - 1.0 / nobs) / (self.mu0 - min(endog))
        eta_star = optimize.brentq(self._find_eta, eta_min, eta_max)
        new_weights = 1.0 / nobs * 1.0 / (1.0 + eta_star * (endog - self.mu0))
        llr = -2 * np.sum(np.log(nobs * new_weights))
        if return_weights:
            return (llr, chi2.sf(llr, 1), new_weights)
        else:
            return (llr, chi2.sf(llr, 1))

    def ci_mean(self, sig=0.05, method='gamma', epsilon=10 ** (-8), gamma_low=-10 ** 10, gamma_high=10 ** 10):
        if False:
            i = 10
            return i + 15
        "\n        Returns the confidence interval for the mean.\n\n        Parameters\n        ----------\n        sig : float\n            significance level. Default is .05\n\n        method : str\n            Root finding method,  Can be 'nested-brent' or\n            'gamma'.  Default is 'gamma'\n\n            'gamma' Tries to solve for the gamma parameter in the\n            Lagrange (see Owen pg 22) and then determine the weights.\n\n            'nested brent' uses brents method to find the confidence\n            intervals but must maximize the likelihood ratio on every\n            iteration.\n\n            gamma is generally much faster.  If the optimizations does not\n            converge, try expanding the gamma_high and gamma_low\n            variable.\n\n        gamma_low : float\n            Lower bound for gamma when finding lower limit.\n            If function returns f(a) and f(b) must have different signs,\n            consider lowering gamma_low.\n\n        gamma_high : float\n            Upper bound for gamma when finding upper limit.\n            If function returns f(a) and f(b) must have different signs,\n            consider raising gamma_high.\n\n        epsilon : float\n            When using 'nested-brent', amount to decrease (increase)\n            from the maximum (minimum) of the data when\n            starting the search.  This is to protect against the\n            likelihood ratio being zero at the maximum (minimum)\n            value of the data.  If data is very small in absolute value\n            (<10 ``**`` -6) consider shrinking epsilon\n\n            When using 'gamma', amount to decrease (increase) the\n            minimum (maximum) by to start the search for gamma.\n            If function returns f(a) and f(b) must have different signs,\n            consider lowering epsilon.\n\n        Returns\n        -------\n        Interval : tuple\n            Confidence interval for the mean\n        "
        endog = self.endog
        sig = 1 - sig
        if method == 'nested-brent':
            self.r0 = chi2.ppf(sig, 1)
            middle = np.mean(endog)
            epsilon_u = (max(endog) - np.mean(endog)) * epsilon
            epsilon_l = (np.mean(endog) - min(endog)) * epsilon
            ulim = optimize.brentq(self._ci_limits_mu, middle, max(endog) - epsilon_u)
            llim = optimize.brentq(self._ci_limits_mu, middle, min(endog) + epsilon_l)
            return (llim, ulim)
        if method == 'gamma':
            self.r0 = chi2.ppf(sig, 1)
            gamma_star_l = optimize.brentq(self._find_gamma, gamma_low, min(endog) - epsilon)
            gamma_star_u = optimize.brentq(self._find_gamma, max(endog) + epsilon, gamma_high)
            weights_low = (endog - gamma_star_l) ** (-1) / np.sum((endog - gamma_star_l) ** (-1))
            weights_high = (endog - gamma_star_u) ** (-1) / np.sum((endog - gamma_star_u) ** (-1))
            mu_low = np.sum(weights_low * endog)
            mu_high = np.sum(weights_high * endog)
            return (mu_low, mu_high)

    def test_var(self, sig2_0, return_weights=False):
        if False:
            return 10
        '\n        Returns  -2 x log-likelihood ratio and the p-value for the\n        hypothesized variance\n\n        Parameters\n        ----------\n        sig2_0 : float\n            Hypothesized variance to be tested\n\n        return_weights : bool\n            If True, returns the weights that maximize the\n            likelihood of observing sig2_0. Default is False\n\n        Returns\n        -------\n        test_results : tuple\n            The  log-likelihood ratio and the p_value  of sig2_0\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> import statsmodels.api as sm\n        >>> random_numbers = np.random.standard_normal(1000)*100\n        >>> el_analysis = sm.emplike.DescStat(random_numbers)\n        >>> hyp_test = el_analysis.test_var(9500)\n        '
        self.sig2_0 = sig2_0
        mu_max = max(self.endog)
        mu_min = min(self.endog)
        llr = optimize.fminbound(self._opt_var, mu_min, mu_max, full_output=1)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        else:
            return (llr, p_val)

    def ci_var(self, lower_bound=None, upper_bound=None, sig=0.05):
        if False:
            i = 10
            return i + 15
        '\n        Returns the confidence interval for the variance.\n\n        Parameters\n        ----------\n        lower_bound : float\n            The minimum value the lower confidence interval can\n            take. The p-value from test_var(lower_bound) must be lower\n            than 1 - significance level. Default is .99 confidence\n            limit assuming normality\n\n        upper_bound : float\n            The maximum value the upper confidence interval\n            can take. The p-value from test_var(upper_bound) must be lower\n            than 1 - significance level.  Default is .99 confidence\n            limit assuming normality\n\n        sig : float\n            The significance level. Default is .05\n\n        Returns\n        -------\n        Interval : tuple\n            Confidence interval for the variance\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> import statsmodels.api as sm\n        >>> random_numbers = np.random.standard_normal(100)\n        >>> el_analysis = sm.emplike.DescStat(random_numbers)\n        >>> el_analysis.ci_var()\n        (0.7539322567470305, 1.229998852496268)\n        >>> el_analysis.ci_var(.5, 2)\n        (0.7539322567469926, 1.2299988524962664)\n\n        Notes\n        -----\n        If the function returns the error f(a) and f(b) must have\n        different signs, consider lowering lower_bound and raising\n        upper_bound.\n        '
        endog = self.endog
        if upper_bound is None:
            upper_bound = (self.nobs - 1) * endog.var() / chi2.ppf(0.0001, self.nobs - 1)
        if lower_bound is None:
            lower_bound = (self.nobs - 1) * endog.var() / chi2.ppf(0.9999, self.nobs - 1)
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_var, lower_bound, endog.var())
        ulim = optimize.brentq(self._ci_limits_var, endog.var(), upper_bound)
        return (llim, ulim)

    def plot_contour(self, mu_low, mu_high, var_low, var_high, mu_step, var_step, levs=[0.2, 0.1, 0.05, 0.01, 0.001]):
        if False:
            return 10
        '\n        Returns a plot of the confidence region for a univariate\n        mean and variance.\n\n        Parameters\n        ----------\n        mu_low : float\n            Lowest value of the mean to plot\n\n        mu_high : float\n            Highest value of the mean to plot\n\n        var_low : float\n            Lowest value of the variance to plot\n\n        var_high : float\n            Highest value of the variance to plot\n\n        mu_step : float\n            Increments to evaluate the mean\n\n        var_step : float\n            Increments to evaluate the mean\n\n        levs : list\n            Which values of significance the contour lines will be drawn.\n            Default is [.2, .1, .05, .01, .001]\n\n        Returns\n        -------\n        Figure\n            The contour plot\n        '
        (fig, ax) = utils.create_mpl_ax()
        ax.set_ylabel('Variance')
        ax.set_xlabel('Mean')
        mu_vect = list(np.arange(mu_low, mu_high, mu_step))
        var_vect = list(np.arange(var_low, var_high, var_step))
        z = []
        for sig0 in var_vect:
            self.sig2_0 = sig0
            for mu0 in mu_vect:
                z.append(self._opt_var(mu0, pval=True))
        z = np.asarray(z).reshape(len(var_vect), len(mu_vect))
        ax.contour(mu_vect, var_vect, z, levels=levs)
        return fig

    def test_skew(self, skew0, return_weights=False):
        if False:
            i = 10
            return i + 15
        '\n        Returns  -2 x log-likelihood and p-value for the hypothesized\n        skewness.\n\n        Parameters\n        ----------\n        skew0 : float\n            Skewness value to be tested\n\n        return_weights : bool\n            If True, function also returns the weights that\n            maximize the likelihood ratio. Default is False.\n\n        Returns\n        -------\n        test_results : tuple\n            The log-likelihood ratio and p_value of skew0\n        '
        self.skew0 = skew0
        start_nuisance = np.array([self.endog.mean(), self.endog.var()])
        llr = optimize.fmin_powell(self._opt_skew, start_nuisance, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def test_kurt(self, kurt0, return_weights=False):
        if False:
            while True:
                i = 10
        '\n        Returns -2 x log-likelihood and the p-value for the hypothesized\n        kurtosis.\n\n        Parameters\n        ----------\n        kurt0 : float\n            Kurtosis value to be tested\n\n        return_weights : bool\n            If True, function also returns the weights that\n            maximize the likelihood ratio. Default is False.\n\n        Returns\n        -------\n        test_results : tuple\n            The log-likelihood ratio and p-value of kurt0\n        '
        self.kurt0 = kurt0
        start_nuisance = np.array([self.endog.mean(), self.endog.var()])
        llr = optimize.fmin_powell(self._opt_kurt, start_nuisance, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def test_joint_skew_kurt(self, skew0, kurt0, return_weights=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns - 2 x log-likelihood and the p-value for the joint\n        hypothesis test for skewness and kurtosis\n\n        Parameters\n        ----------\n        skew0 : float\n            Skewness value to be tested\n        kurt0 : float\n            Kurtosis value to be tested\n\n        return_weights : bool\n            If True, function also returns the weights that\n            maximize the likelihood ratio. Default is False.\n\n        Returns\n        -------\n        test_results : tuple\n            The log-likelihood ratio and p-value  of the joint hypothesis test.\n        '
        self.skew0 = skew0
        self.kurt0 = kurt0
        start_nuisance = np.array([self.endog.mean(), self.endog.var()])
        llr = optimize.fmin_powell(self._opt_skew_kurt, start_nuisance, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 2)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def ci_skew(self, sig=0.05, upper_bound=None, lower_bound=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns the confidence interval for skewness.\n\n        Parameters\n        ----------\n        sig : float\n            The significance level.  Default is .05\n\n        upper_bound : float\n            Maximum value of skewness the upper limit can be.\n            Default is .99 confidence limit assuming normality.\n\n        lower_bound : float\n            Minimum value of skewness the lower limit can be.\n            Default is .99 confidence level assuming normality.\n\n        Returns\n        -------\n        Interval : tuple\n            Confidence interval for the skewness\n\n        Notes\n        -----\n        If function returns f(a) and f(b) must have different signs, consider\n        expanding lower and upper bounds\n        '
        nobs = self.nobs
        endog = self.endog
        if upper_bound is None:
            upper_bound = skew(endog) + 2.5 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5
        if lower_bound is None:
            lower_bound = skew(endog) - 2.5 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_skew, lower_bound, skew(endog))
        ulim = optimize.brentq(self._ci_limits_skew, skew(endog), upper_bound)
        return (llim, ulim)

    def ci_kurt(self, sig=0.05, upper_bound=None, lower_bound=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns the confidence interval for kurtosis.\n\n        Parameters\n        ----------\n\n        sig : float\n            The significance level.  Default is .05\n\n        upper_bound : float\n            Maximum value of kurtosis the upper limit can be.\n            Default is .99 confidence limit assuming normality.\n\n        lower_bound : float\n            Minimum value of kurtosis the lower limit can be.\n            Default is .99 confidence limit assuming normality.\n\n        Returns\n        -------\n        Interval : tuple\n            Lower and upper confidence limit\n\n        Notes\n        -----\n        For small n, upper_bound and lower_bound may have to be\n        provided by the user.  Consider using test_kurt to find\n        values close to the desired significance level.\n\n        If function returns f(a) and f(b) must have different signs, consider\n        expanding the bounds.\n        '
        endog = self.endog
        nobs = self.nobs
        if upper_bound is None:
            upper_bound = kurtosis(endog) + 2.5 * (2.0 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5) * ((nobs ** 2.0 - 1.0) / ((nobs - 3.0) * (nobs + 5.0))) ** 0.5
        if lower_bound is None:
            lower_bound = kurtosis(endog) - 2.5 * (2.0 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5) * ((nobs ** 2.0 - 1.0) / ((nobs - 3.0) * (nobs + 5.0))) ** 0.5
        self.r0 = chi2.ppf(1 - sig, 1)
        llim = optimize.brentq(self._ci_limits_kurt, lower_bound, kurtosis(endog))
        ulim = optimize.brentq(self._ci_limits_kurt, kurtosis(endog), upper_bound)
        return (llim, ulim)

class DescStatMV(_OptFuncts):
    """
    A class for conducting inference on multivariate means and correlation.

    Parameters
    ----------
    endog : ndarray
        Data to be analyzed

    Attributes
    ----------
    endog : ndarray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        if False:
            return 10
        self.endog = endog
        self.nobs = endog.shape[0]

    def mv_test_mean(self, mu_array, return_weights=False):
        if False:
            while True:
                i = 10
        '\n        Returns -2 x log likelihood and the p-value\n        for a multivariate hypothesis test of the mean\n\n        Parameters\n        ----------\n        mu_array  : 1d array\n            Hypothesized values for the mean.  Must have same number of\n            elements as columns in endog\n\n        return_weights : bool\n            If True, returns the weights that maximize the\n            likelihood of mu_array. Default is False.\n\n        Returns\n        -------\n        test_results : tuple\n            The log-likelihood ratio and p-value for mu_array\n        '
        endog = self.endog
        nobs = self.nobs
        if len(mu_array) != endog.shape[1]:
            raise ValueError('mu_array must have the same number of elements as the columns of the data.')
        mu_array = mu_array.reshape(1, endog.shape[1])
        means = np.ones((endog.shape[0], endog.shape[1]))
        means = mu_array * means
        est_vect = endog - means
        start_vals = 1.0 / nobs * np.ones(endog.shape[1])
        eta_star = self._modif_newton(start_vals, est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1 / nobs * 1 / denom
        llr = -2 * np.sum(np.log(nobs * self.new_weights))
        p_val = chi2.sf(llr, mu_array.shape[1])
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        else:
            return (llr, p_val)

    def mv_mean_contour(self, mu1_low, mu1_upp, mu2_low, mu2_upp, step1, step2, levs=(0.001, 0.01, 0.05, 0.1, 0.2), var1_name=None, var2_name=None, plot_dta=False):
        if False:
            print('Hello World!')
        '\n        Creates a confidence region plot for the mean of bivariate data\n\n        Parameters\n        ----------\n        m1_low : float\n            Minimum value of the mean for variable 1\n        m1_upp : float\n            Maximum value of the mean for variable 1\n        mu2_low : float\n            Minimum value of the mean for variable 2\n        mu2_upp : float\n            Maximum value of the mean for variable 2\n        step1 : float\n            Increment of evaluations for variable 1\n        step2 : float\n            Increment of evaluations for variable 2\n        levs : list\n            Levels to be drawn on the contour plot.\n            Default =  (.001, .01, .05, .1, .2)\n        plot_dta : bool\n            If True, makes a scatter plot of the data on\n            top of the contour plot. Defaultis False.\n        var1_name : str\n            Name of variable 1 to be plotted on the x-axis\n        var2_name : str\n            Name of variable 2 to be plotted on the y-axis\n\n        Notes\n        -----\n        The smaller the step size, the more accurate the intervals\n        will be\n\n        If the function returns optimization failed, consider narrowing\n        the boundaries of the plot\n\n        Examples\n        --------\n        >>> import statsmodels.api as sm\n        >>> two_rvs = np.random.standard_normal((20,2))\n        >>> el_analysis = sm.emplike.DescStat(two_rvs)\n        >>> contourp = el_analysis.mv_mean_contour(-2, 2, -2, 2, .1, .1)\n        >>> contourp.show()\n        '
        if self.endog.shape[1] != 2:
            raise ValueError('Data must contain exactly two variables')
        (fig, ax) = utils.create_mpl_ax()
        if var2_name is None:
            ax.set_ylabel('Variable 2')
        else:
            ax.set_ylabel(var2_name)
        if var1_name is None:
            ax.set_xlabel('Variable 1')
        else:
            ax.set_xlabel(var1_name)
        x = np.arange(mu1_low, mu1_upp, step1)
        y = np.arange(mu2_low, mu2_upp, step2)
        pairs = itertools.product(x, y)
        z = []
        for i in pairs:
            z.append(self.mv_test_mean(np.asarray(i))[0])
        (X, Y) = np.meshgrid(x, y)
        z = np.asarray(z)
        z = z.reshape(X.shape[1], Y.shape[0])
        ax.contour(x, y, z.T, levels=levs)
        if plot_dta:
            ax.plot(self.endog[:, 0], self.endog[:, 1], 'bo')
        return fig

    def test_corr(self, corr0, return_weights=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns -2 x log-likelihood ratio and  p-value for the\n        correlation coefficient between 2 variables\n\n        Parameters\n        ----------\n        corr0 : float\n            Hypothesized value to be tested\n\n        return_weights : bool\n            If true, returns the weights that maximize\n            the log-likelihood at the hypothesized value\n        '
        nobs = self.nobs
        endog = self.endog
        if endog.shape[1] != 2:
            raise NotImplementedError('Correlation matrix not yet implemented')
        nuis0 = np.array([endog[:, 0].mean(), endog[:, 0].var(), endog[:, 1].mean(), endog[:, 1].var()])
        x0 = np.zeros(5)
        weights0 = np.array([1.0 / nobs] * int(nobs))
        args = (corr0, endog, nobs, x0, weights0)
        llr = optimize.fmin(self._opt_correl, nuis0, args=args, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def ci_corr(self, sig=0.05, upper_bound=None, lower_bound=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the confidence intervals for the correlation coefficient\n\n        Parameters\n        ----------\n        sig : float\n            The significance level.  Default is .05\n\n        upper_bound : float\n            Maximum value the upper confidence limit can be.\n            Default is  99% confidence limit assuming normality.\n\n        lower_bound : float\n            Minimum value the lower confidence limit can be.\n            Default is 99% confidence limit assuming normality.\n\n        Returns\n        -------\n        interval : tuple\n            Confidence interval for the correlation\n        '
        endog = self.endog
        nobs = self.nobs
        self.r0 = chi2.ppf(1 - sig, 1)
        point_est = np.corrcoef(endog[:, 0], endog[:, 1])[0, 1]
        if upper_bound is None:
            upper_bound = min(0.999, point_est + 2.5 * ((1.0 - point_est ** 2.0) / (nobs - 2.0)) ** 0.5)
        if lower_bound is None:
            lower_bound = max(-0.999, point_est - 2.5 * np.sqrt((1.0 - point_est ** 2.0) / (nobs - 2.0)))
        llim = optimize.brenth(self._ci_limits_corr, lower_bound, point_est)
        ulim = optimize.brenth(self._ci_limits_corr, point_est, upper_bound)
        return (llim, ulim)