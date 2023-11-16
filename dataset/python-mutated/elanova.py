"""
This script contains empirical likelihood ANOVA.

Currently the script only contains one feature that allows the user to compare
means of multiple groups.

General References
------------------

Owen, A. B. (2001). Empirical Likelihood. Chapman and Hall.
"""
import numpy as np
from .descriptive import _OptFuncts
from scipy import optimize
from scipy.stats import chi2

class _ANOVAOpt(_OptFuncts):
    """

    Class containing functions that are optimized over when
    conducting ANOVA.
    """

    def _opt_common_mu(self, mu):
        if False:
            i = 10
            return i + 15
        '\n        Optimizes the likelihood under the null hypothesis that all groups have\n        mean mu.\n\n        Parameters\n        ----------\n        mu : float\n            The common mean.\n\n        Returns\n        -------\n        llr : float\n            -2 times the llr ratio, which is the test statistic.\n        '
        nobs = self.nobs
        endog = self.endog
        num_groups = self.num_groups
        endog_asarray = np.zeros((nobs, num_groups))
        obs_num = 0
        for arr_num in range(len(endog)):
            new_obs_num = obs_num + len(endog[arr_num])
            endog_asarray[obs_num:new_obs_num, arr_num] = endog[arr_num] - mu
            obs_num = new_obs_num
        est_vect = endog_asarray
        wts = np.ones(est_vect.shape[0]) * (1.0 / est_vect.shape[0])
        eta_star = self._modif_newton(np.zeros(num_groups), est_vect, wts)
        denom = 1.0 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr

class ANOVA(_ANOVAOpt):
    """
    A class for ANOVA and comparing means.

    Parameters
    ----------

    endog : list of arrays
        endog should be a list containing 1 dimensional arrays.  Each array
        is the data collected from a certain group.
    """

    def __init__(self, endog):
        if False:
            for i in range(10):
                print('nop')
        self.endog = endog
        self.num_groups = len(self.endog)
        self.nobs = 0
        for i in self.endog:
            self.nobs = self.nobs + len(i)

    def compute_ANOVA(self, mu=None, mu_start=0, return_weights=0):
        if False:
            i = 10
            return i + 15
        '\n        Returns -2 log likelihood, the pvalue and the maximum likelihood\n        estimate for a common mean.\n\n        Parameters\n        ----------\n\n        mu : float\n            If a mu is specified, ANOVA is conducted with mu as the\n            common mean.  Otherwise, the common mean is the maximum\n            empirical likelihood estimate of the common mean.\n            Default is None.\n\n        mu_start : float\n            Starting value for commean mean if specific mu is not specified.\n            Default = 0.\n\n        return_weights : bool\n            if TRUE, returns the weights on observations that maximize the\n            likelihood.  Default is FALSE.\n\n        Returns\n        -------\n\n        res: tuple\n            The log-likelihood, p-value and estimate for the common mean.\n        '
        if mu is not None:
            llr = self._opt_common_mu(mu)
            pval = 1 - chi2.cdf(llr, self.num_groups - 1)
            if return_weights:
                return (llr, pval, mu, self.new_weights)
            else:
                return (llr, pval, mu)
        else:
            res = optimize.fmin_powell(self._opt_common_mu, mu_start, full_output=1, disp=False)
            llr = res[1]
            mu_common = float(np.squeeze(res[0]))
            pval = 1 - chi2.cdf(llr, self.num_groups - 1)
            if return_weights:
                return (llr, pval, mu_common, self.new_weights)
            else:
                return (llr, pval, mu_common)