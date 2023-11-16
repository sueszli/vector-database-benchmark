"""Canonical correlation analysis

author: Yichuan Liu
"""
import numpy as np
from numpy.linalg import svd
import scipy
import pandas as pd
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from .multivariate_ols import multivariate_stats

class CanCorr(Model):
    """
    Canonical correlation analysis using singular value decomposition

    For matrices exog=x and endog=y, find projections x_cancoef and y_cancoef
    such that:

        x1 = x * x_cancoef, x1' * x1 is identity matrix
        y1 = y * y_cancoef, y1' * y1 is identity matrix

    and the correlation between x1 and y1 is maximized.

    Attributes
    ----------
    endog : ndarray
        See Parameters.
    exog : ndarray
        See Parameters.
    cancorr : ndarray
        The canonical correlation values
    y_cancoeff : ndarray
        The canonical coefficients for endog
    x_cancoeff : ndarray
        The canonical coefficients for exog

    References
    ----------
    .. [*] http://numerical.recipes/whp/notes/CanonCorrBySVD.pdf
    .. [*] http://www.csun.edu/~ata20315/psy524/docs/Psy524%20Lecture%208%20CC.pdf
    .. [*] http://www.mathematica-journal.com/2014/06/canonical-correlation-analysis/
    """

    def __init__(self, endog, exog, tolerance=1e-08, missing='none', hasconst=None, **kwargs):
        if False:
            return 10
        super(CanCorr, self).__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)
        self._fit(tolerance)

    def _fit(self, tolerance=1e-08):
        if False:
            while True:
                i = 10
        'Fit the model\n\n        A ValueError is raised if there are singular values smaller than the\n        tolerance. The treatment of singular arrays might change in future.\n\n        Parameters\n        ----------\n        tolerance : float\n            eigenvalue tolerance, values smaller than which is considered 0\n        '
        (nobs, k_yvar) = self.endog.shape
        (nobs, k_xvar) = self.exog.shape
        k = np.min([k_yvar, k_xvar])
        x = np.array(self.exog)
        x = x - x.mean(0)
        y = np.array(self.endog)
        y = y - y.mean(0)
        (ux, sx, vx) = svd(x, 0)
        vx_ds = vx.T
        mask = sx > tolerance
        if mask.sum() < len(mask):
            raise ValueError('exog is collinear.')
        vx_ds[:, mask] /= sx[mask]
        (uy, sy, vy) = svd(y, 0)
        vy_ds = vy.T
        mask = sy > tolerance
        if mask.sum() < len(mask):
            raise ValueError('endog is collinear.')
        vy_ds[:, mask] /= sy[mask]
        (u, s, v) = svd(ux.T.dot(uy), 0)
        self.cancorr = np.array([max(0, min(s[i], 1)) for i in range(len(s))])
        self.x_cancoef = vx_ds.dot(u[:, :k])
        self.y_cancoef = vy_ds.dot(v.T[:, :k])

    def corr_test(self):
        if False:
            i = 10
            return i + 15
        "Approximate F test\n        Perform multivariate statistical tests of the hypothesis that\n        there is no canonical correlation between endog and exog.\n        For each canonical correlation, testing its significance based on\n        Wilks' lambda.\n\n        Returns\n        -------\n        CanCorrTestResults instance\n        "
        (nobs, k_yvar) = self.endog.shape
        (nobs, k_xvar) = self.exog.shape
        eigenvals = np.power(self.cancorr, 2)
        stats = pd.DataFrame(columns=['Canonical Correlation', "Wilks' lambda", 'Num DF', 'Den DF', 'F Value', 'Pr > F'], index=list(range(len(eigenvals) - 1, -1, -1)))
        prod = 1
        for i in range(len(eigenvals) - 1, -1, -1):
            prod *= 1 - eigenvals[i]
            p = k_yvar - i
            q = k_xvar - i
            r = nobs - k_yvar - 1 - (p - q + 1) / 2
            u = (p * q - 2) / 4
            df1 = p * q
            if p ** 2 + q ** 2 - 5 > 0:
                t = np.sqrt(((p * q) ** 2 - 4) / (p ** 2 + q ** 2 - 5))
            else:
                t = 1
            df2 = r * t - 2 * u
            lmd = np.power(prod, 1 / t)
            F = (1 - lmd) / lmd * df2 / df1
            stats.loc[i, 'Canonical Correlation'] = self.cancorr[i]
            stats.loc[i, "Wilks' lambda"] = prod
            stats.loc[i, 'Num DF'] = df1
            stats.loc[i, 'Den DF'] = df2
            stats.loc[i, 'F Value'] = F
            pval = scipy.stats.f.sf(F, df1, df2)
            stats.loc[i, 'Pr > F'] = pval
            "\n            # Wilk's Chi square test of each canonical correlation\n            df = (p - i + 1) * (q - i + 1)\n            chi2 = a * np.log(prod)\n            pval = stats.chi2.sf(chi2, df)\n            stats.loc[i, 'Canonical correlation'] = self.cancorr[i]\n            stats.loc[i, 'Chi-square'] = chi2\n            stats.loc[i, 'DF'] = df\n            stats.loc[i, 'Pr > ChiSq'] = pval\n            "
        ind = stats.index.values[::-1]
        stats = stats.loc[ind, :]
        stats_mv = multivariate_stats(eigenvals, k_yvar, k_xvar, nobs - k_xvar - 1)
        return CanCorrTestResults(stats, stats_mv)

class CanCorrTestResults:
    """
    Canonical correlation results class

    Attributes
    ----------
    stats : DataFrame
        Contain statistical tests results for each canonical correlation
    stats_mv : DataFrame
        Contain the multivariate statistical tests results
    """

    def __init__(self, stats, stats_mv):
        if False:
            print('Hello World!')
        self.stats = stats
        self.stats_mv = stats_mv

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.summary().__str__()

    def summary(self):
        if False:
            while True:
                i = 10
        summ = summary2.Summary()
        summ.add_title('Cancorr results')
        summ.add_df(self.stats)
        summ.add_dict({'': ''})
        summ.add_dict({'Multivariate Statistics and F Approximations': ''})
        summ.add_df(self.stats_mv)
        return summ