from statsmodels.regression.linear_model import OLS
import numpy as np

def _calc_nodewise_row(exog, idx, alpha):
    if False:
        i = 10
        return i + 15
    'calculates the nodewise_row values for the idxth variable, used to\n    estimate approx_inv_cov.\n\n    Parameters\n    ----------\n    exog : array_like\n        The weighted design matrix for the current partition.\n    idx : scalar\n        Index of the current variable.\n    alpha : scalar or array_like\n        The penalty weight.  If a scalar, the same penalty weight\n        applies to all variables in the model.  If a vector, it\n        must have the same length as `params`, and contains a\n        penalty weight for each coefficient.\n\n    Returns\n    -------\n    An array-like object of length p-1\n\n    Notes\n    -----\n\n    nodewise_row_i = arg min 1/(2n) ||exog_i - exog_-i gamma||_2^2\n                             + alpha ||gamma||_1\n    '
    p = exog.shape[1]
    ind = list(range(p))
    ind.pop(idx)
    if not np.isscalar(alpha):
        alpha = alpha[ind]
    tmod = OLS(exog[:, idx], exog[:, ind])
    nodewise_row = tmod.fit_regularized(alpha=alpha).params
    return nodewise_row

def _calc_nodewise_weight(exog, nodewise_row, idx, alpha):
    if False:
        return 10
    'calculates the nodewise_weightvalue for the idxth variable, used to\n    estimate approx_inv_cov.\n\n    Parameters\n    ----------\n    exog : array_like\n        The weighted design matrix for the current partition.\n    nodewise_row : array_like\n        The nodewise_row values for the current variable.\n    idx : scalar\n        Index of the current variable\n    alpha : scalar or array_like\n        The penalty weight.  If a scalar, the same penalty weight\n        applies to all variables in the model.  If a vector, it\n        must have the same length as `params`, and contains a\n        penalty weight for each coefficient.\n\n    Returns\n    -------\n    A scalar\n\n    Notes\n    -----\n\n    nodewise_weight_i = sqrt(1/n ||exog,i - exog_-i nodewise_row||_2^2\n                             + alpha ||nodewise_row||_1)\n    '
    (n, p) = exog.shape
    ind = list(range(p))
    ind.pop(idx)
    if not np.isscalar(alpha):
        alpha = alpha[ind]
    d = np.linalg.norm(exog[:, idx] - exog[:, ind].dot(nodewise_row)) ** 2
    d = np.sqrt(d / n + alpha * np.linalg.norm(nodewise_row, 1))
    return d

def _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l):
    if False:
        return 10
    'calculates the approximate inverse covariance matrix\n\n    Parameters\n    ----------\n    nodewise_row_l : list\n        A list of array-like object where each object corresponds to\n        the nodewise_row values for the corresponding variable, should\n        be length p.\n    nodewise_weight_l : list\n        A list of scalars where each scalar corresponds to the nodewise_weight\n        value for the corresponding variable, should be length p.\n\n    Returns\n    ------\n    An array-like object, p x p matrix\n\n    Notes\n    -----\n\n    nwr = nodewise_row\n    nww = nodewise_weight\n\n    approx_inv_cov_j = - 1 / nww_j [nwr_j,1,...,1,...nwr_j,p]\n    '
    p = len(nodewise_weight_l)
    approx_inv_cov = -np.eye(p)
    for idx in range(p):
        ind = list(range(p))
        ind.pop(idx)
        approx_inv_cov[idx, ind] = nodewise_row_l[idx]
    approx_inv_cov *= -1 / nodewise_weight_l[:, None] ** 2
    return approx_inv_cov

class RegularizedInvCovariance:
    """
    Class for estimating regularized inverse covariance with
    nodewise regression

    Parameters
    ----------
    exog : array_like
        A weighted design matrix for covariance

    Attributes
    ----------
    exog : array_like
        A weighted design matrix for covariance
    alpha : scalar
        Regularizing constant
    """

    def __init__(self, exog):
        if False:
            print('Hello World!')
        self.exog = exog

    def fit(self, alpha=0):
        if False:
            i = 10
            return i + 15
        'estimates the regularized inverse covariance using nodewise\n        regression\n\n        Parameters\n        ----------\n        alpha : scalar\n            Regularizing constant\n        '
        (n, p) = self.exog.shape
        nodewise_row_l = []
        nodewise_weight_l = []
        for idx in range(p):
            nodewise_row = _calc_nodewise_row(self.exog, idx, alpha)
            nodewise_row_l.append(nodewise_row)
            nodewise_weight = _calc_nodewise_weight(self.exog, nodewise_row, idx, alpha)
            nodewise_weight_l.append(nodewise_weight)
        nodewise_row_l = np.array(nodewise_row_l)
        nodewise_weight_l = np.array(nodewise_weight_l)
        approx_inv_cov = _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l)
        self._approx_inv_cov = approx_inv_cov

    def approx_inv_cov(self):
        if False:
            return 10
        return self._approx_inv_cov