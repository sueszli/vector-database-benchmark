"""
Penalty classes for Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

"""
import numpy as np
from scipy.linalg import block_diag
from statsmodels.base._penalties import Penalty

class UnivariateGamPenalty(Penalty):
    """
    Penalty for smooth term in Generalized Additive Models

    Parameters
    ----------
    univariate_smoother : instance
        instance of univariate smoother or spline class
    alpha : float
        default penalty weight, alpha can be provided to each method
    weights:
        TODO: not used and verified, might be removed

    Attributes
    ----------
    Parameters are stored, additionally
    nob s: The number of samples used during the estimation
    n_columns : number of columns in smoother basis
    """

    def __init__(self, univariate_smoother, alpha=1, weights=1):
        if False:
            while True:
                i = 10
        self.weights = weights
        self.alpha = alpha
        self.univariate_smoother = univariate_smoother
        self.nobs = self.univariate_smoother.nobs
        self.n_columns = self.univariate_smoother.dim_basis

    def func(self, params, alpha=None):
        if False:
            for i in range(10):
                print('nop')
        'evaluate penalization at params\n\n        Parameters\n        ----------\n        params : ndarray\n            coefficients for the spline basis in the regression model\n        alpha : float\n            default penalty weight\n\n        Returns\n        -------\n        func : float\n            value of the penalty evaluated at params\n        '
        if alpha is None:
            alpha = self.alpha
        f = params.dot(self.univariate_smoother.cov_der2.dot(params))
        return alpha * f / self.nobs

    def deriv(self, params, alpha=None):
        if False:
            for i in range(10):
                print('nop')
        'evaluate derivative of penalty with respect to params\n\n        Parameters\n        ----------\n        params : ndarray\n            coefficients for the spline basis in the regression model\n        alpha : float\n            default penalty weight\n\n        Returns\n        -------\n        deriv : ndarray\n            derivative, gradient of the penalty with respect to params\n        '
        if alpha is None:
            alpha = self.alpha
        d = 2 * alpha * np.dot(self.univariate_smoother.cov_der2, params)
        d /= self.nobs
        return d

    def deriv2(self, params, alpha=None):
        if False:
            return 10
        'evaluate second derivative of penalty with respect to params\n\n        Parameters\n        ----------\n        params : ndarray\n            coefficients for the spline basis in the regression model\n        alpha : float\n            default penalty weight\n\n        Returns\n        -------\n        deriv2 : ndarray, 2-Dim\n            second derivative, hessian of the penalty with respect to params\n        '
        if alpha is None:
            alpha = self.alpha
        d2 = 2 * alpha * self.univariate_smoother.cov_der2
        d2 /= self.nobs
        return d2

    def penalty_matrix(self, alpha=None):
        if False:
            return 10
        'penalty matrix for the smooth term of a GAM\n\n        Parameters\n        ----------\n        alpha : list of floats or None\n            penalty weights\n\n        Returns\n        -------\n        penalty matrix\n            square penalty matrix for quadratic penalization. The number\n            of rows and columns are equal to the number of columns in the\n            smooth terms, i.e. the number of parameters for this smooth\n            term in the regression model\n        '
        if alpha is None:
            alpha = self.alpha
        return alpha * self.univariate_smoother.cov_der2

class MultivariateGamPenalty(Penalty):
    """
    Penalty for Generalized Additive Models

    Parameters
    ----------
    multivariate_smoother : instance
        instance of additive smoother or spline class
    alpha : list of float
        default penalty weight, list with length equal to the number of smooth
        terms. ``alpha`` can also be provided to each method.
    weights : array_like
        currently not used
        is a list of doubles of the same length as alpha or a list
        of ndarrays where each component has the length equal to the number
        of columns in that component
    start_idx : int
        number of parameters that come before the smooth terms. If the model
        has a linear component, then the parameters for the smooth components
        start at ``start_index``.

    Attributes
    ----------
    Parameters are stored, additionally
    nob s: The number of samples used during the estimation

    dim_basis : number of columns of additive smoother. Number of columns
        in all smoothers.
    k_variables : number of smooth terms
    k_params : total number of parameters in the regression model
    """

    def __init__(self, multivariate_smoother, alpha, weights=None, start_idx=0):
        if False:
            while True:
                i = 10
        if len(multivariate_smoother.smoothers) != len(alpha):
            msg = 'all the input values should be of the same length. len(smoothers)=%d, len(alphas)=%d' % (len(multivariate_smoother.smoothers), len(alpha))
            raise ValueError(msg)
        self.multivariate_smoother = multivariate_smoother
        self.dim_basis = self.multivariate_smoother.dim_basis
        self.k_variables = self.multivariate_smoother.k_variables
        self.nobs = self.multivariate_smoother.nobs
        self.alpha = alpha
        self.start_idx = start_idx
        self.k_params = start_idx + self.dim_basis
        if weights is None:
            self.weights = [1.0 for _ in range(self.k_variables)]
        else:
            import warnings
            warnings.warn('weights is currently ignored')
            self.weights = weights
        self.mask = [np.zeros(self.k_params, dtype=bool) for _ in range(self.k_variables)]
        param_count = start_idx
        for (i, smoother) in enumerate(self.multivariate_smoother.smoothers):
            self.mask[i][param_count:param_count + smoother.dim_basis] = True
            param_count += smoother.dim_basis
        self.gp = []
        for i in range(self.k_variables):
            gp = UnivariateGamPenalty(self.multivariate_smoother.smoothers[i], weights=self.weights[i], alpha=self.alpha[i])
            self.gp.append(gp)

    def func(self, params, alpha=None):
        if False:
            return 10
        'evaluate penalization at params\n\n        Parameters\n        ----------\n        params : ndarray\n            coefficients in the regression model\n        alpha : float or list of floats\n            penalty weights\n\n        Returns\n        -------\n        func : float\n            value of the penalty evaluated at params\n        '
        if alpha is None:
            alpha = [None] * self.k_variables
        cost = 0
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            cost += self.gp[i].func(params_i, alpha=alpha[i])
        return cost

    def deriv(self, params, alpha=None):
        if False:
            for i in range(10):
                print('nop')
        'evaluate derivative of penalty with respect to params\n\n        Parameters\n        ----------\n        params : ndarray\n            coefficients in the regression model\n        alpha : list of floats or None\n            penalty weights\n\n        Returns\n        -------\n        deriv : ndarray\n            derivative, gradient of the penalty with respect to params\n        '
        if alpha is None:
            alpha = [None] * self.k_variables
        grad = [np.zeros(self.start_idx)]
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            grad.append(self.gp[i].deriv(params_i, alpha=alpha[i]))
        return np.concatenate(grad)

    def deriv2(self, params, alpha=None):
        if False:
            for i in range(10):
                print('nop')
        'evaluate second derivative of penalty with respect to params\n\n        Parameters\n        ----------\n        params : ndarray\n            coefficients in the regression model\n        alpha : list of floats or None\n            penalty weights\n\n        Returns\n        -------\n        deriv2 : ndarray, 2-Dim\n            second derivative, hessian of the penalty with respect to params\n        '
        if alpha is None:
            alpha = [None] * self.k_variables
        deriv2 = [np.zeros((self.start_idx, self.start_idx))]
        for i in range(self.k_variables):
            params_i = params[self.mask[i]]
            deriv2.append(self.gp[i].deriv2(params_i, alpha=alpha[i]))
        return block_diag(*deriv2)

    def penalty_matrix(self, alpha=None):
        if False:
            print('Hello World!')
        'penalty matrix for generalized additive model\n\n        Parameters\n        ----------\n        alpha : list of floats or None\n            penalty weights\n\n        Returns\n        -------\n        penalty matrix\n            block diagonal, square penalty matrix for quadratic penalization.\n            The number of rows and columns are equal to the number of\n            parameters in the regression model ``k_params``.\n\n        Notes\n        -----\n        statsmodels does not support backwards compatibility when keywords are\n        used as positional arguments. The order of keywords might change.\n        We might need to add a ``params`` keyword if the need arises.\n        '
        if alpha is None:
            alpha = self.alpha
        s_all = [np.zeros((self.start_idx, self.start_idx))]
        for i in range(self.k_variables):
            s_all.append(self.gp[i].penalty_matrix(alpha=alpha[i]))
        return block_diag(*s_all)