import numpy as np
from prml.linear._regression import Regression

class BayesianRegression(Regression):
    """Bayesian regression model.

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """

    def __init__(self, alpha: float=1.0, beta: float=1.0):
        if False:
            while True:
                i = 10
        'Initialize bayesian linear regression model.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            Precision parameter of the prior, by default 1.\n        beta : float, optional\n            Precision parameter of the likelihood, by default 1.\n        '
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        if False:
            print('Hello World!')
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim: int) -> tuple:
        if False:
            print('Hello World!')
        if self._is_prior_defined():
            return (self.w_mean, self.w_precision)
        else:
            return (np.zeros(ndim), self.alpha * np.eye(ndim))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if False:
            print('Hello World!')
        'Bayesian update of parameters given training dataset.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training data independent variable (N, n_features)\n        y_train :  np.ndarray\n            training data dependent variable\n        '
        (mean_prev, precision_prev) = self._get_prior(np.size(x_train, 1))
        w_precision = precision_prev + self.beta * x_train.T @ x_train
        w_mean = np.linalg.solve(w_precision, precision_prev @ mean_prev + self.beta * x_train.T @ y_train)
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, x: np.ndarray, return_std: bool=False, sample_size: int=None):
        if False:
            return 10
        'Return mean (and standard deviation) of predictive distribution.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            independent variable (N, n_features)\n        return_std : bool, optional\n            flag to return standard deviation (the default is False)\n        sample_size : int, optional\n            number of samples to draw from the predictive distribution\n            (the default is None, no sampling from the distribution)\n\n        Returns\n        -------\n        y : np.ndarray\n            mean of the predictive distribution (N,)\n        y_std : np.ndarray\n            standard deviation of the predictive distribution (N,)\n        y_sample : np.ndarray\n            samples from the predictive distribution (N, sample_size)\n        '
        if sample_size is not None:
            w_sample = np.random.multivariate_normal(self.w_mean, self.w_cov, size=sample_size)
            y_sample = x @ w_sample.T
            return y_sample
        y = x @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(x @ self.w_cov * x, axis=1)
            y_std = np.sqrt(y_var)
            return (y, y_std)
        return y