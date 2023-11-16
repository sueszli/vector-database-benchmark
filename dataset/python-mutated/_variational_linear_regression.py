import numpy as np
from prml.linear._regression import Regression

class VariationalLinearRegression(Regression):
    """Variational bayesian linear regression model.

    p(w,alpha|X,t)
    ~ q(w)q(alpha)
    = N(w|w_mean, w_var)Gamma(alpha|a,b)

    Attributes
    ----------
    a : float
        a parameter of variational posterior gamma distribution
    b : float
        another parameter of variational posterior gamma distribution
    w_mean : (n_features,) ndarray
        mean of variational posterior gaussian distribution
    w_var : (n_features, n_features) ndarray
        variance of variational posterior gaussian distribution
    n_iter : int
        number of iterations performed
    """

    def __init__(self, beta: float=1.0, a0: float=1.0, b0: float=1.0):
        if False:
            for i in range(10):
                print('nop')
        'Initialize variational linear regression model.\n\n        Parameters\n        ----------\n        beta : float\n            precision of observation noise\n        a0 : float\n            a parameter of prior gamma distribution\n            Gamma(alpha|a0,b0)\n        b0 : float\n            another parameter of prior gamma distribution\n            Gamma(alpha|a0,b0)\n        '
        self.beta = beta
        self.a0 = a0
        self.b0 = b0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, iter_max: int=100):
        if False:
            print('Hello World!')
        'Variational bayesian estimation of parameter.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training independent variable (N, D)\n        y_train : np.ndarray\n            training dependent variable (N,)\n        iter_max : int, optional\n            maximum number of iteration (the default is 100)\n        '
        xtx = x_train.T @ x_train
        d = np.size(x_train, 1)
        self.a = self.a0 + 0.5 * d
        self.b = self.b0
        eye = np.eye(d)
        for _ in range(iter_max):
            param = self.b
            self.w_var = np.linalg.inv(self.a * eye / self.b + self.beta * xtx)
            self.w_mean = self.beta * self.w_var @ x_train.T @ y_train
            self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            if np.allclose(self.b, param):
                break

    def predict(self, x: np.ndarray, return_std: bool=False):
        if False:
            print('Hello World!')
        'Return predictions.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            Input independent variable (N, D)\n        return_std : bool, optional\n            return standard deviation of predictive distribution if True\n            (the default is False)\n\n        Returns\n        -------\n        y :  np.ndarray\n            mean of predictive distribution (N,)\n        y_std : np.ndarray\n            standard deviation of predictive distribution (N,)\n        '
        y = x @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(x @ self.w_var * x, axis=1)
            y_std = np.sqrt(y_var)
            return (y, y_std)
        return y