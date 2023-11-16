import typing as tp
import numpy as np
from prml.linear._logistic_regression import LogisticRegression

class VariationalLogisticRegression(LogisticRegression):
    """Variational logistic regression model.

    Graphical Model
    ---------------

    ```txt
    *----------------*
    |                |               ****  alpha
    |     phi_n      |             **    **
    |       **       |            *        *
    |       **       |            *        *
    |       |        |             **    **
    |       |        |               ****
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       v        |                v
    |      ****      |               ****  w
    |    **    **    |             **    **
    |   *        *   |            *        *
    |   *        *<--|------------*        *
    |    **    **    |             **    **
    |  t_n ****      |               ****
    |             N  |
    *----------------*
    ```
    """

    def __init__(self, alpha: tp.Optional[float]=None, a0: float=1.0, b0: float=1.0):
        if False:
            return 10
        'Construct variational logistic regression model.\n\n        Parameters\n        ----------\n        alpha : tp.Optional[float]\n            precision parameter of the prior\n            if None, this is also the subject to estimate\n        a0 : float\n            a parameter of hyper prior Gamma dist.\n            Gamma(alpha|a0,b0)\n            if alpha is not None, this argument will be ignored\n        b0 : float\n            another parameter of hyper prior Gamma dist.\n            Gamma(alpha|a0,b0)\n            if alpha is not None, this argument will be ignored\n        '
        if alpha is not None:
            self.__alpha = alpha
        else:
            self.a0 = a0
            self.b0 = b0

    def fit(self, x_train: np.ndarray, t: np.ndarray, iter_max: int=1000):
        if False:
            for i in range(10):
                print('nop')
        'Variational bayesian estimation of the parameter.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training independent variable (N, D)\n        t : np.ndarray\n            training dependent variable (N,)\n        iter_max : int, optional\n            maximum number of iteration (the default is 1000)\n        '
        (n, d) = x_train.shape
        if hasattr(self, 'a0'):
            self.a = self.a0 + 0.5 * d
        xi = np.random.uniform(-1, 1, size=n)
        eye = np.eye(d)
        param = np.copy(xi)
        for _ in range(iter_max):
            lambda_ = np.tanh(xi) * 0.25 / xi
            self.w_var = np.linalg.inv(eye / self.alpha + 2 * (lambda_ * x_train.T) @ x_train)
            self.w_mean = self.w_var @ np.sum(x_train.T * (t - 0.5), axis=1)
            xi = np.sqrt(np.sum(x_train @ (self.w_var + self.w_mean * self.w_mean[:, None]) * x_train, axis=-1))
            if np.allclose(xi, param):
                break
            else:
                param = np.copy(xi)

    @property
    def alpha(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Return expectation of variational distribution of alpha.\n\n        Returns\n        -------\n        float\n            Expectation of variational distribution of alpha.\n        '
        if hasattr(self, '__alpha'):
            return self.__alpha
        else:
            try:
                self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            except AttributeError:
                self.b = self.b0
            return self.a / self.b

    def proba(self, x: np.ndarray):
        if False:
            i = 10
            return i + 15
        'Return probability of input belonging class 1.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            Input independent variable (N, D)\n\n        Returns\n        -------\n        np.ndarray\n            probability of positive (N,)\n        '
        mu_a = x @ self.w_mean
        var_a = np.sum(x @ self.w_var * x, axis=1)
        y = self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        return y