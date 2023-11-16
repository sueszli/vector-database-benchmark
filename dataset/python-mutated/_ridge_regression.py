import numpy as np
from prml.linear._regression import Regression

class RidgeRegression(Regression):
    """Ridge regression model.

    w* = argmin |t - X @ w| + alpha * |w|_2^2
    """

    def __init__(self, alpha: float=1.0):
        if False:
            i = 10
            return i + 15
        'Initialize ridge linear regression model.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            Coefficient of the prior term, by default 1.\n        '
        self.alpha = alpha

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if False:
            while True:
                i = 10
        'Maximum a posteriori estimation of parameter.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training data independent variable (N, D)\n        y_train : np.ndarray\n            training data dependent variable (N,)\n        '
        eye = np.eye(np.size(x_train, 1))
        self.w = np.linalg.solve(self.alpha * eye + x_train.T @ x_train, x_train.T @ y_train)

    def predict(self, x: np.ndarray):
        if False:
            return 10
        'Return prediction.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            samples to predict their output (N, D)\n\n        Returns\n        -------\n        np.ndarray\n            prediction of each input (N,)\n        '
        return x @ self.w