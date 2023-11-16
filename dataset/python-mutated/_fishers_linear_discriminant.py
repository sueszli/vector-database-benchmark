import typing as tp
import numpy as np
from prml.linear._classifier import Classifier
from prml.rv.gaussian import Gaussian

class FishersLinearDiscriminant(Classifier):
    """Fisher's Linear discriminant model."""

    def __init__(self, w: tp.Optional[np.ndarray]=None, threshold: tp.Optional[float]=None):
        if False:
            i = 10
            return i + 15
        "Initialize fisher's linear discriminant model.\n\n        Parameters\n        ----------\n        w : tp.Optional[np.ndarray], optional\n            Initial parameter, by default None\n        threshold : tp.Optional[float], optional\n            Initial threshold, by default None\n        "
        self.w = w
        self.threshold = threshold

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if False:
            while True:
                i = 10
        'Estimate parameter given training dataset.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training dataset independent variable (N, D)\n        y_train : np.ndarray\n            training dataset dependent variable (N,)\n            binary 0 or 1\n        '
        x0 = x_train[y_train == 0]
        x1 = x_train[y_train == 1]
        m0 = np.mean(x0, axis=0)
        m1 = np.mean(x1, axis=0)
        cov_inclass = np.cov(x0, rowvar=False) + np.cov(x1, rowvar=False)
        self.w = np.linalg.solve(cov_inclass, m1 - m0)
        self.w /= np.linalg.norm(self.w).clip(min=1e-10)
        g0 = Gaussian()
        g0.fit(x0 @ self.w)
        g1 = Gaussian()
        g1.fit(x1 @ self.w)
        root = np.roots([g1.var - g0.var, 2 * (g0.var * g1.mu - g1.var * g0.mu), g1.var * g0.mu ** 2 - g0.var * g1.mu ** 2 - g1.var * g0.var * np.log(g1.var / g0.var)])
        if g0.mu < root[0] < g1.mu or g1.mu < root[0] < g0.mu:
            self.threshold = root[0]
        else:
            self.threshold = root[1]

    def transform(self, x: np.ndarray):
        if False:
            for i in range(10):
                print('nop')
        'Project data.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            independent variable (N, D)\n\n        Returns\n        -------\n        y : np.ndarray\n            projected data (N,)\n        '
        return x @ self.w

    def classify(self, x: np.ndarray):
        if False:
            while True:
                i = 10
        'Classify input data.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            independent variable to be classified (N, D)\n\n        Returns\n        -------\n        np.ndarray\n            binary class for each input (N,)\n        '
        return (x @ self.w > self.threshold).astype(np.int)