from __future__ import annotations
import numpy as np
import pandas as pd
from river import base, proba, utils

class BayesianLinearRegression(base.Regressor):
    """Bayesian linear regression.

    An advantage of Bayesian linear regression over standard linear regression is that features
    do not have to scaled beforehand. Another attractive property is that this flavor of linear
    regression is somewhat insensitive to its hyperparameters. Finally, this model can output
    instead a predictive distribution rather than just a point estimate.

    The downside is that the learning step runs in `O(n^2)` time, whereas the learning step of
    standard linear regression takes `O(n)` time.

    Parameters
    ----------
    alpha
        Prior parameter.
    beta
        Noise parameter.
    smoothing
        Smoothing allows the model to gradually "forget" the past, and focus on the more recent
        data. It thus enables the model to deal with concept drift. Due to the current
        implementation, activating smoothing may slow down the model.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics

    >>> dataset = datasets.TrumpApproval()
    >>> model = linear_model.BayesianLinearRegression()
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.586...

    >>> x, _ = next(iter(dataset))
    >>> model.predict_one(x)
    43.852...

    >>> model.predict_one(x, with_dist=True)
    𝒩(μ=43.85..., σ=1.00...)

    The `smoothing` parameter can be set to make the model robust to drift. The parameter is
    expected to be between 0 and 1. To exemplify, let's generate some simulation data with an
    abrupt concept drift right in the middle.

    >>> import itertools
    >>> import random

    >>> def random_data(coefs, n, seed=42):
    ...     rng = random.Random(seed)
    ...     for _ in range(n):
    ...         x = {i: rng.random() for i, c in enumerate(coefs)}
    ...         y = sum(c * xi for c, xi in zip(coefs, x.values()))
    ...         yield x, y

    Here's how the model performs without any smoothing:

    >>> model = linear_model.BayesianLinearRegression()
    >>> dataset = itertools.chain(
    ...     random_data([0.1, 3], 100),
    ...     random_data([10, -2], 100)
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.284...

    And here's how it performs with some smoothing:

    >>> model = linear_model.BayesianLinearRegression(smoothing=0.8)
    >>> dataset = itertools.chain(
    ...     random_data([0.1, 3], 100),
    ...     random_data([10, -2], 100)
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.159...

    Smoothing allows the model to gradually "forget" the past, and focus on the more recent data.

    Note how this works better than standard linear regression, even when using an aggressive
    learning rate.

    >>> from river import optim
    >>> model = linear_model.LinearRegression(optimizer=optim.SGD(0.5))
    >>> dataset = itertools.chain(
    ...     random_data([0.1, 3], 100),
    ...     random_data([10, -2], 100)
    ... )
    >>> metric = metrics.MAE()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.242...

    References
    ----------
    [^1]: [Pattern Recognition and Machine Learning, page 52 — Christopher M. Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
    [^2]: [Bayesian/Streaming Algorithms — Vincent Warmerdam](https://koaning.io/posts/bayesian-propto-streaming/)
    [^3]: [Bayesian linear regression for practitioners — Max Halford](https://maxhalford.github.io/blog/bayesian-linear-regression/)

    """

    def __init__(self, alpha=1, beta=1, smoothing: float=None):
        if False:
            return 10
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing
        self._ss: dict[tuple[base.typing.FeatureName, base.typing.FeatureName], float] = {}
        self._ss_inv: dict[tuple[base.typing.FeatureName, base.typing.FeatureName], float] = {}
        self._m: dict[base.typing.FeatureName, float] = {}
        self._n = 1

    def _unit_test_skips(self):
        if False:
            return 10
        return {'check_shuffle_features_no_impact', 'check_emerging_features'}

    def _get_arrays(self, features, m=True, ss=True, ss_inv=True):
        if False:
            for i in range(10):
                print('nop')
        m_arr = np.array([self._m.get(i, 0.0) for i in features]) if m else None
        ss_arr = np.array([[self._ss.get(min((i, j), (j, i)), 1.0 / self.alpha if i == j else 0.0) for j in features] for i in features]) if ss else None
        ss_inv_arr = np.array([[self._ss_inv.get(min((i, j), (j, i)), 1.0 / self.alpha if i == j else 0.0) for j in features] for i in features], order='F') if ss_inv else None
        return (m_arr, ss_arr, ss_inv_arr)

    def _set_arrays(self, features, m_arr, ss_arr, ss_inv_arr):
        if False:
            while True:
                i = 10
        for (i, fi) in enumerate(features):
            self._m[fi] = m_arr[i]
            ss_row = ss_arr[i]
            ss_inv_row = ss_inv_arr[i]
            for (j, fj) in enumerate(features):
                self._ss[min((fi, fj), (fj, fi))] = ss_row[j]
                self._ss_inv[min((fi, fj), (fj, fi))] = ss_inv_row[j]

    def learn_one(self, x, y):
        if False:
            while True:
                i = 10
        x_arr = np.array(list(x.values()))
        (m_arr, ss_arr, ss_inv_arr) = self._get_arrays(x.keys())
        bx = self.beta * x_arr
        if self.smoothing is None:
            utils.math.sherman_morrison(A=ss_inv_arr, u=bx, v=x_arr)
            m_arr = ss_inv_arr @ (ss_arr @ m_arr + bx * y)
            ss_arr += np.outer(bx, x_arr)
        else:
            new_ss_arr = self.smoothing * ss_arr + (1 - self.smoothing) * np.outer(bx, x_arr)
            ss_inv_arr = np.linalg.inv(new_ss_arr)
            m_arr = ss_inv_arr @ (self.smoothing * ss_arr @ m_arr + (1 - self.smoothing) * bx * y)
            ss_arr = new_ss_arr
        self._set_arrays(x.keys(), m_arr, ss_arr, ss_inv_arr)
        return self

    def predict_one(self, x, with_dist=False):
        if False:
            print('Hello World!')
        'Predict the output of features `x`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n        with_dist\n            Whether to return a predictive distribution, or instead just the most likely value.\n\n        Returns\n        -------\n        The prediction.\n\n        '
        y_pred_mean = utils.math.dot(self._m, x)
        if not with_dist:
            return y_pred_mean
        x_arr = np.array(list(x.values()))
        (*_, ss_inv_arr) = self._get_arrays(x.keys(), m=False, ss=False)
        y_pred_var = 1 / self.beta + x_arr @ ss_inv_arr @ x_arr.T
        return proba.Gaussian._from_state(n=1, m=y_pred_mean, sig=y_pred_var ** 0.5, ddof=0)

    def predict_many(self, X):
        if False:
            return 10
        (m, *_) = self._get_arrays(X.columns, m=True, ss=False, ss_inv=False)
        return pd.Series(X.values @ m, index=X.index)