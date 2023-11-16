from __future__ import annotations
import copy
import numpy as np
from river import stats

class Cov(stats.base.Bivariate):
    """Covariance.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom.

    Examples
    --------

    >>> from river import stats

    >>> x = [-2.1,  -1,  4.3]
    >>> y = [   3, 1.1, 0.12]

    >>> cov = stats.Cov()

    >>> for xi, yi in zip(x, y):
    ...     print(cov.update(xi, yi).get())
    0.0
    -1.044999
    -4.286

    This class has a `revert` method, and can thus be wrapped by `utils.Rolling`:

    >>> from river import utils

    >>> x = [-2.1,  -1, 4.3, 1, -2.1,  -1, 4.3]
    >>> y = [   3, 1.1, .12, 1,    3, 1.1, .12]

    >>> rcov = utils.Rolling(stats.Cov(), window_size=3)

    >>> for xi, yi in zip(x, y):
    ...     print(rcov.update(xi, yi).get())
    0.0
    -1.045
    -4.286
    -1.382
    -4.589
    -1.415
    -4.286

    Notes
    -----
    The outcomes of the incremental and parallel updates are consistent with numpy's
    batch processing when $\\text{ddof} \\le 1$.

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
    [^2]: Schubert, E. and Gertz, M., 2018, July. Numerically stable parallel computation of
    (co-) variance. In Proceedings of the 30th International Conference on Scientific and
    Statistical Database Management (pp. 1-12).

    """

    def __init__(self, ddof=1):
        if False:
            return 10
        self.ddof = ddof
        self.mean_x = stats.Mean()
        self.mean_y = stats.Mean()
        self.cov = 0

    @property
    def n(self):
        if False:
            while True:
                i = 10
        return self.mean_x.n

    def update(self, x, y, w=1.0):
        if False:
            i = 10
            return i + 15
        dx = x - self.mean_x.get()
        self.mean_x.update(x, w)
        self.mean_y.update(y, w)
        self.cov += w * (dx * (y - self.mean_y.get()) - self.cov) / max(self.n - self.ddof, 1)
        return self

    def revert(self, x, y, w=1.0):
        if False:
            i = 10
            return i + 15
        dx = x - self.mean_x.get()
        self.mean_x.revert(x, w)
        self.mean_y.revert(y, w)
        self.cov -= w * (dx * (y - self.mean_y.get()) - self.cov) / max(self.n - self.ddof, 1)
        return self

    def update_many(self, X: np.ndarray, Y: np.ndarray):
        if False:
            while True:
                i = 10
        dx = X - self.mean_x.get()
        self.mean_x.update_many(X)
        self.mean_y.update_many(Y)
        self.cov += (dx * (Y - self.mean_y.get()) - self.cov).sum() / max(self.n - self.ddof, 1)
        return self

    def get(self):
        if False:
            while True:
                i = 10
        return self.cov

    @classmethod
    def _from_state(cls, n, mean_x, mean_y, cov, *, ddof=1):
        if False:
            print('Hello World!')
        new = cls(ddof=ddof)
        new.mean_x = stats.Mean._from_state(n, mean_x)
        new.mean_y = stats.Mean._from_state(n, mean_y)
        new.cov = cov
        return new

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        old_mean_x = self.mean_x.get()
        old_mean_y = self.mean_y.get()
        old_n = self.n
        self.mean_x += other.mean_x
        self.mean_y += other.mean_y
        if self.n <= self.ddof:
            return self
        scale_a = old_n - self.ddof
        scale_b = other.n - other.ddof
        self.cov = scale_a * self.cov + scale_b * other.cov
        self.cov += (old_mean_x - other.mean_x.get()) * (old_mean_y - other.mean_y.get()) * (old_n * other.n / self.n)
        self.cov /= max(self.n - self.ddof, 1)
        return self

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        result = copy.deepcopy(self)
        result += other
        return result

    def __isub__(self, other):
        if False:
            return 10
        if self.n <= self.ddof:
            return self
        old_n = self.n
        self.mean_x -= other.mean_x
        self.mean_y -= other.mean_y
        if self.n <= self.ddof:
            self.cov = 0
            return self
        scale_x = old_n - self.ddof
        scale_b = other.mean_x.n - other.ddof
        self.cov = scale_x * self.cov - scale_b * other.cov
        self.cov -= (self.mean_x.get() - other.mean_x.get()) * (self.mean_y.get() - other.mean_y.get()) * (self.n * other.mean_x.n / old_n)
        self.cov /= self.n - self.ddof
        return self

    def __sub__(self, other):
        if False:
            return 10
        result = copy.deepcopy(self)
        result -= other
        return result