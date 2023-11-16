from __future__ import annotations
import collections
from river import optim
__all__ = ['NesterovMomentum']

class NesterovMomentum(optim.base.Optimizer):
    """Nesterov Momentum optimizer.

    Parameters
    ----------
    lr
    rho

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()
    >>> optimizer = optim.NesterovMomentum()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression(optimizer)
    ... )
    >>> metric = metrics.F1()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 84.22%

    """

    def __init__(self, lr=0.1, rho=0.9):
        if False:
            print('Hello World!')
        super().__init__(lr)
        self.rho = rho
        self.s = collections.defaultdict(float)

    def look_ahead(self, w):
        if False:
            while True:
                i = 10
        for i in w:
            w[i] -= self.rho * self.s[i]
        return w

    def _step_with_dict(self, w, g):
        if False:
            print('Hello World!')
        for i in w:
            w[i] += self.rho * self.s[i]
        for (i, gi) in g.items():
            self.s[i] = self.rho * self.s[i] + self.learning_rate * gi
            w[i] -= self.s[i]
        return w