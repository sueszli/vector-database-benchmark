from __future__ import annotations
import collections
import math
from river import base, utils
__all__ = ['ALMAClassifier']

class ALMAClassifier(base.Classifier):
    """Approximate Large Margin Algorithm (ALMA).

    Parameters
    ----------
    p
    alpha
    B
    C

    Attributes
    ----------
    w : collections.defaultdict
        The current weights.
    k : int
        The number of instances seen during training.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.ALMAClassifier()
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 82.56%

    References
    ----------
    [^1]: [Gentile, Claudio. "A new approximate maximal margin classification algorithm." Journal of Machine Learning Research 2.Dec (2001): 213-242](http://www.jmlr.org/papers/volume2/gentile01a/gentile01a.pdf)

    """

    def __init__(self, p=2, alpha=0.9, B=1 / 0.9, C=2 ** 0.5):
        if False:
            for i in range(10):
                print('nop')
        self.p = p
        self.alpha = alpha
        self.B = B
        self.C = C
        self.w = collections.defaultdict(float)
        self.k = 1

    def _raw_dot(self, x):
        if False:
            i = 10
            return i + 15
        return utils.math.dot(x, self.w)

    def predict_proba_one(self, x):
        if False:
            while True:
                i = 10
        yp = utils.math.sigmoid(self._raw_dot(x))
        return {False: 1 - yp, True: yp}

    def learn_one(self, x, y):
        if False:
            return 10
        y = int(y or -1)
        gamma = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)
        if y * self._raw_dot(x) < (1 - self.alpha) * gamma:
            eta = self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))
            for (i, xi) in x.items():
                self.w[i] += eta * y * xi
            norm = utils.math.norm(self.w, order=self.p)
            for i in x:
                self.w[i] /= max(1, norm)
            self.k += 1
        return self