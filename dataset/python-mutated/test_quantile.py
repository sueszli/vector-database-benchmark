from __future__ import annotations
import random
from river import stats

def test_issue_1178():
    if False:
        i = 10
        return i + 15
    '\n\n    https://github.com/online-ml/river/issues/1178\n\n    >>> from river import stats\n\n    >>> q = stats.Quantile(0.01)\n    >>> for x in [5, 0, 0, 0, 0, 0, 0, 0]:\n    ...     q = q.update(x)\n    ...     print(q)\n    Quantile: 5.\n    Quantile: 0.\n    Quantile: 0.\n    Quantile: 0.\n    Quantile: 0.\n    Quantile: 0.\n    Quantile: 0.\n    Quantile: 0.\n\n    >>> q = stats.Quantile(0.99)\n    >>> for x in [5, 0, 0, 0, 0, 0, 0, 0]:\n    ...     q = q.update(x)\n    ...     print(q)\n    Quantile: 5.\n    Quantile: 5.\n    Quantile: 5.\n    Quantile: 5.\n    Quantile: 5.\n    Quantile: 0.\n    Quantile: 0.277778\n    Quantile: 0.827546\n\n    '

def test_ge():
    if False:
        while True:
            i = 10
    low = stats.Quantile(0.01)
    high = stats.Quantile(0.99)
    for _ in range(100):
        x = random.random()
        low.update(x)
        high.update(x)
        assert high.get() >= low.get()