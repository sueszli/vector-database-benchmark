from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest
from river import proba

@pytest.mark.parametrize('p', [pytest.param(p, id=f'p={p!r}') for p in [1, 3, 5]])
def test_univariate_multivariate_consistency(p):
    if False:
        i = 10
        return i + 15
    X = pd.DataFrame(np.random.random((30, p)), columns=range(p))
    multi = proba.MultivariateGaussian()
    single = {c: proba.Gaussian() for c in X.columns}
    for x in X.to_dict(orient='records'):
        multi = multi.update(x)
        for (c, s) in single.items():
            s.update(x[c])
    for c in X.columns:
        assert math.isclose(multi.mu[c], single[c].mu)
        assert math.isclose(multi.sigma[c][c], single[c].sigma)