import numpy as np
import pytest
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

@pytest.mark.parametrize('estimator', [GaussianMixture(), BayesianGaussianMixture()])
def test_gaussian_mixture_n_iter(estimator):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = rng.rand(10, 5)
    max_iter = 1
    estimator.set_params(max_iter=max_iter)
    estimator.fit(X)
    assert estimator.n_iter_ == max_iter

@pytest.mark.parametrize('estimator', [GaussianMixture(), BayesianGaussianMixture()])
def test_mixture_n_components_greater_than_n_samples_error(estimator):
    if False:
        i = 10
        return i + 15
    'Check error when n_components <= n_samples'
    rng = np.random.RandomState(0)
    X = rng.rand(10, 5)
    estimator.set_params(n_components=12)
    msg = 'Expected n_samples >= n_components'
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X)