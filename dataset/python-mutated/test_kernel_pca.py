import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_moons
from mlxtend.feature_extraction import RBFKernelPCA as KPCA
(X1, y1) = make_moons(n_samples=50, random_state=1)

def test_default_components():
    if False:
        return 10
    pca = KPCA()
    pca.fit(X1)
    assert pca.X_projected_.shape == X1.shape

def test_default_2components():
    if False:
        i = 10
        return i + 15
    pca = KPCA(n_components=2)
    pca.fit(X1)
    assert pca.X_projected_.shape == (X1.shape[0], 2)

def test_default_0components():
    if False:
        return 10
    with pytest.raises(AttributeError):
        pca = KPCA(n_components=0)
        pca.fit(X1)

def test_proj():
    if False:
        print('Hello World!')
    pca = KPCA(n_components=2)
    pca.fit(X1[:2])
    exp = np.array([[-0.71, -0.71], [0.71, -0.71]])
    assert_almost_equal(pca.X_projected_, exp, decimal=2)

def test_reproj_1():
    if False:
        print('Hello World!')
    pca = KPCA(n_components=2)
    pca.fit(X1)
    exp = pca.transform(X1)
    assert_almost_equal(pca.X_projected_, exp, decimal=2)

def test_reproj_2():
    if False:
        return 10
    pca = KPCA(n_components=2)
    pca.fit(X1)
    exp = pca.transform(X1[1, None])
    assert_almost_equal(pca.X_projected_[1, None], exp, decimal=2)

def test_fail_array_fit():
    if False:
        return 10
    pca = KPCA(n_components=2)
    with pytest.raises(ValueError):
        pca.fit(X1[1])

def test_fail_array_transform():
    if False:
        print('Hello World!')
    pca = KPCA(n_components=2)
    pca.fit(X1)
    with pytest.raises(ValueError):
        pca.transform(X1[1])