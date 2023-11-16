import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from mlxtend.data import iris_data
from mlxtend.feature_extraction import LinearDiscriminantAnalysis as LDA
from mlxtend.preprocessing import standardize
(X, y) = iris_data()
X = standardize(X)

def test_default_components():
    if False:
        i = 10
        return i + 15
    lda = LDA()
    lda.fit(X, y)
    res = lda.fit(X, y).transform(X)
    assert res.shape[1] == 4

def test_default_2components():
    if False:
        for i in range(10):
            print('nop')
    lda = LDA(n_discriminants=2)
    lda.fit(X, y)
    res = lda.fit(X, y).transform(X)
    assert res.shape[1] == 2

def test_default_0components():
    if False:
        i = 10
        return i + 15
    with pytest.raises(AttributeError):
        LDA(n_discriminants=0)

def test_evals():
    if False:
        for i in range(10):
            print('nop')
    lda = LDA(n_discriminants=2)
    lda.fit(X, y).transform(X)
    np.set_printoptions(suppress=True)
    print('%s' % lda.e_vals_)
    assert_almost_equal(lda.e_vals_, [20.9, 0.14, 0.0, 0.0], decimal=2)

def test_fail_array_fit():
    if False:
        for i in range(10):
            print('nop')
    lda = LDA()
    with pytest.raises(ValueError):
        lda.fit(X[1], y[1])

def test_fail_array_transform():
    if False:
        print('Hello World!')
    lda = LDA()
    lda.fit(X, y)
    with pytest.raises(ValueError):
        lda.transform(X[1])