import numpy as np
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import BSR_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
data = [[0, 1, 2, 3, 4], [0, 2, 2, 3, 5], [1, 1, 2, 4, 0]]
data2 = [[-0.13725701]] * 10

@pytest.mark.parametrize('sparse_container', [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_zero_variance(sparse_container):
    if False:
        while True:
            i = 10
    X = data if sparse_container is None else sparse_container(data)
    sel = VarianceThreshold().fit(X)
    assert_array_equal([0, 1, 3, 4], sel.get_support(indices=True))

def test_zero_variance_value_error():
    if False:
        return 10
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1], [0, 1]])

@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
def test_variance_threshold(sparse_container):
    if False:
        while True:
            i = 10
    X = data if sparse_container is None else sparse_container(data)
    X = VarianceThreshold(threshold=0.4).fit_transform(X)
    assert (len(data), 1) == X.shape

@pytest.mark.skipif(np.var(data2) == 0, reason='This test is not valid for this platform, as it relies on numerical instabilities.')
@pytest.mark.parametrize('sparse_container', [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_zero_variance_floating_point_error(sparse_container):
    if False:
        while True:
            i = 10
    X = data2 if sparse_container is None else sparse_container(data2)
    msg = 'No feature in X meets the variance threshold 0.00000'
    with pytest.raises(ValueError, match=msg):
        VarianceThreshold().fit(X)

@pytest.mark.parametrize('sparse_container', [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_variance_nan(sparse_container):
    if False:
        i = 10
        return i + 15
    arr = np.array(data, dtype=np.float64)
    arr[0, 0] = np.nan
    arr[:, 1] = np.nan
    X = arr if sparse_container is None else sparse_container(arr)
    sel = VarianceThreshold().fit(X)
    assert_array_equal([0, 3, 4], sel.get_support(indices=True))