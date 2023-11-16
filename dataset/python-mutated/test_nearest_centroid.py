"""
Testing for the nearest centroid module.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_classification_toy(csr_container):
    if False:
        return 10
    X_csr = csr_container(X)
    T_csr = csr_container(T)
    clf = NearestCentroid()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    clf = NearestCentroid()
    clf.fit(X_csr, y)
    assert_array_equal(clf.predict(T_csr), true_result)
    clf = NearestCentroid()
    clf.fit(X_csr, y)
    assert_array_equal(clf.predict(T), true_result)
    clf = NearestCentroid()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T_csr), true_result)
    clf = NearestCentroid()
    clf.fit(X_csr.tocoo(), y)
    assert_array_equal(clf.predict(T_csr.tolil()), true_result)

@pytest.mark.filterwarnings('ignore:Support for distance metrics:FutureWarning:sklearn')
def test_iris():
    if False:
        i = 10
        return i + 15
    for metric in ('euclidean', 'cosine'):
        clf = NearestCentroid(metric=metric).fit(iris.data, iris.target)
        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.9, 'Failed with score = ' + str(score)

@pytest.mark.filterwarnings('ignore:Support for distance metrics:FutureWarning:sklearn')
def test_iris_shrinkage():
    if False:
        return 10
    for metric in ('euclidean', 'cosine'):
        for shrink_threshold in [None, 0.1, 0.5]:
            clf = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)
            clf = clf.fit(iris.data, iris.target)
            score = np.mean(clf.predict(iris.data) == iris.target)
            assert score > 0.8, 'Failed with score = ' + str(score)

def test_pickle():
    if False:
        print('Hello World!')
    import pickle
    obj = NearestCentroid()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)
    obj2 = pickle.loads(s)
    assert type(obj2) == obj.__class__
    score2 = obj2.score(iris.data, iris.target)
    assert_array_equal(score, score2, 'Failed to generate same score after pickling (classification).')

def test_shrinkage_correct():
    if False:
        print('Hello World!')
    X = np.array([[0, 1], [1, 0], [1, 1], [2, 0], [6, 8]])
    y = np.array([1, 1, 2, 2, 2])
    clf = NearestCentroid(shrink_threshold=0.1)
    clf.fit(X, y)
    expected_result = np.array([[0.778731, 0.8545292], [2.814179, 2.763647]])
    np.testing.assert_array_almost_equal(clf.centroids_, expected_result)

def test_shrinkage_threshold_decoded_y():
    if False:
        return 10
    clf = NearestCentroid(shrink_threshold=0.01)
    y_ind = np.asarray(y)
    y_ind[y_ind == -1] = 0
    clf.fit(X, y_ind)
    centroid_encoded = clf.centroids_
    clf.fit(X, y)
    assert_array_equal(centroid_encoded, clf.centroids_)

def test_predict_translated_data():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    X = rng.rand(50, 50)
    y = rng.randint(0, 3, 50)
    noise = rng.rand(50)
    clf = NearestCentroid(shrink_threshold=0.1)
    clf.fit(X, y)
    y_init = clf.predict(X)
    clf = NearestCentroid(shrink_threshold=0.1)
    X_noise = X + noise
    clf.fit(X_noise, y)
    y_translate = clf.predict(X_noise)
    assert_array_equal(y_init, y_translate)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_manhattan_metric(csr_container):
    if False:
        i = 10
        return i + 15
    X_csr = csr_container(X)
    clf = NearestCentroid(metric='manhattan')
    clf.fit(X, y)
    dense_centroid = clf.centroids_
    clf.fit(X_csr, y)
    assert_array_equal(clf.centroids_, dense_centroid)
    assert_array_equal(dense_centroid, [[-1, -1], [1, 1]])

@pytest.mark.parametrize('metric', sorted(list(NearestCentroid._valid_metrics - {'manhattan', 'euclidean'})))
def test_deprecated_distance_metric_supports(metric):
    if False:
        for i in range(10):
            print('nop')
    clf = NearestCentroid(metric=metric)
    with pytest.warns(FutureWarning, match='Support for distance metrics other than euclidean and manhattan'):
        clf.fit(X, y)

def test_features_zero_var():
    if False:
        print('Hello World!')
    X = np.empty((10, 2))
    X[:, 0] = -0.13725701
    X[:, 1] = -0.9853293
    y = np.zeros(10)
    y[0] = 1
    clf = NearestCentroid(shrink_threshold=0.1)
    with pytest.raises(ValueError):
        clf.fit(X, y)