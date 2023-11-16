from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import manifold, neighbors
from sklearn.datasets import make_blobs
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from sklearn.utils._testing import assert_allclose, assert_array_equal, ignore_warnings
eigen_solvers = ['dense', 'arpack']

def test_barycenter_kneighbors_graph(global_dtype):
    if False:
        return 10
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]], dtype=global_dtype)
    graph = barycenter_kneighbors_graph(X, 1)
    expected_graph = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=global_dtype)
    assert graph.dtype == global_dtype
    assert_allclose(graph.toarray(), expected_graph)
    graph = barycenter_kneighbors_graph(X, 2)
    assert_allclose(np.sum(graph.toarray(), axis=1), np.ones(3))
    pred = np.dot(graph.toarray(), X)
    assert linalg.norm(pred - X) / X.shape[0] < 1

def test_lle_simple_grid(global_dtype):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(42)
    X = np.array(list(product(range(5), repeat=2)))
    X = X + 1e-10 * rng.uniform(size=X.shape)
    X = X.astype(global_dtype, copy=False)
    n_components = 2
    clf = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=n_components, random_state=rng)
    tol = 0.1
    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()
    reconstruction_error = linalg.norm(np.dot(N, X) - X, 'fro')
    assert reconstruction_error < tol
    for solver in eigen_solvers:
        clf.set_params(eigen_solver=solver)
        clf.fit(X)
        assert clf.embedding_.shape[1] == n_components
        reconstruction_error = linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, 'fro') ** 2
        assert reconstruction_error < tol
        assert_allclose(clf.reconstruction_error_, reconstruction_error, atol=0.1)
    noise = rng.randn(*X.shape).astype(global_dtype, copy=False) / 100
    X_reembedded = clf.transform(X + noise)
    assert linalg.norm(X_reembedded - clf.embedding_) < tol

@pytest.mark.parametrize('method', ['standard', 'hessian', 'modified', 'ltsa'])
@pytest.mark.parametrize('solver', eigen_solvers)
def test_lle_manifold(global_dtype, method, solver):
    if False:
        return 10
    rng = np.random.RandomState(0)
    X = np.array(list(product(np.arange(18), repeat=2)))
    X = np.c_[X, X[:, 0] ** 2 / 18]
    X = X + 1e-10 * rng.uniform(size=X.shape)
    X = X.astype(global_dtype, copy=False)
    n_components = 2
    clf = manifold.LocallyLinearEmbedding(n_neighbors=6, n_components=n_components, method=method, random_state=0)
    tol = 1.5 if method == 'standard' else 3
    N = barycenter_kneighbors_graph(X, clf.n_neighbors).toarray()
    reconstruction_error = linalg.norm(np.dot(N, X) - X)
    assert reconstruction_error < tol
    clf.set_params(eigen_solver=solver)
    clf.fit(X)
    assert clf.embedding_.shape[1] == n_components
    reconstruction_error = linalg.norm(np.dot(N, clf.embedding_) - clf.embedding_, 'fro') ** 2
    details = 'solver: %s, method: %s' % (solver, method)
    assert reconstruction_error < tol, details
    assert np.abs(clf.reconstruction_error_ - reconstruction_error) < tol * reconstruction_error, details

def test_pipeline():
    if False:
        for i in range(10):
            print('nop')
    from sklearn import datasets, pipeline
    (X, y) = datasets.make_blobs(random_state=0)
    clf = pipeline.Pipeline([('filter', manifold.LocallyLinearEmbedding(random_state=0)), ('clf', neighbors.KNeighborsClassifier())])
    clf.fit(X, y)
    assert 0.9 < clf.score(X, y)

def test_singular_matrix():
    if False:
        for i in range(10):
            print('nop')
    M = np.ones((200, 3))
    f = ignore_warnings
    with pytest.raises(ValueError, match='Error in determining null-space with ARPACK'):
        f(manifold.locally_linear_embedding(M, n_neighbors=2, n_components=1, method='standard', eigen_solver='arpack'))

def test_integer_input():
    if False:
        print('Hello World!')
    rand = np.random.RandomState(0)
    X = rand.randint(0, 100, size=(20, 3))
    for method in ['standard', 'hessian', 'modified', 'ltsa']:
        clf = manifold.LocallyLinearEmbedding(method=method, n_neighbors=10)
        clf.fit(X)

def test_get_feature_names_out():
    if False:
        i = 10
        return i + 15
    'Check get_feature_names_out for LocallyLinearEmbedding.'
    (X, y) = make_blobs(random_state=0, n_features=4)
    n_components = 2
    iso = manifold.LocallyLinearEmbedding(n_components=n_components)
    iso.fit(X)
    names = iso.get_feature_names_out()
    assert_array_equal([f'locallylinearembedding{i}' for i in range(n_components)], names)