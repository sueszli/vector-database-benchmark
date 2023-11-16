import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, DOK_CONTAINERS, LIL_CONTAINERS

@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS + CSC_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
@pytest.mark.parametrize('sample_size', [None, 'half'])
def test_silhouette(sparse_container, sample_size):
    if False:
        while True:
            i = 10
    dataset = datasets.load_iris()
    (X, y) = (dataset.data, dataset.target)
    if sparse_container is not None:
        X = sparse_container(X)
    sample_size = int(X.shape[0] / 2) if sample_size == 'half' else sample_size
    D = pairwise_distances(X, metric='euclidean')
    score_precomputed = silhouette_score(D, y, metric='precomputed', sample_size=sample_size, random_state=0)
    score_euclidean = silhouette_score(X, y, metric='euclidean', sample_size=sample_size, random_state=0)
    assert score_precomputed > 0
    assert score_euclidean > 0
    assert score_precomputed == pytest.approx(score_euclidean)

def test_cluster_size_1():
    if False:
        for i in range(10):
            print('nop')
    X = [[0.0], [1.0], [1.0], [2.0], [3.0], [3.0]]
    labels = np.array([0, 1, 1, 1, 2, 2])
    silhouette = silhouette_score(X, labels)
    assert not np.isnan(silhouette)
    ss = silhouette_samples(X, labels)
    assert_array_equal(ss, [0, 0.5, 0.5, 0, 1, 1])

def test_silhouette_paper_example():
    if False:
        print('Hello World!')
    lower = [5.58, 7.0, 6.5, 7.08, 7.0, 3.83, 4.83, 5.08, 8.17, 5.83, 2.17, 5.75, 6.67, 6.92, 4.92, 6.42, 5.0, 5.58, 6.0, 4.67, 6.42, 3.42, 5.5, 6.42, 6.42, 5.0, 3.92, 6.17, 2.5, 4.92, 6.25, 7.33, 4.5, 2.25, 6.33, 2.75, 6.08, 6.67, 4.25, 2.67, 6.0, 6.17, 6.17, 6.92, 6.17, 5.25, 6.83, 4.5, 3.75, 5.75, 5.42, 6.08, 5.83, 6.67, 3.67, 4.75, 3.0, 6.08, 6.67, 5.0, 5.58, 4.83, 6.17, 5.67, 6.5, 6.92]
    D = np.zeros((12, 12))
    D[np.tril_indices(12, -1)] = lower
    D += D.T
    names = ['BEL', 'BRA', 'CHI', 'CUB', 'EGY', 'FRA', 'IND', 'ISR', 'USA', 'USS', 'YUG', 'ZAI']
    labels1 = [1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1]
    expected1 = {'USA': 0.43, 'BEL': 0.39, 'FRA': 0.35, 'ISR': 0.3, 'BRA': 0.22, 'EGY': 0.2, 'ZAI': 0.19, 'CUB': 0.4, 'USS': 0.34, 'CHI': 0.33, 'YUG': 0.26, 'IND': -0.04}
    score1 = 0.28
    labels2 = [1, 2, 3, 3, 1, 1, 2, 1, 1, 3, 3, 2]
    expected2 = {'USA': 0.47, 'FRA': 0.44, 'BEL': 0.42, 'ISR': 0.37, 'EGY': 0.02, 'ZAI': 0.28, 'BRA': 0.25, 'IND': 0.17, 'CUB': 0.48, 'USS': 0.44, 'YUG': 0.31, 'CHI': 0.31}
    score2 = 0.33
    for (labels, expected, score) in [(labels1, expected1, score1), (labels2, expected2, score2)]:
        expected = [expected[name] for name in names]
        pytest.approx(expected, silhouette_samples(D, np.array(labels), metric='precomputed'), abs=0.01)
        pytest.approx(score, silhouette_score(D, np.array(labels), metric='precomputed'), abs=0.01)

def test_correct_labelsize():
    if False:
        return 10
    dataset = datasets.load_iris()
    X = dataset.data
    y = np.arange(X.shape[0])
    err_msg = 'Number of labels is %d\\. Valid values are 2 to n_samples - 1 \\(inclusive\\)' % len(np.unique(y))
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)
    y = np.zeros(X.shape[0])
    err_msg = 'Number of labels is %d\\. Valid values are 2 to n_samples - 1 \\(inclusive\\)' % len(np.unique(y))
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)

def test_non_encoded_labels():
    if False:
        i = 10
        return i + 15
    dataset = datasets.load_iris()
    X = dataset.data
    labels = dataset.target
    assert silhouette_score(X, labels * 2 + 10) == silhouette_score(X, labels)
    assert_array_equal(silhouette_samples(X, labels * 2 + 10), silhouette_samples(X, labels))

def test_non_numpy_labels():
    if False:
        print('Hello World!')
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    assert silhouette_score(list(X), list(y)) == silhouette_score(X, y)

@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_silhouette_nonzero_diag(dtype):
    if False:
        for i in range(10):
            print('nop')
    dists = pairwise_distances(np.array([[0.2, 0.1, 0.12, 1.34, 1.11, 1.6]], dtype=dtype).T)
    labels = [0, 0, 0, 1, 1, 1]
    dists[2][2] = np.finfo(dists.dtype).eps * 10
    silhouette_samples(dists, labels, metric='precomputed')
    dists[2][2] = np.finfo(dists.dtype).eps * 1000
    with pytest.raises(ValueError, match='contains non-zero'):
        silhouette_samples(dists, labels, metric='precomputed')

@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_silhouette_samples_precomputed_sparse(sparse_container):
    if False:
        for i in range(10):
            print('nop')
    'Check that silhouette_samples works for sparse matrices correctly.'
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    pdist_dense = pairwise_distances(X)
    pdist_sparse = sparse_container(pdist_dense)
    assert issparse(pdist_sparse)
    output_with_sparse_input = silhouette_samples(pdist_sparse, y, metric='precomputed')
    output_with_dense_input = silhouette_samples(pdist_dense, y, metric='precomputed')
    assert_allclose(output_with_sparse_input, output_with_dense_input)

@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_silhouette_samples_euclidean_sparse(sparse_container):
    if False:
        while True:
            i = 10
    'Check that silhouette_samples works for sparse matrices correctly.'
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    pdist_dense = pairwise_distances(X)
    pdist_sparse = sparse_container(pdist_dense)
    assert issparse(pdist_sparse)
    output_with_sparse_input = silhouette_samples(pdist_sparse, y)
    output_with_dense_input = silhouette_samples(pdist_dense, y)
    assert_allclose(output_with_sparse_input, output_with_dense_input)

@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_silhouette_reduce(sparse_container):
    if False:
        print('Hello World!')
    'Check for non-CSR input to private method `_silhouette_reduce`.'
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    pdist_dense = pairwise_distances(X)
    pdist_sparse = sparse_container(pdist_dense)
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    label_freqs = np.bincount(y)
    with pytest.raises(TypeError, match='Expected CSR matrix. Please pass sparse matrix in CSR format.'):
        _silhouette_reduce(pdist_sparse, start=0, labels=y, label_freqs=label_freqs)

def assert_raises_on_only_one_label(func):
    if False:
        while True:
            i = 10
    'Assert message when there is only one label'
    rng = np.random.RandomState(seed=0)
    with pytest.raises(ValueError, match='Number of labels is'):
        func(rng.rand(10, 2), np.zeros(10))

def assert_raises_on_all_points_same_cluster(func):
    if False:
        i = 10
        return i + 15
    'Assert message when all point are in different clusters'
    rng = np.random.RandomState(seed=0)
    with pytest.raises(ValueError, match='Number of labels is'):
        func(rng.rand(10, 2), np.arange(10))

def test_calinski_harabasz_score():
    if False:
        return 10
    assert_raises_on_only_one_label(calinski_harabasz_score)
    assert_raises_on_all_points_same_cluster(calinski_harabasz_score)
    assert 1.0 == calinski_harabasz_score(np.ones((10, 2)), [0] * 5 + [1] * 5)
    assert 0.0 == calinski_harabasz_score([[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10)
    X = [[0, 0], [1, 1]] * 5 + [[3, 3], [4, 4]] * 5 + [[0, 4], [1, 3]] * 5 + [[3, 1], [4, 0]] * 5
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(calinski_harabasz_score(X, labels), 45 * (40 - 4) / (5 * (4 - 1)))

def test_davies_bouldin_score():
    if False:
        for i in range(10):
            print('nop')
    assert_raises_on_only_one_label(davies_bouldin_score)
    assert_raises_on_all_points_same_cluster(davies_bouldin_score)
    assert davies_bouldin_score(np.ones((10, 2)), [0] * 5 + [1] * 5) == pytest.approx(0.0)
    assert davies_bouldin_score([[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10) == pytest.approx(0.0)
    X = [[0, 0], [1, 1]] * 5 + [[3, 3], [4, 4]] * 5 + [[0, 4], [1, 3]] * 5 + [[3, 1], [4, 0]] * 5
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(davies_bouldin_score(X, labels), 2 * np.sqrt(0.5) / 3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        davies_bouldin_score(X, labels)
    X = [[0, 0], [2, 2], [3, 3], [5, 5]]
    labels = [0, 0, 1, 2]
    pytest.approx(davies_bouldin_score(X, labels), 5.0 / 4 / 3)

def test_silhouette_score_integer_precomputed():
    if False:
        for i in range(10):
            print('nop')
    'Check that silhouette_score works for precomputed metrics that are integers.\n\n    Non-regression test for #22107.\n    '
    result = silhouette_score([[0, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 0, 1], metric='precomputed')
    assert result == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match='contains non-zero'):
        silhouette_score([[1, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 0, 1], metric='precomputed')