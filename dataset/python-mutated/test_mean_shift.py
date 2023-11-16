"""
Testing for mean shift clustering methods

"""
import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
n_clusters = 3
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
(X, _) = make_blobs(n_samples=300, n_features=2, centers=centers, cluster_std=0.4, shuffle=True, random_state=11)

def test_estimate_bandwidth():
    if False:
        for i in range(10):
            print('nop')
    bandwidth = estimate_bandwidth(X, n_samples=200)
    assert 0.9 <= bandwidth <= 1.5

def test_estimate_bandwidth_1sample(global_dtype):
    if False:
        i = 10
        return i + 15
    bandwidth = estimate_bandwidth(X.astype(global_dtype, copy=False), n_samples=1, quantile=0.3)
    assert bandwidth.dtype == X.dtype
    assert bandwidth == pytest.approx(0.0, abs=1e-05)

@pytest.mark.parametrize('bandwidth, cluster_all, expected, first_cluster_label', [(1.2, True, 3, 0), (1.2, False, 4, -1)])
def test_mean_shift(global_dtype, bandwidth, cluster_all, expected, first_cluster_label):
    if False:
        return 10
    X_with_global_dtype = X.astype(global_dtype, copy=False)
    ms = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)
    labels = ms.fit(X_with_global_dtype).labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    assert n_clusters_ == expected
    assert labels_unique[0] == first_cluster_label
    assert ms.cluster_centers_.dtype == global_dtype
    (cluster_centers, labels_mean_shift) = mean_shift(X_with_global_dtype, cluster_all=cluster_all)
    labels_mean_shift_unique = np.unique(labels_mean_shift)
    n_clusters_mean_shift = len(labels_mean_shift_unique)
    assert n_clusters_mean_shift == expected
    assert labels_mean_shift_unique[0] == first_cluster_label
    assert cluster_centers.dtype == global_dtype

def test_parallel(global_dtype):
    if False:
        print('Hello World!')
    centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
    (X, _) = make_blobs(n_samples=50, n_features=2, centers=centers, cluster_std=0.4, shuffle=True, random_state=11)
    X = X.astype(global_dtype, copy=False)
    ms1 = MeanShift(n_jobs=2)
    ms1.fit(X)
    ms2 = MeanShift()
    ms2.fit(X)
    assert_allclose(ms1.cluster_centers_, ms2.cluster_centers_)
    assert ms1.cluster_centers_.dtype == ms2.cluster_centers_.dtype
    assert_array_equal(ms1.labels_, ms2.labels_)

def test_meanshift_predict(global_dtype):
    if False:
        print('Hello World!')
    ms = MeanShift(bandwidth=1.2)
    X_with_global_dtype = X.astype(global_dtype, copy=False)
    labels = ms.fit_predict(X_with_global_dtype)
    labels2 = ms.predict(X_with_global_dtype)
    assert_array_equal(labels, labels2)

def test_meanshift_all_orphans():
    if False:
        for i in range(10):
            print('nop')
    ms = MeanShift(bandwidth=0.1, seeds=[[-9, -9], [-10, -10]])
    msg = 'No point was within bandwidth=0.1'
    with pytest.raises(ValueError, match=msg):
        ms.fit(X)

def test_unfitted():
    if False:
        return 10
    ms = MeanShift()
    assert not hasattr(ms, 'cluster_centers_')
    assert not hasattr(ms, 'labels_')

def test_cluster_intensity_tie(global_dtype):
    if False:
        while True:
            i = 10
    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=global_dtype)
    c1 = MeanShift(bandwidth=2).fit(X)
    X = np.array([[4, 7], [3, 5], [3, 6], [1, 1], [2, 1], [1, 0]], dtype=global_dtype)
    c2 = MeanShift(bandwidth=2).fit(X)
    assert_array_equal(c1.labels_, [1, 1, 1, 0, 0, 0])
    assert_array_equal(c2.labels_, [0, 0, 0, 1, 1, 1])

def test_bin_seeds(global_dtype):
    if False:
        print('Hello World!')
    X = np.array([[1.0, 1.0], [1.4, 1.4], [1.8, 1.2], [2.0, 1.0], [2.1, 1.1], [0.0, 0.0]], dtype=global_dtype)
    ground_truth = {(1.0, 1.0), (2.0, 1.0), (0.0, 0.0)}
    test_bins = get_bin_seeds(X, 1, 1)
    test_result = set((tuple(p) for p in test_bins))
    assert len(ground_truth.symmetric_difference(test_result)) == 0
    ground_truth = {(1.0, 1.0), (2.0, 1.0)}
    test_bins = get_bin_seeds(X, 1, 2)
    test_result = set((tuple(p) for p in test_bins))
    assert len(ground_truth.symmetric_difference(test_result)) == 0
    with warnings.catch_warnings(record=True):
        test_bins = get_bin_seeds(X, 0.01, 1)
    assert_allclose(test_bins, X)
    (X, _) = make_blobs(n_samples=100, n_features=2, centers=[[0, 0], [1, 1]], cluster_std=0.1, random_state=0)
    X = X.astype(global_dtype, copy=False)
    test_bins = get_bin_seeds(X, 1)
    assert_array_equal(test_bins, [[0, 0], [1, 1]])

@pytest.mark.parametrize('max_iter', [1, 100])
def test_max_iter(max_iter):
    if False:
        for i in range(10):
            print('nop')
    (clusters1, _) = mean_shift(X, max_iter=max_iter)
    ms = MeanShift(max_iter=max_iter).fit(X)
    clusters2 = ms.cluster_centers_
    assert ms.n_iter_ <= ms.max_iter
    assert len(clusters1) == len(clusters2)
    for (c1, c2) in zip(clusters1, clusters2):
        assert np.allclose(c1, c2)

def test_mean_shift_zero_bandwidth(global_dtype):
    if False:
        for i in range(10):
            print('nop')
    X = np.array([1, 1, 1, 2, 2, 2, 3, 3], dtype=global_dtype).reshape(-1, 1)
    bandwidth = estimate_bandwidth(X)
    assert bandwidth == 0
    assert get_bin_seeds(X, bin_size=bandwidth) is X
    ms_binning = MeanShift(bin_seeding=True, bandwidth=None).fit(X)
    ms_nobinning = MeanShift(bin_seeding=False).fit(X)
    expected_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    assert v_measure_score(ms_binning.labels_, expected_labels) == pytest.approx(1)
    assert v_measure_score(ms_nobinning.labels_, expected_labels) == pytest.approx(1)
    assert_allclose(ms_binning.cluster_centers_, ms_nobinning.cluster_centers_)