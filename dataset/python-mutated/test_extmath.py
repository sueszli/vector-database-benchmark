import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import assert_allclose, assert_allclose_dense_sparse, assert_almost_equal, assert_array_almost_equal, assert_array_equal, skip_if_32bit
from sklearn.utils.extmath import _deterministic_vector_sign_flip, _incremental_mean_and_var, _randomized_eigsh, _safe_accumulator_op, cartesian, density, log_logistic, randomized_svd, row_norms, safe_sparse_dot, softmax, stable_cumsum, svd_flip, weighted_mode
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS, DOK_CONTAINERS, LIL_CONTAINERS, _mode

@pytest.mark.parametrize('sparse_container', COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS + LIL_CONTAINERS)
def test_density(sparse_container):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    X = rng.randint(10, size=(10, 5))
    X[1, 2] = 0
    X[5, 3] = 0
    assert density(sparse_container(X)) == density(X)

def test_density_deprecated_kwargs():
    if False:
        for i in range(10):
            print('nop')
    'Check that future warning is raised when user enters keyword arguments.'
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.warns(FutureWarning, match='Additional keyword arguments are deprecated in version 1.2 and will be removed in version 1.4.'):
        density(test_array, a=1)

def test_uniform_weights():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    x = rng.randint(10, size=(10, 5))
    weights = np.ones(x.shape)
    for axis in (None, 0, 1):
        (mode, score) = _mode(x, axis)
        (mode2, score2) = weighted_mode(x, weights, axis=axis)
        assert_array_equal(mode, mode2)
        assert_array_equal(score, score2)

def test_random_weights():
    if False:
        while True:
            i = 10
    mode_result = 6
    rng = np.random.RandomState(0)
    x = rng.randint(mode_result, size=(100, 10))
    w = rng.random_sample(x.shape)
    x[:, :5] = mode_result
    w[:, :5] += 1
    (mode, score) = weighted_mode(x, w, axis=1)
    assert_array_equal(mode, mode_result)
    assert_array_almost_equal(score.ravel(), w[:, :5].sum(1))

@pytest.mark.parametrize('dtype', (np.int32, np.int64, np.float32, np.float64))
def test_randomized_svd_low_rank_all_dtypes(dtype):
    if False:
        for i in range(10):
            print('nop')
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    decimal = 5 if dtype == np.float32 else 7
    dtype = np.dtype(dtype)
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=rank, tail_strength=0.0, random_state=0).astype(dtype, copy=False)
    assert X.shape == (n_samples, n_features)
    (U, s, Vt) = linalg.svd(X, full_matrices=False)
    U = U.astype(dtype, copy=False)
    s = s.astype(dtype, copy=False)
    Vt = Vt.astype(dtype, copy=False)
    for normalizer in ['auto', 'LU', 'QR']:
        (Ua, sa, Va) = randomized_svd(X, k, power_iteration_normalizer=normalizer, random_state=0)
        if dtype.kind == 'f':
            assert Ua.dtype == dtype
            assert sa.dtype == dtype
            assert Va.dtype == dtype
        else:
            assert Ua.dtype == np.float64
            assert sa.dtype == np.float64
            assert Va.dtype == np.float64
        assert Ua.shape == (n_samples, k)
        assert sa.shape == (k,)
        assert Va.shape == (k, n_features)
        assert_almost_equal(s[:k], sa, decimal=decimal)
        assert_almost_equal(np.dot(U[:, :k], Vt[:k, :]), np.dot(Ua, Va), decimal=decimal)
        for csr_container in CSR_CONTAINERS:
            X = csr_container(X)
            (Ua, sa, Va) = randomized_svd(X, k, power_iteration_normalizer=normalizer, random_state=0)
            if dtype.kind == 'f':
                assert Ua.dtype == dtype
                assert sa.dtype == dtype
                assert Va.dtype == dtype
            else:
                assert Ua.dtype.kind == 'f'
                assert sa.dtype.kind == 'f'
                assert Va.dtype.kind == 'f'
            assert_almost_equal(s[:rank], sa[:rank], decimal=decimal)

@pytest.mark.parametrize('dtype', (np.int32, np.int64, np.float32, np.float64))
def test_randomized_eigsh(dtype):
    if False:
        print('Hello World!')
    'Test that `_randomized_eigsh` returns the appropriate components'
    rng = np.random.RandomState(42)
    X = np.diag(np.array([1.0, -2.0, 0.0, 3.0], dtype=dtype))
    rand_rot = np.linalg.qr(rng.normal(size=X.shape))[0]
    X = rand_rot @ X @ rand_rot.T
    (eigvals, eigvecs) = _randomized_eigsh(X, n_components=2, selection='module')
    assert eigvals.shape == (2,)
    assert_array_almost_equal(eigvals, [3.0, -2.0])
    assert eigvecs.shape == (4, 2)
    with pytest.raises(NotImplementedError):
        _randomized_eigsh(X, n_components=2, selection='value')

@pytest.mark.parametrize('k', (10, 50, 100, 199, 200))
def test_randomized_eigsh_compared_to_others(k):
    if False:
        print('Hello World!')
    'Check that `_randomized_eigsh` is similar to other `eigsh`\n\n    Tests that for a random PSD matrix, `_randomized_eigsh` provides results\n    comparable to LAPACK (scipy.linalg.eigh) and ARPACK\n    (scipy.sparse.linalg.eigsh).\n\n    Note: some versions of ARPACK do not support k=n_features.\n    '
    n_features = 200
    X = make_sparse_spd_matrix(n_features, random_state=0)
    (eigvals, eigvecs) = _randomized_eigsh(X, n_components=k, selection='module', n_iter=25, random_state=0)
    (eigvals_qr, eigvecs_qr) = _randomized_eigsh(X, n_components=k, n_iter=25, n_oversamples=20, random_state=0, power_iteration_normalizer='QR', selection='module')
    (eigvals_lapack, eigvecs_lapack) = eigh(X, subset_by_index=(n_features - k, n_features - 1))
    indices = eigvals_lapack.argsort()[::-1]
    eigvals_lapack = eigvals_lapack[indices]
    eigvecs_lapack = eigvecs_lapack[:, indices]
    assert eigvals_lapack.shape == (k,)
    assert_array_almost_equal(eigvals, eigvals_lapack, decimal=6)
    assert_array_almost_equal(eigvals_qr, eigvals_lapack, decimal=6)
    assert eigvecs_lapack.shape == (n_features, k)
    dummy_vecs = np.zeros_like(eigvecs).T
    (eigvecs, _) = svd_flip(eigvecs, dummy_vecs)
    (eigvecs_qr, _) = svd_flip(eigvecs_qr, dummy_vecs)
    (eigvecs_lapack, _) = svd_flip(eigvecs_lapack, dummy_vecs)
    assert_array_almost_equal(eigvecs, eigvecs_lapack, decimal=4)
    assert_array_almost_equal(eigvecs_qr, eigvecs_lapack, decimal=6)
    if k < n_features:
        v0 = _init_arpack_v0(n_features, random_state=0)
        (eigvals_arpack, eigvecs_arpack) = eigsh(X, k, which='LA', tol=0, maxiter=None, v0=v0)
        indices = eigvals_arpack.argsort()[::-1]
        eigvals_arpack = eigvals_arpack[indices]
        assert_array_almost_equal(eigvals_lapack, eigvals_arpack, decimal=10)
        eigvecs_arpack = eigvecs_arpack[:, indices]
        (eigvecs_arpack, _) = svd_flip(eigvecs_arpack, dummy_vecs)
        assert_array_almost_equal(eigvecs_arpack, eigvecs_lapack, decimal=8)

@pytest.mark.parametrize('n,rank', [(10, 7), (100, 10), (100, 80), (500, 10), (500, 250), (500, 400)])
def test_randomized_eigsh_reconst_low_rank(n, rank):
    if False:
        print('Hello World!')
    'Check that randomized_eigsh is able to reconstruct a low rank psd matrix\n\n    Tests that the decomposition provided by `_randomized_eigsh` leads to\n    orthonormal eigenvectors, and that a low rank PSD matrix can be effectively\n    reconstructed with good accuracy using it.\n    '
    assert rank < n
    rng = np.random.RandomState(69)
    X = rng.randn(n, rank)
    A = X @ X.T
    (S, V) = _randomized_eigsh(A, n_components=rank, random_state=rng)
    assert_array_almost_equal(np.linalg.norm(V, axis=0), np.ones(S.shape))
    assert_array_almost_equal(V.T @ V, np.diag(np.ones(S.shape)))
    A_reconstruct = V @ np.diag(S) @ V.T
    assert_array_almost_equal(A_reconstruct, A, decimal=6)

@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_row_norms(dtype, csr_container):
    if False:
        return 10
    X = np.random.RandomState(42).randn(100, 100)
    if dtype is np.float32:
        precision = 4
    else:
        precision = 5
    X = X.astype(dtype, copy=False)
    sq_norm = (X ** 2).sum(axis=1)
    assert_array_almost_equal(sq_norm, row_norms(X, squared=True), precision)
    assert_array_almost_equal(np.sqrt(sq_norm), row_norms(X), precision)
    for csr_index_dtype in [np.int32, np.int64]:
        Xcsr = csr_container(X, dtype=dtype)
        if csr_index_dtype is np.int64:
            Xcsr.indptr = Xcsr.indptr.astype(csr_index_dtype, copy=False)
            Xcsr.indices = Xcsr.indices.astype(csr_index_dtype, copy=False)
        assert Xcsr.indices.dtype == csr_index_dtype
        assert Xcsr.indptr.dtype == csr_index_dtype
        assert_array_almost_equal(sq_norm, row_norms(Xcsr, squared=True), precision)
        assert_array_almost_equal(np.sqrt(sq_norm), row_norms(Xcsr), precision)

def test_randomized_svd_low_rank_with_noise():
    if False:
        for i in range(10):
            print('nop')
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=rank, tail_strength=0.1, random_state=0)
    assert X.shape == (n_samples, n_features)
    (_, s, _) = linalg.svd(X, full_matrices=False)
    for normalizer in ['auto', 'none', 'LU', 'QR']:
        (_, sa, _) = randomized_svd(X, k, n_iter=0, power_iteration_normalizer=normalizer, random_state=0)
        assert np.abs(s[:k] - sa).max() > 0.01
        (_, sap, _) = randomized_svd(X, k, power_iteration_normalizer=normalizer, random_state=0)
        assert_almost_equal(s[:k], sap, decimal=3)

def test_randomized_svd_infinite_rank():
    if False:
        return 10
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=rank, tail_strength=1.0, random_state=0)
    assert X.shape == (n_samples, n_features)
    (_, s, _) = linalg.svd(X, full_matrices=False)
    for normalizer in ['auto', 'none', 'LU', 'QR']:
        (_, sa, _) = randomized_svd(X, k, n_iter=0, power_iteration_normalizer=normalizer, random_state=0)
        assert np.abs(s[:k] - sa).max() > 0.1
        (_, sap, _) = randomized_svd(X, k, n_iter=5, power_iteration_normalizer=normalizer, random_state=0)
        assert_almost_equal(s[:k], sap, decimal=3)

def test_randomized_svd_transpose_consistency():
    if False:
        print('Hello World!')
    n_samples = 100
    n_features = 500
    rank = 4
    k = 10
    X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=rank, tail_strength=0.5, random_state=0)
    assert X.shape == (n_samples, n_features)
    (U1, s1, V1) = randomized_svd(X, k, n_iter=3, transpose=False, random_state=0)
    (U2, s2, V2) = randomized_svd(X, k, n_iter=3, transpose=True, random_state=0)
    (U3, s3, V3) = randomized_svd(X, k, n_iter=3, transpose='auto', random_state=0)
    (U4, s4, V4) = linalg.svd(X, full_matrices=False)
    assert_almost_equal(s1, s4[:k], decimal=3)
    assert_almost_equal(s2, s4[:k], decimal=3)
    assert_almost_equal(s3, s4[:k], decimal=3)
    assert_almost_equal(np.dot(U1, V1), np.dot(U4[:, :k], V4[:k, :]), decimal=2)
    assert_almost_equal(np.dot(U2, V2), np.dot(U4[:, :k], V4[:k, :]), decimal=2)
    assert_almost_equal(s2, s3)

def test_randomized_svd_power_iteration_normalizer():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    X = make_low_rank_matrix(100, 500, effective_rank=50, random_state=rng)
    X += 3 * rng.randint(0, 2, size=X.shape)
    n_components = 50
    (U, s, Vt) = randomized_svd(X, n_components, n_iter=2, power_iteration_normalizer='none', random_state=0)
    A = X - U.dot(np.diag(s).dot(Vt))
    error_2 = linalg.norm(A, ord='fro')
    (U, s, Vt) = randomized_svd(X, n_components, n_iter=20, power_iteration_normalizer='none', random_state=0)
    A = X - U.dot(np.diag(s).dot(Vt))
    error_20 = linalg.norm(A, ord='fro')
    assert np.abs(error_2 - error_20) > 100
    for normalizer in ['LU', 'QR', 'auto']:
        (U, s, Vt) = randomized_svd(X, n_components, n_iter=2, power_iteration_normalizer=normalizer, random_state=0)
        A = X - U.dot(np.diag(s).dot(Vt))
        error_2 = linalg.norm(A, ord='fro')
        for i in [5, 10, 50]:
            (U, s, Vt) = randomized_svd(X, n_components, n_iter=i, power_iteration_normalizer=normalizer, random_state=0)
            A = X - U.dot(np.diag(s).dot(Vt))
            error = linalg.norm(A, ord='fro')
            assert 15 > np.abs(error_2 - error)

@pytest.mark.parametrize('sparse_container', DOK_CONTAINERS + LIL_CONTAINERS)
def test_randomized_svd_sparse_warnings(sparse_container):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    X = make_low_rank_matrix(50, 20, effective_rank=10, random_state=rng)
    n_components = 5
    X = sparse_container(X)
    warn_msg = 'Calculating SVD of a {} is expensive. csr_matrix is more efficient.'.format(sparse_container.__name__)
    with pytest.warns(sparse.SparseEfficiencyWarning, match=warn_msg):
        randomized_svd(X, n_components, n_iter=1, power_iteration_normalizer='none')

def test_svd_flip():
    if False:
        for i in range(10):
            print('nop')
    rs = np.random.RandomState(1999)
    n_samples = 20
    n_features = 10
    X = rs.randn(n_samples, n_features)
    (U, S, Vt) = linalg.svd(X, full_matrices=False)
    (U1, V1) = svd_flip(U, Vt, u_based_decision=False)
    assert_almost_equal(np.dot(U1 * S, V1), X, decimal=6)
    XT = X.T
    (U, S, Vt) = linalg.svd(XT, full_matrices=False)
    (U2, V2) = svd_flip(U, Vt, u_based_decision=True)
    assert_almost_equal(np.dot(U2 * S, V2), XT, decimal=6)
    (U_flip1, V_flip1) = svd_flip(U, Vt, u_based_decision=True)
    assert_almost_equal(np.dot(U_flip1 * S, V_flip1), XT, decimal=6)
    (U_flip2, V_flip2) = svd_flip(U, Vt, u_based_decision=False)
    assert_almost_equal(np.dot(U_flip2 * S, V_flip2), XT, decimal=6)

@pytest.mark.parametrize('n_samples, n_features', [(3, 4), (4, 3)])
def test_svd_flip_max_abs_cols(n_samples, n_features, global_random_seed):
    if False:
        print('Hello World!')
    rs = np.random.RandomState(global_random_seed)
    X = rs.randn(n_samples, n_features)
    (U, _, Vt) = linalg.svd(X, full_matrices=False)
    (U1, _) = svd_flip(U, Vt, u_based_decision=True)
    max_abs_U1_row_idx_for_col = np.argmax(np.abs(U1), axis=0)
    assert (U1[max_abs_U1_row_idx_for_col, np.arange(U1.shape[1])] >= 0).all()
    (_, V2) = svd_flip(U, Vt, u_based_decision=False)
    max_abs_V2_col_idx_for_row = np.argmax(np.abs(V2), axis=1)
    assert (V2[np.arange(V2.shape[0]), max_abs_V2_col_idx_for_row] >= 0).all()

def test_randomized_svd_sign_flip():
    if False:
        while True:
            i = 10
    a = np.array([[2.0, 0.0], [0.0, 1.0]])
    (u1, s1, v1) = randomized_svd(a, 2, flip_sign=True, random_state=41)
    for seed in range(10):
        (u2, s2, v2) = randomized_svd(a, 2, flip_sign=True, random_state=seed)
        assert_almost_equal(u1, u2)
        assert_almost_equal(v1, v2)
        assert_almost_equal(np.dot(u2 * s2, v2), a)
        assert_almost_equal(np.dot(u2.T, u2), np.eye(2))
        assert_almost_equal(np.dot(v2.T, v2), np.eye(2))

def test_randomized_svd_sign_flip_with_transpose():
    if False:
        i = 10
        return i + 15

    def max_loading_is_positive(u, v):
        if False:
            for i in range(10):
                print('nop')
        '\n        returns bool tuple indicating if the values maximising np.abs\n        are positive across all rows for u and across all columns for v.\n        '
        u_based = (np.abs(u).max(axis=0) == u.max(axis=0)).all()
        v_based = (np.abs(v).max(axis=1) == v.max(axis=1)).all()
        return (u_based, v_based)
    mat = np.arange(10 * 8).reshape(10, -1)
    (u_flipped, _, v_flipped) = randomized_svd(mat, 3, flip_sign=True, random_state=0)
    (u_based, v_based) = max_loading_is_positive(u_flipped, v_flipped)
    assert u_based
    assert not v_based
    (u_flipped_with_transpose, _, v_flipped_with_transpose) = randomized_svd(mat, 3, flip_sign=True, transpose=True, random_state=0)
    (u_based, v_based) = max_loading_is_positive(u_flipped_with_transpose, v_flipped_with_transpose)
    assert u_based
    assert not v_based

@pytest.mark.parametrize('n', [50, 100, 300])
@pytest.mark.parametrize('m', [50, 100, 300])
@pytest.mark.parametrize('k', [10, 20, 50])
@pytest.mark.parametrize('seed', range(5))
def test_randomized_svd_lapack_driver(n, m, k, seed):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(seed)
    X = rng.rand(n, m)
    (u1, s1, vt1) = randomized_svd(X, k, svd_lapack_driver='gesdd', random_state=0)
    (u2, s2, vt2) = randomized_svd(X, k, svd_lapack_driver='gesvd', random_state=0)
    assert u1.shape == u2.shape
    assert_allclose(u1, u2, atol=0, rtol=0.001)
    assert s1.shape == s2.shape
    assert_allclose(s1, s2, atol=0, rtol=0.001)
    assert vt1.shape == vt2.shape
    assert_allclose(vt1, vt2, atol=0, rtol=0.001)

def test_cartesian():
    if False:
        return 10
    axes = (np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7]))
    true_out = np.array([[1, 4, 6], [1, 4, 7], [1, 5, 6], [1, 5, 7], [2, 4, 6], [2, 4, 7], [2, 5, 6], [2, 5, 7], [3, 4, 6], [3, 4, 7], [3, 5, 6], [3, 5, 7]])
    out = cartesian(axes)
    assert_array_equal(true_out, out)
    x = np.arange(3)
    assert_array_equal(x[:, np.newaxis], cartesian((x,)))

@pytest.mark.parametrize('arrays, output_dtype', [([np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int64)], np.dtype(np.int64)), ([np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.float64)], np.dtype(np.float64)), ([np.array([1, 2, 3], dtype=np.int32), np.array(['x', 'y'], dtype=object)], np.dtype(object))])
def test_cartesian_mix_types(arrays, output_dtype):
    if False:
        while True:
            i = 10
    'Check that the cartesian product works with mixed types.'
    output = cartesian(arrays)
    assert output.dtype == output_dtype

def test_logistic_sigmoid():
    if False:
        for i in range(10):
            print('nop')

    def naive_log_logistic(x):
        if False:
            return 10
        return np.log(expit(x))
    x = np.linspace(-2, 2, 50)
    warn_msg = '`log_logistic` is deprecated and will be removed'
    with pytest.warns(FutureWarning, match=warn_msg):
        assert_array_almost_equal(log_logistic(x), naive_log_logistic(x))
    extreme_x = np.array([-100.0, 100.0])
    with pytest.warns(FutureWarning, match=warn_msg):
        assert_array_almost_equal(log_logistic(extreme_x), [-100, 0])

@pytest.fixture()
def rng():
    if False:
        while True:
            i = 10
    return np.random.RandomState(42)

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_incremental_weighted_mean_and_variance_simple(rng, dtype):
    if False:
        while True:
            i = 10
    mult = 10
    X = rng.rand(1000, 20).astype(dtype) * mult
    sample_weight = rng.rand(X.shape[0]) * mult
    (mean, var, _) = _incremental_mean_and_var(X, 0, 0, 0, sample_weight=sample_weight)
    expected_mean = np.average(X, weights=sample_weight, axis=0)
    expected_var = np.average(X ** 2, weights=sample_weight, axis=0) - expected_mean ** 2
    assert_almost_equal(mean, expected_mean)
    assert_almost_equal(var, expected_var)

@pytest.mark.parametrize('mean', [0, 10000000.0, -10000000.0])
@pytest.mark.parametrize('var', [1, 1e-08, 100000.0])
@pytest.mark.parametrize('weight_loc, weight_scale', [(0, 1), (0, 1e-08), (1, 1e-08), (10, 1), (10000000.0, 1)])
def test_incremental_weighted_mean_and_variance(mean, var, weight_loc, weight_scale, rng):
    if False:
        i = 10
        return i + 15

    def _assert(X, sample_weight, expected_mean, expected_var):
        if False:
            for i in range(10):
                print('nop')
        n = X.shape[0]
        for chunk_size in [1, n // 10 + 1, n // 4 + 1, n // 2 + 1, n]:
            (last_mean, last_weight_sum, last_var) = (0, 0, 0)
            for batch in gen_batches(n, chunk_size):
                (last_mean, last_var, last_weight_sum) = _incremental_mean_and_var(X[batch], last_mean, last_var, last_weight_sum, sample_weight=sample_weight[batch])
            assert_allclose(last_mean, expected_mean)
            assert_allclose(last_var, expected_var, atol=1e-06)
    size = (100, 20)
    weight = rng.normal(loc=weight_loc, scale=weight_scale, size=size[0])
    X = rng.normal(loc=mean, scale=var, size=size)
    expected_mean = _safe_accumulator_op(np.average, X, weights=weight, axis=0)
    expected_var = _safe_accumulator_op(np.average, (X - expected_mean) ** 2, weights=weight, axis=0)
    _assert(X, weight, expected_mean, expected_var)
    X = rng.normal(loc=mean, scale=var, size=size)
    ones_weight = np.ones(size[0])
    expected_mean = _safe_accumulator_op(np.mean, X, axis=0)
    expected_var = _safe_accumulator_op(np.var, X, axis=0)
    _assert(X, ones_weight, expected_mean, expected_var)

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_incremental_weighted_mean_and_variance_ignore_nan(dtype):
    if False:
        while True:
            i = 10
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_weight_sum = np.array([2, 2, 2, 2], dtype=np.int32)
    sample_weights_X = np.ones(3)
    sample_weights_X_nan = np.ones(4)
    X = np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]]).astype(dtype)
    X_nan = np.array([[170, np.nan, 170, 170], [np.nan, 170, 430, 430], [430, 430, np.nan, 300], [300, 300, 300, np.nan]]).astype(dtype)
    (X_means, X_variances, X_count) = _incremental_mean_and_var(X, old_means, old_variances, old_weight_sum, sample_weight=sample_weights_X)
    (X_nan_means, X_nan_variances, X_nan_count) = _incremental_mean_and_var(X_nan, old_means, old_variances, old_weight_sum, sample_weight=sample_weights_X_nan)
    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_variances, X_variances)
    assert_allclose(X_nan_count, X_count)

def test_incremental_variance_update_formulas():
    if False:
        return 10
    A = np.array([[600, 470, 170, 430, 300], [600, 470, 170, 430, 300], [600, 470, 170, 430, 300], [600, 470, 170, 430, 300]]).T
    idx = 2
    X1 = A[:idx, :]
    X2 = A[idx:, :]
    old_means = X1.mean(axis=0)
    old_variances = X1.var(axis=0)
    old_sample_count = np.full(X1.shape[1], X1.shape[0], dtype=np.int32)
    (final_means, final_variances, final_count) = _incremental_mean_and_var(X2, old_means, old_variances, old_sample_count)
    assert_almost_equal(final_means, A.mean(axis=0), 6)
    assert_almost_equal(final_variances, A.var(axis=0), 6)
    assert_almost_equal(final_count, A.shape[0])

def test_incremental_mean_and_variance_ignore_nan():
    if False:
        for i in range(10):
            print('nop')
    old_means = np.array([535.0, 535.0, 535.0, 535.0])
    old_variances = np.array([4225.0, 4225.0, 4225.0, 4225.0])
    old_sample_count = np.array([2, 2, 2, 2], dtype=np.int32)
    X = np.array([[170, 170, 170, 170], [430, 430, 430, 430], [300, 300, 300, 300]])
    X_nan = np.array([[170, np.nan, 170, 170], [np.nan, 170, 430, 430], [430, 430, np.nan, 300], [300, 300, 300, np.nan]])
    (X_means, X_variances, X_count) = _incremental_mean_and_var(X, old_means, old_variances, old_sample_count)
    (X_nan_means, X_nan_variances, X_nan_count) = _incremental_mean_and_var(X_nan, old_means, old_variances, old_sample_count)
    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_variances, X_variances)
    assert_allclose(X_nan_count, X_count)

@skip_if_32bit
def test_incremental_variance_numerical_stability():
    if False:
        return 10

    def np_var(A):
        if False:
            return 10
        return A.var(axis=0)

    def one_pass_var(X):
        if False:
            return 10
        n = X.shape[0]
        exp_x2 = (X ** 2).sum(axis=0) / n
        expx_2 = (X.sum(axis=0) / n) ** 2
        return exp_x2 - expx_2

    def two_pass_var(X):
        if False:
            print('Hello World!')
        mean = X.mean(axis=0)
        Y = X.copy()
        return np.mean((Y - mean) ** 2, axis=0)

    def naive_mean_variance_update(x, last_mean, last_variance, last_sample_count):
        if False:
            for i in range(10):
                print('nop')
        updated_sample_count = last_sample_count + 1
        samples_ratio = last_sample_count / float(updated_sample_count)
        updated_mean = x / updated_sample_count + last_mean * samples_ratio
        updated_variance = last_variance * samples_ratio + (x - last_mean) * (x - updated_mean) / updated_sample_count
        return (updated_mean, updated_variance, updated_sample_count)
    tol = 200
    n_features = 2
    n_samples = 10000
    x1 = np.array(100000000.0, dtype=np.float64)
    x2 = np.log(1e-05, dtype=np.float64)
    A0 = np.full((n_samples // 2, n_features), x1, dtype=np.float64)
    A1 = np.full((n_samples // 2, n_features), x2, dtype=np.float64)
    A = np.vstack((A0, A1))
    assert np.abs(np_var(A) - one_pass_var(A)).max() > tol
    (mean, var, n) = (A0[0, :], np.zeros(n_features), n_samples // 2)
    for i in range(A1.shape[0]):
        (mean, var, n) = naive_mean_variance_update(A1[i, :], mean, var, n)
    assert n == A.shape[0]
    assert np.abs(A.mean(axis=0) - mean).max() > 1e-06
    assert np.abs(np_var(A) - var).max() > tol
    (mean, var) = (A0[0, :], np.zeros(n_features))
    n = np.full(n_features, n_samples // 2, dtype=np.int32)
    for i in range(A1.shape[0]):
        (mean, var, n) = _incremental_mean_and_var(A1[i, :].reshape((1, A1.shape[1])), mean, var, n)
    assert_array_equal(n, A.shape[0])
    assert_array_almost_equal(A.mean(axis=0), mean)
    assert tol > np.abs(np_var(A) - var).max()

def test_incremental_variance_ddof():
    if False:
        return 10
    rng = np.random.RandomState(1999)
    X = rng.randn(50, 10)
    (n_samples, n_features) = X.shape
    for batch_size in [11, 20, 37]:
        steps = np.arange(0, X.shape[0], batch_size)
        if steps[-1] != X.shape[0]:
            steps = np.hstack([steps, n_samples])
        for (i, j) in zip(steps[:-1], steps[1:]):
            batch = X[i:j, :]
            if i == 0:
                incremental_means = batch.mean(axis=0)
                incremental_variances = batch.var(axis=0)
                incremental_count = batch.shape[0]
                sample_count = np.full(batch.shape[1], batch.shape[0], dtype=np.int32)
            else:
                result = _incremental_mean_and_var(batch, incremental_means, incremental_variances, sample_count)
                (incremental_means, incremental_variances, incremental_count) = result
                sample_count += batch.shape[0]
            calculated_means = np.mean(X[:j], axis=0)
            calculated_variances = np.var(X[:j], axis=0)
            assert_almost_equal(incremental_means, calculated_means, 6)
            assert_almost_equal(incremental_variances, calculated_variances, 6)
            assert_array_equal(incremental_count, sample_count)

def test_vector_sign_flip():
    if False:
        print('Hello World!')
    data = np.random.RandomState(36).randn(5, 5)
    max_abs_rows = np.argmax(np.abs(data), axis=1)
    data_flipped = _deterministic_vector_sign_flip(data)
    max_rows = np.argmax(data_flipped, axis=1)
    assert_array_equal(max_abs_rows, max_rows)
    signs = np.sign(data[range(data.shape[0]), max_abs_rows])
    assert_array_equal(data, data_flipped * signs[:, np.newaxis])

def test_softmax():
    if False:
        return 10
    rng = np.random.RandomState(0)
    X = rng.randn(3, 5)
    exp_X = np.exp(X)
    sum_exp_X = np.sum(exp_X, axis=1).reshape((-1, 1))
    assert_array_almost_equal(softmax(X), exp_X / sum_exp_X)

def test_stable_cumsum():
    if False:
        i = 10
        return i + 15
    assert_array_equal(stable_cumsum([1, 2, 3]), np.cumsum([1, 2, 3]))
    r = np.random.RandomState(0).rand(100000)
    with pytest.warns(RuntimeWarning):
        stable_cumsum(r, rtol=0, atol=0)
    A = np.random.RandomState(36).randint(1000, size=(5, 5, 5))
    assert_array_equal(stable_cumsum(A, axis=0), np.cumsum(A, axis=0))
    assert_array_equal(stable_cumsum(A, axis=1), np.cumsum(A, axis=1))
    assert_array_equal(stable_cumsum(A, axis=2), np.cumsum(A, axis=2))

@pytest.mark.parametrize('A_container', [np.array, *CSR_CONTAINERS], ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
@pytest.mark.parametrize('B_container', [np.array, *CSR_CONTAINERS], ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
def test_safe_sparse_dot_2d(A_container, B_container):
    if False:
        return 10
    rng = np.random.RandomState(0)
    A = rng.random_sample((30, 10))
    B = rng.random_sample((10, 20))
    expected = np.dot(A, B)
    A = A_container(A)
    B = B_container(B)
    actual = safe_sparse_dot(A, B, dense_output=True)
    assert_allclose(actual, expected)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_safe_sparse_dot_nd(csr_container):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    A = rng.random_sample((2, 3, 4, 5, 6))
    B = rng.random_sample((6, 7))
    expected = np.dot(A, B)
    B = csr_container(B)
    actual = safe_sparse_dot(A, B)
    assert_allclose(actual, expected)
    A = rng.random_sample((2, 3))
    B = rng.random_sample((4, 5, 3, 6))
    expected = np.dot(A, B)
    A = csr_container(A)
    actual = safe_sparse_dot(A, B)
    assert_allclose(actual, expected)

@pytest.mark.parametrize('container', [np.array, *CSR_CONTAINERS], ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
def test_safe_sparse_dot_2d_1d(container):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    B = rng.random_sample(10)
    A = rng.random_sample((30, 10))
    expected = np.dot(A, B)
    actual = safe_sparse_dot(container(A), B)
    assert_allclose(actual, expected)
    A = rng.random_sample((10, 30))
    expected = np.dot(B, A)
    actual = safe_sparse_dot(B, container(A))
    assert_allclose(actual, expected)

@pytest.mark.parametrize('dense_output', [True, False])
def test_safe_sparse_dot_dense_output(dense_output):
    if False:
        return 10
    rng = np.random.RandomState(0)
    A = sparse.random(30, 10, density=0.1, random_state=rng)
    B = sparse.random(10, 20, density=0.1, random_state=rng)
    expected = A.dot(B)
    actual = safe_sparse_dot(A, B, dense_output=dense_output)
    assert sparse.issparse(actual) == (not dense_output)
    if dense_output:
        expected = expected.toarray()
    assert_allclose_dense_sparse(actual, expected)