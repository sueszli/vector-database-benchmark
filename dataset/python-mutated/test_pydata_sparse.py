import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
try:
    import sparse
except Exception:
    sparse = None
pytestmark = pytest.mark.skipif(sparse is None, reason='pydata/sparse not installed')
msg = 'pydata/sparse (0.8) does not implement necessary operations'
sparse_params = (pytest.param('COO'), pytest.param('DOK', marks=[pytest.mark.xfail(reason=msg)]))
scipy_sparse_classes = [sp.bsr_matrix, sp.csr_matrix, sp.coo_matrix, sp.csc_matrix, sp.dia_matrix, sp.dok_matrix]

@pytest.fixture(params=sparse_params)
def sparse_cls(request):
    if False:
        print('Hello World!')
    return getattr(sparse, request.param)

@pytest.fixture(params=scipy_sparse_classes)
def sp_sparse_cls(request):
    if False:
        return 10
    return request.param

@pytest.fixture
def same_matrix(sparse_cls, sp_sparse_cls):
    if False:
        while True:
            i = 10
    np.random.seed(1234)
    A_dense = np.random.rand(9, 9)
    return (sp_sparse_cls(A_dense), sparse_cls(A_dense))

@pytest.fixture
def matrices(sparse_cls):
    if False:
        i = 10
        return i + 15
    np.random.seed(1234)
    A_dense = np.random.rand(9, 9)
    A_dense = A_dense @ A_dense.T
    A_sparse = sparse_cls(A_dense)
    b = np.random.rand(9)
    return (A_dense, A_sparse, b)

def test_isolve_gmres(matrices):
    if False:
        return 10
    (A_dense, A_sparse, b) = matrices
    (x, info) = splin.gmres(A_sparse, b, atol=1e-15)
    assert info == 0
    assert isinstance(x, np.ndarray)
    assert_allclose(A_sparse @ x, b)

def test_lsmr(matrices):
    if False:
        i = 10
        return i + 15
    (A_dense, A_sparse, b) = matrices
    res0 = splin.lsmr(A_dense, b)
    res = splin.lsmr(A_sparse, b)
    assert_allclose(res[0], res0[0], atol=1.8e-05)

def test_lsmr_output_shape():
    if False:
        i = 10
        return i + 15
    x = splin.lsmr(A=np.ones((10, 1)), b=np.zeros(10), x0=np.ones(1))[0]
    assert_equal(x.shape, (1,))

def test_lsqr(matrices):
    if False:
        for i in range(10):
            print('nop')
    (A_dense, A_sparse, b) = matrices
    res0 = splin.lsqr(A_dense, b)
    res = splin.lsqr(A_sparse, b)
    assert_allclose(res[0], res0[0], atol=1e-05)

def test_eigs(matrices):
    if False:
        for i in range(10):
            print('nop')
    (A_dense, A_sparse, v0) = matrices
    M_dense = np.diag(v0 ** 2)
    M_sparse = A_sparse.__class__(M_dense)
    (w_dense, v_dense) = splin.eigs(A_dense, k=3, v0=v0)
    (w, v) = splin.eigs(A_sparse, k=3, v0=v0)
    assert_allclose(w, w_dense)
    assert_allclose(v, v_dense)
    for M in [M_sparse, M_dense]:
        (w_dense, v_dense) = splin.eigs(A_dense, M=M_dense, k=3, v0=v0)
        (w, v) = splin.eigs(A_sparse, M=M, k=3, v0=v0)
        assert_allclose(w, w_dense)
        assert_allclose(v, v_dense)
        (w_dense, v_dense) = splin.eigsh(A_dense, M=M_dense, k=3, v0=v0)
        (w, v) = splin.eigsh(A_sparse, M=M, k=3, v0=v0)
        assert_allclose(w, w_dense)
        assert_allclose(v, v_dense)

def test_svds(matrices):
    if False:
        return 10
    (A_dense, A_sparse, v0) = matrices
    (u0, s0, vt0) = splin.svds(A_dense, k=2, v0=v0)
    (u, s, vt) = splin.svds(A_sparse, k=2, v0=v0)
    assert_allclose(s, s0)
    assert_allclose(u, u0)
    assert_allclose(vt, vt0)

def test_lobpcg(matrices):
    if False:
        return 10
    (A_dense, A_sparse, x) = matrices
    X = x[:, None]
    (w_dense, v_dense) = splin.lobpcg(A_dense, X)
    (w, v) = splin.lobpcg(A_sparse, X)
    assert_allclose(w, w_dense)
    assert_allclose(v, v_dense)

def test_spsolve(matrices):
    if False:
        for i in range(10):
            print('nop')
    (A_dense, A_sparse, b) = matrices
    b2 = np.random.rand(len(b), 3)
    x0 = splin.spsolve(sp.csc_matrix(A_dense), b)
    x = splin.spsolve(A_sparse, b)
    assert isinstance(x, np.ndarray)
    assert_allclose(x, x0)
    x0 = splin.spsolve(sp.csc_matrix(A_dense), b)
    x = splin.spsolve(A_sparse, b, use_umfpack=True)
    assert isinstance(x, np.ndarray)
    assert_allclose(x, x0)
    x0 = splin.spsolve(sp.csc_matrix(A_dense), b2)
    x = splin.spsolve(A_sparse, b2)
    assert isinstance(x, np.ndarray)
    assert_allclose(x, x0)
    x0 = splin.spsolve(sp.csc_matrix(A_dense), sp.csc_matrix(A_dense))
    x = splin.spsolve(A_sparse, A_sparse)
    assert isinstance(x, type(A_sparse))
    assert_allclose(x.toarray(), x0.toarray())

def test_splu(matrices):
    if False:
        print('Hello World!')
    (A_dense, A_sparse, b) = matrices
    n = len(b)
    sparse_cls = type(A_sparse)
    lu = splin.splu(A_sparse)
    assert isinstance(lu.L, sparse_cls)
    assert isinstance(lu.U, sparse_cls)
    Pr = sparse_cls(sp.csc_matrix((np.ones(n), (lu.perm_r, np.arange(n)))))
    Pc = sparse_cls(sp.csc_matrix((np.ones(n), (np.arange(n), lu.perm_c))))
    A2 = Pr.T @ lu.L @ lu.U @ Pc.T
    assert_allclose(A2.toarray(), A_sparse.toarray())
    z = lu.solve(A_sparse.toarray())
    assert_allclose(z, np.eye(n), atol=1e-10)

def test_spilu(matrices):
    if False:
        for i in range(10):
            print('nop')
    (A_dense, A_sparse, b) = matrices
    sparse_cls = type(A_sparse)
    lu = splin.spilu(A_sparse)
    assert isinstance(lu.L, sparse_cls)
    assert isinstance(lu.U, sparse_cls)
    z = lu.solve(A_sparse.toarray())
    assert_allclose(z, np.eye(len(b)), atol=0.001)

def test_spsolve_triangular(matrices):
    if False:
        return 10
    (A_dense, A_sparse, b) = matrices
    A_sparse = sparse.tril(A_sparse)
    x = splin.spsolve_triangular(A_sparse, b)
    assert_allclose(A_sparse @ x, b)

def test_onenormest(matrices):
    if False:
        print('Hello World!')
    (A_dense, A_sparse, b) = matrices
    est0 = splin.onenormest(A_dense)
    est = splin.onenormest(A_sparse)
    assert_allclose(est, est0)

def test_inv(matrices):
    if False:
        while True:
            i = 10
    (A_dense, A_sparse, b) = matrices
    x0 = splin.inv(sp.csc_matrix(A_dense))
    x = splin.inv(A_sparse)
    assert_allclose(x.toarray(), x0.toarray())

def test_expm(matrices):
    if False:
        for i in range(10):
            print('nop')
    (A_dense, A_sparse, b) = matrices
    x0 = splin.expm(sp.csc_matrix(A_dense))
    x = splin.expm(A_sparse)
    assert_allclose(x.toarray(), x0.toarray())

def test_expm_multiply(matrices):
    if False:
        i = 10
        return i + 15
    (A_dense, A_sparse, b) = matrices
    x0 = splin.expm_multiply(A_dense, b)
    x = splin.expm_multiply(A_sparse, b)
    assert_allclose(x, x0)

def test_eq(same_matrix):
    if False:
        i = 10
        return i + 15
    (sp_sparse, pd_sparse) = same_matrix
    assert (sp_sparse == pd_sparse).all()

def test_ne(same_matrix):
    if False:
        print('Hello World!')
    (sp_sparse, pd_sparse) = same_matrix
    assert not (sp_sparse != pd_sparse).any()