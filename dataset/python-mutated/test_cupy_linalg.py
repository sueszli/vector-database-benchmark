from __future__ import annotations
import numpy as np
import pytest
from packaging.version import parse as parse_version
pytestmark = pytest.mark.gpu
import dask.array as da
from dask.array.utils import assert_eq
cupy = pytest.importorskip('cupy')
cupy_version = parse_version(cupy.__version__)

@pytest.mark.skipif(cupy_version < parse_version('6.1.0'), reason='Requires CuPy 6.1.0+ (with https://github.com/cupy/cupy/pull/2209)')
@pytest.mark.parametrize('m,n,chunks,error_type', [(20, 10, 10, None), (20, 10, (3, 10), None), (20, 10, ((8, 4, 8), 10), None), (40, 10, ((15, 5, 5, 8, 7), 10), None), (128, 2, (16, 2), None), (129, 2, (16, 2), None), (130, 2, (16, 2), None), (131, 2, (16, 2), None), (300, 10, (40, 10), None), (300, 10, (30, 10), None), (300, 10, (20, 10), None), (10, 5, 10, None), (5, 10, 10, None), (10, 10, 10, None), (10, 40, (10, 10), ValueError), (10, 40, (10, 15), ValueError), (10, 40, (10, (15, 5, 5, 8, 7)), ValueError), (20, 20, 10, ValueError)])
def test_tsqr(m, n, chunks, error_type):
    if False:
        i = 10
        return i + 15
    mat = cupy.random.default_rng().random((m, n))
    data = da.from_array(mat, chunks=chunks, name='A', asarray=False)
    m_q = m
    n_q = min(m, n)
    m_r = n_q
    n_r = n
    m_u = m
    n_u = min(m, n)
    n_s = n_q
    m_vh = n_q
    n_vh = n
    d_vh = max(m_vh, n_vh)
    if error_type is None:
        (q, r) = da.linalg.tsqr(data)
        assert_eq((m_q, n_q), q.shape)
        assert_eq((m_r, n_r), r.shape)
        assert_eq(mat, da.dot(q, r))
        assert_eq(cupy.eye(n_q, n_q), da.dot(q.T, q))
        assert_eq(r, np.triu(r.rechunk(r.shape[0])))
        (u, s, vh) = da.linalg.tsqr(data, compute_svd=True)
        s_exact = np.linalg.svd(mat)[1]
        assert_eq(s, s_exact)
        assert_eq((m_u, n_u), u.shape)
        assert_eq((n_s,), s.shape)
        assert_eq((d_vh, d_vh), vh.shape)
        assert_eq(np.eye(n_u, n_u), da.dot(u.T, u), check_type=False)
        assert_eq(np.eye(d_vh, d_vh), da.dot(vh, vh.T), check_type=False)
        assert_eq(mat, da.dot(da.dot(u, da.diag(s)), vh[:n_q]))
    else:
        with pytest.raises(error_type):
            (q, r) = da.linalg.tsqr(data)
        with pytest.raises(error_type):
            (u, s, vh) = da.linalg.tsqr(data, compute_svd=True)

@pytest.mark.parametrize('m_min,n_max,chunks,vary_rows,vary_cols,error_type', [(10, 5, (10, 5), True, False, None), (10, 5, (10, 5), False, True, None), (10, 5, (10, 5), True, True, None), (40, 5, (10, 5), True, False, None), (40, 5, (10, 5), False, True, None), (40, 5, (10, 5), True, True, None), (300, 10, (40, 10), True, False, None), (300, 10, (30, 10), True, False, None), (300, 10, (20, 10), True, False, None), (300, 10, (40, 10), False, True, None), (300, 10, (30, 10), False, True, None), (300, 10, (20, 10), False, True, None), (300, 10, (40, 10), True, True, None), (300, 10, (30, 10), True, True, None), (300, 10, (20, 10), True, True, None)])
def test_tsqr_uncertain(m_min, n_max, chunks, vary_rows, vary_cols, error_type):
    if False:
        i = 10
        return i + 15
    mat = cupy.random.default_rng().random((m_min * 2, n_max))
    (m, n) = (m_min * 2, n_max)
    mat[0:m_min, 0] += 1
    _c0 = mat[:, 0]
    _r0 = mat[0, :]
    c0 = da.from_array(_c0, chunks=m_min, name='c', asarray=False)
    r0 = da.from_array(_r0, chunks=n_max, name='r', asarray=False)
    data = da.from_array(mat, chunks=chunks, name='A', asarray=False)
    if vary_rows:
        data = data[c0 > 0.5, :]
        mat = mat[_c0 > 0.5, :]
        m = mat.shape[0]
    if vary_cols:
        data = data[:, r0 > 0.5]
        mat = mat[:, _r0 > 0.5]
        n = mat.shape[1]
    m_q = m
    n_q = min(m, n)
    m_r = n_q
    n_r = n
    m_u = m
    n_u = min(m, n)
    n_s = n_q
    m_vh = n_q
    n_vh = n
    d_vh = max(m_vh, n_vh)
    if error_type is None:
        (q, r) = da.linalg.tsqr(data)
        q = q.compute()
        r = r.compute()
        assert_eq((m_q, n_q), q.shape)
        assert_eq((m_r, n_r), r.shape)
        assert_eq(mat, np.dot(q, r))
        assert_eq(np.eye(n_q, n_q), np.dot(q.T, q), check_type=False)
        assert_eq(r, np.triu(r))
        (u, s, vh) = da.linalg.tsqr(data, compute_svd=True)
        u = u.compute()
        s = s.compute()
        vh = vh.compute()
        s_exact = np.linalg.svd(mat)[1]
        assert_eq(s, s_exact)
        assert_eq((m_u, n_u), u.shape)
        assert_eq((n_s,), s.shape)
        assert_eq((d_vh, d_vh), vh.shape)
        assert_eq(np.eye(n_u, n_u), np.dot(u.T, u), check_type=False)
        assert_eq(np.eye(d_vh, d_vh), np.dot(vh, vh.T), check_type=False)
        assert_eq(mat, np.dot(np.dot(u, np.diag(s)), vh[:n_q]), check_type=False)
    else:
        with pytest.raises(error_type):
            (q, r) = da.linalg.tsqr(data)
        with pytest.raises(error_type):
            (u, s, vh) = da.linalg.tsqr(data, compute_svd=True)

@pytest.mark.parametrize('m,n,chunks,error_type', [(20, 10, 10, ValueError), (20, 10, (3, 10), ValueError), (20, 10, ((8, 4, 8), 10), ValueError), (40, 10, ((15, 5, 5, 8, 7), 10), ValueError), (128, 2, (16, 2), ValueError), (129, 2, (16, 2), ValueError), (130, 2, (16, 2), ValueError), (131, 2, (16, 2), ValueError), (300, 10, (40, 10), ValueError), (300, 10, (30, 10), ValueError), (300, 10, (20, 10), ValueError), (10, 5, 10, None), (5, 10, 10, None), (10, 10, 10, None), (10, 40, (10, 10), None), (10, 40, (10, 15), None), (10, 40, (10, (15, 5, 5, 8, 7)), None), (20, 20, 10, ValueError)])
def test_sfqr(m, n, chunks, error_type):
    if False:
        for i in range(10):
            print('nop')
    mat = np.random.default_rng().random((m, n))
    data = da.from_array(mat, chunks=chunks, name='A')
    m_q = m
    n_q = min(m, n)
    m_r = n_q
    n_r = n
    m_qtq = n_q
    if error_type is None:
        (q, r) = da.linalg.sfqr(data)
        assert_eq((m_q, n_q), q.shape)
        assert_eq((m_r, n_r), r.shape)
        assert_eq(mat, da.dot(q, r))
        assert_eq(np.eye(m_qtq, m_qtq), da.dot(q.T, q))
        assert_eq(r, da.triu(r.rechunk(r.shape[0])))
    else:
        with pytest.raises(error_type):
            (q, r) = da.linalg.sfqr(data)

@pytest.mark.parametrize('iscomplex', [False, True])
@pytest.mark.parametrize(('nrow', 'ncol', 'chunk'), [(20, 10, 5), (100, 10, 10)])
def test_lstsq(nrow, ncol, chunk, iscomplex):
    if False:
        return 10
    rng = cupy.random.default_rng(1)
    A = rng.integers(1, 20, (nrow, ncol))
    b = rng.integers(1, 20, nrow)
    if iscomplex:
        A = A + 1j * rng.integers(1, 20, A.shape)
        b = b + 1j * rng.integers(1, 20, b.shape)
    dA = da.from_array(A, (chunk, ncol))
    db = da.from_array(b, chunk)
    (x, r, rank, s) = cupy.linalg.lstsq(A, b, rcond=-1)
    (dx, dr, drank, ds) = da.linalg.lstsq(dA, db)
    assert_eq(dx, x)
    assert_eq(dr, r)
    assert drank.compute() == rank
    assert_eq(ds, s)
    A[:, 1] = A[:, 2]
    dA = da.from_array(A, (chunk, ncol))
    db = da.from_array(b, chunk)
    (x, r, rank, s) = cupy.linalg.lstsq(A, b, rcond=cupy.finfo(cupy.double).eps * max(nrow, ncol))
    assert rank == ncol - 1
    (dx, dr, drank, ds) = da.linalg.lstsq(dA, db)
    assert drank.compute() == rank
    A = rng.integers(1, 20, (nrow, ncol))
    b2D = rng.integers(1, 20, (nrow, ncol // 2))
    if iscomplex:
        A = A + 1j * rng.integers(1, 20, A.shape)
        b2D = b2D + 1j * rng.integers(1, 20, b2D.shape)
    dA = da.from_array(A, (chunk, ncol))
    db2D = da.from_array(b2D, (chunk, ncol // 2))
    (x, r, rank, s) = cupy.linalg.lstsq(A, b2D, rcond=-1)
    (dx, dr, drank, ds) = da.linalg.lstsq(dA, db2D)
    assert_eq(dx, x)
    assert_eq(dr, r)
    assert drank.compute() == rank
    assert_eq(ds, s)

def _get_symmat(size):
    if False:
        print('Hello World!')
    rng = cupy.random.default_rng(1)
    A = rng.integers(1, 21, (size, size))
    lA = cupy.tril(A)
    return lA.dot(lA.T)

@pytest.mark.parametrize(('shape', 'chunk'), [(20, 10), (12, 3), (30, 3), (30, 6)])
def test_cholesky(shape, chunk):
    if False:
        while True:
            i = 10
    scipy_linalg = pytest.importorskip('scipy.linalg')
    A = _get_symmat(shape)
    dA = da.from_array(A, (chunk, chunk))
    assert_eq(da.linalg.cholesky(dA), cupy.linalg.cholesky(A).T, check_graph=False, check_chunks=False)
    assert_eq(da.linalg.cholesky(dA, lower=True).map_blocks(cupy.asnumpy), scipy_linalg.cholesky(cupy.asnumpy(A), lower=True), check_graph=False, check_chunks=False)