import os
import re
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest
from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
if os.environ.get('SCIPY_USE_PROPACK'):
    has_propack = True
else:
    has_propack = False
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence

def sorted_svd(m, k, which='LM'):
    if False:
        return 10
    if issparse(m):
        m = m.toarray()
    (u, s, vh) = svd(m)
    if which == 'LM':
        ii = np.argsort(s)[-k:]
    elif which == 'SM':
        ii = np.argsort(s)[:k]
    else:
        raise ValueError(f'unknown which={which!r}')
    return (u[:, ii], s[ii], vh[ii])

def _check_svds(A, k, u, s, vh, which='LM', check_usvh_A=False, check_svd=True, atol=1e-10, rtol=1e-07):
    if False:
        print('Hello World!')
    (n, m) = A.shape
    assert_equal(u.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(vh.shape, (k, m))
    A_rebuilt = (u * s).dot(vh)
    assert_equal(A_rebuilt.shape, A.shape)
    if check_usvh_A:
        assert_allclose(A_rebuilt, A, atol=atol, rtol=rtol)
    uh_u = np.dot(u.T.conj(), u)
    assert_equal(uh_u.shape, (k, k))
    assert_allclose(uh_u, np.identity(k), atol=atol, rtol=rtol)
    vh_v = np.dot(vh, vh.T.conj())
    assert_equal(vh_v.shape, (k, k))
    assert_allclose(vh_v, np.identity(k), atol=atol, rtol=rtol)
    if check_svd:
        (u2, s2, vh2) = sorted_svd(A, k, which)
        assert_allclose(np.abs(u), np.abs(u2), atol=atol, rtol=rtol)
        assert_allclose(s, s2, atol=atol, rtol=rtol)
        assert_allclose(np.abs(vh), np.abs(vh2), atol=atol, rtol=rtol)

def _check_svds_n(A, k, u, s, vh, which='LM', check_res=True, check_svd=True, atol=1e-10, rtol=1e-07):
    if False:
        for i in range(10):
            print('nop')
    (n, m) = A.shape
    assert_equal(u.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(vh.shape, (k, m))
    uh_u = np.dot(u.T.conj(), u)
    assert_equal(uh_u.shape, (k, k))
    error = np.sum(np.abs(uh_u - np.identity(k))) / (k * k)
    assert_allclose(error, 0.0, atol=atol, rtol=rtol)
    vh_v = np.dot(vh, vh.T.conj())
    assert_equal(vh_v.shape, (k, k))
    error = np.sum(np.abs(vh_v - np.identity(k))) / (k * k)
    assert_allclose(error, 0.0, atol=atol, rtol=rtol)
    if check_res:
        ru = A.T.conj() @ u - vh.T.conj() * s
        rus = np.sum(np.abs(ru)) / (n * k)
        rvh = A @ vh.T.conj() - u * s
        rvhs = np.sum(np.abs(rvh)) / (m * k)
        assert_allclose(rus, 0.0, atol=atol, rtol=rtol)
        assert_allclose(rvhs, 0.0, atol=atol, rtol=rtol)
    if check_svd:
        (u2, s2, vh2) = sorted_svd(A, k, which)
        assert_allclose(s, s2, atol=atol, rtol=rtol)
        A_rebuilt_svd = (u2 * s2).dot(vh2)
        A_rebuilt = (u * s).dot(vh)
        assert_equal(A_rebuilt.shape, A.shape)
        error = np.sum(np.abs(A_rebuilt_svd - A_rebuilt)) / (k * k)
        assert_allclose(error, 0.0, atol=atol, rtol=rtol)

class CheckingLinearOperator(LinearOperator):

    def __init__(self, A):
        if False:
            for i in range(10):
                print('nop')
        self.A = A
        self.dtype = A.dtype
        self.shape = A.shape

    def _matvec(self, x):
        if False:
            i = 10
            return i + 15
        assert_equal(max(x.shape), np.size(x))
        return self.A.dot(x)

    def _rmatvec(self, x):
        if False:
            i = 10
            return i + 15
        assert_equal(max(x.shape), np.size(x))
        return self.A.T.conjugate().dot(x)

class SVDSCommonTests:
    solver = None
    _A_empty_msg = '`A` must not be empty.'
    _A_dtype_msg = '`A` must be of floating or complex floating data type'
    _A_type_msg = 'type not understood'
    _A_ndim_msg = 'array must have ndim <= 2'
    _A_validation_inputs = [(np.asarray([[]]), ValueError, _A_empty_msg), (np.asarray([[1, 2], [3, 4]]), ValueError, _A_dtype_msg), ('hi', TypeError, _A_type_msg), (np.asarray([[[1.0, 2.0], [3.0, 4.0]]]), ValueError, _A_ndim_msg)]

    @pytest.mark.parametrize('args', _A_validation_inputs)
    def test_svds_input_validation_A(self, args):
        if False:
            for i in range(10):
                print('nop')
        (A, error_type, message) = args
        with pytest.raises(error_type, match=message):
            svds(A, k=1, solver=self.solver)

    @pytest.mark.parametrize('k', [-1, 0, 3, 4, 5, 1.5, '1'])
    def test_svds_input_validation_k_1(self, k):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(0)
        A = rng.random((4, 3))
        if self.solver == 'propack' and k == 3:
            if not has_propack:
                pytest.skip('PROPACK not enabled')
            res = svds(A, k=k, solver=self.solver)
            _check_svds(A, k, *res, check_usvh_A=True, check_svd=True)
            return
        message = '`k` must be an integer satisfying'
        with pytest.raises(ValueError, match=message):
            svds(A, k=k, solver=self.solver)

    def test_svds_input_validation_k_2(self):
        if False:
            i = 10
            return i + 15
        message = 'int() argument must be a'
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), k=[], solver=self.solver)
        message = 'invalid literal for int()'
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), k='hi', solver=self.solver)

    @pytest.mark.parametrize('tol', (-1, np.inf, np.nan))
    def test_svds_input_validation_tol_1(self, tol):
        if False:
            i = 10
            return i + 15
        message = '`tol` must be a non-negative floating point value.'
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), tol=tol, solver=self.solver)

    @pytest.mark.parametrize('tol', ([], 'hi'))
    def test_svds_input_validation_tol_2(self, tol):
        if False:
            return 10
        message = "'<' not supported between instances"
        with pytest.raises(TypeError, match=message):
            svds(np.eye(10), tol=tol, solver=self.solver)

    @pytest.mark.parametrize('which', ('LA', 'SA', 'ekki', 0))
    def test_svds_input_validation_which(self, which):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='`which` must be in'):
            svds(np.eye(10), which=which, solver=self.solver)

    @pytest.mark.parametrize('transpose', (True, False))
    @pytest.mark.parametrize('n', range(4, 9))
    def test_svds_input_validation_v0_1(self, transpose, n):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(0)
        A = rng.random((5, 7))
        v0 = rng.random(n)
        if transpose:
            A = A.T
        k = 2
        message = '`v0` must have shape'
        required_length = A.shape[0] if self.solver == 'propack' else min(A.shape)
        if n != required_length:
            with pytest.raises(ValueError, match=message):
                svds(A, k=k, v0=v0, solver=self.solver)

    def test_svds_input_validation_v0_2(self):
        if False:
            while True:
                i = 10
        A = np.ones((10, 10))
        v0 = np.ones((1, 10))
        message = '`v0` must have shape'
        with pytest.raises(ValueError, match=message):
            svds(A, k=1, v0=v0, solver=self.solver)

    @pytest.mark.parametrize('v0', ('hi', 1, np.ones(10, dtype=int)))
    def test_svds_input_validation_v0_3(self, v0):
        if False:
            i = 10
            return i + 15
        A = np.ones((10, 10))
        message = '`v0` must be of floating or complex floating data type.'
        with pytest.raises(ValueError, match=message):
            svds(A, k=1, v0=v0, solver=self.solver)

    @pytest.mark.parametrize('maxiter', (-1, 0, 5.5))
    def test_svds_input_validation_maxiter_1(self, maxiter):
        if False:
            while True:
                i = 10
        message = '`maxiter` must be a positive integer.'
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), maxiter=maxiter, solver=self.solver)

    def test_svds_input_validation_maxiter_2(self):
        if False:
            i = 10
            return i + 15
        message = 'int() argument must be a'
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), maxiter=[], solver=self.solver)
        message = 'invalid literal for int()'
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), maxiter='hi', solver=self.solver)

    @pytest.mark.parametrize('rsv', ('ekki', 10))
    def test_svds_input_validation_return_singular_vectors(self, rsv):
        if False:
            while True:
                i = 10
        message = '`return_singular_vectors` must be in'
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), return_singular_vectors=rsv, solver=self.solver)

    @pytest.mark.parametrize('k', [3, 5])
    @pytest.mark.parametrize('which', ['LM', 'SM'])
    def test_svds_parameter_k_which(self, k, which):
        if False:
            i = 10
            return i + 15
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        rng = np.random.default_rng(0)
        A = rng.random((10, 10))
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match='The problem size'):
                res = svds(A, k=k, which=which, solver=self.solver, random_state=0)
        else:
            res = svds(A, k=k, which=which, solver=self.solver, random_state=0)
        _check_svds(A, k, *res, which=which, atol=8e-10)

    def test_svds_parameter_tol(self):
        if False:
            for i in range(10):
                print('nop')
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        return
        n = 100
        k = 3
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        A[A > 0.1] = 0
        A = A @ A.T
        (_, s, _) = svd(A)
        A = csc_matrix(A)

        def err(tol):
            if False:
                return 10
            if self.solver == 'lobpcg' and tol == 0.0001:
                with pytest.warns(UserWarning, match='Exited at iteration'):
                    (_, s2, _) = svds(A, k=k, v0=np.ones(n), solver=self.solver, tol=tol)
            else:
                (_, s2, _) = svds(A, k=k, v0=np.ones(n), solver=self.solver, tol=tol)
            return np.linalg.norm((s2 - s[k - 1::-1]) / s[k - 1::-1])
        tols = [0.0001, 0.01, 1.0]
        accuracies = {'propack': [1e-12, 1e-06, 0.0001], 'arpack': [2e-15, 1e-10, 1e-10], 'lobpcg': [1e-11, 0.001, 10]}
        for (tol, accuracy) in zip(tols, accuracies[self.solver]):
            error = err(tol)
            assert error < accuracy
            assert error > accuracy / 10

    def test_svd_v0(self):
        if False:
            print('Hello World!')
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        n = 100
        k = 1
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        v0a = rng.random(n)
        res1a = svds(A, k, v0=v0a, solver=self.solver, random_state=0)
        res2a = svds(A, k, v0=v0a, solver=self.solver, random_state=1)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)
        v0b = rng.random(n)
        res1b = svds(A, k, v0=v0b, solver=self.solver, random_state=2)
        res2b = svds(A, k, v0=v0b, solver=self.solver, random_state=3)
        for idx in range(3):
            assert_allclose(res1b[idx], res2b[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1b)
        message = 'Arrays are not equal'
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res1b)

    def test_svd_random_state(self):
        if False:
            while True:
                i = 10
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        n = 100
        k = 1
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        res1a = svds(A, k, solver=self.solver, random_state=0)
        res2a = svds(A, k, solver=self.solver, random_state=0)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)
        res1b = svds(A, k, solver=self.solver, random_state=1)
        res2b = svds(A, k, solver=self.solver, random_state=1)
        for idx in range(3):
            assert_allclose(res1b[idx], res2b[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1b)
        message = 'Arrays are not equal'
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res1b)

    @pytest.mark.parametrize('random_state', (0, 1, np.random.RandomState(0), np.random.default_rng(0)))
    def test_svd_random_state_2(self, random_state):
        if False:
            i = 10
            return i + 15
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        n = 100
        k = 1
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        random_state_2 = copy.deepcopy(random_state)
        res1a = svds(A, k, solver=self.solver, random_state=random_state)
        res2a = svds(A, k, solver=self.solver, random_state=random_state_2)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)

    @pytest.mark.parametrize('random_state', (None, np.random.RandomState(0), np.random.default_rng(0)))
    def test_svd_random_state_3(self, random_state):
        if False:
            i = 10
            return i + 15
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        n = 100
        k = 5
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        res1a = svds(A, k, solver=self.solver, random_state=random_state)
        res2a = svds(A, k, solver=self.solver, random_state=random_state)
        _check_svds(A, k, *res1a, atol=2e-10, rtol=1e-06)
        _check_svds(A, k, *res2a, atol=2e-10, rtol=1e-06)
        message = 'Arrays are not equal'
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res2a)

    @pytest.mark.filterwarnings('ignore:Exited postprocessing')
    def test_svd_maxiter(self):
        if False:
            while True:
                i = 10
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        A = np.diag(np.arange(9)).astype(np.float64)
        k = 1
        (u, s, vh) = sorted_svd(A, k)
        if self.solver == 'arpack':
            message = 'ARPACK error -1: No convergence'
            with pytest.raises(ArpackNoConvergence, match=message):
                svds(A, k, ncv=3, maxiter=1, solver=self.solver)
        elif self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match='Exited at iteration'):
                svds(A, k, maxiter=1, solver=self.solver)
        elif self.solver == 'propack':
            message = 'k=1 singular triplets did not converge within'
            with pytest.raises(np.linalg.LinAlgError, match=message):
                svds(A, k, maxiter=1, solver=self.solver)
        (ud, sd, vhd) = svds(A, k, solver=self.solver)
        _check_svds(A, k, ud, sd, vhd, atol=1e-08)
        assert_allclose(np.abs(ud), np.abs(u), atol=1e-08)
        assert_allclose(np.abs(vhd), np.abs(vh), atol=1e-08)
        assert_allclose(np.abs(sd), np.abs(s), atol=1e-09)

    @pytest.mark.parametrize('rsv', (True, False, 'u', 'vh'))
    @pytest.mark.parametrize('shape', ((5, 7), (6, 6), (7, 5)))
    def test_svd_return_singular_vectors(self, rsv, shape):
        if False:
            return 10
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        rng = np.random.default_rng(0)
        A = rng.random(shape)
        k = 2
        (M, N) = shape
        (u, s, vh) = sorted_svd(A, k)
        respect_u = True if self.solver == 'propack' else M <= N
        respect_vh = True if self.solver == 'propack' else M > N
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match='The problem size'):
                if rsv is False:
                    s2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                    assert_allclose(s2, s)
                elif rsv == 'u' and respect_u:
                    (u2, s2, vh2) = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                    assert_allclose(np.abs(u2), np.abs(u))
                    assert_allclose(s2, s)
                    assert vh2 is None
                elif rsv == 'vh' and respect_vh:
                    (u2, s2, vh2) = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                    assert u2 is None
                    assert_allclose(s2, s)
                    assert_allclose(np.abs(vh2), np.abs(vh))
                else:
                    (u2, s2, vh2) = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
                    if u2 is not None:
                        assert_allclose(np.abs(u2), np.abs(u))
                    assert_allclose(s2, s)
                    if vh2 is not None:
                        assert_allclose(np.abs(vh2), np.abs(vh))
        elif rsv is False:
            s2 = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
            assert_allclose(s2, s)
        elif rsv == 'u' and respect_u:
            (u2, s2, vh2) = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
            assert_allclose(np.abs(u2), np.abs(u))
            assert_allclose(s2, s)
            assert vh2 is None
        elif rsv == 'vh' and respect_vh:
            (u2, s2, vh2) = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
            assert u2 is None
            assert_allclose(s2, s)
            assert_allclose(np.abs(vh2), np.abs(vh))
        else:
            (u2, s2, vh2) = svds(A, k, return_singular_vectors=rsv, solver=self.solver, random_state=rng)
            if u2 is not None:
                assert_allclose(np.abs(u2), np.abs(u))
            assert_allclose(s2, s)
            if vh2 is not None:
                assert_allclose(np.abs(vh2), np.abs(vh))
    A1 = [[1, 2, 3], [3, 4, 3], [1 + 1j, 0, 2], [0, 0, 1]]
    A2 = [[1, 2, 3, 8 + 5j], [3 - 2j, 4, 3, 5], [1, 0, 2, 3], [0, 0, 1, 0]]

    @pytest.mark.filterwarnings('ignore:k >= N - 1', reason='needed to demonstrate #16725')
    @pytest.mark.parametrize('A', (A1, A2))
    @pytest.mark.parametrize('k', range(1, 5))
    @pytest.mark.parametrize('real', (True, False))
    @pytest.mark.parametrize('transpose', (False, True))
    @pytest.mark.parametrize('lo_type', (np.asarray, csc_matrix, aslinearoperator))
    def test_svd_simple(self, A, k, real, transpose, lo_type):
        if False:
            while True:
                i = 10
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        A = np.asarray(A)
        A = np.real(A) if real else A
        A = A.T if transpose else A
        A2 = lo_type(A)
        if k > min(A.shape):
            pytest.skip('`k` cannot be greater than `min(A.shape)`')
        if self.solver != 'propack' and k >= min(A.shape):
            pytest.skip('Only PROPACK supports complete SVD')
        if self.solver == 'arpack' and (not real) and (k == min(A.shape) - 1):
            pytest.skip('#16725')
        if self.solver == 'propack' and (np.intp(0).itemsize < 8 and (not real)):
            pytest.skip('PROPACK complex-valued SVD methods not available for 32-bit builds')
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match='The problem size'):
                (u, s, vh) = svds(A2, k, solver=self.solver)
        else:
            (u, s, vh) = svds(A2, k, solver=self.solver)
        _check_svds(A, k, u, s, vh, atol=3e-10)

    def test_svd_linop(self):
        if False:
            print('Hello World!')
        solver = self.solver
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not available')
        nmks = [(6, 7, 3), (9, 5, 4), (10, 8, 5)]

        def reorder(args):
            if False:
                for i in range(10):
                    print('nop')
            (U, s, VH) = args
            j = np.argsort(s)
            return (U[:, j], s[j], VH[j, :])
        for (n, m, k) in nmks:
            A = np.random.RandomState(52).randn(n, m)
            L = CheckingLinearOperator(A)
            if solver == 'propack':
                v0 = np.ones(n)
            else:
                v0 = np.ones(min(A.shape))
            if solver == 'lobpcg':
                with pytest.warns(UserWarning, match='The problem size'):
                    (U1, s1, VH1) = reorder(svds(A, k, v0=v0, solver=solver))
                    (U2, s2, VH2) = reorder(svds(L, k, v0=v0, solver=solver))
            else:
                (U1, s1, VH1) = reorder(svds(A, k, v0=v0, solver=solver))
                (U2, s2, VH2) = reorder(svds(L, k, v0=v0, solver=solver))
            assert_allclose(np.abs(U1), np.abs(U2))
            assert_allclose(s1, s2)
            assert_allclose(np.abs(VH1), np.abs(VH2))
            assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)), np.dot(U2, np.dot(np.diag(s2), VH2)))
            A = np.random.RandomState(1909).randn(n, m)
            L = CheckingLinearOperator(A)
            kwargs = {'v0': v0} if solver not in {None, 'arpack'} else {}
            if self.solver == 'lobpcg':
                with pytest.warns(UserWarning, match='The problem size'):
                    (U1, s1, VH1) = reorder(svds(A, k, which='SM', solver=solver, **kwargs))
                    (U2, s2, VH2) = reorder(svds(L, k, which='SM', solver=solver, **kwargs))
            else:
                (U1, s1, VH1) = reorder(svds(A, k, which='SM', solver=solver, **kwargs))
                (U2, s2, VH2) = reorder(svds(L, k, which='SM', solver=solver, **kwargs))
            assert_allclose(np.abs(U1), np.abs(U2))
            assert_allclose(s1 + 1, s2 + 1)
            assert_allclose(np.abs(VH1), np.abs(VH2))
            assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)), np.dot(U2, np.dot(np.diag(s2), VH2)))
            if k < min(n, m) - 1:
                for (dt, eps) in [(complex, 1e-07), (np.complex64, 0.001)]:
                    if self.solver == 'propack' and np.intp(0).itemsize < 8:
                        pytest.skip('PROPACK complex-valued SVD methods not available for 32-bit builds')
                    rng = np.random.RandomState(1648)
                    A = (rng.randn(n, m) + 1j * rng.randn(n, m)).astype(dt)
                    L = CheckingLinearOperator(A)
                    if self.solver == 'lobpcg':
                        with pytest.warns(UserWarning, match='The problem size'):
                            (U1, s1, VH1) = reorder(svds(A, k, which='LM', solver=solver))
                            (U2, s2, VH2) = reorder(svds(L, k, which='LM', solver=solver))
                    else:
                        (U1, s1, VH1) = reorder(svds(A, k, which='LM', solver=solver))
                        (U2, s2, VH2) = reorder(svds(L, k, which='LM', solver=solver))
                    assert_allclose(np.abs(U1), np.abs(U2), rtol=eps)
                    assert_allclose(s1, s2, rtol=eps)
                    assert_allclose(np.abs(VH1), np.abs(VH2), rtol=eps)
                    assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)), np.dot(U2, np.dot(np.diag(s2), VH2)), rtol=eps)
    SHAPES = ((100, 100), (100, 101), (101, 100))

    @pytest.mark.filterwarnings('ignore:Exited at iteration')
    @pytest.mark.filterwarnings('ignore:Exited postprocessing')
    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('dtype', (float, complex, np.float32))
    def test_small_sigma_sparse(self, shape, dtype):
        if False:
            print('Hello World!')
        solver = self.solver
        if solver == 'propack':
            pytest.skip('PROPACK failures unrelated to PR')
        rng = np.random.default_rng(0)
        k = 5
        (m, n) = shape
        S = random(m, n, density=0.1, random_state=rng)
        if dtype == complex:
            S = +1j * random(m, n, density=0.1, random_state=rng)
        e = np.ones(m)
        e[0:5] *= 10.0 ** np.arange(-5, 0, 1)
        S = spdiags(e, 0, m, m) @ S
        S = S.astype(dtype)
        (u, s, vh) = svds(S, k, which='SM', solver=solver, maxiter=1000)
        c_svd = False
        _check_svds_n(S, k, u, s, vh, which='SM', check_svd=c_svd, atol=0.1)

    @pytest.mark.parametrize('shape', ((6, 5), (5, 5), (5, 6)))
    @pytest.mark.parametrize('dtype', (float, complex))
    def test_svd_LM_ones_matrix(self, shape, dtype):
        if False:
            return 10
        k = 3
        (n, m) = shape
        A = np.ones((n, m), dtype=dtype)
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match='The problem size'):
                (U, s, VH) = svds(A, k, solver=self.solver)
        else:
            (U, s, VH) = svds(A, k, solver=self.solver)
        _check_svds(A, k, U, s, VH, check_usvh_A=True, check_svd=False)
        assert_allclose(np.max(s), np.sqrt(n * m))
        s = np.array(sorted(s)[:-1]) + 1
        z = np.ones_like(s)
        assert_allclose(s, z)

    @pytest.mark.filterwarnings('ignore:k >= N - 1', reason='needed to demonstrate #16725')
    @pytest.mark.parametrize('shape', ((3, 4), (4, 4), (4, 3), (4, 2)))
    @pytest.mark.parametrize('dtype', (float, complex))
    def test_zero_matrix(self, shape, dtype):
        if False:
            for i in range(10):
                print('nop')
        k = 1
        (n, m) = shape
        A = np.zeros((n, m), dtype=dtype)
        if self.solver == 'arpack' and dtype is complex and (k == min(A.shape) - 1):
            pytest.skip('#16725')
        if self.solver == 'propack':
            pytest.skip('PROPACK failures unrelated to PR #16712')
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match='The problem size'):
                (U, s, VH) = svds(A, k, solver=self.solver)
        else:
            (U, s, VH) = svds(A, k, solver=self.solver)
        _check_svds(A, k, U, s, VH, check_usvh_A=True, check_svd=False)
        assert_array_equal(s, 0)

    @pytest.mark.parametrize('shape', ((20, 20), (20, 21), (21, 20)))
    @pytest.mark.parametrize('dtype', (float, complex, np.float32))
    def test_small_sigma(self, shape, dtype):
        if False:
            i = 10
            return i + 15
        if not has_propack:
            pytest.skip('PROPACK not enabled')
        if dtype == complex and self.solver == 'propack':
            pytest.skip('PROPACK unsupported for complex dtype')
        rng = np.random.default_rng(179847540)
        A = rng.random(shape).astype(dtype)
        (u, _, vh) = svd(A, full_matrices=False)
        if dtype == np.float32:
            e = 10.0
        else:
            e = 100.0
        t = e ** (-np.arange(len(vh))).astype(dtype)
        A = (u * t).dot(vh)
        k = 4
        (u, s, vh) = svds(A, k, solver=self.solver, maxiter=100)
        t = np.sum(s > 0)
        assert_equal(t, k)
        _check_svds_n(A, k, u, s, vh, atol=0.001, rtol=1.0, check_svd=False)

    @pytest.mark.filterwarnings('ignore:The problem size')
    @pytest.mark.parametrize('dtype', (float, complex, np.float32))
    def test_small_sigma2(self, dtype):
        if False:
            i = 10
            return i + 15
        if self.solver == 'propack':
            if not has_propack:
                pytest.skip('PROPACK not enabled')
            elif dtype == np.float32:
                pytest.skip('Test failures in CI, see gh-17004')
            elif dtype == complex:
                pytest.skip('PROPACK unsupported for complex dtype')
        rng = np.random.default_rng(179847540)
        dim = 4
        size = 10
        x = rng.random((size, size - dim))
        y = x[:, :dim] * rng.random(dim)
        mat = np.hstack((x, y))
        mat = mat.astype(dtype)
        nz = null_space(mat)
        assert_equal(nz.shape[1], dim)
        (u, s, vh) = svd(mat)
        assert_allclose(s[-dim:], 0, atol=1e-06, rtol=1.0)
        assert_allclose(mat @ vh[-dim:, :].T, 0, atol=1e-06, rtol=1.0)
        sp_mat = csc_matrix(mat)
        (su, ss, svh) = svds(sp_mat, k=dim, which='SM', solver=self.solver)
        assert_allclose(ss, 0, atol=1e-05, rtol=1.0)
        (n, m) = mat.shape
        if n < m:
            assert_allclose(sp_mat.transpose() @ su, 0, atol=1e-05, rtol=1.0)
        assert_allclose(sp_mat @ svh.T, 0, atol=1e-05, rtol=1.0)

class Test_SVDS_once:

    @pytest.mark.parametrize('solver', ['ekki', object])
    def test_svds_input_validation_solver(self, solver):
        if False:
            for i in range(10):
                print('nop')
        message = 'solver must be one of'
        with pytest.raises(ValueError, match=message):
            svds(np.ones((3, 4)), k=2, solver=solver)

class Test_SVDS_ARPACK(SVDSCommonTests):

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.solver = 'arpack'

    @pytest.mark.parametrize('ncv', list(range(-1, 8)) + [4.5, '5'])
    def test_svds_input_validation_ncv_1(self, ncv):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(0)
        A = rng.random((6, 7))
        k = 3
        if ncv in {4, 5}:
            (u, s, vh) = svds(A, k=k, ncv=ncv, solver=self.solver)
            _check_svds(A, k, u, s, vh)
        else:
            message = '`ncv` must be an integer satisfying'
            with pytest.raises(ValueError, match=message):
                svds(A, k=k, ncv=ncv, solver=self.solver)

    def test_svds_input_validation_ncv_2(self):
        if False:
            return 10
        message = 'int() argument must be a'
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), ncv=[], solver=self.solver)
        message = 'invalid literal for int()'
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), ncv='hi', solver=self.solver)

class Test_SVDS_LOBPCG(SVDSCommonTests):

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.solver = 'lobpcg'

    def test_svd_random_state_3(self):
        if False:
            i = 10
            return i + 15
        pytest.xfail('LOBPCG is having trouble with accuracy.')

class Test_SVDS_PROPACK(SVDSCommonTests):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.solver = 'propack'

    def test_svd_LM_ones_matrix(self):
        if False:
            return 10
        message = 'PROPACK does not return orthonormal singular vectors associated with zero singular values.'
        pytest.xfail(message)

    def test_svd_LM_zeros_matrix(self):
        if False:
            i = 10
            return i + 15
        message = 'PROPACK does not return orthonormal singular vectors associated with zero singular values.'
        pytest.xfail(message)