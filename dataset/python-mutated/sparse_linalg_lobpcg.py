import numpy as np
import warnings
from .common import Benchmark, safe_import
with safe_import():
    from scipy.linalg import eigh, cholesky_banded, cho_solve_banded, eig_banded
    from scipy.sparse.linalg import lobpcg, eigsh, LinearOperator
    from scipy.sparse.linalg._special_sparse_arrays import Sakurai, MikotaPair
msg = 'the benchmark code did not converge as expected, the timing is therefore useless'

class Bench(Benchmark):
    params = [[], ['lobpcg', 'eigsh', 'lapack']]
    param_names = ['n', 'solver']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.time_mikota.__func__.params = list(self.params)
        self.time_mikota.__func__.params[0] = [128, 256, 512, 1024, 2048]
        self.time_mikota.__func__.setup = self.setup_mikota
        self.time_sakurai.__func__.params = list(self.params)
        self.time_sakurai.__func__.params[0] = [50]
        self.time_sakurai.__func__.setup = self.setup_sakurai
        self.time_sakurai_inverse.__func__.params = list(self.params)
        self.time_sakurai_inverse.__func__.params[0] = [500, 1000]
        self.time_sakurai_inverse.__func__.setup = self.setup_sakurai_inverse

    def setup_mikota(self, n, solver):
        if False:
            while True:
                i = 10
        self.shape = (n, n)
        mik = MikotaPair(n)
        mik_k = mik.k
        mik_m = mik.m
        self.Ac = mik_k
        self.Aa = mik_k.toarray()
        self.Bc = mik_m
        self.Ba = mik_m.toarray()
        self.Ab = mik_k.tobanded()
        self.eigenvalues = mik.eigenvalues
        if solver == 'lapack' and n > 512:
            raise NotImplementedError()

    def setup_sakurai(self, n, solver):
        if False:
            for i in range(10):
                print('nop')
        self.shape = (n, n)
        sakurai_obj = Sakurai(n, dtype='int')
        self.A = sakurai_obj
        self.Aa = sakurai_obj.toarray()
        self.eigenvalues = sakurai_obj.eigenvalues

    def setup_sakurai_inverse(self, n, solver):
        if False:
            i = 10
            return i + 15
        self.shape = (n, n)
        sakurai_obj = Sakurai(n)
        self.A = sakurai_obj.tobanded().astype(np.float64)
        self.eigenvalues = sakurai_obj.eigenvalues

    def time_mikota(self, n, solver):
        if False:
            return 10

        def a(x):
            if False:
                i = 10
                return i + 15
            return cho_solve_banded((c, False), x)
        m = 10
        ee = self.eigenvalues(m)
        tol = m * n * n * n * np.finfo(float).eps
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, m))
        if solver == 'lobpcg':
            c = cholesky_banded(self.Ab.astype(np.float32))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                (el, _) = lobpcg(self.Ac, X, self.Bc, M=a, tol=0.0001, maxiter=40, largest=False)
            accuracy = max(abs(ee - el) / ee)
            assert accuracy < tol, msg
        elif solver == 'eigsh':
            B = LinearOperator((n, n), matvec=self.Bc, matmat=self.Bc, dtype='float64')
            A = LinearOperator((n, n), matvec=self.Ac, matmat=self.Ac, dtype='float64')
            c = cholesky_banded(self.Ab)
            a_l = LinearOperator((n, n), matvec=a, matmat=a, dtype='float64')
            (ea, _) = eigsh(B, k=m, M=A, Minv=a_l, which='LA', tol=0.0001, maxiter=50, v0=rng.normal(size=(n, 1)))
            accuracy = max(abs(ee - np.sort(1.0 / ea)) / ee)
            assert accuracy < tol, msg
        else:
            (ed, _) = eigh(self.Aa, self.Ba, subset_by_index=(0, m - 1))
            accuracy = max(abs(ee - ed) / ee)
            assert accuracy < tol, msg

    def time_sakurai(self, n, solver):
        if False:
            i = 10
            return i + 15
        m = 3
        ee = self.eigenvalues(m)
        tol = 100 * n * n * n * np.finfo(float).eps
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, m))
        if solver == 'lobpcg':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                (el, _) = lobpcg(self.A, X, tol=1e-09, maxiter=5000, largest=False)
            accuracy = max(abs(ee - el) / ee)
            assert accuracy < tol, msg
        elif solver == 'eigsh':
            a_l = LinearOperator((n, n), matvec=self.A, matmat=self.A, dtype='float64')
            (ea, _) = eigsh(a_l, k=m, which='SA', tol=1e-09, maxiter=15000, v0=rng.normal(size=(n, 1)))
            accuracy = max(abs(ee - ea) / ee)
            assert accuracy < tol, msg
        else:
            (ed, _) = eigh(self.Aa, subset_by_index=(0, m - 1))
            accuracy = max(abs(ee - ed) / ee)
            assert accuracy < tol, msg

    def time_sakurai_inverse(self, n, solver):
        if False:
            print('Hello World!')

        def a(x):
            if False:
                return 10
            return cho_solve_banded((c, False), x)
        m = 3
        ee = self.eigenvalues(m)
        tol = 10 * n * n * n * np.finfo(float).eps
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, m))
        if solver == 'lobpcg':
            c = cholesky_banded(self.A)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                (el, _) = lobpcg(a, X, tol=1e-09, maxiter=8)
            accuracy = max(abs(ee - 1.0 / el) / ee)
            assert accuracy < tol, msg
        elif solver == 'eigsh':
            c = cholesky_banded(self.A)
            a_l = LinearOperator((n, n), matvec=a, matmat=a, dtype='float64')
            (ea, _) = eigsh(a_l, k=m, which='LA', tol=1e-09, maxiter=8, v0=rng.normal(size=(n, 1)))
            accuracy = max(abs(ee - np.sort(1.0 / ea)) / ee)
            assert accuracy < tol, msg
        else:
            (ed, _) = eig_banded(self.A, select='i', select_range=[0, m - 1])
            accuracy = max(abs(ee - ed) / ee)
            assert accuracy < tol, msg