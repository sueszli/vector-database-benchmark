import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
__all__ = ['broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'newton_krylov', 'BroydenFirst', 'KrylovJacobian', 'InverseJacobian']

class NoConvergence(Exception):
    pass

def maxnorm(x):
    if False:
        print('Hello World!')
    return np.absolute(x).max()

def _as_inexact(x):
    if False:
        i = 10
        return i + 15
    'Return `x` as an array, of either floats or complex floats'
    x = asarray(x)
    if not np.issubdtype(x.dtype, np.inexact):
        return asarray(x, dtype=np.float64)
    return x

def _array_like(x, x0):
    if False:
        i = 10
        return i + 15
    'Return ndarray `x` as same array subclass and shape as `x0`'
    x = np.reshape(x, np.shape(x0))
    wrap = getattr(x0, '__array_wrap__', x.__array_wrap__)
    return wrap(x)

def _safe_norm(v):
    if False:
        while True:
            i = 10
    if not np.isfinite(v).all():
        return np.array(np.inf)
    return norm(v)
_doc_parts = dict(params_basic='\n    F : function(x) -> f\n        Function whose root to find; should take and return an array-like\n        object.\n    xin : array_like\n        Initial guess for the solution\n    '.strip(), params_extra="\n    iter : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    verbose : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    f_tol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    f_rtol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    x_tol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    x_rtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in the\n        direction given by the Jacobian approximation. Defaults to 'armijo'.\n    callback : function, optional\n        Optional callback function. It is called on every iteration as\n        ``callback(x, f)`` where `x` is the current solution and `f`\n        the corresponding residual.\n\n    Returns\n    -------\n    sol : ndarray\n        An array (of similar array type as `x0`) containing the final solution.\n\n    Raises\n    ------\n    NoConvergence\n        When a solution was not found.\n\n    ".strip())

def _set_doc(obj):
    if False:
        for i in range(10):
            print('nop')
    if obj.__doc__:
        obj.__doc__ = obj.__doc__ % _doc_parts

def nonlin_solve(F, x0, jacobian='krylov', iter=None, verbose=False, maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None, tol_norm=None, line_search='armijo', callback=None, full_output=False, raise_exception=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find a root of a function, in a way suitable for large-scale problems.\n\n    Parameters\n    ----------\n    %(params_basic)s\n    jacobian : Jacobian\n        A Jacobian approximation: `Jacobian` object or something that\n        `asjacobian` can transform to one. Alternatively, a string specifying\n        which of the builtin Jacobian approximations to use:\n\n            krylov, broyden1, broyden2, anderson\n            diagbroyden, linearmixing, excitingmixing\n\n    %(params_extra)s\n    full_output : bool\n        If true, returns a dictionary `info` containing convergence\n        information.\n    raise_exception : bool\n        If True, a `NoConvergence` exception is raise if no solution is found.\n\n    See Also\n    --------\n    asjacobian, Jacobian\n\n    Notes\n    -----\n    This algorithm implements the inexact Newton method, with\n    backtracking or full line searches. Several Jacobian\n    approximations are available, including Krylov and Quasi-Newton\n    methods.\n\n    References\n    ----------\n    .. [KIM] C. T. Kelley, "Iterative Methods for Linear and Nonlinear\n       Equations". Society for Industrial and Applied Mathematics. (1995)\n       https://archive.siam.org/books/kelley/fr16/\n\n    '
    tol_norm = maxnorm if tol_norm is None else tol_norm
    condition = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol, x_tol=x_tol, x_rtol=x_rtol, iter=iter, norm=tol_norm)
    x0 = _as_inexact(x0)

    def func(z):
        if False:
            print('Hello World!')
        return _as_inexact(F(_array_like(z, x0))).flatten()
    x = x0.flatten()
    dx = np.full_like(x, np.inf)
    Fx = func(x)
    Fx_norm = norm(Fx)
    jacobian = asjacobian(jacobian)
    jacobian.setup(x.copy(), Fx, func)
    if maxiter is None:
        if iter is not None:
            maxiter = iter + 1
        else:
            maxiter = 100 * (x.size + 1)
    if line_search is True:
        line_search = 'armijo'
    elif line_search is False:
        line_search = None
    if line_search not in (None, 'armijo', 'wolfe'):
        raise ValueError('Invalid line search')
    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 0.001
    for n in range(maxiter):
        status = condition.check(Fx, x, dx)
        if status:
            break
        tol = min(eta, eta * Fx_norm)
        dx = -jacobian.solve(Fx, tol=tol)
        if norm(dx) == 0:
            raise ValueError('Jacobian inversion yielded zero vector. This indicates a bug in the Jacobian approximation.')
        if line_search:
            (s, x, Fx, Fx_norm_new) = _nonlin_line_search(func, x, Fx, dx, line_search)
        else:
            s = 1.0
            x = x + dx
            Fx = func(x)
            Fx_norm_new = norm(Fx)
        jacobian.update(x.copy(), Fx)
        if callback:
            callback(x, Fx)
        eta_A = gamma * Fx_norm_new ** 2 / Fx_norm ** 2
        if gamma * eta ** 2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma * eta ** 2))
        Fx_norm = Fx_norm_new
        if verbose:
            sys.stdout.write('%d:  |F(x)| = %g; step %g\n' % (n, tol_norm(Fx), s))
            sys.stdout.flush()
    else:
        if raise_exception:
            raise NoConvergence(_array_like(x, x0))
        else:
            status = 2
    if full_output:
        info = {'nit': condition.iteration, 'fun': Fx, 'status': status, 'success': status == 1, 'message': {1: 'A solution was found at the specified tolerance.', 2: 'The maximum number of iterations allowed has been reached.'}[status]}
        return (_array_like(x, x0), info)
    else:
        return _array_like(x, x0)
_set_doc(nonlin_solve)

def _nonlin_line_search(func, x, Fx, dx, search_type='armijo', rdiff=1e-08, smin=0.01):
    if False:
        i = 10
        return i + 15
    tmp_s = [0]
    tmp_Fx = [Fx]
    tmp_phi = [norm(Fx) ** 2]
    s_norm = norm(x) / norm(dx)

    def phi(s, store=True):
        if False:
            i = 10
            return i + 15
        if s == tmp_s[0]:
            return tmp_phi[0]
        xt = x + s * dx
        v = func(xt)
        p = _safe_norm(v) ** 2
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_Fx[0] = v
        return p

    def derphi(s):
        if False:
            for i in range(10):
                print('nop')
        ds = (abs(s) + s_norm + 1) * rdiff
        return (phi(s + ds, store=False) - phi(s)) / ds
    if search_type == 'wolfe':
        (s, phi1, phi0) = scalar_search_wolfe1(phi, derphi, tmp_phi[0], xtol=0.01, amin=smin)
    elif search_type == 'armijo':
        (s, phi1) = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=smin)
    if s is None:
        s = 1.0
    x = x + s * dx
    if s == tmp_s[0]:
        Fx = tmp_Fx[0]
    else:
        Fx = func(x)
    Fx_norm = norm(Fx)
    return (s, x, Fx, Fx_norm)

class TerminationCondition:
    """
    Termination condition for an iteration. It is terminated if

    - |F| < f_rtol*|F_0|, AND
    - |F| < f_tol

    AND

    - |dx| < x_rtol*|x|, AND
    - |dx| < x_tol

    """

    def __init__(self, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None, iter=None, norm=maxnorm):
        if False:
            return 10
        if f_tol is None:
            f_tol = np.finfo(np.float64).eps ** (1.0 / 3)
        if f_rtol is None:
            f_rtol = np.inf
        if x_tol is None:
            x_tol = np.inf
        if x_rtol is None:
            x_rtol = np.inf
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.norm = norm
        self.iter = iter
        self.f0_norm = None
        self.iteration = 0

    def check(self, f, x, dx):
        if False:
            return 10
        self.iteration += 1
        f_norm = self.norm(f)
        x_norm = self.norm(x)
        dx_norm = self.norm(dx)
        if self.f0_norm is None:
            self.f0_norm = f_norm
        if f_norm == 0:
            return 1
        if self.iter is not None:
            return 2 * (self.iteration > self.iter)
        return int((f_norm <= self.f_tol and f_norm / self.f_rtol <= self.f0_norm) and (dx_norm <= self.x_tol and dx_norm / self.x_rtol <= x_norm))

class Jacobian:
    """
    Common interface for Jacobians or Jacobian approximations.

    The optional methods come useful when implementing trust region
    etc., algorithms that often require evaluating transposes of the
    Jacobian.

    Methods
    -------
    solve
        Returns J^-1 * v
    update
        Updates Jacobian to point `x` (where the function has residual `Fx`)

    matvec : optional
        Returns J * v
    rmatvec : optional
        Returns A^H * v
    rsolve : optional
        Returns A^-H * v
    matmat : optional
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    todense : optional
        Form the dense Jacobian matrix. Necessary for dense trust region
        algorithms, and useful for testing.

    Attributes
    ----------
    shape
        Matrix dimensions (M, N)
    dtype
        Data type of the matrix.
    func : callable, optional
        Function the Jacobian corresponds to

    """

    def __init__(self, **kw):
        if False:
            print('Hello World!')
        names = ['solve', 'update', 'matvec', 'rmatvec', 'rsolve', 'matmat', 'todense', 'shape', 'dtype']
        for (name, value) in kw.items():
            if name not in names:
                raise ValueError('Unknown keyword argument %s' % name)
            if value is not None:
                setattr(self, name, kw[name])
        if hasattr(self, 'todense'):
            self.__array__ = lambda : self.todense()

    def aspreconditioner(self):
        if False:
            for i in range(10):
                print('nop')
        return InverseJacobian(self)

    def solve(self, v, tol=0):
        if False:
            return 10
        raise NotImplementedError

    def update(self, x, F):
        if False:
            return 10
        pass

    def setup(self, x, F, func):
        if False:
            print('Hello World!')
        self.func = func
        self.shape = (F.size, x.size)
        self.dtype = F.dtype
        if self.__class__.setup is Jacobian.setup:
            self.update(x, F)

class InverseJacobian:

    def __init__(self, jacobian):
        if False:
            return 10
        self.jacobian = jacobian
        self.matvec = jacobian.solve
        self.update = jacobian.update
        if hasattr(jacobian, 'setup'):
            self.setup = jacobian.setup
        if hasattr(jacobian, 'rsolve'):
            self.rmatvec = jacobian.rsolve

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        return self.jacobian.shape

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self.jacobian.dtype

def asjacobian(J):
    if False:
        i = 10
        return i + 15
    '\n    Convert given object to one suitable for use as a Jacobian.\n    '
    spsolve = scipy.sparse.linalg.spsolve
    if isinstance(J, Jacobian):
        return J
    elif inspect.isclass(J) and issubclass(J, Jacobian):
        return J()
    elif isinstance(J, np.ndarray):
        if J.ndim > 2:
            raise ValueError('array must have rank <= 2')
        J = np.atleast_2d(np.asarray(J))
        if J.shape[0] != J.shape[1]:
            raise ValueError('array must be square')
        return Jacobian(matvec=lambda v: dot(J, v), rmatvec=lambda v: dot(J.conj().T, v), solve=lambda v, tol=0: solve(J, v), rsolve=lambda v, tol=0: solve(J.conj().T, v), dtype=J.dtype, shape=J.shape)
    elif scipy.sparse.issparse(J):
        if J.shape[0] != J.shape[1]:
            raise ValueError('matrix must be square')
        return Jacobian(matvec=lambda v: J @ v, rmatvec=lambda v: J.conj().T @ v, solve=lambda v, tol=0: spsolve(J, v), rsolve=lambda v, tol=0: spsolve(J.conj().T, v), dtype=J.dtype, shape=J.shape)
    elif hasattr(J, 'shape') and hasattr(J, 'dtype') and hasattr(J, 'solve'):
        return Jacobian(matvec=getattr(J, 'matvec'), rmatvec=getattr(J, 'rmatvec'), solve=J.solve, rsolve=getattr(J, 'rsolve'), update=getattr(J, 'update'), setup=getattr(J, 'setup'), dtype=J.dtype, shape=J.shape)
    elif callable(J):

        class Jac(Jacobian):

            def update(self, x, F):
                if False:
                    print('Hello World!')
                self.x = x

            def solve(self, v, tol=0):
                if False:
                    for i in range(10):
                        print('nop')
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m, v)
                else:
                    raise ValueError('Unknown matrix type')

            def matvec(self, v):
                if False:
                    for i in range(10):
                        print('nop')
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m, v)
                elif scipy.sparse.issparse(m):
                    return m @ v
                else:
                    raise ValueError('Unknown matrix type')

            def rsolve(self, v, tol=0):
                if False:
                    while True:
                        i = 10
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m.conj().T, v)
                else:
                    raise ValueError('Unknown matrix type')

            def rmatvec(self, v):
                if False:
                    print('Hello World!')
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return m.conj().T @ v
                else:
                    raise ValueError('Unknown matrix type')
        return Jac()
    elif isinstance(J, str):
        return dict(broyden1=BroydenFirst, broyden2=BroydenSecond, anderson=Anderson, diagbroyden=DiagBroyden, linearmixing=LinearMixing, excitingmixing=ExcitingMixing, krylov=KrylovJacobian)[J]()
    else:
        raise TypeError('Cannot convert object to a Jacobian')

class GenericBroyden(Jacobian):

    def setup(self, x0, f0, func):
        if False:
            for i in range(10):
                print('nop')
        Jacobian.setup(self, x0, f0, func)
        self.last_f = f0
        self.last_x = x0
        if hasattr(self, 'alpha') and self.alpha is None:
            normf0 = norm(f0)
            if normf0:
                self.alpha = 0.5 * max(norm(x0), 1) / normf0
            else:
                self.alpha = 1.0

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def update(self, x, f):
        if False:
            i = 10
            return i + 15
        df = f - self.last_f
        dx = x - self.last_x
        self._update(x, f, dx, df, norm(dx), norm(df))
        self.last_f = f
        self.last_x = x

class LowRankMatrix:
    """
    A matrix represented as

    .. math:: \\alpha I + \\sum_{n=0}^{n=M} c_n d_n^\\dagger

    However, if the rank of the matrix reaches the dimension of the vectors,
    full matrix representation will be used thereon.

    """

    def __init__(self, alpha, n, dtype):
        if False:
            while True:
                i = 10
        self.alpha = alpha
        self.cs = []
        self.ds = []
        self.n = n
        self.dtype = dtype
        self.collapsed = None

    @staticmethod
    def _matvec(v, alpha, cs, ds):
        if False:
            while True:
                i = 10
        (axpy, scal, dotc) = get_blas_funcs(['axpy', 'scal', 'dotc'], cs[:1] + [v])
        w = alpha * v
        for (c, d) in zip(cs, ds):
            a = dotc(d, v)
            w = axpy(c, w, w.size, a)
        return w

    @staticmethod
    def _solve(v, alpha, cs, ds):
        if False:
            i = 10
            return i + 15
        'Evaluate w = M^-1 v'
        if len(cs) == 0:
            return v / alpha
        (axpy, dotc) = get_blas_funcs(['axpy', 'dotc'], cs[:1] + [v])
        c0 = cs[0]
        A = alpha * np.identity(len(cs), dtype=c0.dtype)
        for (i, d) in enumerate(ds):
            for (j, c) in enumerate(cs):
                A[i, j] += dotc(d, c)
        q = np.zeros(len(cs), dtype=c0.dtype)
        for (j, d) in enumerate(ds):
            q[j] = dotc(d, v)
        q /= alpha
        q = solve(A, q)
        w = v / alpha
        for (c, qc) in zip(cs, q):
            w = axpy(c, w, w.size, -qc)
        return w

    def matvec(self, v):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate w = M v'
        if self.collapsed is not None:
            return np.dot(self.collapsed, v)
        return LowRankMatrix._matvec(v, self.alpha, self.cs, self.ds)

    def rmatvec(self, v):
        if False:
            i = 10
            return i + 15
        'Evaluate w = M^H v'
        if self.collapsed is not None:
            return np.dot(self.collapsed.T.conj(), v)
        return LowRankMatrix._matvec(v, np.conj(self.alpha), self.ds, self.cs)

    def solve(self, v, tol=0):
        if False:
            i = 10
            return i + 15
        'Evaluate w = M^-1 v'
        if self.collapsed is not None:
            return solve(self.collapsed, v)
        return LowRankMatrix._solve(v, self.alpha, self.cs, self.ds)

    def rsolve(self, v, tol=0):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate w = M^-H v'
        if self.collapsed is not None:
            return solve(self.collapsed.T.conj(), v)
        return LowRankMatrix._solve(v, np.conj(self.alpha), self.ds, self.cs)

    def append(self, c, d):
        if False:
            while True:
                i = 10
        if self.collapsed is not None:
            self.collapsed += c[:, None] * d[None, :].conj()
            return
        self.cs.append(c)
        self.ds.append(d)
        if len(self.cs) > c.size:
            self.collapse()

    def __array__(self):
        if False:
            i = 10
            return i + 15
        if self.collapsed is not None:
            return self.collapsed
        Gm = self.alpha * np.identity(self.n, dtype=self.dtype)
        for (c, d) in zip(self.cs, self.ds):
            Gm += c[:, None] * d[None, :].conj()
        return Gm

    def collapse(self):
        if False:
            for i in range(10):
                print('nop')
        'Collapse the low-rank matrix to a full-rank one.'
        self.collapsed = np.array(self)
        self.cs = None
        self.ds = None
        self.alpha = None

    def restart_reduce(self, rank):
        if False:
            while True:
                i = 10
        '\n        Reduce the rank of the matrix by dropping all vectors.\n        '
        if self.collapsed is not None:
            return
        assert rank > 0
        if len(self.cs) > rank:
            del self.cs[:]
            del self.ds[:]

    def simple_reduce(self, rank):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reduce the rank of the matrix by dropping oldest vectors.\n        '
        if self.collapsed is not None:
            return
        assert rank > 0
        while len(self.cs) > rank:
            del self.cs[0]
            del self.ds[0]

    def svd_reduce(self, max_rank, to_retain=None):
        if False:
            i = 10
            return i + 15
        '\n        Reduce the rank of the matrix by retaining some SVD components.\n\n        This corresponds to the "Broyden Rank Reduction Inverse"\n        algorithm described in [1]_.\n\n        Note that the SVD decomposition can be done by solving only a\n        problem whose size is the effective rank of this matrix, which\n        is viable even for large problems.\n\n        Parameters\n        ----------\n        max_rank : int\n            Maximum rank of this matrix after reduction.\n        to_retain : int, optional\n            Number of SVD components to retain when reduction is done\n            (ie. rank > max_rank). Default is ``max_rank - 2``.\n\n        References\n        ----------\n        .. [1] B.A. van der Rotten, PhD thesis,\n           "A limited memory Broyden method to solve high-dimensional\n           systems of nonlinear equations". Mathematisch Instituut,\n           Universiteit Leiden, The Netherlands (2003).\n\n           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf\n\n        '
        if self.collapsed is not None:
            return
        p = max_rank
        if to_retain is not None:
            q = to_retain
        else:
            q = p - 2
        if self.cs:
            p = min(p, len(self.cs[0]))
        q = max(0, min(q, p - 1))
        m = len(self.cs)
        if m < p:
            return
        C = np.array(self.cs).T
        D = np.array(self.ds).T
        (D, R) = qr(D, mode='economic')
        C = dot(C, R.T.conj())
        (U, S, WH) = svd(C, full_matrices=False)
        C = dot(C, inv(WH))
        D = dot(D, WH.T.conj())
        for k in range(q):
            self.cs[k] = C[:, k].copy()
            self.ds[k] = D[:, k].copy()
        del self.cs[q:]
        del self.ds[q:]
_doc_parts['broyden_params'] = '\n    alpha : float, optional\n        Initial guess for the Jacobian is ``(-1/alpha)``.\n    reduction_method : str or tuple, optional\n        Method used in ensuring that the rank of the Broyden matrix\n        stays low. Can either be a string giving the name of the method,\n        or a tuple of the form ``(method, param1, param2, ...)``\n        that gives the name of the method and values for additional parameters.\n\n        Methods available:\n\n            - ``restart``: drop all matrix columns. Has no extra parameters.\n            - ``simple``: drop oldest matrix column. Has no extra parameters.\n            - ``svd``: keep only the most significant SVD components.\n              Takes an extra parameter, ``to_retain``, which determines the\n              number of SVD components to retain when rank reduction is done.\n              Default is ``max_rank - 2``.\n\n    max_rank : int, optional\n        Maximum rank for the Broyden matrix.\n        Default is infinity (i.e., no rank reduction).\n    '.strip()

class BroydenFirst(GenericBroyden):
    """
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as \\"Broyden's good method\\".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden1'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) dx^\\dagger H / ( dx^\\dagger H df)

    which corresponds to Broyden's first Jacobian update

    .. math:: J_+ = J + (df - J dx) dx^\\dagger / dx^\\dagger dx


    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       \\"A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations\\". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden1(fun, [0, 0])
    >>> sol
    array([0.84116396, 0.15883641])

    """

    def __init__(self, alpha=None, reduction_method='restart', max_rank=None):
        if False:
            i = 10
            return i + 15
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.Gm = None
        if max_rank is None:
            max_rank = np.inf
        self.max_rank = max_rank
        if isinstance(reduction_method, str):
            reduce_params = ()
        else:
            reduce_params = reduction_method[1:]
            reduction_method = reduction_method[0]
        reduce_params = (max_rank - 1,) + reduce_params
        if reduction_method == 'svd':
            self._reduce = lambda : self.Gm.svd_reduce(*reduce_params)
        elif reduction_method == 'simple':
            self._reduce = lambda : self.Gm.simple_reduce(*reduce_params)
        elif reduction_method == 'restart':
            self._reduce = lambda : self.Gm.restart_reduce(*reduce_params)
        else:
            raise ValueError("Unknown rank reduction method '%s'" % reduction_method)

    def setup(self, x, F, func):
        if False:
            return 10
        GenericBroyden.setup(self, x, F, func)
        self.Gm = LowRankMatrix(-self.alpha, self.shape[0], self.dtype)

    def todense(self):
        if False:
            return 10
        return inv(self.Gm)

    def solve(self, f, tol=0):
        if False:
            return 10
        r = self.Gm.matvec(f)
        if not np.isfinite(r).all():
            self.setup(self.last_x, self.last_f, self.func)
            return self.Gm.matvec(f)
        return r

    def matvec(self, f):
        if False:
            print('Hello World!')
        return self.Gm.solve(f)

    def rsolve(self, f, tol=0):
        if False:
            return 10
        return self.Gm.rmatvec(f)

    def rmatvec(self, f):
        if False:
            while True:
                i = 10
        return self.Gm.rsolve(f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if False:
            return 10
        self._reduce()
        v = self.Gm.rmatvec(dx)
        c = dx - self.Gm.matvec(df)
        d = v / vdot(df, v)
        self.Gm.append(c, d)

class BroydenSecond(BroydenFirst):
    """
    Find a root of a function, using Broyden's second Jacobian approximation.

    This method is also known as "Broyden's bad method".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden2'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) df^\\dagger / ( df^\\dagger df)

    corresponding to Broyden's second method.

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       "A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden2(fun, [0, 0])
    >>> sol
    array([0.84116365, 0.15883529])

    """

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if False:
            while True:
                i = 10
        self._reduce()
        v = df
        c = dx - self.Gm.matvec(df)
        d = v / df_norm ** 2
        self.Gm.append(c, d)

class Anderson(GenericBroyden):
    """
    Find a root of a function, using (extended) Anderson mixing.

    The Jacobian is formed by for a 'best' solution in the space
    spanned by last `M` vectors. As a result, only a MxM matrix
    inversions and MxN multiplications are required. [Ey]_

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.
    w0 : float, optional
        Regularization parameter for numerical stability.
        Compared to unity, good values of the order of 0.01.
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='anderson'`` in particular.

    References
    ----------
    .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.anderson(fun, [0, 0])
    >>> sol
    array([0.84116588, 0.15883789])

    """

    def __init__(self, alpha=None, w0=0.01, M=5):
        if False:
            while True:
                i = 10
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None
        self.w0 = w0

    def solve(self, f, tol=0):
        if False:
            i = 10
            return i + 15
        dx = -self.alpha * f
        n = len(self.dx)
        if n == 0:
            return dx
        df_f = np.empty(n, dtype=f.dtype)
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)
        try:
            gamma = solve(self.a, df_f)
        except LinAlgError:
            del self.dx[:]
            del self.df[:]
            return dx
        for m in range(n):
            dx += gamma[m] * (self.dx[m] + self.alpha * self.df[m])
        return dx

    def matvec(self, f):
        if False:
            i = 10
            return i + 15
        dx = -f / self.alpha
        n = len(self.dx)
        if n == 0:
            return dx
        df_f = np.empty(n, dtype=f.dtype)
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)
        b = np.empty((n, n), dtype=f.dtype)
        for i in range(n):
            for j in range(n):
                b[i, j] = vdot(self.df[i], self.dx[j])
                if i == j and self.w0 != 0:
                    b[i, j] -= vdot(self.df[i], self.df[i]) * self.w0 ** 2 * self.alpha
        gamma = solve(b, df_f)
        for m in range(n):
            dx += gamma[m] * (self.df[m] + self.dx[m] / self.alpha)
        return dx

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if False:
            i = 10
            return i + 15
        if self.M == 0:
            return
        self.dx.append(dx)
        self.df.append(df)
        while len(self.dx) > self.M:
            self.dx.pop(0)
            self.df.pop(0)
        n = len(self.dx)
        a = np.zeros((n, n), dtype=f.dtype)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    wd = self.w0 ** 2
                else:
                    wd = 0
                a[i, j] = (1 + wd) * vdot(self.df[i], self.df[j])
        a += np.triu(a, 1).T.conj()
        self.a = a

class DiagBroyden(GenericBroyden):
    """
    Find a root of a function, using diagonal Broyden Jacobian approximation.

    The Jacobian approximation is derived from previous iterations, by
    retaining only the diagonal of Broyden matrices.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='diagbroyden'`` in particular.

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.diagbroyden(fun, [0, 0])
    >>> sol
    array([0.84116403, 0.15883384])

    """

    def __init__(self, alpha=None):
        if False:
            while True:
                i = 10
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def setup(self, x, F, func):
        if False:
            for i in range(10):
                print('nop')
        GenericBroyden.setup(self, x, F, func)
        self.d = np.full((self.shape[0],), 1 / self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        if False:
            print('Hello World!')
        return -f / self.d

    def matvec(self, f):
        if False:
            while True:
                i = 10
        return -f * self.d

    def rsolve(self, f, tol=0):
        if False:
            return 10
        return -f / self.d.conj()

    def rmatvec(self, f):
        if False:
            i = 10
            return i + 15
        return -f * self.d.conj()

    def todense(self):
        if False:
            print('Hello World!')
        return np.diag(-self.d)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if False:
            print('Hello World!')
        self.d -= (df + self.d * dx) * dx / dx_norm ** 2

class LinearMixing(GenericBroyden):
    """
    Find a root of a function, using a scalar Jacobian approximation.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        The Jacobian approximation is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='linearmixing'`` in particular.

    """

    def __init__(self, alpha=None):
        if False:
            return 10
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def solve(self, f, tol=0):
        if False:
            while True:
                i = 10
        return -f * self.alpha

    def matvec(self, f):
        if False:
            i = 10
            return i + 15
        return -f / self.alpha

    def rsolve(self, f, tol=0):
        if False:
            print('Hello World!')
        return -f * np.conj(self.alpha)

    def rmatvec(self, f):
        if False:
            i = 10
            return i + 15
        return -f / np.conj(self.alpha)

    def todense(self):
        if False:
            for i in range(10):
                print('nop')
        return np.diag(np.full(self.shape[0], -1 / self.alpha))

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if False:
            while True:
                i = 10
        pass

class ExcitingMixing(GenericBroyden):
    """
    Find a root of a function, using a tuned diagonal Jacobian approximation.

    The Jacobian matrix is diagonal and is tuned on each iteration.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='excitingmixing'`` in particular.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial Jacobian approximation is (-1/alpha).
    alphamax : float, optional
        The entries of the diagonal Jacobian are kept in the range
        ``[alpha, alphamax]``.
    %(params_extra)s
    """

    def __init__(self, alpha=None, alphamax=1.0):
        if False:
            for i in range(10):
                print('nop')
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = None

    def setup(self, x, F, func):
        if False:
            i = 10
            return i + 15
        GenericBroyden.setup(self, x, F, func)
        self.beta = np.full((self.shape[0],), self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        if False:
            print('Hello World!')
        return -f * self.beta

    def matvec(self, f):
        if False:
            return 10
        return -f / self.beta

    def rsolve(self, f, tol=0):
        if False:
            return 10
        return -f * self.beta.conj()

    def rmatvec(self, f):
        if False:
            return 10
        return -f / self.beta.conj()

    def todense(self):
        if False:
            while True:
                i = 10
        return np.diag(-1 / self.beta)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if False:
            for i in range(10):
                print('nop')
        incr = f * self.last_f > 0
        self.beta[incr] += self.alpha
        self.beta[~incr] = self.alpha
        np.clip(self.beta, 0, self.alphamax, out=self.beta)

class KrylovJacobian(Jacobian):
    """
    Find a root of a function, using Krylov approximation for inverse Jacobian.

    This method is suitable for solving large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    rdiff : float, optional
        Relative step size to use in numerical differentiation.
    method : str or callable, optional
        Krylov method to use to approximate the Jacobian.  Can be a string,
        or a function implementing the same interface as the iterative
        solvers in `scipy.sparse.linalg`. If a string, needs to be one of:
        ``'lgmres'``, ``'gmres'``, ``'bicgstab'``, ``'cgs'``, ``'minres'``,
        ``'tfqmr'``.

        The default is `scipy.sparse.linalg.lgmres`.
    inner_maxiter : int, optional
        Parameter to pass to the "inner" Krylov solver: maximum number of
        iterations. Iteration will stop after maxiter steps even if the
        specified tolerance has not been achieved.
    inner_M : LinearOperator or InverseJacobian
        Preconditioner for the inner Krylov iteration.
        Note that you can use also inverse Jacobians as (adaptive)
        preconditioners. For example,

        >>> from scipy.optimize import BroydenFirst, KrylovJacobian
        >>> from scipy.optimize import InverseJacobian
        >>> jac = BroydenFirst()
        >>> kjac = KrylovJacobian(inner_M=InverseJacobian(jac))

        If the preconditioner has a method named 'update', it will be called
        as ``update(x, f)`` after each nonlinear step, with ``x`` giving
        the current point, and ``f`` the current function value.
    outer_k : int, optional
        Size of the subspace kept across LGMRES nonlinear iterations.
        See `scipy.sparse.linalg.lgmres` for details.
    inner_kwargs : kwargs
        Keyword parameters for the "inner" Krylov solver
        (defined with `method`). Parameter names must start with
        the `inner_` prefix which will be stripped before passing on
        the inner method. See, e.g., `scipy.sparse.linalg.gmres` for details.
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='krylov'`` in particular.
    scipy.sparse.linalg.gmres
    scipy.sparse.linalg.lgmres

    Notes
    -----
    This function implements a Newton-Krylov solver. The basic idea is
    to compute the inverse of the Jacobian with an iterative Krylov
    method. These methods require only evaluating the Jacobian-vector
    products, which are conveniently approximated by a finite difference:

    .. math:: J v \\approx (f(x + \\omega*v/|v|) - f(x)) / \\omega

    Due to the use of iterative matrix inverses, these methods can
    deal with large nonlinear problems.

    SciPy's `scipy.sparse.linalg` module offers a selection of Krylov
    solvers to choose from. The default here is `lgmres`, which is a
    variant of restarted GMRES iteration that reuses some of the
    information obtained in the previous Newton steps to invert
    Jacobians in subsequent steps.

    For a review on Newton-Krylov methods, see for example [1]_,
    and for the LGMRES sparse inverse method, see [2]_.

    References
    ----------
    .. [1] C. T. Kelley, Solving Nonlinear Equations with Newton's Method,
           SIAM, pp.57-83, 2003.
           :doi:`10.1137/1.9780898718898.ch3`
    .. [2] D.A. Knoll and D.E. Keyes, J. Comp. Phys. 193, 357 (2004).
           :doi:`10.1016/j.jcp.2003.08.010`
    .. [3] A.H. Baker and E.R. Jessup and T. Manteuffel,
           SIAM J. Matrix Anal. Appl. 26, 962 (2005).
           :doi:`10.1137/S0895479803422014`

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0] + 0.5 * x[1] - 1.0,
    ...             0.5 * (x[1] - x[0]) ** 2]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.newton_krylov(fun, [0, 0])
    >>> sol
    array([0.66731771, 0.66536458])

    """

    def __init__(self, rdiff=None, method='lgmres', inner_maxiter=20, inner_M=None, outer_k=10, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.preconditioner = inner_M
        self.rdiff = rdiff
        self.method = dict(bicgstab=scipy.sparse.linalg.bicgstab, gmres=scipy.sparse.linalg.gmres, lgmres=scipy.sparse.linalg.lgmres, cgs=scipy.sparse.linalg.cgs, minres=scipy.sparse.linalg.minres, tfqmr=scipy.sparse.linalg.tfqmr).get(method, method)
        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)
        if self.method is scipy.sparse.linalg.gmres:
            self.method_kw['restart'] = inner_maxiter
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('atol', 0)
        elif self.method in (scipy.sparse.linalg.gcrotmk, scipy.sparse.linalg.bicgstab, scipy.sparse.linalg.cgs):
            self.method_kw.setdefault('atol', 0)
        elif self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('outer_v', [])
            self.method_kw.setdefault('prepend_outer_v', True)
            self.method_kw.setdefault('store_outer_Av', False)
            self.method_kw.setdefault('atol', 0)
        for (key, value) in kw.items():
            if not key.startswith('inner_'):
                raise ValueError('Unknown parameter %s' % key)
            self.method_kw[key[6:]] = value

    def _update_diff_step(self):
        if False:
            for i in range(10):
                print('nop')
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        if False:
            for i in range(10):
                print('nop')
        nv = norm(v)
        if nv == 0:
            return 0 * v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc * v) - self.f0) / sc
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs, tol=0):
        if False:
            while True:
                i = 10
        if 'tol' in self.method_kw:
            (sol, info) = self.method(self.op, rhs, **self.method_kw)
        else:
            (sol, info) = self.method(self.op, rhs, tol=tol, **self.method_kw)
        return sol

    def update(self, x, f):
        if False:
            i = 10
            return i + 15
        self.x0 = x
        self.f0 = f
        self._update_diff_step()
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'update'):
                self.preconditioner.update(x, f)

    def setup(self, x, f, func):
        if False:
            for i in range(10):
                print('nop')
        Jacobian.setup(self, x, f, func)
        self.x0 = x
        self.f0 = f
        self.op = scipy.sparse.linalg.aslinearoperator(self)
        if self.rdiff is None:
            self.rdiff = np.finfo(x.dtype).eps ** (1.0 / 2)
        self._update_diff_step()
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'setup'):
                self.preconditioner.setup(x, f, func)

def _nonlin_wrapper(name, jac):
    if False:
        print('Hello World!')
    '\n    Construct a solver wrapper with given name and Jacobian approx.\n\n    It inspects the keyword arguments of ``jac.__init__``, and allows to\n    use the same arguments in the wrapper function, in addition to the\n    keyword arguments of `nonlin_solve`\n\n    '
    signature = _getfullargspec(jac.__init__)
    (args, varargs, varkw, defaults, kwonlyargs, kwdefaults, _) = signature
    kwargs = list(zip(args[-len(defaults):], defaults))
    kw_str = ', '.join([f'{k}={v!r}' for (k, v) in kwargs])
    if kw_str:
        kw_str = ', ' + kw_str
    kwkw_str = ', '.join([f'{k}={k}' for (k, v) in kwargs])
    if kwkw_str:
        kwkw_str = kwkw_str + ', '
    if kwonlyargs:
        raise ValueError('Unexpected signature %s' % signature)
    wrapper = "\ndef %(name)s(F, xin, iter=None %(kw)s, verbose=False, maxiter=None,\n             f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,\n             tol_norm=None, line_search='armijo', callback=None, **kw):\n    jac = %(jac)s(%(kwkw)s **kw)\n    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,\n                        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search,\n                        callback)\n"
    wrapper = wrapper % dict(name=name, kw=kw_str, jac=jac.__name__, kwkw=kwkw_str)
    ns = {}
    ns.update(globals())
    exec(wrapper, ns)
    func = ns[name]
    func.__doc__ = jac.__doc__
    _set_doc(func)
    return func
broyden1 = _nonlin_wrapper('broyden1', BroydenFirst)
broyden2 = _nonlin_wrapper('broyden2', BroydenSecond)
anderson = _nonlin_wrapper('anderson', Anderson)
linearmixing = _nonlin_wrapper('linearmixing', LinearMixing)
diagbroyden = _nonlin_wrapper('diagbroyden', DiagBroyden)
excitingmixing = _nonlin_wrapper('excitingmixing', ExcitingMixing)
newton_krylov = _nonlin_wrapper('newton_krylov', KrylovJacobian)