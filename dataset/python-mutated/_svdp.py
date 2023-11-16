"""
Python wrapper for PROPACK
--------------------------

PROPACK is a collection of Fortran routines for iterative computation
of partial SVDs of large matrices or linear operators.

Based on BSD licensed pypropack project:
  http://github.com/jakevdp/pypropack
  Author: Jake Vanderplas <vanderplas@astro.washington.edu>

PROPACK source is BSD licensed, and available at
  http://soi.stanford.edu/~rmunk/PROPACK/
"""
__all__ = ['_svdp']
import numpy as np
from scipy._lib._util import check_random_state
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import LinAlgError
from ._propack import _spropack
from ._propack import _dpropack
from ._propack import _cpropack
from ._propack import _zpropack
_lansvd_dict = {'f': _spropack.slansvd, 'd': _dpropack.dlansvd, 'F': _cpropack.clansvd, 'D': _zpropack.zlansvd}
_lansvd_irl_dict = {'f': _spropack.slansvd_irl, 'd': _dpropack.dlansvd_irl, 'F': _cpropack.clansvd_irl, 'D': _zpropack.zlansvd_irl}
_which_converter = {'LM': 'L', 'SM': 'S'}

class _AProd:
    """
    Wrapper class for linear operator

    The call signature of the __call__ method matches the callback of
    the PROPACK routines.
    """

    def __init__(self, A):
        if False:
            while True:
                i = 10
        try:
            self.A = aslinearoperator(A)
        except TypeError:
            self.A = aslinearoperator(np.asarray(A))

    def __call__(self, transa, m, n, x, y, sparm, iparm):
        if False:
            return 10
        if transa == 'n':
            y[:] = self.A.matvec(x)
        else:
            y[:] = self.A.rmatvec(x)

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        return self.A.shape

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.A.dtype
        except AttributeError:
            return self.A.matvec(np.zeros(self.A.shape[1])).dtype

def _svdp(A, k, which='LM', irl_mode=True, kmax=None, compute_u=True, compute_v=True, v0=None, full_output=False, tol=0, delta=None, eta=None, anorm=0, cgs=False, elr=True, min_relgap=0.002, shifts=None, maxiter=None, random_state=None):
    if False:
        return 10
    '\n    Compute the singular value decomposition of a linear operator using PROPACK\n\n    Parameters\n    ----------\n    A : array_like, sparse matrix, or LinearOperator\n        Operator for which SVD will be computed.  If `A` is a LinearOperator\n        object, it must define both ``matvec`` and ``rmatvec`` methods.\n    k : int\n        Number of singular values/vectors to compute\n    which : {"LM", "SM"}\n        Which singular triplets to compute:\n        - \'LM\': compute triplets corresponding to the `k` largest singular\n                values\n        - \'SM\': compute triplets corresponding to the `k` smallest singular\n                values\n        `which=\'SM\'` requires `irl_mode=True`.  Computes largest singular\n        values by default.\n    irl_mode : bool, optional\n        If `True`, then compute SVD using IRL (implicitly restarted Lanczos)\n        mode.  Default is `True`.\n    kmax : int, optional\n        Maximal number of iterations / maximal dimension of the Krylov\n        subspace. Default is ``10 * k``.\n    compute_u : bool, optional\n        If `True` (default) then compute left singular vectors, `u`.\n    compute_v : bool, optional\n        If `True` (default) then compute right singular vectors, `v`.\n    tol : float, optional\n        The desired relative accuracy for computed singular values.\n        If not specified, it will be set based on machine precision.\n    v0 : array_like, optional\n        Starting vector for iterations: must be of length ``A.shape[0]``.\n        If not specified, PROPACK will generate a starting vector.\n    full_output : bool, optional\n        If `True`, then return sigma_bound.  Default is `False`.\n    delta : float, optional\n        Level of orthogonality to maintain between Lanczos vectors.\n        Default is set based on machine precision.\n    eta : float, optional\n        Orthogonality cutoff.  During reorthogonalization, vectors with\n        component larger than `eta` along the Lanczos vector will be purged.\n        Default is set based on machine precision.\n    anorm : float, optional\n        Estimate of ``||A||``.  Default is `0`.\n    cgs : bool, optional\n        If `True`, reorthogonalization is done using classical Gram-Schmidt.\n        If `False` (default), it is done using modified Gram-Schmidt.\n    elr : bool, optional\n        If `True` (default), then extended local orthogonality is enforced\n        when obtaining singular vectors.\n    min_relgap : float, optional\n        The smallest relative gap allowed between any shift in IRL mode.\n        Default is `0.001`.  Accessed only if ``irl_mode=True``.\n    shifts : int, optional\n        Number of shifts per restart in IRL mode.  Default is determined\n        to satisfy ``k <= min(kmax-shifts, m, n)``.  Must be\n        >= 0, but choosing 0 might lead to performance degradation.\n        Accessed only if ``irl_mode=True``.\n    maxiter : int, optional\n        Maximum number of restarts in IRL mode.  Default is `1000`.\n        Accessed only if ``irl_mode=True``.\n    random_state : {None, int, `numpy.random.Generator`,\n                    `numpy.random.RandomState`}, optional\n\n        Pseudorandom number generator state used to generate resamples.\n\n        If `random_state` is ``None`` (or `np.random`), the\n        `numpy.random.RandomState` singleton is used.\n        If `random_state` is an int, a new ``RandomState`` instance is used,\n        seeded with `random_state`.\n        If `random_state` is already a ``Generator`` or ``RandomState``\n        instance then that instance is used.\n\n    Returns\n    -------\n    u : ndarray\n        The `k` largest (``which="LM"``) or smallest (``which="SM"``) left\n        singular vectors, ``shape == (A.shape[0], 3)``, returned only if\n        ``compute_u=True``.\n    sigma : ndarray\n        The top `k` singular values, ``shape == (k,)``\n    vt : ndarray\n        The `k` largest (``which="LM"``) or smallest (``which="SM"``) right\n        singular vectors, ``shape == (3, A.shape[1])``, returned only if\n        ``compute_v=True``.\n    sigma_bound : ndarray\n        the error bounds on the singular values sigma, returned only if\n        ``full_output=True``.\n\n    '
    if np.iscomplexobj(A) and np.intp(0).itemsize < 8:
        raise TypeError('PROPACK complex-valued SVD methods not available for 32-bit builds')
    random_state = check_random_state(random_state)
    which = which.upper()
    if which not in {'LM', 'SM'}:
        raise ValueError("`which` must be either 'LM' or 'SM'")
    if not irl_mode and which == 'SM':
        raise ValueError("`which`='SM' requires irl_mode=True")
    aprod = _AProd(A)
    typ = aprod.dtype.char
    try:
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]
    except KeyError:
        if np.iscomplexobj(np.empty(0, dtype=typ)):
            typ = np.dtype(complex).char
        else:
            typ = np.dtype(float).char
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]
    (m, n) = aprod.shape
    if k < 1 or k > min(m, n):
        raise ValueError('k must be positive and not greater than m or n')
    if kmax is None:
        kmax = 10 * k
    if maxiter is None:
        maxiter = 1000
    kmax = min(m + 1, n + 1, kmax)
    if kmax < k:
        raise ValueError(f'kmax must be greater than or equal to k, but kmax ({kmax}) < k ({k})')
    jobu = 'y' if compute_u else 'n'
    jobv = 'y' if compute_v else 'n'
    u = np.zeros((m, kmax + 1), order='F', dtype=typ)
    v = np.zeros((n, kmax), order='F', dtype=typ)
    if v0 is None:
        u[:, 0] = random_state.uniform(size=m)
        if np.iscomplexobj(np.empty(0, dtype=typ)):
            u[:, 0] += 1j * random_state.uniform(size=m)
    else:
        try:
            u[:, 0] = v0
        except ValueError:
            raise ValueError(f'v0 must be of length {m}')
    if delta is None:
        delta = np.sqrt(np.finfo(typ).eps)
    if eta is None:
        eta = np.finfo(typ).eps ** 0.75
    if irl_mode:
        doption = np.array((delta, eta, anorm, min_relgap), dtype=typ.lower())
        if shifts is None:
            shifts = kmax - k
        if k > min(kmax - shifts, m, n):
            raise ValueError('shifts must satisfy k <= min(kmax-shifts, m, n)!')
        elif shifts < 0:
            raise ValueError('shifts must be >= 0!')
    else:
        doption = np.array((delta, eta, anorm), dtype=typ.lower())
    ioption = np.array((int(bool(cgs)), int(bool(elr))), dtype='i')
    blocksize = 16
    if compute_u or compute_v:
        lwork = m + n + 9 * kmax + 5 * kmax * kmax + 4 + max(3 * kmax * kmax + 4 * kmax + 4, blocksize * max(m, n))
        liwork = 8 * kmax
    else:
        lwork = m + n + 9 * kmax + 2 * kmax * kmax + 4 + max(m + n, 4 * kmax + 4)
        liwork = 2 * kmax + 1
    work = np.empty(lwork, dtype=typ.lower())
    iwork = np.empty(liwork, dtype=np.int32)
    dparm = np.empty(1, dtype=typ.lower())
    iparm = np.empty(1, dtype=np.int32)
    if typ.isupper():
        zwork = np.empty(m + n + 32 * m, dtype=typ)
        works = (work, zwork, iwork)
    else:
        works = (work, iwork)
    if irl_mode:
        (u, sigma, bnd, v, info) = lansvd_irl(_which_converter[which], jobu, jobv, m, n, shifts, k, maxiter, aprod, u, v, tol, *works, doption, ioption, dparm, iparm)
    else:
        (u, sigma, bnd, v, info) = lansvd(jobu, jobv, m, n, k, aprod, u, v, tol, *works, doption, ioption, dparm, iparm)
    if info > 0:
        raise LinAlgError(f'An invariant subspace of dimension {info} was found.')
    elif info < 0:
        raise LinAlgError(f'k={k} singular triplets did not converge within kmax={kmax} iterations')
    return (u[:, :k], sigma, v[:, :k].conj().T, bnd)