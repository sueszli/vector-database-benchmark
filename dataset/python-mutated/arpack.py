"""
Find a few eigenvectors and eigenvalues of a matrix.


Uses ARPACK: https://github.com/opencollab/arpack-ng

"""
import numpy as np
import warnings
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse._sputils import isdense, is_pydata_spmatrix
from scipy.sparse.linalg import gmres, splu
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock
from . import _arpack
arpack_int = _arpack.timing.nbx.dtype
__docformat__ = 'restructuredtext en'
__all__ = ['eigs', 'eigsh', 'ArpackError', 'ArpackNoConvergence']
_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}
_ndigits = {'f': 5, 'd': 12, 'F': 5, 'D': 12}
DNAUPD_ERRORS = {0: 'Normal exit.', 1: 'Maximum number of iterations taken. All possible eigenvalues of OP has been found. IPARAM(5) returns the number of wanted converged Ritz values.', 2: 'No longer an informational error. Deprecated starting with release 2 of ARPACK.', 3: 'No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV. ', -1: 'N must be positive.', -2: 'NEV must be positive.', -3: 'NCV-NEV >= 2 and less than or equal to N.', -4: 'The maximum number of Arnoldi update iterations allowed must be greater than zero.', -5: " WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'", -6: "BMAT must be one of 'I' or 'G'.", -7: 'Length of private work array WORKL is not sufficient.', -8: 'Error return from LAPACK eigenvalue calculation;', -9: 'Starting vector is zero.', -10: 'IPARAM(7) must be 1,2,3,4.', -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.", -12: 'IPARAM(1) must be equal to 0 or 1.', -13: "NEV and WHICH = 'BE' are incompatible.", -9999: 'Could not build an Arnoldi factorization. IPARAM(5) returns the size of the current Arnoldi factorization. The user is advised to check that enough workspace and array storage has been allocated.'}
SNAUPD_ERRORS = DNAUPD_ERRORS
ZNAUPD_ERRORS = DNAUPD_ERRORS.copy()
ZNAUPD_ERRORS[-10] = 'IPARAM(7) must be 1,2,3.'
CNAUPD_ERRORS = ZNAUPD_ERRORS
DSAUPD_ERRORS = {0: 'Normal exit.', 1: 'Maximum number of iterations taken. All possible eigenvalues of OP has been found.', 2: 'No longer an informational error. Deprecated starting with release 2 of ARPACK.', 3: 'No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV. ', -1: 'N must be positive.', -2: 'NEV must be positive.', -3: 'NCV must be greater than NEV and less than or equal to N.', -4: 'The maximum number of Arnoldi update iterations allowed must be greater than zero.', -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.", -6: "BMAT must be one of 'I' or 'G'.", -7: 'Length of private work array WORKL is not sufficient.', -8: 'Error return from trid. eigenvalue calculation; Informational error from LAPACK routine dsteqr .', -9: 'Starting vector is zero.', -10: 'IPARAM(7) must be 1,2,3,4,5.', -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.", -12: 'IPARAM(1) must be equal to 0 or 1.', -13: "NEV and WHICH = 'BE' are incompatible. ", -9999: 'Could not build an Arnoldi factorization. IPARAM(5) returns the size of the current Arnoldi factorization. The user is advised to check that enough workspace and array storage has been allocated.'}
SSAUPD_ERRORS = DSAUPD_ERRORS
DNEUPD_ERRORS = {0: 'Normal exit.', 1: 'The Schur form computed by LAPACK routine dlahqr could not be reordered by LAPACK routine dtrsen. Re-enter subroutine dneupd  with IPARAM(5)NCV and increase the size of the arrays DR and DI to have dimension at least dimension NCV and allocate at least NCV columns for Z. NOTE: Not necessary if Z and V share the same space. Please notify the authors if this erroroccurs.', -1: 'N must be positive.', -2: 'NEV must be positive.', -3: 'NCV-NEV >= 2 and less than or equal to N.', -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'", -6: "BMAT must be one of 'I' or 'G'.", -7: 'Length of private work WORKL array is not sufficient.', -8: 'Error return from calculation of a real Schur form. Informational error from LAPACK routine dlahqr .', -9: 'Error return from calculation of eigenvectors. Informational error from LAPACK routine dtrevc.', -10: 'IPARAM(7) must be 1,2,3,4.', -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.", -12: "HOWMNY = 'S' not yet implemented", -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.", -14: 'DNAUPD  did not find any eigenvalues to sufficient accuracy.', -15: 'DNEUPD got a different count of the number of converged Ritz values than DNAUPD got.  This indicates the user probably made an error in passing data from DNAUPD to DNEUPD or that the data was modified before entering DNEUPD'}
SNEUPD_ERRORS = DNEUPD_ERRORS.copy()
SNEUPD_ERRORS[1] = 'The Schur form computed by LAPACK routine slahqr could not be reordered by LAPACK routine strsen . Re-enter subroutine dneupd  with IPARAM(5)=NCV and increase the size of the arrays DR and DI to have dimension at least dimension NCV and allocate at least NCV columns for Z. NOTE: Not necessary if Z and V share the same space. Please notify the authors if this error occurs.'
SNEUPD_ERRORS[-14] = 'SNAUPD did not find any eigenvalues to sufficient accuracy.'
SNEUPD_ERRORS[-15] = 'SNEUPD got a different count of the number of converged Ritz values than SNAUPD got.  This indicates the user probably made an error in passing data from SNAUPD to SNEUPD or that the data was modified before entering SNEUPD'
ZNEUPD_ERRORS = {0: 'Normal exit.', 1: 'The Schur form computed by LAPACK routine csheqr could not be reordered by LAPACK routine ztrsen. Re-enter subroutine zneupd with IPARAM(5)=NCV and increase the size of the array D to have dimension at least dimension NCV and allocate at least NCV columns for Z. NOTE: Not necessary if Z and V share the same space. Please notify the authors if this error occurs.', -1: 'N must be positive.', -2: 'NEV must be positive.', -3: 'NCV-NEV >= 1 and less than or equal to N.', -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'", -6: "BMAT must be one of 'I' or 'G'.", -7: 'Length of private work WORKL array is not sufficient.', -8: 'Error return from LAPACK eigenvalue calculation. This should never happened.', -9: 'Error return from calculation of eigenvectors. Informational error from LAPACK routine ztrevc.', -10: 'IPARAM(7) must be 1,2,3', -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.", -12: "HOWMNY = 'S' not yet implemented", -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.", -14: 'ZNAUPD did not find any eigenvalues to sufficient accuracy.', -15: 'ZNEUPD got a different count of the number of converged Ritz values than ZNAUPD got.  This indicates the user probably made an error in passing data from ZNAUPD to ZNEUPD or that the data was modified before entering ZNEUPD'}
CNEUPD_ERRORS = ZNEUPD_ERRORS.copy()
CNEUPD_ERRORS[-14] = 'CNAUPD did not find any eigenvalues to sufficient accuracy.'
CNEUPD_ERRORS[-15] = 'CNEUPD got a different count of the number of converged Ritz values than CNAUPD got.  This indicates the user probably made an error in passing data from CNAUPD to CNEUPD or that the data was modified before entering CNEUPD'
DSEUPD_ERRORS = {0: 'Normal exit.', -1: 'N must be positive.', -2: 'NEV must be positive.', -3: 'NCV must be greater than NEV and less than or equal to N.', -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.", -6: "BMAT must be one of 'I' or 'G'.", -7: 'Length of private work WORKL array is not sufficient.', -8: 'Error return from trid. eigenvalue calculation; Information error from LAPACK routine dsteqr.', -9: 'Starting vector is zero.', -10: 'IPARAM(7) must be 1,2,3,4,5.', -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.", -12: "NEV and WHICH = 'BE' are incompatible.", -14: 'DSAUPD  did not find any eigenvalues to sufficient accuracy.', -15: "HOWMNY must be one of 'A' or 'S' if RVEC = .true.", -16: "HOWMNY = 'S' not yet implemented", -17: 'DSEUPD  got a different count of the number of converged Ritz values than DSAUPD  got.  This indicates the user probably made an error in passing data from DSAUPD  to DSEUPD  or that the data was modified before entering  DSEUPD.'}
SSEUPD_ERRORS = DSEUPD_ERRORS.copy()
SSEUPD_ERRORS[-14] = 'SSAUPD  did not find any eigenvalues to sufficient accuracy.'
SSEUPD_ERRORS[-17] = 'SSEUPD  got a different count of the number of converged Ritz values than SSAUPD  got.  This indicates the user probably made an error in passing data from SSAUPD  to SSEUPD  or that the data was modified before entering  SSEUPD.'
_SAUPD_ERRORS = {'d': DSAUPD_ERRORS, 's': SSAUPD_ERRORS}
_NAUPD_ERRORS = {'d': DNAUPD_ERRORS, 's': SNAUPD_ERRORS, 'z': ZNAUPD_ERRORS, 'c': CNAUPD_ERRORS}
_SEUPD_ERRORS = {'d': DSEUPD_ERRORS, 's': SSEUPD_ERRORS}
_NEUPD_ERRORS = {'d': DNEUPD_ERRORS, 's': SNEUPD_ERRORS, 'z': ZNEUPD_ERRORS, 'c': CNEUPD_ERRORS}
_SEUPD_WHICH = ['LM', 'SM', 'LA', 'SA', 'BE']
_NEUPD_WHICH = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']

class ArpackError(RuntimeError):
    """
    ARPACK error
    """

    def __init__(self, info, infodict=_NAUPD_ERRORS):
        if False:
            for i in range(10):
                print('nop')
        msg = infodict.get(info, 'Unknown error')
        RuntimeError.__init__(self, 'ARPACK error %d: %s' % (info, msg))

class ArpackNoConvergence(ArpackError):
    """
    ARPACK iteration did not converge

    Attributes
    ----------
    eigenvalues : ndarray
        Partial result. Converged eigenvalues.
    eigenvectors : ndarray
        Partial result. Converged eigenvectors.

    """

    def __init__(self, msg, eigenvalues, eigenvectors):
        if False:
            while True:
                i = 10
        ArpackError.__init__(self, -1, {-1: msg})
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

def choose_ncv(k):
    if False:
        while True:
            i = 10
    '\n    Choose number of lanczos vectors based on target number\n    of singular/eigen values and vectors to compute, k.\n    '
    return max(2 * k + 1, 20)

class _ArpackParams:

    def __init__(self, n, k, tp, mode=1, sigma=None, ncv=None, v0=None, maxiter=None, which='LM', tol=0):
        if False:
            for i in range(10):
                print('nop')
        if k <= 0:
            raise ValueError('k must be positive, k=%d' % k)
        if maxiter is None:
            maxiter = n * 10
        if maxiter <= 0:
            raise ValueError('maxiter must be positive, maxiter=%d' % maxiter)
        if tp not in 'fdFD':
            raise ValueError("matrix type must be 'f', 'd', 'F', or 'D'")
        if v0 is not None:
            self.resid = np.array(v0, copy=True)
            info = 1
        else:
            self.resid = np.zeros(n, tp)
            info = 0
        if sigma is None:
            self.sigma = 0
        else:
            self.sigma = sigma
        if ncv is None:
            ncv = choose_ncv(k)
        ncv = min(ncv, n)
        self.v = np.zeros((n, ncv), tp)
        self.iparam = np.zeros(11, arpack_int)
        ishfts = 1
        self.mode = mode
        self.iparam[0] = ishfts
        self.iparam[2] = maxiter
        self.iparam[3] = 1
        self.iparam[6] = mode
        self.n = n
        self.tol = tol
        self.k = k
        self.maxiter = maxiter
        self.ncv = ncv
        self.which = which
        self.tp = tp
        self.info = info
        self.converged = False
        self.ido = 0

    def _raise_no_convergence(self):
        if False:
            while True:
                i = 10
        msg = 'No convergence (%d iterations, %d/%d eigenvectors converged)'
        k_ok = self.iparam[4]
        num_iter = self.iparam[2]
        try:
            (ev, vec) = self.extract(True)
        except ArpackError as err:
            msg = f'{msg} [{err}]'
            ev = np.zeros((0,))
            vec = np.zeros((self.n, 0))
            k_ok = 0
        raise ArpackNoConvergence(msg % (num_iter, k_ok, self.k), ev, vec)

class _SymmetricArpackParams(_ArpackParams):

    def __init__(self, n, k, tp, matvec, mode=1, M_matvec=None, Minv_matvec=None, sigma=None, ncv=None, v0=None, maxiter=None, which='LM', tol=0):
        if False:
            print('Hello World!')
        if mode == 1:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=1')
            if M_matvec is not None:
                raise ValueError('M_matvec cannot be specified for mode=1')
            if Minv_matvec is not None:
                raise ValueError('Minv_matvec cannot be specified for mode=1')
            self.OP = matvec
            self.B = lambda x: x
            self.bmat = 'I'
        elif mode == 2:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=2')
            if M_matvec is None:
                raise ValueError('M_matvec must be specified for mode=2')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode=2')
            self.OP = lambda x: Minv_matvec(matvec(x))
            self.OPa = Minv_matvec
            self.OPb = matvec
            self.B = M_matvec
            self.bmat = 'G'
        elif mode == 3:
            if matvec is not None:
                raise ValueError('matvec must not be specified for mode=3')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode=3')
            if M_matvec is None:
                self.OP = Minv_matvec
                self.OPa = Minv_matvec
                self.B = lambda x: x
                self.bmat = 'I'
            else:
                self.OP = lambda x: Minv_matvec(M_matvec(x))
                self.OPa = Minv_matvec
                self.B = M_matvec
                self.bmat = 'G'
        elif mode == 4:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=4')
            if M_matvec is not None:
                raise ValueError('M_matvec must not be specified for mode=4')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode=4')
            self.OPa = Minv_matvec
            self.OP = lambda x: self.OPa(matvec(x))
            self.B = matvec
            self.bmat = 'G'
        elif mode == 5:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=5')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode=5')
            self.OPa = Minv_matvec
            self.A_matvec = matvec
            if M_matvec is None:
                self.OP = lambda x: Minv_matvec(matvec(x) + sigma * x)
                self.B = lambda x: x
                self.bmat = 'I'
            else:
                self.OP = lambda x: Minv_matvec(matvec(x) + sigma * M_matvec(x))
                self.B = M_matvec
                self.bmat = 'G'
        else:
            raise ValueError('mode=%i not implemented' % mode)
        if which not in _SEUPD_WHICH:
            raise ValueError('which must be one of %s' % ' '.join(_SEUPD_WHICH))
        if k >= n:
            raise ValueError('k must be less than ndim(A), k=%d' % k)
        _ArpackParams.__init__(self, n, k, tp, mode, sigma, ncv, v0, maxiter, which, tol)
        if self.ncv > n or self.ncv <= k:
            raise ValueError('ncv must be k<ncv<=n, ncv=%s' % self.ncv)
        self.workd = _aligned_zeros(3 * n, self.tp)
        self.workl = _aligned_zeros(self.ncv * (self.ncv + 8), self.tp)
        ltr = _type_conv[self.tp]
        if ltr not in ['s', 'd']:
            raise ValueError('Input matrix is not real-valued.')
        self._arpack_solver = _arpack.__dict__[ltr + 'saupd']
        self._arpack_extract = _arpack.__dict__[ltr + 'seupd']
        self.iterate_infodict = _SAUPD_ERRORS[ltr]
        self.extract_infodict = _SEUPD_ERRORS[ltr]
        self.ipntr = np.zeros(11, arpack_int)

    def iterate(self):
        if False:
            i = 10
            return i + 15
        (self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info) = self._arpack_solver(self.ido, self.bmat, self.which, self.k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.info)
        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            if self.mode == 1:
                self.workd[yslice] = self.OP(self.workd[xslice])
            elif self.mode == 2:
                self.workd[xslice] = self.OPb(self.workd[xslice])
                self.workd[yslice] = self.OPa(self.workd[xslice])
            elif self.mode == 5:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                Ax = self.A_matvec(self.workd[xslice])
                self.workd[yslice] = self.OPa(Ax + self.sigma * self.workd[Bxslice])
            else:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            raise ValueError('ARPACK requested user shifts.  Assure ISHIFT==0')
        else:
            self.converged = True
            if self.info == 0:
                pass
            elif self.info == 1:
                self._raise_no_convergence()
            else:
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        if False:
            return 10
        rvec = return_eigenvectors
        ierr = 0
        howmny = 'A'
        sselect = np.zeros(self.ncv, 'int')
        (d, z, ierr) = self._arpack_extract(rvec, howmny, sselect, self.sigma, self.bmat, self.which, self.k, self.tol, self.resid, self.v, self.iparam[0:7], self.ipntr, self.workd[0:2 * self.n], self.workl, ierr)
        if ierr != 0:
            raise ArpackError(ierr, infodict=self.extract_infodict)
        k_ok = self.iparam[4]
        d = d[:k_ok]
        z = z[:, :k_ok]
        if return_eigenvectors:
            return (d, z)
        else:
            return d

class _UnsymmetricArpackParams(_ArpackParams):

    def __init__(self, n, k, tp, matvec, mode=1, M_matvec=None, Minv_matvec=None, sigma=None, ncv=None, v0=None, maxiter=None, which='LM', tol=0):
        if False:
            for i in range(10):
                print('nop')
        if mode == 1:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=1')
            if M_matvec is not None:
                raise ValueError('M_matvec cannot be specified for mode=1')
            if Minv_matvec is not None:
                raise ValueError('Minv_matvec cannot be specified for mode=1')
            self.OP = matvec
            self.B = lambda x: x
            self.bmat = 'I'
        elif mode == 2:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=2')
            if M_matvec is None:
                raise ValueError('M_matvec must be specified for mode=2')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode=2')
            self.OP = lambda x: Minv_matvec(matvec(x))
            self.OPa = Minv_matvec
            self.OPb = matvec
            self.B = M_matvec
            self.bmat = 'G'
        elif mode in (3, 4):
            if matvec is None:
                raise ValueError('matvec must be specified for mode in (3,4)')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode in (3,4)')
            self.matvec = matvec
            if tp in 'DF':
                if mode == 3:
                    self.OPa = Minv_matvec
                else:
                    raise ValueError('mode=4 invalid for complex A')
            elif mode == 3:
                self.OPa = lambda x: np.real(Minv_matvec(x))
            else:
                self.OPa = lambda x: np.imag(Minv_matvec(x))
            if M_matvec is None:
                self.B = lambda x: x
                self.bmat = 'I'
                self.OP = self.OPa
            else:
                self.B = M_matvec
                self.bmat = 'G'
                self.OP = lambda x: self.OPa(M_matvec(x))
        else:
            raise ValueError('mode=%i not implemented' % mode)
        if which not in _NEUPD_WHICH:
            raise ValueError('Parameter which must be one of %s' % ' '.join(_NEUPD_WHICH))
        if k >= n - 1:
            raise ValueError('k must be less than ndim(A)-1, k=%d' % k)
        _ArpackParams.__init__(self, n, k, tp, mode, sigma, ncv, v0, maxiter, which, tol)
        if self.ncv > n or self.ncv <= k + 1:
            raise ValueError('ncv must be k+1<ncv<=n, ncv=%s' % self.ncv)
        self.workd = _aligned_zeros(3 * n, self.tp)
        self.workl = _aligned_zeros(3 * self.ncv * (self.ncv + 2), self.tp)
        ltr = _type_conv[self.tp]
        self._arpack_solver = _arpack.__dict__[ltr + 'naupd']
        self._arpack_extract = _arpack.__dict__[ltr + 'neupd']
        self.iterate_infodict = _NAUPD_ERRORS[ltr]
        self.extract_infodict = _NEUPD_ERRORS[ltr]
        self.ipntr = np.zeros(14, arpack_int)
        if self.tp in 'FD':
            self.rwork = _aligned_zeros(self.ncv, self.tp.lower())
        else:
            self.rwork = None

    def iterate(self):
        if False:
            i = 10
            return i + 15
        if self.tp in 'fd':
            (self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info) = self._arpack_solver(self.ido, self.bmat, self.which, self.k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.info)
        else:
            (self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info) = self._arpack_solver(self.ido, self.bmat, self.which, self.k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.rwork, self.info)
        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            if self.mode in (1, 2):
                self.workd[yslice] = self.OP(self.workd[xslice])
            else:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            raise ValueError('ARPACK requested user shifts.  Assure ISHIFT==0')
        else:
            self.converged = True
            if self.info == 0:
                pass
            elif self.info == 1:
                self._raise_no_convergence()
            else:
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        if False:
            for i in range(10):
                print('nop')
        (k, n) = (self.k, self.n)
        ierr = 0
        howmny = 'A'
        sselect = np.zeros(self.ncv, 'int')
        sigmar = np.real(self.sigma)
        sigmai = np.imag(self.sigma)
        workev = np.zeros(3 * self.ncv, self.tp)
        if self.tp in 'fd':
            dr = np.zeros(k + 1, self.tp)
            di = np.zeros(k + 1, self.tp)
            zr = np.zeros((n, k + 1), self.tp)
            (dr, di, zr, ierr) = self._arpack_extract(return_eigenvectors, howmny, sselect, sigmar, sigmai, workev, self.bmat, self.which, k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.info)
            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)
            nreturned = self.iparam[4]
            d = dr + 1j * di
            z = zr.astype(self.tp.upper())
            if sigmai == 0:
                i = 0
                while i <= k:
                    if abs(d[i].imag) != 0:
                        if i < k:
                            z[:, i] = zr[:, i] + 1j * zr[:, i + 1]
                            z[:, i + 1] = z[:, i].conjugate()
                            i += 1
                        else:
                            nreturned -= 1
                    i += 1
            else:
                i = 0
                while i <= k:
                    if abs(d[i].imag) == 0:
                        d[i] = np.dot(zr[:, i], self.matvec(zr[:, i]))
                    elif i < k:
                        z[:, i] = zr[:, i] + 1j * zr[:, i + 1]
                        z[:, i + 1] = z[:, i].conjugate()
                        d[i] = np.dot(zr[:, i], self.matvec(zr[:, i])) + np.dot(zr[:, i + 1], self.matvec(zr[:, i + 1])) + 1j * (np.dot(zr[:, i], self.matvec(zr[:, i + 1])) - np.dot(zr[:, i + 1], self.matvec(zr[:, i])))
                        d[i + 1] = d[i].conj()
                        i += 1
                    else:
                        nreturned -= 1
                    i += 1
            if nreturned <= k:
                d = d[:nreturned]
                z = z[:, :nreturned]
            else:
                if self.mode in (1, 2):
                    rd = d
                elif self.mode in (3, 4):
                    rd = 1 / (d - self.sigma)
                if self.which in ['LR', 'SR']:
                    ind = np.argsort(rd.real)
                elif self.which in ['LI', 'SI']:
                    ind = np.argsort(abs(rd.imag))
                else:
                    ind = np.argsort(abs(rd))
                if self.which in ['LR', 'LM', 'LI']:
                    ind = ind[-k:][::-1]
                elif self.which in ['SR', 'SM', 'SI']:
                    ind = ind[:k]
                d = d[ind]
                z = z[:, ind]
        else:
            (d, z, ierr) = self._arpack_extract(return_eigenvectors, howmny, sselect, self.sigma, workev, self.bmat, self.which, k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.rwork, ierr)
            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)
            k_ok = self.iparam[4]
            d = d[:k_ok]
            z = z[:, :k_ok]
        if return_eigenvectors:
            return (d, z)
        else:
            return d

def _aslinearoperator_with_dtype(m):
    if False:
        for i in range(10):
            print('nop')
    m = aslinearoperator(m)
    if not hasattr(m, 'dtype'):
        x = np.zeros(m.shape[1])
        m.dtype = (m * x).dtype
    return m

class SpLuInv(LinearOperator):
    """
    SpLuInv:
       helper class to repeatedly solve M*x=b
       using a sparse LU-decomposition of M
    """

    def __init__(self, M):
        if False:
            return 10
        self.M_lu = splu(M)
        self.shape = M.shape
        self.dtype = M.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)

    def _matvec(self, x):
        if False:
            i = 10
            return i + 15
        x = np.asarray(x)
        if self.isreal and np.issubdtype(x.dtype, np.complexfloating):
            return self.M_lu.solve(np.real(x).astype(self.dtype)) + 1j * self.M_lu.solve(np.imag(x).astype(self.dtype))
        else:
            return self.M_lu.solve(x.astype(self.dtype))

class LuInv(LinearOperator):
    """
    LuInv:
       helper class to repeatedly solve M*x=b
       using an LU-decomposition of M
    """

    def __init__(self, M):
        if False:
            while True:
                i = 10
        self.M_lu = lu_factor(M)
        self.shape = M.shape
        self.dtype = M.dtype

    def _matvec(self, x):
        if False:
            for i in range(10):
                print('nop')
        return lu_solve(self.M_lu, x)

def gmres_loose(A, b, tol):
    if False:
        return 10
    '\n    gmres with looser termination condition.\n    '
    b = np.asarray(b)
    min_tol = 1000 * np.sqrt(b.size) * np.finfo(b.dtype).eps
    return gmres(A, b, rtol=max(tol, min_tol), atol=0)

class IterInv(LinearOperator):
    """
    IterInv:
       helper class to repeatedly solve M*x=b
       using an iterative method.
    """

    def __init__(self, M, ifunc=gmres_loose, tol=0):
        if False:
            i = 10
            return i + 15
        self.M = M
        if hasattr(M, 'dtype'):
            self.dtype = M.dtype
        else:
            x = np.zeros(M.shape[1])
            self.dtype = (M * x).dtype
        self.shape = M.shape
        if tol <= 0:
            tol = 2 * np.finfo(self.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        if False:
            while True:
                i = 10
        (b, info) = self.ifunc(self.M, x, tol=self.tol)
        if info != 0:
            raise ValueError('Error in inverting M: function %s did not converge (info = %i).' % (self.ifunc.__name__, info))
        return b

class IterOpInv(LinearOperator):
    """
    IterOpInv:
       helper class to repeatedly solve [A-sigma*M]*x = b
       using an iterative method
    """

    def __init__(self, A, M, sigma, ifunc=gmres_loose, tol=0):
        if False:
            return 10
        self.A = A
        self.M = M
        self.sigma = sigma

        def mult_func(x):
            if False:
                for i in range(10):
                    print('nop')
            return A.matvec(x) - sigma * M.matvec(x)

        def mult_func_M_None(x):
            if False:
                print('Hello World!')
            return A.matvec(x) - sigma * x
        x = np.zeros(A.shape[1])
        if M is None:
            dtype = mult_func_M_None(x).dtype
            self.OP = LinearOperator(self.A.shape, mult_func_M_None, dtype=dtype)
        else:
            dtype = mult_func(x).dtype
            self.OP = LinearOperator(self.A.shape, mult_func, dtype=dtype)
        self.shape = A.shape
        if tol <= 0:
            tol = 2 * np.finfo(self.OP.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        if False:
            i = 10
            return i + 15
        (b, info) = self.ifunc(self.OP, x, tol=self.tol)
        if info != 0:
            raise ValueError('Error in inverting [A-sigma*M]: function %s did not converge (info = %i).' % (self.ifunc.__name__, info))
        return b

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self.OP.dtype

def _fast_spmatrix_to_csc(A, hermitian=False):
    if False:
        i = 10
        return i + 15
    'Convert sparse matrix to CSC (by transposing, if possible)'
    if A.format == 'csr' and hermitian and (not np.issubdtype(A.dtype, np.complexfloating)):
        return A.T
    elif is_pydata_spmatrix(A):
        return A
    else:
        return A.tocsc()

def get_inv_matvec(M, hermitian=False, tol=0):
    if False:
        for i in range(10):
            print('nop')
    if isdense(M):
        return LuInv(M).matvec
    elif issparse(M) or is_pydata_spmatrix(M):
        M = _fast_spmatrix_to_csc(M, hermitian=hermitian)
        return SpLuInv(M).matvec
    else:
        return IterInv(M, tol=tol).matvec

def get_OPinv_matvec(A, M, sigma, hermitian=False, tol=0):
    if False:
        print('Hello World!')
    if sigma == 0:
        return get_inv_matvec(A, hermitian=hermitian, tol=tol)
    if M is None:
        if isdense(A):
            if np.issubdtype(A.dtype, np.complexfloating) or np.imag(sigma) == 0:
                A = np.copy(A)
            else:
                A = A + 0j
            A.flat[::A.shape[1] + 1] -= sigma
            return LuInv(A).matvec
        elif issparse(A) or is_pydata_spmatrix(A):
            A = A - sigma * eye(A.shape[0])
            A = _fast_spmatrix_to_csc(A, hermitian=hermitian)
            return SpLuInv(A).matvec
        else:
            return IterOpInv(_aslinearoperator_with_dtype(A), M, sigma, tol=tol).matvec
    elif not isdense(A) and (not issparse(A)) and (not is_pydata_spmatrix(A)) or (not isdense(M) and (not issparse(M)) and (not is_pydata_spmatrix(A))):
        return IterOpInv(_aslinearoperator_with_dtype(A), _aslinearoperator_with_dtype(M), sigma, tol=tol).matvec
    elif isdense(A) or isdense(M):
        return LuInv(A - sigma * M).matvec
    else:
        OP = A - sigma * M
        OP = _fast_spmatrix_to_csc(OP, hermitian=hermitian)
        return SpLuInv(OP).matvec
_ARPACK_LOCK = ReentrancyLock('Nested calls to eigs/eighs not allowed: ARPACK is not re-entrant')

def eigs(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, OPpart=None):
    if False:
        while True:
            i = 10
    "\n    Find k eigenvalues and eigenvectors of the square matrix A.\n\n    Solves ``A @ x[i] = w[i] * x[i]``, the standard eigenvalue problem\n    for w[i] eigenvalues with corresponding eigenvectors x[i].\n\n    If M is specified, solves ``A @ x[i] = w[i] * M @ x[i]``, the\n    generalized eigenvalue problem for w[i] eigenvalues\n    with corresponding eigenvectors x[i]\n\n    Parameters\n    ----------\n    A : ndarray, sparse matrix or LinearOperator\n        An array, sparse matrix, or LinearOperator representing\n        the operation ``A @ x``, where A is a real or complex square matrix.\n    k : int, optional\n        The number of eigenvalues and eigenvectors desired.\n        `k` must be smaller than N-1. It is not possible to compute all\n        eigenvectors of a matrix.\n    M : ndarray, sparse matrix or LinearOperator, optional\n        An array, sparse matrix, or LinearOperator representing\n        the operation M@x for the generalized eigenvalue problem\n\n            A @ x = w * M @ x.\n\n        M must represent a real symmetric matrix if A is real, and must\n        represent a complex Hermitian matrix if A is complex. For best\n        results, the data type of M should be the same as that of A.\n        Additionally:\n\n            If `sigma` is None, M is positive definite\n\n            If sigma is specified, M is positive semi-definite\n\n        If sigma is None, eigs requires an operator to compute the solution\n        of the linear equation ``M @ x = b``.  This is done internally via a\n        (sparse) LU decomposition for an explicit matrix M, or via an\n        iterative solver for a general linear operator.  Alternatively,\n        the user can supply the matrix or operator Minv, which gives\n        ``x = Minv @ b = M^-1 @ b``.\n    sigma : real or complex, optional\n        Find eigenvalues near sigma using shift-invert mode.  This requires\n        an operator to compute the solution of the linear system\n        ``[A - sigma * M] @ x = b``, where M is the identity matrix if\n        unspecified. This is computed internally via a (sparse) LU\n        decomposition for explicit matrices A & M, or via an iterative\n        solver if either A or M is a general linear operator.\n        Alternatively, the user can supply the matrix or operator OPinv,\n        which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.\n        For a real matrix A, shift-invert can either be done in imaginary\n        mode or real mode, specified by the parameter OPpart ('r' or 'i').\n        Note that when sigma is specified, the keyword 'which' (below)\n        refers to the shifted eigenvalues ``w'[i]`` where:\n\n            If A is real and OPpart == 'r' (default),\n              ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.\n\n            If A is real and OPpart == 'i',\n              ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.\n\n            If A is complex, ``w'[i] = 1/(w[i]-sigma)``.\n\n    v0 : ndarray, optional\n        Starting vector for iteration.\n        Default: random\n    ncv : int, optional\n        The number of Lanczos vectors generated\n        `ncv` must be greater than `k`; it is recommended that ``ncv > 2*k``.\n        Default: ``min(n, max(2*k + 1, 20))``\n    which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional\n        Which `k` eigenvectors and eigenvalues to find:\n\n            'LM' : largest magnitude\n\n            'SM' : smallest magnitude\n\n            'LR' : largest real part\n\n            'SR' : smallest real part\n\n            'LI' : largest imaginary part\n\n            'SI' : smallest imaginary part\n\n        When sigma != None, 'which' refers to the shifted eigenvalues w'[i]\n        (see discussion in 'sigma', above).  ARPACK is generally better\n        at finding large values than small values.  If small eigenvalues are\n        desired, consider using shift-invert mode for better performance.\n    maxiter : int, optional\n        Maximum number of Arnoldi update iterations allowed\n        Default: ``n*10``\n    tol : float, optional\n        Relative accuracy for eigenvalues (stopping criterion)\n        The default value of 0 implies machine precision.\n    return_eigenvectors : bool, optional\n        Return eigenvectors (True) in addition to eigenvalues\n    Minv : ndarray, sparse matrix or LinearOperator, optional\n        See notes in M, above.\n    OPinv : ndarray, sparse matrix or LinearOperator, optional\n        See notes in sigma, above.\n    OPpart : {'r' or 'i'}, optional\n        See notes in sigma, above\n\n    Returns\n    -------\n    w : ndarray\n        Array of k eigenvalues.\n    v : ndarray\n        An array of `k` eigenvectors.\n        ``v[:, i]`` is the eigenvector corresponding to the eigenvalue w[i].\n\n    Raises\n    ------\n    ArpackNoConvergence\n        When the requested convergence is not obtained.\n        The currently converged eigenvalues and eigenvectors can be found\n        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception\n        object.\n\n    See Also\n    --------\n    eigsh : eigenvalues and eigenvectors for symmetric matrix A\n    svds : singular value decomposition for a matrix A\n\n    Notes\n    -----\n    This function is a wrapper to the ARPACK [1]_ SNEUPD, DNEUPD, CNEUPD,\n    ZNEUPD, functions which use the Implicitly Restarted Arnoldi Method to\n    find the eigenvalues and eigenvectors [2]_.\n\n    References\n    ----------\n    .. [1] ARPACK Software, https://github.com/opencollab/arpack-ng\n    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:\n       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted\n       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.\n\n    Examples\n    --------\n    Find 6 eigenvectors of the identity matrix:\n\n    >>> import numpy as np\n    >>> from scipy.sparse.linalg import eigs\n    >>> id = np.eye(13)\n    >>> vals, vecs = eigs(id, k=6)\n    >>> vals\n    array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])\n    >>> vecs.shape\n    (13, 6)\n\n    "
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix (shape={A.shape})')
    if M is not None:
        if M.shape != A.shape:
            raise ValueError('wrong M dimensions %s, should be %s' % (M.shape, A.shape))
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            warnings.warn('M does not have the same type precision as A. This may adversely affect ARPACK convergence')
    n = A.shape[0]
    if k <= 0:
        raise ValueError('k=%d must be greater than 0.' % k)
    if k >= n - 1:
        warnings.warn('k >= N - 1 for N * N square matrix. Attempting to use scipy.linalg.eig instead.', RuntimeWarning)
        if issparse(A):
            raise TypeError('Cannot use scipy.linalg.eig for sparse A with k >= N - 1. Use scipy.linalg.eig(A.toarray()) or reduce k.')
        if isinstance(A, LinearOperator):
            raise TypeError('Cannot use scipy.linalg.eig for LinearOperator A with k >= N - 1.')
        if isinstance(M, LinearOperator):
            raise TypeError('Cannot use scipy.linalg.eig for LinearOperator M with k >= N - 1.')
        return eig(A, b=M, right=return_eigenvectors)
    if sigma is None:
        matvec = _aslinearoperator_with_dtype(A).matvec
        if OPinv is not None:
            raise ValueError('OPinv should not be specified with sigma = None.')
        if OPpart is not None:
            raise ValueError('OPpart should not be specified with sigma = None or complex A')
        if M is None:
            mode = 1
            M_matvec = None
            Minv_matvec = None
            if Minv is not None:
                raise ValueError('Minv should not be specified with M = None.')
        else:
            mode = 2
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = _aslinearoperator_with_dtype(Minv)
                Minv_matvec = Minv.matvec
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    else:
        if np.issubdtype(A.dtype, np.complexfloating):
            if OPpart is not None:
                raise ValueError('OPpart should not be specified with sigma=None or complex A')
            mode = 3
        elif OPpart is None or OPpart.lower() == 'r':
            mode = 3
        elif OPpart.lower() == 'i':
            if np.imag(sigma) == 0:
                raise ValueError("OPpart cannot be 'i' if sigma is real")
            mode = 4
        else:
            raise ValueError("OPpart must be one of ('r','i')")
        matvec = _aslinearoperator_with_dtype(A).matvec
        if Minv is not None:
            raise ValueError('Minv should not be specified when sigma is')
        if OPinv is None:
            Minv_matvec = get_OPinv_matvec(A, M, sigma, hermitian=False, tol=tol)
        else:
            OPinv = _aslinearoperator_with_dtype(OPinv)
            Minv_matvec = OPinv.matvec
        if M is None:
            M_matvec = None
        else:
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    params = _UnsymmetricArpackParams(n, k, A.dtype.char, matvec, mode, M_matvec, Minv_matvec, sigma, ncv, v0, maxiter, which, tol)
    with _ARPACK_LOCK:
        while not params.converged:
            params.iterate()
        return params.extract(return_eigenvectors)

def eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal'):
    if False:
        i = 10
        return i + 15
    "\n    Find k eigenvalues and eigenvectors of the real symmetric square matrix\n    or complex Hermitian matrix A.\n\n    Solves ``A @ x[i] = w[i] * x[i]``, the standard eigenvalue problem for\n    w[i] eigenvalues with corresponding eigenvectors x[i].\n\n    If M is specified, solves ``A @ x[i] = w[i] * M @ x[i]``, the\n    generalized eigenvalue problem for w[i] eigenvalues\n    with corresponding eigenvectors x[i].\n\n    Note that there is no specialized routine for the case when A is a complex\n    Hermitian matrix. In this case, ``eigsh()`` will call ``eigs()`` and return the\n    real parts of the eigenvalues thus obtained.\n\n    Parameters\n    ----------\n    A : ndarray, sparse matrix or LinearOperator\n        A square operator representing the operation ``A @ x``, where ``A`` is\n        real symmetric or complex Hermitian. For buckling mode (see below)\n        ``A`` must additionally be positive-definite.\n    k : int, optional\n        The number of eigenvalues and eigenvectors desired.\n        `k` must be smaller than N. It is not possible to compute all\n        eigenvectors of a matrix.\n\n    Returns\n    -------\n    w : array\n        Array of k eigenvalues.\n    v : array\n        An array representing the `k` eigenvectors.  The column ``v[:, i]`` is\n        the eigenvector corresponding to the eigenvalue ``w[i]``.\n\n    Other Parameters\n    ----------------\n    M : An N x N matrix, array, sparse matrix, or linear operator representing\n        the operation ``M @ x`` for the generalized eigenvalue problem\n\n            A @ x = w * M @ x.\n\n        M must represent a real symmetric matrix if A is real, and must\n        represent a complex Hermitian matrix if A is complex. For best\n        results, the data type of M should be the same as that of A.\n        Additionally:\n\n            If sigma is None, M is symmetric positive definite.\n\n            If sigma is specified, M is symmetric positive semi-definite.\n\n            In buckling mode, M is symmetric indefinite.\n\n        If sigma is None, eigsh requires an operator to compute the solution\n        of the linear equation ``M @ x = b``. This is done internally via a\n        (sparse) LU decomposition for an explicit matrix M, or via an\n        iterative solver for a general linear operator.  Alternatively,\n        the user can supply the matrix or operator Minv, which gives\n        ``x = Minv @ b = M^-1 @ b``.\n    sigma : real\n        Find eigenvalues near sigma using shift-invert mode.  This requires\n        an operator to compute the solution of the linear system\n        ``[A - sigma * M] x = b``, where M is the identity matrix if\n        unspecified.  This is computed internally via a (sparse) LU\n        decomposition for explicit matrices A & M, or via an iterative\n        solver if either A or M is a general linear operator.\n        Alternatively, the user can supply the matrix or operator OPinv,\n        which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.\n        Note that when sigma is specified, the keyword 'which' refers to\n        the shifted eigenvalues ``w'[i]`` where:\n\n            if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.\n\n            if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.\n\n            if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.\n\n        (see further discussion in 'mode' below)\n    v0 : ndarray, optional\n        Starting vector for iteration.\n        Default: random\n    ncv : int, optional\n        The number of Lanczos vectors generated ncv must be greater than k and\n        smaller than n; it is recommended that ``ncv > 2*k``.\n        Default: ``min(n, max(2*k + 1, 20))``\n    which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']\n        If A is a complex Hermitian matrix, 'BE' is invalid.\n        Which `k` eigenvectors and eigenvalues to find:\n\n            'LM' : Largest (in magnitude) eigenvalues.\n\n            'SM' : Smallest (in magnitude) eigenvalues.\n\n            'LA' : Largest (algebraic) eigenvalues.\n\n            'SA' : Smallest (algebraic) eigenvalues.\n\n            'BE' : Half (k/2) from each end of the spectrum.\n\n        When k is odd, return one more (k/2+1) from the high end.\n        When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``\n        (see discussion in 'sigma', above).  ARPACK is generally better\n        at finding large values than small values.  If small eigenvalues are\n        desired, consider using shift-invert mode for better performance.\n    maxiter : int, optional\n        Maximum number of Arnoldi update iterations allowed.\n        Default: ``n*10``\n    tol : float\n        Relative accuracy for eigenvalues (stopping criterion).\n        The default value of 0 implies machine precision.\n    Minv : N x N matrix, array, sparse matrix, or LinearOperator\n        See notes in M, above.\n    OPinv : N x N matrix, array, sparse matrix, or LinearOperator\n        See notes in sigma, above.\n    return_eigenvectors : bool\n        Return eigenvectors (True) in addition to eigenvalues.\n        This value determines the order in which eigenvalues are sorted.\n        The sort order is also dependent on the `which` variable.\n\n            For which = 'LM' or 'SA':\n                If `return_eigenvectors` is True, eigenvalues are sorted by\n                algebraic value.\n\n                If `return_eigenvectors` is False, eigenvalues are sorted by\n                absolute value.\n\n            For which = 'BE' or 'LA':\n                eigenvalues are always sorted by algebraic value.\n\n            For which = 'SM':\n                If `return_eigenvectors` is True, eigenvalues are sorted by\n                algebraic value.\n\n                If `return_eigenvectors` is False, eigenvalues are sorted by\n                decreasing absolute value.\n\n    mode : string ['normal' | 'buckling' | 'cayley']\n        Specify strategy to use for shift-invert mode.  This argument applies\n        only for real-valued A and sigma != None.  For shift-invert mode,\n        ARPACK internally solves the eigenvalue problem\n        ``OP @ x'[i] = w'[i] * B @ x'[i]``\n        and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]\n        into the desired eigenvectors and eigenvalues of the problem\n        ``A @ x[i] = w[i] * M @ x[i]``.\n        The modes are as follows:\n\n            'normal' :\n                OP = [A - sigma * M]^-1 @ M,\n                B = M,\n                w'[i] = 1 / (w[i] - sigma)\n\n            'buckling' :\n                OP = [A - sigma * M]^-1 @ A,\n                B = A,\n                w'[i] = w[i] / (w[i] - sigma)\n\n            'cayley' :\n                OP = [A - sigma * M]^-1 @ [A + sigma * M],\n                B = M,\n                w'[i] = (w[i] + sigma) / (w[i] - sigma)\n\n        The choice of mode will affect which eigenvalues are selected by\n        the keyword 'which', and can also impact the stability of\n        convergence (see [2] for a discussion).\n\n    Raises\n    ------\n    ArpackNoConvergence\n        When the requested convergence is not obtained.\n\n        The currently converged eigenvalues and eigenvectors can be found\n        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception\n        object.\n\n    See Also\n    --------\n    eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A\n    svds : singular value decomposition for a matrix A\n\n    Notes\n    -----\n    This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD\n    functions which use the Implicitly Restarted Lanczos Method to\n    find the eigenvalues and eigenvectors [2]_.\n\n    References\n    ----------\n    .. [1] ARPACK Software, https://github.com/opencollab/arpack-ng\n    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:\n       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted\n       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse.linalg import eigsh\n    >>> identity = np.eye(13)\n    >>> eigenvalues, eigenvectors = eigsh(identity, k=6)\n    >>> eigenvalues\n    array([1., 1., 1., 1., 1., 1.])\n    >>> eigenvectors.shape\n    (13, 6)\n\n    "
    if np.issubdtype(A.dtype, np.complexfloating):
        if mode != 'normal':
            raise ValueError('mode=%s cannot be used with complex matrix A' % mode)
        if which == 'BE':
            raise ValueError("which='BE' cannot be used with complex matrix A")
        elif which == 'LA':
            which = 'LR'
        elif which == 'SA':
            which = 'SR'
        ret = eigs(A, k, M=M, sigma=sigma, which=which, v0=v0, ncv=ncv, maxiter=maxiter, tol=tol, return_eigenvectors=return_eigenvectors, Minv=Minv, OPinv=OPinv)
        if return_eigenvectors:
            return (ret[0].real, ret[1])
        else:
            return ret.real
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix (shape={A.shape})')
    if M is not None:
        if M.shape != A.shape:
            raise ValueError('wrong M dimensions %s, should be %s' % (M.shape, A.shape))
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            warnings.warn('M does not have the same type precision as A. This may adversely affect ARPACK convergence')
    n = A.shape[0]
    if k <= 0:
        raise ValueError('k must be greater than 0.')
    if k >= n:
        warnings.warn('k >= N for N * N square matrix. Attempting to use scipy.linalg.eigh instead.', RuntimeWarning)
        if issparse(A):
            raise TypeError('Cannot use scipy.linalg.eigh for sparse A with k >= N. Use scipy.linalg.eigh(A.toarray()) or reduce k.')
        if isinstance(A, LinearOperator):
            raise TypeError('Cannot use scipy.linalg.eigh for LinearOperator A with k >= N.')
        if isinstance(M, LinearOperator):
            raise TypeError('Cannot use scipy.linalg.eigh for LinearOperator M with k >= N.')
        return eigh(A, b=M, eigvals_only=not return_eigenvectors)
    if sigma is None:
        A = _aslinearoperator_with_dtype(A)
        matvec = A.matvec
        if OPinv is not None:
            raise ValueError('OPinv should not be specified with sigma = None.')
        if M is None:
            mode = 1
            M_matvec = None
            Minv_matvec = None
            if Minv is not None:
                raise ValueError('Minv should not be specified with M = None.')
        else:
            mode = 2
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = _aslinearoperator_with_dtype(Minv)
                Minv_matvec = Minv.matvec
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    else:
        if Minv is not None:
            raise ValueError('Minv should not be specified when sigma is')
        if mode == 'normal':
            mode = 3
            matvec = None
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma, hermitian=True, tol=tol)
            else:
                OPinv = _aslinearoperator_with_dtype(OPinv)
                Minv_matvec = OPinv.matvec
            if M is None:
                M_matvec = None
            else:
                M = _aslinearoperator_with_dtype(M)
                M_matvec = M.matvec
        elif mode == 'buckling':
            mode = 4
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma, hermitian=True, tol=tol)
            else:
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            matvec = _aslinearoperator_with_dtype(A).matvec
            M_matvec = None
        elif mode == 'cayley':
            mode = 5
            matvec = _aslinearoperator_with_dtype(A).matvec
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma, hermitian=True, tol=tol)
            else:
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            if M is None:
                M_matvec = None
            else:
                M_matvec = _aslinearoperator_with_dtype(M).matvec
        else:
            raise ValueError("unrecognized mode '%s'" % mode)
    params = _SymmetricArpackParams(n, k, A.dtype.char, matvec, mode, M_matvec, Minv_matvec, sigma, ncv, v0, maxiter, which, tol)
    with _ARPACK_LOCK:
        while not params.converged:
            params.iterate()
        return params.extract(return_eigenvectors)