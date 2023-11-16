"""Hessian update strategies for quasi-Newton optimization methods."""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import get_blas_funcs
from warnings import warn
__all__ = ['HessianUpdateStrategy', 'BFGS', 'SR1']

class HessianUpdateStrategy:
    """Interface for implementing Hessian update strategies.

    Many optimization methods make use of Hessian (or inverse Hessian)
    approximations, such as the quasi-Newton methods BFGS, SR1, L-BFGS.
    Some of these  approximations, however, do not actually need to store
    the entire matrix or can compute the internal matrix product with a
    given vector in a very efficiently manner. This class serves as an
    abstract interface between the optimization algorithm and the
    quasi-Newton update strategies, giving freedom of implementation
    to store and update the internal matrix as efficiently as possible.
    Different choices of initialization and update procedure will result
    in different quasi-Newton strategies.

    Four methods should be implemented in derived classes: ``initialize``,
    ``update``, ``dot`` and ``get_matrix``.

    Notes
    -----
    Any instance of a class that implements this interface,
    can be accepted by the method ``minimize`` and used by
    the compatible solvers to approximate the Hessian (or
    inverse Hessian) used by the optimization algorithms.
    """

    def initialize(self, n, approx_type):
        if False:
            i = 10
            return i + 15
        "Initialize internal matrix.\n\n        Allocate internal memory for storing and updating\n        the Hessian or its inverse.\n\n        Parameters\n        ----------\n        n : int\n            Problem dimension.\n        approx_type : {'hess', 'inv_hess'}\n            Selects either the Hessian or the inverse Hessian.\n            When set to 'hess' the Hessian will be stored and updated.\n            When set to 'inv_hess' its inverse will be used instead.\n        "
        raise NotImplementedError('The method ``initialize(n, approx_type)`` is not implemented.')

    def update(self, delta_x, delta_grad):
        if False:
            return 10
        "Update internal matrix.\n\n        Update Hessian matrix or its inverse (depending on how 'approx_type'\n        is defined) using information about the last evaluated points.\n\n        Parameters\n        ----------\n        delta_x : ndarray\n            The difference between two points the gradient\n            function have been evaluated at: ``delta_x = x2 - x1``.\n        delta_grad : ndarray\n            The difference between the gradients:\n            ``delta_grad = grad(x2) - grad(x1)``.\n        "
        raise NotImplementedError('The method ``update(delta_x, delta_grad)`` is not implemented.')

    def dot(self, p):
        if False:
            print('Hello World!')
        'Compute the product of the internal matrix with the given vector.\n\n        Parameters\n        ----------\n        p : array_like\n            1-D array representing a vector.\n\n        Returns\n        -------\n        Hp : array\n            1-D represents the result of multiplying the approximation matrix\n            by vector p.\n        '
        raise NotImplementedError('The method ``dot(p)`` is not implemented.')

    def get_matrix(self):
        if False:
            while True:
                i = 10
        "Return current internal matrix.\n\n        Returns\n        -------\n        H : ndarray, shape (n, n)\n            Dense matrix containing either the Hessian\n            or its inverse (depending on how 'approx_type'\n            is defined).\n        "
        raise NotImplementedError('The method ``get_matrix(p)`` is not implemented.')

class FullHessianUpdateStrategy(HessianUpdateStrategy):
    """Hessian update strategy with full dimensional internal representation.
    """
    _syr = get_blas_funcs('syr', dtype='d')
    _syr2 = get_blas_funcs('syr2', dtype='d')
    _symv = get_blas_funcs('symv', dtype='d')

    def __init__(self, init_scale='auto'):
        if False:
            while True:
                i = 10
        self.init_scale = init_scale
        self.first_iteration = None
        self.approx_type = None
        self.B = None
        self.H = None

    def initialize(self, n, approx_type):
        if False:
            return 10
        "Initialize internal matrix.\n\n        Allocate internal memory for storing and updating\n        the Hessian or its inverse.\n\n        Parameters\n        ----------\n        n : int\n            Problem dimension.\n        approx_type : {'hess', 'inv_hess'}\n            Selects either the Hessian or the inverse Hessian.\n            When set to 'hess' the Hessian will be stored and updated.\n            When set to 'inv_hess' its inverse will be used instead.\n        "
        self.first_iteration = True
        self.n = n
        self.approx_type = approx_type
        if approx_type not in ('hess', 'inv_hess'):
            raise ValueError("`approx_type` must be 'hess' or 'inv_hess'.")
        if self.approx_type == 'hess':
            self.B = np.eye(n, dtype=float)
        else:
            self.H = np.eye(n, dtype=float)

    def _auto_scale(self, delta_x, delta_grad):
        if False:
            i = 10
            return i + 15
        s_norm2 = np.dot(delta_x, delta_x)
        y_norm2 = np.dot(delta_grad, delta_grad)
        ys = np.abs(np.dot(delta_grad, delta_x))
        if ys == 0.0 or y_norm2 == 0 or s_norm2 == 0:
            return 1
        if self.approx_type == 'hess':
            return y_norm2 / ys
        else:
            return ys / y_norm2

    def _update_implementation(self, delta_x, delta_grad):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('The method ``_update_implementation`` is not implemented.')

    def update(self, delta_x, delta_grad):
        if False:
            i = 10
            return i + 15
        "Update internal matrix.\n\n        Update Hessian matrix or its inverse (depending on how 'approx_type'\n        is defined) using information about the last evaluated points.\n\n        Parameters\n        ----------\n        delta_x : ndarray\n            The difference between two points the gradient\n            function have been evaluated at: ``delta_x = x2 - x1``.\n        delta_grad : ndarray\n            The difference between the gradients:\n            ``delta_grad = grad(x2) - grad(x1)``.\n        "
        if np.all(delta_x == 0.0):
            return
        if np.all(delta_grad == 0.0):
            warn('delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.', UserWarning)
            return
        if self.first_iteration:
            if self.init_scale == 'auto':
                scale = self._auto_scale(delta_x, delta_grad)
            else:
                scale = float(self.init_scale)
            if self.approx_type == 'hess':
                self.B *= scale
            else:
                self.H *= scale
            self.first_iteration = False
        self._update_implementation(delta_x, delta_grad)

    def dot(self, p):
        if False:
            i = 10
            return i + 15
        'Compute the product of the internal matrix with the given vector.\n\n        Parameters\n        ----------\n        p : array_like\n            1-D array representing a vector.\n\n        Returns\n        -------\n        Hp : array\n            1-D represents the result of multiplying the approximation matrix\n            by vector p.\n        '
        if self.approx_type == 'hess':
            return self._symv(1, self.B, p)
        else:
            return self._symv(1, self.H, p)

    def get_matrix(self):
        if False:
            return 10
        'Return the current internal matrix.\n\n        Returns\n        -------\n        M : ndarray, shape (n, n)\n            Dense matrix containing either the Hessian or its inverse\n            (depending on how `approx_type` was defined).\n        '
        if self.approx_type == 'hess':
            M = np.copy(self.B)
        else:
            M = np.copy(self.H)
        li = np.tril_indices_from(M, k=-1)
        M[li] = M.T[li]
        return M

class BFGS(FullHessianUpdateStrategy):
    """Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.

    Parameters
    ----------
    exception_strategy : {'skip_update', 'damp_update'}, optional
        Define how to proceed when the curvature condition is violated.
        Set it to 'skip_update' to just skip the update. Or, alternatively,
        set it to 'damp_update' to interpolate between the actual BFGS
        result and the unmodified matrix. Both exceptions strategies
        are explained  in [1]_, p.536-537.
    min_curvature : float
        This number, scaled by a normalization factor, defines the
        minimum curvature ``dot(delta_grad, delta_x)`` allowed to go
        unaffected by the exception strategy. By default is equal to
        1e-8 when ``exception_strategy = 'skip_update'`` and equal
        to 0.2 when ``exception_strategy = 'damp_update'``.
    init_scale : {float, 'auto'}
        Matrix scale at first iteration. At the first
        iteration the Hessian matrix or its inverse will be initialized
        with ``init_scale*np.eye(n)``, where ``n`` is the problem dimension.
        Set it to 'auto' in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1]_, p.143.
        By default uses 'auto'.

    Notes
    -----
    The update is based on the description in [1]_, p.140.

    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    def __init__(self, exception_strategy='skip_update', min_curvature=None, init_scale='auto'):
        if False:
            print('Hello World!')
        if exception_strategy == 'skip_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 1e-08
        elif exception_strategy == 'damp_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 0.2
        else:
            raise ValueError("`exception_strategy` must be 'skip_update' or 'damp_update'.")
        super().__init__(init_scale)
        self.exception_strategy = exception_strategy

    def _update_inverse_hessian(self, ys, Hy, yHy, s):
        if False:
            i = 10
            return i + 15
        'Update the inverse Hessian matrix.\n\n        BFGS update using the formula:\n\n            ``H <- H + ((H*y).T*y + s.T*y)/(s.T*y)^2 * (s*s.T)\n                     - 1/(s.T*y) * ((H*y)*s.T + s*(H*y).T)``\n\n        where ``s = delta_x`` and ``y = delta_grad``. This formula is\n        equivalent to (6.17) in [1]_ written in a more efficient way\n        for implementation.\n\n        References\n        ----------\n        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"\n               Second Edition (2006).\n        '
        self.H = self._syr2(-1.0 / ys, s, Hy, a=self.H)
        self.H = self._syr((ys + yHy) / ys ** 2, s, a=self.H)

    def _update_hessian(self, ys, Bs, sBs, y):
        if False:
            i = 10
            return i + 15
        'Update the Hessian matrix.\n\n        BFGS update using the formula:\n\n            ``B <- B - (B*s)*(B*s).T/s.T*(B*s) + y*y^T/s.T*y``\n\n        where ``s`` is short for ``delta_x`` and ``y`` is short\n        for ``delta_grad``. Formula (6.19) in [1]_.\n\n        References\n        ----------\n        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"\n               Second Edition (2006).\n        '
        self.B = self._syr(1.0 / ys, y, a=self.B)
        self.B = self._syr(-1.0 / sBs, Bs, a=self.B)

    def _update_implementation(self, delta_x, delta_grad):
        if False:
            print('Hello World!')
        if self.approx_type == 'hess':
            w = delta_x
            z = delta_grad
        else:
            w = delta_grad
            z = delta_x
        wz = np.dot(w, z)
        Mw = self.dot(w)
        wMw = Mw.dot(w)
        if wMw <= 0.0:
            scale = self._auto_scale(delta_x, delta_grad)
            if self.approx_type == 'hess':
                self.B = scale * np.eye(self.n, dtype=float)
            else:
                self.H = scale * np.eye(self.n, dtype=float)
            Mw = self.dot(w)
            wMw = Mw.dot(w)
        if wz <= self.min_curvature * wMw:
            if self.exception_strategy == 'skip_update':
                return
            elif self.exception_strategy == 'damp_update':
                update_factor = (1 - self.min_curvature) / (1 - wz / wMw)
                z = update_factor * z + (1 - update_factor) * Mw
                wz = np.dot(w, z)
        if self.approx_type == 'hess':
            self._update_hessian(wz, Mw, wMw, z)
        else:
            self._update_inverse_hessian(wz, Mw, wMw, z)

class SR1(FullHessianUpdateStrategy):
    """Symmetric-rank-1 Hessian update strategy.

    Parameters
    ----------
    min_denominator : float
        This number, scaled by a normalization factor,
        defines the minimum denominator magnitude allowed
        in the update. When the condition is violated we skip
        the update. By default uses ``1e-8``.
    init_scale : {float, 'auto'}, optional
        Matrix scale at first iteration. At the first
        iteration the Hessian matrix or its inverse will be initialized
        with ``init_scale*np.eye(n)``, where ``n`` is the problem dimension.
        Set it to 'auto' in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1]_, p.143.
        By default uses 'auto'.

    Notes
    -----
    The update is based on the description in [1]_, p.144-146.

    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    def __init__(self, min_denominator=1e-08, init_scale='auto'):
        if False:
            return 10
        self.min_denominator = min_denominator
        super().__init__(init_scale)

    def _update_implementation(self, delta_x, delta_grad):
        if False:
            print('Hello World!')
        if self.approx_type == 'hess':
            w = delta_x
            z = delta_grad
        else:
            w = delta_grad
            z = delta_x
        Mw = self.dot(w)
        z_minus_Mw = z - Mw
        denominator = np.dot(w, z_minus_Mw)
        if np.abs(denominator) <= self.min_denominator * norm(w) * norm(z_minus_Mw):
            return
        if self.approx_type == 'hess':
            self.B = self._syr(1 / denominator, z_minus_Mw, a=self.B)
        else:
            self.H = self._syr(1 / denominator, z_minus_Mw, a=self.H)