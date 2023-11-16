"""
Newton solver for Generalized Linear Models
"""
import warnings
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg
import scipy.optimize
from ..._loss.loss import HalfSquaredError
from ...exceptions import ConvergenceWarning
from ...utils.optimize import _check_optimize_result
from .._linear_loss import LinearModelLoss

class NewtonSolver(ABC):
    """Newton solver for GLMs.

    This class implements Newton/2nd-order optimization routines for GLMs. Each Newton
    iteration aims at finding the Newton step which is done by the inner solver. With
    Hessian H, gradient g and coefficients coef, one step solves:

        H @ coef_newton = -g

    For our GLM / LinearModelLoss, we have gradient g and Hessian H:

        g = X.T @ loss.gradient + l2_reg_strength * coef
        H = X.T @ diag(loss.hessian) @ X + l2_reg_strength * identity

    Backtracking line search updates coef = coef_old + t * coef_newton for some t in
    (0, 1].

    This is a base class, actual implementations (child classes) may deviate from the
    above pattern and use structure specific tricks.

    Usage pattern:
        - initialize solver: sol = NewtonSolver(...)
        - solve the problem: sol.solve(X, y, sample_weight)

    References
    ----------
    - Jorge Nocedal, Stephen J. Wright. (2006) "Numerical Optimization"
      2nd edition
      https://doi.org/10.1007/978-0-387-40065-5

    - Stephen P. Boyd, Lieven Vandenberghe. (2004) "Convex Optimization."
      Cambridge University Press, 2004.
      https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

    Parameters
    ----------
    coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
        Initial coefficients of a linear model.
        If shape (n_classes * n_dof,), the classes of one feature are contiguous,
        i.e. one reconstructs the 2d-array via
        coef.reshape((n_classes, -1), order="F").

    linear_loss : LinearModelLoss
        The loss to be minimized.

    l2_reg_strength : float, default=0.0
        L2 regularization strength.

    tol : float, default=1e-4
        The optimization problem is solved when each of the following condition is
        fulfilled:
        1. maximum |gradient| <= tol
        2. Newton decrement d: 1/2 * d^2 <= tol

    max_iter : int, default=100
        Maximum number of Newton steps allowed.

    n_threads : int, default=1
        Number of OpenMP threads to use for the computation of the Hessian and gradient
        of the loss function.

    Attributes
    ----------
    coef_old : ndarray of shape coef.shape
        Coefficient of previous iteration.

    coef_newton : ndarray of shape coef.shape
        Newton step.

    gradient : ndarray of shape coef.shape
        Gradient of the loss w.r.t. the coefficients.

    gradient_old : ndarray of shape coef.shape
        Gradient of previous iteration.

    loss_value : float
        Value of objective function = loss + penalty.

    loss_value_old : float
        Value of objective function of previous itertion.

    raw_prediction : ndarray of shape (n_samples,) or (n_samples, n_classes)

    converged : bool
        Indicator for convergence of the solver.

    iteration : int
        Number of Newton steps, i.e. calls to inner_solve

    use_fallback_lbfgs_solve : bool
        If set to True, the solver will resort to call LBFGS to finish the optimisation
        procedure in case of convergence issues.

    gradient_times_newton : float
        gradient @ coef_newton, set in inner_solve and used by line_search. If the
        Newton step is a descent direction, this is negative.
    """

    def __init__(self, *, coef, linear_loss=LinearModelLoss(base_loss=HalfSquaredError(), fit_intercept=True), l2_reg_strength=0.0, tol=0.0001, max_iter=100, n_threads=1, verbose=0):
        if False:
            while True:
                i = 10
        self.coef = coef
        self.linear_loss = linear_loss
        self.l2_reg_strength = l2_reg_strength
        self.tol = tol
        self.max_iter = max_iter
        self.n_threads = n_threads
        self.verbose = verbose

    def setup(self, X, y, sample_weight):
        if False:
            i = 10
            return i + 15
        'Precomputations\n\n        If None, initializes:\n            - self.coef\n        Sets:\n            - self.raw_prediction\n            - self.loss_value\n        '
        (_, _, self.raw_prediction) = self.linear_loss.weight_intercept_raw(self.coef, X)
        self.loss_value = self.linear_loss.loss(coef=self.coef, X=X, y=y, sample_weight=sample_weight, l2_reg_strength=self.l2_reg_strength, n_threads=self.n_threads, raw_prediction=self.raw_prediction)

    @abstractmethod
    def update_gradient_hessian(self, X, y, sample_weight):
        if False:
            i = 10
            return i + 15
        'Update gradient and Hessian.'

    @abstractmethod
    def inner_solve(self, X, y, sample_weight):
        if False:
            for i in range(10):
                print('nop')
        'Compute Newton step.\n\n        Sets:\n            - self.coef_newton\n            - self.gradient_times_newton\n        '

    def fallback_lbfgs_solve(self, X, y, sample_weight):
        if False:
            print('Hello World!')
        'Fallback solver in case of emergency.\n\n        If a solver detects convergence problems, it may fall back to this methods in\n        the hope to exit with success instead of raising an error.\n\n        Sets:\n            - self.coef\n            - self.converged\n        '
        opt_res = scipy.optimize.minimize(self.linear_loss.loss_gradient, self.coef, method='L-BFGS-B', jac=True, options={'maxiter': self.max_iter, 'maxls': 50, 'iprint': self.verbose - 1, 'gtol': self.tol, 'ftol': 64 * np.finfo(np.float64).eps}, args=(X, y, sample_weight, self.l2_reg_strength, self.n_threads))
        self.n_iter_ = _check_optimize_result('lbfgs', opt_res)
        self.coef = opt_res.x
        self.converged = opt_res.status == 0

    def line_search(self, X, y, sample_weight):
        if False:
            for i in range(10):
                print('nop')
        'Backtracking line search.\n\n        Sets:\n            - self.coef_old\n            - self.coef\n            - self.loss_value_old\n            - self.loss_value\n            - self.gradient_old\n            - self.gradient\n            - self.raw_prediction\n        '
        (beta, sigma) = (0.5, 0.00048828125)
        eps = 16 * np.finfo(self.loss_value.dtype).eps
        t = 1
        armijo_term = sigma * self.gradient_times_newton
        (_, _, raw_prediction_newton) = self.linear_loss.weight_intercept_raw(self.coef_newton, X)
        self.coef_old = self.coef
        self.loss_value_old = self.loss_value
        self.gradient_old = self.gradient
        sum_abs_grad_old = -1
        is_verbose = self.verbose >= 2
        if is_verbose:
            print('  Backtracking Line Search')
            print(f'    eps=10 * finfo.eps={eps}')
        for i in range(21):
            self.coef = self.coef_old + t * self.coef_newton
            raw = self.raw_prediction + t * raw_prediction_newton
            (self.loss_value, self.gradient) = self.linear_loss.loss_gradient(coef=self.coef, X=X, y=y, sample_weight=sample_weight, l2_reg_strength=self.l2_reg_strength, n_threads=self.n_threads, raw_prediction=raw)
            loss_improvement = self.loss_value - self.loss_value_old
            check = loss_improvement <= t * armijo_term
            if is_verbose:
                print(f'    line search iteration={i + 1}, step size={t}\n      check loss improvement <= armijo term: {loss_improvement} <= {t * armijo_term} {check}')
            if check:
                break
            tiny_loss = np.abs(self.loss_value_old * eps)
            check = np.abs(loss_improvement) <= tiny_loss
            if is_verbose:
                print(f'      check loss |improvement| <= eps * |loss_old|: {np.abs(loss_improvement)} <= {tiny_loss} {check}')
            if check:
                if sum_abs_grad_old < 0:
                    sum_abs_grad_old = scipy.linalg.norm(self.gradient_old, ord=1)
                sum_abs_grad = scipy.linalg.norm(self.gradient, ord=1)
                check = sum_abs_grad < sum_abs_grad_old
                if is_verbose:
                    print(f'      check sum(|gradient|) < sum(|gradient_old|): {sum_abs_grad} < {sum_abs_grad_old} {check}')
                if check:
                    break
            t *= beta
        else:
            warnings.warn(f'Line search of Newton solver {self.__class__.__name__} at iteration #{self.iteration} did no converge after 21 line search refinement iterations. It will now resort to lbfgs instead.', ConvergenceWarning)
            if self.verbose:
                print('  Line search did not converge and resorts to lbfgs instead.')
            self.use_fallback_lbfgs_solve = True
            return
        self.raw_prediction = raw

    def check_convergence(self, X, y, sample_weight):
        if False:
            print('Hello World!')
        'Check for convergence.\n\n        Sets self.converged.\n        '
        if self.verbose:
            print('  Check Convergence')
        check = np.max(np.abs(self.gradient))
        if self.verbose:
            print(f'    1. max |gradient| {check} <= {self.tol}')
        if check > self.tol:
            return
        d2 = self.coef_newton @ self.hessian @ self.coef_newton
        if self.verbose:
            print(f'    2. Newton decrement {0.5 * d2} <= {self.tol}')
        if 0.5 * d2 > self.tol:
            return
        if self.verbose:
            loss_value = self.linear_loss.loss(coef=self.coef, X=X, y=y, sample_weight=sample_weight, l2_reg_strength=self.l2_reg_strength, n_threads=self.n_threads)
            print(f'  Solver did converge at loss = {loss_value}.')
        self.converged = True

    def finalize(self, X, y, sample_weight):
        if False:
            while True:
                i = 10
        'Finalize the solvers results.\n\n        Some solvers may need this, others not.\n        '
        pass

    def solve(self, X, y, sample_weight):
        if False:
            for i in range(10):
                print('nop')
        'Solve the optimization problem.\n\n        This is the main routine.\n\n        Order of calls:\n            self.setup()\n            while iteration:\n                self.update_gradient_hessian()\n                self.inner_solve()\n                self.line_search()\n                self.check_convergence()\n            self.finalize()\n\n        Returns\n        -------\n        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)\n            Solution of the optimization problem.\n        '
        self.setup(X=X, y=y, sample_weight=sample_weight)
        self.iteration = 1
        self.converged = False
        self.use_fallback_lbfgs_solve = False
        while self.iteration <= self.max_iter and (not self.converged):
            if self.verbose:
                print(f'Newton iter={self.iteration}')
            self.use_fallback_lbfgs_solve = False
            self.update_gradient_hessian(X=X, y=y, sample_weight=sample_weight)
            self.inner_solve(X=X, y=y, sample_weight=sample_weight)
            if self.use_fallback_lbfgs_solve:
                break
            self.line_search(X=X, y=y, sample_weight=sample_weight)
            if self.use_fallback_lbfgs_solve:
                break
            self.check_convergence(X=X, y=y, sample_weight=sample_weight)
            self.iteration += 1
        if not self.converged:
            if self.use_fallback_lbfgs_solve:
                self.fallback_lbfgs_solve(X=X, y=y, sample_weight=sample_weight)
            else:
                warnings.warn(f'Newton solver did not converge after {self.iteration - 1} iterations.', ConvergenceWarning)
        self.iteration -= 1
        self.finalize(X=X, y=y, sample_weight=sample_weight)
        return self.coef

class NewtonCholeskySolver(NewtonSolver):
    """Cholesky based Newton solver.

    Inner solver for finding the Newton step H w_newton = -g uses Cholesky based linear
    solver.
    """

    def setup(self, X, y, sample_weight):
        if False:
            print('Hello World!')
        super().setup(X=X, y=y, sample_weight=sample_weight)
        n_dof = X.shape[1]
        if self.linear_loss.fit_intercept:
            n_dof += 1
        self.gradient = np.empty_like(self.coef)
        self.hessian = np.empty_like(self.coef, shape=(n_dof, n_dof))

    def update_gradient_hessian(self, X, y, sample_weight):
        if False:
            print('Hello World!')
        (_, _, self.hessian_warning) = self.linear_loss.gradient_hessian(coef=self.coef, X=X, y=y, sample_weight=sample_weight, l2_reg_strength=self.l2_reg_strength, n_threads=self.n_threads, gradient_out=self.gradient, hessian_out=self.hessian, raw_prediction=self.raw_prediction)

    def inner_solve(self, X, y, sample_weight):
        if False:
            i = 10
            return i + 15
        if self.hessian_warning:
            warnings.warn(f'The inner solver of {self.__class__.__name__} detected a pointwise hessian with many negative values at iteration #{self.iteration}. It will now resort to lbfgs instead.', ConvergenceWarning)
            if self.verbose:
                print('  The inner solver detected a pointwise Hessian with many negative values and resorts to lbfgs instead.')
            self.use_fallback_lbfgs_solve = True
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', scipy.linalg.LinAlgWarning)
                self.coef_newton = scipy.linalg.solve(self.hessian, -self.gradient, check_finite=False, assume_a='sym')
                self.gradient_times_newton = self.gradient @ self.coef_newton
                if self.gradient_times_newton > 0:
                    if self.verbose:
                        print('  The inner solver found a Newton step that is not a descent direction and resorts to LBFGS steps instead.')
                    self.use_fallback_lbfgs_solve = True
                    return
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgWarning) as e:
            warnings.warn(f'The inner solver of {self.__class__.__name__} stumbled upon a singular or very ill-conditioned Hessian matrix at iteration #{self.iteration}. It will now resort to lbfgs instead.\nFurther options are to use another solver or to avoid such situation in the first place. Possible remedies are removing collinear features of X or increasing the penalization strengths.\nThe original Linear Algebra message was:\n' + str(e), scipy.linalg.LinAlgWarning)
            if self.verbose:
                print('  The inner solver stumbled upon an singular or ill-conditioned Hessian matrix and resorts to LBFGS instead.')
            self.use_fallback_lbfgs_solve = True
            return