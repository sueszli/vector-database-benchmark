"""Dog-leg trust-region optimization."""
import numpy as np
import scipy.linalg
from ._trustregion import _minimize_trust_region, BaseQuadraticSubproblem
__all__ = []

def _minimize_dogleg(fun, x0, args=(), jac=None, hess=None, **trust_region_options):
    if False:
        for i in range(10):
            print('nop')
    '\n    Minimization of scalar function of one or more variables using\n    the dog-leg trust-region algorithm.\n\n    Options\n    -------\n    initial_trust_radius : float\n        Initial trust-region radius.\n    max_trust_radius : float\n        Maximum value of the trust-region radius. No steps that are longer\n        than this value will be proposed.\n    eta : float\n        Trust region related acceptance stringency for proposed steps.\n    gtol : float\n        Gradient norm must be less than `gtol` before successful\n        termination.\n\n    '
    if jac is None:
        raise ValueError('Jacobian is required for dogleg minimization')
    if not callable(hess):
        raise ValueError('Hessian is required for dogleg minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, subproblem=DoglegSubproblem, **trust_region_options)

class DoglegSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by the dogleg method"""

    def cauchy_point(self):
        if False:
            while True:
                i = 10
        '\n        The Cauchy point is minimal along the direction of steepest descent.\n        '
        if self._cauchy_point is None:
            g = self.jac
            Bg = self.hessp(g)
            self._cauchy_point = -(np.dot(g, g) / np.dot(g, Bg)) * g
        return self._cauchy_point

    def newton_point(self):
        if False:
            return 10
        '\n        The Newton point is a global minimum of the approximate function.\n        '
        if self._newton_point is None:
            g = self.jac
            B = self.hess
            cho_info = scipy.linalg.cho_factor(B)
            self._newton_point = -scipy.linalg.cho_solve(cho_info, g)
        return self._newton_point

    def solve(self, trust_radius):
        if False:
            print('Hello World!')
        '\n        Minimize a function using the dog-leg trust-region algorithm.\n\n        This algorithm requires function values and first and second derivatives.\n        It also performs a costly Hessian decomposition for most iterations,\n        and the Hessian is required to be positive definite.\n\n        Parameters\n        ----------\n        trust_radius : float\n            We are allowed to wander only this far away from the origin.\n\n        Returns\n        -------\n        p : ndarray\n            The proposed step.\n        hits_boundary : bool\n            True if the proposed step is on the boundary of the trust region.\n\n        Notes\n        -----\n        The Hessian is required to be positive definite.\n\n        References\n        ----------\n        .. [1] Jorge Nocedal and Stephen Wright,\n               Numerical Optimization, second edition,\n               Springer-Verlag, 2006, page 73.\n        '
        p_best = self.newton_point()
        if scipy.linalg.norm(p_best) < trust_radius:
            hits_boundary = False
            return (p_best, hits_boundary)
        p_u = self.cauchy_point()
        p_u_norm = scipy.linalg.norm(p_u)
        if p_u_norm >= trust_radius:
            p_boundary = p_u * (trust_radius / p_u_norm)
            hits_boundary = True
            return (p_boundary, hits_boundary)
        (_, tb) = self.get_boundaries_intersections(p_u, p_best - p_u, trust_radius)
        p_boundary = p_u + tb * (p_best - p_u)
        hits_boundary = True
        return (p_boundary, hits_boundary)