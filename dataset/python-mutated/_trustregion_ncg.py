"""Newton-CG trust-region optimization."""
import math
import numpy as np
import scipy.linalg
from ._trustregion import _minimize_trust_region, BaseQuadraticSubproblem
__all__ = []

def _minimize_trust_ncg(fun, x0, args=(), jac=None, hess=None, hessp=None, **trust_region_options):
    if False:
        while True:
            i = 10
    '\n    Minimization of scalar function of one or more variables using\n    the Newton conjugate gradient trust-region algorithm.\n\n    Options\n    -------\n    initial_trust_radius : float\n        Initial trust-region radius.\n    max_trust_radius : float\n        Maximum value of the trust-region radius. No steps that are longer\n        than this value will be proposed.\n    eta : float\n        Trust region related acceptance stringency for proposed steps.\n    gtol : float\n        Gradient norm must be less than `gtol` before successful\n        termination.\n\n    '
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG trust-region minimization')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product is required for Newton-CG trust-region minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp, subproblem=CGSteihaugSubproblem, **trust_region_options)

class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""

    def solve(self, trust_radius):
        if False:
            i = 10
            return i + 15
        '\n        Solve the subproblem using a conjugate gradient method.\n\n        Parameters\n        ----------\n        trust_radius : float\n            We are allowed to wander only this far away from the origin.\n\n        Returns\n        -------\n        p : ndarray\n            The proposed step.\n        hits_boundary : bool\n            True if the proposed step is on the boundary of the trust region.\n\n        Notes\n        -----\n        This is algorithm (7.2) of Nocedal and Wright 2nd edition.\n        Only the function that computes the Hessian-vector product is required.\n        The Hessian itself is not required, and the Hessian does\n        not need to be positive semidefinite.\n        '
        p_origin = np.zeros_like(self.jac)
        tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag
        if self.jac_mag < tolerance:
            hits_boundary = False
            return (p_origin, hits_boundary)
        z = p_origin
        r = self.jac
        d = -r
        while True:
            Bd = self.hessp(d)
            dBd = np.dot(d, Bd)
            if dBd <= 0:
                (ta, tb) = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return (p_boundary, hits_boundary)
            r_squared = np.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if scipy.linalg.norm(z_next) >= trust_radius:
                (ta, tb) = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return (p_boundary, hits_boundary)
            r_next = r + alpha * Bd
            r_next_squared = np.dot(r_next, r_next)
            if math.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return (z_next, hits_boundary)
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d
            z = z_next
            r = r_next
            d = d_next