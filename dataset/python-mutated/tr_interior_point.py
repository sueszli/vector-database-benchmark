"""Trust-region interior point method.

References
----------
.. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
       "An interior point algorithm for large-scale nonlinear
       programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
.. [2] Byrd, Richard H., Guanghui Liu, and Jorge Nocedal.
       "On the local behavior of an interior point method for
       nonlinear programming." Numerical analysis 1997 (1997): 37-56.
.. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
       Second Edition (2006).
"""
import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
__all__ = ['tr_interior_point']

class BarrierSubproblem:
    """
    Barrier optimization problem:
        minimize fun(x) - barrier_parameter*sum(log(s))
        subject to: constr_eq(x)     = 0
                  constr_ineq(x) + s = 0
    """

    def __init__(self, x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac, barrier_parameter, tolerance, enforce_feasibility, global_stop_criteria, xtol, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0, jac_eq0):
        if False:
            for i in range(10):
                print('nop')
        self.n_vars = n_vars
        self.x0 = x0
        self.s0 = s0
        self.fun = fun
        self.grad = grad
        self.lagr_hess = lagr_hess
        self.constr = constr
        self.jac = jac
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.enforce_feasibility = enforce_feasibility
        self.global_stop_criteria = global_stop_criteria
        self.xtol = xtol
        self.fun0 = self._compute_function(fun0, constr_ineq0, s0)
        self.grad0 = self._compute_gradient(grad0)
        self.constr0 = self._compute_constr(constr_ineq0, constr_eq0, s0)
        self.jac0 = self._compute_jacobian(jac_eq0, jac_ineq0, s0)
        self.terminate = False

    def update(self, barrier_parameter, tolerance):
        if False:
            for i in range(10):
                print('nop')
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance

    def get_slack(self, z):
        if False:
            i = 10
            return i + 15
        return z[self.n_vars:self.n_vars + self.n_ineq]

    def get_variables(self, z):
        if False:
            print('Hello World!')
        return z[:self.n_vars]

    def function_and_constraints(self, z):
        if False:
            while True:
                i = 10
        'Returns barrier function and constraints at given point.\n\n        For z = [x, s], returns barrier function:\n            function(z) = fun(x) - barrier_parameter*sum(log(s))\n        and barrier constraints:\n            constraints(z) = [   constr_eq(x)     ]\n                             [ constr_ineq(x) + s ]\n\n        '
        x = self.get_variables(z)
        s = self.get_slack(z)
        f = self.fun(x)
        (c_eq, c_ineq) = self.constr(x)
        return (self._compute_function(f, c_ineq, s), self._compute_constr(c_ineq, c_eq, s))

    def _compute_function(self, f, c_ineq, s):
        if False:
            while True:
                i = 10
        s[self.enforce_feasibility] = -c_ineq[self.enforce_feasibility]
        log_s = [np.log(s_i) if s_i > 0 else -np.inf for s_i in s]
        return f - self.barrier_parameter * np.sum(log_s)

    def _compute_constr(self, c_ineq, c_eq, s):
        if False:
            return 10
        return np.hstack((c_eq, c_ineq + s))

    def scaling(self, z):
        if False:
            for i in range(10):
                print('nop')
        'Returns scaling vector.\n        Given by:\n            scaling = [ones(n_vars), s]\n        '
        s = self.get_slack(z)
        diag_elements = np.hstack((np.ones(self.n_vars), s))

        def matvec(vec):
            if False:
                print('Hello World!')
            return diag_elements * vec
        return LinearOperator((self.n_vars + self.n_ineq, self.n_vars + self.n_ineq), matvec)

    def gradient_and_jacobian(self, z):
        if False:
            print('Hello World!')
        'Returns scaled gradient.\n\n        Return scaled gradient:\n            gradient = [             grad(x)             ]\n                       [ -barrier_parameter*ones(n_ineq) ]\n        and scaled Jacobian matrix:\n            jacobian = [  jac_eq(x)  0  ]\n                       [ jac_ineq(x) S  ]\n        Both of them scaled by the previously defined scaling factor.\n        '
        x = self.get_variables(z)
        s = self.get_slack(z)
        g = self.grad(x)
        (J_eq, J_ineq) = self.jac(x)
        return (self._compute_gradient(g), self._compute_jacobian(J_eq, J_ineq, s))

    def _compute_gradient(self, g):
        if False:
            while True:
                i = 10
        return np.hstack((g, -self.barrier_parameter * np.ones(self.n_ineq)))

    def _compute_jacobian(self, J_eq, J_ineq, s):
        if False:
            i = 10
            return i + 15
        if self.n_ineq == 0:
            return J_eq
        elif sps.issparse(J_eq) or sps.issparse(J_ineq):
            J_eq = sps.csr_matrix(J_eq)
            J_ineq = sps.csr_matrix(J_ineq)
            return self._assemble_sparse_jacobian(J_eq, J_ineq, s)
        else:
            S = np.diag(s)
            zeros = np.zeros((self.n_eq, self.n_ineq))
            if sps.issparse(J_ineq):
                J_ineq = J_ineq.toarray()
            if sps.issparse(J_eq):
                J_eq = J_eq.toarray()
            return np.block([[J_eq, zeros], [J_ineq, S]])

    def _assemble_sparse_jacobian(self, J_eq, J_ineq, s):
        if False:
            for i in range(10):
                print('nop')
        'Assemble sparse Jacobian given its components.\n\n        Given ``J_eq``, ``J_ineq`` and ``s`` returns:\n            jacobian = [ J_eq,     0     ]\n                       [ J_ineq, diag(s) ]\n\n        It is equivalent to:\n            sps.bmat([[ J_eq,   None    ],\n                      [ J_ineq, diag(s) ]], "csr")\n        but significantly more efficient for this\n        given structure.\n        '
        (n_vars, n_ineq, n_eq) = (self.n_vars, self.n_ineq, self.n_eq)
        J_aux = sps.vstack([J_eq, J_ineq], 'csr')
        (indptr, indices, data) = (J_aux.indptr, J_aux.indices, J_aux.data)
        new_indptr = indptr + np.hstack((np.zeros(n_eq, dtype=int), np.arange(n_ineq + 1, dtype=int)))
        size = indices.size + n_ineq
        new_indices = np.empty(size)
        new_data = np.empty(size)
        mask = np.full(size, False, bool)
        mask[new_indptr[-n_ineq:] - 1] = True
        new_indices[mask] = n_vars + np.arange(n_ineq)
        new_indices[~mask] = indices
        new_data[mask] = s
        new_data[~mask] = data
        J = sps.csr_matrix((new_data, new_indices, new_indptr), (n_eq + n_ineq, n_vars + n_ineq))
        return J

    def lagrangian_hessian_x(self, z, v):
        if False:
            while True:
                i = 10
        'Returns Lagrangian Hessian (in relation to `x`) -> Hx'
        x = self.get_variables(z)
        v_eq = v[:self.n_eq]
        v_ineq = v[self.n_eq:self.n_eq + self.n_ineq]
        lagr_hess = self.lagr_hess
        return lagr_hess(x, v_eq, v_ineq)

    def lagrangian_hessian_s(self, z, v):
        if False:
            for i in range(10):
                print('nop')
        'Returns scaled Lagrangian Hessian (in relation to`s`) -> S Hs S'
        s = self.get_slack(z)
        primal = self.barrier_parameter
        primal_dual = v[-self.n_ineq:] * s
        return np.where(v[-self.n_ineq:] > 0, primal_dual, primal)

    def lagrangian_hessian(self, z, v):
        if False:
            return 10
        'Returns scaled Lagrangian Hessian'
        Hx = self.lagrangian_hessian_x(z, v)
        if self.n_ineq > 0:
            S_Hs_S = self.lagrangian_hessian_s(z, v)

        def matvec(vec):
            if False:
                print('Hello World!')
            vec_x = self.get_variables(vec)
            vec_s = self.get_slack(vec)
            if self.n_ineq > 0:
                return np.hstack((Hx.dot(vec_x), S_Hs_S * vec_s))
            else:
                return Hx.dot(vec_x)
        return LinearOperator((self.n_vars + self.n_ineq, self.n_vars + self.n_ineq), matvec)

    def stop_criteria(self, state, z, last_iteration_failed, optimality, constr_violation, trust_radius, penalty, cg_info):
        if False:
            print('Hello World!')
        'Stop criteria to the barrier problem.\n        The criteria here proposed is similar to formula (2.3)\n        from [1]_, p.879.\n        '
        x = self.get_variables(z)
        if self.global_stop_criteria(state, x, last_iteration_failed, trust_radius, penalty, cg_info, self.barrier_parameter, self.tolerance):
            self.terminate = True
            return True
        else:
            g_cond = optimality < self.tolerance and constr_violation < self.tolerance
            x_cond = trust_radius < self.xtol
            return g_cond or x_cond

def tr_interior_point(fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac, x0, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0, jac_eq0, stop_criteria, enforce_feasibility, xtol, state, initial_barrier_parameter, initial_tolerance, initial_penalty, initial_trust_radius, factorization_method):
    if False:
        print('Hello World!')
    'Trust-region interior points method.\n\n    Solve problem:\n        minimize fun(x)\n        subject to: constr_ineq(x) <= 0\n                    constr_eq(x) = 0\n    using trust-region interior point method described in [1]_.\n    '
    BOUNDARY_PARAMETER = 0.995
    BARRIER_DECAY_RATIO = 0.2
    TRUST_ENLARGEMENT = 5
    if enforce_feasibility is None:
        enforce_feasibility = np.zeros(n_ineq, bool)
    barrier_parameter = initial_barrier_parameter
    tolerance = initial_tolerance
    trust_radius = initial_trust_radius
    s0 = np.maximum(-1.5 * constr_ineq0, np.ones(n_ineq))
    subprob = BarrierSubproblem(x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac, barrier_parameter, tolerance, enforce_feasibility, stop_criteria, xtol, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0, jac_eq0)
    z = np.hstack((x0, s0))
    (fun0_subprob, constr0_subprob) = (subprob.fun0, subprob.constr0)
    (grad0_subprob, jac0_subprob) = (subprob.grad0, subprob.jac0)
    trust_lb = np.hstack((np.full(subprob.n_vars, -np.inf), np.full(subprob.n_ineq, -BOUNDARY_PARAMETER)))
    trust_ub = np.full(subprob.n_vars + subprob.n_ineq, np.inf)
    while True:
        (z, state) = equality_constrained_sqp(subprob.function_and_constraints, subprob.gradient_and_jacobian, subprob.lagrangian_hessian, z, fun0_subprob, grad0_subprob, constr0_subprob, jac0_subprob, subprob.stop_criteria, state, initial_penalty, trust_radius, factorization_method, trust_lb, trust_ub, subprob.scaling)
        if subprob.terminate:
            break
        trust_radius = max(initial_trust_radius, TRUST_ENLARGEMENT * state.tr_radius)
        barrier_parameter *= BARRIER_DECAY_RATIO
        tolerance *= BARRIER_DECAY_RATIO
        subprob.update(barrier_parameter, tolerance)
        (fun0_subprob, constr0_subprob) = subprob.function_and_constraints(z)
        (grad0_subprob, jac0_subprob) = subprob.gradient_and_jacobian(z)
    x = subprob.get_variables(z)
    return (x, state)