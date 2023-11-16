"""
This file is the CVXPY QP extension of the Cardinal Optimizer
"""
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver

class COPT(QpSolver):
    """
    QP interface for the COPT solver
    """
    MIP_CAPABLE = True
    STATUS_MAP = {1: s.OPTIMAL, 2: s.INFEASIBLE, 3: s.UNBOUNDED, 4: s.INF_OR_UNB, 5: s.SOLVER_ERROR, 6: s.USER_LIMIT, 7: s.OPTIMAL_INACCURATE, 8: s.USER_LIMIT, 9: s.SOLVER_ERROR, 10: s.USER_LIMIT}

    def name(self):
        if False:
            return 10
        '\n        The name of solver.\n        '
        return 'COPT'

    def import_solver(self):
        if False:
            return 10
        '\n        Imports the solver.\n        '
        import coptpy

    def invert(self, solution, inverse_data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the solution to the original problem given the inverse_data.\n        '
        status = solution[s.STATUS]
        attr = {s.SOLVE_TIME: solution[s.SOLVE_TIME], s.NUM_ITERS: solution[s.NUM_ITERS], s.EXTRA_STATS: solution['model']}
        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[COPT.VAR_ID]: solution[s.PRIMAL]}
            if not inverse_data[COPT.IS_MIP]:
                dual_vars = {COPT.DUAL_VAR_ID: solution['y']}
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            print('Hello World!')
        '\n        Returns the result of the call to the solver.\n\n        Parameters\n        ----------\n        data : dict\n            Data used by the solver.\n        warm_start : bool\n            Not used.\n        verbose : bool\n            Should the solver print output?\n        solver_opts : dict\n            Additional arguments for the solver.\n        solver_cache: None\n            None\n\n        Returns\n        -------\n        tuple\n            (status, optimal value, primal, equality dual, inequality dual)\n        '
        import coptpy as copt
        envconfig = copt.EnvrConfig()
        if not verbose:
            envconfig.set('nobanner', '1')
        env = copt.Envr(envconfig)
        model = env.createModel()
        model.setParam(copt.COPT.Param.Logging, verbose)
        P = data[s.P]
        q = data[s.Q]
        A = data[s.A]
        b = data[s.B]
        F = data[s.F]
        g = data[s.G]
        n = data['n_var']
        if A.shape[0] > 0 and F.shape[0] == 0:
            Amat = A
            lhs = b
            rhs = b
        elif A.shape[0] == 0 and F.shape[0] > 0:
            Amat = F
            lhs = np.full(F.shape[0], -copt.COPT.INFINITY)
            rhs = g
        elif A.shape[0] > 0 and F.shape[0] > 0:
            Amat = sp.vstack([A, F])
            Amat = Amat.tocsc()
            lhs = np.hstack((b, np.full(F.shape[0], -copt.COPT.INFINITY)))
            rhs = np.hstack((b, g))
        else:
            Amat = sp.vstack([A, F])
            Amat = Amat.tocsc()
            lhs = None
            rhs = None
        lb = np.full(n, -copt.COPT.INFINITY)
        ub = np.full(n, +copt.COPT.INFINITY)
        vtype = None
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            vtype = np.array([copt.COPT.CONTINUOUS] * n)
            if data[s.BOOL_IDX]:
                vtype[data[s.BOOL_IDX]] = copt.COPT.BINARY
                lb[data[s.BOOL_IDX]] = 0
                ub[data[s.BOOL_IDX]] = 1
            if data[s.INT_IDX]:
                vtype[data[s.INT_IDX]] = copt.COPT.INTEGER
        model.loadMatrix(q, Amat, lhs, rhs, lb, ub, vtype)
        if P.count_nonzero():
            P = P.tocoo()
            model.loadQ(0.5 * P)
        for (key, value) in solver_opts.items():
            model.setParam(key, value)
        solution = {}
        try:
            model.solve()
            if model.status == copt.COPT.INF_OR_UNB and solver_opts.get('reoptimize', True):
                model.setParam(copt.COPT.Param.Presolve, 0)
                model.solve()
            if model.hasmipsol:
                solution[s.VALUE] = model.objval
                solution[s.PRIMAL] = np.array(model.getValues())
            elif model.haslpsol:
                solution[s.VALUE] = model.objval
                solution[s.PRIMAL] = np.array(model.getValues())
                solution['y'] = -np.array(model.getDuals())
        except Exception:
            pass
        solution[s.SOLVE_TIME] = model.solvingtime
        solution[s.NUM_ITERS] = model.barrieriter + model.simplexiter
        solution[s.STATUS] = self.STATUS_MAP.get(model.status, s.SOLVER_ERROR)
        if solution[s.STATUS] == s.USER_LIMIT and model.hasmipsol:
            solution[s.STATUS] = s.OPTIMAL_INACCURATE
        if solution[s.STATUS] == s.USER_LIMIT and (not model.hasmipsol):
            solution[s.STATUS] = s.INFEASIBLE_INACCURATE
        solution['model'] = model
        return solution