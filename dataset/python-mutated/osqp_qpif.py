import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver

class OSQP(QpSolver):
    """QP interface for the OSQP solver"""
    STATUS_MAP = {1: s.OPTIMAL, 2: s.OPTIMAL_INACCURATE, -2: s.SOLVER_ERROR, -3: s.INFEASIBLE, 3: s.INFEASIBLE_INACCURATE, -4: s.UNBOUNDED, 4: s.UNBOUNDED_INACCURATE, -6: s.USER_LIMIT, -5: s.SOLVER_ERROR, -10: s.SOLVER_ERROR}

    def name(self):
        if False:
            return 10
        return s.OSQP

    def import_solver(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        import osqp
        osqp

    def invert(self, solution, inverse_data):
        if False:
            while True:
                i = 10
        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = solution
        status = self.STATUS_MAP.get(solution.info.status_val, s.SOLVER_ERROR)
        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.obj_val + inverse_data[s.OFFSET]
            primal_vars = {OSQP.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))}
            dual_vars = {OSQP.DUAL_VAR_ID: solution.y}
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            for i in range(10):
                print('nop')
        import osqp
        P = data[s.P]
        q = data[s.Q]
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        data['Ax'] = A
        uA = np.concatenate((data[s.B], data[s.G]))
        data['u'] = uA
        lA = np.concatenate([data[s.B], -np.inf * np.ones(data[s.G].shape)])
        data['l'] = lA
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-05)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-05)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)
        if warm_start and solver_cache is not None and (self.name() in solver_cache):
            (solver, old_data, results) = solver_cache[self.name()]
            new_args = {}
            for key in ['q', 'l', 'u']:
                if any(data[key] != old_data[key]):
                    new_args[key] = data[key]
            factorizing = False
            if P.data.shape != old_data[s.P].data.shape or any(P.data != old_data[s.P].data):
                P_triu = sp.triu(P).tocsc()
                new_args['Px'] = P_triu.data
                factorizing = True
            if A.data.shape != old_data['Ax'].data.shape or any(A.data != old_data['Ax'].data):
                new_args['Ax'] = A.data
                factorizing = True
            if new_args:
                solver.update(**new_args)
            status = self.STATUS_MAP.get(results.info.status_val, s.SOLVER_ERROR)
            if status == s.OPTIMAL:
                solver.warm_start(results.x, results.y)
            solver_opts['polish'] = solver_opts.get('polish', factorizing)
            solver.update_settings(verbose=verbose, **solver_opts)
        else:
            solver_opts['polish'] = solver_opts.get('polish', True)
            solver = osqp.OSQP()
            try:
                solver.setup(P, q, A, lA, uA, verbose=verbose, **solver_opts)
            except ValueError as e:
                raise SolverError(e)
        results = solver.solve()
        if solver_cache is not None:
            solver_cache[self.name()] = (solver, data, results)
        return results