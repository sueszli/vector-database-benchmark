"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import time
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers import scs_conif
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version

class DIFFCP(scs_conif.SCS):
    """An interface for the DIFFCP solver, a differentiable wrapper of SCS and ECOS.
    """
    STATUS_MAP = {'Solved': s.OPTIMAL, 'Solved/Inaccurate': s.OPTIMAL_INACCURATE, 'Unbounded': s.UNBOUNDED, 'Unbounded/Inaccurate': s.UNBOUNDED_INACCURATE, 'Infeasible': s.INFEASIBLE, 'Infeasible/Inaccurate': s.INFEASIBLE_INACCURATE, 'Failure': s.SOLVER_ERROR, 'Indeterminate': s.SOLVER_ERROR, 'Interrupted': s.SOLVER_ERROR}

    def name(self):
        if False:
            i = 10
            return i + 15
        'The name of the solver.\n        '
        return s.DIFFCP

    def import_solver(self) -> None:
        if False:
            return 10
        'Imports the solver.\n        '
        import diffcp
        patch_version = int(diffcp.__version__.split('.')[2])
        if patch_version < 15:
            raise ImportError('diffcp >= 1.0.15 is required')

    def supports_quad_obj(self) -> bool:
        if False:
            print('Hello World!')
        'Does not support a quadratic objective.\n        '
        return False

    def apply(self, problem):
        if False:
            while True:
                i = 10
        (problem, data, inv_data) = self._prepare_data_and_inv_data(problem)
        (c, d, A, b) = problem.apply_parameters(keep_zeros=True)
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A
        data[s.B] = b
        return (data, inv_data)

    def invert(self, solution, inverse_data):
        if False:
            for i in range(10):
                print('nop')
        'Returns the solution to the original problem given the inverse_data.\n        '
        attr = {}
        if solution['solve_method'] == s.SCS:
            import scs
            if Version(scs.__version__) < Version('3.0.0'):
                status = scs_conif.SCS.STATUS_MAP[solution['info']['statusVal']]
                attr[s.SOLVE_TIME] = solution['info']['solveTime']
                attr[s.SETUP_TIME] = solution['info']['setupTime']
            else:
                status = scs_conif.SCS.STATUS_MAP[solution['info']['status_val']]
                attr[s.SOLVE_TIME] = solution['info']['solve_time']
                attr[s.SETUP_TIME] = solution['info']['setup_time']
        elif solution['solve_method'] == s.ECOS:
            status = self.STATUS_MAP[solution['info']['status']]
            attr[s.SOLVE_TIME] = solution['info']['solveTime']
            attr[s.SETUP_TIME] = solution['info']['setupTime']
        attr[s.NUM_ITERS] = solution['info']['iter']
        attr[s.EXTRA_STATS] = solution
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['info']['pobj']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[DIFFCP.VAR_ID]: solution['x']}
            eq_dual_vars = utilities.get_dual_values(solution['y'][:inverse_data[ConicSolver.DIMS].zero], self.extract_dual_value, inverse_data[DIFFCP.EQ_CONSTR])
            ineq_dual_vars = utilities.get_dual_values(solution['y'][inverse_data[ConicSolver.DIMS].zero:], self.extract_dual_value, inverse_data[DIFFCP.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            while True:
                i = 10
        'Returns the result of the call to the solver.\n\n        Parameters\n        ----------\n        data : dict\n            Data generated via an apply call.\n        warm_start : Bool\n            Whether to warm_start diffcp.\n        verbose : Bool\n            Control the verbosity.\n        solver_opts : dict\n            SCS-specific solver options.\n\n        Returns\n        -------\n        The result returned by a call to scs.solve().\n        '
        import diffcp
        A = data[s.A]
        b = data[s.B]
        c = data[s.C]
        cones = scs_conif.dims_to_solver_dict(data[ConicSolver.DIMS])
        solver_opts['solve_method'] = solver_opts.get('solve_method', s.SCS)
        warm_start_tuple = None
        if solver_opts['solve_method'] == s.SCS:
            import scs
            if Version(scs.__version__) < Version('3.0.0'):
                solver_opts['eps'] = solver_opts.get('eps', 0.0001)
            else:
                solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-05)
                solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-05)
            if warm_start and solver_cache is not None and (self.name() in solver_cache):
                warm_start_tuple = (solver_cache[self.name()]['x'], solver_cache[self.name()]['y'], solver_cache[self.name()]['s'])
        start = time.time()
        results = diffcp.solve_and_derivative_internal(A, b, c, cones, verbose=verbose, warm_start=warm_start_tuple, raise_on_error=False, **solver_opts)
        end = time.time()
        results['TOT_TIME'] = end - start
        results['solve_method'] = solver_opts['solve_method']
        if solver_cache is not None:
            solver_cache[self.name()] = results
        return results