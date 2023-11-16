"""
Copyright 2013 Steven Diamond

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
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers import GLPK
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

class GLPK_MI(GLPK):
    """An interface for the GLPK MI solver.
    """
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    def name(self):
        if False:
            i = 10
            return i + 15
        'The name of the solver.\n        '
        return s.GLPK_MI

    def apply(self, problem):
        if False:
            for i in range(10):
                print('nop')
        'Returns a new problem and data for inverting the new solution.\n\n        Returns\n        -------\n        tuple\n            (dict of arguments needed for the solver, inverse data)\n        '
        (data, inv_data) = super(GLPK_MI, self).apply(problem)
        var = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in var.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in var.integer_idx]
        return (data, inv_data)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            for i in range(10):
                print('nop')
        import cvxopt
        import cvxopt.solvers
        old_options = cvxopt.glpk.options.copy()
        if verbose:
            cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_ON'
        else:
            cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
        data = self._prepare_cvxopt_matrices(data)
        if 'max_iters' in solver_opts:
            solver_opts['maxiters'] = solver_opts['max_iters']
        for (key, value) in solver_opts.items():
            cvxopt.glpk.options[key] = value
        try:
            results_tup = cvxopt.glpk.ilp(data[s.C], data[s.G], data[s.H], data[s.A], data[s.B], set((int(i) for i in data[s.INT_IDX])), set((int(i) for i in data[s.BOOL_IDX])))
            results_dict = {}
            results_dict['status'] = results_tup[0]
            results_dict['x'] = results_tup[1]
        except ValueError:
            results_dict = {'status': 'unknown'}
        self._restore_solver_options(old_options)
        solution = {}
        status = self.STATUS_MAP[results_dict['status']]
        solution[s.STATUS] = status
        if solution[s.STATUS] in s.SOLUTION_PRESENT:
            solution[s.PRIMAL] = intf.cvxopt2dense(results_dict['x'])
            primal_val = (data[s.C].T * results_dict['x'])[0]
            solution[s.VALUE] = primal_val
        return solution

    def invert(self, solution, inverse_data):
        if False:
            for i in range(10):
                print('nop')
        'Returns the solution to the original problem given the inverse_data.\n        '
        status = solution['status']
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            return Solution(status, opt_val, primal_vars, None, {})
        else:
            return failure_solution(status)

    @staticmethod
    def _restore_solver_options(old_options) -> None:
        if False:
            while True:
                i = 10
        import cvxopt.glpk
        for (key, value) in list(cvxopt.glpk.options.items()):
            if key in old_options:
                cvxopt.glpk.options[key] = old_options[key]
            else:
                del cvxopt.glpk.options[key]