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
import cvxpy.settings as s
from cvxpy.constraints import PSD, NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver import Solver

def dims_to_solver_dict(cone_dims):
    if False:
        while True:
            i = 10
    cones = {'f': cone_dims.zero, 'l': cone_dims.nonneg, 's': cone_dims.psd}
    return cones

class SDPA(ConicSolver):
    """An interface for the SDPA solver.
    """
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [PSD]
    STATUS_MAP = {'pdOPT': s.OPTIMAL, 'noINFO': s.SOLVER_ERROR, 'pFEAS': s.OPTIMAL_INACCURATE, 'dFEAS': s.OPTIMAL_INACCURATE, 'pdFEAS': s.OPTIMAL_INACCURATE, 'pdINF': s.INFEASIBLE, 'pFEAS_dINF': s.UNBOUNDED, 'pINF_dFEAS': s.INFEASIBLE, 'pUNBD': s.UNBOUNDED, 'dUNBD': s.INFEASIBLE}

    def name(self):
        if False:
            print('Hello World!')
        'The name of the solver.\n        '
        return s.SDPA

    def import_solver(self) -> None:
        if False:
            i = 10
            return i + 15
        'Imports the solver.\n        '
        import sdpap

    def accepts(self, problem) -> bool:
        if False:
            return 10
        'Can SDPA solve the problem?\n        '
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        if False:
            while True:
                i = 10
        'Returns a new problem and data for inverting the new solution.\n\n        Returns\n        -------\n        tuple\n            (dict of arguments needed for the solver, inverse data)\n        '
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}
        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims
        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[PSD]
        (c, d, A, b) = problem.apply_parameters()
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = A
        if data[s.A].shape[0] == 0:
            data[s.A] = None
        data[s.B] = b.flatten()
        if data[s.B].shape[0] == 0:
            data[s.B] = None
        return (data, inv_data)

    def invert(self, solution, inverse_data):
        if False:
            for i in range(10):
                print('nop')
        'Returns the solution to the original problem given the inverse_data.\n        '
        status = solution['status']
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value']
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            eq_dual = utilities.get_dual_values(solution['eq_dual'], utilities.extract_dual_value, inverse_data[Solver.EQ_CONSTR])
            leq_dual = utilities.get_dual_values(solution['ineq_dual'], utilities.extract_dual_value, inverse_data[Solver.NEQ_CONSTR])
            eq_dual.update(leq_dual)
            dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            return 10
        '\n        CVXPY represents cone programs as\n            (P) min_x { c^T x : A x + b \\in K } + d\n\n        SDPA Python takes a conic program in CLP format:\n            (P) min_x { c^T x : A x - b \\in J, x \\in K }\n\n        CVXPY (P) -> CLP (P), by\n            - flipping sign of b\n            - setting J of CLP (P) to K of CVXPY (P)\n            - setting K of CLP (P) to a free cone\n\n        CLP format is a generalization of the SeDuMi format. Both formats are explained at\n        https://sdpa-python.github.io/docs/formats/\n\n        Internally, SDPA Python will reduce CLP form to SeDuMi dual form using `clp_toLMI`.\n        In SeDuMi format, the dual is in LMI form. In SDPA format, the primal is in LMI form.\n        The backend (i.e. `libsdpa.a` or `libsdpa_gmp.a`) uses the SDPA format.\n\n        For details on the reverse relationship between SDPA and SeDuMi formats, please see\n        https://sdpa-python.github.io/docs/formats/sdpa_sedumi.html\n        '
        import sdpap
        from scipy import matrix
        data[s.DIMS] = dims_to_solver_dict(data[s.DIMS])
        (A, b, c, dims) = (data[s.A], data[s.B], data[s.C], data[s.DIMS])
        K = sdpap.SymCone(f=c.shape[0])
        J = sdpap.SymCone(f=dims['f'], l=dims['l'], s=tuple(dims['s']))
        if 'print' not in solver_opts:
            solver_opts['print'] = 'display' if verbose else 'no'
        (x, y, sdpapinfo, timeinfo, sdpainfo) = sdpap.solve(A, -matrix(b), matrix(c), K, J, solver_opts)
        solution = {}
        solution[s.STATUS] = self.STATUS_MAP[sdpapinfo['phasevalue']]
        if solution[s.STATUS] in s.SOLUTION_PRESENT:
            x = x.toarray()
            y = y.toarray()
            solution[s.VALUE] = sdpapinfo['primalObj']
            solution[s.PRIMAL] = x
            solution[s.EQ_DUAL] = y[:dims['f']]
            solution[s.INEQ_DUAL] = y[dims['f']:]
        return solution