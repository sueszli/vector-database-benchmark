import cvxpy.settings as s
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.solver import Solver

class ConstantSolver(Solver):
    """TODO(akshayka): Documentation."""
    MIP_CAPABLE = True

    def accepts(self, problem) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(problem.variables()) == 0

    def apply(self, problem):
        if False:
            i = 10
            return i + 15
        return (problem, [])

    def invert(self, solution, inverse_data):
        if False:
            return 10
        return solution

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'CONSTANT_SOLVER'

    def import_solver(self) -> None:
        if False:
            return 10
        return

    def is_installed(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            i = 10
            return i + 15
        return self.solve(data, warm_start, verbose, solver_opts)

    def solve(self, problem, warm_start: bool, verbose: bool, solver_opts):
        if False:
            while True:
                i = 10
        if all((c.value() for c in problem.constraints)):
            return Solution(s.OPTIMAL, problem.objective.value, {}, {}, {})
        else:
            return Solution(s.INFEASIBLE, None, {}, {}, {})