"""
Copyright 2017 Robin Verschueren

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
from cvxpy.expressions import cvxtypes
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.reductions.reduction import Reduction

class FlipObjective(Reduction):
    """Flip a minimization objective to a maximization and vice versa.
     """

    def accepts(self, problem) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def apply(self, problem):
        if False:
            while True:
                i = 10
        ':math:`\\max(f(x)) = -\\min(-f(x))`\n\n        Parameters\n        ----------\n        problem : Problem\n            The problem whose objective is to be flipped.\n\n        Returns\n        -------\n        Problem\n            A problem with a flipped objective.\n        list\n            The inverse data.\n        '
        is_maximize = type(problem.objective) == Maximize
        objective = Minimize if is_maximize else Maximize
        problem = cvxtypes.problem()(objective(-problem.objective.expr), problem.constraints)
        return (problem, [])

    def invert(self, solution, inverse_data):
        if False:
            print('Hello World!')
        'Map the solution of the flipped problem to that of the original.\n\n        Parameters\n        ----------\n        solution : Solution\n            A solution object.\n        inverse_data : list\n            The inverse data returned by an invocation to apply.\n\n        Returns\n        -------\n        Solution\n            A solution to the original problem.\n        '
        if solution.opt_val is not None:
            solution.opt_val = -solution.opt_val
        return solution