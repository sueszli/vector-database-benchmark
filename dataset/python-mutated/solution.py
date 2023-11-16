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
import numpy as np
import cvxpy.settings as s
INF_OR_UNB_MESSAGE = '\n    The problem is either infeasible or unbounded, but the solver\n    cannot tell which. Disable any solver-specific presolve methods\n    and re-solve to determine the precise problem status.\n\n    For GUROBI and CPLEX you can automatically perform this re-solve\n    with the keyword argument prob.solve(reoptimize=True, ...).\n    '

def failure_solution(status, attr=None) -> 'Solution':
    if False:
        print('Hello World!')
    'Factory function for infeasible or unbounded solutions.\n\n    Parameters\n    ----------\n    status : str\n        The problem status.\n\n    Returns\n    -------\n    Solution\n        A solution object.\n    '
    if status in [s.INFEASIBLE, s.INFEASIBLE_INACCURATE]:
        opt_val = np.inf
    elif status in [s.UNBOUNDED, s.UNBOUNDED_INACCURATE]:
        opt_val = -np.inf
    else:
        opt_val = None
    if attr is None:
        attr = {}
    if status == s.INFEASIBLE_OR_UNBOUNDED:
        attr['message'] = INF_OR_UNB_MESSAGE
    return Solution(status, opt_val, {}, {}, attr)

class Solution:
    """A solution to an optimization problem.

    Attributes
    ----------
    status : str
        The status code.
    opt_val : float
        The optimal value.
    primal_vars : dict of id to NumPy ndarray
        A map from variable ids to optimal values.
    dual_vars : dict of id to NumPy ndarray
        A map from constraint ids to dual values.
    attr : dict
        Miscelleneous information propagated up from a solver.
    """

    def __init__(self, status, opt_val, primal_vars, dual_vars, attr) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.status = status
        self.opt_val = opt_val
        self.primal_vars = primal_vars
        self.dual_vars = dual_vars
        self.attr = attr

    def copy(self) -> 'Solution':
        if False:
            for i in range(10):
                print('nop')
        return Solution(self.status, self.opt_val, self.primal_vars, self.dual_vars, self.attr)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return 'Solution(status=%s, opt_val=%s, primal_vars=%s, dual_vars=%s, attr=%s)' % (self.status, self.opt_val, self.primal_vars, self.dual_vars, self.attr)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Solution(%s, %s, %s, %s)' % (self.status, self.primal_vars, self.dual_vars, self.attr)