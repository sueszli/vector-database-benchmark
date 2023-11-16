"""
Copyright 2017 Fair Isaac Corp.

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
from collections import namedtuple
from cvxpy.problems.problem import Problem
from cvxpy.utilities.deterministic import unique_list
CachedProblem = namedtuple('CachedProblem', ['objective', 'constraints'])
SolveResult = namedtuple('SolveResult', ['opt_value', 'status', 'primal_values', 'dual_values'])

class XpressProblem(Problem):
    """A convex optimization problem associated with the Xpress Optimizer

    Attributes
    ----------
    objective : Minimize or Maximize
        The expression to minimize or maximize.
    constraints : list
        The constraints on the problem variables.
    """
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, objective, constraints=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(XpressProblem, self).__init__(objective, constraints)
        self._iis = None

    def _reset_iis(self) -> None:
        if False:
            print('Hello World!')
        'Clears the iis information\n        '
        self._iis = None
        self._transferRow = None

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'XpressProblem(%s, %s)' % (repr(self.objective), repr(self.constraints))

    def __neg__(self) -> 'XpressProblem':
        if False:
            return 10
        return XpressProblem(-self.objective, self.constraints)

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other == 0:
            return self
        elif not isinstance(other, XpressProblem):
            raise NotImplementedError()
        return XpressProblem(self.objective + other.objective, unique_list(self.constraints + other.constraints))

    def __sub__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, XpressProblem):
            raise NotImplementedError()
        return XpressProblem(self.objective - other.objective, unique_list(self.constraints + other.constraints))

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return XpressProblem(self.objective * other, self.constraints)

    def __div__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return XpressProblem(self.objective * (1.0 / other), self.constraints)