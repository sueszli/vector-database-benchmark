"""
Copyright 2017 Steven Diamond

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
from typing import List, Tuple
import numpy as np
import cvxpy.utilities.performance_utils as perf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression

class indicator(Expression):
    """An expression representing the convex function I(constraints) = 0
       if constraints hold, +infty otherwise.

    Parameters
    ----------
    constraints : list
       A list of constraint objects.
    err_tol:
       A numeric tolerance for determining whether the constraints hold.
    """

    def __init__(self, constraints: List[Constraint], err_tol: float=0.001) -> None:
        if False:
            print('Hello World!')
        self.args = constraints
        self.err_tol = err_tol
        super(indicator, self).__init__()

    @perf.compute_once
    def is_constant(self) -> bool:
        if False:
            return 10
        'The Indicator is constant if all constraints have constant args.\n        '
        all_args = sum([c.args for c in self.args], [])
        return all([arg.is_constant() for arg in all_args])

    def is_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression convex?\n        '
        return True

    def is_concave(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression concave?\n        '
        return False

    def is_log_log_convex(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def is_log_log_concave(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def is_nonneg(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the expression positive?\n        '
        return True

    def is_nonpos(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression negative?\n        '
        return False

    def is_imag(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the Leaf imaginary?\n        '
        return False

    def is_complex(self) -> bool:
        if False:
            return 10
        'Is the Leaf complex valued?\n        '
        return False

    def get_data(self) -> List[float]:
        if False:
            print('Hello World!')
        'Returns info needed to reconstruct the expression besides the args.\n        '
        return [self.err_tol]

    @property
    def shape(self) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        'Returns the (row, col) dimensions of the expression.\n        '
        return ()

    def name(self) -> str:
        if False:
            print('Hello World!')
        'Returns the string representation of the expression.\n        '
        return f'Indicator({self.args})'

    def domain(self) -> List[Constraint]:
        if False:
            print('Hello World!')
        'A list of constraints describing the closure of the region\n           where the expression is finite.\n        '
        return self.args

    @property
    def value(self) -> float:
        if False:
            i = 10
            return i + 15
        'Returns the numeric value of the expression.\n\n        Returns:\n            A numpy matrix or a scalar.\n        '
        if all((cons.value(tolerance=self.err_tol) for cons in self.args)):
            return 0.0
        else:
            return np.infty

    @property
    def grad(self):
        if False:
            print('Hello World!')
        'Gives the (sub/super)gradient of the expression w.r.t. each variable.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n        None indicates variable values unknown or outside domain.\n\n        Returns:\n            A map of variable to SciPy CSC sparse matrix or None.\n        '
        raise NotImplementedError()