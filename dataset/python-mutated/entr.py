"""
Copyright 2013 Steven Diamond, Eric Chu

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
from scipy.special import xlogy
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint

class entr(Elementwise):
    """Elementwise :math:`-x\\log x`.
    """

    def __init__(self, x) -> None:
        if False:
            return 10
        super(entr, self).__init__(x)

    def numeric(self, values):
        if False:
            while True:
                i = 10
        x = values[0]
        results = -xlogy(x, x)
        if np.isscalar(results):
            if np.isnan(results):
                return -np.inf
        else:
            results[np.isnan(results)] = -np.inf
        return results

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            while True:
                i = 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (False, False)

    def is_atom_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom convex?\n        '
        return False

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom concave?\n        '
        return True

    def is_incr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def _grad(self, values):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        rows = self.args[0].size
        cols = self.size
        if np.min(values[0]) <= 0:
            return [None]
        else:
            grad_vals = -np.log(values[0]) - 1
            return [entr.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self) -> List[Constraint]:
        if False:
            for i in range(10):
                print('nop')
        'Returns constraints describing the domain of the node.\n        '
        return [self.args[0] >= 0]