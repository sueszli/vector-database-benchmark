"""
Copyright 2022, the CVXPY authors

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
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint

class xexp(Elementwise):
    """Elementwise :math:`{x}*e^{x}`.
    """

    def __init__(self, x) -> None:
        if False:
            i = 10
            return i + 15
        super(xexp, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        if False:
            while True:
                i = 10
        return values[0] * np.exp(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            i = 10
            return i + 15
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom convex?\n        '
        return self.args[0].is_nonneg()

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom concave?\n        '
        return False

    def is_atom_log_log_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom log-log concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            while True:
                i = 10
        'Is the composition non-decreasing in argument idx?\n        '
        return True

    def is_decr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def _grad(self, values):
        if False:
            i = 10
            return i + 15
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        rows = self.args[0].size
        cols = self.size
        grad_vals = np.exp(values[0]) * (1 + values[0])
        return [xexp.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self) -> List[Constraint]:
        if False:
            i = 10
            return i + 15
        'Returns constraints describing the domain of the node.\n        '
        return [self.args[0] >= 0]