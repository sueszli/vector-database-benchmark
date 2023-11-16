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
from scipy import linalg as LA
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint

class condition_number(Atom):
    """ Condition Number; :math:`\\lambda_{\\max}(A) / \\lambda_{\\min}(A)`.
        Requires that A be a Positive Semidefinite Matrix.
    """

    def __init__(self, A) -> None:
        if False:
            i = 10
            return i + 15
        super(condition_number, self).__init__(A)

    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Returns the condition number of A.\n\n        Requires that A be a Positive Semidefinite Matrix.\n        '
        lo = hi = self.args[0].shape[0] - 1
        max_eigen = LA.eigvalsh(values[0], eigvals=(lo, hi))[0]
        min_eigen = -LA.eigvalsh(-values[0], eigvals=(lo, hi))[0]
        return max_eigen / min_eigen

    def _domain(self) -> List[Constraint]:
        if False:
            while True:
                i = 10
        'Returns constraints describing the domain of the node.\n        '
        return [self.args[0].H == self.args[0], self.args[0] >> 0]

    def _grad(self, values):
        if False:
            return 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        raise NotImplementedError

    def validate_arguments(self) -> None:
        if False:
            print('Hello World!')
        'Verify that the argument A is square.\n        '
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError(f'The argument {self.args[0].name()} to condition_number must be a square matrix.')

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            return 10
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            return 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            return 10
        'Is the atom convex?\n        '
        return False

    def is_atom_concave(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom concave?\n        '
        return False

    def is_atom_quasiconvex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom quasiconvex?\n        '
        return True

    def is_incr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-increasing in argument idx?\n        '
        return False