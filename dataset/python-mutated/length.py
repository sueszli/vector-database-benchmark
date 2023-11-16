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
from typing import Tuple
import numpy as np
import cvxpy.settings as s
from cvxpy.atoms.atom import Atom

class length(Atom):
    """Length of a vector (index of last nonzero, ones-based).
    """

    def __init__(self, x) -> None:
        if False:
            return 10
        super(length, self).__init__(x)
        if not self.args[0].is_vector():
            raise ValueError('`length` can only be applied to vectors.')

    @Atom.numpy_numeric
    def numeric(self, values) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the length of x.\n        '
        outside_tol = np.abs(values[0]) > s.ATOM_EVAL_TOL
        return np.max(np.nonzero(outside_tol)) + 1

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            i = 10
            return i + 15
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom convex?\n        '
        return False

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom concave?\n        '
        return False

    def is_atom_quasiconvex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom quasiconvex?\n        '
        return True

    def is_atom_quasiconcave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom quasiconvex?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            while True:
                i = 10
        'Is the composition non-decreasing in argument idx?\n        '
        return self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        return self.args[idx].is_nonpos()

    def _grad(self, values) -> None:
        if False:
            for i in range(10):
                print('nop')
        return None