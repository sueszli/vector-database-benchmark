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
from cvxpy.atoms.atom import Atom

class sign(Atom):
    """Sign of an expression (-1 for x <= 0, +1 for x > 0).
    """

    def __init__(self, x) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(sign, self).__init__(x)

    @Atom.numpy_numeric
    def numeric(self, values):
        if False:
            return 10
        'Returns the sign of x.\n        '
        x = values[0].copy()
        x[x > 0] = 1.0
        x[x <= 0] = -1.0
        return x

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            return 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        'Is the atom quasiconvex?\n        '
        return self.args[0].is_scalar()

    def is_atom_quasiconcave(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom quasiconvex?\n        '
        return self.args[0].is_scalar()

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

    def _grad(self, values) -> None:
        if False:
            print('Hello World!')
        return None