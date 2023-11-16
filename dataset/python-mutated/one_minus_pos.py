"""
Copyright 2018 Akshay Agrawal

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
import scipy.sparse as sp
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.atom import Atom

def diff_pos(x, y):
    if False:
        print('Hello World!')
    'The difference :math:`x - y` with domain `\\{x, y : x > y > 0\\}`.\n\n    This atom is log-log concave.\n\n    Parameters\n    ----------\n    x : :class:`~cvxpy.expressions.expression.Expression`\n        An Expression.\n    y : :class:`~cvxpy.expressions.expression.Expression`\n        An Expression.\n    '
    return multiply(x, one_minus_pos(y / x))

class one_minus_pos(Atom):
    """The difference :math:`1 - x` with domain `\\{x : 0 < x < 1\\}`.

    This atom is log-log concave.

    Parameters
    ----------
    x : :class:`~cvxpy.expressions.expression.Expression`
        An Expression.
    """

    def __init__(self, x) -> None:
        if False:
            while True:
                i = 10
        super(one_minus_pos, self).__init__(x)
        self.args[0] = x
        self._ones = np.ones(self.args[0].shape)

    def numeric(self, values):
        if False:
            while True:
                i = 10
        return self._ones - values[0]

    def _grad(self, values):
        if False:
            print('Hello World!')
        del values
        return sp.csc_matrix(-1.0 * self._ones)

    def name(self) -> str:
        if False:
            while True:
                i = 10
        return '%s(%s)' % (self.__class__.__name__, self.args[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the (row, col) shape of the expression.\n        '
        return self.args[0].shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            print('Hello World!')
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            return 10
        'Is the atom convex?\n        '
        return False

    def is_atom_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom concave?\n        '
        return False

    def is_atom_log_log_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom log-log convex?\n        '
        return False

    def is_atom_log_log_concave(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom log-log concave?\n        '
        return True

    def is_incr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-increasing in argument idx?\n        '
        return True