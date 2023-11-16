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
from typing import Optional, Tuple
import numpy as np
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.expressions import cvxtypes

class max(AxisAtom):
    """:math:`\\max_{i,j}\\{X_{i,j}\\}`.
    """
    __EXPR_AXIS_ERROR__ = '\n\n    The second argument to "max" was a cvxpy Expression, when it should have\n    been an int or None. This is probably a result of calling "cp.max" when you\n    should call "cp.maximum". The difference is that cp.max represents the\n    maximum entry in a single vector or matrix, while cp.maximum represents\n    the entry-wise max of a sequence of arguments that all have the same shape.\n\n    '

    def __init__(self, x, axis: Optional[int]=None, keepdims: bool=False) -> None:
        if False:
            print('Hello World!')
        if isinstance(axis, cvxtypes.expression()):
            raise ValueError(max.__EXPR_AXIS_ERROR__)
        super(max, self).__init__(x, axis=axis, keepdims=keepdims)

    @Atom.numpy_numeric
    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Returns the largest entry in x.\n        '
        return values[0].max(axis=self.axis, keepdims=self.keepdims)

    def _grad(self, values):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        return self._axis_grad(values)

    def _column_grad(self, value):
        if False:
            print('Hello World!')
        'Gives the (sub/super)gradient of the atom w.r.t. a column argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            value: A numeric value for a column.\n\n        Returns:\n            A NumPy ndarray or None.\n        '
        value = np.array(value).ravel(order='F')
        idx = np.argmax(value)
        D = np.zeros((value.size, 1))
        D[idx] = 1
        return D

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            for i in range(10):
                print('nop')
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self) -> bool:
        if False:
            return 10
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom concave?\n        '
        return False

    def is_atom_log_log_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            return 10
        'Is the atom log-log concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        return True

    def is_decr(self, idx) -> bool:
        if False:
            while True:
                i = 10
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def is_pwl(self) -> bool:
        if False:
            return 10
        'Is the atom piecewise linear?\n        '
        return self.args[0].is_pwl()