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
from typing import Tuple
import numpy as np
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom

class cummax(AxisAtom):
    """Cumulative maximum.
    """

    def __init__(self, x, axis: int=0) -> None:
        if False:
            print('Hello World!')
        super(cummax, self).__init__(x, axis=axis)

    @Atom.numpy_numeric
    def numeric(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Returns the largest entry in x.\n        '
        return np.maximum.accumulate(values[0], axis=self.axis)

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        'The same as the input.\n        '
        return self.args[0].shape

    def _grad(self, values):
        if False:
            i = 10
            return i + 15
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        return self._axis_grad(values)

    def _column_grad(self, value):
        if False:
            print('Hello World!')
        'Gives the (sub/super)gradient of the atom w.r.t. a column argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            value: A numeric value for a column.\n\n        Returns:\n            A NumPy ndarray or None.\n        '
        value = np.array(value).ravel(order='F')
        maxes = np.maximum.accumulate(value)
        D = np.zeros((value.size, 1))
        D[0] = 1
        if value.size > 1:
            D[1:] = maxes[1:] > maxes[:-1]
        return D

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            return 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def get_data(self):
        if False:
            print('Hello World!')
        'Returns the axis being summed.\n        '
        return [self.axis]

    def is_atom_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-decreasing in argument idx?\n        '
        return True

    def is_decr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-increasing in argument idx?\n        '
        return False