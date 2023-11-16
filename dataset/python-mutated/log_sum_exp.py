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
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom

class log_sum_exp(AxisAtom):
    """:math:`\\log\\sum_i e^{x_i}`

    """

    def __init__(self, x, axis=None, keepdims: bool=False) -> None:
        if False:
            print('Hello World!')
        super(log_sum_exp, self).__init__(x, axis=axis, keepdims=keepdims)

    @Atom.numpy_numeric
    def numeric(self, values):
        if False:
            print('Hello World!')
        'Evaluates e^x elementwise, sums, and takes the log.\n        '
        return logsumexp(values[0], axis=self.axis, keepdims=self.keepdims)

    def _grad(self, values):
        if False:
            print('Hello World!')
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        return self._axis_grad(values)

    def _column_grad(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Gives the (sub/super)gradient of the atom w.r.t. a column argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            value: A numeric value for a column.\n\n        Returns:\n            A NumPy ndarray or None.\n        '
        denom = np.exp(logsumexp(value, axis=None, keepdims=True))
        nom = np.exp(value)
        D = nom / denom
        return D

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            while True:
                i = 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (self.args[0].is_nonneg(), False)

    def is_atom_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            return 10
        'Is the atom concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-decreasing in argument idx?\n        '
        return True

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        return False