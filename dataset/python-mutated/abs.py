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
from .elementwise import Elementwise

class abs(Elementwise):
    """ Elementwise absolute value """
    _allow_complex = True

    def __init__(self, x) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(abs, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        if False:
            return 10
        return np.absolute(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            for i in range(10):
                print('nop')
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

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
            while True:
                i = 10
        'Is the composition non-decreasing in argument idx?\n        '
        return self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-increasing in argument idx?\n        '
        return self.args[idx].is_nonpos()

    def is_pwl(self) -> bool:
        if False:
            return 10
        'Is the atom piecewise linear?\n        '
        return self.args[0].is_pwl() and (self.args[0].is_real() or self.args[0].is_imag())

    def _grad(self, values):
        if False:
            return 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        rows = self.expr.size
        cols = self.size
        D = np.zeros(self.expr.shape)
        D += values[0] > 0
        D -= values[0] < 0
        return [abs.elemwise_grad_to_diag(D, rows, cols)]