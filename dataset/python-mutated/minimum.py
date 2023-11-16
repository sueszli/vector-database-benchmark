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
import sys
from typing import Any, List, Tuple
import numpy as np
from cvxpy.atoms.elementwise.elementwise import Elementwise
if sys.version_info >= (3, 0):
    from functools import reduce

class minimum(Elementwise):
    """Elementwise minimum of a sequence of expressions.
    """

    def __init__(self, arg1, arg2, *args) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Requires at least 2 arguments.\n        '
        super(minimum, self).__init__(arg1, arg2, *args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Returns the elementwise maximum.\n        '
        return reduce(np.minimum, values)

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            print('Hello World!')
        'Returns sign (is positive, is negative) of the expression.\n        '
        is_pos = all((arg.is_nonneg() for arg in self.args))
        is_neg = any((arg.is_nonpos() for arg in self.args))
        return (is_pos, is_neg)

    def is_atom_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom convex?\n        '
        return False

    def is_atom_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom concave?\n        '
        return True

    def is_atom_log_log_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom log-log convex?\n        '
        return False

    def is_atom_log_log_concave(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom log-log concave?\n        '
        return True

    def is_incr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-decreasing in argument idx?\n        '
        return True

    def is_decr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def is_pwl(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom piecewise linear?\n        '
        return all((arg.is_pwl() for arg in self.args))

    def _grad(self, values) -> List[Any]:
        if False:
            i = 10
            return i + 15
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        min_vals = np.array(self.numeric(values))
        unused = np.array(np.ones(min_vals.shape), dtype=bool)
        grad_list = []
        for (idx, value) in enumerate(values):
            rows = self.args[idx].size
            cols = self.size
            grad_vals = (value == min_vals) & unused
            unused[value == min_vals] = 0
            grad_list += [minimum.elemwise_grad_to_diag(grad_vals, rows, cols)]
        return grad_list