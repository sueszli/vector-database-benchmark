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
from cvxpy.atoms.elementwise.elementwise import Elementwise

class logistic(Elementwise):
    """:math:`\\log(1 + e^{x})`

    This is a special case of log(sum(exp)) that is evaluates to a vector rather
    than to a scalar which is useful for logistic regression.
    """

    def __init__(self, x) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(logistic, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        if False:
            return 10
        'Evaluates e^x elementwise, adds 1, and takes the log.\n        '
        return np.logaddexp(0, values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            return 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            return 10
        'Is the atom concave?\n        '
        return False

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

    def _grad(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        rows = self.args[0].size
        cols = self.size
        grad_vals = np.exp(values[0] - np.logaddexp(0, values[0]))
        return [logistic.elemwise_grad_to_diag(grad_vals, rows, cols)]