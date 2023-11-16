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
import numpy as np
from scipy import linalg as LA
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.sum_largest import sum_largest

class lambda_sum_largest(lambda_max):
    """Sum of the largest k eigenvalues.
    """
    _allow_complex = True

    def __init__(self, X, k) -> None:
        if False:
            print('Hello World!')
        self.k = k
        super(lambda_sum_largest, self).__init__(X)

    def validate_arguments(self) -> None:
        if False:
            while True:
                i = 10
        'Verify that the argument A is square.\n        '
        X = self.args[0]
        if not X.ndim == 2 or X.shape[0] != X.shape[1]:
            raise ValueError('First argument must be a square matrix.')
        elif int(self.k) != self.k or self.k <= 0:
            raise ValueError('Second argument must be a positive integer.')

    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Returns the largest eigenvalue of A.\n\n        Requires that A be symmetric.\n        '
        eigs = LA.eigvalsh(values[0])
        return sum_largest(eigs, self.k).value

    def get_data(self):
        if False:
            i = 10
            return i + 15
        'Returns the parameter k.\n        '
        return [self.k]

    def _grad(self, values):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        raise NotImplementedError()

    @property
    def value(self):
        if False:
            return 10
        if not np.allclose(self.args[0].value, self.args[0].value.T.conj()):
            raise ValueError('Input matrix was not Hermitian/symmetric.')
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()