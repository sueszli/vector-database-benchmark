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
from cvxpy.atoms.affine.affine_atom import AffAtom

class real(AffAtom):
    """Extracts the real part of an expression.
    """

    def __init__(self, expr) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(real, self).__init__(expr)

    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Convert the vector constant into a diagonal matrix.\n        '
        return np.real(values[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the shape of the expression.\n        '
        return self.args[0].shape

    def is_imag(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression imaginary?\n        '
        return False

    def is_complex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression complex valued?\n        '
        return False

    def is_symmetric(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression symmetric?\n        '
        return self.args[0].is_hermitian()