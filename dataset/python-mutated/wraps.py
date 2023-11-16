"""
Copyright 2013 Steven Diamond, 2022 the CVXPY authors.

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
from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint

class Wrap(AffAtom):
    """A no-op wrapper to assert properties.
    """

    def __init__(self, arg) -> None:
        if False:
            return 10
        return super(Wrap, self).__init__(arg)

    def is_atom_log_log_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def numeric(self, values):
        if False:
            return 10
        ' Returns input.\n        '
        return values[0]

    def is_complex(self) -> bool:
        if False:
            return 10
        return self.args[0].is_complex()

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            for i in range(10):
                print('nop')
        'Shape of input.\n        '
        return self.args[0].shape

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            while True:
                i = 10
        'Stack the expressions horizontally.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (arg_objs[0], [])

class nonneg_wrap(Wrap):
    """Asserts that the expression is nonnegative.
    """

    def is_nonneg(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

class nonpos_wrap(Wrap):
    """Asserts that the expression is nonpositive.
    """

    def is_nonpos(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

class psd_wrap(Wrap):
    """Asserts that a square matrix is PSD.
    """

    def validate_arguments(self) -> None:
        if False:
            print('Hello World!')
        arg = self.args[0]
        ndim_test = len(arg.shape) == 2
        if not ndim_test:
            raise ValueError('The input must be a square matrix.')
        elif arg.shape[0] != arg.shape[1]:
            raise ValueError('The input must be a square matrix.')

    def is_psd(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def is_nsd(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def is_symmetric(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self.args[0].is_complex()

    def is_hermitian(self) -> bool:
        if False:
            while True:
                i = 10
        return True

class symmetric_wrap(Wrap):
    """Asserts that a real square matrix is symmetric
    """

    def validate_arguments(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        validate_real_square(self.args[0])

    def is_symmetric(self) -> bool:
        if False:
            return 10
        return True

    def is_hermitian(self) -> bool:
        if False:
            return 10
        return True

class hermitian_wrap(Wrap):
    """Asserts that a square matrix is Hermitian.
    """

    def validate_arguments(self) -> None:
        if False:
            while True:
                i = 10
        arg = self.args[0]
        ndim_test = len(arg.shape) == 2
        if not ndim_test:
            raise ValueError('The input must be a square matrix.')
        elif arg.shape[0] != arg.shape[1]:
            raise ValueError('The input must be a square matrix.')

    def is_hermitian(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

class skew_symmetric_wrap(Wrap):
    """Asserts that X is a real square matrix, satisfying X + X.T == 0.
    """

    def validate_arguments(self) -> None:
        if False:
            return 10
        validate_real_square(self.args[0])

    def is_skew_symmetric(self) -> bool:
        if False:
            while True:
                i = 10
        return True

def validate_real_square(arg):
    if False:
        print('Hello World!')
    ndim_test = len(arg.shape) == 2
    if not ndim_test:
        raise ValueError('The input must be a square matrix.')
    elif arg.shape[0] != arg.shape[1]:
        raise ValueError('The input must be a square matrix.')
    elif not arg.is_real():
        raise ValueError('The input must be a real matrix.')