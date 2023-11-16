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
import abc
from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom

class Elementwise(Atom):
    """ Abstract base class for elementwise atoms. """
    __metaclass__ = abc.ABCMeta

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        'Shape is the same as the sum of the arguments.\n        '
        return u.shape.sum_shapes([arg.shape for arg in self.args])

    def validate_arguments(self) -> None:
        if False:
            return 10
        '\n        Verify that all the shapes are the same\n        or can be promoted.\n        '
        u.shape.sum_shapes([arg.shape for arg in self.args])
        super(Elementwise, self).validate_arguments()

    def is_symmetric(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression symmetric?\n        '
        symm_args = all((arg.is_symmetric() for arg in self.args))
        return self.shape[0] == self.shape[1] and symm_args

    @staticmethod
    def elemwise_grad_to_diag(value, rows, cols):
        if False:
            while True:
                i = 10
        'Converts elementwise gradient into a diagonal matrix for Atom._grad()\n\n        Args:\n            value: A scalar or NumPy matrix.\n\n        Returns:\n            A SciPy CSC sparse matrix.\n        '
        if not np.isscalar(value):
            value = value.ravel(order='F')
        return sp.dia_matrix((np.atleast_1d(value), [0]), shape=(rows, cols)).tocsc()

    @staticmethod
    def _promote(arg, shape: Tuple[int, ...]):
        if False:
            return 10
        'Promotes the lin op if necessary.\n\n        Parameters\n        ----------\n        arg : LinOp\n            LinOp to promote.\n        shape : tuple\n            The shape desired.\n\n        Returns\n        -------\n        tuple\n            Promoted LinOp.\n        '
        if arg.shape != shape:
            return lu.promote(arg, shape)
        else:
            return arg