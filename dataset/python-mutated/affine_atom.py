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
from typing import Any, List, Tuple
import scipy.sparse as sp
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.constants import Constant
from cvxpy.utilities import performance_utils as perf

class AffAtom(Atom):
    """ Abstract base class for affine atoms. """
    __metaclass__ = abc.ABCMeta
    _allow_complex = True

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            i = 10
            return i + 15
        'By default, the sign is the most general of all the argument signs.\n        '
        return u.sign.sum_signs([arg for arg in self.args])

    def is_imag(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the expression imaginary?\n        '
        return all((arg.is_imag() for arg in self.args))

    def is_complex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the expression complex valued?\n        '
        return any((arg.is_complex() for arg in self.args))

    def is_atom_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            return 10
        'Is the atom concave?\n        '
        return True

    def is_incr(self, idx) -> bool:
        if False:
            while True:
                i = 10
        'Is the composition non-decreasing in argument idx?\n        '
        return True

    def is_decr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def is_quadratic(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return all((arg.is_quadratic() for arg in self.args))

    def has_quadratic_term(self) -> bool:
        if False:
            while True:
                i = 10
        'Does the affine head of the expression contain a quadratic term?\n\n        The affine head is all nodes with a path to the root node\n        that does not pass through any non-affine atom. If the root node\n        is non-affine, then the affine head is the root alone.\n        '
        return any((arg.has_quadratic_term() for arg in self.args))

    def is_qpwa(self) -> bool:
        if False:
            print('Hello World!')
        return all((arg.is_qpwa() for arg in self.args))

    def is_pwl(self) -> bool:
        if False:
            while True:
                i = 10
        return all((arg.is_pwl() for arg in self.args))

    @perf.compute_once
    def is_psd(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression a positive semidefinite matrix?\n        '
        for (idx, arg) in enumerate(self.args):
            if not (self.is_incr(idx) and arg.is_psd() or (self.is_decr(idx) and arg.is_nsd())):
                return False
        return True

    @perf.compute_once
    def is_nsd(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression a positive semidefinite matrix?\n        '
        for (idx, arg) in enumerate(self.args):
            if not (self.is_decr(idx) and arg.is_psd() or (self.is_incr(idx) and arg.is_nsd())):
                return False
        return True

    def _grad(self, values) -> List[Any]:
        if False:
            return 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        fake_args = []
        var_offsets = {}
        offset = 0
        for (idx, arg) in enumerate(self.args):
            if arg.is_constant():
                fake_args += [Constant(arg.value).canonical_form[0]]
            else:
                fake_args += [lu.create_var(arg.shape, idx)]
                var_offsets[idx] = offset
                offset += arg.size
        var_length = offset
        (fake_expr, _) = self.graph_implementation(fake_args, self.shape, self.get_data())
        param_to_size = {lo.CONSTANT_ID: 1}
        param_to_col = {lo.CONSTANT_ID: 0}
        canon_mat = canonInterface.get_problem_matrix([fake_expr], var_length, var_offsets, param_to_size, param_to_col, self.size)
        shape = (var_length + 1, self.size)
        stacked_grad = canon_mat.reshape(shape).tocsc()[:-1, :]
        grad_list = []
        start = 0
        for arg in self.args:
            if arg.is_constant():
                grad_shape = (arg.size, shape[1])
                if grad_shape == (1, 1):
                    grad_list += [0]
                else:
                    grad_list += [sp.coo_matrix(grad_shape, dtype='float64')]
            else:
                stop = start + arg.size
                grad_list += [stacked_grad[start:stop, :]]
                start = stop
        return grad_list