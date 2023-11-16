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
from __future__ import division
import operator as op
from functools import reduce
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.reshape import deep_flatten, reshape
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DCPError
from cvxpy.expressions.constants.parameter import is_param_affine, is_param_free
from cvxpy.expressions.expression import Expression

class BinaryOperator(AffAtom):
    """
    Base class for expressions involving binary operators. (other than addition)

    """
    OP_NAME = 'BINARY_OP'

    def __init__(self, lh_exp, rh_exp) -> None:
        if False:
            while True:
                i = 10
        super(BinaryOperator, self).__init__(lh_exp, rh_exp)

    def name(self):
        if False:
            while True:
                i = 10
        pretty_args = []
        for a in self.args:
            if isinstance(a, (AddExpression, DivExpression)):
                pretty_args.append('(' + a.name() + ')')
            else:
                pretty_args.append(a.name())
        return pretty_args[0] + ' ' + self.OP_NAME + ' ' + pretty_args[1]

    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Applies the binary operator to the values.\n        '
        return reduce(self.OP_FUNC, values)

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            while True:
                i = 10
        'Default to rules for times.\n        '
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_imag(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression imaginary?\n        '
        return self.args[0].is_imag() and self.args[1].is_real() or (self.args[0].is_real() and self.args[1].is_imag())

    def is_complex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression complex valued?\n        '
        return (self.args[0].is_complex() or self.args[1].is_complex()) and (not (self.args[0].is_imag() and self.args[1].is_imag()))

def matmul(lh_exp, rh_exp) -> 'MulExpression':
    if False:
        while True:
            i = 10
    'Matrix multiplication.'
    return MulExpression(lh_exp, rh_exp)

class MulExpression(BinaryOperator):
    """Matrix multiplication.

    The semantics of multiplication are exactly as those of NumPy's
    matmul function, except here multiplication by a scalar is permitted.
    MulExpression objects can be created by using the '*' operator of
    the Expression class.

    Parameters
    ----------
    lh_exp : Expression
        The left-hand side of the multiplication.
    rh_exp : Expression
        The right-hand side of the multiplication.
    """
    OP_NAME = '@'
    OP_FUNC = op.mul

    def numeric(self, values):
        if False:
            print('Hello World!')
        'Matrix multiplication.\n        '
        if values[0].shape == () or values[1].shape == () or intf.is_sparse(values[0]) or intf.is_sparse(values[1]):
            return values[0] * values[1]
        else:
            return np.matmul(values[0], values[1])

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        'Returns the (row, col) shape of the expression.\n        '
        return u.shape.mul_shapes(self.args[0].shape, self.args[1].shape)

    def is_atom_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Multiplication is convex (affine) in its arguments only if one of\n           the arguments is constant.\n        '
        if u.scopes.dpp_scope_active():
            x = self.args[0]
            y = self.args[1]
            return (x.is_constant() or y.is_constant()) or (is_param_affine(x) and is_param_free(y)) or (is_param_affine(y) and is_param_free(x))
        else:
            return self.args[0].is_constant() or self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'If the multiplication atom is convex, then it is affine.\n        '
        return self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom log-log concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-decreasing in argument idx?\n        '
        return self.args[1 - idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        return self.args[1 - idx].is_nonpos()

    def _grad(self, values):
        if False:
            print('Hello World!')
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        if self.args[0].is_constant() or self.args[1].is_constant():
            return super(MulExpression, self)._grad(values)
        X = values[0]
        Y = values[1]
        DX_rows = self.args[0].size
        cols = self.args[0].size
        DX = sp.dok_matrix((DX_rows, cols))
        for k in range(self.args[0].shape[0]):
            DX[k::self.args[0].shape[0], k::self.args[0].shape[0]] = Y
        DX = sp.csc_matrix(DX)
        cols = 1 if len(self.args[1].shape) == 1 else self.args[1].shape[1]
        DY = sp.block_diag([X.T for k in range(cols)], 'csc')
        return [DX, DY]

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            while True:
                i = 10
        'Multiply the linear expressions.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if self.args[0].is_constant():
            return (lu.mul_expr(lhs, rhs, shape), [])
        elif self.args[1].is_constant():
            return (lu.rmul_expr(lhs, rhs, shape), [])
        else:
            raise DCPError('Product of two non-constant expressions is not DCP.')

class multiply(MulExpression):
    """ Multiplies two expressions elementwise.
    """

    def __init__(self, lh_expr, rh_expr) -> None:
        if False:
            i = 10
            return i + 15
        (lh_expr, rh_expr) = self.broadcast(lh_expr, rh_expr)
        super(multiply, self).__init__(lh_expr, rh_expr)

    def is_atom_log_log_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            return 10
        'Is the atom log-log concave?\n        '
        return True

    def is_atom_quasiconvex(self) -> bool:
        if False:
            return 10
        return (self.args[0].is_constant() or self.args[1].is_constant()) or (self.args[0].is_nonneg() and self.args[1].is_nonpos()) or (self.args[0].is_nonpos() and self.args[1].is_nonneg())

    def is_atom_quasiconcave(self) -> bool:
        if False:
            while True:
                i = 10
        return (self.args[0].is_constant() or self.args[1].is_constant()) or all((arg.is_nonneg() for arg in self.args)) or all((arg.is_nonpos() for arg in self.args))

    def numeric(self, values):
        if False:
            print('Hello World!')
        'Multiplies the values elementwise.\n        '
        if sp.issparse(values[0]):
            return values[0].multiply(values[1])
        elif sp.issparse(values[1]):
            return values[1].multiply(values[0])
        else:
            return np.multiply(values[0], values[1])

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            return 10
        'The sum of the argument dimensions - 1.\n        '
        return u.shape.sum_shapes([arg.shape for arg in self.args])

    def is_psd(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression a positive semidefinite matrix?\n        '
        return self.args[0].is_psd() and self.args[1].is_psd() or (self.args[0].is_nsd() and self.args[1].is_nsd())

    def is_nsd(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression a negative semidefinite matrix?\n        '
        return self.args[0].is_psd() and self.args[1].is_nsd() or (self.args[0].is_nsd() and self.args[1].is_psd())

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            print('Hello World!')
        'Multiply the expressions elementwise.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of exprraints)\n        '
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if self.args[0].is_constant():
            return (lu.multiply(lhs, rhs), [])
        elif self.args[1].is_constant():
            return (lu.multiply(rhs, lhs), [])
        else:
            raise DCPError('Product of two non-constant expressions is not DCP.')

class DivExpression(BinaryOperator):
    """Division by scalar.

    Can be created by using the / operator of expression.
    """
    OP_NAME = '/'
    OP_FUNC = np.divide

    def __init__(self, lh_expr, rh_expr) -> None:
        if False:
            i = 10
            return i + 15
        (lh_expr, rh_expr) = self.broadcast(lh_expr, rh_expr)
        super(DivExpression, self).__init__(lh_expr, rh_expr)

    def numeric(self, values):
        if False:
            i = 10
            return i + 15
        'Divides numerator by denominator.\n        '
        for i in range(2):
            if sp.issparse(values[i]):
                values[i] = values[i].todense().A
        return np.divide(values[0], values[1])

    def is_quadratic(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.args[0].is_quadratic() and self.args[1].is_constant()

    def has_quadratic_term(self) -> bool:
        if False:
            print('Hello World!')
        'Can be a quadratic term if divisor is constant.'
        return self.args[0].has_quadratic_term() and self.args[1].is_constant()

    def is_qpwa(self) -> bool:
        if False:
            print('Hello World!')
        return self.args[0].is_qpwa() and self.args[1].is_constant()

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the (row, col) shape of the expression.\n        '
        return self.args[0].shape

    def is_atom_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Division is convex (affine) in its arguments only if\n           the denominator is constant.\n        '
        return self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        return self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        if False:
            return 10
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            return 10
        'Is the atom log-log concave?\n        '
        return True

    def is_atom_quasiconvex(self) -> bool:
        if False:
            return 10
        return self.args[1].is_nonneg() or self.args[1].is_nonpos()

    def is_atom_quasiconcave(self) -> bool:
        if False:
            print('Hello World!')
        return self.is_atom_quasiconvex()

    def is_incr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-decreasing in argument idx?\n        '
        if idx == 0:
            return self.args[1].is_nonneg()
        else:
            return self.args[0].is_nonpos()

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        if idx == 0:
            return self.args[1].is_nonpos()
        else:
            return self.args[0].is_nonneg()

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            return 10
        'Multiply the linear expressions.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (lu.div_expr(arg_objs[0], arg_objs[1]), [])

def scalar_product(x, y):
    if False:
        return 10
    '\n    Return the standard inner product (or "scalar product") of (x,y).\n\n    Parameters\n    ----------\n    x : Expression, int, float, NumPy ndarray, or nested list thereof.\n        The conjugate-linear argument to the inner product.\n    y : Expression, int, float, NumPy ndarray, or nested list thereof.\n        The linear argument to the inner product.\n\n    Returns\n    -------\n    expr : Expression\n        The standard inner product of (x,y), conjugate-linear in x.\n        We always have ``expr.shape == ()``.\n\n    Notes\n    -----\n    The arguments ``x`` and ``y`` can be nested lists; these lists\n    will be flattened independently of one another.\n\n    For example, if ``x = [[a],[b]]`` and  ``y = [c, d]`` (with ``a,b,c,d``\n    real scalars), then this function returns an Expression representing\n    ``a * c + b * d``.\n    '
    x = deep_flatten(x)
    y = deep_flatten(y)
    prod = multiply(conj(x), y)
    return cvxpy_sum(prod)

def outer(x, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the outer product of (x,y).\n\n    Parameters\n    ----------\n    x : Expression, int, float, NumPy ndarray, or nested list thereof.\n        Input is flattened if not already a vector.\n        The linear argument to the outer product.\n    y : Expression, int, float, NumPy ndarray, or nested list thereof.\n        Input is flattened if not already a vector.\n        The transposed-linear argument to the outer product.\n\n    Returns\n    -------\n    expr : Expression\n        The outer product of (x,y), linear in x and transposed-linear in y.\n    '
    x = Expression.cast_to_const(x)
    if x.ndim > 1:
        raise ValueError('x must be a vector.')
    y = Expression.cast_to_const(y)
    if y.ndim > 1:
        raise ValueError('y must be a vector.')
    x = reshape(x, (x.size, 1))
    y = reshape(y, (1, y.size))
    return x @ y