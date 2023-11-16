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
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable

def get_diff_mat(dim: int, axis: int) -> sp.csc_matrix:
    if False:
        for i in range(10):
            print('nop')
    'Return a sparse matrix representation of first order difference operator.\n\n    Parameters\n    ----------\n    dim : int\n       The length of the matrix dimensions.\n    axis : int\n       The axis to take the difference along.\n\n    Returns\n    -------\n    SciPy CSC matrix\n        A square matrix representing first order difference.\n    '
    val_arr = []
    row_arr = []
    col_arr = []
    for i in range(dim):
        val_arr.append(1.0)
        row_arr.append(i)
        col_arr.append(i)
        if i > 0:
            val_arr.append(-1.0)
            row_arr.append(i)
            col_arr.append(i - 1)
    mat = sp.csc_matrix((val_arr, (row_arr, col_arr)), (dim, dim))
    if axis == 0:
        return mat
    else:
        return mat.T

class cumsum(AffAtom, AxisAtom):
    """Cumulative sum.

    Attributes
    ----------
    expr : CVXPY expression
        The expression being summed.
    axis : int
        The axis to sum across if 2D.
    """

    def __init__(self, expr: Expression, axis: int=0) -> None:
        if False:
            return 10
        super(cumsum, self).__init__(expr, axis)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Convolve the two values.\n        '
        return np.cumsum(values[0], axis=self.axis)

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            return 10
        'The same as the input.\n        '
        return self.args[0].shape

    def _grad(self, values):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        dim = values[0].shape[self.axis]
        mat = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i + 1):
                mat[i, j] = 1
        var = Variable(self.args[0].shape)
        if self.axis == 0:
            grad = MulExpression(mat, var)._grad(values)[1]
        else:
            grad = MulExpression(var, mat.T)._grad(values)[0]
        return [grad]

    def get_data(self):
        if False:
            return 10
        'Returns the axis being summed.\n        '
        return [self.axis]

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            return 10
        'Cumulative sum via difference matrix.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        Y = lu.create_var(shape)
        axis = data[0]
        dim = shape[axis]
        diff_mat = get_diff_mat(dim, axis)
        diff_mat = lu.create_const(diff_mat, (dim, dim), sparse=True)
        if axis == 0:
            diff = lu.mul_expr(diff_mat, Y)
        else:
            diff = lu.rmul_expr(Y, diff_mat)
        return (Y, [lu.create_eq(arg_objs[0], diff)])