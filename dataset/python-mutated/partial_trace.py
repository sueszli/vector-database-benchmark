"""
Copyright 2022, adapted from Convex.jl.

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
from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom

def _term(expr, j: int, dims: Tuple[int], axis: Optional[int]=0):
    if False:
        while True:
            i = 10
    'Helper function for partial trace.\n\n    Parameters\n    ----------\n    expr : :class:`~cvxpy.expressions.expression.Expression`\n        The 2D expression to take the partial trace of.\n    j : int\n        Term in the partial trace sum.\n    dims : tuple of ints.\n        A tuple of integers encoding the dimensions of each subsystem.\n    axis : int\n        The index of the subsystem to be traced out\n        from the tensor product that defines expr.\n    '
    a = sp.coo_matrix(([1.0], ([0], [0])))
    b = sp.coo_matrix(([1.0], ([0], [0])))
    for (i_axis, dim) in enumerate(dims):
        if i_axis == axis:
            v = sp.coo_matrix(([1], ([j], [0])), shape=(dim, 1))
            a = sp.kron(a, v.T)
            b = sp.kron(b, v)
        else:
            eye_mat = sp.eye(dim)
            a = sp.kron(a, eye_mat)
            b = sp.kron(b, eye_mat)
    return a @ expr @ b

def partial_trace(expr, dims: Tuple[int], axis: Optional[int]=0):
    if False:
        return 10
    '\n    Assumes :math:`\\texttt{expr} = X_1 \\otimes \\cdots \\otimes X_n` is a 2D Kronecker\n    product composed of :math:`n = \\texttt{len(dims)}` implicit subsystems.\n    Letting :math:`k = \\texttt{axis}`, the returned expression represents\n    the *partial trace* of :math:`\\texttt{expr}` along its :math:`k^{\\text{th}}` implicit subsystem:\n\n    .. math::\n\n        \\text{tr}(X_k) (X_1 \\otimes \\cdots \\otimes X_{k-1} \\otimes X_{k+1} \\otimes \\cdots \\otimes X_n).\n\n    Parameters\n    ----------\n    expr : :class:`~cvxpy.expressions.expression.Expression`\n        The 2D expression to take the partial trace of.\n    dims : tuple of ints.\n        A tuple of integers encoding the dimensions of each subsystem.\n    axis : int\n        The index of the subsystem to be traced out\n        from the tensor product that defines expr.\n    '
    expr = Atom.cast_to_const(expr)
    if expr.ndim < 2 or expr.shape[0] != expr.shape[1]:
        raise ValueError('Only supports square matrices.')
    if axis < 0 or axis >= len(dims):
        raise ValueError(f'Invalid axis argument, should be between 0 and {len(dims)}, got {axis}.')
    if expr.shape[0] != np.prod(dims):
        raise ValueError("Dimension of system doesn't correspond to dimension of subsystems.")
    return sum([_term(expr, j, dims, axis) for j in range(dims[axis])])