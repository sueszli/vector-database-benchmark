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
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.diag import diag_mat, diag_vec
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri
from cvxpy.atoms.elementwise.log import log
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.canonicalizers.log_canon import log_canon

def log_det_canon(expr, args):
    if False:
        print('Hello World!')
    'Reduces the atom to an affine expression and list of constraints.\n\n    Creates the equivalent problem::\n\n       maximize    sum(log(D[i, i]))\n       subject to: D diagonal\n                   diag(D) = diag(Z)\n                   Z is upper triangular.\n                   [D Z; Z.T A] is positive semidefinite\n\n    The problem computes the LDL factorization:\n\n    .. math::\n\n       A = (Z^TD^{-1})D(D^{-1}Z)\n\n    This follows from the inequality:\n\n    .. math::\n\n       \\det(A) >= \\det(D) + \\det([D, Z; Z^T, A])/\\det(D)\n               >= \\det(D)\n\n    because (Z^TD^{-1})D(D^{-1}Z) is a feasible D, Z that achieves\n    det(A) = det(D) and the objective maximizes det(D).\n\n    Parameters\n    ----------\n    expr : log_det\n    args : list\n        The arguments for the expression\n\n    Returns\n    -------\n    tuple\n        (Variable for objective, list of constraints)\n    '
    A = args[0]
    (n, _) = A.shape
    z = Variable(shape=(n * (n + 1) // 2,))
    Z = vec_to_upper_tri(z, strict=False)
    d = diag_mat(Z)
    D = diag_vec(d)
    X = bmat([[D, Z], [Z.T, A]])
    constraints = [PSD(X)]
    log_expr = log(d)
    (obj, constr) = log_canon(log_expr, log_expr.args)
    constraints += constr
    return (sum(obj), constraints)