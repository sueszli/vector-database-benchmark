"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from cvxpy.atoms.affine.diag import diag
from cvxpy.atoms.affine.vec import vec
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.utilities.perspective_utils import form_cone_constraint

def perspective_canon(expr, args):
    if False:
        i = 10
        return i + 15
    from cvxpy.problems.problem import Problem
    aux_prob = Problem((Minimize if expr.f.is_convex() else Maximize)(expr.f))
    solver_opts = {'use_quad_obj': False}
    chain = aux_prob._construct_chain(solver_opts=solver_opts, ignore_dpp=True)
    chain.reductions = chain.reductions[:-1]
    prob_canon = chain.apply(aux_prob)[0]
    c = prob_canon.c.toarray().flatten()[:-1]
    d = prob_canon.c.toarray().flatten()[-1]
    Ab = prob_canon.A.toarray().reshape((-1, len(c) + 1), order='F')
    (A, b) = (Ab[:, :-1], Ab[:, -1])
    t = Variable()
    s = args[0]
    x_canon = prob_canon.x
    constraints = []
    if A.shape[0] > 0:
        x_pers = A @ x_canon + s * b
        i = 0
        for con in prob_canon.constraints:
            sz = con.size
            var_slice = x_pers[i:i + sz]
            pers_constraint = form_cone_constraint(var_slice, con)
            constraints.append(pers_constraint)
            i += sz
    constraints.append(-c @ x_canon + t - s * d >= 0)
    end_inds = sorted(prob_canon.var_id_to_col.values()) + [x_canon.shape[0]]
    for var in expr.f.variables():
        start_ind = prob_canon.var_id_to_col[var.id]
        end_ind = end_inds[end_inds.index(start_ind) + 1]
        if var.attributes['diag']:
            constraints += [diag(var) == x_canon[start_ind:end_ind]]
        elif var.is_symmetric() and var.size > 1:
            n = var.shape[0]
            inds = np.triu_indices(n, k=0)
            constraints += [var[inds] == x_canon[start_ind:end_ind]]
        else:
            constraints.append(vec(var) == x_canon[start_ind:end_ind])
    return ((1 if expr.f.is_convex() else -1) * t, constraints)