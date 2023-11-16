"""Preconditioned Conjugate Gradient."""
import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('linalg.experimental.conjugate_gradient')
@dispatch.add_dispatch_support
def conjugate_gradient(operator, rhs, preconditioner=None, x=None, tol=1e-05, max_iter=20, name='conjugate_gradient'):
    if False:
        i = 10
        return i + 15
    'Conjugate gradient solver.\n\n  Solves a linear system of equations `A*x = rhs` for self-adjoint, positive\n  definite matrix `A` and right-hand side vector `rhs`, using an iterative,\n  matrix-free algorithm where the action of the matrix A is represented by\n  `operator`. The iteration terminates when either the number of iterations\n  exceeds `max_iter` or when the residual norm has been reduced to `tol`\n  times its initial value, i.e. \\\\(||rhs - A x_k|| <= tol ||rhs||\\\\).\n\n  Args:\n    operator: A `LinearOperator` that is self-adjoint and positive definite.\n    rhs: A possibly batched vector of shape `[..., N]` containing the right-hand\n      size vector.\n    preconditioner: A `LinearOperator` that approximates the inverse of `A`.\n      An efficient preconditioner could dramatically improve the rate of\n      convergence. If `preconditioner` represents matrix `M`(`M` approximates\n      `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate\n      `A^{-1}x`. For this to be useful, the cost of applying `M` should be\n      much lower than computing `A^{-1}` directly.\n    x: A possibly batched vector of shape `[..., N]` containing the initial\n      guess for the solution.\n    tol: A float scalar convergence tolerance.\n    max_iter: An integer giving the maximum number of iterations.\n    name: A name scope for the operation.\n\n  Returns:\n    output: A namedtuple representing the final state with fields:\n      - i: A scalar `int32` `Tensor`. Number of iterations executed.\n      - x: A rank-1 `Tensor` of shape `[..., N]` containing the computed\n          solution.\n      - r: A rank-1 `Tensor` of shape `[.., M]` containing the residual vector.\n      - p: A rank-1 `Tensor` of shape `[..., N]`. `A`-conjugate basis vector.\n      - gamma: \\\\(r \\dot M \\dot r\\\\), equivalent to  \\\\(||r||_2^2\\\\) when\n        `preconditioner=None`.\n  '
    if not (operator.is_self_adjoint and operator.is_positive_definite):
        raise ValueError('Expected a self-adjoint, positive definite operator.')
    cg_state = collections.namedtuple('CGState', ['i', 'x', 'r', 'p', 'gamma'])

    def stopping_criterion(i, state):
        if False:
            i = 10
            return i + 15
        return math_ops.logical_and(i < max_iter, math_ops.reduce_any(linalg.norm(state.r, axis=-1) > tol))

    def dot(x, y):
        if False:
            while True:
                i = 10
        return array_ops.squeeze(math_ops.matvec(x[..., array_ops.newaxis], y, adjoint_a=True), axis=-1)

    def cg_step(i, state):
        if False:
            while True:
                i = 10
        z = math_ops.matvec(operator, state.p)
        alpha = state.gamma / dot(state.p, z)
        x = state.x + alpha[..., array_ops.newaxis] * state.p
        r = state.r - alpha[..., array_ops.newaxis] * z
        if preconditioner is None:
            q = r
        else:
            q = preconditioner.matvec(r)
        gamma = dot(r, q)
        beta = gamma / state.gamma
        p = q + beta[..., array_ops.newaxis] * state.p
        return (i + 1, cg_state(i + 1, x, r, p, gamma))
    with ops.name_scope(name):
        broadcast_shape = array_ops.broadcast_dynamic_shape(array_ops.shape(rhs)[:-1], operator.batch_shape_tensor())
        if preconditioner is not None:
            broadcast_shape = array_ops.broadcast_dynamic_shape(broadcast_shape, preconditioner.batch_shape_tensor())
        broadcast_rhs_shape = array_ops.concat([broadcast_shape, [array_ops.shape(rhs)[-1]]], axis=-1)
        r0 = array_ops.broadcast_to(rhs, broadcast_rhs_shape)
        tol *= linalg.norm(r0, axis=-1)
        if x is None:
            x = array_ops.zeros(broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
        else:
            r0 = rhs - math_ops.matvec(operator, x)
        if preconditioner is None:
            p0 = r0
        else:
            p0 = math_ops.matvec(preconditioner, r0)
        gamma0 = dot(r0, p0)
        i = constant_op.constant(0, dtype=dtypes.int32)
        state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0)
        (_, state) = while_loop.while_loop(stopping_criterion, cg_step, [i, state])
        return cg_state(state.i, x=state.x, r=state.r, p=state.p, gamma=state.gamma)