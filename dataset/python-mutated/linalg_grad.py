"""Gradients for operators defined in linalg_ops.py.

Useful reference for derivative formulas is (Mike Giles, 2008).

Ionescu et al. (2015) provide a detailed derivation of formulas for
backpropagating through spectral layers (SVD and Eig).

References:
  An extended collection of matrix derivative results for
  forward and reverse mode automatic differentiation:
    [Mike Giles, 2008]
    (https://ora.ox.ac.uk/objects/uuid:8d0c0a29-c92b-4153-a1d2-38b276e93124)
    ([pdf](http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf))
  Matrix Backpropagation for Deep Networks with Structured Layers
    [Ionescu et al., 2015]
    (https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.html)
    ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.pdf))
  Training Deep Networks with Structured Layers by Matrix Backpropagation:
    [Ionescu et al., 2015](https://arxiv.org/abs/1509.07838)
    ([pdf](https://arxiv.org/pdf/1509.07838.pdf))
"""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg

@ops.RegisterGradient('MatrixInverse')
def _MatrixInverseGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for MatrixInverse.'
    ainv = op.outputs[0]
    op_adjoint = op.get_attr('adjoint')
    return -math_ops.matmul(ainv, math_ops.matmul(grad, ainv, adjoint_a=op_adjoint, adjoint_b=not op_adjoint), adjoint_a=not op_adjoint)

@ops.RegisterGradient('Einsum')
def _EinsumGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Gradient for Einsum.'
    ellipsis = '...'

    def _GetAxisFromLabel(subscripts, label):
        if False:
            for i in range(10):
                print('nop')
        'Returns the axis (possibly negative) corresponding to a label.\n\n    Returns the axis index of the axis label if it is before an ellipsis (or if\n    the ellipsis is not present), and the negative index if it occurs after the\n    ellipsis. E.g. index of `b` in `ab...cd`, is `1`, but that of `c` is `-2`.\n\n    For multiple occurrences, returns the leftmost one. If not found, returns\n    None.\n\n    Args:\n      subscripts: A string denoting the einsum subscript (e.g. `ab...cd`)\n      label: The single character axis label.\n    '
        splits = subscripts.split(ellipsis)
        index = splits[0].find(label)
        if index != -1:
            return index
        if len(splits) < 2:
            return None
        index = splits[1].find(label)
        if index != -1:
            return index - len(splits[1])
        return None

    def _GetBcastSubshape(subscripts):
        if False:
            return 10
        'Returns a tuple denoting the slice mapping to ellipsis.\n\n    For a given subscript, returns a tuple (start, end) denoting the start\n    axis index and the (negative) end axis index respectively. For any input\n    Tensor `x` described by the subscript, `x[start:end]` would be the slice\n    represented by the ellipsis. E.g. For `ab...cd` returns `[1, -2]`.\n\n    If ellipsis is not present in `subscripts`, returns `(0, 0)`.\n\n    Args:\n      subscripts: A string denoting the einsum subscript.\n    '
        start = subscripts.find(ellipsis)
        if start == -1:
            return (0, 0)
        remaining = len(subscripts) - (start + len(ellipsis))
        end = -remaining if remaining > 0 else None
        return (start, end)

    def _GetReducedSubscripts(reduced_label_set, input_shape, subscripts):
        if False:
            i = 10
            return i + 15
        'Returns reduced subscripts and their corresponding dimensions and axes.\n\n    Given a set of axis labels, returns their concatenated subscript, their\n    corresponding dimensions from input_shape, and their corresponding axes.\n    Note that the concatenated subscript `reduced_subs` may have axis labels\n    from `reduced_label_set` in any order. For example, for the reduced label\n    set `{b, d}`, subscripts `aabbcd` and input shape `[2,2,5,5,3,4]`, returns\n    subscripts `bd`, dimensions `[5,4]` and axes `[2,5]`.\n\n    Args:\n      reduced_label_set: Set of axis labels which appear in `subscripts`.\n      input_shape: A `Tensor` representing the shape of the einsum operand\n        corresponding to `subscripts`.\n      subscripts: A string denoting the einsum subscript.\n\n    Returns:\n      reduced_subs: Subscripts formed by a concatenation of labels in\n        `reduced_label_set`.\n      reduced_dims: Dimensions from `input_shape` corresponding to each label\n        in `reduced_subs`.\n      reduced_axes: Axes described by `subscripts` corresponding to each label\n        in `reduced_subs`. If there are multiple occurrences in `subscripts`,\n        we consider only the leftmost one.\n\n    '
        reduced_subs = ''.join(list(reduced_label_set))
        reduced_axes = [_GetAxisFromLabel(subscripts, s) for s in reduced_subs]
        reduced_dims = array_ops_stack.stack([input_shape[ax] for ax in reduced_axes])
        return (reduced_subs, reduced_dims, reduced_axes)

    def _GetGradReduced(output_grad, output_subs, input_subs, input_shape, reduced_label_set):
        if False:
            while True:
                i = 10
        'Returns the gradient wrt input for a unary einsum with reductions.\n\n    Args:\n      output_grad: The gradient wrt the output of a unary einsum operation.\n      output_subs: The output subscript. (E.g. `ac` for equation `abc->ac`).\n      input_subs: The input subscript. (E.g. `abc` for equation `abc->ac`).\n      input_shape: A `Tensor` representing the shape of the input operand.\n      reduced_label_set: The set of axis labels appearing in `input_subs` but\n        not in `output_subs`.\n    '
        (reduced_subs, reduced_dims, reduced_axes) = _GetReducedSubscripts(reduced_label_set, input_shape, input_subs)
        has_repeated_labels = len(set(input_subs)) + len(set(output_subs)) < len(input_subs) + len(output_subs)
        input_subs_without_reduced_labels = ''.join([s for s in input_subs if s not in reduced_label_set])
        if not has_repeated_labels and input_subs_without_reduced_labels == output_subs:
            reduced_shape = math_ops.reduced_shape(input_shape, ops.convert_to_tensor(reduced_axes))
            return array_ops.broadcast_to(array_ops.reshape(output_grad, reduced_shape), input_shape)
        grad_shape_with_reduced_labels = array_ops.concat([reduced_dims, array_ops.shape(output_grad)], axis=0)
        reduced_shape = array_ops.concat([array_ops.ones(len(reduced_label_set), dtype=dtypes.int32), array_ops.shape(output_grad)], axis=0)
        broadcasted_grad = array_ops.broadcast_to(array_ops.reshape(output_grad, reduced_shape), grad_shape_with_reduced_labels)
        return gen_linalg_ops.einsum([broadcasted_grad], '{}->{}'.format(reduced_subs + output_subs, input_subs))

    def _GetGradWrt(output_grad, other_operand, input_shape, input_subs, other_subs, output_subs):
        if False:
            return 10
        'Returns the gradient wrt an input operand for a binary einsum.\n\n    This function does not handle (un)broadcasting. This must be done separately\n    on the returned gradient.\n\n    Args:\n      output_grad: The gradient wrt the output of a binary einsum operation.\n      other_operand: The complementary `Tensor` operand i.e. which is not the\n        input operand.\n      input_shape: A `Tensor` representing the shape of input operand.\n      input_subs: The subscripts of the input operand.\n      other_subs: The subscripts of the complementary operand.\n      output_subs: The output subscripts.\n    '
        reduced_label_set = set(input_subs).difference(set(output_subs + other_subs + '.'))
        left_subs = ''.join((s for s in input_subs if s not in reduced_label_set))
        grad_reduced = gen_linalg_ops.einsum([output_grad, other_operand], '{},{}->{}'.format(output_subs, other_subs, left_subs))
        if not reduced_label_set:
            return grad_reduced
        return _GetGradReduced(grad_reduced, left_subs, input_subs, input_shape, reduced_label_set)
    equation = op.get_attr('equation')
    if isinstance(equation, bytes):
        equation = equation.decode()
    (input_subs, output_subs) = equation.split('->')
    if len(op.inputs) == 1:
        input_shape = array_ops.shape(op.inputs[0])
        reduced_label_set = set(input_subs).difference(set(output_subs + ellipsis))
        if not reduced_label_set:
            return gen_linalg_ops.einsum([grad], '{}->{}'.format(output_subs, input_subs))
        return _GetGradReduced(grad, output_subs, input_subs, input_shape, reduced_label_set)
    (x_subs, y_subs) = input_subs.split(',')
    if ellipsis in output_subs:
        if ellipsis not in x_subs:
            x_subs += ellipsis
        if ellipsis not in y_subs:
            y_subs += ellipsis
    (x, y) = (op.inputs[0], op.inputs[1])
    if grad.dtype.is_complex:
        x = math_ops.conj(x)
        y = math_ops.conj(y)
    x_shape = array_ops.shape(x)
    y_shape = array_ops.shape(y)
    grad_x = _GetGradWrt(grad, y, x_shape, x_subs, y_subs, output_subs)
    grad_y = _GetGradWrt(grad, x, y_shape, y_subs, x_subs, output_subs)
    if ellipsis not in output_subs:
        return (grad_x, grad_y)
    (bx_start, bx_end) = _GetBcastSubshape(x_subs)
    (by_start, by_end) = _GetBcastSubshape(y_subs)
    x_shape_static = x.get_shape()
    y_shape_static = y.get_shape()
    if x_shape_static.is_fully_defined() and y_shape_static.is_fully_defined() and (x_shape_static[bx_start:bx_end] == y_shape_static[by_start:by_end]):
        return (grad_x, grad_y)
    (rx, ry) = array_ops.broadcast_gradient_args(x_shape[bx_start:bx_end], y_shape[by_start:by_end])
    grad_x = array_ops.reshape(math_ops.reduce_sum(grad_x, bx_start + rx), x_shape)
    grad_y = array_ops.reshape(math_ops.reduce_sum(grad_y, by_start + ry), y_shape)
    return (grad_x, grad_y)

@ops.RegisterGradient('MatrixDeterminant')
def _MatrixDeterminantGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for MatrixDeterminant.'
    a = op.inputs[0]
    c = op.outputs[0]
    a_adj_inv = linalg_ops.matrix_inverse(a, adjoint=True)
    multipliers = array_ops.reshape(grad * c, array_ops.concat([array_ops.shape(c), [1, 1]], 0))
    return multipliers * a_adj_inv

@ops.RegisterGradient('MatrixSquareRoot')
def _MatrixSquareRootGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for MatrixSquareRoot.'

    def _KroneckerProduct(b1, b2):
        if False:
            i = 10
            return i + 15
        'Computes the Kronecker product of two batches of square matrices.'
        b1_shape = array_ops.shape(b1)
        b2_shape = array_ops.shape(b2)
        b1_order = b1_shape[-1]
        b2_order = b2_shape[-1]
        shape_slice_size = [math_ops.subtract(array_ops.size(b1_shape), 2)]
        shape_slice = array_ops.slice(b1_shape, [0], shape_slice_size)
        b1_reshape_shape = array_ops.concat([shape_slice, [b1_order], [1], [b1_order], [1]], 0)
        b2_reshape_shape = array_ops.concat([shape_slice, [1], [b2_order], [1], [b2_order]], 0)
        b1_reshape = array_ops.reshape(b1, b1_reshape_shape)
        b2_reshape = array_ops.reshape(b2, b2_reshape_shape)
        order_prod = b1_order * b2_order
        kprod_shape = array_ops.concat([shape_slice, [order_prod], [order_prod]], 0)
        return array_ops.reshape(b1_reshape * b2_reshape, kprod_shape)
    sqrtm = op.outputs[0]
    shape = array_ops.shape(sqrtm)
    order = shape[-1]
    matrix_count = math_ops.reduce_prod(shape[0:-2])
    eye = linalg_ops.eye(order, dtype=sqrtm.dtype)
    eye_flat = array_ops.reshape(eye, [-1])
    eye_tiled = array_ops.tile(eye_flat, [matrix_count])
    eye_batch = array_ops.reshape(eye_tiled, shape)
    sqrtm_transpose = array_ops.matrix_transpose(sqrtm)
    k1 = _KroneckerProduct(eye_batch, sqrtm_transpose)
    k2 = _KroneckerProduct(sqrtm, eye_batch)
    ksum = math_ops.add(k1, k2)
    shape_slice_size = [math_ops.subtract(array_ops.size(shape), 2)]
    shape_slice = array_ops.slice(shape, [0], shape_slice_size)
    shape_vec_da = array_ops.concat([shape_slice, [order * order], [1]], 0)
    vec_da = array_ops.reshape(array_ops.matrix_transpose(grad), shape_vec_da)
    vec_dsqrtm = linalg_ops.matrix_solve(ksum, vec_da)
    dsqrtm_transpose = array_ops.reshape(vec_dsqrtm, shape)
    return array_ops.matrix_transpose(dsqrtm_transpose)

@ops.RegisterGradient('LogMatrixDeterminant')
def _LogMatrixDeterminantGrad(op: ops.Operation, _, grad_b):
    if False:
        while True:
            i = 10
    'Gradient for LogMatrixDeterminant.'
    a = op.inputs[0]
    c = op.outputs[1]
    a_adj_inv = linalg_ops.matrix_inverse(a, adjoint=True)
    multipliers = array_ops.reshape(grad_b, array_ops.concat([array_ops.shape(c), [1, 1]], 0))
    return multipliers * a_adj_inv

@ops.RegisterGradient('Cholesky')
def _CholeskyGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for Cholesky.'
    l = op.outputs[0]
    num_rows = array_ops.shape(l)[-1]
    batch_shape = array_ops.shape(l)[:-2]
    l_inverse = linalg_ops.matrix_triangular_solve(l, linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=l.dtype))
    middle = math_ops.matmul(l, grad, adjoint_a=True)
    middle = array_ops.matrix_set_diag(middle, 0.5 * array_ops.matrix_diag_part(middle))
    middle = array_ops.matrix_band_part(middle, -1, 0)
    grad_a = math_ops.matmul(math_ops.matmul(l_inverse, middle, adjoint_a=True), l_inverse)
    grad_a += _linalg.adjoint(grad_a)
    return grad_a * 0.5

@ops.RegisterGradient('Qr')
def _QrGrad(op: ops.Operation, dq, dr):
    if False:
        i = 10
        return i + 15
    'Gradient for Qr.'
    (q, r) = op.outputs
    if r.shape.ndims is None or r.shape.as_list()[-2] is None or r.shape.as_list()[-1] is None:
        raise NotImplementedError(f'QrGrad not implemented with dynamic shapes. Received r.shape: {r.shape}')
    if r.shape.dims[-2].value > r.shape.dims[-1].value and q.shape.dims[-2].value == q.shape.dims[-1].value:
        raise NotImplementedError(f'QrGrad not implemented when nrows > ncols and full_matrices is true. Received r.shape={r.shape} with nrows={r.shape.dims[-2]}and ncols={r.shape.dims[-1]}.')

    def _TriangularSolve(x, r):
        if False:
            return 10
        'Equiv to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri.'
        return _linalg.adjoint(linalg_ops.matrix_triangular_solve(r, _linalg.adjoint(x), lower=False, adjoint=False))

    def _QrGradSquareAndDeepMatrices(q, r, dq, dr):
        if False:
            i = 10
            return i + 15
        'Gradient for matrix orders num_rows >= num_cols\n    and full_matrices is false.\n    '
        qdq = math_ops.matmul(q, dq, adjoint_a=True)
        qdq_ = qdq - _linalg.adjoint(qdq)
        rdr = math_ops.matmul(r, dr, adjoint_b=True)
        rdr_ = rdr - _linalg.adjoint(rdr)
        tril = array_ops.matrix_band_part(qdq_ + rdr_, -1, 0)
        grad_a = math_ops.matmul(q, dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - math_ops.matmul(q, qdq), r)
        ret = grad_a + grad_b
        if q.dtype.is_complex:
            m = rdr - _linalg.adjoint(qdq)
            eyem = _linalg.set_diag(array_ops.zeros_like(m), _linalg.diag_part(m))
            correction = eyem - math_ops.cast(math_ops.real(eyem), q.dtype)
            ret = ret + _TriangularSolve(math_ops.matmul(q, _linalg.adjoint(correction)), r)
        return ret
    (num_rows, num_cols) = (q.shape.dims[-2].value, r.shape.dims[-1])
    if num_rows >= num_cols:
        return _QrGradSquareAndDeepMatrices(q, r, dq, dr)
    a = op.inputs[0]
    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = math_ops.matmul(q, dv)
    dx = _QrGradSquareAndDeepMatrices(q, u, dq + math_ops.matmul(y, dv, adjoint_b=True), du)
    return array_ops.concat([dx, dy], axis=-1)

@ops.RegisterGradient('MatrixSolve')
def _MatrixSolveGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for MatrixSolve.'
    a = op.inputs[0]
    adjoint_a = op.get_attr('adjoint')
    c = op.outputs[0]
    grad_b = linalg_ops.matrix_solve(a, grad, adjoint=not adjoint_a)
    if adjoint_a:
        grad_a = -math_ops.matmul(c, grad_b, adjoint_b=True)
    else:
        grad_a = -math_ops.matmul(grad_b, c, adjoint_b=True)
    return (grad_a, grad_b)

@ops.RegisterGradient('MatrixSolveLs')
def _MatrixSolveLsGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Gradients for MatrixSolveLs.'

    def _Overdetermined(op: ops.Operation, grad):
        if False:
            i = 10
            return i + 15
        'Gradients for the overdetermined case of MatrixSolveLs.\n\n    This is the backprop for the solution to the normal equations of the first\n    kind:\n       X = F(A, B) = (A^T * A + lambda * I)^{-1} * A^T * B\n    which solve the least squares problem\n       min ||A * X - B||_F^2 + lambda ||X||_F^2.\n    '
        a = op.inputs[0]
        b = op.inputs[1]
        x = op.outputs[0]
        l2_regularizer = math_ops.cast(op.inputs[2], a.dtype.base_dtype)
        chol = linalg_ops._RegularizedGramianCholesky(a, l2_regularizer=l2_regularizer, first_kind=True)
        z = linalg_ops.cholesky_solve(chol, grad)
        xzt = math_ops.matmul(x, z, adjoint_b=True)
        zx_sym = xzt + array_ops.matrix_transpose(xzt)
        grad_a = -math_ops.matmul(a, zx_sym) + math_ops.matmul(b, z, adjoint_b=True)
        grad_b = math_ops.matmul(a, z)
        return (grad_a, grad_b, None)

    def _Underdetermined(op: ops.Operation, grad):
        if False:
            while True:
                i = 10
        'Gradients for the underdetermined case of MatrixSolveLs.\n\n    This is the backprop for the solution to the normal equations of the second\n    kind:\n      X = F(A, B) = A * (A*A^T + lambda*I)^{-1} * B\n    that (for lambda=0) solve the least squares problem\n      min ||X||_F subject to A*X = B.\n    '
        a = op.inputs[0]
        b = op.inputs[1]
        l2_regularizer = math_ops.cast(op.inputs[2], a.dtype.base_dtype)
        chol = linalg_ops._RegularizedGramianCholesky(a, l2_regularizer=l2_regularizer, first_kind=False)
        grad_b = linalg_ops.cholesky_solve(chol, math_ops.matmul(a, grad))
        tmp = linalg_ops.cholesky_solve(chol, b)
        a1 = math_ops.matmul(tmp, a, adjoint_a=True)
        a1 = -math_ops.matmul(grad_b, a1)
        a2 = grad - math_ops.matmul(a, grad_b, adjoint_a=True)
        a2 = math_ops.matmul(tmp, a2, adjoint_b=True)
        grad_a = a1 + a2
        return (grad_a, grad_b, None)
    fast = op.get_attr('fast')
    if fast is False:
        raise ValueError('Gradient not defined for fast=False')
    matrix_shape = op.inputs[0].get_shape()[-2:]
    if matrix_shape.is_fully_defined():
        if matrix_shape[-2] >= matrix_shape[-1]:
            return _Overdetermined(op, grad)
        else:
            return _Underdetermined(op, grad)
    else:
        matrix_shape = array_ops.shape(op.inputs[0])[-2:]
        return cond.cond(matrix_shape[-2] >= matrix_shape[-1], lambda : _Overdetermined(op, grad), lambda : _Underdetermined(op, grad))

@ops.RegisterGradient('BandedTriangularSolve')
def _BandedTriangularSolveGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for BandedTriangularSolve.'
    a = op.inputs[0]
    b = op.inputs[1]
    num_bands = array_ops.shape(a)[-2]
    adjoint_a = op.get_attr('adjoint')
    lower_a = op.get_attr('lower')
    c = op.outputs[0]
    grad_b = linalg_ops.banded_triangular_solve(a, grad, lower=lower_a, adjoint=not adjoint_a)
    if adjoint_a:
        grad_a = -math_ops.matmul(c, grad_b, adjoint_b=True)
    else:
        grad_a = -math_ops.matmul(grad_b, c, adjoint_b=True)
    if lower_a:
        grad_a = array_ops.matrix_diag_part(grad_a, k=(-(num_bands - 1), 0), align='LEFT_RIGHT')
    else:
        grad_a = array_ops.matrix_diag_part(grad_a, k=(0, num_bands - 1), align='LEFT_RIGHT')
    if a.shape.is_fully_defined() and b.shape.is_fully_defined() and (a.shape[:-2] == b.shape[:-2]):
        return (grad_a, grad_b)
    a_shape = array_ops.shape(a)
    b_shape = array_ops.shape(b)
    (ra, rb) = array_ops.broadcast_gradient_args(a_shape[:-2], b_shape[:-2])
    grad_a = array_ops.reshape(math_ops.reduce_sum(grad_a, axis=ra), a_shape)
    grad_b = array_ops.reshape(math_ops.reduce_sum(grad_b, axis=rb), b_shape)
    return (grad_a, grad_b)

@ops.RegisterGradient('MatrixTriangularSolve')
def _MatrixTriangularSolveGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for MatrixTriangularSolve.'
    a = op.inputs[0]
    b = op.inputs[1]
    adjoint_a = op.get_attr('adjoint')
    lower_a = op.get_attr('lower')
    c = op.outputs[0]
    grad_b = linalg_ops.matrix_triangular_solve(a, grad, lower=lower_a, adjoint=not adjoint_a)
    if adjoint_a:
        grad_a = -math_ops.matmul(c, grad_b, adjoint_b=True)
    else:
        grad_a = -math_ops.matmul(grad_b, c, adjoint_b=True)
    if lower_a:
        grad_a = array_ops.matrix_band_part(grad_a, -1, 0)
    else:
        grad_a = array_ops.matrix_band_part(grad_a, 0, -1)
    if a.shape.is_fully_defined() and b.shape.is_fully_defined() and (a.shape[:-2] == b.shape[:-2]):
        return (grad_a, grad_b)
    a_shape = array_ops.shape(a)
    b_shape = array_ops.shape(b)
    (ra, rb) = array_ops.broadcast_gradient_args(a_shape[:-2], b_shape[:-2])
    grad_a = array_ops.reshape(math_ops.reduce_sum(grad_a, axis=ra), a_shape)
    grad_b = array_ops.reshape(math_ops.reduce_sum(grad_b, axis=rb), b_shape)
    return (grad_a, grad_b)

def _SafeReciprocal(x, epsilon=1e-20):
    if False:
        print('Hello World!')
    return x * math_ops.reciprocal(x * x + epsilon)

@ops.RegisterGradient('Eig')
def _EigGrad(op: ops.Operation, grad_e, grad_v):
    if False:
        print('Hello World!')
    'Gradient for Eig.\n\n  Based on eq. 4.77 from paper by\n  Christoph Boeddeker et al.\n  https://arxiv.org/abs/1701.00392\n  See also\n  "Computation of eigenvalue and eigenvector derivatives\n  for a general complex-valued eigensystem" by Nico van der Aa.\n  As for now only distinct eigenvalue case is considered.\n  '
    e = op.outputs[0]
    compute_v = op.get_attr('compute_v')
    with ops.control_dependencies([grad_e, grad_v]):
        if compute_v:
            v = op.outputs[1]
            vt = _linalg.adjoint(v)
            f = array_ops.matrix_set_diag(_SafeReciprocal(array_ops.expand_dims(e, -2) - array_ops.expand_dims(e, -1)), array_ops.zeros_like(e))
            f = math_ops.conj(f)
            vgv = math_ops.matmul(vt, grad_v)
            mid = array_ops.matrix_diag(grad_e)
            diag_grad_part = array_ops.matrix_diag(array_ops.matrix_diag_part(math_ops.cast(math_ops.real(vgv), vgv.dtype)))
            mid += f * (vgv - math_ops.matmul(math_ops.matmul(vt, v), diag_grad_part))
            grad_a = linalg_ops.matrix_solve(vt, math_ops.matmul(mid, vt))
        else:
            (_, v) = linalg_ops.eig(op.inputs[0])
            vt = _linalg.adjoint(v)
            grad_a = linalg_ops.matrix_solve(vt, math_ops.matmul(array_ops.matrix_diag(grad_e), vt))
        return math_ops.cast(grad_a, op.inputs[0].dtype)

@ops.RegisterGradient('SelfAdjointEigV2')
def _SelfAdjointEigV2Grad(op: ops.Operation, grad_e, grad_v):
    if False:
        i = 10
        return i + 15
    'Gradient for SelfAdjointEigV2.'
    e = op.outputs[0]
    compute_v = op.get_attr('compute_v')
    with ops.control_dependencies([grad_e, grad_v]):
        if compute_v:
            v = op.outputs[1]
            f = array_ops.matrix_set_diag(_SafeReciprocal(array_ops.expand_dims(e, -2) - array_ops.expand_dims(e, -1)), array_ops.zeros_like(e))
            grad_a = math_ops.matmul(v, math_ops.matmul(array_ops.matrix_diag(grad_e) + f * math_ops.matmul(v, grad_v, adjoint_a=True), v, adjoint_b=True))
        else:
            (_, v) = linalg_ops.self_adjoint_eig(op.inputs[0])
            grad_a = math_ops.matmul(v, math_ops.matmul(array_ops.matrix_diag(grad_e), v, adjoint_b=True))
        grad_a = array_ops.matrix_band_part(grad_a + _linalg.adjoint(grad_a), -1, 0)
        grad_a = array_ops.matrix_set_diag(grad_a, 0.5 * array_ops.matrix_diag_part(grad_a))
        return grad_a

@ops.RegisterGradient('Svd')
def _SvdGrad(op: ops.Operation, grad_s, grad_u, grad_v):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for the singular value decomposition.'
    a = op.inputs[0]
    a_shape = a.get_shape().with_rank_at_least(2)
    grad_s = math_ops.cast(grad_s, a.dtype)
    grad_s_mat = array_ops.matrix_diag(grad_s)
    if not op.get_attr('compute_uv'):
        (s, u, v) = linalg_ops.svd(a, compute_uv=True)
        grad_a = math_ops.matmul(u, math_ops.matmul(grad_s_mat, v, adjoint_b=True))
        grad_a.set_shape(a_shape)
        return grad_a
    full_matrices = op.get_attr('full_matrices')
    grad_u_shape = grad_u.get_shape().with_rank_at_least(2)
    grad_v_shape = grad_v.get_shape().with_rank_at_least(2)
    m = a_shape.dims[-2].merge_with(grad_u_shape[-2])
    n = a_shape.dims[-1].merge_with(grad_v_shape[-2])
    batch_shape = a_shape[:-2].merge_with(grad_u_shape[:-2]).merge_with(grad_v_shape[:-2])
    a_shape = batch_shape.concatenate([m, n])
    m = a_shape.dims[-2].value
    n = a_shape.dims[-1].value
    if m is None or n is None:
        raise NotImplementedError('SVD gradient has not been implemented for input with unknown inner matrix shape.')
    s = op.outputs[0]
    u = op.outputs[1]
    v = op.outputs[2]
    s = math_ops.cast(s, a.dtype)
    use_adjoint = False
    if m > n:
        use_adjoint = True
        (m, n) = (n, m)
        (u, v) = (v, u)
        (grad_u, grad_v) = (grad_v, grad_u)
    with ops.control_dependencies([grad_s, grad_u, grad_v]):
        if full_matrices and abs(m - n) > 1:
            raise NotImplementedError(f'svd gradient is not implemented for abs(m - n) > 1 when full_matrices is True. Received: m={m} and n={n} from op input={a} with shape={a_shape}.')
        s_mat = array_ops.matrix_diag(s)
        s2 = math_ops.square(s)
        s_shape = array_ops.shape(s)
        f = array_ops.matrix_set_diag(_SafeReciprocal(array_ops.expand_dims(s2, -2) - array_ops.expand_dims(s2, -1)), array_ops.zeros_like(s))
        s_inv_mat = array_ops.matrix_diag(_SafeReciprocal(s))
        v1 = v[..., :, :m]
        grad_v1 = grad_v[..., :, :m]
        u_gu = math_ops.matmul(u, grad_u, adjoint_a=True)
        v_gv = math_ops.matmul(v1, grad_v1, adjoint_a=True)
        f_u = f * u_gu
        f_v = f * v_gv
        term1_nouv = grad_s_mat + math_ops.matmul(f_u + _linalg.adjoint(f_u), s_mat) + math_ops.matmul(s_mat, f_v + _linalg.adjoint(f_v))
        term1 = math_ops.matmul(u, math_ops.matmul(term1_nouv, v1, adjoint_b=True))
        if m == n:
            grad_a_before_transpose = term1
        else:
            gv1t = array_ops.matrix_transpose(grad_v1, conjugate=True)
            gv1t_v1 = math_ops.matmul(gv1t, v1)
            term2_nous = gv1t - math_ops.matmul(gv1t_v1, v1, adjoint_b=True)
            if full_matrices:
                v2 = v[..., :, m:n]
                grad_v2 = grad_v[..., :, m:n]
                v1t_gv2 = math_ops.matmul(v1, grad_v2, adjoint_a=True)
                term2_nous -= math_ops.matmul(v1t_gv2, v2, adjoint_b=True)
            u_s_inv = math_ops.matmul(u, s_inv_mat)
            term2 = math_ops.matmul(u_s_inv, term2_nous)
            grad_a_before_transpose = term1 + term2
        if a.dtype.is_complex:
            eye = _linalg.eye(s_shape[-1], batch_shape=s_shape[:-1], dtype=a.dtype)
            l = eye * v_gv
            term3_nouv = math_ops.matmul(s_inv_mat, _linalg.adjoint(l) - l)
            term3 = 1 / 2.0 * math_ops.matmul(u, math_ops.matmul(term3_nouv, v1, adjoint_b=True))
            grad_a_before_transpose += term3
        if use_adjoint:
            grad_a = array_ops.matrix_transpose(grad_a_before_transpose, conjugate=True)
        else:
            grad_a = grad_a_before_transpose
        grad_a.set_shape(a_shape)
        return grad_a

def _LeftShift(x):
    if False:
        while True:
            i = 10
    'Shifts next-to-last dimension to the left, adding zero on the right.'
    rank = array_ops.rank(x)
    zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
    pad = array_ops.concat([zeros, array_ops.constant([[0, 1], [0, 0]])], axis=0)
    return array_ops.pad(x[..., 1:, :], pad)

def _RightShift(x):
    if False:
        return 10
    'Shifts next-to-last dimension to the right, adding zero on the left.'
    rank = array_ops.rank(x)
    zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
    pad = array_ops.concat([zeros, array_ops.constant([[1, 0], [0, 0]])], axis=0)
    return array_ops.pad(x[..., :-1, :], pad)

@ops.RegisterGradient('TridiagonalMatMul')
def _TridiagonalMatMulGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for TridiagonalMatMul.'
    superdiag_conj = array_ops.matrix_transpose(op.inputs[0], conjugate=True)
    maindiag_conj = array_ops.matrix_transpose(op.inputs[1], conjugate=True)
    subdiag_conj = array_ops.matrix_transpose(op.inputs[2], conjugate=True)
    rhs_conj = math_ops.conj(op.inputs[3])
    superdiag_grad = math_ops.reduce_sum(_LeftShift(rhs_conj) * grad, axis=-1)
    maindiag_grad = math_ops.reduce_sum(rhs_conj * grad, axis=-1)
    subdiag_grad = math_ops.reduce_sum(_RightShift(rhs_conj) * grad, axis=-1)
    rhs_grad = _RightShift(superdiag_conj * grad) + maindiag_conj * grad + _LeftShift(subdiag_conj * grad)
    superdiag_grad = array_ops.expand_dims(superdiag_grad, -2)
    maindiag_grad = array_ops.expand_dims(maindiag_grad, -2)
    subdiag_grad = array_ops.expand_dims(subdiag_grad, -2)
    return (superdiag_grad, maindiag_grad, subdiag_grad, rhs_grad)

@ops.RegisterGradient('TridiagonalSolve')
def _TridiagonalSolveGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for TridiagonalSolveGrad.'
    diags = op.inputs[0]
    x = op.outputs[0]
    partial_pivoting = op.get_attr('partial_pivoting')
    perturb_singular = op.get_attr('perturb_singular')
    diags_transposed = _TransposeTridiagonalMatrix(diags)
    grad_rhs = linalg_ops.tridiagonal_solve(diags_transposed, grad, partial_pivoting=partial_pivoting, perturb_singular=perturb_singular)
    grad_diags = -_MatmulExtractingThreeDiagonals(grad_rhs, x)
    return (grad_diags, grad_rhs)

def _TransposeTridiagonalMatrix(diags):
    if False:
        i = 10
        return i + 15
    'Transposes a tridiagonal matrix.\n\n  Args:\n    diags: the diagonals of the input matrix in the compact form (see\n      linalg_ops.tridiagonal_solve).\n\n  Returns:\n    Diagonals of the transposed matrix in the compact form.\n  '
    diag = diags[..., 1, :]
    if diags.shape.is_fully_defined():
        zeros = array_ops.zeros(list(diags.shape[:-2]) + [1], dtype=diags.dtype)
        superdiag = array_ops.concat((diags[..., 2, 1:], zeros), axis=-1)
        subdiag = array_ops.concat((zeros, diags[..., 0, :-1]), axis=-1)
    else:
        rank = array_ops.rank(diags)
        zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
        superdiag_pad = array_ops.concat((zeros, array_ops.constant([[0, 1]])), axis=0)
        superdiag = array_ops.pad(diags[..., 2, 1:], superdiag_pad)
        subdiag_pad = array_ops.concat((zeros, array_ops.constant([[1, 0]])), axis=0)
        subdiag = array_ops.pad(diags[..., 0, :-1], subdiag_pad)
    return array_ops_stack.stack([superdiag, diag, subdiag], axis=-2)

def _MatmulExtractingThreeDiagonals(x, y_tr):
    if False:
        return 10
    'Multiplies matrices and extracts three diagonals from the product.\n\n  With sizes M x K and K x M, this function takes O(MK) time and O(M) space,\n  while using math_ops.matmul, and then extracting the diagonals would take\n  O(M^2 K) time and O(M^2) space.\n\n  Args:\n    x: first matrix\n    y_tr: second matrix transposed\n\n  Returns:\n    Diagonals of the product in compact format (see\n    linalg_ops.tridiagonal_solve)\n\n  '
    diag = math_ops.reduce_sum(x * y_tr, axis=-1)
    if y_tr.shape.is_fully_defined():
        zeros = array_ops.zeros(list(x.shape[:-2]) + [1, x.shape[-1]], dtype=x.dtype)
        superdiag = math_ops.reduce_sum(x * array_ops.concat((y_tr[..., 1:, :], zeros), axis=-2), axis=-1)
        subdiag = math_ops.reduce_sum(x * array_ops.concat((zeros, y_tr[..., :-1, :]), axis=-2), axis=-1)
    else:
        rank = array_ops.rank(y_tr)
        zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
        superdiag_pad = array_ops.concat((zeros, array_ops.constant([[0, 1], [0, 0]])), axis=0)
        superdiag = math_ops.reduce_sum(x * array_ops.pad(y_tr[..., 1:, :], superdiag_pad), axis=-1)
        subdiag_pad = array_ops.concat((zeros, array_ops.constant([[1, 0], [0, 0]])), axis=0)
        subdiag = math_ops.reduce_sum(x * array_ops.pad(y_tr[..., :-1, :], subdiag_pad), axis=-1)
    return array_ops_stack.stack([superdiag, diag, subdiag], axis=-2)