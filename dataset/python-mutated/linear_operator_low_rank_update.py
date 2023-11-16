"""Perturb a `LinearOperator` with a rank `K` update."""
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperatorLowRankUpdate']

@tf_export('linalg.LinearOperatorLowRankUpdate')
@linear_operator.make_composite_tensor
class LinearOperatorLowRankUpdate(linear_operator.LinearOperator):
    """Perturb a `LinearOperator` with a rank `K` update.

  This operator acts like a [batch] matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `M x N` matrix.

  `LinearOperatorLowRankUpdate` represents `A = L + U D V^H`, where

  ```
  L, is a LinearOperator representing [batch] M x N matrices
  U, is a [batch] M x K matrix.  Typically K << M.
  D, is a [batch] K x K matrix.
  V, is a [batch] N x K matrix.  Typically K << N.
  V^H is the Hermitian transpose (adjoint) of V.
  ```

  If `M = N`, determinants and solves are done using the matrix determinant
  lemma and Woodbury identities, and thus require L and D to be non-singular.

  Solves and determinants will be attempted unless the "is_non_singular"
  property of L and D is False.

  In the event that L and D are positive-definite, and U = V, solves and
  determinants can be done using a Cholesky factorization.

  ```python
  # Create a 3 x 3 diagonal linear operator.
  diag_operator = LinearOperatorDiag(
      diag_update=[1., 2., 3.], is_non_singular=True, is_self_adjoint=True,
      is_positive_definite=True)

  # Perturb with a rank 2 perturbation
  operator = LinearOperatorLowRankUpdate(
      operator=diag_operator,
      u=[[1., 2.], [-1., 3.], [0., 0.]],
      diag_update=[11., 12.],
      v=[[1., 2.], [-1., 3.], [10., 10.]])

  operator.shape
  ==> [3, 3]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [3, 4] Tensor
  operator.matmul(x)
  ==> Shape [3, 4] Tensor
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  ### Performance

  Suppose `operator` is a `LinearOperatorLowRankUpdate` of shape `[M, N]`,
  made from a rank `K` update of `base_operator` which performs `.matmul(x)` on
  `x` having `x.shape = [N, R]` with `O(L_matmul*N*R)` complexity (and similarly
  for `solve`, `determinant`.  Then, if `x.shape = [N, R]`,

  * `operator.matmul(x)` is `O(L_matmul*N*R + K*N*R)`

  and if `M = N`,

  * `operator.solve(x)` is `O(L_matmul*N*R + N*K*R + K^2*R + K^3)`
  * `operator.determinant()` is `O(L_determinant + L_solve*N*K + K^2*N + K^3)`

  If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular`, `self_adjoint`, `positive_definite`,
  `diag_update_positive` and `square`. These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

    def __init__(self, base_operator, u, diag_update=None, v=None, is_diag_update_positive=None, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name='LinearOperatorLowRankUpdate'):
        if False:
            return 10
        'Initialize a `LinearOperatorLowRankUpdate`.\n\n    This creates a `LinearOperator` of the form `A = L + U D V^H`, with\n    `L` a `LinearOperator`, `U, V` both [batch] matrices, and `D` a [batch]\n    diagonal matrix.\n\n    If `L` is non-singular, solves and determinants are available.\n    Solves/determinants both involve a solve/determinant of a `K x K` system.\n    In the event that L and D are self-adjoint positive-definite, and U = V,\n    this can be done using a Cholesky factorization.  The user should set the\n    `is_X` matrix property hints, which will trigger the appropriate code path.\n\n    Args:\n      base_operator:  Shape `[B1,...,Bb, M, N]`.\n      u:  Shape `[B1,...,Bb, M, K]` `Tensor` of same `dtype` as `base_operator`.\n        This is `U` above.\n      diag_update:  Optional shape `[B1,...,Bb, K]` `Tensor` with same `dtype`\n        as `base_operator`.  This is the diagonal of `D` above.\n         Defaults to `D` being the identity operator.\n      v:  Optional `Tensor` of same `dtype` as `u` and shape `[B1,...,Bb, N, K]`\n         Defaults to `v = u`, in which case the perturbation is symmetric.\n         If `M != N`, then `v` must be set since the perturbation is not square.\n      is_diag_update_positive:  Python `bool`.\n        If `True`, expect `diag_update > 0`.\n      is_non_singular:  Expect that this operator is non-singular.\n        Default is `None`, unless `is_positive_definite` is auto-set to be\n        `True` (see below).\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  Default is `None`, unless `base_operator` is self-adjoint\n        and `v = None` (meaning `u=v`), in which case this defaults to `True`.\n      is_positive_definite:  Expect that this operator is positive definite.\n        Default is `None`, unless `base_operator` is positive-definite\n        `v = None` (meaning `u=v`), and `is_diag_update_positive`, in which case\n        this defaults to `True`.\n        Note that we say an operator is positive definite when the quadratic\n        form `x^H A x` has positive real part for all nonzero `x`.\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name: A name for this `LinearOperator`.\n\n    Raises:\n      ValueError:  If `is_X` flags are set in an inconsistent way.\n    '
        parameters = dict(base_operator=base_operator, u=u, diag_update=diag_update, v=v, is_diag_update_positive=is_diag_update_positive, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        dtype = base_operator.dtype
        if diag_update is not None:
            if is_diag_update_positive and dtype.is_complex:
                logging.warn('Note: setting is_diag_update_positive with a complex dtype means that diagonal is real and positive.')
        if diag_update is None:
            if is_diag_update_positive is False:
                raise ValueError("Default diagonal is the identity, which is positive.  However, user set 'is_diag_update_positive' to False.")
            is_diag_update_positive = True
        self._use_cholesky = base_operator.is_positive_definite and base_operator.is_self_adjoint and is_diag_update_positive and (v is None)
        if base_operator.is_self_adjoint and v is None and (not dtype.is_complex):
            if is_self_adjoint is False:
                raise ValueError('A = L + UDU^H, with L self-adjoint and D real diagonal.  Since UDU^H is self-adjoint, this must be a self-adjoint operator.')
            is_self_adjoint = True
        if self._use_cholesky:
            if is_positive_definite is False or is_self_adjoint is False or is_non_singular is False:
                raise ValueError('Arguments imply this is self-adjoint positive-definite operator.')
            is_positive_definite = True
            is_self_adjoint = True
        with ops.name_scope(name):
            self._u = linear_operator_util.convert_nonref_to_tensor(u, name='u')
            if v is None:
                self._v = self._u
            else:
                self._v = linear_operator_util.convert_nonref_to_tensor(v, name='v')
            if diag_update is None:
                self._diag_update = None
            else:
                self._diag_update = linear_operator_util.convert_nonref_to_tensor(diag_update, name='diag_update')
            self._base_operator = base_operator
            super(LinearOperatorLowRankUpdate, self).__init__(dtype=self._base_operator.dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)
            self._set_diag_operators(diag_update, is_diag_update_positive)
            self._is_diag_update_positive = is_diag_update_positive
            self._check_shapes()

    def _check_shapes(self):
        if False:
            for i in range(10):
                print('nop')
        'Static check that shapes are compatible.'
        uv_shape = array_ops.broadcast_static_shape(self.u.shape, self.v.shape)
        batch_shape = array_ops.broadcast_static_shape(self.base_operator.batch_shape, uv_shape[:-2])
        tensor_shape.Dimension(self.base_operator.domain_dimension).assert_is_compatible_with(uv_shape[-2])
        if self._diag_update is not None:
            tensor_shape.dimension_at_index(uv_shape, -1).assert_is_compatible_with(self._diag_update.shape[-1])
            array_ops.broadcast_static_shape(batch_shape, self._diag_update.shape[:-1])

    def _set_diag_operators(self, diag_update, is_diag_update_positive):
        if False:
            i = 10
            return i + 15
        'Set attributes self._diag_update and self._diag_operator.'
        if diag_update is not None:
            self._diag_operator = linear_operator_diag.LinearOperatorDiag(self._diag_update, is_positive_definite=is_diag_update_positive)
        else:
            if tensor_shape.dimension_value(self.u.shape[-1]) is not None:
                r = tensor_shape.dimension_value(self.u.shape[-1])
            else:
                r = array_ops.shape(self.u)[-1]
            self._diag_operator = linear_operator_identity.LinearOperatorIdentity(num_rows=r, dtype=self.dtype)

    @property
    def u(self):
        if False:
            for i in range(10):
                print('nop')
        'If this operator is `A = L + U D V^H`, this is the `U`.'
        return self._u

    @property
    def v(self):
        if False:
            print('Hello World!')
        'If this operator is `A = L + U D V^H`, this is the `V`.'
        return self._v

    @property
    def is_diag_update_positive(self):
        if False:
            for i in range(10):
                print('nop')
        'If this operator is `A = L + U D V^H`, this hints `D > 0` elementwise.'
        return self._is_diag_update_positive

    @property
    def diag_update(self):
        if False:
            print('Hello World!')
        'If this operator is `A = L + U D V^H`, this is the diagonal of `D`.'
        return self._diag_update

    @property
    def diag_operator(self):
        if False:
            for i in range(10):
                print('nop')
        'If this operator is `A = L + U D V^H`, this is `D`.'
        return self._diag_operator

    @property
    def base_operator(self):
        if False:
            print('Hello World!')
        'If this operator is `A = L + U D V^H`, this is the `L`.'
        return self._base_operator

    def _assert_self_adjoint(self):
        if False:
            return 10
        if self.u is self.v and self.diag_update is None:
            return self.base_operator.assert_self_adjoint()
        return super(LinearOperatorLowRankUpdate, self).assert_self_adjoint()

    def _shape(self):
        if False:
            for i in range(10):
                print('nop')
        batch_shape = array_ops.broadcast_static_shape(self.base_operator.batch_shape, self.diag_operator.batch_shape)
        batch_shape = array_ops.broadcast_static_shape(batch_shape, self.u.shape[:-2])
        batch_shape = array_ops.broadcast_static_shape(batch_shape, self.v.shape[:-2])
        return batch_shape.concatenate(self.base_operator.shape[-2:])

    def _shape_tensor(self):
        if False:
            while True:
                i = 10
        batch_shape = array_ops.broadcast_dynamic_shape(self.base_operator.batch_shape_tensor(), self.diag_operator.batch_shape_tensor())
        batch_shape = array_ops.broadcast_dynamic_shape(batch_shape, array_ops.shape(self.u)[:-2])
        batch_shape = array_ops.broadcast_dynamic_shape(batch_shape, array_ops.shape(self.v)[:-2])
        return array_ops.concat([batch_shape, self.base_operator.shape_tensor()[-2:]], axis=0)

    def _get_uv_as_tensors(self):
        if False:
            while True:
                i = 10
        'Get (self.u, self.v) as tensors (in case they were refs).'
        u = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.u)
        if self.v is self.u:
            v = u
        else:
            v = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.v)
        return (u, v)

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            for i in range(10):
                print('nop')
        (u, v) = self._get_uv_as_tensors()
        l = self.base_operator
        d = self.diag_operator
        leading_term = l.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
        if adjoint:
            uh_x = math_ops.matmul(u, x, adjoint_a=True, adjoint_b=adjoint_arg)
            d_uh_x = d.matmul(uh_x, adjoint=adjoint)
            v_d_uh_x = math_ops.matmul(v, d_uh_x)
            return leading_term + v_d_uh_x
        else:
            vh_x = math_ops.matmul(v, x, adjoint_a=True, adjoint_b=adjoint_arg)
            d_vh_x = d.matmul(vh_x, adjoint=adjoint)
            u_d_vh_x = math_ops.matmul(u, d_vh_x)
            return leading_term + u_d_vh_x

    def _determinant(self):
        if False:
            return 10
        if self.is_positive_definite:
            return math_ops.exp(self.log_abs_determinant())
        (u, v) = self._get_uv_as_tensors()
        det_c = linalg_ops.matrix_determinant(self._make_capacitance(u=u, v=v))
        det_d = self.diag_operator.determinant()
        det_l = self.base_operator.determinant()
        return det_c * det_d * det_l

    def _diag_part(self):
        if False:
            return 10
        (u, v) = self._get_uv_as_tensors()
        product = u * math_ops.conj(v)
        if self.diag_update is not None:
            product *= array_ops.expand_dims(self.diag_update, axis=-2)
        return math_ops.reduce_sum(product, axis=-1) + self.base_operator.diag_part()

    def _log_abs_determinant(self):
        if False:
            for i in range(10):
                print('nop')
        (u, v) = self._get_uv_as_tensors()
        log_abs_det_d = self.diag_operator.log_abs_determinant()
        log_abs_det_l = self.base_operator.log_abs_determinant()
        if self._use_cholesky:
            chol_cap_diag = array_ops.matrix_diag_part(linalg_ops.cholesky(self._make_capacitance(u=u, v=v)))
            log_abs_det_c = 2 * math_ops.reduce_sum(math_ops.log(chol_cap_diag), axis=[-1])
        else:
            det_c = linalg_ops.matrix_determinant(self._make_capacitance(u=u, v=v))
            log_abs_det_c = math_ops.log(math_ops.abs(det_c))
            if self.dtype.is_complex:
                log_abs_det_c = math_ops.cast(log_abs_det_c, dtype=self.dtype)
        return log_abs_det_c + log_abs_det_d + log_abs_det_l

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            return 10
        if self.base_operator.is_non_singular is False:
            raise ValueError('Solve not implemented unless this is a perturbation of a non-singular LinearOperator.')
        l = self.base_operator
        if adjoint:
            (v, u) = self._get_uv_as_tensors()
            capacitance = self._make_capacitance(u=v, v=u)
        else:
            (u, v) = self._get_uv_as_tensors()
            capacitance = self._make_capacitance(u=u, v=v)
        linv_rhs = l.solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        vh_linv_rhs = math_ops.matmul(v, linv_rhs, adjoint_a=True)
        if self._use_cholesky:
            capinv_vh_linv_rhs = linalg_ops.cholesky_solve(linalg_ops.cholesky(capacitance), vh_linv_rhs)
        else:
            capinv_vh_linv_rhs = linear_operator_util.matrix_solve_with_broadcast(capacitance, vh_linv_rhs, adjoint=adjoint)
        u_capinv_vh_linv_rhs = math_ops.matmul(u, capinv_vh_linv_rhs)
        linv_u_capinv_vh_linv_rhs = l.solve(u_capinv_vh_linv_rhs, adjoint=adjoint)
        return linv_rhs - linv_u_capinv_vh_linv_rhs

    def _make_capacitance(self, u, v):
        if False:
            return 10
        linv_u = self.base_operator.solve(u)
        vh_linv_u = math_ops.matmul(v, linv_u, adjoint_a=True)
        capacitance = self._diag_operator.inverse().add_to_tensor(vh_linv_u)
        return capacitance

    @property
    def _composite_tensor_fields(self):
        if False:
            for i in range(10):
                print('nop')
        return ('base_operator', 'u', 'diag_update', 'v', 'is_diag_update_positive')

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            i = 10
            return i + 15
        return {'base_operator': 0, 'u': 2, 'diag_update': 1, 'v': 2}