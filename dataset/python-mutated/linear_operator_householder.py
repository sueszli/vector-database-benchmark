"""`LinearOperator` acting like a Householder transformation."""
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperatorHouseholder']

@tf_export('linalg.LinearOperatorHouseholder')
@linear_operator.make_composite_tensor
class LinearOperatorHouseholder(linear_operator.LinearOperator):
    """`LinearOperator` acting like a [batch] of Householder transformations.

  This operator acts like a [batch] of householder reflections with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorHouseholder` is initialized with a (batch) vector.

  A Householder reflection, defined via a vector `v`, which reflects points
  in `R^n` about the hyperplane orthogonal to `v` and through the origin.

  ```python
  # Create a 2 x 2 householder transform.
  vec = [1 / np.sqrt(2), 1. / np.sqrt(2)]
  operator = LinearOperatorHouseholder(vec)

  operator.to_dense()
  ==> [[0.,  -1.]
       [-1., -0.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

    def __init__(self, reflection_axis, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name='LinearOperatorHouseholder'):
        if False:
            print('Hello World!')
        'Initialize a `LinearOperatorHouseholder`.\n\n    Args:\n      reflection_axis:  Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.\n        The vector defining the hyperplane to reflect about.\n        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,\n        `complex128`.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  This is autoset to true\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices\n        This is autoset to false.\n      is_square:  Expect that this operator acts like square [batch] matrices.\n        This is autoset to true.\n      name: A name for this `LinearOperator`.\n\n    Raises:\n      ValueError:  `is_self_adjoint` is not `True`, `is_positive_definite` is\n        not `False` or `is_square` is not `True`.\n    '
        parameters = dict(reflection_axis=reflection_axis, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        with ops.name_scope(name, values=[reflection_axis]):
            self._reflection_axis = linear_operator_util.convert_nonref_to_tensor(reflection_axis, name='reflection_axis')
            self._check_reflection_axis(self._reflection_axis)
            if is_self_adjoint is False:
                raise ValueError('A Householder operator is always self adjoint.')
            else:
                is_self_adjoint = True
            if is_positive_definite is True:
                raise ValueError('A Householder operator is always non-positive definite.')
            else:
                is_positive_definite = False
            if is_square is False:
                raise ValueError('A Householder operator is always square.')
            is_square = True
            super(LinearOperatorHouseholder, self).__init__(dtype=self._reflection_axis.dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    def _check_reflection_axis(self, reflection_axis):
        if False:
            for i in range(10):
                print('nop')
        'Static check of reflection_axis.'
        if reflection_axis.shape.ndims is not None and reflection_axis.shape.ndims < 1:
            raise ValueError('Argument reflection_axis must have at least 1 dimension.  Found: %s' % reflection_axis)

    def _shape(self):
        if False:
            return 10
        d_shape = self._reflection_axis.shape
        return d_shape.concatenate(d_shape[-1:])

    def _shape_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        d_shape = array_ops.shape(self._reflection_axis)
        k = d_shape[-1]
        return array_ops.concat((d_shape, [k]), 0)

    def _assert_non_singular(self):
        if False:
            for i in range(10):
                print('nop')
        return control_flow_ops.no_op('assert_non_singular')

    def _assert_positive_definite(self):
        if False:
            print('Hello World!')
        raise errors.InvalidArgumentError(node_def=None, op=None, message='Householder operators are always non-positive definite.')

    def _assert_self_adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        return control_flow_ops.no_op('assert_self_adjoint')

    def _linop_adjoint(self) -> 'LinearOperatorHouseholder':
        if False:
            for i in range(10):
                print('nop')
        return self

    def _linop_inverse(self) -> 'LinearOperatorHouseholder':
        if False:
            return 10
        return self

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            i = 10
            return i + 15
        reflection_axis = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.reflection_axis)
        x = linalg.adjoint(x) if adjoint_arg else x
        normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
        mat = normalized_axis[..., array_ops.newaxis]
        x_dot_normalized_v = math_ops.matmul(mat, x, adjoint_a=True)
        return x - 2 * mat * x_dot_normalized_v

    def _trace(self):
        if False:
            print('Hello World!')
        shape = self.shape_tensor()
        return math_ops.cast(self._domain_dimension_tensor(shape=shape) - 2, self.dtype) * array_ops.ones(shape=self._batch_shape_tensor(shape=shape), dtype=self.dtype)

    def _determinant(self):
        if False:
            for i in range(10):
                print('nop')
        return -array_ops.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)

    def _log_abs_determinant(self):
        if False:
            for i in range(10):
                print('nop')
        return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            return 10
        return self._matmul(rhs, adjoint, adjoint_arg)

    def _to_dense(self):
        if False:
            i = 10
            return i + 15
        reflection_axis = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.reflection_axis)
        normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
        mat = normalized_axis[..., array_ops.newaxis]
        matrix = -2 * math_ops.matmul(mat, mat, adjoint_b=True)
        return array_ops.matrix_set_diag(matrix, 1.0 + array_ops.matrix_diag_part(matrix))

    def _diag_part(self):
        if False:
            return 10
        reflection_axis = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.reflection_axis)
        normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
        return 1.0 - 2 * normalized_axis * math_ops.conj(normalized_axis)

    def _eigvals(self):
        if False:
            for i in range(10):
                print('nop')
        result_shape = array_ops.shape(self.reflection_axis)
        n = result_shape[-1]
        ones_shape = array_ops.concat([result_shape[:-1], [n - 1]], axis=-1)
        neg_shape = array_ops.concat([result_shape[:-1], [1]], axis=-1)
        eigvals = array_ops.ones(shape=ones_shape, dtype=self.dtype)
        eigvals = array_ops.concat([-array_ops.ones(shape=neg_shape, dtype=self.dtype), eigvals], axis=-1)
        return eigvals

    def _cond(self):
        if False:
            i = 10
            return i + 15
        return array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)

    @property
    def reflection_axis(self):
        if False:
            for i in range(10):
                print('nop')
        return self._reflection_axis

    @property
    def _composite_tensor_fields(self):
        if False:
            while True:
                i = 10
        return ('reflection_axis',)

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            return 10
        return {'reflection_axis': 1}