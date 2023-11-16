"""`LinearOperator` acting like a Toeplitz matrix."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperatorToeplitz']

@tf_export('linalg.LinearOperatorToeplitz')
@linear_operator.make_composite_tensor
class LinearOperatorToeplitz(linear_operator.LinearOperator):
    """`LinearOperator` acting like a [batch] of toeplitz matrices.

  This operator acts like a [batch] Toeplitz matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of toeplitz matrices

  Toeplitz means that `A` has constant diagonals. Hence, `A` can be generated
  with two vectors. One represents the first column of the matrix, and the
  other represents the first row.

  Below is a 4 x 4 example:

  ```
  A = |a b c d|
      |e a b c|
      |f e a b|
      |g f e a|
  ```

  #### Example of a Toeplitz operator.

  ```python
  # Create a 3 x 3 Toeplitz operator.
  col = [1., 2., 3.]
  row = [1., 4., -9.]
  operator = LinearOperatorToeplitz(col, row)

  operator.to_dense()
  ==> [[1., 4., -9.],
       [2., 1., 4.],
       [3., 2., 1.]]

  operator.shape
  ==> [3, 3]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [3, 4] Tensor
  operator.matmul(x)
  ==> Shape [3, 4] Tensor
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

    def __init__(self, col, row, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name='LinearOperatorToeplitz'):
        if False:
            while True:
                i = 10
        'Initialize a `LinearOperatorToeplitz`.\n\n    Args:\n      col: Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.\n        The first column of the operator. Allowed dtypes: `float16`, `float32`,\n          `float64`, `complex64`, `complex128`. Note that the first entry of\n          `col` is assumed to be the same as the first entry of `row`.\n      row: Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.\n        The first row of the operator. Allowed dtypes: `float16`, `float32`,\n          `float64`, `complex64`, `complex128`. Note that the first entry of\n          `row` is assumed to be the same as the first entry of `col`.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  If `diag.dtype` is real, this is auto-set to `True`.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name: A name for this `LinearOperator`.\n    '
        parameters = dict(col=col, row=row, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        with ops.name_scope(name, values=[row, col]):
            self._row = linear_operator_util.convert_nonref_to_tensor(row, name='row')
            self._col = linear_operator_util.convert_nonref_to_tensor(col, name='col')
            self._check_row_col(self._row, self._col)
            if is_square is False:
                raise ValueError('Only square Toeplitz operators currently supported.')
            is_square = True
            super(LinearOperatorToeplitz, self).__init__(dtype=self._row.dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    def _check_row_col(self, row, col):
        if False:
            return 10
        'Static check of row and column.'
        for (name, tensor) in [['row', row], ['col', col]]:
            if tensor.shape.ndims is not None and tensor.shape.ndims < 1:
                raise ValueError('Argument {} must have at least 1 dimension.  Found: {}'.format(name, tensor))
        if row.shape[-1] is not None and col.shape[-1] is not None:
            if row.shape[-1] != col.shape[-1]:
                raise ValueError('Expected square matrix, got row and col with mismatched dimensions.')

    def _shape(self):
        if False:
            print('Hello World!')
        v_shape = array_ops.broadcast_static_shape(self.row.shape, self.col.shape)
        return v_shape.concatenate(v_shape[-1:])

    def _shape_tensor(self, row=None, col=None):
        if False:
            while True:
                i = 10
        row = self.row if row is None else row
        col = self.col if col is None else col
        v_shape = array_ops.broadcast_dynamic_shape(array_ops.shape(row), array_ops.shape(col))
        k = v_shape[-1]
        return array_ops.concat((v_shape, [k]), 0)

    def _assert_self_adjoint(self):
        if False:
            return 10
        return check_ops.assert_equal(self.row, self.col, message='row and col are not the same, and so this operator is not self-adjoint.')

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            for i in range(10):
                print('nop')
        x = linalg.adjoint(x) if adjoint_arg else x
        expanded_x = array_ops.concat([x, array_ops.zeros_like(x)], axis=-2)
        col = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.col)
        row = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.row)
        circulant_col = array_ops.concat([col, array_ops.zeros_like(col[..., 0:1]), array_ops.reverse(row[..., 1:], axis=[-1])], axis=-1)
        circulant = linear_operator_circulant.LinearOperatorCirculant(fft_ops.fft(_to_complex(circulant_col)), input_output_dtype=row.dtype)
        result = circulant.matmul(expanded_x, adjoint=adjoint, adjoint_arg=False)
        shape = self._shape_tensor(row=row, col=col)
        return math_ops.cast(result[..., :self._domain_dimension_tensor(shape=shape), :], self.dtype)

    def _trace(self):
        if False:
            for i in range(10):
                print('nop')
        return math_ops.cast(self.domain_dimension_tensor(), dtype=self.dtype) * self.col[..., 0]

    def _diag_part(self):
        if False:
            while True:
                i = 10
        diag_entry = self.col[..., 0:1]
        return diag_entry * array_ops.ones([self.domain_dimension_tensor()], self.dtype)

    def _to_dense(self):
        if False:
            print('Hello World!')
        row = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.row)
        col = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.col)
        total_shape = array_ops.broadcast_dynamic_shape(array_ops.shape(row), array_ops.shape(col))
        n = array_ops.shape(row)[-1]
        row = array_ops.broadcast_to(row, total_shape)
        col = array_ops.broadcast_to(col, total_shape)
        elements = array_ops.concat([array_ops.reverse(col, axis=[-1]), row[..., 1:]], axis=-1)
        indices = math_ops.mod(math_ops.range(0, n) + math_ops.range(n - 1, -1, -1)[..., array_ops.newaxis], 2 * n - 1)
        return array_ops.gather(elements, indices, axis=-1)

    @property
    def col(self):
        if False:
            while True:
                i = 10
        return self._col

    @property
    def row(self):
        if False:
            return 10
        return self._row

    @property
    def _composite_tensor_fields(self):
        if False:
            for i in range(10):
                print('nop')
        return ('col', 'row')

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            for i in range(10):
                print('nop')
        return {'col': 1, 'row': 1}

def _to_complex(x):
    if False:
        i = 10
        return i + 15
    dtype = dtypes.complex64
    if x.dtype in [dtypes.float64, dtypes.complex128]:
        dtype = dtypes.complex128
    return math_ops.cast(x, dtype)