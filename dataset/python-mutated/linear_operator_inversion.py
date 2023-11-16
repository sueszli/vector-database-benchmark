"""Inverts a non-singular `LinearOperator`."""
from tensorflow.python.framework import ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperatorInversion']

@tf_export('linalg.LinearOperatorInversion')
@linear_operator.make_composite_tensor
class LinearOperatorInversion(linear_operator.LinearOperator):
    """`LinearOperator` representing the inverse of another operator.

  This operator represents the inverse of another operator.

  ```python
  # Create a 2 x 2 linear operator.
  operator = LinearOperatorFullMatrix([[1., 0.], [0., 2.]])
  operator_inv = LinearOperatorInversion(operator)

  operator_inv.to_dense()
  ==> [[1., 0.]
       [0., 0.5]]

  operator_inv.shape
  ==> [2, 2]

  operator_inv.log_abs_determinant()
  ==> - log(2)

  x = ... Shape [2, 4] Tensor
  operator_inv.matmul(x)
  ==> Shape [2, 4] Tensor, equal to operator.solve(x)
  ```

  #### Performance

  The performance of `LinearOperatorInversion` depends on the underlying
  operators performance:  `solve` and `matmul` are swapped, and determinant is
  inverted.

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

    def __init__(self, operator, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None):
        if False:
            i = 10
            return i + 15
        'Initialize a `LinearOperatorInversion`.\n\n    `LinearOperatorInversion` is initialized with an operator `A`.  The `solve`\n    and `matmul` methods are effectively swapped.  E.g.\n\n    ```\n    A = MyLinearOperator(...)\n    B = LinearOperatorInversion(A)\n    x = [....]  # a vector\n\n    assert A.matvec(x) == B.solvevec(x)\n    ```\n\n    Args:\n      operator: `LinearOperator` object. If `operator.is_non_singular == False`,\n        an exception is raised.  We do allow `operator.is_non_singular == None`,\n        in which case this operator will have `is_non_singular == None`.\n        Similarly for `is_self_adjoint` and `is_positive_definite`.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name: A name for this `LinearOperator`. Default is `operator.name +\n        "_inv"`.\n\n    Raises:\n      ValueError:  If `operator.is_non_singular` is False.\n    '
        parameters = dict(operator=operator, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        self._operator = operator
        if operator.is_non_singular is False or is_non_singular is False:
            raise ValueError(f'Argument `is_non_singular` or argument `operator` must have supplied hint `is_non_singular` equal to `True` or `None`. Found `operator.is_non_singular`: {operator.is_non_singular}, `is_non_singular`: {is_non_singular}.')
        if operator.is_square is False or is_square is False:
            raise ValueError(f'Argument `is_square` or argument `operator` must have supplied hint `is_square` equal to `True` or `None`. Found `operator.is_square`: {operator.is_square}, `is_square`: {is_square}.')
        combine_hint = linear_operator_util.use_operator_or_provided_hint_unless_contradicting
        is_square = combine_hint(operator, 'is_square', is_square, 'An operator is square if and only if its inverse is square.')
        is_non_singular = combine_hint(operator, 'is_non_singular', is_non_singular, 'An operator is non-singular if and only if its inverse is non-singular.')
        is_self_adjoint = combine_hint(operator, 'is_self_adjoint', is_self_adjoint, 'An operator is self-adjoint if and only if its inverse is self-adjoint.')
        is_positive_definite = combine_hint(operator, 'is_positive_definite', is_positive_definite, 'An operator is positive-definite if and only if its inverse is positive-definite.')
        if name is None:
            name = operator.name + '_inv'
        with ops.name_scope(name):
            super(LinearOperatorInversion, self).__init__(dtype=operator.dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    @property
    def operator(self) -> 'LinearOperatorInversion':
        if False:
            return 10
        'The operator before inversion.'
        return self._operator

    def _linop_inverse(self) -> linear_operator.LinearOperator:
        if False:
            while True:
                i = 10
        return self.operator

    def _linop_solve(self, left_operator: 'LinearOperatorInversion', right_operator: linear_operator.LinearOperator) -> linear_operator.LinearOperator:
        if False:
            for i in range(10):
                print('nop')
        'Solve inverse of generic `LinearOperator`s.'
        return left_operator.operator.matmul(right_operator)

    def _assert_non_singular(self):
        if False:
            print('Hello World!')
        return self.operator.assert_non_singular()

    def _assert_positive_definite(self):
        if False:
            while True:
                i = 10
        return self.operator.assert_positive_definite()

    def _assert_self_adjoint(self):
        if False:
            i = 10
            return i + 15
        return self.operator.assert_self_adjoint()

    def _shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operator.shape

    def _shape_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operator.shape_tensor()

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            i = 10
            return i + 15
        return self.operator.solve(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _determinant(self):
        if False:
            while True:
                i = 10
        return 1.0 / self.operator.determinant()

    def _log_abs_determinant(self):
        if False:
            print('Hello World!')
        return -1.0 * self.operator.log_abs_determinant()

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            i = 10
            return i + 15
        return self.operator.matmul(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _eigvals(self):
        if False:
            print('Hello World!')
        return 1.0 / self.operator.eigvals()

    def _cond(self):
        if False:
            while True:
                i = 10
        return self.operator.cond()

    @property
    def _composite_tensor_fields(self):
        if False:
            while True:
                i = 10
        return ('operator',)

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            print('Hello World!')
        return {'operator': 0}