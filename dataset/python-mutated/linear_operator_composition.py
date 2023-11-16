"""Composes one or more `LinearOperators`."""
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperatorComposition']

@tf_export('linalg.LinearOperatorComposition')
@linear_operator.make_composite_tensor
class LinearOperatorComposition(linear_operator.LinearOperator):
    """Composes one or more `LinearOperators`.

  This operator composes one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator` with action defined by:

  ```
  op_composed(x) := op1(op2(...(opJ(x)...))
  ```

  If `opj` acts like [batch] matrix `Aj`, then `op_composed` acts like the
  [batch] matrix formed with the multiplication `A1 A2...AJ`.

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then we must have
  `N_j = M_{j+1}`, in which case the composed operator has shape equal to
  `broadcast_batch_shape + [M_1, N_J]`, where `broadcast_batch_shape` is the
  mutual broadcast of `batch_shape_j`, `j = 1,...,J`, assuming the intermediate
  batch shapes broadcast.  Even if the composed shape is well defined, the
  composed operator's methods may fail due to lack of broadcasting ability in
  the defining operators' methods.

  ```python
  # Create a 2 x 2 linear operator composed of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  operator = LinearOperatorComposition([operator_1, operator_2])

  operator.to_dense()
  ==> [[1., 2.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 5 linear operators.
  matrix_45 = tf.random.normal(shape=[2, 3, 4, 5])
  operator_45 = LinearOperatorFullMatrix(matrix)

  # Create a [2, 3] batch of 5 x 6 linear operators.
  matrix_56 = tf.random.normal(shape=[2, 3, 5, 6])
  operator_56 = LinearOperatorFullMatrix(matrix_56)

  # Compose to create a [2, 3] batch of 4 x 6 operators.
  operator_46 = LinearOperatorComposition([operator_45, operator_56])

  # Create a shape [2, 3, 6, 2] vector.
  x = tf.random.normal(shape=[2, 3, 6, 2])
  operator.matmul(x)
  ==> Shape [2, 3, 4, 2] Tensor
  ```

  #### Performance

  The performance of `LinearOperatorComposition` on any operation is equal to
  the sum of the individual operators' operations.


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

    def __init__(self, operators, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None):
        if False:
            while True:
                i = 10
        'Initialize a `LinearOperatorComposition`.\n\n    `LinearOperatorComposition` is initialized with a list of operators\n    `[op_1,...,op_J]`.  For the `matmul` method to be well defined, the\n    composition `op_i.matmul(op_{i+1}(x))` must be defined.  Other methods have\n    similar constraints.\n\n    Args:\n      operators:  Iterable of `LinearOperator` objects, each with\n        the same `dtype` and composable shape.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name: A name for this `LinearOperator`.  Default is the individual\n        operators names joined with `_o_`.\n\n    Raises:\n      TypeError:  If all operators do not have the same `dtype`.\n      ValueError:  If `operators` is empty.\n    '
        parameters = dict(operators=operators, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        check_ops.assert_proper_iterable(operators)
        operators = list(operators)
        if not operators:
            raise ValueError('Expected a non-empty list of operators. Found: %s' % operators)
        self._operators = operators
        dtype = operators[0].dtype
        for operator in operators:
            if operator.dtype != dtype:
                name_type = (str((o.name, o.dtype)) for o in operators)
                raise TypeError('Expected all operators to have the same dtype.  Found %s' % '   '.join(name_type))
        if all((operator.is_non_singular for operator in operators)):
            if is_non_singular is False:
                raise ValueError('The composition of non-singular operators is always non-singular.')
            is_non_singular = True
        if _composition_must_be_self_adjoint(operators):
            if is_self_adjoint is False:
                raise ValueError('The composition was determined to be self-adjoint but user provided incorrect `False` hint.')
            is_self_adjoint = True
        if linear_operator_util.is_aat_form(operators):
            if is_square is False:
                raise ValueError('The composition was determined have the form A @ A.H, hence it must be square. The user provided an incorrect `False` hint.')
            is_square = True
        if linear_operator_util.is_aat_form(operators) and is_non_singular:
            if is_positive_definite is False:
                raise ValueError('The composition was determined to be non-singular and have the form A @ A.H, hence it must be positive-definite. The user provided an incorrect `False` hint.')
            is_positive_definite = True
        if name is None:
            name = '_o_'.join((operator.name for operator in operators))
        with ops.name_scope(name):
            super(LinearOperatorComposition, self).__init__(dtype=dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    @property
    def operators(self):
        if False:
            i = 10
            return i + 15
        return self._operators

    def _shape(self):
        if False:
            return 10
        domain_dimension = self.operators[0].domain_dimension
        for operator in self.operators[1:]:
            domain_dimension.assert_is_compatible_with(operator.range_dimension)
            domain_dimension = operator.domain_dimension
        matrix_shape = tensor_shape.TensorShape([self.operators[0].range_dimension, self.operators[-1].domain_dimension])
        batch_shape = self.operators[0].batch_shape
        for operator in self.operators[1:]:
            batch_shape = common_shapes.broadcast_shape(batch_shape, operator.batch_shape)
        return batch_shape.concatenate(matrix_shape)

    def _shape_tensor(self):
        if False:
            print('Hello World!')
        if self.shape.is_fully_defined():
            return ops.convert_to_tensor(self.shape.as_list(), dtype=dtypes.int32, name='shape')
        matrix_shape = array_ops_stack.stack([self.operators[0].range_dimension_tensor(), self.operators[-1].domain_dimension_tensor()])
        zeros = array_ops.zeros(shape=self.operators[0].batch_shape_tensor())
        for operator in self.operators[1:]:
            zeros += array_ops.zeros(shape=operator.batch_shape_tensor())
        batch_shape = array_ops.shape(zeros)
        return array_ops.concat((batch_shape, matrix_shape), 0)

    def _linop_cholesky(self) -> linear_operator.LinearOperator:
        if False:
            print('Hello World!')
        'Computes Cholesky(LinearOperatorComposition).'

        def _is_llt_product(self):
            if False:
                print('Hello World!')
            'Determines if linop = L @ L.H for L = LinearOperatorLowerTriangular.'
            if len(self.operators) != 2:
                return False
            if not linear_operator_util.is_aat_form(self.operators):
                return False
            return isinstance(self.operators[0], linear_operator_lower_triangular.LinearOperatorLowerTriangular)
        if not _is_llt_product(self):
            return linear_operator_lower_triangular.LinearOperatorLowerTriangular(linalg_ops.cholesky(self.to_dense()), is_non_singular=True, is_self_adjoint=False, is_square=True)
        left_op = self.operators[0]
        if left_op.is_positive_definite:
            return left_op
        diag_sign = array_ops.expand_dims(math_ops.sign(left_op.diag_part()), axis=-2)
        return linear_operator_lower_triangular.LinearOperatorLowerTriangular(tril=left_op.tril / diag_sign, is_non_singular=left_op.is_non_singular, is_self_adjoint=left_op.is_self_adjoint, is_positive_definite=True if left_op.is_positive_definite else None, is_square=True)

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            for i in range(10):
                print('nop')
        if adjoint:
            matmul_order_list = self.operators
        else:
            matmul_order_list = list(reversed(self.operators))
        result = matmul_order_list[0].matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
        for operator in matmul_order_list[1:]:
            result = operator.matmul(result, adjoint=adjoint)
        return result

    def _determinant(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.operators[0].determinant()
        for operator in self.operators[1:]:
            result *= operator.determinant()
        return result

    def _log_abs_determinant(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.operators[0].log_abs_determinant()
        for operator in self.operators[1:]:
            result += operator.log_abs_determinant()
        return result

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            print('Hello World!')
        if adjoint:
            solve_order_list = list(reversed(self.operators))
        else:
            solve_order_list = self.operators
        solution = solve_order_list[0].solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        for operator in solve_order_list[1:]:
            solution = operator.solve(solution, adjoint=adjoint)
        return solution

    def _assert_non_singular(self):
        if False:
            while True:
                i = 10
        if all((operator.is_square for operator in self.operators)):
            asserts = [operator.assert_non_singular() for operator in self.operators]
            return control_flow_ops.group(asserts)
        return super(LinearOperatorComposition, self)._assert_non_singular()

    @property
    def _composite_tensor_fields(self):
        if False:
            print('Hello World!')
        return ('operators',)

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            while True:
                i = 10
        return {'operators': [0] * len(self.operators)}

def _composition_must_be_self_adjoint(operators):
    if False:
        while True:
            i = 10
    'Runs some checks to see if composition operators must be SA.\n\n  Args:\n    operators: List of LinearOperators.\n\n  Returns:\n    True if the composition must be SA. False if it is not SA OR if we did not\n      determine whether the composition is SA.\n  '
    if len(operators) == 1 and operators[0].is_self_adjoint:
        return True
    if linear_operator_util.is_aat_form(operators):
        return True
    return False