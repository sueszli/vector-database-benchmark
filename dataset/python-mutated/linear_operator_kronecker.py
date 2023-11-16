"""Construct the Kronecker product of one or more `LinearOperators`."""
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperatorKronecker']

def _prefer_static_shape(x):
    if False:
        while True:
            i = 10
    if x.shape.is_fully_defined():
        return x.shape
    return array_ops.shape(x)

def _prefer_static_concat_shape(first_shape, second_shape_int_list):
    if False:
        i = 10
        return i + 15
    'Concatenate a shape with a list of integers as statically as possible.\n\n  Args:\n    first_shape: `TensorShape` or `Tensor` instance. If a `TensorShape`,\n      `first_shape.is_fully_defined()` must return `True`.\n    second_shape_int_list: `list` of scalar integer `Tensor`s.\n\n  Returns:\n    `Tensor` representing concatenating `first_shape` and\n      `second_shape_int_list` as statically as possible.\n  '
    second_shape_int_list_static = [tensor_util.constant_value(s) for s in second_shape_int_list]
    if isinstance(first_shape, tensor_shape.TensorShape) and all((s is not None for s in second_shape_int_list_static)):
        return first_shape.concatenate(second_shape_int_list_static)
    return array_ops.concat([first_shape, second_shape_int_list], axis=0)

@tf_export('linalg.LinearOperatorKronecker')
@linear_operator.make_composite_tensor
class LinearOperatorKronecker(linear_operator.LinearOperator):
    """Kronecker product between two `LinearOperators`.

  This operator composes one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator` representing the Kronecker product:
  `op1 x op2 x .. opJ` (we omit parentheses as the Kronecker product is
  associative).

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then the composed operator
  will have shape equal to `broadcast_batch_shape + [prod M_j, prod N_j]`,
  where the product is over all operators.

  ```python
  # Create a 4 x 4 linear operator composed of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [2., 1.]])
  operator = LinearOperatorKronecker([operator_1, operator_2])

  operator.to_dense()
  ==> [[1., 0., 2., 0.],
       [2., 1., 4., 2.],
       [3., 0., 4., 0.],
       [6., 3., 8., 4.]]

  operator.shape
  ==> [4, 4]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [4, 2] Tensor
  operator.matmul(x)
  ==> Shape [4, 2] Tensor

  # Create a [2, 3] batch of 4 x 5 linear operators.
  matrix_45 = tf.random.normal(shape=[2, 3, 4, 5])
  operator_45 = LinearOperatorFullMatrix(matrix)

  # Create a [2, 3] batch of 5 x 6 linear operators.
  matrix_56 = tf.random.normal(shape=[2, 3, 5, 6])
  operator_56 = LinearOperatorFullMatrix(matrix_56)

  # Compose to create a [2, 3] batch of 20 x 30 operators.
  operator_large = LinearOperatorKronecker([operator_45, operator_56])

  # Create a shape [2, 3, 20, 2] vector.
  x = tf.random.normal(shape=[2, 3, 6, 2])
  operator_large.matmul(x)
  ==> Shape [2, 3, 30, 2] Tensor
  ```

  #### Performance

  The performance of `LinearOperatorKronecker` on any operation is equal to
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
        'Initialize a `LinearOperatorKronecker`.\n\n    `LinearOperatorKronecker` is initialized with a list of operators\n    `[op_1,...,op_J]`.\n\n    Args:\n      operators:  Iterable of `LinearOperator` objects, each with\n        the same `dtype` and composable shape, representing the Kronecker\n        factors.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix\\\n            #Extension_for_non_symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name: A name for this `LinearOperator`.  Default is the individual\n        operators names joined with `_x_`.\n\n    Raises:\n      TypeError:  If all operators do not have the same `dtype`.\n      ValueError:  If `operators` is empty.\n    '
        parameters = dict(operators=operators, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        check_ops.assert_proper_iterable(operators)
        operators = list(operators)
        if not operators:
            raise ValueError(f'Argument `operators` must be a list of >=1 operators. Received: {operators}.')
        self._operators = operators
        dtype = operators[0].dtype
        for operator in operators:
            if operator.dtype != dtype:
                name_type = (str((o.name, o.dtype)) for o in operators)
                raise TypeError(f'Expected every operation in argument `operators` to have the same dtype. Received {list(name_type)}.')
        if all((operator.is_non_singular for operator in operators)):
            if is_non_singular is False:
                raise ValueError(f'The Kronecker product of non-singular operators is always non-singular. Expected argument `is_non_singular` to be True. Received: {is_non_singular}.')
            is_non_singular = True
        if all((operator.is_self_adjoint for operator in operators)):
            if is_self_adjoint is False:
                raise ValueError(f'The Kronecker product of self-adjoint operators is always self-adjoint. Expected argument `is_self_adjoint` to be True. Received: {is_self_adjoint}.')
            is_self_adjoint = True
        if all((operator.is_positive_definite for operator in operators)):
            if is_positive_definite is False:
                raise ValueError(f'The Kronecker product of positive-definite operators is always positive-definite. Expected argument `is_positive_definite` to be True. Received: {is_positive_definite}.')
            is_positive_definite = True
        if name is None:
            name = operators[0].name
            for operator in operators[1:]:
                name += '_x_' + operator.name
        with ops.name_scope(name):
            super(LinearOperatorKronecker, self).__init__(dtype=dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    @property
    def operators(self):
        if False:
            print('Hello World!')
        return self._operators

    def _shape(self):
        if False:
            for i in range(10):
                print('nop')
        domain_dimension = self.operators[0].domain_dimension
        for operator in self.operators[1:]:
            domain_dimension = domain_dimension * operator.domain_dimension
        range_dimension = self.operators[0].range_dimension
        for operator in self.operators[1:]:
            range_dimension = range_dimension * operator.range_dimension
        matrix_shape = tensor_shape.TensorShape([range_dimension, domain_dimension])
        batch_shape = self.operators[0].batch_shape
        for operator in self.operators[1:]:
            batch_shape = common_shapes.broadcast_shape(batch_shape, operator.batch_shape)
        return batch_shape.concatenate(matrix_shape)

    def _shape_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        domain_dimension = self.operators[0].domain_dimension_tensor()
        for operator in self.operators[1:]:
            domain_dimension = domain_dimension * operator.domain_dimension_tensor()
        range_dimension = self.operators[0].range_dimension_tensor()
        for operator in self.operators[1:]:
            range_dimension = range_dimension * operator.range_dimension_tensor()
        matrix_shape = [range_dimension, domain_dimension]
        batch_shape = self.operators[0].batch_shape_tensor()
        for operator in self.operators[1:]:
            batch_shape = array_ops.broadcast_dynamic_shape(batch_shape, operator.batch_shape_tensor())
        return array_ops.concat((batch_shape, matrix_shape), 0)

    def _linop_adjoint(self) -> 'LinearOperatorKronecker':
        if False:
            i = 10
            return i + 15
        return LinearOperatorKronecker(operators=[operator.adjoint() for operator in self.operators], is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True)

    def _linop_cholesky(self) -> 'LinearOperatorKronecker':
        if False:
            i = 10
            return i + 15
        return LinearOperatorKronecker(operators=[operator.cholesky() for operator in self.operators], is_non_singular=True, is_self_adjoint=None, is_square=True)

    def _linop_inverse(self) -> 'LinearOperatorKronecker':
        if False:
            return 10
        return LinearOperatorKronecker(operators=[operator.inverse() for operator in self.operators], is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True)

    def _solve_matmul_internal(self, x, solve_matmul_fn, adjoint=False, adjoint_arg=False):
        if False:
            while True:
                i = 10
        output = x
        if adjoint_arg:
            if self.dtype.is_complex:
                output = math_ops.conj(output)
        else:
            output = linalg.transpose(output)
        for o in reversed(self.operators):
            if adjoint:
                operator_dimension = o.range_dimension_tensor()
            else:
                operator_dimension = o.domain_dimension_tensor()
            output_shape = _prefer_static_shape(output)
            if tensor_util.constant_value(operator_dimension) is not None:
                operator_dimension = tensor_util.constant_value(operator_dimension)
                if output.shape[-2] is not None and output.shape[-1] is not None:
                    dim = int(output.shape[-2] * output_shape[-1] // operator_dimension)
            else:
                dim = math_ops.cast(output_shape[-2] * output_shape[-1] // operator_dimension, dtype=dtypes.int32)
            output_shape = _prefer_static_concat_shape(output_shape[:-2], [dim, operator_dimension])
            output = array_ops.reshape(output, shape=output_shape)
            if self.dtype.is_complex:
                output = math_ops.conj(output)
            output = solve_matmul_fn(o, output, adjoint=adjoint, adjoint_arg=True)
        if adjoint_arg:
            col_dim = _prefer_static_shape(x)[-2]
        else:
            col_dim = _prefer_static_shape(x)[-1]
        if adjoint:
            row_dim = self.domain_dimension_tensor()
        else:
            row_dim = self.range_dimension_tensor()
        matrix_shape = [row_dim, col_dim]
        output = array_ops.reshape(output, _prefer_static_concat_shape(_prefer_static_shape(output)[:-2], matrix_shape))
        if x.shape.is_fully_defined():
            if adjoint_arg:
                column_dim = x.shape[-2]
            else:
                column_dim = x.shape[-1]
            broadcast_batch_shape = common_shapes.broadcast_shape(x.shape[:-2], self.batch_shape)
            if adjoint:
                matrix_dimensions = [self.domain_dimension, column_dim]
            else:
                matrix_dimensions = [self.range_dimension, column_dim]
            output.set_shape(broadcast_batch_shape.concatenate(matrix_dimensions))
        return output

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            while True:
                i = 10

        def matmul_fn(o, x, adjoint, adjoint_arg):
            if False:
                return 10
            return o.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
        return self._solve_matmul_internal(x=x, solve_matmul_fn=matmul_fn, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            return 10

        def solve_fn(o, rhs, adjoint, adjoint_arg):
            if False:
                while True:
                    i = 10
            return o.solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        return self._solve_matmul_internal(x=rhs, solve_matmul_fn=solve_fn, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _determinant(self):
        if False:
            for i in range(10):
                print('nop')
        total = self.domain_dimension_tensor()
        determinant = 1.0
        for operator in self.operators:
            determinant = determinant * operator.determinant() ** math_ops.cast(total / operator.domain_dimension_tensor(), dtype=operator.dtype)
        return determinant

    def _log_abs_determinant(self):
        if False:
            return 10
        total = self.domain_dimension_tensor()
        log_abs_det = 0.0
        for operator in self.operators:
            log_abs_det += operator.log_abs_determinant() * math_ops.cast(total / operator.domain_dimension_tensor(), dtype=operator.dtype)
        return log_abs_det

    def _trace(self):
        if False:
            return 10
        trace = 1.0
        for operator in self.operators:
            trace = trace * operator.trace()
        return trace

    def _diag_part(self):
        if False:
            i = 10
            return i + 15
        diag_part = self.operators[0].diag_part()
        for operator in self.operators[1:]:
            diag_part = diag_part[..., :, array_ops.newaxis]
            op_diag_part = operator.diag_part()[..., array_ops.newaxis, :]
            diag_part = diag_part * op_diag_part
            diag_part = array_ops.reshape(diag_part, shape=array_ops.concat([array_ops.shape(diag_part)[:-2], [-1]], axis=0))
        if self.range_dimension > self.domain_dimension:
            diag_dimension = self.domain_dimension
        else:
            diag_dimension = self.range_dimension
        diag_part.set_shape(self.batch_shape.concatenate(diag_dimension))
        return diag_part

    def _to_dense(self):
        if False:
            print('Hello World!')
        product = self.operators[0].to_dense()
        for operator in self.operators[1:]:
            product = product[..., :, array_ops.newaxis, :, array_ops.newaxis]
            op_to_mul = operator.to_dense()[..., array_ops.newaxis, :, array_ops.newaxis, :]
            product = product * op_to_mul
            product_shape = _prefer_static_shape(product)
            shape = _prefer_static_concat_shape(product_shape[:-4], [product_shape[-4] * product_shape[-3], product_shape[-2] * product_shape[-1]])
            product = array_ops.reshape(product, shape=shape)
        product.set_shape(self.shape)
        return product

    def _eigvals(self):
        if False:
            for i in range(10):
                print('nop')
        eigvals = [operator.eigvals() for operator in self.operators]
        product = eigvals[0]
        for eigval in eigvals[1:]:
            product = product[..., array_ops.newaxis]
            product = product * eigval[..., array_ops.newaxis, :]
            product = array_ops.reshape(product, shape=array_ops.concat([array_ops.shape(product)[:-2], [-1]], axis=0))
        product.set_shape(self.shape[:-1])
        return product

    def _assert_non_singular(self):
        if False:
            i = 10
            return i + 15
        if all((operator.is_square for operator in self.operators)):
            asserts = [operator.assert_non_singular() for operator in self.operators]
            return control_flow_ops.group(asserts)
        else:
            raise errors.InvalidArgumentError(node_def=None, op=None, message='All Kronecker factors must be square for the product to be invertible. Expected hint `is_square` to be True for every operator in argument `operators`.')

    def _assert_self_adjoint(self):
        if False:
            print('Hello World!')
        if all((operator.is_square for operator in self.operators)):
            asserts = [operator.assert_self_adjoint() for operator in self.operators]
            return control_flow_ops.group(asserts)
        else:
            raise errors.InvalidArgumentError(node_def=None, op=None, message='All Kronecker factors must be square for the product to be invertible. Expected hint `is_square` to be True for every operator in argument `operators`.')

    @property
    def _composite_tensor_fields(self):
        if False:
            return 10
        return ('operators',)

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            for i in range(10):
                print('nop')
        return {'operators': [0] * len(self.operators)}