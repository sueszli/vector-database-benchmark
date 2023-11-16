"""Base class for linear operators."""
import abc
import contextlib
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import property_hint_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperator']

class _LinearOperatorGradient(composite_tensor_gradient.CompositeTensorGradient):
    """Composite tensor gradient for `LinearOperator`."""

    def get_gradient_components(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value._type_spec._to_components(value)

    def replace_gradient_components(self, value, components):
        if False:
            print('Hello World!')
        flat_components = nest.flatten(components)
        if all((c is None for c in flat_components)):
            return None
        value_components = value._type_spec._to_components(value)
        flat_grad_components = []
        for (gc, vc) in zip(flat_components, nest.flatten(value_components)):
            if gc is None:
                flat_grad_components.append(nest.map_structure(lambda x: array_ops.zeros_like(x, dtype=value.dtype), vc, expand_composites=True))
            else:
                flat_grad_components.append(gc)
        grad_components = nest.pack_sequence_as(value_components, flat_grad_components)
        return value._type_spec._from_components(grad_components)

@tf_export('linalg.LinearOperator')
class LinearOperator(module.Module, composite_tensor.CompositeTensor, metaclass=abc.ABCMeta):
    """Base class defining a [batch of] linear operator[s].

  Subclasses of `LinearOperator` provide access to common methods on a
  (batch) matrix, without the need to materialize the matrix.  This allows:

  * Matrix free computations
  * Operators that take advantage of special structure, while providing a
    consistent API to users.

  #### Subclassing

  To enable a public method, subclasses should implement the leading-underscore
  version of the method.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable
  `matmul(x, adjoint=False, name="matmul")` a subclass should implement
  `_matmul(x, adjoint=False)`.

  #### Performance contract

  Subclasses should only implement the assert methods
  (e.g. `assert_non_singular`) if they can be done in less than `O(N^3)`
  time.

  Class docstrings should contain an explanation of computational complexity.
  Since this is a high-performance library, attention should be paid to detail,
  and explanations can include constants as well as Big-O notation.

  #### Shape compatibility

  `LinearOperator` subclasses should operate on a [batch] matrix with
  compatible shape.  Class docstrings should define what is meant by compatible
  shape.  Some subclasses may not support batching.

  Examples:

  `x` is a batch matrix with compatible shape for `matmul` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  x.shape =   [B1,...,Bb] + [N, R]
  ```

  `rhs` is a batch matrix with compatible shape for `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  rhs.shape =   [B1,...,Bb] + [M, R]
  ```

  #### Example docstring for subclasses.

  This operator acts like a (batch) matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of identifying and working with compatible arguments the shape is
  relevant.

  Examples:

  ```python
  some_tensor = ... shape = ????
  operator = MyLinOp(some_tensor)

  operator.shape()
  ==> [2, 4, 4]

  operator.log_abs_determinant()
  ==> Shape [2] Tensor

  x = ... Shape [2, 4, 5] Tensor

  operator.matmul(x)
  ==> Shape [2, 4, 5] Tensor
  ```

  #### Shape compatibility

  This operator acts on batch matrices with compatible shape.
  FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

  #### Performance

  FILL THIS IN

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

  #### Initialization parameters

  All subclasses of `LinearOperator` are expected to pass a `parameters`
  argument to `super().__init__()`.  This should be a `dict` containing
  the unadulterated arguments passed to the subclass `__init__`.  For example,
  `MyLinearOperator` with an initializer should look like:

  ```python
  def __init__(self, operator, is_square=False, name=None):
     parameters = dict(
         operator=operator,
         is_square=is_square,
         name=name
     )
     ...
     super().__init__(..., parameters=parameters)
  ```

   Users can then access `my_linear_operator.parameters` to see all arguments
   passed to its initializer.
  """

    @deprecation.deprecated_args(None, 'Do not pass `graph_parents`.  They will  no longer be used.', 'graph_parents')
    def __init__(self, dtype, graph_parents=None, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None, parameters=None):
        if False:
            i = 10
            return i + 15
        'Initialize the `LinearOperator`.\n\n    **This is a private method for subclass use.**\n    **Subclasses should copy-paste this `__init__` documentation.**\n\n    Args:\n      dtype: The type of the this `LinearOperator`.  Arguments to `matmul` and\n        `solve` will have to be this type.\n      graph_parents: (Deprecated) Python list of graph prerequisites of this\n        `LinearOperator` Typically tensors that are passed during initialization\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  If `dtype` is real, this is equivalent to being symmetric.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name: A name for this `LinearOperator`.\n      parameters: Python `dict` of parameters used to instantiate this\n        `LinearOperator`.\n\n    Raises:\n      ValueError:  If any member of graph_parents is `None` or not a `Tensor`.\n      ValueError:  If hints are set incorrectly.\n    '
        if is_positive_definite:
            if is_non_singular is False:
                raise ValueError('A positive definite matrix is always non-singular.')
            is_non_singular = True
        if is_non_singular:
            if is_square is False:
                raise ValueError('A non-singular matrix is always square.')
            is_square = True
        if is_self_adjoint:
            if is_square is False:
                raise ValueError('A self-adjoint matrix is always square.')
            is_square = True
        self._is_square_set_or_implied_by_hints = is_square
        if graph_parents is not None:
            self._set_graph_parents(graph_parents)
        else:
            self._graph_parents = []
        self._dtype = dtypes.as_dtype(dtype).base_dtype if dtype else dtype
        self._is_non_singular = is_non_singular
        self._is_self_adjoint = is_self_adjoint
        self._is_positive_definite = is_positive_definite
        self._parameters = self._no_dependency(parameters)
        self._parameters_sanitized = False
        self._name = name or type(self).__name__

    @contextlib.contextmanager
    def _name_scope(self, name=None):
        if False:
            print('Hello World!')
        'Helper function to standardize op scope.'
        full_name = self.name
        if name is not None:
            full_name += '/' + name
        with ops.name_scope(full_name) as scope:
            yield scope

    @property
    def parameters(self):
        if False:
            i = 10
            return i + 15
        'Dictionary of parameters used to instantiate this `LinearOperator`.'
        return dict(self._parameters)

    @property
    def dtype(self):
        if False:
            return 10
        'The `DType` of `Tensor`s handled by this `LinearOperator`.'
        return self._dtype

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'Name prepended to all ops created by this `LinearOperator`.'
        return self._name

    @property
    @deprecation.deprecated(None, 'Do not call `graph_parents`.')
    def graph_parents(self):
        if False:
            return 10
        'List of graph dependencies of this `LinearOperator`.'
        return self._graph_parents

    @property
    def is_non_singular(self):
        if False:
            for i in range(10):
                print('nop')
        return self._is_non_singular

    @property
    def is_self_adjoint(self):
        if False:
            i = 10
            return i + 15
        return self._is_self_adjoint

    @property
    def is_positive_definite(self):
        if False:
            return 10
        return self._is_positive_definite

    @property
    def is_square(self):
        if False:
            for i in range(10):
                print('nop')
        'Return `True/False` depending on if this operator is square.'
        auto_square_check = self.domain_dimension == self.range_dimension
        if self._is_square_set_or_implied_by_hints is False and auto_square_check:
            raise ValueError('User set is_square hint to False, but the operator was square.')
        if self._is_square_set_or_implied_by_hints is None:
            return auto_square_check
        return self._is_square_set_or_implied_by_hints

    @abc.abstractmethod
    def _shape(self):
        if False:
            return 10
        raise NotImplementedError('_shape is not implemented.')

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        '`TensorShape` of this `LinearOperator`.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns\n    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.shape`.\n\n    Returns:\n      `TensorShape`, statically determined, may be undefined.\n    '
        return self._shape()

    def _shape_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('_shape_tensor is not implemented.')

    def shape_tensor(self, name='shape_tensor'):
        if False:
            print('Hello World!')
        'Shape of this `LinearOperator`, determined at runtime.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding\n    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `int32` `Tensor`\n    '
        with self._name_scope(name):
            if self.shape.is_fully_defined():
                return linear_operator_util.shape_tensor(self.shape.as_list())
            else:
                return self._shape_tensor()

    @property
    def batch_shape(self):
        if False:
            while True:
                i = 10
        '`TensorShape` of batch dimensions of this `LinearOperator`.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns\n    `TensorShape([B1,...,Bb])`, equivalent to `A.shape[:-2]`\n\n    Returns:\n      `TensorShape`, statically determined, may be undefined.\n    '
        return self.shape[:-2]

    def batch_shape_tensor(self, name='batch_shape_tensor'):
        if False:
            while True:
                i = 10
        'Shape of batch dimensions of this operator, determined at runtime.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding\n    `[B1,...,Bb]`.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `int32` `Tensor`\n    '
        with self._name_scope(name):
            return self._batch_shape_tensor()

    def _batch_shape_tensor(self, shape=None):
        if False:
            while True:
                i = 10
        if self.batch_shape.is_fully_defined():
            return linear_operator_util.shape_tensor(self.batch_shape.as_list(), name='batch_shape')
        else:
            shape = self.shape_tensor() if shape is None else shape
            return shape[:-2]

    @property
    def tensor_rank(self, name='tensor_rank'):
        if False:
            print('Hello World!')
        'Rank (in the sense of tensors) of matrix corresponding to this operator.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      Python integer, or None if the tensor rank is undefined.\n    '
        with self._name_scope(name):
            return self.shape.ndims

    def tensor_rank_tensor(self, name='tensor_rank_tensor'):
        if False:
            print('Hello World!')
        'Rank (in the sense of tensors) of matrix corresponding to this operator.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `int32` `Tensor`, determined at runtime.\n    '
        with self._name_scope(name):
            return self._tensor_rank_tensor()

    def _tensor_rank_tensor(self, shape=None):
        if False:
            while True:
                i = 10
        if self.tensor_rank is not None:
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(self.tensor_rank)
        else:
            shape = self.shape_tensor() if shape is None else shape
            return array_ops.size(shape)

    @property
    def domain_dimension(self):
        if False:
            for i in range(10):
                print('nop')
        'Dimension (in the sense of vector spaces) of the domain of this operator.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.\n\n    Returns:\n      `Dimension` object.\n    '
        if self.shape.rank is None:
            return tensor_shape.Dimension(None)
        else:
            return self.shape.dims[-1]

    def domain_dimension_tensor(self, name='domain_dimension_tensor'):
        if False:
            for i in range(10):
                print('nop')
        'Dimension (in the sense of vector spaces) of the domain of this operator.\n\n    Determined at runtime.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `int32` `Tensor`\n    '
        with self._name_scope(name):
            return self._domain_dimension_tensor()

    def _domain_dimension_tensor(self, shape=None):
        if False:
            print('Hello World!')
        dim_value = tensor_shape.dimension_value(self.domain_dimension)
        if dim_value is not None:
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(dim_value)
        else:
            shape = self.shape_tensor() if shape is None else shape
            return shape[-1]

    @property
    def range_dimension(self):
        if False:
            return 10
        'Dimension (in the sense of vector spaces) of the range of this operator.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.\n\n    Returns:\n      `Dimension` object.\n    '
        if self.shape.dims:
            return self.shape.dims[-2]
        else:
            return tensor_shape.Dimension(None)

    def range_dimension_tensor(self, name='range_dimension_tensor'):
        if False:
            print('Hello World!')
        'Dimension (in the sense of vector spaces) of the range of this operator.\n\n    Determined at runtime.\n\n    If this operator acts like the batch matrix `A` with\n    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `int32` `Tensor`\n    '
        with self._name_scope(name):
            return self._range_dimension_tensor()

    def _range_dimension_tensor(self, shape=None):
        if False:
            i = 10
            return i + 15
        dim_value = tensor_shape.dimension_value(self.range_dimension)
        if dim_value is not None:
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(dim_value)
        else:
            shape = self.shape_tensor() if shape is None else shape
            return shape[-2]

    def _assert_non_singular(self):
        if False:
            while True:
                i = 10
        'Private default implementation of _assert_non_singular.'
        logging.warn('Using (possibly slow) default implementation of assert_non_singular.  Requires conversion to a dense matrix and O(N^3) operations.')
        if self._can_use_cholesky():
            return self.assert_positive_definite()
        else:
            singular_values = linalg_ops.svd(self.to_dense(), compute_uv=False)
            cond = math_ops.reduce_max(singular_values, axis=-1) / math_ops.reduce_min(singular_values, axis=-1)
            return check_ops.assert_less(cond, self._max_condition_number_to_be_non_singular(), message='Singular matrix up to precision epsilon.')

    def _max_condition_number_to_be_non_singular(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the maximum condition number that we consider nonsingular.'
        with ops.name_scope('max_nonsingular_condition_number'):
            dtype_eps = np.finfo(self.dtype.as_numpy_dtype).eps
            eps = math_ops.cast(math_ops.reduce_max([100.0, math_ops.cast(self.range_dimension_tensor(), self.dtype), math_ops.cast(self.domain_dimension_tensor(), self.dtype)]), self.dtype) * dtype_eps
            return 1.0 / eps

    def assert_non_singular(self, name='assert_non_singular'):
        if False:
            print('Hello World!')
        'Returns an `Op` that asserts this operator is non singular.\n\n    This operator is considered non-singular if\n\n    ```\n    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,\n    eps := np.finfo(self.dtype.as_numpy_dtype).eps\n    ```\n\n    Args:\n      name:  A string name to prepend to created ops.\n\n    Returns:\n      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if\n        the operator is singular.\n    '
        with self._name_scope(name):
            return self._assert_non_singular()

    def _assert_positive_definite(self):
        if False:
            while True:
                i = 10
        'Default implementation of _assert_positive_definite.'
        logging.warn('Using (possibly slow) default implementation of assert_positive_definite.  Requires conversion to a dense matrix and O(N^3) operations.')
        if self.is_self_adjoint:
            return check_ops.assert_positive(array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense())), message='Matrix was not positive definite.')
        raise NotImplementedError('assert_positive_definite is not implemented.')

    def assert_positive_definite(self, name='assert_positive_definite'):
        if False:
            print('Hello World!')
        'Returns an `Op` that asserts this operator is positive definite.\n\n    Here, positive definite means that the quadratic form `x^H A x` has positive\n    real part for all nonzero `x`.  Note that we do not require the operator to\n    be self-adjoint to be positive definite.\n\n    Args:\n      name:  A name to give this `Op`.\n\n    Returns:\n      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if\n        the operator is not positive definite.\n    '
        with self._name_scope(name):
            return self._assert_positive_definite()

    def _assert_self_adjoint(self):
        if False:
            for i in range(10):
                print('nop')
        dense = self.to_dense()
        logging.warn('Using (possibly slow) default implementation of assert_self_adjoint.  Requires conversion to a dense matrix.')
        return check_ops.assert_equal(dense, linalg.adjoint(dense), message='Matrix was not equal to its adjoint.')

    def assert_self_adjoint(self, name='assert_self_adjoint'):
        if False:
            return 10
        'Returns an `Op` that asserts this operator is self-adjoint.\n\n    Here we check that this operator is *exactly* equal to its hermitian\n    transpose.\n\n    Args:\n      name:  A string name to prepend to created ops.\n\n    Returns:\n      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if\n        the operator is not self-adjoint.\n    '
        with self._name_scope(name):
            return self._assert_self_adjoint()

    def _check_input_dtype(self, arg):
        if False:
            while True:
                i = 10
        'Check that arg.dtype == self.dtype.'
        if arg.dtype.base_dtype != self.dtype:
            raise TypeError('Expected argument to have dtype %s.  Found: %s in tensor %s' % (self.dtype, arg.dtype, arg))

    @abc.abstractmethod
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('_matmul is not implemented.')

    def matmul(self, x, adjoint=False, adjoint_arg=False, name='matmul'):
        if False:
            return 10
        'Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.\n\n    ```python\n    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]\n    operator = LinearOperator(...)\n    operator.shape = [..., M, N]\n\n    X = ... # shape [..., N, R], batch matrix, R > 0.\n\n    Y = operator.matmul(X)\n    Y.shape\n    ==> [..., M, R]\n\n    Y[..., :, r] = sum_j A[..., :, j] X[j, r]\n    ```\n\n    Args:\n      x: `LinearOperator` or `Tensor` with compatible shape and same `dtype` as\n        `self`. See class docstring for definition of compatibility.\n      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.\n      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is\n        the hermitian transpose (transposition and complex conjugation).\n      name:  A name for this `Op`.\n\n    Returns:\n      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`\n        as `self`.\n    '
        if isinstance(x, LinearOperator):
            left_operator = self.adjoint() if adjoint else self
            right_operator = x.adjoint() if adjoint_arg else x
            if right_operator.range_dimension is not None and left_operator.domain_dimension is not None and (right_operator.range_dimension != left_operator.domain_dimension):
                raise ValueError('Operators are incompatible. Expected `x` to have dimension {} but got {}.'.format(left_operator.domain_dimension, right_operator.range_dimension))
            with self._name_scope(name):
                return self._linop_matmul(left_operator, right_operator)
        with self._name_scope(name):
            x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name='x')
            self._check_input_dtype(x)
            self_dim = -2 if adjoint else -1
            arg_dim = -1 if adjoint_arg else -2
            tensor_shape.dimension_at_index(self.shape, self_dim).assert_is_compatible_with(x.shape[arg_dim])
            return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _linop_matmul(self, left_operator: 'LinearOperator', right_operator: 'LinearOperator') -> 'LinearOperator':
        if False:
            i = 10
            return i + 15
        if hasattr(right_operator, '_ones_diag') and (not hasattr(right_operator, 'multiplier')):
            return left_operator
        elif hasattr(right_operator, '_zeros_diag'):
            if not right_operator.is_square or not left_operator.is_square:
                raise ValueError('Matmul with non-square `LinearOperator`s or non-square `LinearOperatorZeros` not supported at this time.')
            return right_operator
        else:
            is_square = property_hint_util.is_square(left_operator, right_operator)
            is_non_singular = None
            is_self_adjoint = None
            is_positive_definite = None
            if is_square:
                is_non_singular = property_hint_util.combined_non_singular_hint(left_operator, right_operator)
            elif is_square is False:
                is_non_singular = False
                is_self_adjoint = False
                is_positive_definite = False
            from tensorflow.python.ops.linalg import linear_operator_composition
            return linear_operator_composition.LinearOperatorComposition(operators=[left_operator, right_operator], is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square)

    def __matmul__(self, other):
        if False:
            return 10
        return self.matmul(other)

    def _matvec(self, x, adjoint=False):
        if False:
            while True:
                i = 10
        x_mat = array_ops.expand_dims(x, axis=-1)
        y_mat = self.matmul(x_mat, adjoint=adjoint)
        return array_ops.squeeze(y_mat, axis=-1)

    def matvec(self, x, adjoint=False, name='matvec'):
        if False:
            print('Hello World!')
        'Transform [batch] vector `x` with left multiplication:  `x --> Ax`.\n\n    ```python\n    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]\n    operator = LinearOperator(...)\n\n    X = ... # shape [..., N], batch vector\n\n    Y = operator.matvec(X)\n    Y.shape\n    ==> [..., M]\n\n    Y[..., :] = sum_j A[..., :, j] X[..., j]\n    ```\n\n    Args:\n      x: `Tensor` with compatible shape and same `dtype` as `self`.\n        `x` is treated as a [batch] vector meaning for every set of leading\n        dimensions, the last dimension defines a vector.\n        See class docstring for definition of compatibility.\n      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.\n      name:  A name for this `Op`.\n\n    Returns:\n      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.\n    '
        with self._name_scope(name):
            x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name='x')
            self._check_input_dtype(x)
            self_dim = -2 if adjoint else -1
            tensor_shape.dimension_at_index(self.shape, self_dim).assert_is_compatible_with(x.shape[-1])
            return self._matvec(x, adjoint=adjoint)

    def _determinant(self):
        if False:
            while True:
                i = 10
        logging.warn('Using (possibly slow) default implementation of determinant.  Requires conversion to a dense matrix and O(N^3) operations.')
        if self._can_use_cholesky():
            return math_ops.exp(self.log_abs_determinant())
        return linalg_ops.matrix_determinant(self.to_dense())

    def determinant(self, name='det'):
        if False:
            for i in range(10):
                print('nop')
        'Determinant for every batch member.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.\n\n    Raises:\n      NotImplementedError:  If `self.is_square` is `False`.\n    '
        if self.is_square is False:
            raise NotImplementedError('Determinant not implemented for an operator that is expected to not be square.')
        with self._name_scope(name):
            return self._determinant()

    def _log_abs_determinant(self):
        if False:
            for i in range(10):
                print('nop')
        logging.warn('Using (possibly slow) default implementation of determinant.  Requires conversion to a dense matrix and O(N^3) operations.')
        if self._can_use_cholesky():
            diag = array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense()))
            return 2 * math_ops.reduce_sum(math_ops.log(diag), axis=[-1])
        (_, log_abs_det) = linalg.slogdet(self.to_dense())
        return log_abs_det

    def log_abs_determinant(self, name='log_abs_det'):
        if False:
            print('Hello World!')
        'Log absolute value of determinant for every batch member.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.\n\n    Raises:\n      NotImplementedError:  If `self.is_square` is `False`.\n    '
        if self.is_square is False:
            raise NotImplementedError('Determinant not implemented for an operator that is expected to not be square.')
        with self._name_scope(name):
            return self._log_abs_determinant()

    def _dense_solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            return 10
        'Solve by conversion to a dense matrix.'
        if self.is_square is False:
            raise NotImplementedError('Solve is not yet implemented for non-square operators.')
        rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
        if self._can_use_cholesky():
            return linalg_ops.cholesky_solve(linalg_ops.cholesky(self.to_dense()), rhs)
        return linear_operator_util.matrix_solve_with_broadcast(self.to_dense(), rhs, adjoint=adjoint)

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            return 10
        'Default implementation of _solve.'
        logging.warn('Using (possibly slow) default implementation of solve.  Requires conversion to a dense matrix and O(N^3) operations.')
        return self._dense_solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def solve(self, rhs, adjoint=False, adjoint_arg=False, name='solve'):
        if False:
            for i in range(10):
                print('nop')
        "Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.\n\n    The returned `Tensor` will be close to an exact solution if `A` is well\n    conditioned. Otherwise closeness will vary. See class docstring for details.\n\n    Examples:\n\n    ```python\n    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]\n    operator = LinearOperator(...)\n    operator.shape = [..., M, N]\n\n    # Solve R > 0 linear systems for every member of the batch.\n    RHS = ... # shape [..., M, R]\n\n    X = operator.solve(RHS)\n    # X[..., :, r] is the solution to the r'th linear system\n    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]\n\n    operator.matmul(X)\n    ==> RHS\n    ```\n\n    Args:\n      rhs: `Tensor` with same `dtype` as this operator and compatible shape.\n        `rhs` is treated like a [batch] matrix meaning for every set of leading\n        dimensions, the last two dimensions defines a matrix.\n        See class docstring for definition of compatibility.\n      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint\n        of this `LinearOperator`:  `A^H X = rhs`.\n      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`\n        is the hermitian transpose (transposition and complex conjugation).\n      name:  A name scope to use for ops added by this method.\n\n    Returns:\n      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.\n\n    Raises:\n      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.\n    "
        if self.is_non_singular is False:
            raise NotImplementedError('Exact solve not implemented for an operator that is expected to be singular.')
        if self.is_square is False:
            raise NotImplementedError('Exact solve not implemented for an operator that is expected to not be square.')
        if isinstance(rhs, LinearOperator):
            left_operator = self.adjoint() if adjoint else self
            right_operator = rhs.adjoint() if adjoint_arg else rhs
            if right_operator.range_dimension is not None and left_operator.domain_dimension is not None and (right_operator.range_dimension != left_operator.domain_dimension):
                raise ValueError('Operators are incompatible. Expected `rhs` to have dimension {} but got {}.'.format(left_operator.domain_dimension, right_operator.range_dimension))
            with self._name_scope(name):
                return self._linop_solve(left_operator, right_operator)
        with self._name_scope(name):
            rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(rhs, name='rhs')
            self._check_input_dtype(rhs)
            self_dim = -1 if adjoint else -2
            arg_dim = -1 if adjoint_arg else -2
            tensor_shape.dimension_at_index(self.shape, self_dim).assert_is_compatible_with(rhs.shape[arg_dim])
            return self._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _linop_solve(self, left_operator: 'LinearOperator', right_operator: 'LinearOperator') -> 'LinearOperator':
        if False:
            i = 10
            return i + 15
        if hasattr(right_operator, '_ones_diag') and (not hasattr(right_operator, 'multiplier')):
            return left_operator.inverse()
        is_square = property_hint_util.is_square(left_operator, right_operator)
        is_non_singular = None
        is_self_adjoint = None
        is_positive_definite = None
        if is_square:
            is_non_singular = property_hint_util.combined_non_singular_hint(left_operator, right_operator)
        elif is_square is False:
            is_non_singular = False
            is_self_adjoint = False
            is_positive_definite = False
        from tensorflow.python.ops.linalg import linear_operator_composition
        from tensorflow.python.ops.linalg import linear_operator_inversion
        return linear_operator_composition.LinearOperatorComposition(operators=[linear_operator_inversion.LinearOperatorInversion(left_operator), right_operator], is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square)

    def _solvevec(self, rhs, adjoint=False):
        if False:
            return 10
        'Default implementation of _solvevec.'
        rhs_mat = array_ops.expand_dims(rhs, axis=-1)
        solution_mat = self.solve(rhs_mat, adjoint=adjoint)
        return array_ops.squeeze(solution_mat, axis=-1)

    def solvevec(self, rhs, adjoint=False, name='solve'):
        if False:
            while True:
                i = 10
        'Solve single equation with best effort: `A X = rhs`.\n\n    The returned `Tensor` will be close to an exact solution if `A` is well\n    conditioned. Otherwise closeness will vary. See class docstring for details.\n\n    Examples:\n\n    ```python\n    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]\n    operator = LinearOperator(...)\n    operator.shape = [..., M, N]\n\n    # Solve one linear system for every member of the batch.\n    RHS = ... # shape [..., M]\n\n    X = operator.solvevec(RHS)\n    # X is the solution to the linear system\n    # sum_j A[..., :, j] X[..., j] = RHS[..., :]\n\n    operator.matvec(X)\n    ==> RHS\n    ```\n\n    Args:\n      rhs: `Tensor` with same `dtype` as this operator.\n        `rhs` is treated like a [batch] vector meaning for every set of leading\n        dimensions, the last dimension defines a vector.  See class docstring\n        for definition of compatibility regarding batch dimensions.\n      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint\n        of this `LinearOperator`:  `A^H X = rhs`.\n      name:  A name scope to use for ops added by this method.\n\n    Returns:\n      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.\n\n    Raises:\n      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.\n    '
        with self._name_scope(name):
            rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(rhs, name='rhs')
            self._check_input_dtype(rhs)
            self_dim = -1 if adjoint else -2
            tensor_shape.dimension_at_index(self.shape, self_dim).assert_is_compatible_with(rhs.shape[-1])
            return self._solvevec(rhs, adjoint=adjoint)

    def adjoint(self, name: str='adjoint') -> 'LinearOperator':
        if False:
            for i in range(10):
                print('nop')
        'Returns the adjoint of the current `LinearOperator`.\n\n    Given `A` representing this `LinearOperator`, return `A*`.\n    Note that calling `self.adjoint()` and `self.H` are equivalent.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `LinearOperator` which represents the adjoint of this `LinearOperator`.\n    '
        if self.is_self_adjoint is True:
            return self
        with self._name_scope(name):
            return self._linop_adjoint()
    H = property(adjoint, None)

    def _linop_adjoint(self) -> 'LinearOperator':
        if False:
            i = 10
            return i + 15
        from tensorflow.python.ops.linalg import linear_operator_adjoint
        return linear_operator_adjoint.LinearOperatorAdjoint(self, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=self.is_square)

    def inverse(self, name: str='inverse') -> 'LinearOperator':
        if False:
            i = 10
            return i + 15
        'Returns the Inverse of this `LinearOperator`.\n\n    Given `A` representing this `LinearOperator`, return a `LinearOperator`\n    representing `A^-1`.\n\n    Args:\n      name: A name scope to use for ops added by this method.\n\n    Returns:\n      `LinearOperator` representing inverse of this matrix.\n\n    Raises:\n      ValueError: When the `LinearOperator` is not hinted to be `non_singular`.\n    '
        if self.is_square is False:
            raise ValueError('Cannot take the Inverse: This operator represents a non square matrix.')
        if self.is_non_singular is False:
            raise ValueError('Cannot take the Inverse: This operator represents a singular matrix.')
        with self._name_scope(name):
            return self._linop_inverse()

    def _linop_inverse(self) -> 'LinearOperator':
        if False:
            for i in range(10):
                print('nop')
        from tensorflow.python.ops.linalg import linear_operator_inversion
        return linear_operator_inversion.LinearOperatorInversion(self, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=self.is_square)

    def cholesky(self, name: str='cholesky') -> 'LinearOperator':
        if False:
            for i in range(10):
                print('nop')
        'Returns a Cholesky factor as a `LinearOperator`.\n\n    Given `A` representing this `LinearOperator`, if `A` is positive definite\n    self-adjoint, return `L`, where `A = L L^T`, i.e. the cholesky\n    decomposition.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      `LinearOperator` which represents the lower triangular matrix\n      in the Cholesky decomposition.\n\n    Raises:\n      ValueError: When the `LinearOperator` is not hinted to be positive\n        definite and self adjoint.\n    '
        if not self._can_use_cholesky():
            raise ValueError('Cannot take the Cholesky decomposition: Not a positive definite self adjoint matrix.')
        with self._name_scope(name):
            return self._linop_cholesky()

    def _linop_cholesky(self) -> 'LinearOperator':
        if False:
            return 10
        from tensorflow.python.ops.linalg import linear_operator_lower_triangular
        return linear_operator_lower_triangular.LinearOperatorLowerTriangular(linalg_ops.cholesky(self.to_dense()), is_non_singular=True, is_self_adjoint=False, is_square=True)

    def _to_dense(self):
        if False:
            while True:
                i = 10
        'Generic and often inefficient implementation.  Override often.'
        if self.batch_shape.is_fully_defined():
            batch_shape = self.batch_shape
        else:
            batch_shape = self.batch_shape_tensor()
        dim_value = tensor_shape.dimension_value(self.domain_dimension)
        if dim_value is not None:
            n = dim_value
        else:
            n = self.domain_dimension_tensor()
        eye = linalg_ops.eye(num_rows=n, batch_shape=batch_shape, dtype=self.dtype)
        return self.matmul(eye)

    def to_dense(self, name='to_dense'):
        if False:
            for i in range(10):
                print('nop')
        'Return a dense (batch) matrix representing this operator.'
        with self._name_scope(name):
            return self._to_dense()

    def _diag_part(self):
        if False:
            while True:
                i = 10
        'Generic and often inefficient implementation.  Override often.'
        return array_ops.matrix_diag_part(self.to_dense())

    def diag_part(self, name='diag_part'):
        if False:
            print('Hello World!')
        'Efficiently get the [batch] diagonal part of this operator.\n\n    If this operator has shape `[B1,...,Bb, M, N]`, this returns a\n    `Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where\n    `diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.\n\n    ```\n    my_operator = LinearOperatorDiag([1., 2.])\n\n    # Efficiently get the diagonal\n    my_operator.diag_part()\n    ==> [1., 2.]\n\n    # Equivalent, but inefficient method\n    tf.linalg.diag_part(my_operator.to_dense())\n    ==> [1., 2.]\n    ```\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      diag_part:  A `Tensor` of same `dtype` as self.\n    '
        with self._name_scope(name):
            return self._diag_part()

    def _trace(self):
        if False:
            for i in range(10):
                print('nop')
        return math_ops.reduce_sum(self.diag_part(), axis=-1)

    def trace(self, name='trace'):
        if False:
            return 10
        'Trace of the linear operator, equal to sum of `self.diag_part()`.\n\n    If the operator is square, this is also the sum of the eigenvalues.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.\n    '
        with self._name_scope(name):
            return self._trace()

    def _add_to_tensor(self, x):
        if False:
            print('Hello World!')
        return self.to_dense() + x

    def add_to_tensor(self, x, name='add_to_tensor'):
        if False:
            print('Hello World!')
        'Add matrix represented by this operator to `x`.  Equivalent to `A + x`.\n\n    Args:\n      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.\n      name:  A name to give this `Op`.\n\n    Returns:\n      A `Tensor` with broadcast shape and same `dtype` as `self`.\n    '
        with self._name_scope(name):
            x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name='x')
            self._check_input_dtype(x)
            return self._add_to_tensor(x)

    def _eigvals(self):
        if False:
            print('Hello World!')
        return linalg_ops.self_adjoint_eigvals(self.to_dense())

    def eigvals(self, name='eigvals'):
        if False:
            print('Hello World!')
        'Returns the eigenvalues of this linear operator.\n\n    If the operator is marked as self-adjoint (via `is_self_adjoint`)\n    this computation can be more efficient.\n\n    Note: This currently only supports self-adjoint operators.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      Shape `[B1,...,Bb, N]` `Tensor` of same `dtype` as `self`.\n    '
        if not self.is_self_adjoint:
            raise NotImplementedError('Only self-adjoint matrices are supported.')
        with self._name_scope(name):
            return self._eigvals()

    def _cond(self):
        if False:
            return 10
        if not self.is_self_adjoint:
            vals = linalg_ops.svd(self.to_dense(), compute_uv=False)
        else:
            vals = math_ops.abs(self._eigvals())
        return math_ops.reduce_max(vals, axis=-1) / math_ops.reduce_min(vals, axis=-1)

    def cond(self, name='cond'):
        if False:
            for i in range(10):
                print('nop')
        'Returns the condition number of this linear operator.\n\n    Args:\n      name:  A name for this `Op`.\n\n    Returns:\n      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.\n    '
        with self._name_scope(name):
            return self._cond()

    def _can_use_cholesky(self):
        if False:
            return 10
        return self.is_self_adjoint and self.is_positive_definite

    def _set_graph_parents(self, graph_parents):
        if False:
            while True:
                i = 10
        'Set self._graph_parents.  Called during derived class init.\n\n    This method allows derived classes to set graph_parents, without triggering\n    a deprecation warning (which is invoked if `graph_parents` is passed during\n    `__init__`.\n\n    Args:\n      graph_parents: Iterable over Tensors.\n    '
        graph_parents = [] if graph_parents is None else graph_parents
        for (i, t) in enumerate(graph_parents):
            if t is None or not (linear_operator_util.is_ref(t) or tensor_util.is_tf_type(t)):
                raise ValueError('Graph parent item %d is not a Tensor; %s.' % (i, t))
        self._graph_parents = graph_parents

    @property
    def _composite_tensor_fields(self):
        if False:
            i = 10
            return i + 15
        'A tuple of parameter names to rebuild the `LinearOperator`.\n\n    The tuple contains the names of kwargs to the `LinearOperator`\'s constructor\n    that the `TypeSpec` needs to rebuild the `LinearOperator` instance.\n\n    "is_non_singular", "is_self_adjoint", "is_positive_definite", and\n    "is_square" are common to all `LinearOperator` subclasses and may be\n    omitted.\n    '
        return ()

    @property
    def _composite_tensor_prefer_static_fields(self):
        if False:
            while True:
                i = 10
        'A tuple of names referring to parameters that may be treated statically.\n\n    This is a subset of `_composite_tensor_fields`, and contains the names of\n    of `Tensor`-like args to the `LinearOperator`s constructor that may be\n    stored as static values, if they are statically known. These are typically\n    shapes or axis values.\n    '
        return ()

    @property
    def _type_spec(self):
        if False:
            i = 10
            return i + 15
        pass

    def _convert_variables_to_tensors(self):
        if False:
            while True:
                i = 10
        "Recursively converts ResourceVariables in the LinearOperator to Tensors.\n\n    The usage of `self._type_spec._from_components` violates the contract of\n    `CompositeTensor`, since it is called on a different nested structure\n    (one containing only `Tensor`s) than `self.type_spec` specifies (one that\n    may contain `ResourceVariable`s). Since `LinearOperator`'s\n    `_from_components` method just passes the contents of the nested structure\n    to `__init__` to rebuild the operator, and any `LinearOperator` that may be\n    instantiated with `ResourceVariables` may also be instantiated with\n    `Tensor`s, this usage is valid.\n\n    Returns:\n      tensor_operator: `self` with all internal Variables converted to Tensors.\n    "
        components = self._type_spec._to_components(self)
        tensor_components = variable_utils.convert_variables_to_tensors(components)
        return self._type_spec._from_components(tensor_components)

    def __getitem__(self, slices):
        if False:
            while True:
                i = 10
        return slicing.batch_slice(self, params_overrides={}, slices=slices)

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            i = 10
            return i + 15
        'A dict of names to number of dimensions contributing to an operator.\n\n    This is a dictionary of parameter names to `int`s specifying the\n    number of right-most dimensions contributing to the **matrix** shape of the\n    densified operator.\n    If the parameter is a `Tensor`, this is mapped to an `int`.\n    If the parameter is a `LinearOperator` (called `A`), this specifies the\n    number of batch dimensions of `A` contributing to this `LinearOperator`s\n    matrix shape.\n    If the parameter is a structure, this is a structure of the same type of\n    `int`s.\n    '
        return ()
    __composite_gradient__ = _LinearOperatorGradient()

class _LinearOperatorSpec(type_spec.BatchableTypeSpec):
    """A tf.TypeSpec for `LinearOperator` objects."""
    __slots__ = ('_param_specs', '_non_tensor_params', '_prefer_static_fields')

    def __init__(self, param_specs, non_tensor_params, prefer_static_fields):
        if False:
            print('Hello World!')
        "Initializes a new `_LinearOperatorSpec`.\n\n    Args:\n      param_specs: Python `dict` of `tf.TypeSpec` instances that describe\n        kwargs to the `LinearOperator`'s constructor that are `Tensor`-like or\n        `CompositeTensor` subclasses.\n      non_tensor_params: Python `dict` containing non-`Tensor` and non-\n        `CompositeTensor` kwargs to the `LinearOperator`'s constructor.\n      prefer_static_fields: Python `tuple` of strings corresponding to the names\n        of `Tensor`-like args to the `LinearOperator`s constructor that may be\n        stored as static values, if known. These are typically shapes, indices,\n        or axis values.\n    "
        self._param_specs = param_specs
        self._non_tensor_params = non_tensor_params
        self._prefer_static_fields = prefer_static_fields

    @classmethod
    def from_operator(cls, operator):
        if False:
            while True:
                i = 10
        'Builds a `_LinearOperatorSpec` from a `LinearOperator` instance.\n\n    Args:\n      operator: An instance of `LinearOperator`.\n\n    Returns:\n      linear_operator_spec: An instance of `_LinearOperatorSpec` to be used as\n        the `TypeSpec` of `operator`.\n    '
        validation_fields = ('is_non_singular', 'is_self_adjoint', 'is_positive_definite', 'is_square')
        kwargs = _extract_attrs(operator, keys=set(operator._composite_tensor_fields + validation_fields))
        non_tensor_params = {}
        param_specs = {}
        for (k, v) in list(kwargs.items()):
            type_spec_or_v = _extract_type_spec_recursively(v)
            is_tensor = [isinstance(x, type_spec.TypeSpec) for x in nest.flatten(type_spec_or_v)]
            if all(is_tensor):
                param_specs[k] = type_spec_or_v
            elif not any(is_tensor):
                non_tensor_params[k] = v
            else:
                raise NotImplementedError(f'Field {k} contains a mix of `Tensor` and  non-`Tensor` values.')
        return cls(param_specs=param_specs, non_tensor_params=non_tensor_params, prefer_static_fields=operator._composite_tensor_prefer_static_fields)

    def _to_components(self, obj):
        if False:
            i = 10
            return i + 15
        return _extract_attrs(obj, keys=list(self._param_specs))

    def _from_components(self, components):
        if False:
            i = 10
            return i + 15
        kwargs = dict(self._non_tensor_params, **components)
        return self.value_type(**kwargs)

    @property
    def _component_specs(self):
        if False:
            i = 10
            return i + 15
        return self._param_specs

    def _serialize(self):
        if False:
            return 10
        return (self._param_specs, self._non_tensor_params, self._prefer_static_fields)

    def _copy(self, **overrides):
        if False:
            for i in range(10):
                print('nop')
        kwargs = {'param_specs': self._param_specs, 'non_tensor_params': self._non_tensor_params, 'prefer_static_fields': self._prefer_static_fields}
        kwargs.update(overrides)
        return type(self)(**kwargs)

    def _batch(self, batch_size):
        if False:
            print('Hello World!')
        'Returns a TypeSpec representing a batch of objects with this TypeSpec.'
        return self._copy(param_specs=nest.map_structure(lambda spec: spec._batch(batch_size), self._param_specs))

    def _unbatch(self, batch_size):
        if False:
            print('Hello World!')
        'Returns a TypeSpec representing a single element of this TypeSpec.'
        return self._copy(param_specs=nest.map_structure(lambda spec: spec._unbatch(), self._param_specs))

def make_composite_tensor(cls, module_name='tf.linalg'):
    if False:
        print('Hello World!')
    'Class decorator to convert `LinearOperator`s to `CompositeTensor`.'
    spec_name = '{}Spec'.format(cls.__name__)
    spec_type = type(spec_name, (_LinearOperatorSpec,), {'value_type': cls})
    type_spec_registry.register('{}.{}'.format(module_name, spec_name))(spec_type)
    cls._type_spec = property(spec_type.from_operator)
    return cls

def _extract_attrs(op, keys):
    if False:
        return 10
    "Extract constructor kwargs to reconstruct `op`.\n\n  Args:\n    op: A `LinearOperator` instance.\n    keys: A Python `tuple` of strings indicating the names of the constructor\n      kwargs to extract from `op`.\n\n  Returns:\n    kwargs: A Python `dict` of kwargs to `op`'s constructor, keyed by `keys`.\n  "
    kwargs = {}
    not_found = object()
    for k in keys:
        srcs = [getattr(op, k, not_found), getattr(op, '_' + k, not_found), getattr(op, 'parameters', {}).get(k, not_found)]
        if any((v is not not_found for v in srcs)):
            kwargs[k] = [v for v in srcs if v is not not_found][0]
        else:
            raise ValueError(f"Could not determine an appropriate value for field `{k}` in object  `{op}`. Looked for \n 1. an attr called `{k}`,\n 2. an attr called `_{k}`,\n 3. an entry in `op.parameters` with key '{k}'.")
        if k in op._composite_tensor_prefer_static_fields and kwargs[k] is not None:
            if tensor_util.is_tensor(kwargs[k]):
                static_val = tensor_util.constant_value(kwargs[k])
                if static_val is not None:
                    kwargs[k] = static_val
        if isinstance(kwargs[k], (np.ndarray, np.generic)):
            kwargs[k] = kwargs[k].tolist()
    return kwargs

def _extract_type_spec_recursively(value):
    if False:
        return 10
    'Return (collection of) `TypeSpec`(s) for `value` if it includes `Tensor`s.\n\n  If `value` is a `Tensor` or `CompositeTensor`, return its `TypeSpec`. If\n  `value` is a collection containing `Tensor` values, recursively supplant them\n  with their respective `TypeSpec`s in a collection of parallel stucture.\n\n  If `value` is none of the above, return it unchanged.\n\n  Args:\n    value: a Python `object` to (possibly) turn into a (collection of)\n    `tf.TypeSpec`(s).\n\n  Returns:\n    spec: the `TypeSpec` or collection of `TypeSpec`s corresponding to `value`\n    or `value`, if no `Tensor`s are found.\n  '
    if isinstance(value, composite_tensor.CompositeTensor):
        return value._type_spec
    if isinstance(value, variables.Variable):
        return resource_variable_ops.VariableSpec(value.shape, dtype=value.dtype, trainable=value.trainable)
    if tensor_util.is_tensor(value):
        return tensor_spec.TensorSpec(value.shape, value.dtype)
    if isinstance(value, list):
        return list((_extract_type_spec_recursively(v) for v in value))
    if isinstance(value, data_structures.TrackableDataStructure):
        return _extract_type_spec_recursively(value.__wrapped__)
    if isinstance(value, tuple):
        return type(value)((_extract_type_spec_recursively(x) for x in value))
    if isinstance(value, dict):
        return type(value)(((k, _extract_type_spec_recursively(v)) for (k, v) in value.items()))
    return value

@dispatch.dispatch_for_types(linalg.adjoint, LinearOperator)
def _adjoint(matrix, name=None):
    if False:
        for i in range(10):
            print('nop')
    return matrix.adjoint(name)

@dispatch.dispatch_for_types(linalg.cholesky, LinearOperator)
def _cholesky(input, name=None):
    if False:
        i = 10
        return i + 15
    return input.cholesky(name)

@dispatch.dispatch_for_types(linalg.diag_part, LinearOperator)
def _diag_part(input, name='diag_part', k=0, padding_value=0, align='RIGHT_LEFT'):
    if False:
        return 10
    return input.diag_part(name)

@dispatch.dispatch_for_types(linalg.det, LinearOperator)
def _det(input, name=None):
    if False:
        print('Hello World!')
    return input.determinant(name)

@dispatch.dispatch_for_types(linalg.inv, LinearOperator)
def _inverse(input, adjoint=False, name=None):
    if False:
        while True:
            i = 10
    inv = input.inverse(name)
    if adjoint:
        inv = inv.adjoint()
    return inv

@dispatch.dispatch_for_types(linalg.logdet, LinearOperator)
def _logdet(matrix, name=None):
    if False:
        print('Hello World!')
    if matrix.is_positive_definite and matrix.is_self_adjoint:
        return matrix.log_abs_determinant(name)
    raise ValueError('Expected matrix to be self-adjoint positive definite.')

@dispatch.dispatch_for_types(math_ops.matmul, LinearOperator)
def _matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, output_type=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    if transpose_a or transpose_b:
        raise ValueError('Transposing not supported at this time.')
    if a_is_sparse or b_is_sparse:
        raise ValueError('Sparse methods not supported at this time.')
    if not isinstance(a, LinearOperator):
        adjoint_matmul = b.matmul(a, adjoint=not adjoint_b, adjoint_arg=not adjoint_a, name=name)
        return linalg.adjoint(adjoint_matmul)
    return a.matmul(b, adjoint=adjoint_a, adjoint_arg=adjoint_b, name=name)

@dispatch.dispatch_for_types(linalg.solve, LinearOperator)
def _solve(matrix, rhs, adjoint=False, name=None):
    if False:
        i = 10
        return i + 15
    if not isinstance(matrix, LinearOperator):
        raise ValueError('Passing in `matrix` as a Tensor and `rhs` as a LinearOperator is not supported.')
    return matrix.solve(rhs, adjoint=adjoint, name=name)

@dispatch.dispatch_for_types(linalg.trace, LinearOperator)
def _trace(x, name=None):
    if False:
        while True:
            i = 10
    return x.trace(name)