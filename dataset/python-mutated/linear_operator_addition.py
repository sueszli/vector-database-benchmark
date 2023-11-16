"""Add one or more `LinearOperators` efficiently."""
import abc
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
__all__ = []

def add_operators(operators, operator_name=None, addition_tiers=None, name=None):
    if False:
        print('Hello World!')
    'Efficiently add one or more linear operators.\n\n  Given operators `[A1, A2,...]`, this `Op` returns a possibly shorter list of\n  operators `[B1, B2,...]` such that\n\n  ```sum_k Ak.matmul(x) = sum_k Bk.matmul(x).```\n\n  The operators `Bk` result by adding some of the `Ak`, as allowed by\n  `addition_tiers`.\n\n  Example of efficient adding of diagonal operators.\n\n  ```python\n  A1 = LinearOperatorDiag(diag=[1., 1.], name="A1")\n  A2 = LinearOperatorDiag(diag=[2., 2.], name="A2")\n\n  # Use two tiers, the first contains an Adder that returns Diag.  Since both\n  # A1 and A2 are Diag, they can use this Adder.  The second tier will not be\n  # used.\n  addition_tiers = [\n      [_AddAndReturnDiag()],\n      [_AddAndReturnMatrix()]]\n  B_list = add_operators([A1, A2], addition_tiers=addition_tiers)\n\n  len(B_list)\n  ==> 1\n\n  B_list[0].__class__.__name__\n  ==> \'LinearOperatorDiag\'\n\n  B_list[0].to_dense()\n  ==> [[3., 0.],\n       [0., 3.]]\n\n  B_list[0].name\n  ==> \'Add/A1__A2/\'\n  ```\n\n  Args:\n    operators:  Iterable of `LinearOperator` objects with same `dtype`, domain\n      and range dimensions, and broadcastable batch shapes.\n    operator_name:  String name for returned `LinearOperator`.  Defaults to\n      concatenation of "Add/A__B/" that indicates the order of addition steps.\n    addition_tiers:  List tiers, like `[tier_0, tier_1, ...]`, where `tier_i`\n      is a list of `Adder` objects.  This function attempts to do all additions\n      in tier `i` before trying tier `i + 1`.\n    name:  A name for this `Op`.  Defaults to `add_operators`.\n\n  Returns:\n    Subclass of `LinearOperator`.  Class and order of addition may change as new\n      (and better) addition strategies emerge.\n\n  Raises:\n    ValueError:  If `operators` argument is empty.\n    ValueError:  If shapes are incompatible.\n  '
    if addition_tiers is None:
        addition_tiers = _DEFAULT_ADDITION_TIERS
    check_ops.assert_proper_iterable(operators)
    operators = list(reversed(operators))
    if len(operators) < 1:
        raise ValueError(f'Argument `operators` must contain at least one operator. Received: {operators}.')
    if not all((isinstance(op, linear_operator.LinearOperator) for op in operators)):
        raise TypeError(f'Argument `operators` must contain only LinearOperator instances. Received: {operators}.')
    _static_check_for_same_dimensions(operators)
    _static_check_for_broadcastable_batch_shape(operators)
    with ops.name_scope(name or 'add_operators'):
        ops_to_try_at_next_tier = list(operators)
        for tier in addition_tiers:
            ops_to_try_at_this_tier = ops_to_try_at_next_tier
            ops_to_try_at_next_tier = []
            while ops_to_try_at_this_tier:
                op1 = ops_to_try_at_this_tier.pop()
                (op2, adder) = _pop_a_match_at_tier(op1, ops_to_try_at_this_tier, tier)
                if op2 is not None:
                    new_operator = adder.add(op1, op2, operator_name)
                    ops_to_try_at_this_tier.append(new_operator)
                else:
                    ops_to_try_at_next_tier.append(op1)
        return ops_to_try_at_next_tier

def _pop_a_match_at_tier(op1, operator_list, tier):
    if False:
        while True:
            i = 10
    for i in range(1, len(operator_list) + 1):
        op2 = operator_list[-i]
        for adder in tier:
            if adder.can_add(op1, op2):
                return (operator_list.pop(-i), adder)
    return (None, None)

def _infer_hints_allowing_override(op1, op2, hints):
    if False:
        for i in range(10):
            print('nop')
    'Infer hints from op1 and op2.  hints argument is an override.\n\n  Args:\n    op1:  LinearOperator\n    op2:  LinearOperator\n    hints:  _Hints object holding "is_X" boolean hints to use for returned\n      operator.\n      If some hint is None, try to set using op1 and op2.  If the\n      hint is provided, ignore op1 and op2 hints.  This allows an override\n      of previous hints, but does not allow forbidden hints (e.g. you still\n      cannot say a real diagonal operator is not self-adjoint.\n\n  Returns:\n    _Hints object.\n  '
    hints = hints or _Hints()
    if hints.is_self_adjoint is None:
        is_self_adjoint = op1.is_self_adjoint and op2.is_self_adjoint
    else:
        is_self_adjoint = hints.is_self_adjoint
    if hints.is_positive_definite is None:
        is_positive_definite = op1.is_positive_definite and op2.is_positive_definite
    else:
        is_positive_definite = hints.is_positive_definite
    if is_positive_definite and hints.is_positive_definite is None:
        is_non_singular = True
    else:
        is_non_singular = hints.is_non_singular
    return _Hints(is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite)

def _static_check_for_same_dimensions(operators):
    if False:
        for i in range(10):
            print('nop')
    'ValueError if operators determined to have different dimensions.'
    if len(operators) < 2:
        return
    domain_dimensions = [(op.name, tensor_shape.dimension_value(op.domain_dimension)) for op in operators if tensor_shape.dimension_value(op.domain_dimension) is not None]
    if len(set((value for (name, value) in domain_dimensions))) > 1:
        raise ValueError(f'All `operators` must have the same `domain_dimension`. Received: {domain_dimensions}.')
    range_dimensions = [(op.name, tensor_shape.dimension_value(op.range_dimension)) for op in operators if tensor_shape.dimension_value(op.range_dimension) is not None]
    if len(set((value for (name, value) in range_dimensions))) > 1:
        raise ValueError(f'All operators must have the same `range_dimension`. Received: {range_dimensions}.')

def _static_check_for_broadcastable_batch_shape(operators):
    if False:
        return 10
    'ValueError if operators determined to have non-broadcastable shapes.'
    if len(operators) < 2:
        return
    batch_shape = operators[0].batch_shape
    for op in operators[1:]:
        batch_shape = array_ops.broadcast_static_shape(batch_shape, op.batch_shape)

class _Hints:
    """Holds 'is_X' flags that every LinearOperator is initialized with."""

    def __init__(self, is_non_singular=None, is_positive_definite=None, is_self_adjoint=None):
        if False:
            i = 10
            return i + 15
        self.is_non_singular = is_non_singular
        self.is_positive_definite = is_positive_definite
        self.is_self_adjoint = is_self_adjoint

class _Adder(metaclass=abc.ABCMeta):
    """Abstract base class to add two operators.

  Each `Adder` acts independently, adding everything it can, paying no attention
  as to whether another `Adder` could have done the addition more efficiently.
  """

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__

    @abc.abstractmethod
    def can_add(self, op1, op2):
        if False:
            i = 10
            return i + 15
        'Returns `True` if this `Adder` can add `op1` and `op2`.  Else `False`.'
        pass

    @abc.abstractmethod
    def _add(self, op1, op2, operator_name, hints):
        if False:
            for i in range(10):
                print('nop')
        pass

    def add(self, op1, op2, operator_name, hints=None):
        if False:
            for i in range(10):
                print('nop')
        'Return new `LinearOperator` acting like `op1 + op2`.\n\n    Args:\n      op1:  `LinearOperator`\n      op2:  `LinearOperator`, with `shape` and `dtype` such that adding to\n        `op1` is allowed.\n      operator_name:  `String` name to give to returned `LinearOperator`\n      hints:  `_Hints` object.  Returned `LinearOperator` will be created with\n        these hints.\n\n    Returns:\n      `LinearOperator`\n    '
        updated_hints = _infer_hints_allowing_override(op1, op2, hints)
        if operator_name is None:
            operator_name = 'Add/' + op1.name + '__' + op2.name + '/'
        scope_name = self.name
        if scope_name.startswith('_'):
            scope_name = scope_name[1:]
        with ops.name_scope(scope_name):
            return self._add(op1, op2, operator_name, updated_hints)

class _AddAndReturnScaledIdentity(_Adder):
    """Handles additions resulting in an Identity family member.

  The Identity (`LinearOperatorScaledIdentity`, `LinearOperatorIdentity`) family
  is closed under addition.  This `Adder` respects that, and returns an Identity
  """

    def can_add(self, op1, op2):
        if False:
            return 10
        types = {_type(op1), _type(op2)}
        return not types.difference(_IDENTITY_FAMILY)

    def _add(self, op1, op2, operator_name, hints):
        if False:
            return 10
        if _type(op1) == _SCALED_IDENTITY:
            multiplier_1 = op1.multiplier
        else:
            multiplier_1 = array_ops.ones(op1.batch_shape_tensor(), dtype=op1.dtype)
        if _type(op2) == _SCALED_IDENTITY:
            multiplier_2 = op2.multiplier
        else:
            multiplier_2 = array_ops.ones(op2.batch_shape_tensor(), dtype=op2.dtype)
        return linear_operator_identity.LinearOperatorScaledIdentity(num_rows=op1.range_dimension_tensor(), multiplier=multiplier_1 + multiplier_2, is_non_singular=hints.is_non_singular, is_self_adjoint=hints.is_self_adjoint, is_positive_definite=hints.is_positive_definite, name=operator_name)

class _AddAndReturnDiag(_Adder):
    """Handles additions resulting in a Diag operator."""

    def can_add(self, op1, op2):
        if False:
            print('Hello World!')
        types = {_type(op1), _type(op2)}
        return not types.difference(_DIAG_LIKE)

    def _add(self, op1, op2, operator_name, hints):
        if False:
            return 10
        return linear_operator_diag.LinearOperatorDiag(diag=op1.diag_part() + op2.diag_part(), is_non_singular=hints.is_non_singular, is_self_adjoint=hints.is_self_adjoint, is_positive_definite=hints.is_positive_definite, name=operator_name)

class _AddAndReturnTriL(_Adder):
    """Handles additions resulting in a TriL operator."""

    def can_add(self, op1, op2):
        if False:
            i = 10
            return i + 15
        types = {_type(op1), _type(op2)}
        return not types.difference(_DIAG_LIKE.union({_TRIL}))

    def _add(self, op1, op2, operator_name, hints):
        if False:
            while True:
                i = 10
        if _type(op1) in _EFFICIENT_ADD_TO_TENSOR:
            (op_add_to_tensor, op_other) = (op1, op2)
        else:
            (op_add_to_tensor, op_other) = (op2, op1)
        return linear_operator_lower_triangular.LinearOperatorLowerTriangular(tril=op_add_to_tensor.add_to_tensor(op_other.to_dense()), is_non_singular=hints.is_non_singular, is_self_adjoint=hints.is_self_adjoint, is_positive_definite=hints.is_positive_definite, name=operator_name)

class _AddAndReturnMatrix(_Adder):
    """"Handles additions resulting in a `LinearOperatorFullMatrix`."""

    def can_add(self, op1, op2):
        if False:
            i = 10
            return i + 15
        return isinstance(op1, linear_operator.LinearOperator) and isinstance(op2, linear_operator.LinearOperator)

    def _add(self, op1, op2, operator_name, hints):
        if False:
            print('Hello World!')
        if _type(op1) in _EFFICIENT_ADD_TO_TENSOR:
            (op_add_to_tensor, op_other) = (op1, op2)
        else:
            (op_add_to_tensor, op_other) = (op2, op1)
        return linear_operator_full_matrix.LinearOperatorFullMatrix(matrix=op_add_to_tensor.add_to_tensor(op_other.to_dense()), is_non_singular=hints.is_non_singular, is_self_adjoint=hints.is_self_adjoint, is_positive_definite=hints.is_positive_definite, name=operator_name)
_IDENTITY = 'identity'
_SCALED_IDENTITY = 'scaled_identity'
_DIAG = 'diag'
_TRIL = 'tril'
_MATRIX = 'matrix'
_DIAG_LIKE = {_DIAG, _IDENTITY, _SCALED_IDENTITY}
_IDENTITY_FAMILY = {_IDENTITY, _SCALED_IDENTITY}
_EFFICIENT_ADD_TO_TENSOR = _DIAG_LIKE
SUPPORTED_OPERATORS = [linear_operator_diag.LinearOperatorDiag, linear_operator_lower_triangular.LinearOperatorLowerTriangular, linear_operator_full_matrix.LinearOperatorFullMatrix, linear_operator_identity.LinearOperatorIdentity, linear_operator_identity.LinearOperatorScaledIdentity]

def _type(operator):
    if False:
        return 10
    'Returns the type name constant (e.g. _TRIL) for operator.'
    if isinstance(operator, linear_operator_diag.LinearOperatorDiag):
        return _DIAG
    if isinstance(operator, linear_operator_lower_triangular.LinearOperatorLowerTriangular):
        return _TRIL
    if isinstance(operator, linear_operator_full_matrix.LinearOperatorFullMatrix):
        return _MATRIX
    if isinstance(operator, linear_operator_identity.LinearOperatorIdentity):
        return _IDENTITY
    if isinstance(operator, linear_operator_identity.LinearOperatorScaledIdentity):
        return _SCALED_IDENTITY
    raise TypeError(f'Expected operator to be one of [LinearOperatorDiag, LinearOperatorLowerTriangular, LinearOperatorFullMatrix, LinearOperatorIdentity, LinearOperatorScaledIdentity]. Received: {operator}')
_DEFAULT_ADDITION_TIERS = [[_AddAndReturnScaledIdentity()], [_AddAndReturnDiag()], [_AddAndReturnTriL()], [_AddAndReturnMatrix()]]