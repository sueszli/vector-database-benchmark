""" Python operator tables

These are mostly used to resolve the operator in the module operator and to know the list
of operations allowed.

"""
import operator
from nuitka.PythonVersions import python_version
binary_operator_functions = {'Add': operator.add, 'Sub': operator.sub, 'Pow': operator.pow, 'Mult': operator.mul, 'FloorDiv': operator.floordiv, 'TrueDiv': operator.truediv, 'Mod': operator.mod, 'LShift': operator.lshift, 'RShift': operator.rshift, 'BitAnd': operator.and_, 'BitOr': operator.or_, 'BitXor': operator.xor, 'Divmod': divmod, 'IAdd': operator.iadd, 'ISub': operator.isub, 'IPow': operator.ipow, 'IMult': operator.imul, 'IFloorDiv': operator.ifloordiv, 'ITrueDiv': operator.itruediv, 'IMod': operator.imod, 'ILShift': operator.ilshift, 'IRShift': operator.irshift, 'IBitAnd': operator.iand, 'IBitOr': operator.ior, 'IBitXor': operator.ixor}
if python_version < 768:
    binary_operator_functions['OldDiv'] = operator.div
    binary_operator_functions['IOldDiv'] = operator.idiv
if python_version >= 848:
    binary_operator_functions['MatMult'] = operator.matmul
    binary_operator_functions['IMatMult'] = operator.imatmul
unary_operator_functions = {'UAdd': operator.pos, 'USub': operator.neg, 'Invert': operator.invert, 'Repr': repr, 'Not': operator.not_, 'Abs': operator.abs}
rich_comparison_functions = {'Lt': operator.lt, 'LtE': operator.le, 'Eq': operator.eq, 'NotEq': operator.ne, 'Gt': operator.gt, 'GtE': operator.ge}
other_comparison_functions = {'Is': operator.is_, 'IsNot': operator.is_not, 'In': lambda value1, value2: value1 in value2, 'NotIn': lambda value1, value2: value1 not in value2}
comparison_inversions = {'Is': 'IsNot', 'IsNot': 'Is', 'In': 'NotIn', 'NotIn': 'In', 'Lt': 'GtE', 'GtE': 'Lt', 'Eq': 'NotEq', 'NotEq': 'Eq', 'Gt': 'LtE', 'LtE': 'Gt', 'exception_match': 'exception_mismatch', 'exception_mismatch': 'exception_match'}
rich_comparison_arg_swaps = {'Lt': 'Gt', 'GtE': 'LtE', 'Eq': 'Eq', 'NotEq': 'NotEq', 'Gt': 'Lt', 'LtE': 'GtE'}
all_comparison_functions = dict(rich_comparison_functions)
all_comparison_functions.update(other_comparison_functions)

def matchException(left, right):
    if False:
        i = 10
        return i + 15
    if python_version >= 768:
        if type(right) is tuple:
            for element in right:
                if not isinstance(BaseException, element):
                    raise TypeError('catching classes that do not inherit from BaseException is not allowed')
        elif not isinstance(BaseException, right):
            raise TypeError('catching classes that do not inherit from BaseException is not allowed')
    import os
    os._exit(16)
all_comparison_functions['exception_match'] = matchException