from .mixin_factory import _create_delegating_mixin
BinaryOperand = _create_delegating_mixin('BinaryOperand', 'Mixin encapsulating binary operations.', 'BINARY_OPERATION', '_binaryop', {'__add__', '__sub__', '__mul__', '__matmul__', '__truediv__', '__floordiv__', '__mod__', '__pow__', '__and__', '__xor__', '__or__', '__radd__', '__rsub__', '__rmul__', '__rmatmul__', '__rtruediv__', '__rfloordiv__', '__rmod__', '__rpow__', '__rand__', '__rxor__', '__ror__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__'})

def _binaryop(self, other, op: str):
    if False:
        return 10
    'The core binary_operation function.\n\n    Must be overridden by subclasses, the default implementation raises a\n    NotImplementedError.\n    '
    raise NotImplementedError

def _check_reflected_op(op):
    if False:
        for i in range(10):
            print('nop')
    if (reflect := (op[2] == 'r' and op != '__rshift__')):
        op = op[:2] + op[3:]
    return (reflect, op)
BinaryOperand._binaryop = _binaryop
BinaryOperand._check_reflected_op = staticmethod(_check_reflected_op)