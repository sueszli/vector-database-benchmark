"""Schema utilities to get builtin code from operator code."""
from tensorflow.python.util import all_util

def get_builtin_code_from_operator_code(opcode):
    if False:
        while True:
            i = 10
    'Return the builtin code of the given operator code.\n\n  The following method is introduced to resolve op builtin code shortage\n  problem. The new builtin operator will be assigned to the extended builtin\n  code field in the flatbuffer schema. Those methods helps to hide builtin code\n  details.\n\n  Args:\n    opcode: Operator code.\n\n  Returns:\n    The builtin code of the given operator code.\n  '
    if hasattr(opcode, 'BuiltinCode') and callable(opcode.BuiltinCode):
        return max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())
    return max(opcode.builtinCode, opcode.deprecatedBuiltinCode)
_allowed_symbols = ['get_builtin_code_from_operator_code']
all_util.remove_undocumented(__name__, _allowed_symbols)