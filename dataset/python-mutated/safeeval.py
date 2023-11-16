from __future__ import division
_const_codes = ['POP_TOP', 'ROT_TWO', 'ROT_THREE', 'ROT_FOUR', 'DUP_TOP', 'BUILD_LIST', 'BUILD_MAP', 'MAP_ADD', 'BUILD_TUPLE', 'BUILD_SET', 'BUILD_CONST_KEY_MAP', 'BUILD_STRING', 'LOAD_CONST', 'RETURN_VALUE', 'STORE_SUBSCR', 'STORE_MAP', 'LIST_TO_TUPLE', 'LIST_EXTEND', 'SET_UPDATE', 'DICT_UPDATE', 'DICT_MERGE', 'COPY', 'RESUME']
_expr_codes = _const_codes + ['UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_NOT', 'UNARY_INVERT', 'BINARY_POWER', 'BINARY_MULTIPLY', 'BINARY_DIVIDE', 'BINARY_FLOOR_DIVIDE', 'BINARY_TRUE_DIVIDE', 'BINARY_MODULO', 'BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_LSHIFT', 'BINARY_RSHIFT', 'BINARY_AND', 'BINARY_XOR', 'BINARY_OR', 'BINARY_OP']
_values_codes = _expr_codes + ['LOAD_NAME']
import six

def _get_opcodes(codeobj):
    if False:
        for i in range(10):
            print('nop')
    '_get_opcodes(codeobj) -> [opcodes]\n\n    Extract the actual opcodes as a list from a code object\n\n    >>> c = compile("[1 + 2, (1,2)]", "", "eval")\n    >>> _get_opcodes(c)\n    [100, 100, 103, 83]\n    '
    import dis
    if hasattr(dis, 'get_instructions'):
        return [ins.opcode for ins in dis.get_instructions(codeobj)]
    i = 0
    opcodes = []
    s = codeobj.co_code
    while i < len(s):
        code = six.indexbytes(s, i)
        opcodes.append(code)
        if code >= dis.HAVE_ARGUMENT:
            i += 3
        else:
            i += 1
    return opcodes

def test_expr(expr, allowed_codes):
    if False:
        print('Hello World!')
    'test_expr(expr, allowed_codes) -> codeobj\n\n    Test that the expression contains only the listed opcodes.\n    If the expression is valid and contains only allowed codes,\n    return the compiled code object. Otherwise raise a ValueError\n    '
    import dis
    allowed_codes = [dis.opmap[c] for c in allowed_codes if c in dis.opmap]
    try:
        c = compile(expr, '', 'eval')
    except SyntaxError:
        raise ValueError('%r is not a valid expression' % expr)
    codes = _get_opcodes(c)
    for code in codes:
        if code not in allowed_codes:
            raise ValueError('opcode %s not allowed' % dis.opname[code])
    return c

def const(expr):
    if False:
        i = 10
        return i + 15
    'const(expression) -> value\n\n    Safe Python constant evaluation\n\n    Evaluates a string that contains an expression describing\n    a Python constant. Strings that are not valid Python expressions\n    or that contain other code besides the constant raise ValueError.\n\n    Examples:\n\n        >>> const("10")\n        10\n        >>> const("[1,2, (3,4), {\'foo\':\'bar\'}]")\n        [1, 2, (3, 4), {\'foo\': \'bar\'}]\n        >>> const("[1]+[2]")\n        Traceback (most recent call last):\n        ...\n        ValueError: opcode BINARY_ADD not allowed\n    '
    c = test_expr(expr, _const_codes)
    return eval(c)

def expr(expr):
    if False:
        i = 10
        return i + 15
    'expr(expression) -> value\n\n    Safe Python expression evaluation\n\n    Evaluates a string that contains an expression that only\n    uses Python constants. This can be used to e.g. evaluate\n    a numerical expression from an untrusted source.\n\n    Examples:\n\n        >>> expr("1+2")\n        3\n        >>> expr("[1,2]*2")\n        [1, 2, 1, 2]\n        >>> expr("__import__(\'sys\').modules")\n        Traceback (most recent call last):\n        ...\n        ValueError: opcode LOAD_NAME not allowed\n    '
    c = test_expr(expr, _expr_codes)
    return eval(c)

def values(expr, env):
    if False:
        print('Hello World!')
    'values(expression, dict) -> value\n\n    Safe Python expression evaluation\n\n    Evaluates a string that contains an expression that only\n    uses Python constants and values from a supplied dictionary.\n    This can be used to e.g. evaluate e.g. an argument to a syscall.\n\n    Note: This is potentially unsafe if e.g. the __add__ method has side\n          effects.\n\n    Examples:\n\n        >>> values("A + 4", {\'A\': 6})\n        10\n        >>> class Foo:\n        ...    def __add__(self, other):\n        ...        print("Firing the missiles")\n        >>> values("A + 1", {\'A\': Foo()})\n        Firing the missiles\n        >>> values("A.x", {\'A\': Foo()})\n        Traceback (most recent call last):\n        ...\n        ValueError: opcode LOAD_ATTR not allowed\n    '
    env = dict(env)
    env['__builtins__'] = {}
    c = test_expr(expr, _values_codes)
    return eval(c, env)