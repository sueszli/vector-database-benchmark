"""
Process raw qstr file and output qstr data with length, hash and data bytes.

This script works with Python 2.6, 2.7, 3.3 and 3.4.
"""
from __future__ import print_function
import re
import sys
import platform
if platform.python_version_tuple()[0] == '2':
    bytes_cons = lambda val, enc=None: bytearray(val)
    from htmlentitydefs import codepoint2name
elif platform.python_version_tuple()[0] == '3':
    bytes_cons = bytes
    from html.entities import codepoint2name
codepoint2name[ord('-')] = 'hyphen'
codepoint2name[ord(' ')] = 'space'
codepoint2name[ord("'")] = 'squot'
codepoint2name[ord(',')] = 'comma'
codepoint2name[ord('.')] = 'dot'
codepoint2name[ord(':')] = 'colon'
codepoint2name[ord(';')] = 'semicolon'
codepoint2name[ord('/')] = 'slash'
codepoint2name[ord('%')] = 'percent'
codepoint2name[ord('#')] = 'hash'
codepoint2name[ord('(')] = 'paren_open'
codepoint2name[ord(')')] = 'paren_close'
codepoint2name[ord('[')] = 'bracket_open'
codepoint2name[ord(']')] = 'bracket_close'
codepoint2name[ord('{')] = 'brace_open'
codepoint2name[ord('}')] = 'brace_close'
codepoint2name[ord('*')] = 'star'
codepoint2name[ord('!')] = 'bang'
codepoint2name[ord('\\')] = 'backslash'
codepoint2name[ord('+')] = 'plus'
codepoint2name[ord('$')] = 'dollar'
codepoint2name[ord('=')] = 'equals'
codepoint2name[ord('?')] = 'question'
codepoint2name[ord('@')] = 'at_sign'
codepoint2name[ord('^')] = 'caret'
codepoint2name[ord('|')] = 'pipe'
codepoint2name[ord('~')] = 'tilde'
static_qstr_list = ['', '__dir__', '\n', ' ', '*', '/', '<module>', '_', '__call__', '__class__', '__delitem__', '__enter__', '__exit__', '__getattr__', '__getitem__', '__hash__', '__init__', '__int__', '__iter__', '__len__', '__main__', '__module__', '__name__', '__new__', '__next__', '__qualname__', '__repr__', '__setitem__', '__str__', 'ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'EOFError', 'Ellipsis', 'Exception', 'GeneratorExit', 'ImportError', 'IndentationError', 'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'NameError', 'NoneType', 'NotImplementedError', 'OSError', 'OverflowError', 'RuntimeError', 'StopIteration', 'SyntaxError', 'SystemExit', 'TypeError', 'ValueError', 'ZeroDivisionError', 'abs', 'all', 'any', 'append', 'args', 'bool', 'builtins', 'bytearray', 'bytecode', 'bytes', 'callable', 'chr', 'classmethod', 'clear', 'close', 'const', 'copy', 'count', 'dict', 'dir', 'divmod', 'end', 'endswith', 'eval', 'exec', 'extend', 'find', 'format', 'from_bytes', 'get', 'getattr', 'globals', 'hasattr', 'hash', 'id', 'index', 'insert', 'int', 'isalpha', 'isdigit', 'isinstance', 'islower', 'isspace', 'issubclass', 'isupper', 'items', 'iter', 'join', 'key', 'keys', 'len', 'list', 'little', 'locals', 'lower', 'lstrip', 'main', 'map', 'micropython', 'next', 'object', 'open', 'ord', 'pop', 'popitem', 'pow', 'print', 'range', 'read', 'readinto', 'readline', 'remove', 'replace', 'repr', 'reverse', 'rfind', 'rindex', 'round', 'rsplit', 'rstrip', 'self', 'send', 'sep', 'set', 'setattr', 'setdefault', 'sort', 'sorted', 'split', 'start', 'startswith', 'staticmethod', 'step', 'stop', 'str', 'strip', 'sum', 'super', 'throw', 'to_bytes', 'tuple', 'type', 'update', 'upper', 'utf-8', 'value', 'values', 'write', 'zip']
operator_qstr_list = {'__bool__', '__pos__', '__neg__', '__invert__', '__abs__', '__float__', '__complex__', '__sizeof__', '__lt__', '__gt__', '__eq__', '__le__', '__ge__', '__ne__', '__contains__', '__iadd__', '__isub__', '__imul__', '__imatmul__', '__ifloordiv__', '__itruediv__', '__imod__', '__ipow__', '__ior__', '__ixor__', '__iand__', '__ilshift__', '__irshift__', '__add__', '__sub__', '__mul__', '__matmul__', '__floordiv__', '__truediv__', '__mod__', '__divmod__', '__pow__', '__or__', '__xor__', '__and__', '__lshift__', '__rshift__', '__radd__', '__rsub__', '__rmul__', '__rmatmul__', '__rfloordiv__', '__rtruediv__', '__rmod__', '__rpow__', '__ror__', '__rxor__', '__rand__', '__rlshift__', '__rrshift__', '__get__', '__set__', '__delete__'}

def compute_hash(qstr, bytes_hash):
    if False:
        print('Hello World!')
    hash = 5381
    for b in qstr:
        hash = hash * 33 ^ b
    return hash & (1 << 8 * bytes_hash) - 1 or 1

def qstr_escape(qst):
    if False:
        return 10

    def esc_char(m):
        if False:
            while True:
                i = 10
        c = ord(m.group(0))
        try:
            name = codepoint2name[c]
        except KeyError:
            name = '0x%02x' % c
        return '_' + name + '_'
    return re.sub('[^A-Za-z0-9_]', esc_char, qst)
static_qstr_list_ident = list(map(qstr_escape, static_qstr_list))

def parse_input_headers(infiles):
    if False:
        i = 10
        return i + 15
    qcfgs = {}
    qstrs = {}
    for infile in infiles:
        with open(infile, 'rt') as f:
            for line in f:
                line = line.strip()
                match = re.match('^QCFG\\((.+), (.+)\\)', line)
                if match:
                    value = match.group(2)
                    if value[0] == '(' and value[-1] == ')':
                        value = value[1:-1]
                    qcfgs[match.group(1)] = value
                    continue
                match = re.match('^Q\\((.*)\\)$', line)
                if not match:
                    continue
                qstr = match.group(1)
                if qstr == '\\n':
                    qstr = '\n'
                elif qstr == '\\r\\n':
                    qstr = '\r\n'
                ident = qstr_escape(qstr)
                if ident in static_qstr_list_ident:
                    continue
                if ident in qstrs:
                    continue
                qstrs[ident] = (ident, qstr)
    if not qcfgs:
        sys.stderr.write('ERROR: Empty preprocessor output - check for errors above\n')
        sys.exit(1)
    return (qcfgs, qstrs)

def escape_bytes(qstr, qbytes):
    if False:
        while True:
            i = 10
    if all((32 <= ord(c) <= 126 and c != '\\' and (c != '"') for c in qstr)):
        return qstr
    else:
        return ''.join(('\\x%02x' % b for b in qbytes))

def make_bytes(cfg_bytes_len, cfg_bytes_hash, qstr):
    if False:
        for i in range(10):
            print('nop')
    qbytes = bytes_cons(qstr, 'utf8')
    qlen = len(qbytes)
    qhash = compute_hash(qbytes, cfg_bytes_hash)
    if qlen >= 1 << 8 * cfg_bytes_len:
        print('qstr is too long:', qstr)
        assert False
    qdata = escape_bytes(qstr, qbytes)
    return '%d, %d, "%s"' % (qhash, qlen, qdata)

def print_qstr_data(qcfgs, qstrs):
    if False:
        return 10
    cfg_bytes_len = int(qcfgs['BYTES_IN_LEN'])
    cfg_bytes_hash = int(qcfgs['BYTES_IN_HASH'])
    print('// This file was automatically generated by makeqstrdata.py')
    print('')
    print('QDEF0(MP_QSTRnull, 0, 0, "")')
    for qstr in static_qstr_list:
        qbytes = make_bytes(cfg_bytes_len, cfg_bytes_hash, qstr)
        print('QDEF0(MP_QSTR_%s, %s)' % (qstr_escape(qstr), qbytes))
    for (ident, qstr) in sorted(qstrs.values(), key=lambda x: x[1]):
        qbytes = make_bytes(cfg_bytes_len, cfg_bytes_hash, qstr)
        pool = 0 if qstr in operator_qstr_list else 1
        print('QDEF%d(MP_QSTR_%s, %s)' % (pool, ident, qbytes))

def do_work(infiles):
    if False:
        for i in range(10):
            print('nop')
    (qcfgs, qstrs) = parse_input_headers(infiles)
    print_qstr_data(qcfgs, qstrs)
if __name__ == '__main__':
    do_work(sys.argv[1:])