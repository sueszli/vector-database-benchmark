"""
Python 3.6 bytecode decompiler scanner

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.

This sets up opcodes Python's 3.6 and calls a generalized
scanner routine for Python 3.
"""
from __future__ import print_function
from uncompyle6.scanners.scanner3 import Scanner3
from xdis.opcodes import opcode_36 as opc
JUMP_OPS = opc.JUMP_OPS

class Scanner36(Scanner3):

    def __init__(self, show_asm=None, is_pypy=False):
        if False:
            return 10
        Scanner3.__init__(self, (3, 6), show_asm, is_pypy)
        return

    def ingest(self, co, classname=None, code_objects={}, show_asm=None):
        if False:
            while True:
                i = 10
        '\n        Create "tokens" the bytecode of an Python code object. Largely these\n        are the opcode name, but in some cases that has been modified to make parsing\n        easier.\n        returning a list of uncompyle6 Token\'s.\n\n        Some transformations are made to assist the deparsing grammar:\n           -  various types of LOAD_CONST\'s are categorized in terms of what they load\n           -  COME_FROM instructions are added to assist parsing control structures\n           -  operands with stack argument counts or flag masks are appended to the opcode name, e.g.:\n              *  BUILD_LIST, BUILD_SET\n              *  MAKE_FUNCTION and FUNCTION_CALLS append the number of positional arguments\n           -  EXTENDED_ARGS instructions are removed\n\n        Also, when we encounter certain tokens, we add them to a set which will cause custom\n        grammar rules. Specifically, variable arg tokens like MAKE_FUNCTION or BUILD_LIST\n        cause specific rules for the specific number of arguments they take.\n        '
        (tokens, customize) = Scanner3.ingest(self, co, classname, code_objects, show_asm)
        not_pypy36 = not (self.version[:2] == (3, 6) and self.is_pypy)
        for t in tokens:
            if not_pypy36 and t.op == self.opc.CALL_FUNCTION_EX and t.attr & 1:
                t.kind = 'CALL_FUNCTION_EX_KW'
                pass
            elif t.op == self.opc.BUILD_STRING:
                t.kind = 'BUILD_STRING_%s' % t.attr
            elif t.op == self.opc.CALL_FUNCTION_KW:
                t.kind = 'CALL_FUNCTION_KW_%s' % t.attr
            elif t.op == self.opc.FORMAT_VALUE:
                if t.attr & 4:
                    t.kind = 'FORMAT_VALUE_ATTR'
                    pass
            elif not_pypy36 and t.op == self.opc.BUILD_MAP_UNPACK_WITH_CALL:
                t.kind = 'BUILD_MAP_UNPACK_WITH_CALL_%d' % t.attr
            elif not_pypy36 and t.op == self.opc.BUILD_TUPLE_UNPACK_WITH_CALL:
                t.kind = 'BUILD_TUPLE_UNPACK_WITH_CALL_%d' % t.attr
            pass
        return (tokens, customize)
if __name__ == '__main__':
    from xdis.version_info import PYTHON_VERSION_TRIPLE, version_tuple_to_str
    if PYTHON_VERSION_TRIPLE[:2] == (3, 6):
        import inspect
        co = inspect.currentframe().f_code
        (tokens, customize) = Scanner36().ingest(co)
        for t in tokens:
            print(t.format())
        pass
    else:
        print('Need to be Python 3.6 to demo; I am version %s' % version_tuple_to_str())