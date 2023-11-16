"""
Python 3.7 bytecode decompiler scanner.

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.

This sets up opcodes Python's 3.7 and calls a generalized
scanner routine for Python 3.
"""
from typing import Tuple
from uncompyle6.scanner import CONST_COLLECTIONS, Token
from uncompyle6.scanners.scanner37base import Scanner37Base
from xdis.opcodes import opcode_37 as opc
JUMP_OPs = opc.JUMP_OPS

class Scanner37(Scanner37Base):

    def __init__(self, show_asm=None, debug='', is_pypy=False):
        if False:
            return 10
        Scanner37Base.__init__(self, (3, 7), show_asm, debug, is_pypy)
        self.debug = debug
        return
    pass

    def bound_collection_from_tokens(self, tokens: list, next_tokens: list, t: Token, i: int, collection_type: str) -> list:
        if False:
            for i in range(10):
                print('nop')
        count = t.attr
        assert isinstance(count, int)
        assert count <= i
        if collection_type == 'CONST_DICT':
            count += 1
        if count < 5:
            return next_tokens + [t]
        collection_start = i - count
        for j in range(collection_start, i):
            if tokens[j].kind not in ('LOAD_CODE', 'LOAD_CONST', 'LOAD_FAST', 'LOAD_GLOBAL', 'LOAD_NAME', 'LOAD_STR'):
                return next_tokens + [t]
        collection_enum = CONST_COLLECTIONS.index(collection_type)
        new_tokens = next_tokens[:-count]
        start_offset = tokens[collection_start].offset
        new_tokens.append(Token(opname='COLLECTION_START', attr=collection_enum, pattr=collection_type, offset=f'{start_offset}_0', linestart=False, has_arg=True, has_extended_arg=False, opc=self.opc))
        for j in range(collection_start, i):
            new_tokens.append(Token(opname='ADD_VALUE', attr=tokens[j].attr, pattr=tokens[j].pattr, offset=tokens[j].offset, linestart=tokens[j].linestart, has_arg=True, has_extended_arg=False, opc=self.opc))
        new_tokens.append(Token(opname=f'BUILD_{collection_type}', attr=t.attr, pattr=t.pattr, offset=t.offset, linestart=t.linestart, has_arg=t.has_arg, has_extended_arg=False, opc=t.opc))
        return new_tokens

    def ingest(self, bytecode, classname=None, code_objects={}, show_asm=None) -> Tuple[list, dict]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create "tokens" the bytecode of an Python code object. Largely these\n        are the opcode name, but in some cases that has been modified to make parsing\n        easier.\n        returning a list of uncompyle6 Token\'s.\n\n        Some transformations are made to assist the deparsing grammar:\n           -  various types of LOAD_CONST\'s are categorized in terms of what they load\n           -  COME_FROM instructions are added to assist parsing control structures\n           -  operands with stack argument counts or flag masks are appended to the opcode name, e.g.:\n              *  BUILD_LIST, BUILD_SET\n              *  MAKE_FUNCTION and FUNCTION_CALLS append the number of positional arguments\n           -  EXTENDED_ARGS instructions are removed\n\n        Also, when we encounter certain tokens, we add them to a set which will cause custom\n        grammar rules. Specifically, variable arg tokens like MAKE_FUNCTION or BUILD_LIST\n        cause specific rules for the specific number of arguments they take.\n        '
        (tokens, customize) = Scanner37Base.ingest(self, bytecode, classname, code_objects, show_asm)
        new_tokens = []
        for (i, t) in enumerate(tokens):
            if t.op in (self.opc.BUILD_CONST_KEY_MAP, self.opc.BUILD_LIST, self.opc.BUILD_SET):
                collection_type = 'DICT' if t.kind.startswith('BUILD_CONST_KEY_MAP') else t.kind.split('_')[1]
                new_tokens = self.bound_collection_from_tokens(tokens, new_tokens, t, i, f'CONST_{collection_type}')
                continue
            if t.op == self.opc.CALL_FUNCTION_EX and t.attr & 1:
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
            elif t.op == self.opc.BUILD_MAP_UNPACK_WITH_CALL:
                t.kind = 'BUILD_MAP_UNPACK_WITH_CALL_%d' % t.attr
            elif not self.is_pypy and t.op == self.opc.BUILD_TUPLE_UNPACK_WITH_CALL:
                t.kind = 'BUILD_TUPLE_UNPACK_WITH_CALL_%d' % t.attr
            new_tokens.append(t)
        return (new_tokens, customize)
if __name__ == '__main__':
    from xdis.version_info import PYTHON_VERSION_TRIPLE, version_tuple_to_str
    if PYTHON_VERSION_TRIPLE[:2] == (3, 7):
        import inspect
        co = inspect.currentframe().f_code
        (tokens, customize) = Scanner37().ingest(co)
        for t in tokens:
            print(t.format())
        pass
    else:
        print(f'Need to be Python 3.7 to demo; I am version {version_tuple_to_str()}.')