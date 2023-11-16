"""
Python 3.8 bytecode decompiler scanner.

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.

This sets up opcodes Python's 3.8 and calls a generalized
scanner routine for Python 3.7 and up.
"""
from typing import Dict, Tuple
from uncompyle6.scanners.tok import off2int
from uncompyle6.scanners.scanner37 import Scanner37
from uncompyle6.scanners.scanner37base import Scanner37Base
from xdis.opcodes import opcode_38 as opc
JUMP_OPs = opc.JUMP_OPS

class Scanner38(Scanner37):

    def __init__(self, show_asm=None, debug='', is_pypy=False):
        if False:
            i = 10
            return i + 15
        Scanner37Base.__init__(self, (3, 8), show_asm, debug, is_pypy)
        self.debug = debug
        return
    pass

    def ingest(self, bytecode, classname=None, code_objects={}, show_asm=None) -> Tuple[list, dict]:
        if False:
            print('Hello World!')
        '\n        Create "tokens" the bytecode of an Python code object. Largely these\n        are the opcode name, but in some cases that has been modified to make parsing\n        easier.\n        returning a list of uncompyle6 Token\'s.\n\n        Some transformations are made to assist the deparsing grammar:\n           -  various types of LOAD_CONST\'s are categorized in terms of what they load\n           -  COME_FROM instructions are added to assist parsing control structures\n           -  operands with stack argument counts or flag masks are appended to the opcode name, e.g.:\n              *  BUILD_LIST, BUILD_SET\n              *  MAKE_FUNCTION and FUNCTION_CALLS append the number of positional arguments\n           -  EXTENDED_ARGS instructions are removed\n\n        Also, when we encounter certain tokens, we add them to a set which will cause custom\n        grammar rules. Specifically, variable arg tokens like MAKE_FUNCTION or BUILD_LIST\n        cause specific rules for the specific number of arguments they take.\n        '
        (tokens, customize) = super(Scanner38, self).ingest(bytecode, classname, code_objects, show_asm)
        jump_back_targets: Dict[int, int] = {}
        for token in tokens:
            if token.kind == 'JUMP_BACK':
                jump_back_targets[token.attr] = token.offset
                pass
            pass
        if self.debug and jump_back_targets:
            print(jump_back_targets)
        loop_ends = []
        next_end = tokens[len(tokens) - 1].off2int() + 10
        new_tokens = []
        for (i, token) in enumerate(tokens):
            opname = token.kind
            offset = token.offset
            if offset == next_end:
                loop_ends.pop()
                if self.debug:
                    print(f"{'  ' * len(loop_ends)}remove loop offset {offset}")
                    pass
                next_end = loop_ends[-1] if len(loop_ends) else tokens[len(tokens) - 1].off2int() + 10
            if offset in jump_back_targets:
                next_end = off2int(jump_back_targets[offset], prefer_last=False)
                if self.debug:
                    print(f"{'  ' * len(loop_ends)}adding loop offset {offset} ending at {next_end}")
                loop_ends.append(next_end)
            if opname in ('JUMP_FORWARD', 'JUMP_ABSOLUTE') and len(loop_ends):
                jump_target = token.attr
                if opname == 'JUMP_ABSOLUTE' and jump_target <= next_end:
                    new_tokens.append(token)
                    continue
                if i + 1 < len(tokens) and tokens[i + 1] == 'JUMP_BACK':
                    jump_back_index = i + 1
                else:
                    jump_back_index = self.offset2tok_index[jump_target] - 1
                    while tokens[jump_back_index].kind.startswith('COME_FROM_'):
                        jump_back_index -= 1
                        pass
                    pass
                jump_back_token = tokens[jump_back_index]
                break_loop = token.linestart and jump_back_token != 'JUMP_BACK'
                if break_loop or (jump_back_token == 'JUMP_BACK' and jump_back_token.attr < token.off2int()):
                    token.kind = 'BREAK_LOOP'
                pass
            new_tokens.append(token)
        return (new_tokens, customize)
if __name__ == '__main__':
    from xdis.version_info import PYTHON_VERSION_TRIPLE, version_tuple_to_str
    if PYTHON_VERSION_TRIPLE[:2] == (3, 8):
        import inspect
        co = inspect.currentframe().f_code
        (tokens, customize) = Scanner38().ingest(co)
        for t in tokens:
            print(t.format())
        pass
    else:
        print(f'Need to be Python 3.8 to demo; I am version {version_tuple_to_str()}.')