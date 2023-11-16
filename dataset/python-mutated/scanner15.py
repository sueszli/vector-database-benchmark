"""
Python 1.5 bytecode decompiler massaging.

This massages tokenized 1.5 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner21 as scan
from xdis.opcodes import opcode_15
JUMP_OPS = opcode_15.JUMP_OPS

class Scanner15(scan.Scanner21):

    def __init__(self, show_asm=False):
        if False:
            i = 10
            return i + 15
        scan.Scanner21.__init__(self, show_asm)
        self.opc = opcode_15
        self.opname = opcode_15.opname
        self.version = (1, 5)
        self.genexpr_name = '<generator expression>'
        return

    def ingest(self, co, classname=None, code_objects={}, show_asm=None):
        if False:
            i = 10
            return i + 15
        '\n        Create "tokens" the bytecode of an Python code object. Largely these\n        are the opcode name, but in some cases that has been modified to make parsing\n        easier.\n        returning a list of uncompyle6 Token\'s.\n\n        Some transformations are made to assist the deparsing grammar:\n           -  various types of LOAD_CONST\'s are categorized in terms of what they load\n           -  COME_FROM instructions are added to assist parsing control structures\n           -  operands with stack argument counts or flag masks are appended to the\n              opcode name, e.g.:\n              *  BUILD_LIST, BUILD_SET\n              *  MAKE_FUNCTION and FUNCTION_CALLS append the number of positional\n                 arguments\n           -  EXTENDED_ARGS instructions are removed\n\n        Also, when we encounter certain tokens, we add them to a set which will cause\n        custom grammar rules. Specifically, variable arg tokens like MAKE_FUNCTION or\n        BUILD_LIST cause specific rules for the specific number of arguments they take.\n        '
        (tokens, customize) = scan.Scanner21.ingest(self, co, classname, code_objects, show_asm)
        for t in tokens:
            if t.op == self.opc.UNPACK_LIST:
                t.kind = 'UNPACK_LIST_%d' % t.attr
            pass
        return (tokens, customize)