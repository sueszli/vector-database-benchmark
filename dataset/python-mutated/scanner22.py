"""
Python 2.2 bytecode massaging.

This massages tokenized 2.2 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner23 as scan
from xdis.opcodes import opcode_22
JUMP_OPS = opcode_22.JUMP_OPS

class Scanner22(scan.Scanner23):

    def __init__(self, show_asm=False):
        if False:
            print('Hello World!')
        scan.Scanner23.__init__(self, show_asm)
        self.opc = opcode_22
        self.opname = opcode_22.opname
        self.version = (2, 2)
        self.genexpr_name = '<generator expression>'
        self.parent_ingest = self.ingest
        self.ingest = self.ingest22
        return

    def ingest22(self, co, classname=None, code_objects={}, show_asm=None):
        if False:
            while True:
                i = 10
        '\n        Create "tokens" the bytecode of an Python code object. Largely these\n        are the opcode name, but in some cases that has been modified to make parsing\n        easier.\n        returning a list of uncompyle6 Token\'s.\n\n        Some transformations are made to assist the deparsing grammar:\n           -  various types of LOAD_CONST\'s are categorized in terms of what they load\n           -  COME_FROM instructions are added to assist parsing control structures\n           -  operands with stack argument counts or flag masks are appended to the opcode name, e.g.:\n              *  BUILD_LIST, BUILD_SET\n              *  MAKE_FUNCTION and FUNCTION_CALLS append the number of positional arguments\n           -  EXTENDED_ARGS instructions are removed\n\n        Also, when we encounter certain tokens, we add them to a set which will cause custom\n        grammar rules. Specifically, variable arg tokens like MAKE_FUNCTION or BUILD_LIST\n        cause specific rules for the specific number of arguments they take.\n        '
        (tokens, customize) = self.parent_ingest(co, classname, code_objects, show_asm)
        tokens = [t for t in tokens if t.kind != 'SET_LINENO']
        return (tokens, customize)