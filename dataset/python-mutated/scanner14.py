"""
Python 1.4 bytecode decompiler massaging.

This massages tokenized 1.4 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner15 as scan
from xdis.opcodes import opcode_14
JUMP_OPS = opcode_14.JUMP_OPS

class Scanner14(scan.Scanner15):

    def __init__(self, show_asm=False):
        if False:
            i = 10
            return i + 15
        scan.Scanner15.__init__(self, show_asm)
        self.opc = opcode_14
        self.opname = opcode_14.opname
        self.version = (1, 4)
        self.genexpr_name = '<generator expression>'
        return