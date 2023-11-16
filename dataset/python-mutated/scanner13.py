"""
Python 1.3 bytecode decompiler massaging.

This massages tokenized 1.3 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner14 as scan
from xdis.opcodes import opcode_13
JUMP_OPS = opcode_13.JUMP_OPS

class Scanner13(scan.Scanner14):

    def __init__(self, show_asm=False):
        if False:
            return 10
        scan.Scanner14.__init__(self, show_asm)
        self.opc = opcode_13
        self.opname = opcode_13.opname
        self.version = (1, 3)
        return