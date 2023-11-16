"""
Python 1.0 bytecode decompiler massaging.

This massages tokenized 1.0 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner11 as scan
from xdis.opcodes import opcode_10
JUMP_OPS = opcode_10.JUMP_OPS

class Scanner10(scan.Scanner11):

    def __init__(self, show_asm=False):
        if False:
            print('Hello World!')
        scan.Scanner11.__init__(self, show_asm)
        self.opc = opcode_10
        self.opname = opcode_10.opname
        self.version = (1, 0)
        return