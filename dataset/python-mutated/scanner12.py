"""
Python 1.2 bytecode decompiler massaging.

This massages tokenized 1.2 bytecode to make it more amenable for
grammar parsing.

"""
import uncompyle6.scanners.scanner13 as scan
from xdis.opcodes import opcode_11
JUMP_OPS = opcode_11.JUMP_OPS

class Scanner12(scan.Scanner13):

    def __init__(self, show_asm=False):
        if False:
            print('Hello World!')
        scan.Scanner14.__init__(self, show_asm)
        self.opc = opcode_11
        self.opname = opcode_11.opname
        self.version = (1, 2)
        return