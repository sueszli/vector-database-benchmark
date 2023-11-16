"""
Python 2.3 bytecode massaging.

This massages tokenized 2.3 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner24 as scan
from xdis.opcodes import opcode_23
JUMP_OPS = opcode_23.JUMP_OPS

class Scanner23(scan.Scanner24):

    def __init__(self, show_asm=False):
        if False:
            for i in range(10):
                print('nop')
        scan.Scanner24.__init__(self, show_asm)
        self.opc = opcode_23
        self.opname = opcode_23.opname
        self.version = (2, 3)
        self.genexpr_name = '<generator expression>'
        return