"""
Python 2.1 bytecode massaging.

This massages tokenized 2.1 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner22 as scan
from xdis.opcodes import opcode_21
JUMP_OPS = opcode_21.JUMP_OPS

class Scanner21(scan.Scanner22):

    def __init__(self, show_asm=False):
        if False:
            while True:
                i = 10
        scan.Scanner22.__init__(self, show_asm)
        self.opc = opcode_21
        self.opname = opcode_21.opname
        self.version = (2, 1)
        self.genexpr_name = '<generator expression>'
        return