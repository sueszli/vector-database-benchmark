"""Python 3.9 bytecode decompiler scanner.

Does some token massaging of xdis-disassembled instructions to make
things easier for decompilation.

This sets up opcodes Python's 3.9 and calls a generalized
scanner routine for Python 3.7 and up.
"""
from uncompyle6.scanners.scanner38 import Scanner38
from uncompyle6.scanners.scanner37base import Scanner37Base
from xdis.opcodes import opcode_38 as opc
JUMP_OPs = opc.JUMP_OPS

class Scanner39(Scanner38):

    def __init__(self, show_asm=None):
        if False:
            return 10
        Scanner37Base.__init__(self, (3, 9), show_asm)
        return
    pass
if __name__ == '__main__':
    print('Note: Python 3.9 decompilation not supported')