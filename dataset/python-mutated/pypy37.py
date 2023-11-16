"""
Python PyPy 3.7 decompiler scanner.

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.
"""
import uncompyle6.scanners.scanner37 as scan
from xdis.opcodes import opcode_37pypy as opc
JUMP_OPs = opc.JUMP_OPS

class ScannerPyPy37(scan.Scanner37):

    def __init__(self, show_asm):
        if False:
            i = 10
            return i + 15
        scan.Scanner37.__init__(self, show_asm, is_pypy=True)
        self.version = (3, 7)
        self.opc = opc
        self.is_pypy = True
        return