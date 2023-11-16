"""
Python PyPy 3.5 decompiler scanner.

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.
"""
import uncompyle6.scanners.scanner35 as scan
from xdis.opcodes import opcode_35 as opc
JUMP_OPs = opc.JUMP_OPS

class ScannerPyPy35(scan.Scanner35):

    def __init__(self, show_asm):
        if False:
            i = 10
            return i + 15
        scan.Scanner35.__init__(self, show_asm, is_pypy=True)
        self.version = (3, 5)
        return