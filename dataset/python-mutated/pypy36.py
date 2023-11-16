"""
Python PyPy 3.6 decompiler scanner.

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.
"""
import uncompyle6.scanners.scanner36 as scan
from xdis.opcodes import opcode_35 as opc
JUMP_OPs = opc.JUMP_OPS

class ScannerPyPy36(scan.Scanner36):

    def __init__(self, show_asm):
        if False:
            for i in range(10):
                print('nop')
        scan.Scanner36.__init__(self, show_asm, is_pypy=True)
        self.version = (3, 6)
        return