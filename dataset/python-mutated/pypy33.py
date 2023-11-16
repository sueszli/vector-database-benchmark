"""
Python PyPy 3.3 decompiler scanner.

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.
"""
import uncompyle6.scanners.scanner33 as scan
from xdis.opcodes import opcode_33pypy as opc
JUMP_OPs = map(lambda op: opc.opname[op], opc.hasjrel + opc.hasjabs)

class ScannerPyPy33(scan.Scanner33):

    def __init__(self, show_asm):
        if False:
            return 10
        scan.Scanner33.__init__(self, show_asm, is_pypy=True)
        self.version = (3, 3)
        self.opc = opc
        return