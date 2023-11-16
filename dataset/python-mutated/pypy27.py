"""
Python PyPy 2.7 bytecode scanner/deparser

This overlaps Python's 2.7's dis module, but it can be run from
Python 3 and other versions of Python. Also, we save token
information for later use in deparsing.
"""
import uncompyle6.scanners.scanner27 as scan
from xdis.opcodes import opcode_27pypy
JUMP_OPS = opcode_27pypy.JUMP_OPS

class ScannerPyPy27(scan.Scanner27):

    def __init__(self, show_asm):
        if False:
            return 10
        scan.Scanner27.__init__(self, show_asm, is_pypy=True)
        self.version = (2, 7)
        return