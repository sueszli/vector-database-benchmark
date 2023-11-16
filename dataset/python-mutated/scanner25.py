"""
Python 2.5 bytecode massaging.

This overlaps Python's 2.5's dis module, but it can be run from
Python 3 and other versions of Python. Also, we save token
information for later use in deparsing.
"""
import uncompyle6.scanners.scanner26 as scan
from xdis.opcodes import opcode_25
JUMP_OPS = opcode_25.JUMP_OPS

class Scanner25(scan.Scanner26):

    def __init__(self, show_asm=False):
        if False:
            while True:
                i = 10
        self.opc = opcode_25
        self.opname = opcode_25.opname
        scan.Scanner26.__init__(self, show_asm)
        self.version = (2, 5)
        return
if __name__ == '__main__':
    from xdis.version_info import PYTHON_VERSION_TRIPLE, version_tuple_to_str
    if PYTHON_VERSION_TRIPLE[:2] == (2, 5):
        import inspect
        co = inspect.currentframe().f_code
        (tokens, customize) = Scanner25().ingest(co)
        for t in tokens:
            print(t.format())
        pass
    else:
        print('Need to be Python 2.5 to demo; I am version %s' % version_tuple_to_str())