"""
Python 2.4 bytecode massaging.

This massages tokenized 2.7 bytecode to make it more amenable for
grammar parsing.
"""
import uncompyle6.scanners.scanner25 as scan
from xdis.opcodes import opcode_24
JUMP_OPS = opcode_24.JUMP_OPS

class Scanner24(scan.Scanner25):

    def __init__(self, show_asm=False):
        if False:
            i = 10
            return i + 15
        scan.Scanner25.__init__(self, show_asm)
        self.opc = opcode_24
        self.opname = opcode_24.opname
        self.version = (2, 4)
        self.genexpr_name = '<generator expression>'
        return
if __name__ == '__main__':
    from xdis.version_info import PYTHON_VERSION_TRIPLE, version_tuple_to_str
    if PYTHON_VERSION_TRIPLE[:2] == (2, 4):
        import inspect
        co = inspect.currentframe().f_code
        (tokens, customize) = Scanner24().ingest(co)
        for t in tokens:
            print(t.format())
        pass
    else:
        print('Need to be Python 2.4 to demo; I am version %s' % version_tuple_to_str())