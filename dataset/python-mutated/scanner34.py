"""
Python 3.4 bytecode decompiler scanner

Does some additional massaging of xdis-disassembled instructions to
make things easier for decompilation.

This sets up opcodes Python's 3.4 and calls a generalized
scanner routine for Python 3.
"""
from __future__ import print_function
from xdis.opcodes import opcode_34 as opc
JUMP_OPS = opc.JUMP_OPS
from uncompyle6.scanners.scanner3 import Scanner3

class Scanner34(Scanner3):

    def __init__(self, show_asm=None):
        if False:
            i = 10
            return i + 15
        Scanner3.__init__(self, (3, 4), show_asm)
        return
    pass
if __name__ == '__main__':
    from xdis.version_info import PYTHON_VERSION_TRIPLE, version_tuple_to_str
    if PYTHON_VERSION_TRIPLE[:2] == (3, 4):
        import inspect
        co = inspect.currentframe().f_code
        (tokens, customize) = Scanner34().ingest(co)
        for t in tokens:
            print(t.format())
        pass
    else:
        print('Need to be Python 3.4 to demo; I am version %s' % version_tuple_to_str())