import sys
from asm_test import Asm_Test_64

class Test_DIV(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        MOV RBX, 0x11223344556677\n        DIV RBX\n        RET\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.RAX == 2039
        assert self.myjit.cpu.RDX == 1088
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_DIV]]