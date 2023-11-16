import sys
from asm_test import Asm_Test
from pdb import pm

class Test_UBFM1(Asm_Test):
    TXT = '\nmain:\n       MOVZ    X0, 0x5600\n       UBFM    X0, X0, 8, 15\n       RET     LR\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.X0 == 86
        pass

class Test_UBFM2(Asm_Test):
    TXT = '\nmain:\n       MOVZ    X0, 0x56\n       UBFM    X0, X0, 4, 55\n       RET     LR\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.X0 == 5
        pass
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_UBFM1, Test_UBFM2]]