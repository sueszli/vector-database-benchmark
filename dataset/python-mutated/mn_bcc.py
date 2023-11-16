import sys
from asm_test import Asm_Test

class Test_BCC(Asm_Test):
    MYSTRING = 'test string'
    TXT = '\n    main:\n      ADDIU   A0, V0, mystr\nstrlen:\n      LBU     V0, 0(A0)\n      BEQ     V0, ZERO, SKIP\n      ADDU    V1, ZERO, ZERO\nloop:\n      ADDIU   A0, A0, 1\n      LBU     V0, 0(A0)\n      BNE     V0, ZERO, loop\n      ADDIU   V1, V1, 1\nSKIP:\n      JR      RA\n      ADDU    V0, V1, ZERO\n\n    mystr:\n    .string "%s"\n    ' % MYSTRING

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.V0 == len(self.MYSTRING)
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_BCC]]