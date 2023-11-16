import sys
from asm_test import Asm_Test_32

class Test_FADD(Asm_Test_32):
    TXT = '\n    main:\n       ; test float\n       PUSH 0\n       FLD1\n       FLD1\n       FADD ST, ST(1)\n       FIST  DWORD PTR [ESP]\n       POP  EAX\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.EAX == 2
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_FADD]]