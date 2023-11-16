import sys
from asm_test import Asm_Test_32

class Test_PMINU(Asm_Test_32):
    TXT = '\n    main:\n       CALL   next\n       .byte 0x88, 0x78, 0x66, 0x56, 0x44, 0x3F, 0xFF, 0x1F\n       .byte 0x89, 0x77, 0x66, 0x55, 0xF9, 0x33, 0x22, 0x11\n    next:\n       POP    EBP\n       MOVQ   MM0, QWORD PTR [EBP]\n       MOVQ   MM1, MM0\n       PMINUB MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.MM0 == 2305631096976865416
        assert self.myjit.cpu.MM1 == 1234605616436508552
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PMINU]]