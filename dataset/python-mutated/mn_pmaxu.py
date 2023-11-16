import sys
from asm_test import Asm_Test_32

class Test_PMAXU(Asm_Test_32):
    TXT = '\n    main:\n       CALL   next\n       .byte 0x88, 0x76, 0x66, 0x54, 0x44, 0x32, 0x00, 0x10\n       .byte 0x87, 0x77, 0x66, 0x55, 0x40, 0x33, 0x22, 0x11\n    next:\n       POP    EBP\n       MOVQ   MM0, QWORD PTR [EBP]\n       MOVQ   MM1, MM0\n       PMAXUB MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.MM0 == 1152976773662013064
        assert self.myjit.cpu.MM1 == 1234605616436508552
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PMAXU]]