import sys
from asm_test import Asm_Test_32

class Test_PSHUFB(Asm_Test_32):
    TXT = '\n    main:\n       CALL   next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0\n    next:\n       POP    EBP\n       MOVQ   MM0, QWORD PTR [EBP]\n       MOVQ   MM1, MM0\n       PSHUFB MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 9833440827789222417
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PSHUFB]]