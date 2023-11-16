import sys
from asm_test import Asm_Test_32

class Test_PUNPCKHBW(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x01, 0x02, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA\n    next:\n       POP       EBP\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PUNPCKHBW MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 12254781819181063492

class Test_PUNPCKHWD(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x01, 0x02, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA\n    next:\n       POP       EBP\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PUNPCKHWD MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 12302445648256250692

class Test_PUNPCKHDQ(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x01, 0x02, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA\n    next:\n       POP       EBP\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PUNPCKHDQ MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 12302652056939934532

class Test_PUNPCKLBW(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x01, 0x02, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA\n    next:\n       POP       EBP\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PUNPCKLBW MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 17173913567640355208

class Test_PUNPCKLWD(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x01, 0x02, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA\n    next:\n       POP       EBP\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PUNPCKLWD MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 17221577396696741768

class Test_PUNPCKLDQ(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x01, 0x02, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA\n    next:\n       POP       EBP\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PUNPCKLDQ MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 17221485704839067528
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PUNPCKHBW, Test_PUNPCKHWD, Test_PUNPCKHDQ, Test_PUNPCKLBW, Test_PUNPCKLWD, Test_PUNPCKLDQ]]