import sys
from asm_test import Asm_Test_32

class Test_PSRL(Asm_Test_32):
    TXT = '\n    main:\n       CALL   next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x4, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0\n    next:\n       POP    EBP\n       MOVQ   MM0, QWORD PTR [EBP]\n       MOVQ   MM1, MM0\n       MOVQ   MM2, MM0\n       MOVQ   MM3, MM0\n       PSRLW  MM1, QWORD PTR [EBP+0x8]\n       PSRLD  MM2, QWORD PTR [EBP+0x8]\n       PSRLQ  MM3, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 77127665581426552
        assert self.myjit.cpu.MM2 == 77162849953539960
        assert self.myjit.cpu.MM3 == 77162851027281784

class Test_PSLL(Asm_Test_32):
    TXT = '\n    main:\n       CALL   next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x4, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0\n    next:\n       POP    EBP\n       MOVQ   MM0, QWORD PTR [EBP]\n       MOVQ   MM1, MM0\n       MOVQ   MM2, MM0\n       MOVQ   MM3, MM0\n       PSLLW  MM1, QWORD PTR [EBP+0x8]\n       PSLLD  MM2, QWORD PTR [EBP+0x8]\n       PSLLQ  MM3, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 1306101342869158016
        assert self.myjit.cpu.MM2 == 1306945767799748736
        assert self.myjit.cpu.MM3 == 1306945789274585216
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PSRL, Test_PSLL]]