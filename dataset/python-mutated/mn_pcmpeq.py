import sys
from asm_test import Asm_Test_32

class Test_PCMPEQB(Asm_Test_32):
    TXT = '\n    main:\n       CALL    next\n       .byte 0x88, 0x78, 0x66, 0x56, 0x44, 0x3F, 0xFF, 0x11\n       .byte 0x89, 0x77, 0x66, 0x55, 0xF9, 0x33, 0x22, 0x11\n    next:\n       POP     EBP\n       MOVQ    MM0, QWORD PTR [EBP]\n       MOVQ    MM1, MM0\n       PCMPEQB MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.MM0 == 1296824780445874312
        assert self.myjit.cpu.MM1 == 18374686479688335360

class Test_PCMPEQW(Asm_Test_32):
    TXT = '\n    main:\n       CALL    next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x3F, 0x22, 0x11\n       .byte 0x89, 0x77, 0x66, 0x55, 0xF9, 0x33, 0x22, 0x11\n    next:\n       POP     EBP\n       MOVQ    MM0, QWORD PTR [EBP]\n       MOVQ    MM1, MM0\n       PCMPEQW MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.MM0 == 1234618810576041864
        assert self.myjit.cpu.MM1 == 18446462603027742720

class Test_PCMPEQD(Asm_Test_32):
    TXT = '\n    main:\n       CALL    next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x3F, 0x22, 0x11\n       .byte 0x88, 0x77, 0x66, 0x55, 0xF9, 0x33, 0x22, 0x11\n    next:\n       POP     EBP\n       MOVQ    MM0, QWORD PTR [EBP]\n       MOVQ    MM1, MM0\n       PCMPEQD MM1, QWORD PTR [EBP+0x8]\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.MM0 == 1234618810576041864
        assert self.myjit.cpu.MM1 == 4294967295

class Test_PCMPEQQ(Asm_Test_32):
    TXT = '\n    main:\n       MOVD       XMM0, ESI\n       MOVD       XMM1, EDI\n       PCMPEQQ    XMM0, XMM1\n       JZ         ret\n       MOV        EAX, 1\n    ret:\n       RET\n    '

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        val = 1
        self.myjit.cpu.ESI = 287454020
        self.myjit.cpu.EDI = 287454021
        self.myjit.cpu.XMM0 = val

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.XMM0 == 340282366920938463444927863358058659840
        assert self.myjit.cpu.XMM1 == 287454021
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PCMPEQB, Test_PCMPEQW, Test_PCMPEQD, Test_PCMPEQQ]]