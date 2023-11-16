import sys
from asm_test import Asm_Test_32

class Test_get_set_128(Asm_Test_32):
    TXT = '\n    main:\n       MOVD       XMM0, ESI\n       MOVD       XMM1, EDI\n       PCMPEQQ    XMM0, XMM1\n       JZ         ret\n       MOV        EAX, 1\n\n       PUSH       0x11112222\n       PUSH       0x33334444\n       PUSH       0x55556666\n       PUSH       0x77778888\n       MOVAPS     XMM2, XMMWORD PTR [ESP]\n       ADD        ESP, 0x10\n    ret:\n       RET\n    '

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        val = 1
        self.myjit.cpu.ESI = 287454020
        self.myjit.cpu.EDI = 287454021
        self.myjit.cpu.XMM0 = val
        assert self.myjit.cpu.XMM0 == val
        assert self.myjit.cpu.get_gpreg()['XMM0'] == val

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.XMM0 == 340282366920938463444927863358058659840
        assert self.myjit.cpu.XMM1 == 287454021
        assert self.myjit.cpu.get_gpreg()['XMM0'] == 340282366920938463444927863358058659840
        assert self.myjit.cpu.get_gpreg()['XMM1'] == 287454021
        assert self.myjit.cpu.get_gpreg()['XMM2'] == 22685837286468424649968941046919825544
        assert self.myjit.cpu.get_gpreg()['XMM2'] == 22685837286468424649968941046919825544
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_get_set_128]]