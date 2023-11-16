import sys
from asm_test import Asm_Test_32

class Test_PMOVMSKB(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0xE6, 0x55, 0xC4, 0x33, 0x22, 0x11\n       .byte 0x01, 0x02, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA\n    next:\n       POP       EBP\n       MOV       EAX, 0xFFFFFFFF\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PMOVMSKB  EAX, MM1\n       RET\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.MM0 == 1234606166200711048
        assert self.myjit.cpu.MM1 == 1234606166200711048
        assert self.myjit.cpu.EAX == 21
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PMOVMSKB]]