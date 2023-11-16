import sys
from asm_test import Asm_Test_32

class Test_PINSRB(Asm_Test_32):
    TXT = '\n    main:\n       CALL      next\n       .byte 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11\n       .byte 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01\n    next:\n       POP       EBP\n       MOVQ      MM0, QWORD PTR [EBP]\n       MOVQ      MM1, MM0\n       PINSRW    MM1, QWORD PTR [EBP+0x8], 2\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.MM0 == 1234605616436508552
        assert self.myjit.cpu.MM1 == 1234556980226848648
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PINSRB]]