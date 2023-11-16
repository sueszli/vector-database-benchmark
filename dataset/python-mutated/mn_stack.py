import sys
from asm_test import Asm_Test_32

class Test_PUSHPOP(Asm_Test_32):
    TXT = '\n    main:\n       MOV     EBP, ESP\n       PUSH    0x11223344\n       POP     EAX\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       PUSHW   0x1122\n       POPW    AX\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       PUSH    SS\n       POP     EAX\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       PUSHW   SS\n       POPW    AX\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       PUSHFD\n       POP     EAX\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       PUSHFW\n       POPW    AX\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       PUSH    EAX\n       POPFD\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       PUSHW   AX\n       POPFW\n       CMP     EBP, ESP\n       JNZ     BAD\n\n       RET\n\nBAD:\n       INT     0x3\n       RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.ESP - 4 == self.myjit.cpu.EBP
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PUSHPOP]]