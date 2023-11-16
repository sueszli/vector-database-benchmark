import sys
from asm_test import Asm_Test_32

class Test_DAA(Asm_Test_32):
    TXT = '\n    main:\n       MOV     EBP, ESP\n       LEA     ESI, DWORD PTR [array_al]\n    loop:\n\n       ; load original cf\n       LODSB\n       MOV     BL, AL\n       ; load original af\n       LODSB\n       SHL     AL, 4\n       OR      AL, BL\n       MOV     AH, AL\n       SAHF\n       ; load original al\n       LODSB\n\n       DAA\n       MOV     BL, AL\n\n       LAHF\n       MOV     CL, AH\n\n       ; test cf\n       LODSB\n       MOV     DL, CL\n       AND     DL, 1\n       CMP     DL, AL\n       JNZ BAD\n\n       MOV     DL, CL\n       SHR     DL, 4\n       AND     DL, 1\n       ; test af\n       LODSB\n       CMP     DL, AL\n       JNZ BAD\n\n       ; test value\n       LODSB\n       CMP     AL, BL\n       JNZ BAD\n\n       CMP     ESI, array_al_end\n       JB      loop\n\n\n    end:\n       RET\n\nBAD:\n       INT     0x3\n       RET\n\narray_al:\n.byte 0, 1, 0x08, 0, 1, 0x0E\n.byte 0, 1, 0x09, 0, 1, 0x0F\n.byte 0, 1, 0x0A, 0, 1, 0x10\n.byte 0, 1, 0x98, 0, 1, 0x9E\n.byte 0, 1, 0x99, 0, 1, 0x9F\n.byte 0, 1, 0x9A, 1, 1, 0x00\narray_al_end:\n.long 0\n    '

    def check(self):
        if False:
            return 10
        pass
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_DAA]]