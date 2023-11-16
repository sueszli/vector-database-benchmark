import sys
from asm_test import Asm_Test_64

class Test_CMOVZ_OK(Asm_Test_64):
    TXT = '\nmain:\n        MOV   RAX, 0x8877665544332211\n        MOV   RBX, RAX\n        MOV   RAX, 0xAABBCCDDEEFF0011\n        XOR   RCX, RCX\n        CMOVZ RAX, RBX\n        RET\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.RAX == 9833440827789222417

class Test_CMOVZ_KO(Asm_Test_64):
    TXT = '\nmain:\n        MOV   RAX, 0x8877665544332211\n        MOV   RBX, RAX\n        MOV   RAX, 0xAABBCCDDEEFF0011\n        XOR   RCX, RCX\n        INC   RCX\n        CMOVZ RAX, RBX\n        RET\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.RAX == 12302652060662169617

class Test_CMOVZ_OK_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV   RAX, 0x8877665544332211\n        MOV   RBX, RAX\n        MOV   RAX, 0xAABBCCDDEEFF0011\n        XOR   RCX, RCX\n        CMOVZ EAX, EBX\n        RET\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.RAX == 1144201745

class Test_CMOVZ_KO_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV   RAX, 0x8877665544332211\n        MOV   RBX, RAX\n        MOV   RAX, 0xAABBCCDDEEFF0011\n        XOR   RCX, RCX\n        INC   RCX\n        CMOVZ EAX, EBX\n        RET\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.RAX == 4009689105
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_CMOVZ_OK, Test_CMOVZ_KO, Test_CMOVZ_OK_64_32, Test_CMOVZ_KO_64_32]]