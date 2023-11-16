import sys
from asm_test import Asm_Test_64

class Test_ROR_0(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        ROR RAX, 0\n        RET\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.RAX == 9833440827789222417

class Test_ROR_8(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        ROR RAX, 8\n        RET\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.RAX == 1263390976878326562

class Test_ROR_X8(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        MOV CL, 16\n        ROR RAX, CL\n        RET\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.RAX == 2454893318292980787

class Test_SHR_0(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        SHR RAX, 0\n        RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 9833440827789222417

class Test_SHR_8(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        SHR RAX, 8\n        RET\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.RAX == 38411878233551650

class Test_SHR_X8(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        MOV CL, 16\n        SHR RAX, CL\n        RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 150046399349811

class Test_ROR_0_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        ROR EAX, 0\n        RET\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.RAX == 1144201745

class Test_ROR_8_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        ROR EAX, 8\n        RET\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.RAX == 289682210

class Test_ROR_X8_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        MOV CL, 16\n        ROR EAX, CL\n        RET\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.RAX == 571556915

class Test_SHR_0_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        SHR EAX, 0\n        RET\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.RAX == 1144201745

class Test_SHR_8_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        SHR EAX, 8\n        RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 4469538

class Test_SHR_X8_64_32(Asm_Test_64):
    TXT = '\nmain:\n        MOV RAX, 0x8877665544332211\n        MOV CL, 16\n        SHR EAX, CL\n        RET\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.RAX == 17459

class Test_SHLD(Asm_Test_64):
    TXT = '\nmain:\n        MOV         RAX, 0x1234FDB512345678\n        MOV         RDX, RAX\n        MOV         RAX, 0x21AD96F921AD3D34\n        MOV         RSI, RAX\n        MOV         RAX, 0x0000000000000021\n        MOV         RCX, RAX\n        SHLD        EDX, ESI, CL\n        RET\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RDX == 610839792
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_ROR_0, Test_ROR_8, Test_ROR_X8, Test_SHR_0, Test_SHR_8, Test_SHR_X8, Test_ROR_0_64_32, Test_ROR_8_64_32, Test_ROR_X8_64_32, Test_SHR_0_64_32, Test_SHR_8_64_32, Test_SHR_X8_64_32, Test_SHLD]]