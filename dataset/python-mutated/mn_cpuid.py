import sys
from asm_test import Asm_Test_32

class Test_CPUID(Asm_Test_32):
    """Check for cpuid support (and not for arbitrary returned values)"""
    TXT = '\n    main:\n       XOR EAX, EAX\n       CPUID\n       RET\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.EAX == 10
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_CPUID]]