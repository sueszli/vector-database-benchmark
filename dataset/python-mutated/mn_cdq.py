import sys
from asm_test import Asm_Test_16, Asm_Test_32, Asm_Test_64
from miasm.core.utils import pck16, pck32

class Test_CBW_16(Asm_Test_16):
    MYSTRING = 'test CBW 16'

    def prepare(self):
        if False:
            return 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            print('Hello World!')
        self.myjit.cpu.EAX = 2271560481
        self.myjit.cpu.EDX = 287454020
    TXT = '\n    main:\n       CBW\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.EAX == 2271543329
        assert self.myjit.cpu.EDX == 287454020

class Test_CBW_16_signed(Asm_Test_16):
    MYSTRING = 'test CBW 16 signed'

    def prepare(self):
        if False:
            return 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.cpu.EAX = 2271560577
        self.myjit.cpu.EDX = 287454020
    TXT = '\n    main:\n       CBW\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.EAX == 2271608705
        assert self.myjit.cpu.EDX == 287454020

class Test_CBW_32(Asm_Test_32):
    MYSTRING = 'test CBW 32'

    def prepare(self):
        if False:
            return 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.cpu.EAX = 2271560481
        self.myjit.cpu.EDX = 287454020
    TXT = '\n    main:\n       CBW\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.EAX == 2271543329
        assert self.myjit.cpu.EDX == 287454020

class Test_CBW_32_signed(Asm_Test_32):
    MYSTRING = 'test CBW 32 signed'

    def prepare(self):
        if False:
            i = 10
            return i + 15
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            return 10
        self.myjit.cpu.EAX = 2271560577
        self.myjit.cpu.EDX = 287454020
    TXT = '\n    main:\n       CBW\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.EAX == 2271608705
        assert self.myjit.cpu.EDX == 287454020

class Test_CDQ_32(Asm_Test_32):
    MYSTRING = 'test cdq 32'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            return 10
        self.myjit.cpu.EAX = 2003125025
        self.myjit.cpu.EDX = 287454020
    TXT = '\n    main:\n       CDQ\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.EAX == 2003125025
        assert self.myjit.cpu.EDX == 0

class Test_CDQ_32_signed(Asm_Test_32):
    MYSTRING = 'test cdq 32 signed'

    def prepare(self):
        if False:
            i = 10
            return i + 15
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            while True:
                i = 10
        self.myjit.cpu.EAX = 2271560481
        self.myjit.cpu.EDX = 287454020
    TXT = '\n    main:\n       CDQ\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.EAX == 2271560481
        assert self.myjit.cpu.EDX == 4294967295

class Test_CDQ_64(Asm_Test_64):
    MYSTRING = 'test cdq 64'

    def prepare(self):
        if False:
            return 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.cpu.RAX = 1311768466870846241
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CDQ\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 1311768466870846241
        assert self.myjit.cpu.RDX == 0

class Test_CDQ_64_signed(Asm_Test_64):
    MYSTRING = 'test cdq 64 signed'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        self.myjit.cpu.RAX = 1311768467139281697
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CDQ\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.RAX == 1311768467139281697
        assert self.myjit.cpu.RDX == 4294967295

class Test_CDQE_64(Asm_Test_64):
    MYSTRING = 'test cdq 64'

    def prepare(self):
        if False:
            i = 10
            return i + 15
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            print('Hello World!')
        self.myjit.cpu.RAX = 1311768466870846241
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CDQE\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 2003125025
        assert self.myjit.cpu.RDX == 1234605616436508552

class Test_CDQE_64_signed(Asm_Test_64):
    MYSTRING = 'test cdq 64 signed'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            while True:
                i = 10
        self.myjit.cpu.RAX = 1311768467139281697
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CDQE\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.RAX == 18446744071686144801
        assert self.myjit.cpu.RDX == 1234605616436508552

class Test_CWD_32(Asm_Test_32):
    MYSTRING = 'test cdq 32'

    def prepare(self):
        if False:
            print('Hello World!')
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        self.myjit.cpu.EAX = 2271560481
        self.myjit.cpu.EDX = 305419896
    TXT = '\n    main:\n       CWD\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.RAX == 2271560481
        assert self.myjit.cpu.RDX == 305397760

class Test_CWD_32_signed(Asm_Test_32):
    MYSTRING = 'test cdq 32'

    def prepare(self):
        if False:
            print('Hello World!')
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        self.myjit.cpu.EAX = 2271576865
        self.myjit.cpu.EDX = 305419896
    TXT = '\n    main:\n       CWD\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 2271576865
        assert self.myjit.cpu.RDX == 305463295

class Test_CWD_32(Asm_Test_32):
    MYSTRING = 'test cdq 32'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.cpu.EAX = 2271560481
        self.myjit.cpu.EDX = 305419896
    TXT = '\n    main:\n       CWD\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 2271560481
        assert self.myjit.cpu.RDX == 305397760

class Test_CWDE_32(Asm_Test_32):
    MYSTRING = 'test cwde 32'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        self.myjit.cpu.EAX = 2271560481
        self.myjit.cpu.EDX = 287454020
    TXT = '\n    main:\n       CWDE\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.RAX == 17185
        assert self.myjit.cpu.RDX == 287454020

class Test_CWDE_32_signed(Asm_Test_32):
    MYSTRING = 'test cwde 32 signed'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            print('Hello World!')
        self.myjit.cpu.RAX = 2271576865
        self.myjit.cpu.RDX = 287454020
    TXT = '\n    main:\n       CWDE\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.EAX == 4294935329
        assert self.myjit.cpu.RDX == 287454020

class Test_CWDE_64(Asm_Test_64):
    MYSTRING = 'test cwde 64'

    def prepare(self):
        if False:
            return 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            while True:
                i = 10
        self.myjit.cpu.RAX = 1311768467139281697
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CWDE\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.RAX == 17185
        assert self.myjit.cpu.RDX == 1234605616436508552

class Test_CWDE_64_signed(Asm_Test_64):
    MYSTRING = 'test cwde 64 signed'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            return 10
        self.myjit.cpu.RAX = 1311768467139298081
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CWDE\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.RAX == 4294935329
        assert self.myjit.cpu.RDX == 1234605616436508552

class Test_CQO_64(Asm_Test_64):
    MYSTRING = 'test cwde 64'

    def prepare(self):
        if False:
            i = 10
            return i + 15
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.myjit.cpu.RAX = 1311768467139281697
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CQO\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.RAX == 1311768467139281697
        assert self.myjit.cpu.RDX == 0

class Test_CQO_64_signed(Asm_Test_64):
    MYSTRING = 'test cwde 64 signed'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.myjit.lifter.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            while True:
                i = 10
        self.myjit.cpu.RAX = 9382218999387226913
        self.myjit.cpu.RDX = 1234605616436508552
    TXT = '\n    main:\n       CQO\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            i = 10
            return i + 15
        assert self.myjit.cpu.RAX == 9382218999387226913
        assert self.myjit.cpu.RDX == 18446744073709551615
if __name__ == '__main__':
    tests = [Test_CBW_16, Test_CBW_16_signed, Test_CBW_32, Test_CBW_32_signed, Test_CWD_32, Test_CWD_32_signed, Test_CWDE_32, Test_CWDE_32_signed, Test_CWDE_64, Test_CWDE_64_signed, Test_CDQ_32, Test_CDQ_32_signed, Test_CDQ_64, Test_CDQ_64_signed, Test_CDQE_64, Test_CDQE_64_signed]
    if sys.argv[1] not in ['gcc']:
        tests += [Test_CQO_64, Test_CQO_64_signed]
    [test(*sys.argv[1:])() for test in tests]