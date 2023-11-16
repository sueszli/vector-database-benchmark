import sys
from asm_test import Asm_Test_16, Asm_Test_32
from miasm.core.utils import pck16, pck32

def init_regs(test):
    if False:
        for i in range(10):
            print('nop')
    test.myjit.cpu.EAX = 286331153
    test.myjit.cpu.EBX = 572662306
    test.myjit.cpu.ECX = 858993459
    test.myjit.cpu.EDX = 1145324612
    test.myjit.cpu.ESI = 1431655765
    test.myjit.cpu.EDI = 1717986918
    test.myjit.cpu.EBP = 2004318071
    test.stk_origin = test.myjit.cpu.ESP

class Test_PUSHAD_32(Asm_Test_32):
    MYSTRING = 'test pushad 32'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        init_regs(self)
        self.buf = b''
        for reg_name in reversed(['EAX', 'ECX', 'EDX', 'EBX', 'ESP', 'EBP', 'ESI', 'EDI']):
            self.buf += pck32(getattr(self.myjit.cpu, reg_name))
    TXT = '\n    main:\n       PUSHAD\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.ESP == self.stk_origin - 4 * 8
        buf = self.myjit.vm.get_mem(self.myjit.cpu.ESP, 4 * 8)
        assert buf == self.buf

class Test_PUSHA_32(Asm_Test_32):
    MYSTRING = 'test pusha 32'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            return 10
        init_regs(self)
        self.buf = b''
        for reg_name in reversed(['AX', 'CX', 'DX', 'BX', 'SP', 'BP', 'SI', 'DI']):
            self.buf += pck16(getattr(self.myjit.cpu, reg_name))
    TXT = '\n    main:\n       PUSHA\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.ESP == self.stk_origin - 2 * 8
        buf = self.myjit.vm.get_mem(self.myjit.cpu.ESP, 2 * 8)
        assert buf == self.buf

class Test_PUSHA_16(Asm_Test_16):
    MYSTRING = 'test pusha 16'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            while True:
                i = 10
        init_regs(self)
        self.buf = b''
        for reg_name in reversed(['AX', 'CX', 'DX', 'BX', 'SP', 'BP', 'SI', 'DI']):
            self.buf += pck16(getattr(self.myjit.cpu, reg_name))
    TXT = '\n    main:\n       PUSHA\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.ESP == self.stk_origin - 2 * 8
        buf = self.myjit.vm.get_mem(self.myjit.cpu.SP, 2 * 8)
        assert buf == self.buf

class Test_PUSHAD_16(Asm_Test_16):
    MYSTRING = 'test pushad 16'

    def prepare(self):
        if False:
            return 10
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            print('Hello World!')
        init_regs(self)
        self.buf = b''
        for reg_name in reversed(['EAX', 'ECX', 'EDX', 'EBX', 'ESP', 'EBP', 'ESI', 'EDI']):
            self.buf += pck32(getattr(self.myjit.cpu, reg_name))
    TXT = '\n    main:\n       PUSHAD\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.ESP == self.stk_origin - 4 * 8
        buf = self.myjit.vm.get_mem(self.myjit.cpu.SP, 4 * 8)
        assert buf == self.buf

class Test_PUSH_mode32_32(Asm_Test_32):
    MYSTRING = 'test push mode32 32'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        init_regs(self)
        self.buf = b''
        self.buf += pck32(287454020)
    TXT = '\n    main:\n       PUSH 0x11223344\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.ESP == self.stk_origin - 4
        buf = self.myjit.vm.get_mem(self.myjit.cpu.ESP, 4)
        assert buf == self.buf

class Test_PUSH_mode32_16(Asm_Test_32):
    MYSTRING = 'test push mode32 16'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        init_regs(self)
        self.buf = b''
        self.buf += pck16(4386)
    TXT = '\n    main:\n       PUSHW 0x1122\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.ESP == self.stk_origin - 2
        buf = self.myjit.vm.get_mem(self.myjit.cpu.ESP, 2)
        assert buf == self.buf

class Test_PUSH_mode16_16(Asm_Test_16):
    MYSTRING = 'test push mode16 16'

    def prepare(self):
        if False:
            i = 10
            return i + 15
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        init_regs(self)
        self.buf = b''
        self.buf += pck16(4386)
    TXT = '\n    main:\n       PUSHW 0x1122\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.ESP == self.stk_origin - 2
        buf = self.myjit.vm.get_mem(self.myjit.cpu.ESP, 2)
        assert buf == self.buf

class Test_PUSH_mode16_32(Asm_Test_16):
    MYSTRING = 'test push mode16 32'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        init_regs(self)
        self.buf = b''
        self.buf += pck32(287454020)
    TXT = '\n    main:\n       .byte 0x66, 0x68, 0x44, 0x33, 0x22, 0x11\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.ESP == self.stk_origin - 4
        buf = self.myjit.vm.get_mem(self.myjit.cpu.ESP, 4)
        assert buf == self.buf

class Test_POP_mode32_32(Asm_Test_32):
    MYSTRING = 'test pop mode32 32'

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = 287454020
        self.myjit.push_uint32_t(self.value)
        init_regs(self)
    TXT = '\n    main:\n       POP EAX\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            while True:
                i = 10
        assert self.myjit.cpu.ESP == self.stk_origin + 4
        assert self.myjit.cpu.EAX == self.value

class Test_POP_mode32_16(Asm_Test_32):
    MYSTRING = 'test pop mode32 16'

    def prepare(self):
        if False:
            i = 10
            return i + 15
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        self.value = 4386
        self.myjit.push_uint16_t(self.value)
        init_regs(self)
    TXT = '\n    main:\n       POPW AX\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.ESP == self.stk_origin + 2
        assert self.myjit.cpu.AX == self.value

class Test_POP_mode16_16(Asm_Test_16):
    MYSTRING = 'test pop mode16 16'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        self.value = 4386
        self.myjit.push_uint16_t(self.value)
        init_regs(self)
    TXT = '\n    main:\n       POPW AX\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            return 10
        assert self.myjit.cpu.ESP == self.stk_origin + 2
        assert self.myjit.cpu.AX == self.value

class Test_POP_mode16_32(Asm_Test_16):
    MYSTRING = 'test pop mode16 32'

    def prepare(self):
        if False:
            while True:
                i = 10
        self.loc_db.add_location('lbl_ret', self.ret_addr)

    def test_init(self):
        if False:
            print('Hello World!')
        self.value = 287454020
        self.myjit.cpu.SP -= 4
        self.myjit.vm.set_mem(self.myjit.cpu.SP, pck32(self.value))
        init_regs(self)
    TXT = '\n    main:\n       POP EAX\n       JMP lbl_ret\n    '

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.ESP == self.stk_origin + 4
        assert self.myjit.cpu.EAX == self.value
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_PUSHA_16, Test_PUSHA_32, Test_PUSHAD_16, Test_PUSHAD_32, Test_PUSH_mode32_32, Test_PUSH_mode32_16, Test_PUSH_mode16_16, Test_PUSH_mode16_32, Test_POP_mode32_32, Test_POP_mode32_16, Test_POP_mode16_16, Test_POP_mode16_32]]