import unittest
from manticore.native.cpu.abstractcpu import ConcretizeRegister
from manticore.native.cpu.x86 import AMD64Cpu
from manticore.native.memory import *
from manticore.core.smtlib.solver import Z3Solver
solver = Z3Solver.instance()

class CPUTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    class ROOperand:
        """Mocking class for operand ronly"""

        def __init__(self, size, value):
            if False:
                for i in range(10):
                    print('nop')
            self.size = size
            self.value = value

        def read(self):
            if False:
                while True:
                    i = 10
            return self.value & (1 << self.size) - 1

    class RWOperand(ROOperand):
        """Mocking class for operand rw"""

        def write(self, value):
            if False:
                for i in range(10):
                    print('nop')
            self.value = value & (1 << self.size) - 1
            return self.value

    def test_MOVHPD_1(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_1\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347996113, 'IVATE\x00\x00\x00')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 340282366920937259247406077050255658055
        cpu.RDI = 140737347996105
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347996113:140737347996121], [b'I', b'V', b'A', b'T', b'E', b'\x00', b'\x00', b'\x00'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 5492818941963568420245782219847)
        self.assertEqual(cpu.RDI, 140737347996105)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_10(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_10\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 340282366842019785943813698650299255879
        cpu.RDI = 140737347995854
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RDI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_11(self):
        if False:
            return 10
        'Instruction MOVHPD_11\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 88109632480871197291218000195730623559
        cpu.RSI = 140737347995854
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RSI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_12(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_12\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 340282366842019785943813698650299255879
        cpu.RDI = 140737347995854
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RDI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_13(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_13\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347981312, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347983578, 'tart_mai')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 8313472711475879775
        cpu.RDI = 140737347983570
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347983578:140737347983586], [b't', b'a', b'r', b't', b'_', b'm', b'a', b'i'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 140074810698054820722452200425796689759)
        self.assertEqual(cpu.RDI, 140737347983570)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_14(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_14\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347977216, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347979931, '\x00acct\x00_n')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 1004656093761930814559
        cpu.RSI = 140737347979923
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347979931:140737347979939], [b'\x00', b'a', b'c', b'c', b't', b'\x00', b'_', b'n'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 146708356959127564005328096862462043231)
        self.assertEqual(cpu.RSI, 140737347979923)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_15(self):
        if False:
            i = 10
            return i + 15
        'Instruction MOVHPD_15\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347989504, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347990254, 'nable_se')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 1003428846375260675935
        cpu.RSI = 140737347990246
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347990254:140737347990262], [b'n', b'a', b'b', b'l', b'e', b'_', b's', b'e'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 134851076577508085086976746042965122911)
        self.assertEqual(cpu.RSI, 140737347990246)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_16(self):
        if False:
            i = 10
            return i + 15
        'Instruction MOVHPD_16\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 88109632480871197291218000195730623559
        cpu.RSI = 140737347995854
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RSI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_17(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_17\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351872512, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351874161, '_dso_for')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 7236837539639485535
        cpu.RDI = 140737351874153
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737351874161:140737351874169], [b'_', b'd', b's', b'o', b'_', b'f', b'o', b'r'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 152110412837725123259047000460919333983)
        self.assertEqual(cpu.RDI, 140737351874153)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_18(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_18\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 88109632480871197291218000195730623559
        cpu.RSI = 140737347995854
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RSI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_19(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_19\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351872512, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351874384, 'obal_ro\x00')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 7811316963618353759
        cpu.RDI = 140737351874376
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737351874384:140737351874392], [b'o', b'b', b'a', b'l', b'_', b'r', b'o', b'\x00'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 578664706209732724830403288697696863)
        self.assertEqual(cpu.RDI, 140737351874376)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_2\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 340282366842019785943813698650299255879
        cpu.RDI = 140737347995854
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RDI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_20(self):
        if False:
            i = 10
            return i + 15
        'Instruction MOVHPD_20\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995831, '-x86-64.')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 8679965255892034668
        cpu.RDI = 140737347995823
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347995831:140737347995839], [b'-', b'x', b'8', b'6', b'-', b'6', b'4', b'.'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 61415586074916309421369241318231729260)
        self.assertEqual(cpu.RDI, 140737347995823)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_21(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_21\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737349521408, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737349524016, '6\x00__vdso')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 133537424963509100900314316955479591244
        cpu.RSI = 140737349524008
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737349524016:140737349524024], [b'6', b'\x00', b'_', b'_', b'v', b'd', b's', b'o'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 148143459290256633805182000720633547084)
        self.assertEqual(cpu.RSI, 140737349524008)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_3(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_3\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 340282366842019785943813698650299255879
        cpu.RDI = 140737347995854
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RDI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_4(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_4\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 88109632480871197291218000195730623559
        cpu.RSI = 140737347995854
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RSI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_5(self):
        if False:
            return 10
        'Instruction MOVHPD_5\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.mmap(140737354113024, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        mem.write(140737354113804, '6\x00\x00\x00\x00\x00\x02\x00')
        cpu.XMM1 = 340282366842019785943813698740812663116
        cpu.RDI = 140737354113796
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(mem[140737354113804:140737354113812], [b'6', b'\x00', b'\x00', b'\x00', b'\x00', b'\x00', b'\x02', b'\x00'])
        self.assertEqual(cpu.XMM1, 10384593717070654710068880547400012)
        self.assertEqual(cpu.RDI, 140737354113796)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_MOVHPD_6(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_6\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 88109632480871197291218000195730623559
        cpu.RSI = 140737347995854
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RSI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_7(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_7\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347995862, '2.5\x00GLIB')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        cpu.XMM2 = 88109632480871197291218000195730623559
        cpu.RSI = 140737347995854
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737347995862:140737347995870], [b'2', b'.', b'5', b'\x00', b'G', b'L', b'I', b'B'])
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(cpu.XMM2, 88109632480871197291218000195730623559)
        self.assertEqual(cpu.RSI, 140737347995854)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_8(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_8\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.mmap(140737354100736, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        mem.write(140737354101928, '_64-linu')
        cpu.XMM2 = 3907004821653777455
        cpu.RSI = 140737354101920
        cpu.RIP = 140737351985491
        cpu.execute()
        self.assertEqual(mem[140737351985491:140737351985496], [b'f', b'\x0f', b'\x16', b'V', b'\x08'])
        self.assertEqual(mem[140737354101928:140737354101936], [b'_', b'6', b'4', b'-', b'l', b'i', b'n', b'u'])
        self.assertEqual(cpu.XMM2, 156092966384913869483545010807748783151)
        self.assertEqual(cpu.RSI, 140737354101920)
        self.assertEqual(cpu.RIP, 140737351985496)

    def test_MOVHPD_9(self):
        if False:
            return 10
        'Instruction MOVHPD_9\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347981312, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737347982101, 'emalign\x00')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        cpu.XMM1 = 340282366842019785948365997980487802719
        cpu.RDI = 140737347982093
        cpu.RIP = 140737351985486
        cpu.execute()
        self.assertEqual(mem[140737347982101:140737347982109], [b'e', b'm', b'a', b'l', b'i', b'g', b'n', b'\x00'])
        self.assertEqual(mem[140737351985486:140737351985491], [b'f', b'\x0f', b'\x16', b'O', b'\x08'])
        self.assertEqual(cpu.XMM1, 573250095127234633104266320675626847)
        self.assertEqual(cpu.RDI, 140737347982093)
        self.assertEqual(cpu.RIP, 140737351985491)

    def test_PSLLDQ_1(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_1\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 1
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 72057594037927936)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_10(self):
        if False:
            i = 10
            return i + 15
        'Instruction PSLLDQ_10\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 61723168909761380161964749838612430848)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_11(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_11\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 61723168909761380161964749838612430848)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_12(self):
        if False:
            return 10
        'Instruction PSLLDQ_12\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 61723168909761380161964749838612430848)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_13(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_13\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 1
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 72057594037927936)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_14(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PSLLDQ_14\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 61723168909761380161964749838612430848)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_15(self):
        if False:
            return 10
        'Instruction PSLLDQ_15\n        Groups: sse2\n        0x7ffff7df389d:     pslldq  xmm2, 4\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989405, 'f\x0fsú\x04')
        cpu.XMM2 = 15576890578022371418309123164483122
        cpu.RIP = 140737351989405
        cpu.execute()
        self.assertEqual(mem[140737351989405:140737351989410], [b'f', b'\x0f', b's', b'\xfa', b'\x04'])
        self.assertEqual(cpu.XMM2, 10384752173395664791945953216036864)
        self.assertEqual(cpu.RIP, 140737351989410)

    def test_PSLLDQ_16(self):
        if False:
            return 10
        'Instruction PSLLDQ_16\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 61723168909761380161964749838612430848)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_17(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_17\n        Groups: sse2\n        0x7ffff7df39dd:     pslldq  xmm2, 3\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989725, 'f\x0fsú\x03')
        cpu.XMM2 = 97429698321087917969083959610337675008
        cpu.RIP = 140737351989725
        cpu.execute()
        self.assertEqual(mem[140737351989725:140737351989730], [b'f', b'\x0f', b's', b'\xfa', b'\x03'])
        self.assertEqual(cpu.XMM2, 276128700049446162655260478745346048)
        self.assertEqual(cpu.RIP, 140737351989730)

    def test_PSLLDQ_18(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PSLLDQ_18\n        Groups: sse2\n        0x7ffff7df389d:     pslldq  xmm2, 4\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989405, 'f\x0fsú\x04')
        cpu.XMM2 = 136076131895038381799925370591722039395
        cpu.RIP = 140737351989405
        cpu.execute()
        self.assertEqual(mem[140737351989405:140737351989410], [b'f', b'\x0f', b's', b'\xfa', b'\x04'])
        self.assertEqual(cpu.XMM2, 126278919537221597046423674937331941376)
        self.assertEqual(cpu.RIP, 140737351989410)

    def test_PSLLDQ_19(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_19\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 1
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 72057594037927936)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_2(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_2\n        Groups: sse2\n        0x7ffff7df2f70:     pslldq  xmm2, 0xb\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351987056, 'f\x0fsú\x0b')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351987056
        cpu.execute()
        self.assertEqual(mem[140737351987056:140737351987061], [b'f', b'\x0f', b's', b'\xfa', b'\x0b'])
        self.assertEqual(cpu.XMM2, 132104554884493019491015862172149350400)
        self.assertEqual(cpu.RIP, 140737351987061)

    def test_PSLLDQ_20(self):
        if False:
            i = 10
            return i + 15
        'Instruction PSLLDQ_20\n        Groups: sse2\n        0x7ffff7df3970:     pslldq  xmm2, 3\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989616, 'f\x0fsú\x03')
        cpu.XMM2 = 66702505917742520628121034217238130281
        cpu.RIP = 140737351989616
        cpu.execute()
        self.assertEqual(mem[140737351989616:140737351989621], [b'f', b'\x0f', b's', b'\xfa', b'\x03'])
        self.assertEqual(cpu.XMM2, 153101124148370467217615035531131879424)
        self.assertEqual(cpu.RIP, 140737351989621)

    def test_PSLLDQ_21(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_21\n        Groups: sse2\n        0x7ffff7df3830:     pslldq  xmm2, 4\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989296, 'f\x0fsú\x04')
        cpu.XMM2 = 126625887935385241370692449898806329929
        cpu.RIP = 140737351989296
        cpu.execute()
        self.assertEqual(mem[140737351989296:140737351989301], [b'f', b'\x0f', b's', b'\xfa', b'\x04'])
        self.assertEqual(cpu.XMM2, 101389984890772213670702594761716400128)
        self.assertEqual(cpu.RIP, 140737351989301)

    def test_PSLLDQ_3(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PSLLDQ_3\n        Groups: sse2\n        0x7ffff7df3ab0:     pslldq  xmm2, 2\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989936, 'f\x0fsú\x02')
        cpu.XMM2 = 131595932217195380898632096716893942628
        cpu.RIP = 140737351989936
        cpu.execute()
        self.assertEqual(mem[140737351989936:140737351989941], [b'f', b'\x0f', b's', b'\xfa', b'\x02'])
        self.assertEqual(cpu.XMM2, 154706541852064556987039687627872927744)
        self.assertEqual(cpu.RIP, 140737351989941)

    def test_PSLLDQ_4(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_4\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 61723168909761380161964749838612430848)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_5(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_5\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 140163140585241516644150668835041143808
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 61723168909761380161964749838612430848)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_6(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PSLLDQ_6\n        Groups: sse2\n        0x7ffff7df389d:     pslldq  xmm2, 4\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989405, 'f\x0fsú\x04')
        cpu.XMM2 = 15576890578022371418309123164483122
        cpu.RIP = 140737351989405
        cpu.execute()
        self.assertEqual(mem[140737351989405:140737351989410], [b'f', b'\x0f', b's', b'\xfa', b'\x04'])
        self.assertEqual(cpu.XMM2, 10384752173395664791945953216036864)
        self.assertEqual(cpu.RIP, 140737351989410)

    def test_PSLLDQ_7(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_7\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = 1
        cpu.RIP = 140737351988336
        cpu.execute()
        self.assertEqual(mem[140737351988336:140737351988341], [b'f', b'\x0f', b's', b'\xfa', b'\x07'])
        self.assertEqual(cpu.XMM2, 72057594037927936)
        self.assertEqual(cpu.RIP, 140737351988341)

    def test_PSLLDQ_8(self):
        if False:
            return 10
        'Instruction PSLLDQ_8\n        Groups: sse2\n        0x7ffff7df39dd:     pslldq  xmm2, 3\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989725, 'f\x0fsú\x03')
        cpu.XMM2 = 154696117092760325124648776676031882095
        cpu.RIP = 140737351989725
        cpu.execute()
        self.assertEqual(mem[140737351989725:140737351989730], [b'f', b'\x0f', b's', b'\xfa', b'\x03'])
        self.assertEqual(cpu.XMM2, 148107273809595710738464457560820809728)
        self.assertEqual(cpu.RIP, 140737351989730)

    def test_PSLLDQ_9(self):
        if False:
            return 10
        'Instruction PSLLDQ_9\n        Groups: sse2\n        0x7ffff7df3c5d:     pslldq  xmm2, 1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351990365, 'f\x0fsú\x01')
        cpu.XMM2 = 138432768838165356457729754383509712233
        cpu.RIP = 140737351990365
        cpu.execute()
        self.assertEqual(mem[140737351990365:140737351990370], [b'f', b'\x0f', b's', b'\xfa', b'\x01'])
        self.assertEqual(cpu.XMM2, 49422662792731052987857949274592340224)
        self.assertEqual(cpu.RIP, 140737351990370)

    def test_MOVHPD_1_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_1\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996113)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996114)
        value = cs.new_bitvec(8)
        cs.add(value == 86)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996115)
        value = cs.new_bitvec(8)
        cs.add(value == 65)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996116)
        value = cs.new_bitvec(8)
        cs.add(value == 84)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996117)
        value = cs.new_bitvec(8)
        cs.add(value == 69)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996118)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996119)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347996120)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        mem.write(140737351985489, 'O\x08')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 340282366920937259247406077050255658055)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347996105)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347996115, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(140737347996116, 8) == ord('T'))
        condition = Operators.AND(condition, cpu.read_int(140737347996117, 8) == ord('E'))
        condition = Operators.AND(condition, cpu.read_int(140737347996118, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347996119, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347996120, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347996113, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347996114, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.XMM1 == 5492818941963568420245782219847)
        condition = Operators.AND(condition, cpu.RDI == 140737347996105)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_10_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_10\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 340282366842019785943813698650299255879)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347995854)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM1 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RDI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_11_symbolic(self):
        if False:
            return 10
        'Instruction MOVHPD_11\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 88109632480871197291218000195730623559)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347995854)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM2 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RSI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_12_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_12\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 340282366842019785943813698650299255879)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347995854)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM1 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RDI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_13_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_13\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347981312, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983578)
        value = cs.new_bitvec(8)
        cs.add(value == 116)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983579)
        value = cs.new_bitvec(8)
        cs.add(value == 97)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983580)
        value = cs.new_bitvec(8)
        cs.add(value == 114)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983581)
        value = cs.new_bitvec(8)
        cs.add(value == 116)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983582)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983583)
        value = cs.new_bitvec(8)
        cs.add(value == 109)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983584)
        value = cs.new_bitvec(8)
        cs.add(value == 97)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347983585)
        value = cs.new_bitvec(8)
        cs.add(value == 105)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 8313472711475879775)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347983570)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347983578, 8) == ord('t'))
        condition = Operators.AND(condition, cpu.read_int(140737347983579, 8) == ord('a'))
        condition = Operators.AND(condition, cpu.read_int(140737347983580, 8) == ord('r'))
        condition = Operators.AND(condition, cpu.read_int(140737347983581, 8) == ord('t'))
        condition = Operators.AND(condition, cpu.read_int(140737347983582, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737347983583, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(140737347983584, 8) == ord('a'))
        condition = Operators.AND(condition, cpu.read_int(140737347983585, 8) == ord('i'))
        condition = Operators.AND(condition, cpu.XMM1 == 140074810698054820722452200425796689759)
        condition = Operators.AND(condition, cpu.RDI == 140737347983570)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_14_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_14\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347977216, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979931)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979932)
        value = cs.new_bitvec(8)
        cs.add(value == 97)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979933)
        value = cs.new_bitvec(8)
        cs.add(value == 99)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979934)
        value = cs.new_bitvec(8)
        cs.add(value == 99)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979935)
        value = cs.new_bitvec(8)
        cs.add(value == 116)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979936)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979937)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347979938)
        value = cs.new_bitvec(8)
        cs.add(value == 110)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 1004656093761930814559)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347979923)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347979931, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347979932, 8) == ord('a'))
        condition = Operators.AND(condition, cpu.read_int(140737347979933, 8) == ord('c'))
        condition = Operators.AND(condition, cpu.read_int(140737347979934, 8) == ord('c'))
        condition = Operators.AND(condition, cpu.read_int(140737347979935, 8) == ord('t'))
        condition = Operators.AND(condition, cpu.read_int(140737347979936, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347979937, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737347979938, 8) == ord('n'))
        condition = Operators.AND(condition, cpu.XMM2 == 146708356959127564005328096862462043231)
        condition = Operators.AND(condition, cpu.RSI == 140737347979923)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_15_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction MOVHPD_15\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347989504, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990254)
        value = cs.new_bitvec(8)
        cs.add(value == 110)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990255)
        value = cs.new_bitvec(8)
        cs.add(value == 97)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990256)
        value = cs.new_bitvec(8)
        cs.add(value == 98)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990257)
        value = cs.new_bitvec(8)
        cs.add(value == 108)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990258)
        value = cs.new_bitvec(8)
        cs.add(value == 101)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990259)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990260)
        value = cs.new_bitvec(8)
        cs.add(value == 115)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347990261)
        value = cs.new_bitvec(8)
        cs.add(value == 101)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 1003428846375260675935)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347990246)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347990254, 8) == ord('n'))
        condition = Operators.AND(condition, cpu.read_int(140737347990255, 8) == ord('a'))
        condition = Operators.AND(condition, cpu.read_int(140737347990256, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(140737347990257, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(140737347990258, 8) == ord('e'))
        condition = Operators.AND(condition, cpu.read_int(140737347990259, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737347990260, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737347990261, 8) == ord('e'))
        condition = Operators.AND(condition, cpu.XMM2 == 134851076577508085086976746042965122911)
        condition = Operators.AND(condition, cpu.RSI == 140737347990246)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_16_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_16\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 88109632480871197291218000195730623559)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347995854)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM2 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RSI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_17_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_17\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351872512, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874161)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874162)
        value = cs.new_bitvec(8)
        cs.add(value == 100)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874163)
        value = cs.new_bitvec(8)
        cs.add(value == 115)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874164)
        value = cs.new_bitvec(8)
        cs.add(value == 111)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874165)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874166)
        value = cs.new_bitvec(8)
        cs.add(value == 102)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874167)
        value = cs.new_bitvec(8)
        cs.add(value == 111)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874168)
        value = cs.new_bitvec(8)
        cs.add(value == 114)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 7236837539639485535)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737351874153)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737351874161, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737351874162, 8) == ord('d'))
        condition = Operators.AND(condition, cpu.read_int(140737351874163, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351874164, 8) == ord('o'))
        condition = Operators.AND(condition, cpu.read_int(140737351874165, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737351874166, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351874167, 8) == ord('o'))
        condition = Operators.AND(condition, cpu.read_int(140737351874168, 8) == ord('r'))
        condition = Operators.AND(condition, cpu.XMM1 == 152110412837725123259047000460919333983)
        condition = Operators.AND(condition, cpu.RDI == 140737351874153)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_18_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_18\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 88109632480871197291218000195730623559)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347995854)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM2 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RSI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_19_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction MOVHPD_19\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351872512, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874384)
        value = cs.new_bitvec(8)
        cs.add(value == 111)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874385)
        value = cs.new_bitvec(8)
        cs.add(value == 98)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874386)
        value = cs.new_bitvec(8)
        cs.add(value == 97)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874387)
        value = cs.new_bitvec(8)
        cs.add(value == 108)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874388)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874389)
        value = cs.new_bitvec(8)
        cs.add(value == 114)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874390)
        value = cs.new_bitvec(8)
        cs.add(value == 111)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737351874391)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        mem.write(140737351985488, '\x16O\x08')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 7811316963618353759)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737351874376)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737351874387, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(140737351874388, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737351874389, 8) == ord('r'))
        condition = Operators.AND(condition, cpu.read_int(140737351874390, 8) == ord('o'))
        condition = Operators.AND(condition, cpu.read_int(140737351874391, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737351874384, 8) == ord('o'))
        condition = Operators.AND(condition, cpu.read_int(140737351874385, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(140737351874386, 8) == ord('a'))
        condition = Operators.AND(condition, cpu.XMM1 == 578664706209732724830403288697696863)
        condition = Operators.AND(condition, cpu.RDI == 140737351874376)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_2_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_2\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 340282366842019785943813698650299255879)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347995854)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM1 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RDI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_20_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction MOVHPD_20\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995831)
        value = cs.new_bitvec(8)
        cs.add(value == 45)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995832)
        value = cs.new_bitvec(8)
        cs.add(value == 120)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995833)
        value = cs.new_bitvec(8)
        cs.add(value == 56)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995834)
        value = cs.new_bitvec(8)
        cs.add(value == 54)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995835)
        value = cs.new_bitvec(8)
        cs.add(value == 45)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995836)
        value = cs.new_bitvec(8)
        cs.add(value == 54)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995837)
        value = cs.new_bitvec(8)
        cs.add(value == 52)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995838)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 8679965255892034668)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347995823)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995831, 8) == ord('-'))
        condition = Operators.AND(condition, cpu.read_int(140737347995832, 8) == ord('x'))
        condition = Operators.AND(condition, cpu.read_int(140737347995833, 8) == ord('8'))
        condition = Operators.AND(condition, cpu.read_int(140737347995834, 8) == ord('6'))
        condition = Operators.AND(condition, cpu.read_int(140737347995835, 8) == ord('-'))
        condition = Operators.AND(condition, cpu.read_int(140737347995836, 8) == ord('6'))
        condition = Operators.AND(condition, cpu.read_int(140737347995837, 8) == ord('4'))
        condition = Operators.AND(condition, cpu.read_int(140737347995838, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.XMM1 == 61415586074916309421369241318231729260)
        condition = Operators.AND(condition, cpu.RDI == 140737347995823)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_21_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_21\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737349521408, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524016)
        value = cs.new_bitvec(8)
        cs.add(value == 54)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524017)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524018)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524019)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524020)
        value = cs.new_bitvec(8)
        cs.add(value == 118)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524021)
        value = cs.new_bitvec(8)
        cs.add(value == 100)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524022)
        value = cs.new_bitvec(8)
        cs.add(value == 115)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737349524023)
        value = cs.new_bitvec(8)
        cs.add(value == 111)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 133537424963509100900314316955479591244)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737349524008)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737349524016, 8) == ord('6'))
        condition = Operators.AND(condition, cpu.read_int(140737349524017, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737349524018, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737349524019, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737349524020, 8) == ord('v'))
        condition = Operators.AND(condition, cpu.read_int(140737349524021, 8) == ord('d'))
        condition = Operators.AND(condition, cpu.read_int(140737349524022, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737349524023, 8) == ord('o'))
        condition = Operators.AND(condition, cpu.XMM2 == 148143459290256633805182000720633547084)
        condition = Operators.AND(condition, cpu.RSI == 140737349524008)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_3_symbolic(self):
        if False:
            return 10
        'Instruction MOVHPD_3\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 340282366842019785943813698650299255879)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347995854)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM1 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RDI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_4_symbolic(self):
        if False:
            return 10
        'Instruction MOVHPD_4\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 88109632480871197291218000195730623559)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347995854)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM2 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RSI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_5_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction MOVHPD_5\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.mmap(140737354113024, 4096, 'rwx')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113804)
        value = cs.new_bitvec(8)
        cs.add(value == 54)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113805)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113806)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        mem.write(140737351985487, '\x0f')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113808)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113809)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        mem.write(140737351985490, '\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113811)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        mem.write(140737351985486, 'f')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113807)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        mem.write(140737351985488, '\x16O')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354113810)
        value = cs.new_bitvec(8)
        cs.add(value == 2)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 340282366842019785943813698740812663116)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737354113796)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737354113804, 8) == ord('6'))
        condition = Operators.AND(condition, cpu.read_int(140737354113805, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737354113806, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737354113811, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737354113807, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737354113808, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737354113809, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737354113810, 8) == ord('\x02'))
        condition = Operators.AND(condition, cpu.XMM1 == 10384593717070654710068880547400012)
        condition = Operators.AND(condition, cpu.RDI == 140737354113796)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_6_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction MOVHPD_6\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 88109632480871197291218000195730623559)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347995854)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM2 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RSI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_7_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_7\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347993600, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995862)
        value = cs.new_bitvec(8)
        cs.add(value == 50)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995863)
        value = cs.new_bitvec(8)
        cs.add(value == 46)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995864)
        value = cs.new_bitvec(8)
        cs.add(value == 53)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995865)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995866)
        value = cs.new_bitvec(8)
        cs.add(value == 71)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995867)
        value = cs.new_bitvec(8)
        cs.add(value == 76)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995868)
        value = cs.new_bitvec(8)
        cs.add(value == 73)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347995869)
        value = cs.new_bitvec(8)
        cs.add(value == 66)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 88109632480871197291218000195730623559)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737347995854)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737347995863, 8) == ord('.'))
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737347995862, 8) == ord('2'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347995864, 8) == ord('5'))
        condition = Operators.AND(condition, cpu.read_int(140737347995865, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.read_int(140737347995866, 8) == ord('G'))
        condition = Operators.AND(condition, cpu.read_int(140737347995867, 8) == ord('L'))
        condition = Operators.AND(condition, cpu.read_int(140737347995868, 8) == ord('I'))
        condition = Operators.AND(condition, cpu.read_int(140737347995869, 8) == ord('B'))
        condition = Operators.AND(condition, cpu.XMM2 == 88109632480871197291218000195730623559)
        condition = Operators.AND(condition, cpu.RSI == 140737347995854)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_8_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction MOVHPD_8\n        Groups: sse2\n        0x7ffff7df2953:     movhpd  xmm2, qword ptr [rsi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.mmap(140737354100736, 4096, 'rwx')
        mem.write(140737351985491, 'f\x0f\x16V\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101928)
        value = cs.new_bitvec(8)
        cs.add(value == 95)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101929)
        value = cs.new_bitvec(8)
        cs.add(value == 54)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101930)
        value = cs.new_bitvec(8)
        cs.add(value == 52)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101931)
        value = cs.new_bitvec(8)
        cs.add(value == 45)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101932)
        value = cs.new_bitvec(8)
        cs.add(value == 108)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101933)
        value = cs.new_bitvec(8)
        cs.add(value == 105)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101934)
        value = cs.new_bitvec(8)
        cs.add(value == 110)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737354101935)
        value = cs.new_bitvec(8)
        cs.add(value == 117)
        mem[addr] = value
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 3907004821653777455)
        cpu.RSI = cs.new_bitvec(64)
        cs.add(cpu.RSI == 140737354101920)
        cpu.RIP = 140737351985491
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985491, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985492, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985493, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985494, 8) == ord('V'))
        condition = Operators.AND(condition, cpu.read_int(140737351985495, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737354101928, 8) == ord('_'))
        condition = Operators.AND(condition, cpu.read_int(140737354101929, 8) == ord('6'))
        condition = Operators.AND(condition, cpu.read_int(140737354101930, 8) == ord('4'))
        condition = Operators.AND(condition, cpu.read_int(140737354101931, 8) == ord('-'))
        condition = Operators.AND(condition, cpu.read_int(140737354101932, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(140737354101933, 8) == ord('i'))
        condition = Operators.AND(condition, cpu.read_int(140737354101934, 8) == ord('n'))
        condition = Operators.AND(condition, cpu.read_int(140737354101935, 8) == ord('u'))
        condition = Operators.AND(condition, cpu.XMM2 == 156092966384913869483545010807748783151)
        condition = Operators.AND(condition, cpu.RSI == 140737354101920)
        condition = Operators.AND(condition, cpu.RIP == 140737351985496)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_MOVHPD_9_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction MOVHPD_9\n        Groups: sse2\n        0x7ffff7df294e:     movhpd  xmm1, qword ptr [rdi + 8]\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737347981312, 4096, 'rwx')
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351985486, 'f\x0f\x16O\x08')
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982101)
        value = cs.new_bitvec(8)
        cs.add(value == 101)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982102)
        value = cs.new_bitvec(8)
        cs.add(value == 109)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982103)
        value = cs.new_bitvec(8)
        cs.add(value == 97)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982104)
        value = cs.new_bitvec(8)
        cs.add(value == 108)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982105)
        value = cs.new_bitvec(8)
        cs.add(value == 105)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982106)
        value = cs.new_bitvec(8)
        cs.add(value == 103)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982107)
        value = cs.new_bitvec(8)
        cs.add(value == 110)
        mem[addr] = value
        addr = cs.new_bitvec(64)
        cs.add(addr == 140737347982108)
        value = cs.new_bitvec(8)
        cs.add(value == 0)
        mem[addr] = value
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 340282366842019785948365997980487802719)
        cpu.RDI = cs.new_bitvec(64)
        cs.add(cpu.RDI == 140737347982093)
        cpu.RIP = 140737351985486
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351985486, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985487, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351985488, 8) == ord('\x16'))
        condition = Operators.AND(condition, cpu.read_int(140737351985489, 8) == ord('O'))
        condition = Operators.AND(condition, cpu.read_int(140737351985490, 8) == ord('\x08'))
        condition = Operators.AND(condition, cpu.read_int(140737347982101, 8) == ord('e'))
        condition = Operators.AND(condition, cpu.read_int(140737347982102, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(140737347982103, 8) == ord('a'))
        condition = Operators.AND(condition, cpu.read_int(140737347982104, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(140737347982105, 8) == ord('i'))
        condition = Operators.AND(condition, cpu.read_int(140737347982106, 8) == ord('g'))
        condition = Operators.AND(condition, cpu.read_int(140737347982107, 8) == ord('n'))
        condition = Operators.AND(condition, cpu.read_int(140737347982108, 8) == ord('\x00'))
        condition = Operators.AND(condition, cpu.XMM1 == 573250095127234633104266320675626847)
        condition = Operators.AND(condition, cpu.RDI == 140737347982093)
        condition = Operators.AND(condition, cpu.RIP == 140737351985491)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_1_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PSLLDQ_1\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 1)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 72057594037927936)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_10_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_10\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 61723168909761380161964749838612430848)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_11_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_11\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 61723168909761380161964749838612430848)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_12_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_12\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 61723168909761380161964749838612430848)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_13_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_13\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 1)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 72057594037927936)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_14_symbolic(self):
        if False:
            return 10
        'Instruction PSLLDQ_14\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 61723168909761380161964749838612430848)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_15_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PSLLDQ_15\n        Groups: sse2\n        0x7ffff7df389d:     pslldq  xmm2, 4\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989405, 'f\x0fsú\x04')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 15576890578022371418309123164483122)
        cpu.RIP = 140737351989405
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989408, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989409, 8) == ord('\x04'))
        condition = Operators.AND(condition, cpu.read_int(140737351989405, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989406, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989407, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.XMM2 == 10384752173395664791945953216036864)
        condition = Operators.AND(condition, cpu.RIP == 140737351989410)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_16_symbolic(self):
        if False:
            return 10
        'Instruction PSLLDQ_16\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 61723168909761380161964749838612430848)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_17_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_17\n        Groups: sse2\n        0x7ffff7df39dd:     pslldq  xmm2, 3\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989725, 'f\x0fsú\x03')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 97429698321087917969083959610337675008)
        cpu.RIP = 140737351989725
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989728, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989729, 8) == ord('\x03'))
        condition = Operators.AND(condition, cpu.read_int(140737351989725, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989726, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989727, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.XMM2 == 276128700049446162655260478745346048)
        condition = Operators.AND(condition, cpu.RIP == 140737351989730)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_18_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_18\n        Groups: sse2\n        0x7ffff7df389d:     pslldq  xmm2, 4\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989405, 'f\x0fsú\x04')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 136076131895038381799925370591722039395)
        cpu.RIP = 140737351989405
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989408, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989409, 8) == ord('\x04'))
        condition = Operators.AND(condition, cpu.read_int(140737351989405, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989406, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989407, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.XMM2 == 126278919537221597046423674937331941376)
        condition = Operators.AND(condition, cpu.RIP == 140737351989410)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_19_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_19\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 1)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 72057594037927936)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_2_symbolic(self):
        if False:
            return 10
        'Instruction PSLLDQ_2\n        Groups: sse2\n        0x7ffff7df2f70:     pslldq  xmm2, 0xb\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351983104, 4096, 'rwx')
        mem.write(140737351987056, 'f\x0fsú\x0b')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351987056
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351987056, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351987057, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351987058, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351987059, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351987060, 8) == ord('\x0b'))
        condition = Operators.AND(condition, cpu.XMM2 == 132104554884493019491015862172149350400)
        condition = Operators.AND(condition, cpu.RIP == 140737351987061)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_20_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PSLLDQ_20\n        Groups: sse2\n        0x7ffff7df3970:     pslldq  xmm2, 3\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989616, 'f\x0fsú\x03')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 66702505917742520628121034217238130281)
        cpu.RIP = 140737351989616
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989616, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989617, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989618, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351989619, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989620, 8) == ord('\x03'))
        condition = Operators.AND(condition, cpu.XMM2 == 153101124148370467217615035531131879424)
        condition = Operators.AND(condition, cpu.RIP == 140737351989621)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_21_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PSLLDQ_21\n        Groups: sse2\n        0x7ffff7df3830:     pslldq  xmm2, 4\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989296, 'f\x0fsú\x04')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 126625887935385241370692449898806329929)
        cpu.RIP = 140737351989296
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989296, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989297, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989298, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351989299, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989300, 8) == ord('\x04'))
        condition = Operators.AND(condition, cpu.XMM2 == 101389984890772213670702594761716400128)
        condition = Operators.AND(condition, cpu.RIP == 140737351989301)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_3_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PSLLDQ_3\n        Groups: sse2\n        0x7ffff7df3ab0:     pslldq  xmm2, 2\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989936, 'f\x0fsú\x02')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 131595932217195380898632096716893942628)
        cpu.RIP = 140737351989936
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989936, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989937, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989938, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351989939, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989940, 8) == ord('\x02'))
        condition = Operators.AND(condition, cpu.XMM2 == 154706541852064556987039687627872927744)
        condition = Operators.AND(condition, cpu.RIP == 140737351989941)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_4_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_4\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 61723168909761380161964749838612430848)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_5_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PSLLDQ_5\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 140163140585241516644150668835041143808)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 61723168909761380161964749838612430848)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_6_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PSLLDQ_6\n        Groups: sse2\n        0x7ffff7df389d:     pslldq  xmm2, 4\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989405, 'f\x0fsú\x04')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 15576890578022371418309123164483122)
        cpu.RIP = 140737351989405
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989408, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989409, 8) == ord('\x04'))
        condition = Operators.AND(condition, cpu.read_int(140737351989405, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989406, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989407, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.XMM2 == 10384752173395664791945953216036864)
        condition = Operators.AND(condition, cpu.RIP == 140737351989410)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_7_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_7\n        Groups: sse2\n        0x7ffff7df3470:     pslldq  xmm2, 7\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351988336, 'f\x0fsú\x07')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 1)
        cpu.RIP = 140737351988336
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351988336, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988337, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351988338, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.read_int(140737351988339, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351988340, 8) == ord('\x07'))
        condition = Operators.AND(condition, cpu.XMM2 == 72057594037927936)
        condition = Operators.AND(condition, cpu.RIP == 140737351988341)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_8_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PSLLDQ_8\n        Groups: sse2\n        0x7ffff7df39dd:     pslldq  xmm2, 3\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351989725, 'f\x0fsú\x03')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 154696117092760325124648776676031882095)
        cpu.RIP = 140737351989725
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351989728, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351989729, 8) == ord('\x03'))
        condition = Operators.AND(condition, cpu.read_int(140737351989725, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989726, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351989727, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.XMM2 == 148107273809595710738464457560820809728)
        condition = Operators.AND(condition, cpu.RIP == 140737351989730)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PSLLDQ_9_symbolic(self):
        if False:
            return 10
        'Instruction PSLLDQ_9\n        Groups: sse2\n        0x7ffff7df3c5d:     pslldq  xmm2, 1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(140737351987200, 4096, 'rwx')
        mem.write(140737351990365, 'f\x0fsú\x01')
        cpu.XMM2 = cs.new_bitvec(128)
        cs.add(cpu.XMM2 == 138432768838165356457729754383509712233)
        cpu.RIP = 140737351990365
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(140737351990368, 8) == ord('ú'))
        condition = Operators.AND(condition, cpu.read_int(140737351990369, 8) == ord('\x01'))
        condition = Operators.AND(condition, cpu.read_int(140737351990365, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(140737351990366, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(140737351990367, 8) == ord('s'))
        condition = Operators.AND(condition, cpu.XMM2 == 49422662792731052987857949274592340224)
        condition = Operators.AND(condition, cpu.RIP == 140737351990370)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))
if __name__ == '__main__':
    unittest.main()