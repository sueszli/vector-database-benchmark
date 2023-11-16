import unittest
from manticore.native.cpu.arm import Armv7RegisterFile as RF

class Armv7RFTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        if False:
            while True:
                i = 10
        self.r = RF()

    def test_init_state(self):
        if False:
            return 10
        self.assertEqual(self.r.read('R0'), 0)

    def test_write_read(self):
        if False:
            return 10
        self.r.write('R0', 1)
        self.assertEqual(self.r.read('R0'), 1)

    def test_write_read_sp(self):
        if False:
            while True:
                i = 10
        self.r.write('SP', 1)
        self.assertEqual(self.r.read('SP'), 1)

    def test_flag_wr(self):
        if False:
            return 10
        self.r.write('APSR_Z', True)
        self.assertEqual(self.r.read('APSR_Z'), True)

    def test_flag_wr_f(self):
        if False:
            while True:
                i = 10
        self.r.write('APSR_Z', False)
        self.assertEqual(self.r.read('APSR_Z'), False)

    def test_bad_reg_name(self):
        if False:
            i = 10
            return i + 15
        if __debug__:
            exception = AssertionError
        else:
            exception = KeyError
        with self.assertRaises(exception):
            nonexistant_reg = 'Pc'
            self.r.read(nonexistant_reg)

    def test_flag_wr_aspr(self):
        if False:
            print('Hello World!')
        self.r.write('APSR', 4294967295)
        self.assertEqual(self.r.read('APSR'), 4026531840)
        self.assertEqual(self.r.read('APSR_V'), True)
        self.assertEqual(self.r.read('APSR_C'), True)
        self.assertEqual(self.r.read('APSR_Z'), True)
        self.assertEqual(self.r.read('APSR_N'), True)
        self.r.write('APSR_N', False)
        self.assertEqual(self.r.read('APSR'), 1879048192)
        self.r.write('APSR_Z', False)
        self.assertEqual(self.r.read('APSR'), 805306368)
        self.r.write('APSR_C', False)
        self.assertEqual(self.r.read('APSR'), 268435456)
        self.r.write('APSR_V', False)
        self.assertEqual(self.r.read('APSR'), 0)

    def test_register_independence_wr(self):
        if False:
            for i in range(10):
                print('nop')
        regs = ('R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15')
        aliases = {'SB': 'R9', 'SL': 'R10', 'FP': 'R11', 'IP': 'R12', 'STACK': 'R13', 'SP': 'R13', 'LR': 'R14', 'PC': 'R15'}
        for j in range(16):
            for i in range(16):
                if i == j:
                    self.r.write(regs[i], 1094861636)
                else:
                    self.r.write(regs[i], 0)
            for (a, b) in aliases.items():
                self.assertEqual(self.r.read(a), self.r.read(b))
            for i in range(16):
                if i == j:
                    self.assertEqual(self.r.read(regs[i]), 1094861636)
                else:
                    self.assertEqual(self.r.read(regs[i]), 0)