import unittest
from manticore.native.cpu import bitwise

class BitwiseTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_mask(self):
        if False:
            while True:
                i = 10
        masked = bitwise.Mask(8)
        self.assertEqual(masked, 255)

    def test_get_bits(self):
        if False:
            i = 10
            return i + 15
        val = 2864434397
        result = bitwise.GetNBits(val, 8)
        self.assertEqual(result, 221)

    def test_lsl_nocarry(self):
        if False:
            while True:
                i = 10
        val = 43520
        (result, carry) = bitwise.LSL_C(val, 4, 32)
        self.assertEqual(result, 696320)
        self.assertEqual(carry, 0)

    def test_lsl_carry(self):
        if False:
            return 10
        val = 2147483648
        (result, carry) = bitwise.LSL_C(val, 1, 32)
        print(hex(result), '', hex(carry))
        self.assertEqual(result, 0)
        self.assertEqual(carry, 1)

    def test_lsr_nocarry(self):
        if False:
            print('Hello World!')
        val = 65527
        (result, carry) = bitwise.LSR_C(val, 4, 32)
        self.assertEqual(result, 4095)
        self.assertEqual(carry, 0)

    def test_lsr_carry(self):
        if False:
            print('Hello World!')
        val = 65528
        (result, carry) = bitwise.LSR_C(val, 4, 32)
        self.assertEqual(result, 4095)
        self.assertEqual(carry, 1)

    def test_asr_nocarry(self):
        if False:
            i = 10
            return i + 15
        val = 240
        (result, carry) = bitwise.ASR_C(val, 4, 32)
        self.assertEqual(result, 15)
        self.assertEqual(carry, 0)

    def test_asr_carry(self):
        if False:
            while True:
                i = 10
        val = 3
        (result, carry) = bitwise.ASR_C(val, 1, 32)
        self.assertEqual(result, 1)
        self.assertEqual(carry, 1)

    def test_ror_nocarry(self):
        if False:
            for i in range(10):
                print('nop')
        val = 240
        (result, carry) = bitwise.ROR_C(val, 4, 32)
        print(hex(result))
        self.assertEqual(result, 15)
        self.assertEqual(carry, 0)

    def test_ror_carry(self):
        if False:
            print('Hello World!')
        val = 3
        (result, carry) = bitwise.ROR_C(val, 1, 32)
        print(hex(result))
        self.assertEqual(result, 2147483649)
        self.assertEqual(carry, 1)

    def test_rrx_nocarry(self):
        if False:
            i = 10
            return i + 15
        val = 15
        (result, carry) = bitwise.RRX_C(val, 0, 32)
        self.assertEqual(result, 7)
        self.assertEqual(carry, 1)

    def test_rrx_carry(self):
        if False:
            print('Hello World!')
        val = 1
        (result, carry) = bitwise.RRX_C(val, 1, 32)
        print(hex(result))
        self.assertEqual(result, 2147483648)
        self.assertEqual(carry, 1)

    def test_sint(self):
        if False:
            for i in range(10):
                print('nop')
        val = 4294967295
        result = bitwise.SInt(val, 32)
        self.assertEqual(result, -1)

    def test_sint_2(self):
        if False:
            return 10
        val = 4294967294
        result = bitwise.SInt(val, 32)
        self.assertEqual(result, -2)