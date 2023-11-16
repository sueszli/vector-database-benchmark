import unittest
from manticore.core.smtlib import *
from manticore.native.cpu.register import Register

class RegisterTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_rd(self):
        if False:
            for i in range(10):
                print('nop')
        r = Register(32)
        self.assertEqual(r.read(), 0)

    def test_basic_write(self):
        if False:
            i = 10
            return i + 15
        r = Register(32)
        r.write(1)
        self.assertEqual(r.read(), 1)

    def test_truncate(self):
        if False:
            while True:
                i = 10
        r = Register(32)
        r.write(2 ** 32)
        self.assertEqual(r.read(), 0)

    def test_largest_write(self):
        if False:
            for i in range(10):
                print('nop')
        r = Register(32)
        r.write(4294967295)
        self.assertEqual(r.read(), 4294967295)

    def test_flag(self):
        if False:
            i = 10
            return i + 15
        r = Register(1)
        self.assertEqual(r.read(), False)

    def test_flag_write(self):
        if False:
            return 10
        r = Register(1)
        r.write(True)
        self.assertEqual(r.read(), True)

    def test_flag_trunc(self):
        if False:
            while True:
                i = 10
        r = Register(1)
        r.write(3)
        self.assertEqual(r.read(), True)

    def test_bool_write_nonflag(self):
        if False:
            while True:
                i = 10
        r = Register(32)
        r.write(True)
        self.assertEqual(r.read(), True)

    def test_Bool(self):
        if False:
            for i in range(10):
                print('nop')
        r = Register(32)
        b = BoolVariable(name='B')
        r.write(b)
        self.assertIs(r.read(), b)

    def test_bitvec_flag(self):
        if False:
            for i in range(10):
                print('nop')
        r = Register(1)
        b = BitVecConstant(size=32, value=0)
        r.write(b)
        self.assertTrue(isinstance(r.read(), Bool))

    def test_bitvec(self):
        if False:
            i = 10
            return i + 15
        r = Register(32)
        b = BitVecConstant(size=32, value=0)
        r.write(b)
        self.assertIs(r.read(), b)

    def test_bad_write(self):
        if False:
            i = 10
            return i + 15
        r = Register(32)
        with self.assertRaises(TypeError):
            r.write(dict())