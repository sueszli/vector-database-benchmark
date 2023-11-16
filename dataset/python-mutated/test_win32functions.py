"""Tests for win32functions.py"""
import unittest
import sys
import ctypes
sys.path.append('.')
from pywinauto.windows.win32structures import Structure
from pywinauto.windows.win32structures import POINT
from pywinauto.windows.win32structures import RECT
from pywinauto.windows.win32functions import MakeLong, HiWord, LoWord

class Win32FunctionsTestCases(unittest.TestCase):
    """Unit tests for the win32function methods"""

    def testMakeLong(self):
        if False:
            return 10
        data = ((0, (0, 0)), (1, (0, 1)), (65536, (1, 0)), (65535, (0, 65535)), (4294901760, (65535, 0)), (4294967295, (65535, 65535)), (0, (65536, 65536)))
        for (result, (hi, lo)) in data:
            self.assertEqual(result, MakeLong(hi, lo))

    def testMakeLong_zero(self):
        if False:
            i = 10
            return i + 15
        'test that makelong(0,0)'
        self.assertEqual(0, MakeLong(0, 0))

    def testMakeLong_lowone(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure MakeLong() function works with low word == 1'
        self.assertEqual(1, MakeLong(0, 1))

    def testMakeLong_highone(self):
        if False:
            print('Hello World!')
        'Make sure MakeLong() function works with high word == 1'
        self.assertEqual(65536, MakeLong(1, 0))

    def testMakeLong_highbig(self):
        if False:
            i = 10
            return i + 15
        'Make sure MakeLong() function works with big numder in high word'
        self.assertEqual(4294901760, MakeLong(65535, 0))

    def testMakeLong_lowbig(self):
        if False:
            i = 10
            return i + 15
        'Make sure MakeLong() function works with big numder in low word'
        self.assertEqual(65535, MakeLong(0, 65535))

    def testMakeLong_big(self):
        if False:
            return 10
        'Make sure MakeLong() function works with big numders in 2 words'
        self.assertEqual(4294967295, MakeLong(65535, 65535))

    def testLowWord_zero(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, LoWord(0))

    def testLowWord_one(self):
        if False:
            print('Hello World!')
        self.assertEqual(1, LoWord(1))

    def testLowWord_big(self):
        if False:
            return 10
        self.assertEqual(1, LoWord(MakeLong(65535, 1)))

    def testLowWord_vbig(self):
        if False:
            print('Hello World!')
        self.assertEqual(65535, LoWord(MakeLong(65535, 65535)))

    def testHiWord_zero(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, HiWord(0))

    def testHiWord_one(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(0, HiWord(1))

    def testHiWord_bigone(self):
        if False:
            while True:
                i = 10
        self.assertEqual(1, HiWord(65536))

    def testHiWord_big(self):
        if False:
            print('Hello World!')
        self.assertEqual(65535, HiWord(MakeLong(65535, 1)))

    def testHiWord_vbig(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(65535, HiWord(MakeLong(65535, 65535)))

    def testPOINTindexation(self):
        if False:
            return 10
        p = POINT(1, 2)
        self.assertEqual(p[0], p.x)
        self.assertEqual(p[1], p.y)
        self.assertEqual(p[-2], p.x)
        self.assertEqual(p[-1], p.y)
        self.assertRaises(IndexError, lambda : p[2])
        self.assertRaises(IndexError, lambda : p[-3])

    def testPOINTiteration(self):
        if False:
            print('Hello World!')
        p = POINT(1, 2)
        self.assertEqual([1, 2], [i for i in p])

    def testPOINTcomparision(self):
        if False:
            print('Hello World!')
        'Test POINT comparision operations'
        p0 = POINT(1, 2)
        p1 = POINT(0, 2)
        self.assertNotEqual(p0, p1)
        p1.x = p0.x
        self.assertEqual(p0, p1)
        self.assertEqual(p0, (1, 2))
        self.assertNotEqual(p0, (0, 2))
        self.assertNotEqual(p0, 1)

    def test_RECT_hash(self):
        if False:
            print('Hello World!')
        'Test RECT is not hashable'
        self.assertRaises(TypeError, hash, RECT())

    def test_RECT_eq(self):
        if False:
            print('Hello World!')
        r0 = RECT(1, 2, 3, 4)
        self.assertEqual(r0, RECT(1, 2, 3, 4))
        self.assertEqual(r0, [1, 2, 3, 4])
        self.assertNotEqual(r0, RECT(1, 2, 3, 5))
        self.assertNotEqual(r0, [1, 2, 3, 5])
        self.assertNotEqual(r0, [1, 2, 3])
        self.assertNotEqual(r0, [1, 2, 3, 4, 5])
        r0.bottom = 5
        self.assertEqual(r0, RECT(1, 2, 3, 5))
        self.assertEqual(r0, (1, 2, 3, 5))

    def test_RECT_repr(self):
        if False:
            while True:
                i = 10
        'Test RECT repr'
        r0 = RECT(0)
        self.assertEqual(r0.__repr__(), '<RECT L0, T0, R0, B0>')

    def test_RECT_iter(self):
        if False:
            return 10
        'Test RECT is iterable'
        r = RECT(1, 2, 3, 4)
        (left, top, right, bottom) = r
        self.assertEqual(left, r.left)
        self.assertEqual(right, r.right)
        self.assertEqual(top, r.top)
        self.assertEqual(bottom, r.bottom)

    def test_Structure(self):
        if False:
            return 10

        class Structure0(Structure):
            _fields_ = [('f0', ctypes.c_int)]

        class Structure1(Structure):
            _fields_ = [('f1', ctypes.c_int)]
        s0 = Structure0(0)
        self.assertEqual(str(s0), '%20s\t%s' % ('f0', s0.f0))
        s1 = Structure1(0)
        self.assertNotEqual(s0, s1)
        s0._fields_.append(('f1', ctypes.c_int))
        self.assertNotEqual(s0, [0, 1])
if __name__ == '__main__':
    unittest.main()