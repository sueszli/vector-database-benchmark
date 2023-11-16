from distutils.util import get_platform
import os
import sys
import unittest
import numpy as np
(major, minor) = [int(d) for d in np.__version__.split('.')[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError
libDir = 'lib.{}-{}.{}'.format(get_platform(), *sys.version_info[:2])
sys.path.insert(0, os.path.join('build', libDir))
import Farray

class FarrayTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.nrows = 5
        self.ncols = 4
        self.array = Farray.Farray(self.nrows, self.ncols)

    def testConstructor1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Farray size constructor'
        self.assertTrue(isinstance(self.array, Farray.Farray))

    def testConstructor2(self):
        if False:
            return 10
        'Test Farray copy constructor'
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i + j
        arrayCopy = Farray.Farray(self.array)
        self.assertTrue(arrayCopy == self.array)

    def testConstructorBad1(self):
        if False:
            i = 10
            return i + 15
        'Test Farray size constructor, negative nrows'
        self.assertRaises(ValueError, Farray.Farray, -4, 4)

    def testConstructorBad2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Farray size constructor, negative ncols'
        self.assertRaises(ValueError, Farray.Farray, 4, -4)

    def testNrows(self):
        if False:
            while True:
                i = 10
        'Test Farray nrows method'
        self.assertTrue(self.array.nrows() == self.nrows)

    def testNcols(self):
        if False:
            while True:
                i = 10
        'Test Farray ncols method'
        self.assertTrue(self.array.ncols() == self.ncols)

    def testLen(self):
        if False:
            i = 10
            return i + 15
        'Test Farray __len__ method'
        self.assertTrue(len(self.array) == self.nrows * self.ncols)

    def testSetGet(self):
        if False:
            i = 10
            return i + 15
        'Test Farray __setitem__, __getitem__ methods'
        m = self.nrows
        n = self.ncols
        for i in range(m):
            for j in range(n):
                self.array[i, j] = i * j
        for i in range(m):
            for j in range(n):
                self.assertTrue(self.array[i, j] == i * j)

    def testSetBad1(self):
        if False:
            i = 10
            return i + 15
        'Test Farray __setitem__ method, negative row'
        self.assertRaises(IndexError, self.array.__setitem__, (-1, 3), 0)

    def testSetBad2(self):
        if False:
            return 10
        'Test Farray __setitem__ method, negative col'
        self.assertRaises(IndexError, self.array.__setitem__, (1, -3), 0)

    def testSetBad3(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Farray __setitem__ method, out-of-range row'
        self.assertRaises(IndexError, self.array.__setitem__, (self.nrows + 1, 0), 0)

    def testSetBad4(self):
        if False:
            i = 10
            return i + 15
        'Test Farray __setitem__ method, out-of-range col'
        self.assertRaises(IndexError, self.array.__setitem__, (0, self.ncols + 1), 0)

    def testGetBad1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Farray __getitem__ method, negative row'
        self.assertRaises(IndexError, self.array.__getitem__, (-1, 3))

    def testGetBad2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Farray __getitem__ method, negative col'
        self.assertRaises(IndexError, self.array.__getitem__, (1, -3))

    def testGetBad3(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Farray __getitem__ method, out-of-range row'
        self.assertRaises(IndexError, self.array.__getitem__, (self.nrows + 1, 0))

    def testGetBad4(self):
        if False:
            return 10
        'Test Farray __getitem__ method, out-of-range col'
        self.assertRaises(IndexError, self.array.__getitem__, (0, self.ncols + 1))

    def testAsString(self):
        if False:
            while True:
                i = 10
        'Test Farray asString method'
        result = '[ [ 0, 1, 2, 3 ],\n  [ 1, 2, 3, 4 ],\n  [ 2, 3, 4, 5 ],\n  [ 3, 4, 5, 6 ],\n  [ 4, 5, 6, 7 ] ]\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i + j
        self.assertTrue(self.array.asString() == result)

    def testStr(self):
        if False:
            return 10
        'Test Farray __str__ method'
        result = '[ [ 0, -1, -2, -3 ],\n  [ 1, 0, -1, -2 ],\n  [ 2, 1, 0, -1 ],\n  [ 3, 2, 1, 0 ],\n  [ 4, 3, 2, 1 ] ]\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i - j
        self.assertTrue(str(self.array) == result)

    def testView(self):
        if False:
            print('Hello World!')
        'Test Farray view method'
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array[i, j] = i + j
        a = self.array.view()
        self.assertTrue(isinstance(a, np.ndarray))
        self.assertTrue(a.flags.f_contiguous)
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.assertTrue(a[i, j] == i + j)
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FarrayTestCase))
    print('Testing Classes of Module Farray')
    print('NumPy version', np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))