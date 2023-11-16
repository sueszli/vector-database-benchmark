import sys
import unittest
import numpy as np
(major, minor) = [int(d) for d in np.__version__.split('.')[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError
import Array

class Array1TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.length = 5
        self.array1 = Array.Array1(self.length)

    def testConstructor0(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array1 default constructor'
        a = Array.Array1()
        self.assertTrue(isinstance(a, Array.Array1))
        self.assertTrue(len(a) == 0)

    def testConstructor1(self):
        if False:
            while True:
                i = 10
        'Test Array1 length constructor'
        self.assertTrue(isinstance(self.array1, Array.Array1))

    def testConstructor2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array1 array constructor'
        na = np.arange(self.length)
        aa = Array.Array1(na)
        self.assertTrue(isinstance(aa, Array.Array1))

    def testConstructor3(self):
        if False:
            return 10
        'Test Array1 copy constructor'
        for i in range(self.array1.length()):
            self.array1[i] = i
        arrayCopy = Array.Array1(self.array1)
        self.assertTrue(arrayCopy == self.array1)

    def testConstructorBad(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array1 length constructor, negative'
        self.assertRaises(ValueError, Array.Array1, -4)

    def testLength(self):
        if False:
            print('Hello World!')
        'Test Array1 length method'
        self.assertTrue(self.array1.length() == self.length)

    def testLen(self):
        if False:
            print('Hello World!')
        'Test Array1 __len__ method'
        self.assertTrue(len(self.array1) == self.length)

    def testResize0(self):
        if False:
            while True:
                i = 10
        'Test Array1 resize method, length'
        newLen = 2 * self.length
        self.array1.resize(newLen)
        self.assertTrue(len(self.array1) == newLen)

    def testResize1(self):
        if False:
            while True:
                i = 10
        'Test Array1 resize method, array'
        a = np.zeros((2 * self.length,), dtype='l')
        self.array1.resize(a)
        self.assertTrue(len(self.array1) == a.size)

    def testResizeBad(self):
        if False:
            while True:
                i = 10
        'Test Array1 resize method, negative length'
        self.assertRaises(ValueError, self.array1.resize, -5)

    def testSetGet(self):
        if False:
            while True:
                i = 10
        'Test Array1 __setitem__, __getitem__ methods'
        n = self.length
        for i in range(n):
            self.array1[i] = i * i
        for i in range(n):
            self.assertTrue(self.array1[i] == i * i)

    def testSetBad1(self):
        if False:
            while True:
                i = 10
        'Test Array1 __setitem__ method, negative index'
        self.assertRaises(IndexError, self.array1.__setitem__, -1, 0)

    def testSetBad2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array1 __setitem__ method, out-of-range index'
        self.assertRaises(IndexError, self.array1.__setitem__, self.length + 1, 0)

    def testGetBad1(self):
        if False:
            print('Hello World!')
        'Test Array1 __getitem__ method, negative index'
        self.assertRaises(IndexError, self.array1.__getitem__, -1)

    def testGetBad2(self):
        if False:
            return 10
        'Test Array1 __getitem__ method, out-of-range index'
        self.assertRaises(IndexError, self.array1.__getitem__, self.length + 1)

    def testAsString(self):
        if False:
            return 10
        'Test Array1 asString method'
        for i in range(self.array1.length()):
            self.array1[i] = i + 1
        self.assertTrue(self.array1.asString() == '[ 1, 2, 3, 4, 5 ]')

    def testStr(self):
        if False:
            while True:
                i = 10
        'Test Array1 __str__ method'
        for i in range(self.array1.length()):
            self.array1[i] = i - 2
        self.assertTrue(str(self.array1) == '[ -2, -1, 0, 1, 2 ]')

    def testView(self):
        if False:
            return 10
        'Test Array1 view method'
        for i in range(self.array1.length()):
            self.array1[i] = i + 1
        a = self.array1.view()
        self.assertTrue(isinstance(a, np.ndarray))
        self.assertTrue(len(a) == self.length)
        self.assertTrue((a == [1, 2, 3, 4, 5]).all())

class Array2TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.nrows = 5
        self.ncols = 4
        self.array2 = Array.Array2(self.nrows, self.ncols)

    def testConstructor0(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array2 default constructor'
        a = Array.Array2()
        self.assertTrue(isinstance(a, Array.Array2))
        self.assertTrue(len(a) == 0)

    def testConstructor1(self):
        if False:
            print('Hello World!')
        'Test Array2 nrows, ncols constructor'
        self.assertTrue(isinstance(self.array2, Array.Array2))

    def testConstructor2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array2 array constructor'
        na = np.zeros((3, 4), dtype='l')
        aa = Array.Array2(na)
        self.assertTrue(isinstance(aa, Array.Array2))

    def testConstructor3(self):
        if False:
            while True:
                i = 10
        'Test Array2 copy constructor'
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array2[i][j] = i * j
        arrayCopy = Array.Array2(self.array2)
        self.assertTrue(arrayCopy == self.array2)

    def testConstructorBad1(self):
        if False:
            while True:
                i = 10
        'Test Array2 nrows, ncols constructor, negative nrows'
        self.assertRaises(ValueError, Array.Array2, -4, 4)

    def testConstructorBad2(self):
        if False:
            print('Hello World!')
        'Test Array2 nrows, ncols constructor, negative ncols'
        self.assertRaises(ValueError, Array.Array2, 4, -4)

    def testNrows(self):
        if False:
            i = 10
            return i + 15
        'Test Array2 nrows method'
        self.assertTrue(self.array2.nrows() == self.nrows)

    def testNcols(self):
        if False:
            i = 10
            return i + 15
        'Test Array2 ncols method'
        self.assertTrue(self.array2.ncols() == self.ncols)

    def testLen(self):
        if False:
            i = 10
            return i + 15
        'Test Array2 __len__ method'
        self.assertTrue(len(self.array2) == self.nrows * self.ncols)

    def testResize0(self):
        if False:
            i = 10
            return i + 15
        'Test Array2 resize method, size'
        newRows = 2 * self.nrows
        newCols = 2 * self.ncols
        self.array2.resize(newRows, newCols)
        self.assertTrue(len(self.array2) == newRows * newCols)

    def testResize1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array2 resize method, array'
        a = np.zeros((2 * self.nrows, 2 * self.ncols), dtype='l')
        self.array2.resize(a)
        self.assertTrue(len(self.array2) == a.size)

    def testResizeBad1(self):
        if False:
            return 10
        'Test Array2 resize method, negative nrows'
        self.assertRaises(ValueError, self.array2.resize, -5, 5)

    def testResizeBad2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array2 resize method, negative ncols'
        self.assertRaises(ValueError, self.array2.resize, 5, -5)

    def testSetGet1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array2 __setitem__, __getitem__ methods'
        m = self.nrows
        n = self.ncols
        array1 = []
        a = np.arange(n, dtype='l')
        for i in range(m):
            array1.append(Array.Array1(i * a))
        for i in range(m):
            self.array2[i] = array1[i]
        for i in range(m):
            self.assertTrue(self.array2[i] == array1[i])

    def testSetGet2(self):
        if False:
            print('Hello World!')
        'Test Array2 chained __setitem__, __getitem__ methods'
        m = self.nrows
        n = self.ncols
        for i in range(m):
            for j in range(n):
                self.array2[i][j] = i * j
        for i in range(m):
            for j in range(n):
                self.assertTrue(self.array2[i][j] == i * j)

    def testSetBad1(self):
        if False:
            print('Hello World!')
        'Test Array2 __setitem__ method, negative index'
        a = Array.Array1(self.ncols)
        self.assertRaises(IndexError, self.array2.__setitem__, -1, a)

    def testSetBad2(self):
        if False:
            print('Hello World!')
        'Test Array2 __setitem__ method, out-of-range index'
        a = Array.Array1(self.ncols)
        self.assertRaises(IndexError, self.array2.__setitem__, self.nrows + 1, a)

    def testGetBad1(self):
        if False:
            print('Hello World!')
        'Test Array2 __getitem__ method, negative index'
        self.assertRaises(IndexError, self.array2.__getitem__, -1)

    def testGetBad2(self):
        if False:
            i = 10
            return i + 15
        'Test Array2 __getitem__ method, out-of-range index'
        self.assertRaises(IndexError, self.array2.__getitem__, self.nrows + 1)

    def testAsString(self):
        if False:
            while True:
                i = 10
        'Test Array2 asString method'
        result = '[ [ 0, 1, 2, 3 ],\n  [ 1, 2, 3, 4 ],\n  [ 2, 3, 4, 5 ],\n  [ 3, 4, 5, 6 ],\n  [ 4, 5, 6, 7 ] ]\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array2[i][j] = i + j
        self.assertTrue(self.array2.asString() == result)

    def testStr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array2 __str__ method'
        result = '[ [ 0, -1, -2, -3 ],\n  [ 1, 0, -1, -2 ],\n  [ 2, 1, 0, -1 ],\n  [ 3, 2, 1, 0 ],\n  [ 4, 3, 2, 1 ] ]\n'
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.array2[i][j] = i - j
        self.assertTrue(str(self.array2) == result)

    def testView(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Array2 view method'
        a = self.array2.view()
        self.assertTrue(isinstance(a, np.ndarray))
        self.assertTrue(len(a) == self.nrows)

class ArrayZTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.length = 5
        self.array3 = Array.ArrayZ(self.length)

    def testConstructor0(self):
        if False:
            print('Hello World!')
        'Test ArrayZ default constructor'
        a = Array.ArrayZ()
        self.assertTrue(isinstance(a, Array.ArrayZ))
        self.assertTrue(len(a) == 0)

    def testConstructor1(self):
        if False:
            print('Hello World!')
        'Test ArrayZ length constructor'
        self.assertTrue(isinstance(self.array3, Array.ArrayZ))

    def testConstructor2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ArrayZ array constructor'
        na = np.arange(self.length, dtype=np.complex128)
        aa = Array.ArrayZ(na)
        self.assertTrue(isinstance(aa, Array.ArrayZ))

    def testConstructor3(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ArrayZ copy constructor'
        for i in range(self.array3.length()):
            self.array3[i] = complex(i, -i)
        arrayCopy = Array.ArrayZ(self.array3)
        self.assertTrue(arrayCopy == self.array3)

    def testConstructorBad(self):
        if False:
            return 10
        'Test ArrayZ length constructor, negative'
        self.assertRaises(ValueError, Array.ArrayZ, -4)

    def testLength(self):
        if False:
            return 10
        'Test ArrayZ length method'
        self.assertTrue(self.array3.length() == self.length)

    def testLen(self):
        if False:
            print('Hello World!')
        'Test ArrayZ __len__ method'
        self.assertTrue(len(self.array3) == self.length)

    def testResize0(self):
        if False:
            return 10
        'Test ArrayZ resize method, length'
        newLen = 2 * self.length
        self.array3.resize(newLen)
        self.assertTrue(len(self.array3) == newLen)

    def testResize1(self):
        if False:
            i = 10
            return i + 15
        'Test ArrayZ resize method, array'
        a = np.zeros((2 * self.length,), dtype=np.complex128)
        self.array3.resize(a)
        self.assertTrue(len(self.array3) == a.size)

    def testResizeBad(self):
        if False:
            print('Hello World!')
        'Test ArrayZ resize method, negative length'
        self.assertRaises(ValueError, self.array3.resize, -5)

    def testSetGet(self):
        if False:
            i = 10
            return i + 15
        'Test ArrayZ __setitem__, __getitem__ methods'
        n = self.length
        for i in range(n):
            self.array3[i] = i * i
        for i in range(n):
            self.assertTrue(self.array3[i] == i * i)

    def testSetBad1(self):
        if False:
            while True:
                i = 10
        'Test ArrayZ __setitem__ method, negative index'
        self.assertRaises(IndexError, self.array3.__setitem__, -1, 0)

    def testSetBad2(self):
        if False:
            return 10
        'Test ArrayZ __setitem__ method, out-of-range index'
        self.assertRaises(IndexError, self.array3.__setitem__, self.length + 1, 0)

    def testGetBad1(self):
        if False:
            while True:
                i = 10
        'Test ArrayZ __getitem__ method, negative index'
        self.assertRaises(IndexError, self.array3.__getitem__, -1)

    def testGetBad2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ArrayZ __getitem__ method, out-of-range index'
        self.assertRaises(IndexError, self.array3.__getitem__, self.length + 1)

    def testAsString(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ArrayZ asString method'
        for i in range(self.array3.length()):
            self.array3[i] = complex(i + 1, -i - 1)
        self.assertTrue(self.array3.asString() == '[ (1,-1), (2,-2), (3,-3), (4,-4), (5,-5) ]')

    def testStr(self):
        if False:
            i = 10
            return i + 15
        'Test ArrayZ __str__ method'
        for i in range(self.array3.length()):
            self.array3[i] = complex(i - 2, (i - 2) * 2)
        self.assertTrue(str(self.array3) == '[ (-2,-4), (-1,-2), (0,0), (1,2), (2,4) ]')

    def testView(self):
        if False:
            i = 10
            return i + 15
        'Test ArrayZ view method'
        for i in range(self.array3.length()):
            self.array3[i] = complex(i + 1, i + 2)
        a = self.array3.view()
        self.assertTrue(isinstance(a, np.ndarray))
        self.assertTrue(len(a) == self.length)
        self.assertTrue((a == [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j]).all())
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Array1TestCase))
    suite.addTest(unittest.makeSuite(Array2TestCase))
    suite.addTest(unittest.makeSuite(ArrayZTestCase))
    print('Testing Classes of Module Array')
    print('NumPy version', np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))