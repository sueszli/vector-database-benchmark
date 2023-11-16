import sys
import unittest
import numpy as np
(major, minor) = [int(d) for d in np.__version__.split('.')[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError
import Vector

class VectorTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        unittest.TestCase.__init__(self, methodName)
        self.typeStr = 'double'
        self.typeCode = 'd'

    def testLength(self):
        if False:
            while True:
                i = 10
        'Test length function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        length = Vector.__dict__[self.typeStr + 'Length']
        self.assertEqual(length([5, 12, 0]), 13)

    def testLengthBadList(self):
        if False:
            print('Hello World!')
        'Test length function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        length = Vector.__dict__[self.typeStr + 'Length']
        self.assertRaises(BadListError, length, [5, 'twelve', 0])

    def testLengthWrongSize(self):
        if False:
            for i in range(10):
                print('nop')
        'Test length function with wrong size'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        length = Vector.__dict__[self.typeStr + 'Length']
        self.assertRaises(TypeError, length, [5, 12])

    def testLengthWrongDim(self):
        if False:
            while True:
                i = 10
        'Test length function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        length = Vector.__dict__[self.typeStr + 'Length']
        self.assertRaises(TypeError, length, [[1, 2], [3, 4]])

    def testLengthNonContainer(self):
        if False:
            for i in range(10):
                print('nop')
        'Test length function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        length = Vector.__dict__[self.typeStr + 'Length']
        self.assertRaises(TypeError, length, None)

    def testProd(self):
        if False:
            i = 10
            return i + 15
        'Test prod function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        prod = Vector.__dict__[self.typeStr + 'Prod']
        self.assertEqual(prod([1, 2, 3, 4]), 24)

    def testProdBadList(self):
        if False:
            return 10
        'Test prod function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        prod = Vector.__dict__[self.typeStr + 'Prod']
        self.assertRaises(BadListError, prod, [[1, 'two'], ['e', 'pi']])

    def testProdWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test prod function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        prod = Vector.__dict__[self.typeStr + 'Prod']
        self.assertRaises(TypeError, prod, [[1, 2], [8, 9]])

    def testProdNonContainer(self):
        if False:
            i = 10
            return i + 15
        'Test prod function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        prod = Vector.__dict__[self.typeStr + 'Prod']
        self.assertRaises(TypeError, prod, None)

    def testSum(self):
        if False:
            i = 10
            return i + 15
        'Test sum function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        sum = Vector.__dict__[self.typeStr + 'Sum']
        self.assertEqual(sum([5, 6, 7, 8]), 26)

    def testSumBadList(self):
        if False:
            for i in range(10):
                print('nop')
        'Test sum function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        sum = Vector.__dict__[self.typeStr + 'Sum']
        self.assertRaises(BadListError, sum, [3, 4, 5, 'pi'])

    def testSumWrongDim(self):
        if False:
            return 10
        'Test sum function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        sum = Vector.__dict__[self.typeStr + 'Sum']
        self.assertRaises(TypeError, sum, [[3, 4], [5, 6]])

    def testSumNonContainer(self):
        if False:
            for i in range(10):
                print('nop')
        'Test sum function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        sum = Vector.__dict__[self.typeStr + 'Sum']
        self.assertRaises(TypeError, sum, True)

    def testReverse(self):
        if False:
            return 10
        'Test reverse function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        reverse = Vector.__dict__[self.typeStr + 'Reverse']
        vector = np.array([1, 2, 4], self.typeCode)
        reverse(vector)
        self.assertEqual((vector == [4, 2, 1]).all(), True)

    def testReverseWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test reverse function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        reverse = Vector.__dict__[self.typeStr + 'Reverse']
        vector = np.array([[1, 2], [3, 4]], self.typeCode)
        self.assertRaises(TypeError, reverse, vector)

    def testReverseWrongSize(self):
        if False:
            return 10
        'Test reverse function with wrong size'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        reverse = Vector.__dict__[self.typeStr + 'Reverse']
        vector = np.array([9, 8, 7, 6, 5, 4], self.typeCode)
        self.assertRaises(TypeError, reverse, vector)

    def testReverseWrongType(self):
        if False:
            i = 10
            return i + 15
        'Test reverse function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        reverse = Vector.__dict__[self.typeStr + 'Reverse']
        vector = np.array([1, 2, 4], 'c')
        self.assertRaises(TypeError, reverse, vector)

    def testReverseNonArray(self):
        if False:
            return 10
        'Test reverse function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        reverse = Vector.__dict__[self.typeStr + 'Reverse']
        self.assertRaises(TypeError, reverse, [2, 4, 6])

    def testOnes(self):
        if False:
            return 10
        'Test ones function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ones = Vector.__dict__[self.typeStr + 'Ones']
        vector = np.zeros(5, self.typeCode)
        ones(vector)
        np.testing.assert_array_equal(vector, np.array([1, 1, 1, 1, 1]))

    def testOnesWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ones function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ones = Vector.__dict__[self.typeStr + 'Ones']
        vector = np.zeros((5, 5), self.typeCode)
        self.assertRaises(TypeError, ones, vector)

    def testOnesWrongType(self):
        if False:
            i = 10
            return i + 15
        'Test ones function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ones = Vector.__dict__[self.typeStr + 'Ones']
        vector = np.zeros((5, 5), 'c')
        self.assertRaises(TypeError, ones, vector)

    def testOnesNonArray(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ones function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ones = Vector.__dict__[self.typeStr + 'Ones']
        self.assertRaises(TypeError, ones, [2, 4, 6, 8])

    def testZeros(self):
        if False:
            i = 10
            return i + 15
        'Test zeros function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        zeros = Vector.__dict__[self.typeStr + 'Zeros']
        vector = np.ones(5, self.typeCode)
        zeros(vector)
        np.testing.assert_array_equal(vector, np.array([0, 0, 0, 0, 0]))

    def testZerosWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test zeros function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        zeros = Vector.__dict__[self.typeStr + 'Zeros']
        vector = np.ones((5, 5), self.typeCode)
        self.assertRaises(TypeError, zeros, vector)

    def testZerosWrongType(self):
        if False:
            i = 10
            return i + 15
        'Test zeros function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        zeros = Vector.__dict__[self.typeStr + 'Zeros']
        vector = np.ones(6, 'c')
        self.assertRaises(TypeError, zeros, vector)

    def testZerosNonArray(self):
        if False:
            while True:
                i = 10
        'Test zeros function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        zeros = Vector.__dict__[self.typeStr + 'Zeros']
        self.assertRaises(TypeError, zeros, [1, 3, 5, 7, 9])

    def testEOSplit(self):
        if False:
            return 10
        'Test eoSplit function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        eoSplit = Vector.__dict__[self.typeStr + 'EOSplit']
        (even, odd) = eoSplit([1, 2, 3])
        self.assertEqual((even == [1, 0, 3]).all(), True)
        self.assertEqual((odd == [0, 2, 0]).all(), True)

    def testTwos(self):
        if False:
            i = 10
            return i + 15
        'Test twos function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        twos = Vector.__dict__[self.typeStr + 'Twos']
        vector = twos(5)
        self.assertEqual((vector == [2, 2, 2, 2, 2]).all(), True)

    def testTwosNonInt(self):
        if False:
            print('Hello World!')
        'Test twos function with non-integer dimension'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        twos = Vector.__dict__[self.typeStr + 'Twos']
        self.assertRaises(TypeError, twos, 5.0)

    def testThrees(self):
        if False:
            i = 10
            return i + 15
        'Test threes function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        threes = Vector.__dict__[self.typeStr + 'Threes']
        vector = threes(6)
        self.assertEqual((vector == [3, 3, 3, 3, 3, 3]).all(), True)

    def testThreesNonInt(self):
        if False:
            i = 10
            return i + 15
        'Test threes function with non-integer dimension'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        threes = Vector.__dict__[self.typeStr + 'Threes']
        self.assertRaises(TypeError, threes, 'threes')

class scharTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'schar'
        self.typeCode = 'b'

class ucharTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'uchar'
        self.typeCode = 'B'

class shortTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'short'
        self.typeCode = 'h'

class ushortTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'ushort'
        self.typeCode = 'H'

class intTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'int'
        self.typeCode = 'i'

class uintTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'uint'
        self.typeCode = 'I'

class longTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'long'
        self.typeCode = 'l'

class ulongTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'ulong'
        self.typeCode = 'L'

class longLongTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'longLong'
        self.typeCode = 'q'

class ulongLongTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'ulongLong'
        self.typeCode = 'Q'

class floatTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'float'
        self.typeCode = 'f'

class doubleTestCase(VectorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        VectorTestCase.__init__(self, methodName)
        self.typeStr = 'double'
        self.typeCode = 'd'
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(scharTestCase))
    suite.addTest(unittest.makeSuite(ucharTestCase))
    suite.addTest(unittest.makeSuite(shortTestCase))
    suite.addTest(unittest.makeSuite(ushortTestCase))
    suite.addTest(unittest.makeSuite(intTestCase))
    suite.addTest(unittest.makeSuite(uintTestCase))
    suite.addTest(unittest.makeSuite(longTestCase))
    suite.addTest(unittest.makeSuite(ulongTestCase))
    suite.addTest(unittest.makeSuite(longLongTestCase))
    suite.addTest(unittest.makeSuite(ulongLongTestCase))
    suite.addTest(unittest.makeSuite(floatTestCase))
    suite.addTest(unittest.makeSuite(doubleTestCase))
    print('Testing 1D Functions of Module Vector')
    print('NumPy version', np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))