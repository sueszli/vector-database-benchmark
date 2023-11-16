import sys
import unittest
import numpy as np
(major, minor) = [int(d) for d in np.__version__.split('.')[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError
import Matrix

class MatrixTestCase(unittest.TestCase):

    def __init__(self, methodName='runTests'):
        if False:
            print('Hello World!')
        unittest.TestCase.__init__(self, methodName)
        self.typeStr = 'double'
        self.typeCode = 'd'

    def testDet(self):
        if False:
            i = 10
            return i + 15
        'Test det function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        det = Matrix.__dict__[self.typeStr + 'Det']
        matrix = [[8, 7], [6, 9]]
        self.assertEqual(det(matrix), 30)

    def testDetBadList(self):
        if False:
            for i in range(10):
                print('nop')
        'Test det function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        det = Matrix.__dict__[self.typeStr + 'Det']
        matrix = [[8, 7], ['e', 'pi']]
        self.assertRaises(BadListError, det, matrix)

    def testDetWrongDim(self):
        if False:
            print('Hello World!')
        'Test det function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        det = Matrix.__dict__[self.typeStr + 'Det']
        matrix = [8, 7]
        self.assertRaises(TypeError, det, matrix)

    def testDetWrongSize(self):
        if False:
            i = 10
            return i + 15
        'Test det function with wrong size'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        det = Matrix.__dict__[self.typeStr + 'Det']
        matrix = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
        self.assertRaises(TypeError, det, matrix)

    def testDetNonContainer(self):
        if False:
            print('Hello World!')
        'Test det function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        det = Matrix.__dict__[self.typeStr + 'Det']
        self.assertRaises(TypeError, det, None)

    def testMax(self):
        if False:
            while True:
                i = 10
        'Test max function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Matrix.__dict__[self.typeStr + 'Max']
        matrix = [[6, 5, 4], [3, 2, 1]]
        self.assertEqual(max(matrix), 6)

    def testMaxBadList(self):
        if False:
            return 10
        'Test max function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Matrix.__dict__[self.typeStr + 'Max']
        matrix = [[6, 'five', 4], ['three', 2, 'one']]
        self.assertRaises(BadListError, max, matrix)

    def testMaxNonContainer(self):
        if False:
            while True:
                i = 10
        'Test max function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Matrix.__dict__[self.typeStr + 'Max']
        self.assertRaises(TypeError, max, None)

    def testMaxWrongDim(self):
        if False:
            return 10
        'Test max function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Matrix.__dict__[self.typeStr + 'Max']
        self.assertRaises(TypeError, max, [0, 1, 2, 3])

    def testMin(self):
        if False:
            while True:
                i = 10
        'Test min function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Matrix.__dict__[self.typeStr + 'Min']
        matrix = [[9, 8], [7, 6], [5, 4]]
        self.assertEqual(min(matrix), 4)

    def testMinBadList(self):
        if False:
            while True:
                i = 10
        'Test min function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Matrix.__dict__[self.typeStr + 'Min']
        matrix = [['nine', 'eight'], ['seven', 'six']]
        self.assertRaises(BadListError, min, matrix)

    def testMinWrongDim(self):
        if False:
            i = 10
            return i + 15
        'Test min function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Matrix.__dict__[self.typeStr + 'Min']
        self.assertRaises(TypeError, min, [1, 3, 5, 7, 9])

    def testMinNonContainer(self):
        if False:
            print('Hello World!')
        'Test min function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Matrix.__dict__[self.typeStr + 'Min']
        self.assertRaises(TypeError, min, False)

    def testScale(self):
        if False:
            print('Hello World!')
        'Test scale function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Matrix.__dict__[self.typeStr + 'Scale']
        matrix = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]], self.typeCode)
        scale(matrix, 4)
        self.assertEqual((matrix == [[4, 8, 12], [8, 4, 8], [12, 8, 4]]).all(), True)

    def testScaleWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test scale function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Matrix.__dict__[self.typeStr + 'Scale']
        matrix = np.array([1, 2, 2, 1], self.typeCode)
        self.assertRaises(TypeError, scale, matrix)

    def testScaleWrongSize(self):
        if False:
            print('Hello World!')
        'Test scale function with wrong size'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Matrix.__dict__[self.typeStr + 'Scale']
        matrix = np.array([[1, 2], [2, 1]], self.typeCode)
        self.assertRaises(TypeError, scale, matrix)

    def testScaleWrongType(self):
        if False:
            for i in range(10):
                print('nop')
        'Test scale function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Matrix.__dict__[self.typeStr + 'Scale']
        matrix = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]], 'c')
        self.assertRaises(TypeError, scale, matrix)

    def testScaleNonArray(self):
        if False:
            for i in range(10):
                print('nop')
        'Test scale function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Matrix.__dict__[self.typeStr + 'Scale']
        matrix = [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
        self.assertRaises(TypeError, scale, matrix)

    def testFloor(self):
        if False:
            print('Hello World!')
        'Test floor function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Matrix.__dict__[self.typeStr + 'Floor']
        matrix = np.array([[6, 7], [8, 9]], self.typeCode)
        floor(matrix, 7)
        np.testing.assert_array_equal(matrix, np.array([[7, 7], [8, 9]]))

    def testFloorWrongDim(self):
        if False:
            i = 10
            return i + 15
        'Test floor function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Matrix.__dict__[self.typeStr + 'Floor']
        matrix = np.array([6, 7, 8, 9], self.typeCode)
        self.assertRaises(TypeError, floor, matrix)

    def testFloorWrongType(self):
        if False:
            i = 10
            return i + 15
        'Test floor function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Matrix.__dict__[self.typeStr + 'Floor']
        matrix = np.array([[6, 7], [8, 9]], 'c')
        self.assertRaises(TypeError, floor, matrix)

    def testFloorNonArray(self):
        if False:
            for i in range(10):
                print('nop')
        'Test floor function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Matrix.__dict__[self.typeStr + 'Floor']
        matrix = [[6, 7], [8, 9]]
        self.assertRaises(TypeError, floor, matrix)

    def testCeil(self):
        if False:
            return 10
        'Test ceil function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Matrix.__dict__[self.typeStr + 'Ceil']
        matrix = np.array([[1, 2], [3, 4]], self.typeCode)
        ceil(matrix, 3)
        np.testing.assert_array_equal(matrix, np.array([[1, 2], [3, 3]]))

    def testCeilWrongDim(self):
        if False:
            print('Hello World!')
        'Test ceil function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Matrix.__dict__[self.typeStr + 'Ceil']
        matrix = np.array([1, 2, 3, 4], self.typeCode)
        self.assertRaises(TypeError, ceil, matrix)

    def testCeilWrongType(self):
        if False:
            while True:
                i = 10
        'Test ceil function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Matrix.__dict__[self.typeStr + 'Ceil']
        matrix = np.array([[1, 2], [3, 4]], 'c')
        self.assertRaises(TypeError, ceil, matrix)

    def testCeilNonArray(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ceil function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Matrix.__dict__[self.typeStr + 'Ceil']
        matrix = [[1, 2], [3, 4]]
        self.assertRaises(TypeError, ceil, matrix)

    def testLUSplit(self):
        if False:
            for i in range(10):
                print('nop')
        'Test luSplit function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        luSplit = Matrix.__dict__[self.typeStr + 'LUSplit']
        (lower, upper) = luSplit([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual((lower == [[1, 0, 0], [4, 5, 0], [7, 8, 9]]).all(), True)
        self.assertEqual((upper == [[0, 2, 3], [0, 0, 6], [0, 0, 0]]).all(), True)

class scharTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'schar'
        self.typeCode = 'b'

class ucharTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'uchar'
        self.typeCode = 'B'

class shortTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'short'
        self.typeCode = 'h'

class ushortTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'ushort'
        self.typeCode = 'H'

class intTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'int'
        self.typeCode = 'i'

class uintTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'uint'
        self.typeCode = 'I'

class longTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'long'
        self.typeCode = 'l'

class ulongTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'ulong'
        self.typeCode = 'L'

class longLongTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'longLong'
        self.typeCode = 'q'

class ulongLongTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'ulongLong'
        self.typeCode = 'Q'

class floatTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        MatrixTestCase.__init__(self, methodName)
        self.typeStr = 'float'
        self.typeCode = 'f'

class doubleTestCase(MatrixTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        MatrixTestCase.__init__(self, methodName)
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
    print('Testing 2D Functions of Module Matrix')
    print('NumPy version', np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))