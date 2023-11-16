from math import sqrt
import sys
import unittest
import numpy as np
(major, minor) = [int(d) for d in np.__version__.split('.')[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError
import Tensor

class TensorTestCase(unittest.TestCase):

    def __init__(self, methodName='runTests'):
        if False:
            print('Hello World!')
        unittest.TestCase.__init__(self, methodName)
        self.typeStr = 'double'
        self.typeCode = 'd'
        self.result = sqrt(28.0 / 8)

    def testNorm(self):
        if False:
            for i in range(10):
                print('nop')
        'Test norm function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + 'Norm']
        tensor = [[[0, 1], [2, 3]], [[3, 2], [1, 0]]]
        if isinstance(self.result, int):
            self.assertEqual(norm(tensor), self.result)
        else:
            self.assertAlmostEqual(norm(tensor), self.result, 6)

    def testNormBadList(self):
        if False:
            while True:
                i = 10
        'Test norm function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + 'Norm']
        tensor = [[[0, 'one'], [2, 3]], [[3, 'two'], [1, 0]]]
        self.assertRaises(BadListError, norm, tensor)

    def testNormWrongDim(self):
        if False:
            while True:
                i = 10
        'Test norm function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + 'Norm']
        tensor = [[0, 1, 2, 3], [3, 2, 1, 0]]
        self.assertRaises(TypeError, norm, tensor)

    def testNormWrongSize(self):
        if False:
            i = 10
            return i + 15
        'Test norm function with wrong size'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + 'Norm']
        tensor = [[[0, 1, 0], [2, 3, 2]], [[3, 2, 3], [1, 0, 1]]]
        self.assertRaises(TypeError, norm, tensor)

    def testNormNonContainer(self):
        if False:
            for i in range(10):
                print('nop')
        'Test norm function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        norm = Tensor.__dict__[self.typeStr + 'Norm']
        self.assertRaises(TypeError, norm, None)

    def testMax(self):
        if False:
            i = 10
            return i + 15
        'Test max function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + 'Max']
        tensor = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        self.assertEqual(max(tensor), 8)

    def testMaxBadList(self):
        if False:
            print('Hello World!')
        'Test max function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + 'Max']
        tensor = [[[1, 'two'], [3, 4]], [[5, 'six'], [7, 8]]]
        self.assertRaises(BadListError, max, tensor)

    def testMaxNonContainer(self):
        if False:
            print('Hello World!')
        'Test max function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + 'Max']
        self.assertRaises(TypeError, max, None)

    def testMaxWrongDim(self):
        if False:
            print('Hello World!')
        'Test max function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        max = Tensor.__dict__[self.typeStr + 'Max']
        self.assertRaises(TypeError, max, [0, -1, 2, -3])

    def testMin(self):
        if False:
            while True:
                i = 10
        'Test min function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + 'Min']
        tensor = [[[9, 8], [7, 6]], [[5, 4], [3, 2]]]
        self.assertEqual(min(tensor), 2)

    def testMinBadList(self):
        if False:
            for i in range(10):
                print('nop')
        'Test min function with bad list'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + 'Min']
        tensor = [[['nine', 8], [7, 6]], [['five', 4], [3, 2]]]
        self.assertRaises(BadListError, min, tensor)

    def testMinNonContainer(self):
        if False:
            print('Hello World!')
        'Test min function with non-container'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + 'Min']
        self.assertRaises(TypeError, min, True)

    def testMinWrongDim(self):
        if False:
            return 10
        'Test min function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        min = Tensor.__dict__[self.typeStr + 'Min']
        self.assertRaises(TypeError, min, [[1, 3], [5, 7]])

    def testScale(self):
        if False:
            return 10
        'Test scale function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + 'Scale']
        tensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], self.typeCode)
        scale(tensor, 4)
        self.assertEqual((tensor == [[[4, 0, 4], [0, 4, 0], [4, 0, 4]], [[0, 4, 0], [4, 0, 4], [0, 4, 0]], [[4, 0, 4], [0, 4, 0], [4, 0, 4]]]).all(), True)

    def testScaleWrongType(self):
        if False:
            for i in range(10):
                print('nop')
        'Test scale function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + 'Scale']
        tensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], 'c')
        self.assertRaises(TypeError, scale, tensor)

    def testScaleWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test scale function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + 'Scale']
        tensor = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]], self.typeCode)
        self.assertRaises(TypeError, scale, tensor)

    def testScaleWrongSize(self):
        if False:
            print('Hello World!')
        'Test scale function with wrong size'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + 'Scale']
        tensor = np.array([[[1, 0], [0, 1], [1, 0]], [[0, 1], [1, 0], [0, 1]], [[1, 0], [0, 1], [1, 0]]], self.typeCode)
        self.assertRaises(TypeError, scale, tensor)

    def testScaleNonArray(self):
        if False:
            i = 10
            return i + 15
        'Test scale function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        scale = Tensor.__dict__[self.typeStr + 'Scale']
        self.assertRaises(TypeError, scale, True)

    def testFloor(self):
        if False:
            while True:
                i = 10
        'Test floor function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + 'Floor']
        tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], self.typeCode)
        floor(tensor, 4)
        np.testing.assert_array_equal(tensor, np.array([[[4, 4], [4, 4]], [[5, 6], [7, 8]]]))

    def testFloorWrongType(self):
        if False:
            i = 10
            return i + 15
        'Test floor function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + 'Floor']
        tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 'c')
        self.assertRaises(TypeError, floor, tensor)

    def testFloorWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test floor function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + 'Floor']
        tensor = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], self.typeCode)
        self.assertRaises(TypeError, floor, tensor)

    def testFloorNonArray(self):
        if False:
            i = 10
            return i + 15
        'Test floor function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        floor = Tensor.__dict__[self.typeStr + 'Floor']
        self.assertRaises(TypeError, floor, object)

    def testCeil(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ceil function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + 'Ceil']
        tensor = np.array([[[9, 8], [7, 6]], [[5, 4], [3, 2]]], self.typeCode)
        ceil(tensor, 5)
        np.testing.assert_array_equal(tensor, np.array([[[5, 5], [5, 5]], [[5, 4], [3, 2]]]))

    def testCeilWrongType(self):
        if False:
            while True:
                i = 10
        'Test ceil function with wrong type'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + 'Ceil']
        tensor = np.array([[[9, 8], [7, 6]], [[5, 4], [3, 2]]], 'c')
        self.assertRaises(TypeError, ceil, tensor)

    def testCeilWrongDim(self):
        if False:
            while True:
                i = 10
        'Test ceil function with wrong dimensions'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + 'Ceil']
        tensor = np.array([[9, 8], [7, 6], [5, 4], [3, 2]], self.typeCode)
        self.assertRaises(TypeError, ceil, tensor)

    def testCeilNonArray(self):
        if False:
            return 10
        'Test ceil function with non-array'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        ceil = Tensor.__dict__[self.typeStr + 'Ceil']
        tensor = [[[9, 8], [7, 6]], [[5, 4], [3, 2]]]
        self.assertRaises(TypeError, ceil, tensor)

    def testLUSplit(self):
        if False:
            i = 10
            return i + 15
        'Test luSplit function'
        print(self.typeStr, '... ', end=' ', file=sys.stderr)
        luSplit = Tensor.__dict__[self.typeStr + 'LUSplit']
        (lower, upper) = luSplit([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        self.assertEqual((lower == [[[1, 1], [1, 0]], [[1, 0], [0, 0]]]).all(), True)
        self.assertEqual((upper == [[[0, 0], [0, 1]], [[0, 1], [1, 1]]]).all(), True)

class scharTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'schar'
        self.typeCode = 'b'
        self.result = int(self.result)

class ucharTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'uchar'
        self.typeCode = 'B'
        self.result = int(self.result)

class shortTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'short'
        self.typeCode = 'h'
        self.result = int(self.result)

class ushortTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'ushort'
        self.typeCode = 'H'
        self.result = int(self.result)

class intTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'int'
        self.typeCode = 'i'
        self.result = int(self.result)

class uintTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'uint'
        self.typeCode = 'I'
        self.result = int(self.result)

class longTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'long'
        self.typeCode = 'l'
        self.result = int(self.result)

class ulongTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'ulong'
        self.typeCode = 'L'
        self.result = int(self.result)

class longLongTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'longLong'
        self.typeCode = 'q'
        self.result = int(self.result)

class ulongLongTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'ulongLong'
        self.typeCode = 'Q'
        self.result = int(self.result)

class floatTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        TensorTestCase.__init__(self, methodName)
        self.typeStr = 'float'
        self.typeCode = 'f'

class doubleTestCase(TensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        TensorTestCase.__init__(self, methodName)
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
    print('Testing 3D Functions of Module Tensor')
    print('NumPy version', np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))