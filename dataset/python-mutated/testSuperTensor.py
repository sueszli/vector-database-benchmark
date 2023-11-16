import sys
import unittest
import numpy as np
(major, minor) = [int(d) for d in np.__version__.split('.')[:2]]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError
import SuperTensor

class SuperTensorTestCase(unittest.TestCase):

    def __init__(self, methodName='runTests'):
        if False:
            return 10
        unittest.TestCase.__init__(self, methodName)
        self.typeStr = 'double'
        self.typeCode = 'd'

    def testNorm(self):
        if False:
            print('Hello World!')
        'Test norm function'
        print(self.typeStr, '... ', file=sys.stderr)
        norm = SuperTensor.__dict__[self.typeStr + 'Norm']
        supertensor = np.arange(2 * 2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2, 2))
        answer = np.array([np.sqrt(np.sum(supertensor.astype('d') * supertensor) / 16.0)], dtype=self.typeCode)[0]
        self.assertAlmostEqual(norm(supertensor), answer, 6)

    def testNormBadList(self):
        if False:
            print('Hello World!')
        'Test norm function with bad list'
        print(self.typeStr, '... ', file=sys.stderr)
        norm = SuperTensor.__dict__[self.typeStr + 'Norm']
        supertensor = [[[[0, 'one'], [2, 3]], [[3, 'two'], [1, 0]]], [[[0, 'one'], [2, 3]], [[3, 'two'], [1, 0]]]]
        self.assertRaises(BadListError, norm, supertensor)

    def testNormWrongDim(self):
        if False:
            print('Hello World!')
        'Test norm function with wrong dimensions'
        print(self.typeStr, '... ', file=sys.stderr)
        norm = SuperTensor.__dict__[self.typeStr + 'Norm']
        supertensor = np.arange(2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2))
        self.assertRaises(TypeError, norm, supertensor)

    def testNormWrongSize(self):
        if False:
            return 10
        'Test norm function with wrong size'
        print(self.typeStr, '... ', file=sys.stderr)
        norm = SuperTensor.__dict__[self.typeStr + 'Norm']
        supertensor = np.arange(3 * 2 * 2, dtype=self.typeCode).reshape((3, 2, 2))
        self.assertRaises(TypeError, norm, supertensor)

    def testNormNonContainer(self):
        if False:
            print('Hello World!')
        'Test norm function with non-container'
        print(self.typeStr, '... ', file=sys.stderr)
        norm = SuperTensor.__dict__[self.typeStr + 'Norm']
        self.assertRaises(TypeError, norm, None)

    def testMax(self):
        if False:
            print('Hello World!')
        'Test max function'
        print(self.typeStr, '... ', file=sys.stderr)
        max = SuperTensor.__dict__[self.typeStr + 'Max']
        supertensor = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]
        self.assertEqual(max(supertensor), 8)

    def testMaxBadList(self):
        if False:
            print('Hello World!')
        'Test max function with bad list'
        print(self.typeStr, '... ', file=sys.stderr)
        max = SuperTensor.__dict__[self.typeStr + 'Max']
        supertensor = [[[[1, 'two'], [3, 4]], [[5, 'six'], [7, 8]]], [[[1, 'two'], [3, 4]], [[5, 'six'], [7, 8]]]]
        self.assertRaises(BadListError, max, supertensor)

    def testMaxNonContainer(self):
        if False:
            for i in range(10):
                print('nop')
        'Test max function with non-container'
        print(self.typeStr, '... ', file=sys.stderr)
        max = SuperTensor.__dict__[self.typeStr + 'Max']
        self.assertRaises(TypeError, max, None)

    def testMaxWrongDim(self):
        if False:
            i = 10
            return i + 15
        'Test max function with wrong dimensions'
        print(self.typeStr, '... ', file=sys.stderr)
        max = SuperTensor.__dict__[self.typeStr + 'Max']
        self.assertRaises(TypeError, max, [0, -1, 2, -3])

    def testMin(self):
        if False:
            for i in range(10):
                print('nop')
        'Test min function'
        print(self.typeStr, '... ', file=sys.stderr)
        min = SuperTensor.__dict__[self.typeStr + 'Min']
        supertensor = [[[[9, 8], [7, 6]], [[5, 4], [3, 2]]], [[[9, 8], [7, 6]], [[5, 4], [3, 2]]]]
        self.assertEqual(min(supertensor), 2)

    def testMinBadList(self):
        if False:
            return 10
        'Test min function with bad list'
        print(self.typeStr, '... ', file=sys.stderr)
        min = SuperTensor.__dict__[self.typeStr + 'Min']
        supertensor = [[[['nine', 8], [7, 6]], [['five', 4], [3, 2]]], [[['nine', 8], [7, 6]], [['five', 4], [3, 2]]]]
        self.assertRaises(BadListError, min, supertensor)

    def testMinNonContainer(self):
        if False:
            return 10
        'Test min function with non-container'
        print(self.typeStr, '... ', file=sys.stderr)
        min = SuperTensor.__dict__[self.typeStr + 'Min']
        self.assertRaises(TypeError, min, True)

    def testMinWrongDim(self):
        if False:
            i = 10
            return i + 15
        'Test min function with wrong dimensions'
        print(self.typeStr, '... ', file=sys.stderr)
        min = SuperTensor.__dict__[self.typeStr + 'Min']
        self.assertRaises(TypeError, min, [[1, 3], [5, 7]])

    def testScale(self):
        if False:
            print('Hello World!')
        'Test scale function'
        print(self.typeStr, '... ', file=sys.stderr)
        scale = SuperTensor.__dict__[self.typeStr + 'Scale']
        supertensor = np.arange(3 * 3 * 3 * 3, dtype=self.typeCode).reshape((3, 3, 3, 3))
        answer = supertensor.copy() * 4
        scale(supertensor, 4)
        self.assertEqual((supertensor == answer).all(), True)

    def testScaleWrongType(self):
        if False:
            while True:
                i = 10
        'Test scale function with wrong type'
        print(self.typeStr, '... ', file=sys.stderr)
        scale = SuperTensor.__dict__[self.typeStr + 'Scale']
        supertensor = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], 'c')
        self.assertRaises(TypeError, scale, supertensor)

    def testScaleWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test scale function with wrong dimensions'
        print(self.typeStr, '... ', file=sys.stderr)
        scale = SuperTensor.__dict__[self.typeStr + 'Scale']
        supertensor = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]], self.typeCode)
        self.assertRaises(TypeError, scale, supertensor)

    def testScaleWrongSize(self):
        if False:
            print('Hello World!')
        'Test scale function with wrong size'
        print(self.typeStr, '... ', file=sys.stderr)
        scale = SuperTensor.__dict__[self.typeStr + 'Scale']
        supertensor = np.array([[[1, 0], [0, 1], [1, 0]], [[0, 1], [1, 0], [0, 1]], [[1, 0], [0, 1], [1, 0]]], self.typeCode)
        self.assertRaises(TypeError, scale, supertensor)

    def testScaleNonArray(self):
        if False:
            while True:
                i = 10
        'Test scale function with non-array'
        print(self.typeStr, '... ', file=sys.stderr)
        scale = SuperTensor.__dict__[self.typeStr + 'Scale']
        self.assertRaises(TypeError, scale, True)

    def testFloor(self):
        if False:
            print('Hello World!')
        'Test floor function'
        print(self.typeStr, '... ', file=sys.stderr)
        supertensor = np.arange(2 * 2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2, 2))
        answer = supertensor.copy()
        answer[answer < 4] = 4
        floor = SuperTensor.__dict__[self.typeStr + 'Floor']
        floor(supertensor, 4)
        np.testing.assert_array_equal(supertensor, answer)

    def testFloorWrongType(self):
        if False:
            print('Hello World!')
        'Test floor function with wrong type'
        print(self.typeStr, '... ', file=sys.stderr)
        floor = SuperTensor.__dict__[self.typeStr + 'Floor']
        supertensor = np.ones(2 * 2 * 2 * 2, dtype='c').reshape((2, 2, 2, 2))
        self.assertRaises(TypeError, floor, supertensor)

    def testFloorWrongDim(self):
        if False:
            print('Hello World!')
        'Test floor function with wrong type'
        print(self.typeStr, '... ', file=sys.stderr)
        floor = SuperTensor.__dict__[self.typeStr + 'Floor']
        supertensor = np.arange(2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2))
        self.assertRaises(TypeError, floor, supertensor)

    def testFloorNonArray(self):
        if False:
            i = 10
            return i + 15
        'Test floor function with non-array'
        print(self.typeStr, '... ', file=sys.stderr)
        floor = SuperTensor.__dict__[self.typeStr + 'Floor']
        self.assertRaises(TypeError, floor, object)

    def testCeil(self):
        if False:
            print('Hello World!')
        'Test ceil function'
        print(self.typeStr, '... ', file=sys.stderr)
        supertensor = np.arange(2 * 2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2, 2))
        answer = supertensor.copy()
        answer[answer > 5] = 5
        ceil = SuperTensor.__dict__[self.typeStr + 'Ceil']
        ceil(supertensor, 5)
        np.testing.assert_array_equal(supertensor, answer)

    def testCeilWrongType(self):
        if False:
            while True:
                i = 10
        'Test ceil function with wrong type'
        print(self.typeStr, '... ', file=sys.stderr)
        ceil = SuperTensor.__dict__[self.typeStr + 'Ceil']
        supertensor = np.ones(2 * 2 * 2 * 2, 'c').reshape((2, 2, 2, 2))
        self.assertRaises(TypeError, ceil, supertensor)

    def testCeilWrongDim(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ceil function with wrong dimensions'
        print(self.typeStr, '... ', file=sys.stderr)
        ceil = SuperTensor.__dict__[self.typeStr + 'Ceil']
        supertensor = np.arange(2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2))
        self.assertRaises(TypeError, ceil, supertensor)

    def testCeilNonArray(self):
        if False:
            print('Hello World!')
        'Test ceil function with non-array'
        print(self.typeStr, '... ', file=sys.stderr)
        ceil = SuperTensor.__dict__[self.typeStr + 'Ceil']
        supertensor = np.arange(2 * 2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2, 2)).tolist()
        self.assertRaises(TypeError, ceil, supertensor)

    def testLUSplit(self):
        if False:
            print('Hello World!')
        'Test luSplit function'
        print(self.typeStr, '... ', file=sys.stderr)
        luSplit = SuperTensor.__dict__[self.typeStr + 'LUSplit']
        supertensor = np.ones(2 * 2 * 2 * 2, dtype=self.typeCode).reshape((2, 2, 2, 2))
        answer_upper = [[[[0, 0], [0, 1]], [[0, 1], [1, 1]]], [[[0, 1], [1, 1]], [[1, 1], [1, 1]]]]
        answer_lower = [[[[1, 1], [1, 0]], [[1, 0], [0, 0]]], [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]
        (lower, upper) = luSplit(supertensor)
        self.assertEqual((lower == answer_lower).all(), True)
        self.assertEqual((upper == answer_upper).all(), True)

class scharTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'schar'
        self.typeCode = 'b'

class ucharTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'uchar'
        self.typeCode = 'B'

class shortTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'short'
        self.typeCode = 'h'

class ushortTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'ushort'
        self.typeCode = 'H'

class intTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            print('Hello World!')
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'int'
        self.typeCode = 'i'

class uintTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            i = 10
            return i + 15
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'uint'
        self.typeCode = 'I'

class longTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'long'
        self.typeCode = 'l'

class ulongTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'ulong'
        self.typeCode = 'L'

class longLongTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'longLong'
        self.typeCode = 'q'

class ulongLongTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'ulongLong'
        self.typeCode = 'Q'

class floatTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        SuperTensorTestCase.__init__(self, methodName)
        self.typeStr = 'float'
        self.typeCode = 'f'

class doubleTestCase(SuperTensorTestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        SuperTensorTestCase.__init__(self, methodName)
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
    print('Testing 4D Functions of Module SuperTensor')
    print('NumPy version', np.__version__)
    print()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(bool(result.errors + result.failures))