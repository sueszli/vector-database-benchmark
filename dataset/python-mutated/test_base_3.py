from calculus.base_3 import Calculation
from twisted.trial import unittest

class CalculationTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.calc = Calculation()

    def _test(self, operation, a, b, expected):
        if False:
            while True:
                i = 10
        result = operation(a, b)
        self.assertEqual(result, expected)

    def _test_error(self, operation):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, operation, 'foo', 2)
        self.assertRaises(TypeError, operation, 'bar', 'egg')
        self.assertRaises(TypeError, operation, [3], [8, 2])
        self.assertRaises(TypeError, operation, {'e': 3}, {'r': 't'})

    def test_add(self):
        if False:
            print('Hello World!')
        self._test(self.calc.add, 3, 8, 11)

    def test_subtract(self):
        if False:
            return 10
        self._test(self.calc.subtract, 7, 3, 4)

    def test_multiply(self):
        if False:
            while True:
                i = 10
        self._test(self.calc.multiply, 6, 9, 54)

    def test_divide(self):
        if False:
            while True:
                i = 10
        self._test(self.calc.divide, 12, 5, 2)

    def test_errorAdd(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_error(self.calc.add)

    def test_errorSubtract(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_error(self.calc.subtract)

    def test_errorMultiply(self):
        if False:
            i = 10
            return i + 15
        self._test_error(self.calc.multiply)

    def test_errorDivide(self):
        if False:
            print('Hello World!')
        self._test_error(self.calc.divide)