from calculus.base_2 import Calculation
from twisted.trial import unittest

class CalculationTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.calc = Calculation()

    def _test(self, operation, a, b, expected):
        if False:
            i = 10
            return i + 15
        result = operation(a, b)
        self.assertEqual(result, expected)

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(self.calc.add, 3, 8, 11)

    def test_subtract(self):
        if False:
            while True:
                i = 10
        self._test(self.calc.subtract, 7, 3, 4)

    def test_multiply(self):
        if False:
            while True:
                i = 10
        self._test(self.calc.multiply, 6, 9, 54)

    def test_divide(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(self.calc.divide, 12, 5, 2)