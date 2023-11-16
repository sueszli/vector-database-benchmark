from calculus.base_2 import Calculation
from twisted.trial import unittest

class CalculationTestCase(unittest.TestCase):

    def test_add(self):
        if False:
            while True:
                i = 10
        calc = Calculation()
        result = calc.add(3, 8)
        self.assertEqual(result, 11)

    def test_subtract(self):
        if False:
            for i in range(10):
                print('nop')
        calc = Calculation()
        result = calc.subtract(7, 3)
        self.assertEqual(result, 4)

    def test_multiply(self):
        if False:
            print('Hello World!')
        calc = Calculation()
        result = calc.multiply(12, 5)
        self.assertEqual(result, 60)

    def test_divide(self):
        if False:
            i = 10
            return i + 15
        calc = Calculation()
        result = calc.divide(12, 5)
        self.assertEqual(result, 2)