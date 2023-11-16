from algorithms.maths.polynomial import Monomial
from fractions import Fraction
import math
import unittest

class TestSuite(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.m1 = Monomial({})
        self.m2 = Monomial({1: 1}, 2)
        self.m3 = Monomial({1: 2, 2: -1}, 1.5)
        self.m4 = Monomial({1: 1, 2: 2, 3: -2}, 3)
        self.m5 = Monomial({2: 1, 3: 0}, Fraction(2, 3))
        self.m6 = Monomial({1: 0, 2: 0, 3: 0}, -2.27)
        self.m7 = Monomial({1: 2, 7: 2}, -math.pi)
        self.m8 = Monomial({150: 5, 170: 2, 10000: 3}, 0)
        self.m9 = 2
        self.m10 = math.pi
        self.m11 = Fraction(3, 8)
        self.m12 = 0
        self.m13 = Monomial({1: 1}, -2)
        self.m14 = Monomial({1: 2}, 3)
        self.m15 = Monomial({1: 1}, 3)
        self.m16 = Monomial({1: 2, 7: 2}, math.pi)
        self.m17 = Monomial({1: -1})

    def test_monomial_addition(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, lambda x, y: x + y, self.m1, self.m2)
        self.assertRaises(ValueError, lambda x, y: x + y, self.m2, self.m3)
        self.assertRaises(ValueError, lambda x, y: x + y, self.m2, self.m14)
        self.assertEqual(self.m13 + self.m2, self.m1)
        self.assertEqual(self.m1 + self.m1, self.m1)
        self.assertEqual(self.m7 + self.m7, Monomial({1: 2, 7: 2}, -2 * math.pi))
        self.assertEqual(self.m8, self.m1)
        self.assertRaises(ValueError, lambda x, y: x + y, self.m2, self.m9)
        self.assertRaises(TypeError, lambda x, y: x + y, self.m9, self.m2)
        self.assertEqual(self.m1 + self.m9, Monomial({}, 2))
        self.assertEqual(self.m1 + self.m12, Monomial({}, 0))
        return

    def test_monomial_subtraction(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, lambda x, y: x - y, self.m1, self.m2)
        self.assertRaises(ValueError, lambda x, y: x - y, self.m2, self.m3)
        self.assertRaises(ValueError, lambda x, y: x - y, self.m2, self.m14)
        self.assertEqual(self.m2 - self.m2, self.m1)
        self.assertEqual(self.m2 - self.m2, Monomial({}, 0))
        self.assertEqual(self.m1 - self.m1, self.m1)
        self.assertEqual(self.m2 - self.m15, Monomial({1: 1}, -1))
        self.assertEqual(self.m16 - self.m7, Monomial({1: 2, 7: 2}, 2 * math.pi))
        self.assertRaises(ValueError, lambda x, y: x - y, self.m2, self.m9)
        self.assertRaises(TypeError, lambda x, y: x - y, self.m9, self.m2)
        self.assertEqual(self.m1 - self.m9, Monomial({}, -2))
        self.assertEqual(self.m1 - self.m12, Monomial({}, 0))
        return

    def test_monomial_multiplication(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.m2 * self.m13, Monomial({1: 2}, -4))
        self.assertEqual(self.m2 * self.m17, Monomial({}, 2))
        self.assertEqual(self.m8 * self.m5, self.m1)
        self.assertEqual(self.m1 * self.m2, self.m1)
        self.assertEqual(self.m7 * self.m3, Monomial({1: 4, 2: -1, 7: 2}, -1.5 * math.pi))
        return

    def test_monomial_inverse(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, lambda x: x.inverse(), self.m1)
        self.assertRaises(ValueError, lambda x: x.inverse(), self.m8)
        self.assertRaises(ValueError, lambda x: x.inverse(), Monomial({}, self.m12))
        self.assertEqual(self.m7.inverse(), Monomial({1: -2, 7: -2}, -1 / math.pi))
        self.assertEqual(self.m5.inverse(), Monomial({2: -1}, Fraction(3, 2)))
        self.assertEqual(self.m5.inverse(), Monomial({2: -1}, 1.5))
        self.assertTrue(self.m6.inverse(), Monomial({}, Fraction(-100, 227)))
        self.assertEqual(self.m6.inverse(), Monomial({}, -1 / 2.27))
        return

    def test_monomial_division(self):
        if False:
            return 10
        self.assertRaises(ValueError, lambda x, y: x.__truediv__(y), self.m2, self.m1)
        self.assertRaises(ValueError, lambda x, y: x.__truediv__(y), self.m2, self.m8)
        self.assertRaises(ValueError, lambda x, y: x.__truediv__(y), self.m2, self.m12)
        self.assertEqual(self.m7 / self.m3, Monomial({2: 1, 7: 2}, -2 * math.pi / 3))
        self.assertEqual(self.m14 / self.m13, Monomial({1: 1}) * Fraction(-3, 2))
        return

    def test_monomial_substitution(self):
        if False:
            i = 10
            return i + 15
        self.assertAlmostEqual(self.m7.substitute(2), -16 * math.pi, delta=1e-09)
        self.assertAlmostEqual(self.m7.substitute(1.5), 1.5 ** 4 * -math.pi, delta=1e-09)
        self.assertAlmostEqual(self.m7.substitute(Fraction(-1, 2)), Fraction(-1, 2) ** 4 * -math.pi, delta=1e-09)
        self.assertAlmostEqual(self.m7.substitute({1: 3, 7: 0}), 3 ** 2 * 0 ** 2 * -math.pi, delta=1e-09)
        self.assertAlmostEqual(self.m7.substitute({1: 3, 7: 0, 2: 2}), 3 ** 2 * 0 ** 2 * -math.pi, delta=1e-09)
        self.assertRaises(ValueError, lambda x, y: x.substitute(y), self.m7, {1: 3, 2: 2})
        self.assertRaises(ValueError, lambda x, y: x.substitute(y), self.m7, {2: 2})
        self.assertEqual(self.m8.substitute(2), 0)
        self.assertEqual(self.m8.substitute({1231: 2, 1: 2}), 0)
        return

    def test_monomial_all_variables(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.m5.all_variables(), {2})
        self.assertEqual(self.m6.all_variables(), set())
        self.assertEqual(self.m8.all_variables(), set())
        return

    def test_monomial_clone(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.m3, self.m3.clone())
        self.assertEqual(self.m1, self.m8.clone())
        self.assertEqual(self.m1, self.m1.clone())
        self.assertEqual(self.m8, self.m1.clone())
        self.assertEqual(self.m8, self.m8.clone())
        return
if __name__ == '__main__':
    unittest.main()