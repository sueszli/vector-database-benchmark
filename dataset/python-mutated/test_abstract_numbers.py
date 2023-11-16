"""Unit tests for numbers.py."""
import math
import operator
import unittest
from numbers import Complex, Real, Rational, Integral

class TestNumbers(unittest.TestCase):

    def test_int(self):
        if False:
            return 10
        self.assertTrue(issubclass(int, Integral))
        self.assertTrue(issubclass(int, Complex))
        self.assertEqual(7, int(7).real)
        self.assertEqual(0, int(7).imag)
        self.assertEqual(7, int(7).conjugate())
        self.assertEqual(-7, int(-7).conjugate())
        self.assertEqual(7, int(7).numerator)
        self.assertEqual(1, int(7).denominator)

    def test_float(self):
        if False:
            return 10
        self.assertFalse(issubclass(float, Rational))
        self.assertTrue(issubclass(float, Real))
        self.assertEqual(7.3, float(7.3).real)
        self.assertEqual(0, float(7.3).imag)
        self.assertEqual(7.3, float(7.3).conjugate())
        self.assertEqual(-7.3, float(-7.3).conjugate())

    def test_complex(self):
        if False:
            print('Hello World!')
        self.assertFalse(issubclass(complex, Real))
        self.assertTrue(issubclass(complex, Complex))
        (c1, c2) = (complex(3, 2), complex(4, 1))
        self.assertRaises(TypeError, math.trunc, c1)
        self.assertRaises(TypeError, operator.mod, c1, c2)
        self.assertRaises(TypeError, divmod, c1, c2)
        self.assertRaises(TypeError, operator.floordiv, c1, c2)
        self.assertRaises(TypeError, float, c1)
        self.assertRaises(TypeError, int, c1)
if __name__ == '__main__':
    unittest.main()