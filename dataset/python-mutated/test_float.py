from math import log10
from odoo.tests.common import TransactionCase
from odoo.tools import float_compare, float_is_zero, float_repr, float_round

class TestFloatPrecision(TransactionCase):
    """ Tests on float precision. """

    def test_rounding_02(self):
        if False:
            while True:
                i = 10
        ' Test rounding methods with 2 digits. '
        currency = self.env.ref('base.EUR')

        def try_round(amount, expected):
            if False:
                i = 10
                return i + 15
            digits = max(0, -int(log10(currency.rounding)))
            result = float_repr(currency.round(amount), precision_digits=digits)
            self.assertEqual(result, expected, 'Rounding error: got %s, expected %s' % (result, expected))
        try_round(2.674, '2.67')
        try_round(2.675, '2.68')
        try_round(-2.675, '-2.68')
        try_round(0.001, '0.00')
        try_round(-0.001, '-0.00')
        try_round(0.0049, '0.00')
        try_round(0.005, '0.01')
        try_round(-0.005, '-0.01')

        def try_zero(amount, expected):
            if False:
                i = 10
                return i + 15
            self.assertEqual(currency.is_zero(amount), expected, 'Rounding error: %s should be zero!' % amount)
        try_zero(0.01, False)
        try_zero(-0.01, False)
        try_zero(0.001, True)
        try_zero(-0.001, True)
        try_zero(0.0046, True)
        try_zero(-0.0046, True)
        try_zero(2.68 - 2.675, False)
        try_zero(2.68 - 2.676, True)
        try_zero(2.676 - 2.68, True)
        try_zero(2.675 - 2.68, False)

        def try_compare(amount1, amount2, expected):
            if False:
                print('Hello World!')
            self.assertEqual(currency.compare_amounts(amount1, amount2), expected, 'Rounding error, compare_amounts(%s,%s) should be %s' % (amount1, amount2, expected))
        try_compare(0.001, 0.001, 0)
        try_compare(-0.001, -0.001, 0)
        try_compare(0.001, 0.002, 0)
        try_compare(-0.001, -0.002, 0)
        try_compare(2.675, 2.68, 0)
        try_compare(2.676, 2.68, 0)
        try_compare(-2.676, -2.68, 0)
        try_compare(2.674, 2.68, -1)
        try_compare(-2.674, -2.68, 1)
        try_compare(3, 2.68, 1)
        try_compare(-3, -2.68, -1)
        try_compare(0.01, 0, 1)
        try_compare(-0.01, 0, -1)

    def test_rounding_03(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test rounding methods with 3 digits. '

        def try_round(amount, expected, digits=3, method='HALF-UP'):
            if False:
                return 10
            value = float_round(amount, precision_digits=digits, rounding_method=method)
            result = float_repr(value, precision_digits=digits)
            self.assertEqual(result, expected, 'Rounding error: got %s, expected %s' % (result, expected))
        try_round(2.6745, '2.675')
        try_round(-2.6745, '-2.675')
        try_round(2.6744, '2.674')
        try_round(-2.6744, '-2.674')
        try_round(0.0004, '0.000')
        try_round(-0.0004, '-0.000')
        try_round(357.4555, '357.456')
        try_round(-357.4555, '-357.456')
        try_round(457.4554, '457.455')
        try_round(-457.4554, '-457.455')
        try_round(8.175, '8.175', method='UP')
        try_round(8.1751, '8.176', method='UP')
        try_round(-8.175, '-8.175', method='UP')
        try_round(-8.1751, '-8.176', method='UP')
        try_round(-6.0, '-6.000', method='UP')
        try_round(1.8, '2', 0, method='UP')
        try_round(-1.8, '-2', 0, method='UP')
        fractions = [0.0, 0.015, 0.01499, 0.675, 0.67499, 0.4555, 0.4555, 0.45555]
        expecteds = ['.00', '.02', '.01', '.68', '.67', '.46', '.456', '.4556']
        precisions = [2, 2, 2, 2, 2, 2, 3, 4]
        for magnitude in range(7):
            for i in xrange(len(fractions)):
                (frac, exp, prec) = (fractions[i], expecteds[i], precisions[i])
                for sign in [-1, 1]:
                    for x in xrange(0, 10000, 97):
                        n = x * 10 ** magnitude
                        f = sign * (n + frac)
                        f_exp = ('-' if f != 0 and sign == -1 else '') + str(n) + exp
                        try_round(f, f_exp, digits=prec)

        def try_zero(amount, expected):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(float_is_zero(amount, precision_digits=3), expected, 'Rounding error: %s should be zero!' % amount)
        try_zero(0.0002, True)
        try_zero(-0.0002, True)
        try_zero(0.00034, True)
        try_zero(0.0005, False)
        try_zero(-0.0005, False)
        try_zero(0.0008, False)
        try_zero(-0.0008, False)

        def try_compare(amount1, amount2, expected):
            if False:
                i = 10
                return i + 15
            self.assertEqual(float_compare(amount1, amount2, precision_digits=3), expected, 'Rounding error, compare_amounts(%s,%s) should be %s' % (amount1, amount2, expected))
        try_compare(0.0003, 0.0004, 0)
        try_compare(-0.0003, -0.0004, 0)
        try_compare(0.0002, 0.0005, -1)
        try_compare(-0.0002, -0.0005, 1)
        try_compare(0.0009, 0.0004, 1)
        try_compare(-0.0009, -0.0004, -1)
        try_compare(557.4555, 557.4556, 0)
        try_compare(-557.4555, -557.4556, 0)
        try_compare(657.4444, 657.445, -1)
        try_compare(-657.4444, -657.445, 1)

        def try_round(amount, expected, precision_rounding=None):
            if False:
                for i in range(10):
                    print('nop')
            value = float_round(amount, precision_rounding=precision_rounding)
            result = float_repr(value, precision_digits=2)
            self.assertEqual(result, expected, 'Rounding error: got %s, expected %s' % (result, expected))
        try_round(-457.4554, '-457.45', precision_rounding=0.05)
        try_round(457.444, '457.50', precision_rounding=0.5)
        try_round(457.3, '455.00', precision_rounding=5)
        try_round(457.5, '460.00', precision_rounding=5)
        try_round(457.1, '456.00', precision_rounding=3)

    def test_rounding_04(self):
        if False:
            while True:
                i = 10
        ' check that proper rounding is performed for float persistence '
        currency = self.env.ref('base.EUR')
        currency_rate = self.env['res.currency.rate']

        def try_roundtrip(value, expected):
            if False:
                while True:
                    i = 10
            rate = currency_rate.create({'name': '2000-01-01', 'rate': value, 'currency_id': currency.id})
            self.assertEqual(rate.rate, expected, 'Roundtrip error: got %s back from db, expected %s' % (rate, expected))
        try_roundtrip(2.6748955, 2.674896)
        try_roundtrip(-2.6748955, -2.674896)
        try_roundtrip(10000.999999, 10000.999999)
        try_roundtrip(-10000.999999, -10000.999999)

    def test_rounding_invalid(self):
        if False:
            while True:
                i = 10
        ' verify that invalid parameters are forbidden '
        with self.assertRaises(AssertionError):
            float_is_zero(0.01, precision_digits=3, precision_rounding=0.01)
        with self.assertRaises(AssertionError):
            float_compare(0.01, 0.02, precision_digits=3, precision_rounding=0.01)
        with self.assertRaises(AssertionError):
            float_round(0.01, precision_digits=3, precision_rounding=0.01)