import stock
import unittest

class TestStock(unittest.TestCase):

    def test_create(self):
        if False:
            i = 10
            return i + 15
        s = stock.Stock('GOOG', 100, 490.1)
        self.assertEqual(s.name, 'GOOG')
        self.assertEqual(s.shares, 100)
        self.assertEqual(s.price, 490.1)

    def test_create_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        s = stock.Stock(name='GOOG', shares=100, price=490.1)
        self.assertEqual(s.name, 'GOOG')
        self.assertEqual(s.shares, 100)
        self.assertEqual(s.price, 490.1)

    def test_cost(self):
        if False:
            print('Hello World!')
        s = stock.Stock('GOOG', 100, 490.1)
        self.assertEqual(s.cost, 49010.0)

    def test_sell(self):
        if False:
            return 10
        s = stock.Stock('GOOG', 100, 490.1)
        s.sell(25)
        self.assertEqual(s.shares, 75)

    def test_from_row(self):
        if False:
            return 10
        s = stock.Stock.from_row(['GOOG', '100', '490.1'])
        self.assertEqual(s.name, 'GOOG')
        self.assertEqual(s.shares, 100)
        self.assertEqual(s.price, 490.1)

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        s = stock.Stock('GOOG', 100, 490.1)
        self.assertEqual(repr(s), "Stock('GOOG', 100, 490.1)")

    def test_eq(self):
        if False:
            for i in range(10):
                print('nop')
        a = stock.Stock('GOOG', 100, 490.1)
        b = stock.Stock('GOOG', 100, 490.1)
        self.assertTrue(a == b)

    def test_shares_badtype(self):
        if False:
            return 10
        s = stock.Stock('GOOG', 100, 490.1)
        with self.assertRaises(TypeError):
            s.shares = '50'

    def test_shares_badvalue(self):
        if False:
            for i in range(10):
                print('nop')
        s = stock.Stock('GOOG', 100, 490.1)
        with self.assertRaises(ValueError):
            s.shares = -50

    def test_price_badtype(self):
        if False:
            print('Hello World!')
        s = stock.Stock('GOOG', 100, 490.1)
        with self.assertRaises(TypeError):
            s.price = '45.23'

    def test_price_badvalue(self):
        if False:
            while True:
                i = 10
        s = stock.Stock('GOOG', 100, 490.1)
        with self.assertRaises(ValueError):
            s.price = -45.23

    def test_bad_attribute(self):
        if False:
            while True:
                i = 10
        s = stock.Stock('GOOG', 100, 490.1)
        with self.assertRaises(AttributeError):
            s.share = 100
if __name__ == '__main__':
    unittest.main()