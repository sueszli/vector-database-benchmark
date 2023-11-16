import unittest
import tushare.stock.fundamental as fd

class Test(unittest.TestCase):

    def set_data(self):
        if False:
            print('Hello World!')
        self.code = '600848'
        self.start = '2015-01-03'
        self.end = '2015-04-07'
        self.year = 2014
        self.quarter = 4

    def test_get_stock_basics(self):
        if False:
            i = 10
            return i + 15
        print(fd.get_stock_basics())
if __name__ == '__main__':
    unittest.main()