import unittest
import tushare.stock.shibor as fd

class Test(unittest.TestCase):

    def set_data(self):
        if False:
            return 10
        self.year = 2014

    def test_shibor_data(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        fd.shibor_data(self.year)

    def test_shibor_quote_data(self):
        if False:
            while True:
                i = 10
        self.set_data()
        fd.shibor_quote_data(self.year)

    def test_shibor_ma_data(self):
        if False:
            return 10
        self.set_data()
        fd.shibor_ma_data(self.year)

    def test_lpr_data(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        fd.lpr_data(self.year)

    def test_lpr_ma_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data()
        fd.lpr_ma_data(self.year)
if __name__ == '__main__':
    unittest.main()