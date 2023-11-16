"""
Created on 2015/3/14
@author: Jimmy Liu
"""
import unittest
from tushare.stock import reference as fd

class Test(unittest.TestCase):

    def set_data(self):
        if False:
            while True:
                i = 10
        self.code = '600848'
        self.start = '2015-01-03'
        self.end = '2015-04-07'
        self.year = 2014
        self.quarter = 4
        self.top = 60
        self.show_content = True

    def test_profit_data(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        print(fd.profit_data(top=self.top))

    def test_forecast_data(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        print(fd.forecast_data(self.year, self.quarter))

    def test_xsg_data(self):
        if False:
            i = 10
            return i + 15
        print(fd.xsg_data())

    def test_fund_holdings(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        print(fd.fund_holdings(self.year, self.quarter))

    def test_new_stocksa(self):
        if False:
            return 10
        print(fd.new_stocks())

    def test_sh_margin_details(self):
        if False:
            return 10
        self.set_data()
        print(fd.sh_margin_details(self.start, self.end, self.code))

    def test_sh_margins(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        print(fd.sh_margins(self.start, self.end))

    def test_sz_margins(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data()
        print(fd.sz_margins(self.start, self.end))

    def test_sz_margin_details(self):
        if False:
            return 10
        self.set_data()
        print(fd.sz_margin_details(self.end))
if __name__ == '__main__':
    unittest.main()