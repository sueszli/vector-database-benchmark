"""
Created on 2015/3/14
@author: Jimmy Liu
"""
import unittest
import tushare.stock.trading as fd

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

    def test_get_hist_data(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        print(fd.get_hist_data(self.code, self.start))

    def test_get_tick_data(self):
        if False:
            return 10
        self.set_data()
        print(fd.get_tick_data(self.code, self.end))

    def test_get_today_all(self):
        if False:
            i = 10
            return i + 15
        print(fd.get_today_all())

    def test_get_realtime_quotesa(self):
        if False:
            print('Hello World!')
        self.set_data()
        print(fd.get_realtime_quotes(self.code))

    def test_get_h_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data()
        print(fd.get_h_data(self.code, self.start, self.end))

    def test_get_today_ticks(self):
        if False:
            print('Hello World!')
        self.set_data()
        print(fd.get_today_ticks(self.code))
if __name__ == '__main__':
    unittest.main()