"""
Created on 2015/3/14
@author: Jimmy Liu
"""
import unittest
import tushare.stock.billboard as fd

class Test(unittest.TestCase):

    def set_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.date = '2015-06-12'
        self.days = 5

    def test_top_list(self):
        if False:
            print('Hello World!')
        self.set_data()
        print(fd.top_list(self.date))

    def test_cap_tops(self):
        if False:
            print('Hello World!')
        self.set_data()
        print(fd.cap_tops(self.days))

    def test_broker_tops(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        print(fd.broker_tops(self.days))

    def test_inst_tops(self):
        if False:
            i = 10
            return i + 15
        self.set_data()
        print(fd.inst_tops(self.days))

    def test_inst_detail(self):
        if False:
            while True:
                i = 10
        print(fd.inst_detail())
if __name__ == '__main__':
    unittest.main()