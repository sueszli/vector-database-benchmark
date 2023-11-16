"""
Created on 2015/3/14
@author: Jimmy Liu
"""
import unittest
import tushare.stock.newsevent as fd

class Test(unittest.TestCase):

    def set_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.code = '600848'
        self.start = '2015-01-03'
        self.end = '2015-04-07'
        self.year = 2014
        self.quarter = 4
        self.top = 60
        self.show_content = True

    def test_get_latest_news(self):
        if False:
            return 10
        self.set_data()
        print(fd.get_latest_news(self.top, self.show_content))

    def test_get_notices(self):
        if False:
            print('Hello World!')
        self.set_data()
        df = fd.get_notices(self.code)
        print(fd.notice_content(df.ix[0]['url']))

    def test_guba_sina(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data()
        print(fd.guba_sina(self.show_content))
if __name__ == '__main__':
    unittest.main()