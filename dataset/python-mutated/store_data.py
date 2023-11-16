__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import tushare as ts
import sqlite3

class TS_DB:

    def __init__(self):
        if False:
            print('Hello World!')
        self.db = sqlite3.connect('testdb.db')

    def save_csv(self, code):
        if False:
            for i in range(10):
                print('nop')
        df = ts.get_k_data(code, start='2016-01-01', end='2016-12-28')
        filename = code + '.csv'
        df.to_sql('newtable', self.db, flavor='sqlite')
if __name__ == '__main__':
    obj = TS_DB()
    obj.save_csv('300333')