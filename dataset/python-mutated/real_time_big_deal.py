__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import datetime
import tushare as ts
import pandas as pd
import time, os, threading
import numpy as np
from toolkit import Toolkit
pd.set_option('display.max_rows', None)

class BigMonitor:

    def __init__(self):
        if False:
            print('Hello World!')
        path = os.path.join(os.getcwd(), 'data')
        if os.path.exists(path) == False:
            os.mkdir(path)
            print('Please put data under data folder')
            exit()
        os.chdir(path)
        self.stockList = Toolkit.read_stock('mystock.csv')
        self.bases = pd.read_csv('bases.csv', dtype={'code': np.str})

    def loop(self, code):
        if False:
            return 10
        name = self.bases[self.bases['code'] == code]['name'].values[0]
        print(name)
        while 1:
            time.sleep(2)
            df_t1 = ts.get_realtime_quotes(code)
            v1 = long(df_t1['volume'].values[0])
            p1 = float(df_t1['price'].values[0])
            time.sleep(2)
            df_t2 = ts.get_realtime_quotes(code)
            v2 = long(df_t2['volume'].values[0])
            p2 = float(df_t2['price'].values[0])
            delta_v = (v2 - v1) / 100
            price_v = p2 - p1
            if delta_v > 1000:
                print(datetime.datetime.now().strftime('%H:%M:%S'))
                print('Big deal on %s' % name)
                print(delta_v, 'price diff', price_v)

    def multi_thread(self, code_list):
        if False:
            while True:
                i = 10
        thread_list = []
        for i in code_list:
            t = threading.Thread(target=self.loop, args=(i,))
            thread_list.append(t)
        for j in thread_list:
            j.start()

    def testcase(self):
        if False:
            for i in range(10):
                print('nop')
        self.multi_thread(self.stockList)

def main():
    if False:
        print('Hello World!')
    obj = BigMonitor()
    obj.testcase()
main()