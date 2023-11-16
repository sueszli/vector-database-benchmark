__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from threading import Thread
from multiprocessing import Pool, Queue, Process, Manager
import multiprocessing
multiprocessing.freeze_support()

class CheckStock:

    def __init__(self):
        if False:
            return 10
        self.base = pd.read_csv('bases.csv', dtype={'code': np.str})
        '\n        if len(self.id)!=6:\n            print("Wrong stock code")\n            exit()\n        '

    def multi_thread(self):
        if False:
            i = 10
            return i + 15
        with open('stock_list.txt') as f:
            stock_list = f.readlines()
        ratio_list = []
        for i in stock_list:
            i = i.strip()
            ratio_list.append(self.get_info(i))
        return ratio_list

    def get_info(self, id):
        if False:
            while True:
                i = 10
        print(id)
        try:
            df = ts.get_today_ticks(id)
            print('len of df ', len(df))
            if len(df) == 0:
                print('Pause of exchange')
                return (id, 'pause')
        except Exception as e:
            print(e)
            print('ERROR')
            return (id, 'pause')
        "\n        print('\n')\n        max_p=df['price'].max()\n        print(max_p)\n        min_p=df['price'].min()\n        print(min_p)\n        #print(df)\n        "
        buy = df[df['type'] == '买盘']['volume'].sum()
        sell = df[df['type'] == '卖盘']['volume'].sum()
        neutral = df[df['type'] == '中性盘']['volume'].sum()
        start = df[-1:]
        vol_0 = start['volume'].sum()
        total = buy + sell + neutral + vol_0
        sum_all = df['volume'].sum()
        ratio = round((buy - sell) * 1.0 / sell * 100, 2)
        return (id, ratio)
        "\n        df['price'].plot()\n        plt.grid()\n\n        plt.show()\n        "

    def multi_process(self):
        if False:
            i = 10
            return i + 15
        stock_list = []
        with open('stock_list.txt') as f:
            stock_list = f.readlines()
        stock_list = map(lambda x: x.strip(), stock_list)
        '\n        p=Pool(len(stock_list))\n        result=p.map(self.get_info,stock_list)\n        p.close()\n        p.join()\n        '
        p = Pool(len(stock_list))
        result = []
        for i in stock_list:
            t = p.apply_async(self.get_info, args=(i,))
            result.append(t)
        p.close()
        p.join()
        print(result)
        '\n        for j in p_list:\n            j.start()\n        for k in p_list:\n            k.join()\n        '
        print(result)

    def show_name(self):
        if False:
            for i in range(10):
                print('nop')
        stock_list = self.multi_thread()
        for st in stock_list:
            print('code: ', st[0])
            name = self.base[self.base['code'] == st[0]]['name'].values[0]
            print('name: ', name)
            print('ratio: ', st[1])
            if st[1] > 30:
                print('WOW, more than 30')
            print('\n')

    def sinle_thread(self, start, end):
        if False:
            while True:
                i = 10
        for i in range(start, end):
            (id, ratio) = self.get_info(self.all_code[i])
            if ratio == 'pause':
                continue
            if ratio > 30:
                print(self.base[self.base['code'] == id]['name'].values[0], ' buy more than 30 percent')

    def scan_all(self):
        if False:
            i = 10
            return i + 15
        self.all_code = self.base['code'].values
        thread_num = 500
        all_num = len(self.all_code)
        each_thread = all_num / thread_num
        thread_list = []
        for i in range(thread_num):
            t = Thread(target=self.sinle_thread, args=(i * each_thread, (i + 1) * each_thread))
            thread_list.append(t)
        for j in thread_list:
            j.start()
        for k in thread_list:
            k.join()

    def monitor(self):
        if False:
            print('Hello World!')
        ratio_list = self.multi_thread()
        for js in ratio_list:
            if js[1] > 30:
                print(js[0])

def sub_process_ratio(i, q):
    if False:
        print('Hello World!')
    print('Start')
    try:
        df = ts.get_today_ticks(i)
        if len(df) == 0:
            print('Pause of exchange')
            return (i, 'pause')
    except Exception as e:
        print(e)
        print('ERROR')
        return (id, 'pause')
    "\n    print('\n')\n    max_p=df['price'].max()\n    print(max_p)\n    min_p=df['price'].min()\n    print(min_p)\n    #print(df)\n    "
    buy = df[df['type'] == '买盘']['volume'].sum()
    sell = df[df['type'] == '卖盘']['volume'].sum()
    neutral = df[df['type'] == '中性盘']['volume'].sum()
    start = df[-1:]
    vol_0 = start['volume'].sum()
    total = buy + sell + neutral + vol_0
    sum_all = df['volume'].sum()
    ratio = round((buy - sell) * 1.0 / sell * 100, 2)
    s = [i, ratio]
    print(s)
    q.put(s)

def testcase1(i, j, q):
    if False:
        for i in range(10):
            print('nop')
    print(i, j)
    q.put(i)

def multi_process():
    if False:
        i = 10
        return i + 15
    with open('stock_list.txt') as f:
        stock_list = f.readlines()
    stock_list = map(lambda x: x.strip(), stock_list)
    print(stock_list)
    '\n    p=Pool(len(stock_list))\n    result=p.map(self.get_info,stock_list)\n    p.close()\n    p.join()\n    '
    p = Pool(len(stock_list))
    result = []
    manager = Manager()
    q = manager.Queue()
    for i in stock_list:
        p.apply_async(sub_process_ratio, args=(i, q))
    p.close()
    p.join()
    while q.empty() == False:
        print('get')
        print(q.get())
if __name__ == '__main__':
    multi_process()