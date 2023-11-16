import random
import time
import tushare as ts
import pandas as pd
import os, datetime, math
import numpy as np
import logging
from configure.settings import DBSelector, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, REDIS_HOST
import redis
from threading import Thread
from common.BaseService import BaseService
DB = DBSelector()
engine = DB.get_engine('history', 'qq')
conn = ts.get_apis()
MYSQL_DB = 'history'
cursor = DB.get_mysql_conn(MYSQL_DB, 'qq').cursor()

class Kline(BaseService):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Kline, self).__init__('log/kline.log')
        path = os.path.join(os.getcwd(), 'data')
        self.today_date = datetime.datetime.now().strftime('%Y-%m-%d')
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)

    def store_base_data(self, target):
        if False:
            print('Hello World!')
        self.all_info = ts.get_stock_basics()
        self.all_info = self.all_info.reset_index()
        print(self.all_info)
        if target == 'sql':
            self.all_info.to_sql('tb_baseinfo', engine, if_exists='replace')
        elif target == 'csv':
            self.all_info.to_csv('baseInfo.csv')
        else:
            logging.info('sql or csv option. Not get right argument')

    def store_hist_data(self):
        if False:
            for i in range(10):
                print('nop')
        read_cmd = 'select * from tb_baseInfo;'
        df = pd.read_sql(read_cmd, engine)
        for i in range(len(df)):
            (code, name, start_date) = (df.loc[i]['code'], df.loc[i]['name'], df.loc[i]['timeToMarket'])
            self.get_hist_data(code, name, start_date)
            print(code, name, start_date)

    def get_hist_data(self, code, name, start_data):
        if False:
            while True:
                i = 10
        try:
            start_data = datetime.datetime.strptime(str(start_data), '%Y%m%d').strftime('%Y-%m-%d')
            df = ts.bar(code, conn=conn, start_date=start_data, adj='qfq')
            print(df)
        except Exception as e:
            print(e)
            return
        df.insert(1, 'name', name)
        df = df.reset_index()
        try:
            df.to_sql(code, engine, if_exists='append')
        except Exception as e:
            print(e)

    def inital_data(self, target):
        if False:
            for i in range(10):
                print('nop')
        if target == 'sql':
            self.today = pd.read_csv(self.today_date + '.csv', dtype={'code': np.str})
            self.all = pd.read_csv('bases.csv', dtype={'code': np.str})

    def _xiayingxian(self, row, ratio):
        if False:
            for i in range(10):
                print('nop')
        '\n        下影线的逻辑 ratio 下影线的长度比例，数字越大，下影线越长\n        row: series类型\n        '
        open_p = float(row['open'])
        closed = float(row['close'])
        low = float(row['low'])
        high = float(row['high'])
        p = min(closed, open_p)
        try:
            diff = (p - low) * 1.0 / (high - low)
            diff = round(diff, 3)
        except ZeroDivisionError:
            diff = 0
        if diff > ratio:
            xiayinxian_engine = DB.get_engine('db_selection', 'qq')
            (date, code, name, ocupy_ration, standards) = (row['datetime'], row['code'], row['name'], diff, ratio)
            df = pd.DataFrame({'datetime': [date], 'code': [code], 'name': [name], 'ocupy_ration': [ocupy_ration], 'standards': [standards]})
            try:
                df1 = pd.read_sql_table('xiayingxian', xiayinxian_engine, index_col='index')
                df = pd.concat([df1, df])
            except Exception as e:
                print(e)
            df = df.reset_index(drop=True)
            df.to_sql('xiayingxian', xiayinxian_engine, if_exists='replace')
            return row

    def store_data_not(self):
        if False:
            print('Hello World!')
        df = self._xiayingxian()
        df.to_csv('xiayinxian.csv')

    def redis_init(self):
        if False:
            print('Hello World!')
        rds = redis.StrictRedis(REDIS_HOST, 6379, db=0)
        rds_2 = redis.StrictRedis(REDIS_HOST, 6379, db=1)
        for i in rds.keys():
            d = dict({i: rds.get(i)})
            rds_2.lpush('codes', d)

    def get_hist_line(self, date):
        if False:
            for i in range(10):
                print('nop')
        print('Starting to capture')
        cmd = "select * from `{}` where datetime = '{}'"
        r0 = redis.StrictRedis(REDIS_HOST, 6379, db=0)
        for code in r0.keys():
            try:
                cursor.execute(cmd.format(code, date))
            except Exception as e:
                continue
            data = cursor.fetchall()
            try:
                data_row = data[0]
            except Exception as e:
                continue
            d = dict(zip(('datetime', 'code', 'name', 'open', 'close', 'high', 'low'), data_row[1:8]))
            self._xiayingxian(d, 0.7)

    def add_code_redis(self):
        if False:
            return 10
        rds = redis.StrictRedis(REDIS_HOST, 6379, db=0)
        rds_1 = redis.StrictRedis(REDIS_HOST, 6379, db=1)
        df = ts.get_stock_basics()
        df = df.reset_index()
        if rds.dbsize() != 0:
            rds.flushdb()
        if rds_1.dbsize() != 0:
            rds_1.flushdb()
        for i in range(len(df)):
            (code, name, timeToMarket) = (df.loc[i]['code'], df.loc[i]['name'], df.loc[i]['timeToMarket'])
            d = dict({code: ':'.join([name, str(timeToMarket)])})
            rds.set(code, name)
            rds_1.lpush('codes', d)

def get_hist_data(code, name, start_data):
    if False:
        print('Hello World!')
    try:
        df = ts.bar(code, conn=conn, start_date=start_data, adj='qfq')
    except Exception as e:
        print(e)
        return
    hist_con = DB.get_engine('history')
    df.insert(1, 'name', name)
    df = df.reset_index()
    df2 = pd.read_sql_table(code, hist_con, index_col='index')
    try:
        new_df = pd.concat([df, df2])
        new_df = new_df.reset_index(drop=True)
        new_df.to_sql(code, engine, if_exists='replace')
    except Exception as e:
        print(e)
        return

class StockThread(Thread):

    def __init__(self, loop):
        if False:
            while True:
                i = 10
        Thread.__init__(self)
        self.rds = redis.StrictRedis(REDIS_HOST, 6379, db=1)
        self.loop_count = loop

    def run(self):
        if False:
            i = 10
            return i + 15
        self.loops()

    def loops(self):
        if False:
            for i in range(10):
                print('nop')
        start_date = '2017-11-21'
        while 1:
            try:
                item = self.rds.lpop('codes')
                print(item)
            except Exception as e:
                print(e)
                break
            d = eval(item)
            k = d.keys()[0]
            v = d[k]
            name = v.split(':')[0].strip()
            get_hist_data(k, name, start_date)
THREAD_NUM = 4

def StoreData():
    if False:
        for i in range(10):
            print('nop')
    threads = []
    for i in range(THREAD_NUM):
        t = StockThread(i)
        t.start()
        threads.append(t)
    for j in range(THREAD_NUM):
        threads[j].join()
    print('done')

def main():
    if False:
        for i in range(10):
            print('nop')
    obj = Kline()
if __name__ == '__main__':
    main()