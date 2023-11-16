__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n交割单处理 保存交割单到数据库\n'
import os
import datetime
import pandas as pd
import numpy as np
import re
from configure.settings import DBSelector
import fire
pd.set_option('display.max_rows', None)

class DeliveryOrder:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.gj_table = 'tb_delivery_gj_django'
        self.hb_table = 'tb_delivery_hb_django'
        self.db_init()

    def db_init(self):
        if False:
            i = 10
            return i + 15
        DB = DBSelector()
        self.engine = DB.get_engine('db_stock', 'qq')
        self.conn = DB.get_mysql_conn('db_stock', 'qq')

    def setpath(self, path):
        if False:
            while True:
                i = 10
        path = os.path.join(os.getcwd(), path)
        if os.path.exists(path) == False:
            os.mkdir(path)
        os.chdir(path)

    def merge_data_HuaBao(self, filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            df = pd.read_csv(filename, encoding='gbk')
        except Exception as e:
            print(e)
            raise OSError('打开文件失败')
        df = df.reset_index(drop='True')
        df = df.dropna(subset=['成交时间'])
        df['成交日期'] = df['成交日期'].astype(np.str) + df['成交时间']
        df['成交日期'] = df['成交日期'].map(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'))
        try:
            df['成交日期'] = pd.to_datetime(df['成交日期'])
        except Exception as e:
            print(e)
        del df['股东代码']
        del df['成交时间']
        df = df[(df['委托类别'] == '买入') | (df['委托类别'] == '卖出')]
        df = df.fillna(0)
        df = df.sort_values(by='成交日期', ascending=False)
        cursor = self.conn.cursor()
        insert_cmd = f'\n               insert into {self.hb_table} (成交日期,证券代码,证券名称,委托类别,成交数量,成交价格,成交金额,发生金额,佣金,印花税,过户费,其他费) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        check_dup = f'\n               select * from {self.hb_table} where 成交日期=%s and 证券代码=%s and 委托类别=%s and 成交数量=%s and 发生金额=%s\n               '
        for (index, row) in df.iterrows():
            date = row['成交日期']
            date = date.to_pydatetime()
            cursor.execute(check_dup, (date, row['证券代码'], row['委托类别'], row['成交数量'], row['发生金额']))
            if cursor.fetchall():
                print('有重复数据，忽略')
                continue
            else:
                cursor.execute(insert_cmd, (date, row['证券代码'], row['证券名称'], row['委托类别'], row['成交数量'], row['成交价格'], row['成交金额'], row['发生金额'], row['佣金'], row['印花税'], row['过户费'], row['其他费']))
        self.conn.commit()
        self.conn.close()

    def years_ht(self):
        if False:
            while True:
                i = 10
        df_list = []
        for i in range(1, 2):
            filename = 'HT_2018-05_week4-5.xls'
            try:
                t = pd.read_table(filename, encoding='gbk', dtype={'证券代码': np.str})
            except Exception as e:
                print(e)
                continue
            df_list.append(t)
        df = pd.concat(df_list)
        df = df.reset_index()
        df['成交日期'] = map(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'), df['成交日期'])
        df = df[df['摘要'] != '申购配号']
        df = df[df['摘要'] != '质押回购拆出']
        df = df[df['摘要'] != '拆出质押购回']
        del df['合同编号']
        del df['备注']
        del df['股东帐户']
        del df['结算汇率']
        del df['Unnamed: 16']
        df = df.sort_values(by='成交日期')
        df = df.set_index('成交日期')
        df.to_sql('tb_delivery_HT', self.engine, if_exists='append')

    def caculation(self, df):
        if False:
            while True:
                i = 10
        fee = df['手续费'].sum() + df['印花税'].sum() + df['其他杂费'].sum()
        print(fee)

    def month(self):
        if False:
            i = 10
            return i + 15
        pass

    def years_gj(self):
        if False:
            while True:
                i = 10
        df_list = []
        for i in range(2, 12):
            filename = 'GJ_2018_%s.csv' % str(i).zfill(2)
            try:
                t = pd.read_csv(filename, encoding='gbk', dtype={'证券代码': np.str})
            except Exception as e:
                print(e)
            df_list.append(t)
        df = pd.concat(df_list)
        df = df.reset_index(drop='True')
        df['成交日期'] = df['成交日期'].astype(np.str) + df['成交时间']
        df['成交日期'] = df['成交日期'].map(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'))
        try:
            df['成交日期'] = pd.to_datetime(df['成交日期'])
        except Exception as e:
            print(e)
        del df['股东帐户']
        del df['成交时间']
        df = df.sort_values(by='成交日期', ascending=False)
        df = df.set_index('成交日期')
        df.to_sql('tb_delivery_gj', self.engine, if_exists='replace')

    def file_exists(self, filepath):
        if False:
            i = 10
            return i + 15
        return True if os.path.exists(filepath) else False

    def years_gj_each_month_day(self, filename):
        if False:
            return 10
        if not self.file_exists(filename):
            raise ValueError('路径不存在')
        try:
            df = pd.read_csv(filename, encoding='gbk', dtype={'证券代码': np.str})
        except Exception as e:
            print(e)
            raise ValueError('读取文件错误')
        df = df.reset_index(drop='True')
        df['成交日期'] = df['成交日期'].astype(np.str) + df['成交时间']
        df['成交日期'] = df['成交日期'].map(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'))
        try:
            df['成交日期'] = pd.to_datetime(df['成交日期'])
        except Exception as e:
            print(e)
        del df['股东帐户']
        del df['成交时间']
        df = df.fillna(0)
        df = df[(df['操作'] != '申购配号') & (df['操作'] != '拆出质押购回') & (df['操作'] != '质押回购拆出')]
        df = df.sort_values(by='成交日期', ascending=False)
        cursor = self.conn.cursor()
        insert_cmd = f'\n        insert into {self.gj_table} (成交日期,证券代码,证券名称,操作,成交数量,成交均价,成交金额,余额,发生金额,手续费,印花税,过户费,本次金额,其他费用,交易市场) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        check_dup = f'\n        select * from {self.gj_table} where 成交日期=%s and 证券代码=%s and 操作=%s and 成交数量=%s and 余额=%s\n        '
        for (index, row) in df.iterrows():
            date = row['成交日期']
            date = date.to_pydatetime()
            cursor.execute(check_dup, (date, row['证券代码'], row['操作'], row['成交数量'], row['余额']))
            if cursor.fetchall():
                print('有重复数据，忽略')
            else:
                cursor.execute(insert_cmd, (date, row['证券代码'], row['证券名称'], row['操作'], row['成交数量'], row['成交均价'], row['成交金额'], row['余额'], row['发生金额'], row['手续费'], row['印花税'], row['过户费'], row['本次金额'], row['其他费用'], row['交易市场']))
        self.conn.commit()
        self.conn.close()

    def pretty(self):
        if False:
            for i in range(10):
                print('nop')
        df = pd.read_sql('tb_delivery_GJ', self.engine, index_col='成交日期')
        del df['index']
        df.to_sql('tb_delivery_GJ', self.engine, if_exists='replace')

    def data_sync(self):
        if False:
            while True:
                i = 10
        cursor = self.conn.cursor()
        select_cmd = 'select * from tb_delivery_gj'
        cursor.execute(select_cmd)
        ret = list(cursor.fetchall())
        print('new db ', len(ret))
        select_cmd2 = 'select * from tb_delivery_gj_django'
        cursor.execute(select_cmd2)
        ret2 = list(cursor.fetchall())
        print('old db ', len(ret2))
        ret_copy = ret.copy()
        for item in ret:
            for item2 in ret2:
                if item[0] == item2[0] and item[1] == item2[1] and (item[2] == item2[2]) and (item[4] == item2[4]) and (item[5] == item2[5]):
                    try:
                        ret_copy.remove(item)
                    except Exception as e:
                        continue
        for i in ret_copy:
            update_sql = '\n            insert into tb_delivery_gj_django (成交日期,证券代码,证券名称,操作,成交数量,成交均价,成交金额,)\n            '
        print('diff len ', len(ret_copy))

    def bank_account(self):
        if False:
            return 10
        folder_path = os.path.join(os.path.dirname(__file__), 'private')
        os.chdir(folder_path)
        df_list = []
        for file in os.listdir(folder_path):
            if re.search('2', file.decode('gbk')):
                df = pd.read_table(file, encoding='gbk')
                df_list.append(df)
        total_df = pd.concat(df_list)
        del total_df['货币单位']
        del total_df['合同编号']
        del total_df['Unnamed: 8']
        del total_df['银行名称']
        total_df['发生金额'] = map(lambda x, y: x * -1 if y == '证券转银行' else x, total_df['发生金额'], total_df['操作'])
        total_df['委托时间'] = map(lambda x: str(x).zfill(6), total_df['委托时间'])
        total_df['日期'] = map(lambda x, y: str(x) + ' ' + y, total_df['日期'], total_df['委托时间'])
        total_df['日期'] = pd.to_datetime(total_df['日期'], format='%Y%m%d %H%M%S')
        total_df = total_df.set_index('日期')
        df = total_df[total_df['备注'] == '成功[[0000]交易成功]']
        del df['备注']
        del df['委托时间']
        df.to_sql('tb_bank_cash', self.engine, if_exists='replace')

def GJfunc(obj, path, name):
    if False:
        i = 10
        return i + 15
    obj.setpath(path)
    obj.years_gj_each_month_day(filename=name)

def HBfunc(obj, path, name):
    if False:
        for i in range(10):
            print('nop')
    obj.setpath(path)
    obj.merge_data_HuaBao(filename=name)

def main(broker, name):
    if False:
        print('Hello World!')
    '\n    broker: HB GJ\n    name\n    '
    obj = DeliveryOrder()
    base_path = f'private/{datetime.date.today().year}/'
    path = base_path + broker
    if broker == 'GJ':
        GJfunc(obj, path, name)
    elif broker == 'HB':
        HBfunc(obj, path, name)
if __name__ == '__main__':
    fire.Fire(main)