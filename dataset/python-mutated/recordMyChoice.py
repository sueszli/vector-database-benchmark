__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import datetime
import os
import xlrd
import time
from xlutils.copy import copy
import tushare as ts
from configure.settings import get_mysql_conn
import codecs
from configure.settings import LLogger
logger = LLogger('recordMyChoice.log')

class Prediction_rate:

    def __init__(self):
        if False:
            print('Hello World!')
        self.today_stock = ts.get_today_all()
        now = datetime.datetime.now()
        self.today = now.strftime('%Y-%m-%d')
        self.path = os.path.join(os.getcwd(), 'data')
        self.filename = os.path.join(self.path, 'recordMyChoice.xls')

    def stock_pool(self, stock_list):
        if False:
            while True:
                i = 10
        pass

    def first_record(self, stockID):
        if False:
            i = 10
            return i + 15
        wb = xlrd.open_workbook(self.filename)
        table = wb.sheets()[0]
        nrow = table.nrows
        ncol = table.ncols
        print('%d*%d' % (nrow, ncol))
        row_start = nrow
        wb_copy = copy(wb)
        sheet = wb_copy.get_sheet(0)
        mystock = self.today_stock[self.today_stock['code'] == stockID]
        name = mystock['name'].values[0]
        in_price = mystock['trade'].values[0]
        current_price = in_price
        profit = 0.0
        content = [self.today, stockID, name, in_price, current_price, profit]
        for i in range(len(content)):
            sheet.write(row_start, i, content[i])
        row_start = row_start + 1
        wb_copy.save(self.filename)

    def update(self):
        if False:
            print('Hello World!')
        pass
'\n持股信息保存到Mysql数据库, 更新，删除\n'

class StockRecord:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.conn = get_mysql_conn('db_stock', local=True)
        self.cur = self.conn.cursor()
        self.table_name = 'tb_profit'
        self.today = datetime.datetime.now().strftime('%Y-%m-%d')

    def holding_stock_sql(self):
        if False:
            i = 10
            return i + 15
        path = os.path.join(os.path.dirname(__file__), 'data', 'mystock.csv')
        if not os.path.exists(path):
            return
        create_table_cmd = 'CREATE TABLE IF NOT EXISTS `tb_profit` (`证券代码` CHAR (6),`证券名称` VARCHAR (16), `保本价` FLOAT,`股票余额` INT,`盈亏比例` FLOAT,`盈亏` FLOAT, `市值` FLOAT);'
        try:
            self.cur.execute(create_table_cmd)
            self.conn.commit()
        except Exception as e:
            logger.log(e)
            self.conn.rollback()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        for i in range(1, len(content)):
            (code, name, safe_price, count) = content[i].strip().split(',')[:4]
            print(code, name, safe_price, count)
            insert_cmd = 'INSERT INTO `tb_profit`  (`证券代码`,`证券名称`,`保本价`,`股票余额`) VALUES("%s","%s","%s","%s");' % (code.zfill(6), name, safe_price, count)
            self._exe(insert_cmd)

    def delete(self, content):
        if False:
            return 10
        name = u'证券名称'
        cmd = u'DELETE FROM `{}` WHERE `{}` = "{}"'.format(self.table_name, name, content)
        self._exe(cmd)

    def insert(self, code, name, safe_price, count):
        if False:
            print('Hello World!')
        '\n\n        :param code: 代码\n        :param name: 名称\n        :param safe_price: 保本价\n        :param count: 股票数目\n        :return: None\n        '
        insert_cmd = 'INSERT INTO `tb_profit`  (`证券代码`,`证券名称`,`保本价`,`股票余额`) VALUES("%s","%s","%s","%s");' % (code.zfill(6), name, safe_price, count)
        self._exe(insert_cmd)

    def _exe(self, cmd):
        if False:
            while True:
                i = 10
        try:
            self.cur.execute(cmd)
            self.conn.commit()
        except Exception as e:
            logger.log(e)
            self.conn.rollback()
        return self.cur

    def update_daily(self):
        if False:
            while True:
                i = 10
        add_cols = 'ALTER TABLE `{}` ADD `{}` FLOAT;'.format(self.table_name, self.today)
        self._exe(add_cols)
        api = ts.get_apis()
        cmd = 'SELECT * FROM `{}`'.format(self.table_name)
        cur = self._exe(cmd)
        for i in cur.fetchall():
            (code, name, safe_price, count, profit_ratio, profit, values, current_price, earn) = i[:9]
            df = ts.quotes(code, conn=api)
            current_price = round(float(df['price'].values[0]), 2)
            values = current_price * count
            last_close = df['last_close'].values[0]
            earn = (current_price - last_close) * count
            profit = (current_price - safe_price) * count
            profit_ratio = round(float(current_price - safe_price) / safe_price * 100, 2)
            update_cmd = 'UPDATE {} SET `盈亏比例`={} ,`盈亏`={}, `市值` ={}, `现价` = {},`{}`={} where `证券代码`= {};'.format(self.table_name, profit_ratio, profit, values, current_price, self.today, earn, code)
            self._exe(update_cmd)
        ts.close_apis(api)

    def update_item(self, code, content):
        if False:
            while True:
                i = 10
        cmd = 'UPDATE `{}` SET `保本价`={} where `证券代码`={};'.format(self.table_name, content, code)
        self._exe(cmd)

    def update_sold(self):
        if False:
            i = 10
            return i + 15
        cur = self.conn.cursor()
        tb_name = 'tb_sold_stock'
        cur.execute('select * from {}'.format(tb_name))
        content = cur.fetchall()
        db_daily = get_mysql_conn('db_daily')
        db_cursor = db_daily.cursor()
        stock_table = datetime.datetime.now().strftime('%Y-%m-%d')
        for i in content:
            cmd = "select `trade` from `{}` where `code`='{}'".format(stock_table, i[0])
            print(cmd)
            db_cursor.execute(cmd)
            ret = db_cursor.fetchone()
            sold_price = i[3]
            percentange = round(float(ret[0] - sold_price) / sold_price * 100, 2)
            update_cmd = "update  `{}` set `当前价`={} ,`卖出后涨跌幅`= {} where `代码`='{}'".format(tb_name, ret[0], percentange, i[0])
            print(update_cmd)
            cur.execute(update_cmd)
            self.conn.commit()
if __name__ == '__main__':
    if ts.is_holiday(datetime.datetime.now().strftime('%Y-%m-%d')):
        exit(0)
    obj = StockRecord()
    obj.update_daily()
    obj.update_sold()