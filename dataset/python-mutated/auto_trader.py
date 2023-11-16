import datetime
import logging
import time
import pymongo
import easyquotation
import easytrader
import pandas as pd
from config import PROGRAM_PATH, MONGO_PORT, MONGO_HOST
from configure.settings import DBSelector
SELL = 7
DB = DBSelector()

class AutoTrader:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.today = datetime.date.today().strftime('%Y-%m-%d')
        self.logger = self.llogger('log/auto_trader_{}'.format(self.today))
        self.logger.info('程序启动')
        self.user = easytrader.use('gj_client')
        self.user.prepare('user.json')
        self.yesterday = datetime.datetime.now() + datetime.timedelta(days=-1)
        self.yesterday = self.yesterday.strftime('%Y-%m-%d')

    def get_close_price(self):
        if False:
            for i in range(10):
                print('nop')
        conn = DB.get_mysql_conn('db_jisilu', 'qq')
        cursor = conn.cursor()
        cmd = 'select 可转债代码,可转债价格 from `tb_jsl_{}`'.format(self.yesterday)
        try:
            cursor.execute(cmd)
            result = cursor.fetchall()
        except Exception as e:
            return None
        else:
            d = {}
            for item in result:
                d[item[0]] = item[1]
            return d

    def set_ceiling(self):
        if False:
            while True:
                i = 10
        position = self.get_position()
        code_price = self.get_close_price()
        for each_stock in position:
            try:
                code = each_stock.get('证券代码')
                amount = int(each_stock.get('可用余额', 0))
                if amount <= 0.1:
                    continue
                close_price = code_price.get(code, None)
                buy_price = round(close_price * (1 + SELL * 0.01), 1)
                self.user.sell(code, price=buy_price, amount=amount)
            except Exception as e:
                self.logger.error(e)

    def get_candidates(self):
        if False:
            i = 10
            return i + 15
        stock_candidate_df = pd.read_sql('tb_stock_candidates', con=self.engine)
        stock_candidate_df = stock_candidate_df.sort_values(by='可转债价格')
        return stock_candidate_df

    def get_market_data(self):
        if False:
            return 10
        market_data_df = pd.read_sql('tb_bond_jisilu', con=self.engine)
        return market_data_df

    def get_blacklist(self):
        if False:
            while True:
                i = 10
        black_list_df = pd.read_sql('tb_bond_blacklist', con=self.engine)
        return black_list_df['code'].values

    def morning_start(self, p):
        if False:
            for i in range(10):
                print('nop')
        codes = self.stock_candidates['可转债代码']
        prices = self.stock_candidates['可转债价格']
        code_price_dict = dict(zip(codes, prices))
        count = 0
        while 1:
            count += 1
            logging.info('Looping {}'.format(count))
            for (code, price) in code_price_dict.copy().items():
                if code not in self.blacklist_bond:
                    deal_detail = self.q.stocks(code)
                    close = deal_detail.get(code, {}).get('close')
                    ask = deal_detail.get(code, {}).get('ask1')
                    bid = deal_detail.get(code, {}).get('bid1')
                    current_percent = (ask - close) / close * 100
                    if current_percent <= p:
                        self.logger.info('>>>>代码{}, 当前价格{}, 开盘跌幅{}'.format(code, bid, current_percent))
                        try:
                            print('code {} buy price {}'.format(code, ask))
                            self.user.buy(code, price=ask + 0.1, amount=10)
                        except Exception as e:
                            self.logger.error('>>>>买入{}出错'.format(code))
                            self.logger.error(e)
                        else:
                            del code_price_dict[code]
            if not code_price_dict:
                break
            time.sleep(20)

    def get_position(self):
        if False:
            i = 10
            return i + 15
        "\n        [{'证券代码': '128012', '证券名称': '辉丰转债', '股票余额': 10.0, '可用余额': 10.0,\n        '市价': 97.03299999999999, '冻结数量': 0, '参考盈亏': 118.77, '参考成本价': 85.156,\n        '参考盈亏比例(%)': 13.947000000000001, '市值': 970.33, '买入成本': 85.156, '市场代码': 1,\n        '交易市场': '深圳Ａ股', '股东帐户': '0166448046', '实际数量': 10, 'Unnamed: 15': ''}\n        :return:\n        "
        return self.user.position

    def get_position_df(self):
        if False:
            print('Hello World!')
        position_list = self.get_position()
        df = pd.DataFrame(position_list)
        return df

    def save_position(self):
        if False:
            print('Hello World!')
        self.engine = DB.get_engine('db_position', 'qq')
        df = self.get_position_df()
        try:
            df.to_sql('tb_position_{}'.format(self.today), con=self.engine, if_exists='replace')
        except Exception as e:
            self.logger.error(e)

    def llogger(self, filename):
        if False:
            i = 10
            return i + 15
        logger = logging.getLogger(filename)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - Line:%(lineno)d:-%(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(filename + '.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger

    def end(self):
        if False:
            for i in range(10):
                print('nop')
        self.logger.info('程序退出')
if __name__ == '__main__':
    trader = AutoTrader()
    trader.set_ceiling()