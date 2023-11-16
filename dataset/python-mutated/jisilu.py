__author__ = 'Rocky'
'\nhttp://30daydo.com\nContact: weigesysu@qq.com\n'
import sys
sys.path.append('..')
import re
import time
import datetime
import pandas as pd
from configure.settings import DBSelector, config
from configure.util import send_from_aliyun
from sqlalchemy import VARCHAR
from common.BaseService import BaseService
from datahub.jsl_login import login

class Jisilu(BaseService):

    def __init__(self, check_holiday=False, remote='qq'):
        if False:
            while True:
                i = 10
        super(Jisilu, self).__init__(logfile='log/jisilu.log')
        if check_holiday:
            self.check_holiday()
        self.date = datetime.datetime.now().strftime('%Y-%m-%d')
        self.timestamp = int(time.time() * 1000)
        self.url = 'https://www.jisilu.cn/data/cbnew/cb_list_new/?___jsl=LST___t={}'.format(self.timestamp)
        self.pre_release_url = 'https://www.jisilu.cn/data/cbnew/pre_list/?___jsl=LST___t={}'.format(self.timestamp)
        self.remote = remote
        self.DB = DBSelector()
        self.get_session()

    @property
    def headers(self):
        if False:
            for i in range(10):
                print('nop')
        _header = {'Host': 'www.jisilu.cn', 'Connection': 'keep-alive', 'Pragma': 'no-cache', 'Cache-Control': 'no-cache', 'Accept': 'application/json,text/javascript,*/*;q=0.01', 'Origin': 'https://www.jisilu.cn', 'X-Requested-With': 'XMLHttpRequest', 'User-Agent': 'Mozilla/5.0(WindowsNT6.1;WOW64)AppleWebKit/537.36(KHTML,likeGecko)Chrome/67.0.3396.99Safari/537.36', 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8', 'Referer': 'https://www.jisilu.cn/login/', 'Accept-Encoding': 'gzip,deflate,br', 'Accept-Language': 'zh,en;q=0.9,en-US;q=0.8'}
        return _header

    def check_holiday(self):
        if False:
            return 10
        if self.is_weekday():
            self.logger.info('Start')
        else:
            self.logger.info('Holidy')
            exit(0)

    def get_session(self):
        if False:
            for i in range(10):
                print('nop')
        self.session = login(config['jsl_monitor']['JSL_USER'], config['jsl_monitor']['JSL_PASSWORD'])

    def download(self, url, data, retry=5):
        if False:
            for i in range(10):
                print('nop')
        for i in range(retry):
            try:
                r = self.session.post(url, headers=self.headers, data=data)
                if not r.text or r.status_code != 200:
                    continue
                else:
                    return r
            except Exception as e:
                self.logger.info(e)
                self.notify(title=f'下载失败 {self.__class__}')
                continue
        return None

    def daily_update(self, adjust_no_use=True):
        if False:
            return 10
        post_data = {'fprice': None, 'tprice': None, 'curr_iss_amt': None, 'volume': None, 'svolume': None, 'premium_rt': None, 'ytm_rt': None, 'rating_cd': None, 'is_search': 'N', 'btype': 'C', 'listed': 'Y', 'qflag': 'N', 'sw_cd': None, 'bond_ids': None, 'rp': 50}
        js = self.download(self.url, data=post_data)
        if not js:
            return None
        ret = js.json()
        bond_list = ret.get('rows', {})
        df = self.data_parse(bond_list, adjust_no_use)
        self.store_mysql(df)

    def data_parse(self, bond_list, adjust_no_use):
        if False:
            for i in range(10):
                print('nop')
        cell_list = []
        for item in bond_list:
            cell_list.append(pd.Series(item.get('cell')))
        df = pd.DataFrame(cell_list)
        if adjust_no_use:
            df['price'] = df['price'].astype('float64')
            df['convert_price'] = df['convert_price'].astype('float64')
            df['premium_rt'] = df['premium_rt'].astype('float64')
            df['force_redeem_price'] = df['force_redeem_price'].astype('float64')
            rename_columns = {'bond_id': '可转债代码', 'bond_nm': '可转债名称', 'price': '可转债价格', 'stock_nm': '正股名称', 'stock_id': '正股代码', 'sprice': '正股现价', 'sincrease_rt': '正股涨跌幅', 'convert_price': '最新转股价', 'premium_rt': '溢价率', 'increase_rt': '可转债涨幅', 'convert_value': '转股价值', 'dblow': '双低', 'put_convert_price': '回售触发价', 'convert_dt': '转股起始日', 'maturity_dt': '到期时间', 'volume': '成交额(万元)', 'force_redeem_price': '强赎价格', 'year_left': '剩余时间', 'rating_cd': '评级', 'turnover_rt': '换手率', 'convert_price_tips': '下修提示', 'adj_cnt': '提出下调次数', 'svolume': '正股成交量', 'convert_amt_ratio': '转债剩余占总市值比', 'curr_iss_amt': '剩余规模', 'orig_iss_amt': '发行规模', 'option_tip': '期权价值', 'bond_nm_tip': '强赎提示', 'redeem_dt': '强赎日期', 'list_dt': '上市日期', 'ytm_rt': '到期收益率', 'redeem_icon': '强赎标志', 'margin_flg': '是否两融标的', 'adj_scnt': '下修成功次数', 'convert_cd_tip': '转股日期提示', 'ref_yield_info': '参考YTM'}
            df = df.rename(columns=rename_columns)
            df = df[list(rename_columns.values())]
            df['更新日期'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        df = df.set_index('可转债代码', drop=True)
        return df

    def to_excel(self, df):
        if False:
            print('Hello World!')
        try:
            df.to_excel(f'jisilu_{self.date}.xlsx', encoding='utf8')
        except Exception as e:
            print(e)

    def store_mysql(self, df):
        if False:
            i = 10
            return i + 15
        TABLE_DICT = {'qq': {'fix_db': 'db_stock', 'daily_db': 'db_jisilu'}, 'ptrade': {'fix_db': 'ptrade', 'daily_db': 'db_jisilu_end'}}
        try:
            engine = self.DB.get_engine(TABLE_DICT.get(self.remote).get('daily_db'), self.remote)
            df.to_sql('tb_jsl_{}'.format(self.date), engine, if_exists='replace', dtype={'可转债代码': VARCHAR(10)})
            engine = self.DB.get_engine(TABLE_DICT.get(self.remote).get('fix_db'), self.remote)
            df.to_sql('tb_bond_jisilu', engine, if_exists='replace', dtype={'可转债代码': VARCHAR(10)})
        except Exception as e:
            self.logger.info(e)
            send_from_aliyun(title='jisilu可转债', content='写入数据库出错')

    def init_release_table(self, conn):
        if False:
            while True:
                i = 10
        creat_table = '\n        create table if not exists tb_bond_release (\n        可转债代码 varchar(10),\n        可转债名称 varchar(10),\n        集思录建议 varchar(500),\n        包销比例 float(6,3),\n        中签率 float(6,3),\n        上市日期 varchar(20),\n        申购户数（万户） int,\n        单账户中签（顶格） float(6,3),\n        股东配售率 float(6,3),\n        评级 varchar(8),\n        现价比转股价 float(6,3),\n        抓取时间 datetime\n        );\n        '
        self.execute(creat_table, (), conn)

    def get_conn(self):
        if False:
            for i in range(10):
                print('nop')
        return self.DB.get_mysql_conn('db_stock', self.remote)

    def release_data(self):
        if False:
            while True:
                i = 10
        conn = self.get_conn()
        self.init_release_table(conn)
        post_data = {'cb_type_Y': 'Y', 'progress': '', 'rp': 22}
        r = self.download(url=self.pre_release_url, data=post_data)
        js_data = r.json()
        rows = js_data.get('rows')
        self.save_release_data(rows, conn)

    def save_release_data(self, rows, conn):
        if False:
            while True:
                i = 10
        for items in rows:
            item = items.get('cell')
            single_draw = item.get('single_draw')
            if single_draw:
                jsl_advise_text = item.get('jsl_advise_text')
                underwriter_rt = self.convert_float(item.get('underwriter_rt'))
                bond_nm = item.get('bond_nm')
                lucky_draw_rt = self.convert_float(item.get('lucky_draw_rt'))
                if lucky_draw_rt:
                    lucky_draw_rt = lucky_draw_rt * 100
                list_date = item.get('list_date')
                valid_apply = self.convert_float(item.get('valid_apply'))
                single_draw = self.convert_float(item.get('single_draw'))
                ration_rt = self.convert_float(item.get('ration_rt'))
                rating_cd = item.get('rating_cd')
                bond_id = item.get('bond_id')
                pma_rt = self.convert_float(item.get('pma_rt'))
                update_time = datetime.datetime.now()
                if self.check_bond_exist(bond_id, conn):
                    if self.check_update(bond_id, conn):
                        update_data = (underwriter_rt, list_date, update_time, bond_id)
                        self.update_release_data(update_data, conn)
                    else:
                        continue
                else:
                    insert_data_tuple = (bond_id, bond_nm, jsl_advise_text, underwriter_rt, lucky_draw_rt, list_date, valid_apply, single_draw, ration_rt, rating_cd, pma_rt, update_time)
                    self.insert_release_data(insert_data_tuple, conn)

    def check_update(self, bond_id, conn):
        if False:
            i = 10
            return i + 15
        check_update = 'select * from tb_bond_release where 可转债代码=%s and 包销比例 is null'
        return self.execute(check_update, bond_id, conn)

    def update_release_data(self, update_data, conn):
        if False:
            print('Hello World!')
        '更新发布数据'
        update_sql = 'update tb_bond_release set 包销比例=%s , 上市日期=%s ,抓取时间=%s where 可转债代码 = %s'
        self.execute(update_sql, update_data, conn)

    def insert_release_data(self, data, conn):
        if False:
            for i in range(10):
                print('nop')
        '插入发布数据'
        insert_sql = 'insert into tb_bond_release (可转债代码 , 可转债名称 , 集思录建议 , 包销比例 , 中签率 ,上市日期 ,申购户数（万户）, 单账户中签（顶格）, 股东配售率 ,评级 ,  现价比转股价,抓取时间) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        self.execute(insert_sql, data, conn)

    def check_bond_exist(self, bond_id, conn):
        if False:
            print('Hello World!')
        '\n        判断债券是否存在\n        '
        check_exist = 'select * from tb_bond_release where 可转债代码=%s'
        return self.execute(check_exist, bond_id, conn)

    def execute(self, cmd, data, conn):
        if False:
            while True:
                i = 10
        cursor = conn.cursor()
        if not isinstance(data, tuple):
            data = (data,)
        try:
            cursor.execute(cmd, data)
        except Exception as e:
            conn.rollback()
            self.logger.error('执行数据库错误 {}'.format(e))
            ret = None
        else:
            ret = cursor.fetchall()
            conn.commit()
        return ret

    def convert_float(self, x):
        if False:
            i = 10
            return i + 15
        if not x:
            return None
        if '%' in x:
            ration = 100
        else:
            ration = 1
        x = re.sub('%', '', x)
        try:
            ret = float(x) * ration
        except Exception as e:
            self.logger.error('转换失败{}'.format(e))
            ret = None
        return ret

def main():
    if False:
        for i in range(10):
            print('nop')
    obj = Jisilu(check_holiday=False, remote='qq')
    obj.daily_update()
if __name__ == '__main__':
    main()