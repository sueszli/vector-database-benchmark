"""
@author:xda
@file:fund_share_update.py
@time:2021/01/20
"""
# 基金份额
import sys
import math
import re

sys.path.append('..')
from configure.settings import DBSelector, config_dict
from common.BaseService import BaseService
import requests
import warnings
import datetime
from fund.LOF_Model import Base, FundBaseInfoModel, ShareModel

warnings.filterwarnings("ignore")
from sqlalchemy.orm import sessionmaker


class Fund(BaseService):
    def __init__(self, first_use=False):
        super(Fund, self).__init__(f'../log/{self.__class__.__name__}.log')
        self.first_use = first_use
        self.engine = self.get_engine()
        self.enableProxy = False

    @staticmethod
    def get_engine():
        return DBSelector().get_engine('db_stock')

    def set_proxy_enable(self):
        self.enableProxy = True
        self.proxy_ip = config_dict('proxy_ip')
        self.set_proxy_param(self.proxy_ip)

    def create_table(self):
        # 初始化数据库连接:
        Base.metadata.create_all(self.engine)  # 创建表结构

    def get_session(self):
        return sessionmaker(bind=self.engine)

    def get(self, url, _json=False, binary=False, retry=5):
        start = 0
        while start < retry:
            try:
                if self.enableProxy:
                    proxy = self.get_proxy()
                else:
                    proxy = None

                response = requests.get(url,
                                        headers=self.headers,
                                        proxies=proxy,
                                        # verify=False
                                        )
            except Exception as e:
                self.logger.error(e)
                start += 1

            else:
                if _json:
                    content = response.json()
                else:
                    content = response.text

                return content

        if start == retry:
            self.logger.error('重试太多')
            return None


class SZFundShare(Fund):
    '''
    doc URL地址
    http://fund.szse.cn/marketdata/fundslist/index.html?catalogId=1000_lf&selectJjlb=LOF&r=1616062435559
    '''

    def __init__(self, first_use=False):
        super(SZFundShare, self).__init__(first_use)
        # self.url = 'http://fund.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=1000_lf&TABKEY=tab1&PAGENO={}&selectJjlb=LOF&random=0.019172632634173903'
        self.all_fund_url = 'http://fund.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=1000_lf&TABKEY=tab1&PAGENO={}&random=0.1292751130110099'
        self.session = requests.Session()
        self.logger.info('start...sz fund')
        self.LAST_TEXT = ''

        if self.first_use:
            self.create_table()

        self.db_session = self.get_session()
        self.sess = self.db_session()
        self.logger.info(f'{self.today} start to crawl....')
        self.set_proxy_enable()

    @property
    def headers(self):
        _header = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh,en;q=0.9,en-US;q=0.8,zh-CN;q=0.7",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Host": "fund.szse.cn",
            "Pragma": "no-cache",
            "Referer": "http://fund.szse.cn/marketdata/fundslist/index.html?catalogId=1000_lf&selectJjlb=ETF",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36",
            "X-Request-Type": "ajax",
            "X-Requested-With": "XMLHttpRequest",
        }
        return _header

    @staticmethod
    def convert(float_str):

        try:
            return_float = float(float_str)
        except Exception as e:
            return_float = None
        return return_float

    def json_parse(self, js_data):
        # TODO 如果当前是周一怎么办？
        date = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        # 手动算的前一天 ？
        if js_data is None:
            raise ValueError('数据为空')

        data = js_data[0].get('data', [])

        if not data:
            self.stop = True
            return None

        for item in data:
            jjlb = item['jjlb']  # 基金类别
            tzlb = item['tzlb']  # 投资类别
            ssrq = item['ssrq']  # 上市日期

            name = self.extract_name(item['jjjcurl'])

            dqgm = self.convert_number(item['dqgm'])  # 当前规模

            glrmc = self.extract_glrmc(item['glrmc'])  # 管理人名称

            code = self.extract_code(item['sys_key'])

            yield (jjlb, tzlb, ssrq, dqgm, glrmc, code, name, date)

    def extract_name(self, name):
        return re.search('<u>(.*?)</u>', name).group(1)

    def extract_code(self, code):
        return re.search('<u>(\d{6})</u>', code).group(1)

    def extract_glrmc(self, glrmc):
        if re.search(('\<a.*?\>(.*?)\</a\>'), glrmc):
            glrmc = re.search(('\<a.*?\>(.*?)\</a\>'), glrmc).group(1).strip()
        return glrmc

    def model_process(self, jjlb, tzlb, ssrq, dqgm, glrmc, code, name, date):

        obj = self.sess.query(FundBaseInfoModel).filter_by(code=code).first()
        # 为的捕获新出的基金，避免遗漏
        if not obj:

            base_info = FundBaseInfoModel(
                code=code,
                name=name,
                category=jjlb,
                invest_type=tzlb,
                manager_name=glrmc,
                issue_date=ssrq,
            )
            try:
                self.sess.add(base_info)
                self.sess.commit()
            except Exception as e:
                print(e)

        # 更新份额表
        if not self.sess.query(ShareModel).filter_by(code=code, date=date).first():
            share_info = ShareModel(
                code=code,
                date=date,
                share=dqgm,
                crawltime=datetime.datetime.now(),
            )
            try:
                self.sess.add(share_info)
                self.sess.commit()
            except Exception as e:
                print(e)

    @staticmethod
    def convert_number(s):
        return float(s.replace(',', ''))

    def run(self):
        page = 1
        self.stop = False
        while not self.stop:
            content = self.get(self.all_fund_url.format(page), _json=True)
            for item in self.json_parse(content):
                self.model_process(*item)

            page += 1


class SHFundShare(Fund):
    '''
    上交所的基金LOF
    '''

    def __init__(self, kind, date, first_use=False):
        super(SHFundShare, self).__init__(first_use)

        self.lof_url = 'http://query.sse.com.cn/commonQuery.do?=&jsonCallBack=jsonpCallback1681&sqlId=COMMON_SSE_FUND_LOF_SCALE_CX_S&pageHelp.pageSize=10000&FILEDATE={}&_=161146986468'
        self.etf_url = 'http://query.sse.com.cn/commonQuery.do?jsonCallBack=jsonpCallback28550&isPagination=true&pageHelp.pageSize=25&pageHelp.pageNo={}&pageHelp.cacheSize=1&sqlId=COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L&STAT_DATE={}&pageHelp.beginPage={}&pageHelp.endPage=30&_=1611473902414'

        # self.today_ = '20210122' # LOF
        if date == 'now':
            last_day = datetime.datetime.now() + datetime.timedelta(days=-1)
            self.today_etf = last_day.strftime('%Y-%m-%d')
            self.today_lof = last_day.strftime('%Y%m%d')
        else:
            print('not now, history data')
            self.today_etf = date
            self.today_lof = date

        # self.today ='2021-01-22' # ETF

        self.ETF_COUNT_PER_PAGE = 25
        self.url_option_dict = {
            'ETF': {'url': self.etf_url, 'date': self.today_etf},  # 2021-03-17 ETF
            'LOF': {'url': self.lof_url, 'date': self.today_lof}  # 20210316 LOF
        }

        self.kind = kind.lower()
        self.session = requests.Session()
        self.logger.info('start...sh fund')
        self.LAST_TEXT = ''

        if self.first_use:
            self.create_table()

        self.db_session = self.get_session()
        self.sess = self.db_session()

    @property
    def headers(self):
        return {
            "Host": "query.sse.com.cn",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": "http://www.sse.com.cn/market/funddata/volumn/lofvolumn/",
        }

    def crawl_lof(self):
        options = self.url_option_dict['LOF']
        date = options.get('date')
        url = options.get('url')
        content = self.get(url.format(date), _json=False)
        js_data = self.jsonp2json(content)
        self.process_lof(js_data)

    def process_lof(self, js_data):
        result = js_data.get('result')
        for item in result:
            code = item['FUND_CODE']
            name = item['FUND_ABBR']
            date = item['TRADE_DATE']

            try:
                share = float(item['INTERNAL_VOL'].replace(',', ''))
            except Exception as e:
                print(e)
                share = None

            self.process_model(code, name, date, share, 'LOF')

    def crawl_etf(self):
        options = self.url_option_dict['ETF']
        date = options.get('date')
        url = options.get('url')
        current_page = 1
        while True:
            content = self.get(url.format(current_page, date, current_page), _json=False)
            js_data = self.jsonp2json(content)
            total_count = js_data.get('pageHelp').get('total')
            print(f'page : {current_page}')
            self.process_etf(js_data)

            max_page = math.ceil(total_count / self.ETF_COUNT_PER_PAGE)  # 每页 10个

            if current_page > max_page:
                break

            current_page += 1

    def process_etf(self, js_data):
        result = js_data.get('result')
        for item in result:
            code = item['SEC_CODE']
            name = item['SEC_NAME']
            date = item['STAT_DATE']
            share = item['TOT_VOL']
            try:
                share = float(share)
            except Exception as e:
                print(e)

            self.process_model(code, name, date, share, 'ETF')

    def run(self):
        'LOF 与 ETF'
        # for type_, options in self.url_option_dict.items():
        if self.kind == 'etf':
            self.logger.info('crawling etf .....')
            self.crawl_etf()

        if self.kind == 'lof':
            self.logger.info('crawling lof .....')
            self.crawl_lof()

    def process_model(self, code, name, date, share, type_):
        obj = self.sess.query(FundBaseInfoModel).filter_by(code=code).first()
        if not obj:
            obj = FundBaseInfoModel(
                code=code,
                name=name,
                category=type_,
                invest_type='',
                manager_name='',
                issue_date=None,
            )
            try:
                self.sess.add(obj)
            except Exception as e:
                print(e)
            else:
                self.sess.commit()
                print(f'插入一条记录{code}，{date}')

        if not self.sess.query(ShareModel).filter_by(code=code, date=date).first():

            share_info = ShareModel(
                code=code,
                date=date,
                share=share,
                crawltime=datetime.datetime.now(),
            )
            try:
                self.sess.add(share_info)
            except Exception as e:
                print(e)
            else:
                print(f'插入一条记录{code}，{date}')
                self.sess.commit()


def patch_fix_missing_data():
    '''
    补充丢失数据
    '''
    days = 90
    for day in range(1, days):
        # etf
        # date=(datetime.datetime.now() + datetime.timedelta(days=-1*day)).strftime('%Y-%m-%d')
        # kind='ETF'

        date = (datetime.datetime.now() + datetime.timedelta(days=-1 * day)).strftime('%Y%m%d')
        kind = 'LOF'

        app = SHFundShare(first_use=False, kind=kind, date=date)
        app.run()


if __name__ == '__main__':
    app = SZFundShare(first_use=False)
    app.run()
    # kind='LOF'
    # date='now'
    # app = SHFundShare(first_use=False,kind=kind,date=date)
    # app.run()
    # patch_fix_missing_data()
