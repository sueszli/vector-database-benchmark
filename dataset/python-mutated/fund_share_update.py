"""# -*- coding"""
'\n@author:xda\n@file:fund_share_update.py\n@time:2021/01/20\n'
import sys
sys.path.append('..')
from configure.settings import DBSelector
from common.BaseService import BaseService
import requests
import warnings
import datetime
import math
import re
warnings.filterwarnings('ignore')
from sqlalchemy.orm import relationship
from sqlalchemy import Column, INTEGER, VARCHAR, DATE, DateTime, ForeignKey, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class FundBaseInfoModel(Base):
    __tablename__ = 'LOF_BaseInfo'
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    code = Column(VARCHAR(6), comment='基金代码', unique=True)
    name = Column(VARCHAR(40), comment='基金名称')
    category = Column(VARCHAR(8), comment='基金类别')
    invest_type = Column(VARCHAR(6), comment='投资类别')
    manager_name = Column(VARCHAR(48), comment='管理人呢名称')
    issue_date = Column(DATE, comment='上市日期')
    child = relationship('ShareModel')

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'<{self.code}><{self.name}>'

class ShareModel(Base):
    __tablename__ = 'LOF_Share'
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    code = Column(VARCHAR(6), ForeignKey('LOF_BaseInfo.code'), comment='代码')
    date = Column(DATE, comment='份额日期')
    share = Column(FLOAT, comment='份额 单位：万份')
    parent = relationship('FundBaseInfoModel')
    crawltime = Column(DateTime, comment='爬取日期')

class Fund(BaseService):

    def __init__(self, first_use=False):
        if False:
            i = 10
            return i + 15
        super(Fund, self).__init__(f'../log/{self.__class__.__name__}.log')
        self.first_use = first_use
        self.engine = self.get_engine()

    def get_engine(self):
        if False:
            while True:
                i = 10
        return DBSelector().get_engine('db_stock')

    def create_table(self):
        if False:
            return 10
        Base.metadata.create_all(self.engine)

    def get_session(self):
        if False:
            print('Hello World!')
        return sessionmaker(bind=self.engine)

    def get(self, url, retry=5, js=True):
        if False:
            return 10
        start = 0
        while start < retry:
            try:
                response = self.session.get(url, headers=self.headers, verify=False)
            except Exception as e:
                self.logger.error(e)
                start += 1
            else:
                if js:
                    content = response.json()
                else:
                    content = response.text
                return content
        if start == retry:
            self.logger.error('重试太多')
            return None

class SZFundShare(Fund):

    def __init__(self, first_use=False):
        if False:
            print('Hello World!')
        super(SZFundShare, self).__init__(first_use)
        self.all_fund_url = 'http://fund.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=1000_lf&TABKEY=tab1&PAGENO={}&random=0.1292751130110099'
        self.session = requests.Session()
        self.logger.info('start...sz fund')
        self.LAST_TEXT = ''
        if self.first_use:
            self.create_table()
        self.db_session = self.get_session()
        self.sess = self.db_session()
        self.logger.info(f'{self.today} start to crawl....')

    @property
    def headers(self):
        if False:
            print('Hello World!')
        _header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh,en;q=0.9,en-US;q=0.8,zh-CN;q=0.7', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Content-Type': 'application/json', 'Host': 'fund.szse.cn', 'Pragma': 'no-cache', 'Referer': 'http://fund.szse.cn/marketdata/fundslist/index.html?catalogId=1000_lf&selectJjlb=ETF', 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36', 'X-Request-Type': 'ajax', 'X-Requested-With': 'XMLHttpRequest'}
        return _header

    def convert(self, float_str):
        if False:
            print('Hello World!')
        try:
            return_float = float(float_str)
        except:
            return_float = None
        return return_float

    def json_parse(self, js_data):
        if False:
            return 10
        date = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        data = js_data[0].get('data', [])
        if not data:
            self.stop = True
            return None
        for item in data:
            jjlb = item['jjlb']
            tzlb = item['tzlb']
            ssrq = item['ssrq']
            name = self.extract_name(item['jjjcurl'])
            dqgm = self.convert_number(item['dqgm'])
            glrmc = self.extract_glrmc(item['glrmc'])
            code = self.extract_code(item['sys_key'])
            yield (jjlb, tzlb, ssrq, dqgm, glrmc, code, name, date)

    def extract_name(self, name):
        if False:
            while True:
                i = 10
        return re.search('<u>(.*?)</u>', name).group(1)

    def extract_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        return re.search('<u>(\\d{6})</u>', code).group(1)

    def extract_glrmc(self, glrmc):
        if False:
            while True:
                i = 10
        if re.search('\\<a.*?\\>(.*?)\\</a\\>', glrmc):
            glrmc = re.search('\\<a.*?\\>(.*?)\\</a\\>', glrmc).group(1).strip()
        return glrmc

    def model_process(self, jjlb, tzlb, ssrq, dqgm, glrmc, code, name, date):
        if False:
            while True:
                i = 10
        obj = self.sess.query(FundBaseInfoModel).filter_by(code=code).first()
        if not obj:
            base_info = FundBaseInfoModel(code=code, name=name, category=jjlb, invest_type=tzlb, manager_name=glrmc, issue_date=ssrq)
            self.sess.add(base_info)
            self.sess.commit()
        share_info = ShareModel(code=code, date=date, share=dqgm, crawltime=datetime.datetime.now())
        self.sess.add(share_info)
        self.sess.commit()

    def convert_number(self, s):
        if False:
            return 10
        return float(s.replace(',', ''))

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        page = 1
        self.stop = False
        while not self.stop:
            content = self.get(self.all_fund_url.format(page))
            for item in self.json_parse(content):
                self.model_process(*item)
            page += 1

class SHFundShare(Fund):

    def __init__(self, kind, date, first_use=False):
        if False:
            while True:
                i = 10
        super(SHFundShare, self).__init__(first_use)
        self.lof_url = 'http://query.sse.com.cn/commonQuery.do?=&jsonCallBack=jsonpCallback1681&sqlId=COMMON_SSE_FUND_LOF_SCALE_CX_S&pageHelp.pageSize=10000&FILEDATE={}&_=161146986468'
        self.etf_url = 'http://query.sse.com.cn/commonQuery.do?jsonCallBack=jsonpCallback28550&isPagination=true&pageHelp.pageSize=25&pageHelp.pageNo={}&pageHelp.cacheSize=1&sqlId=COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L&STAT_DATE={}&pageHelp.beginPage={}&pageHelp.endPage=30&_=1611473902414'
        if date == 'now':
            self.today_ = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y%m%d')
        else:
            self.today_ = self.today = date
        self.ETF_COUNT_PER_PAGE = 25
        self.url_option_dict = {'ETF': {'url': self.etf_url, 'date': self.today}, 'LOF': {'url': self.lof_url, 'date': self.today_}}
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
        if False:
            for i in range(10):
                print('nop')
        return {'Host': 'query.sse.com.cn', 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0', 'Accept': '*/*', 'Accept-Language': 'en-US,en;q=0.5', 'Accept-Encoding': 'gzip, deflate', 'Connection': 'keep-alive', 'Referer': 'http://www.sse.com.cn/market/funddata/volumn/lofvolumn/'}

    def crawl_lof(self):
        if False:
            return 10
        options = self.url_option_dict['LOF']
        date = options.get('date')
        url = options.get('url')
        content = self.get(url.format(date), js=False)
        js_data = self.jsonp2json(content)
        self.process_lof(js_data)

    def process_lof(self, js_data):
        if False:
            for i in range(10):
                print('nop')
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
        if False:
            for i in range(10):
                print('nop')
        options = self.url_option_dict['ETF']
        date = options.get('date')
        url = options.get('url')
        current_page = 1
        while True:
            content = self.get(url.format(current_page, date, current_page), js=False)
            js_data = self.jsonp2json(content)
            total_count = js_data.get('pageHelp').get('total')
            print(f'page : {current_page}')
            self.process_etf(js_data)
            max_page = math.ceil(total_count / self.ETF_COUNT_PER_PAGE)
            if current_page > max_page:
                break
            current_page += 1

    def process_etf(self, js_data):
        if False:
            print('Hello World!')
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
        if False:
            return 10
        'LOF 与 ETF'
        if self.kind == 'etf':
            self.logger.info('crawling etf .....')
            self.crawl_etf()
        if self.kind == 'lof':
            self.logger.info('crawling lof .....')
            self.crawl_lof()

    def process_model(self, code, name, date, share, type_):
        if False:
            while True:
                i = 10
        obj = self.sess.query(FundBaseInfoModel).filter_by(code=code).first()
        if not obj:
            obj = FundBaseInfoModel(code=code, name=name, category=type_, invest_type='', manager_name='', issue_date=None)
            try:
                self.sess.add(obj)
            except Exception as e:
                print(e)
            else:
                self.sess.commit()
                print(f'插入一条记录{code}，{date}')
        if not self.sess.query(ShareModel).filter_by(code=code, date=date).first():
            share_info = ShareModel(code=code, date=date, share=share, crawltime=datetime.datetime.now())
            try:
                self.sess.add(share_info)
            except Exception as e:
                print(e)
            else:
                print(f'插入一条记录{code}，{date}')
                self.sess.commit()
if __name__ == '__main__':
    app = SZFundShare(first_use=False)
    app.run()
    app = SHFundShare(first_use=False)
    app.run()