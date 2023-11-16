import sys
import execjs
import fire
import pymongo
from parsel import Selector
sys.path.append('..')
import requests
import datetime
import time
import json
from configure.settings import DBSelector
from common.BaseService import BaseService
import loguru
import re
LOG = loguru.logger

class TTFund(BaseService):
    """'
    爬取天天基金网的排名数据
    """

    def __init__(self, key='股票'):
        if False:
            return 10
        super(TTFund, self).__init__()
        self.ft_dict = {'混合': 'hh', '股票': 'gp', 'qdii': 'qdii', 'lof': 'lof', 'fof': 'fof', '指数': 'zs', '债券': 'zq'}
        self.key = key
        self.date_format = datetime.datetime.now().strftime('%Y_%m_%d')
        self.date_format = '2021_12_15'
        self.doc = self.mongo()['db_stock']['ttjj_rank_{}'.format(self.date_format)]

    @property
    def headers(self):
        if False:
            i = 10
            return i + 15
        return {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh,en;q=0.9,en-US;q=0.8,zh-CN;q=0.7', 'Cache-Control': 'no-cache', 'Cookie': 'AUTH_FUND.EASTMONEY.COM_GSJZ=AUTH*TTJJ*TOKEN; em_hq_fls=js; HAList=a-sh-603707-%u5065%u53CB%u80A1%u4EFD%2Ca-sz-300999-%u91D1%u9F99%u9C7C%2Ca-sh-605338-%u5DF4%u6BD4%u98DF%u54C1%2Ca-sh-600837-%u6D77%u901A%u8BC1%u5238%2Ca-sh-600030-%u4E2D%u4FE1%u8BC1%u5238%2Ca-sz-300059-%u4E1C%u65B9%u8D22%u5BCC%2Cd-hk-06185; EMFUND1=null; EMFUND2=null; EMFUND3=null; EMFUND4=null; qgqp_b_id=956b72f8de13e912a4fc731a7845a6f8; searchbar_code=163407_588080_501077_163406_001665_001664_007049_004433_005827_110011; EMFUND0=null; EMFUND5=02-24%2019%3A30%3A19@%23%24%u5357%u65B9%u6709%u8272%u91D1%u5C5EETF%u8054%u63A5C@%23%24004433; EMFUND6=02-24%2021%3A46%3A42@%23%24%u5357%u65B9%u4E2D%u8BC1%u7533%u4E07%u6709%u8272%u91D1%u5C5EETF@%23%24512400; EMFUND7=02-24%2021%3A58%3A27@%23%24%u6613%u65B9%u8FBE%u84DD%u7B79%u7CBE%u9009%u6DF7%u5408@%23%24005827; EMFUND8=03-05%2015%3A33%3A29@%23%24%u6613%u65B9%u8FBE%u4E2D%u5C0F%u76D8%u6DF7%u5408@%23%24110011; EMFUND9=03-05 23:47:41@#$%u5929%u5F18%u4F59%u989D%u5B9D%u8D27%u5E01@%23%24000198; ASP.NET_SessionId=ntwtbzdkb0vpkzvil2a3h1ip; st_si=44251094035925; st_asi=delete; st_pvi=77351447730109; st_sp=2020-08-16%2015%3A54%3A02; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=3; st_psi=20210309200219784-0-8081344721', 'Host': 'fund.eastmoney.com', 'Pragma': 'no-cache', 'Proxy-Connection': 'keep-alive', 'Referer': 'http://fund.eastmoney.com/data/fundranking.html', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}

    def mongo(self):
        if False:
            print('Hello World!')
        return DBSelector().mongo('qq')

    def rank(self):
        if False:
            i = 10
            return i + 15
        time_interval = 'jnzf'
        self.category_rank(self.key, time_interval)

    def category_rank(self, key, time_interval):
        if False:
            i = 10
            return i + 15
        ft = self.ft_dict[key]
        td_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        td_dt = datetime.datetime.strptime(td_str, '%Y-%m-%d')
        last_dt = td_dt - datetime.timedelta(days=365)
        last_str = datetime.datetime.strftime(last_dt, '%Y-%m-%d')
        rank_url = 'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft={0}&rs=&gs=0&sc={1}&st=desc&sd={2}&ed={3}&qdii=&tabSubtype=,,,,,&pi=1&pn=10000&dx=1'.format(ft, time_interval, last_str, td_str)
        content = self.get(url=rank_url)
        rank_data = self.parse(content)
        rank_list = self.key_remap(rank_data, key)
        self.save_data(rank_list)

    def save_data(self, rank_list):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.doc.insert_many(rank_list)
        except Exception as e:
            print(e)

    def parse(self, content):
        if False:
            return 10
        js_content = execjs.compile(content)
        rank = js_content.eval('rankData')
        return rank.get('datas', [])

    def key_remap(self, rank_data, type_):
        if False:
            print('Hello World!')
        '\n        映射key value\n        '
        colums = ['基金代码', '基金简称', '缩写', '日期', '单位净值', '累计净值', '日增长率(%)', '近1周增幅', '近1月增幅', '近3月增幅', '近6月增幅', '近1年增幅', '近2年增幅', '近3年增幅', '今年来', '成立来', '成立日期', '购买手续费折扣', '自定义', '手续费原价？', '手续费折后？', '布吉岛1', '布吉岛2', '布吉岛3', '布吉岛4']
        return_rank_data = []
        for rank in rank_data:
            rand_dict = {}
            rand_dict['type'] = type_
            rand_dict['crawl_date'] = self.today
            rank_ = rank.split(',')
            for (index, colum) in enumerate(colums):
                rand_dict[colum] = rank_[index]
            return_rank_data.append(rand_dict)
        return return_rank_data

    def turnover_rate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        换手率\n        http://api.fund.eastmoney.com/f10/JJHSL/?callback=jQuery18301549281364854147_1639139836416&fundcode={}&pageindex=1&pagesize=20&_=1639139836475\n        '
        self.DB = self.get_turnover_db()
        for code in self.doc.find({'type': self.key}, {'_id': 0, '基金代码': 1}).sort([('_id', pymongo.ASCENDING)]):
            if self.is_crawl(self.DB, code['基金代码'], 'code'):
                continue
            print('爬取{}'.format(code['基金代码']))
            self.__turnover_rate(code['基金代码'])

    def is_crawl(self, db, code, cond):
        if False:
            while True:
                i = 10
        return True if db.find_one({cond: code}) else False

    def __turnover_rate(self, code):
        if False:
            for i in range(10):
                print('nop')
        url = 'http://api.fund.eastmoney.com/f10/JJHSL/?callback=jQuery18301549281364854147_1639139836416&fundcode={}&pageindex=1&pagesize=100&_=1639139836475'.format(code)
        ret_txt = self.get(url, _json=False)
        self.__parse_turnover_data(ret_txt, code)

    def get_turnover_db(self):
        if False:
            i = 10
            return i + 15
        return DBSelector().mongo('qq')['db_stock']['turnover_{}'.format(self.date_format)]

    def __parse_turnover_data(self, jquery_data, code):
        if False:
            while True:
                i = 10
        js_format = jquery_data[jquery_data.find('{'):jquery_data.rfind('}') + 1]
        js_data = json.loads(js_format)
        turnover_rate_dict = {}
        turnover_rate_dict['code'] = code
        turnover_rate_dict['kind'] = self.key
        turnover_rate_dict['turnover_rate'] = js_data['Data']
        turnover_rate_dict['update'] = datetime.datetime.now()
        self.DB.insert(turnover_rate_dict)

    def fund_detail(self, db, code):
        if False:
            for i in range(10):
                print('nop')
        url = 'http://fundf10.eastmoney.com/jbgk_{}.html'.format(code)

        def __get(url, headers, retry=5):
            if False:
                return 10
            start = 0
            while start < retry:
                try:
                    r = requests.get(url=url, headers=headers)
                except Exception as e:
                    print('base class error', e)
                    time.sleep(1)
                    start += 1
                    continue
                else:
                    return r.text
            return None
        headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9', 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh,en;q=0.9,en-US;q=0.8,zh-CN;q=0.7', 'Host': 'fundf10.eastmoney.com', 'Cookie': 'em_hq_fls=js; searchbar_code=005827; qgqp_b_id=98846d680cc781b1e4a70c935431c5c1; intellpositionL=1170.55px; intellpositionT=555px; HAList=a-sz-123030-%u4E5D%u6D32%u8F6C%u503A%2Ca-sz-300776-%u5E1D%u5C14%u6FC0%u5149%2Ca-sz-300130-%u65B0%u56FD%u90FD%2Ca-sz-300473-%u5FB7%u5C14%u80A1%u4EFD%2Ca-sz-300059-%u4E1C%u65B9%u8D22%u5BCC%2Ca-sz-000411-%u82F1%u7279%u96C6%u56E2%2Ca-sz-300587-%u5929%u94C1%u80A1%u4EFD%2Ca-sz-000060-%u4E2D%u91D1%u5CAD%u5357%2Ca-sz-002707-%u4F17%u4FE1%u65C5%u6E38%2Ca-sh-605080-%u6D59%u5927%u81EA%u7136%2Ca-sz-001201-%u4E1C%u745E%u80A1%u4EFD%2Ca-sz-300981-%u4E2D%u7EA2%u533B%u7597; em-quote-version=topspeed; st_si=90568564737268; st_asi=delete; ASP.NET_SessionId=otnhaxvqrwnmj4nuorygjua4; EMFUND0=11-29%2015%3A40%3A32@%23%24%u5DE5%u94F6%u4E0A%u8BC1%u592E%u4F01ETF@%23%24510060; EMFUND1=12-11%2000%3A51%3A58@%23%24%u524D%u6D77%u5F00%u6E90%u65B0%u7ECF%u6D4E%u6DF7%u5408A@%23%24000689; EMFUND2=12-11%2000%3A57%3A17@%23%24%u4E2D%u4FE1%u5EFA%u6295%u667A%u4FE1%u7269%u8054%u7F51A@%23%24001809; EMFUND3=12-11%2000%3A56%3A12@%23%24%u9E4F%u534E%u4E2D%u8BC1A%u80A1%u8D44%u6E90%u4EA7%u4E1A%u6307%u6570%28LOF%29A@%23%24160620; EMFUND4=12-11%2000%3A47%3A36@%23%24%u4E2D%u4FE1%u4FDD%u8BDA%u7A33%u9E3FA@%23%24006011; EMFUND5=12-11%2000%3A54%3A13@%23%24%u878D%u901A%u6DF1%u8BC1100%u6307%u6570A@%23%24161604; EMFUND6=12-11%2000%3A55%3A27@%23%24%u56FD%u6CF0%u7EB3%u65AF%u8FBE%u514B100%u6307%u6570@%23%24160213; EMFUND7=12-15%2023%3A05%3A04@%23%24%u534E%u5546%u65B0%u5174%u6D3B%u529B%u6DF7%u5408@%23%24001933; EMFUND8=12-15%2023%3A14%3A53@%23%24%u91D1%u4FE1%u6C11%u5174%u503A%u5238A@%23%24004400; EMFUND9=12-15 23:15:15@#$%u5929%u5F18%u4E2D%u8BC1%u5149%u4F0F%u4EA7%u4E1A%u6307%u6570A@%23%24011102; st_pvi=77351447730109; st_sp=2020-08-16%2015%3A54%3A02; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=10; st_psi=20211215231519394-112200305283-4710014236', 'Referer': 'http://fund.eastmoney.com/', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'}
        content = __get(url, headers)
        (built_date, scale) = self.parse_detail_info(content)
        db.insert_one({'成立日期': built_date, '规模': scale, '基金代码': code, 'type': self.key, 'update': datetime.datetime.now()})

    def parse_detail_info(self, content):
        if False:
            return 10
        resp = Selector(text=content)
        labels = resp.xpath('//div[@class="bs_gl"]/p/label')
        if len(labels) < 5:
            print('解析报错')
            return ('', '')
        built_date = labels[0].xpath('./span/text()').extract_first()
        scale = labels[4].xpath('./span/text()').extract_first()
        scale = scale.strip()
        return (built_date, scale)

    def update_basic_info(self):
        if False:
            while True:
                i = 10
        pass

    def get_basic_db(self):
        if False:
            i = 10
            return i + 15
        return DBSelector().mongo('qq')['db_stock']['ttjj_basic']

    def basic_info(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        基本数据\n        '
        self.basic_DB = self.get_basic_db()
        for code in self.doc.find({'type': self.key}, {'_id': 0, '基金代码': 1}).sort([('_id', pymongo.ASCENDING)]):
            if self.is_crawl(self.basic_DB, code['基金代码'], '基金代码'):
                continue
            LOG.info('爬取{}'.format(code['基金代码']))
            self.fund_detail(self.basic_DB, code['基金代码'])

    def convert_data_type(self):
        if False:
            return 10
        '\n        转换mongodb的字段\n        '
        for item in self.doc.find({}, {'成立来': 1}):
            try:
                p1 = float(item['成立来'])
            except:
                p1 = None
            self.doc.update_one({'_id': item['_id']}, {'$set': {'成立来': p1}})

    def get_fund(self, page):
        if False:
            while True:
                i = 10
        url = 'http://fund.eastmoney.com/Data/Fund_JJJZ_Data.aspx?t=1&lx=1&letter=&gsid=&text=&sort=zdf,desc&page={},200&dt=1640059130666&atfc=&onlySale=0'
        content = self.get(url.format(page), _json=False)
        js_content = execjs.compile(content)
        db = js_content.eval('db')
        fund_list = db.get('datas', [])
        for item in fund_list:
            name = item[1]
            if re.search('定增', name):
                print(name)

    def get_funds(self):
        if False:
            while True:
                i = 10
        for i in range(66):
            self.get_fund(i)
            time.sleep(1)

def main(kind, option):
    if False:
        i = 10
        return i + 15
    _dict = {1: '指数', 2: '股票', 3: '混合', 4: 'qdii', 5: 'lof', 6: 'fof', 7: '债券'}
    app = TTFund(key=_dict.get(kind, '股票'))
    if option == 'basic':
        LOG.info('获取{}排名'.format(_dict.get(kind)))
        app.rank()
    elif option == 'turnover':
        LOG.info('获取换手率')
        app.turnover_rate()
    elif option == 'info':
        LOG.info('获取基本信息')
        app.basic_info()
    else:
        LOG.error('请输入正确参数')
if __name__ == '__main__':
    app = TTFund()
    app.convert_data_type()