import random
import re
import datetime
import demjson
import requests
import time
import sys
sys.path.append('..')
from configure.settings import DBSelector
from common.BaseService import BaseService
from configure.util import send_from_aliyun
import warnings
warnings.filterwarnings('ignore')
now = datetime.datetime.now()
TODAY = now.strftime('%Y-%m-%d')
_time = now.strftime('%H:%M:%S')
if _time < '11:30:00':
    TODAY += 'morning'
elif _time < '14:45:00':
    TODAY += 'noon'
else:
    TODAY += 'close'
NOTIFY_HOUR = 13
MAX_PAGE = 50
try:
    DB = DBSelector()
    conn = DB.get_mysql_conn('db_fund', 'qq')
    cursor = conn.cursor()
except Exception as e:
    print(e)

class TencentFundSpider(BaseService):

    def __init__(self):
        if False:
            return 10
        super(TencentFundSpider, self).__init__(f'../log/{self.__class__.__name__}.log')
        self.create_table()
        self.session = requests.Session()
        self.logger.info('start...qq fund')
        self.LAST_TEXT = ''

    @property
    def headers(self):
        if False:
            return 10
        _headers = {'Connection': 'keep-alive', 'Pragma': 'no-cache', 'Cache-Control': 'no-cache', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36', 'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh,en;q=0.9,en-US;q=0.8'}
        return _headers

    def create_table(self):
        if False:
            i = 10
            return i + 15
        create_table = 'create table if not EXISTS `{}` (`基金代码` varchar(20) PRIMARY KEY,`基金简称` varchar(100),`最新规模-万` float,`实时价格` float,`涨跌幅` float,`成交额-万` float,`净值日期` VARCHAR(10),`单位净值` float,`累计净值` float,`折溢价率` float ,`申购状态` VARCHAR(20),`申赎状态` varchar(20),`基金经理` VARCHAR(200),`成立日期` VARCHAR(20), `管理人名称` VARCHAR(200),`实时估值` INT,`更新时间` VARCHAR(20),`数据源` VARCHAR(20) );'.format(TODAY)
        self.execute(create_table, (), conn, self.logger)

    def crawl_fund_info_by_code_table(self):
        if False:
            print('Hello World!')
        code_list = self.get_fund_code(valid=False)
        for code in code_list:
            self.get_info_by_code(code)

    def get_fund_code(self, valid=True):
        if False:
            return 10
        query_cmd = 'select code from fund_main_code'
        if valid:
            query_cmd = query_cmd + 'where valid=1'
        result = self.execute(query_cmd, (), conn, self.logger)
        code_list = []
        for row in result:
            code_list.append(row[0])
        return code_list

    def convert(self, float_str):
        if False:
            while True:
                i = 10
        try:
            return_float = float(float_str)
        except:
            return_float = None
        return return_float

    def insert_data(self, jjdm, jjjc, zxgm, zxjg, jgzffd, cj_total_amount, jzrq, dwjz, ljjz, zyjl, sgzt, shzt, jjjl, clrq, glrmc):
        if False:
            i = 10
            return i + 15
        update_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_realtime = 1
        zxgm = self.convert(zxgm)
        zxjg = self.convert(zxjg)
        jgzffd = self.convert(jgzffd)
        cj_total_amount = self.convert(cj_total_amount)
        dwjz = self.convert(dwjz)
        ljjz = self.convert(ljjz)
        zyjl = self.convert(zyjl)
        source = '腾讯基金'
        insert_data = 'insert into `{}` VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'.format(TODAY)
        if jjdm is None:
            return
        self.execute(insert_data, (jjdm, jjjc, zxgm, zxjg, jgzffd, cj_total_amount, jzrq, dwjz, ljjz, zyjl, sgzt, shzt, jjjl, clrq, glrmc, is_realtime, update_time, source), conn, self.logger)

    def check_exist(self, code):
        if False:
            return 10
        check_code_exists = 'select count(*) from `{}` WHERE `基金代码`=%s'.format(TODAY)
        cursor.execute(check_code_exists, code)
        ret = cursor.fetchone()
        return ret

    def get(self, url, params, retry=5, js=False):
        if False:
            while True:
                i = 10
        start = 0
        while start < retry:
            try:
                response = self.session.get(url, headers=self.headers, params=params, verify=False)
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

    def crawl(self):
        if False:
            return 10
        '\n        废弃 网页内容不存在了\n        '
        url = 'http://stock.gtimg.cn/data/index.php'
        for p in range(1, MAX_PAGE):
            params = (('appn', 'rank'), ('t', 'ranklof/chr'), ('p', p), ('o', '0'), ('l', '40'), ('v', 'list_data'))
            content = self.get(url, params)
            if content is None:
                continue
            if content == self.LAST_TEXT:
                break
            self.LAST_TEXT = content
            ls_data = re.search('var list_data=(.*?);', content, re.S)
            if ls_data:
                ret = ls_data.group(1)
            else:
                self.logger.error('解析出错')
                continue
            js = demjson.decode(ret)
            query_string = js.get('data')
            time.sleep(5 * random.random())
            for code in query_string.split(','):
                self.get_info_by_code(code)

    def get_info_by_code(self, code):
        if False:
            i = 10
            return i + 15
        code_set = set()
        if code not in code_set:
            code_set.add(code)
        else:
            return
        ret = self.check_exist(code)
        if ret[0] > 0:
            return
        detail_url = 'http://gu.qq.com/{}'
        content = self.get(url=detail_url.format(code), params=None)
        if content is None:
            self.logger.error('请求内容为空')
            return
        self.parse_content_and_save(content)

    def parse_content_and_save(self, content):
        if False:
            i = 10
            return i + 15
        (jjdm, jjjc, zxgm, zxjg, jgzffd, cj_total_amount, jzrq, dwjz, ljjz, zyjl, sgzt, shzt, jjjl, clrq, glrmc) = self.parse_html(content)
        self.insert_data(jjdm, jjjc, zxgm, zxjg, jgzffd, cj_total_amount, jzrq, dwjz, ljjz, zyjl, sgzt, shzt, jjjl, clrq, glrmc)

    def parse_html(self, content):
        if False:
            return 10
        search_str = re.search('<script>SSR\\["hqpanel"\\]=(.*?)</script>', content)
        if search_str:
            s = search_str.group(1)
            js_ = demjson.decode(s)
            sub_js = js_.get('data').get('data').get('data')
            zxjg = sub_js.get('zxjg')
            jgzffd = sub_js.get('jgzffd')
            cj_total_amount = sub_js.get('cj_total_amount')
            zyjl = float(sub_js.get('zyjl', 0)) * 100
            info = js_.get('data').get('data').get('info')
            jjdm = info.get('jjdm')
            jjjc = info.get('jjjc')
            zxgm = info.get('zxgm')
            dwjz = info.get('dwjz')
            ljjz = info.get('ljjz')
            sgzt = info.get('sgzt')
            shzt = info.get('shzt')
            jjjl = info.get('jjjl')
            clrq = info.get('clrq')
            glrmc = info.get('glrmc')
            jzrq = info.get('jzrq')
            return (jjdm, jjjc, zxgm, zxjg, jgzffd, cj_total_amount, jzrq, dwjz, ljjz, zyjl, sgzt, shzt, jjjl, clrq, glrmc)
        else:
            return [None] * 15

    def change_table_field(self, table):
        if False:
            i = 10
            return i + 15
        add_column1 = 'alter table `{}` add column `实时净值` float'.format(table)
        add_column2 = 'alter table `{}` add column `溢价率` float'.format(table)
        self.execute(add_column1, (), conn, self.logger)
        self.execute(add_column2, (), conn, self.logger)

    def get_fund_info(self, table):
        if False:
            print('Hello World!')
        query = 'select `基金代码`,`基金简称`,`实时价格` from `{}`'.format(table)
        return self.execute(query, (), conn, self.logger)

    def udpate_db(self, table, jz, yjl, is_realtime, code):
        if False:
            print('Hello World!')
        update_sql = 'update `{}` set `实时净值`= %s,`溢价率`=%s ,`实时估值`=%s where  `基金代码`=%s'.format(table)
        self.execute(update_sql, (jz, yjl, is_realtime, code), conn, self.logger)

    def update_netvalue(self):
        if False:
            print('Hello World!')
        '\n        更新净值\n        :param table:\n        :return:\n        '
        table = TODAY
        self.change_table_field(table)
        all_fund_info = self.get_fund_info(table)
        for item in all_fund_info:
            (jz, yjl, is_realtime, code) = self.get_netvalue(table, item)
            self.udpate_db(table, jz, yjl, is_realtime, code)
        self.logger.info('更新成功')
        self.notice_me(TODAY)

    def get_netvalue(self, table, item):
        if False:
            print('Hello World!')
        code = item[0]
        is_realtime = 1
        realtime_price = item[2]
        url = 'http://web.ifzq.gtimg.cn/fund/newfund/fundSsgz/getSsgz?app=web&symbol=jj{}'
        js = self.get(url=url.format(code), params=None, js=True)
        data = js.get('data')
        if data:
            try:
                data_list = data.get('data')
            except Exception as e:
                self.logger.error(e)
                jz = None
                yjl = None
            else:
                last_one = data_list[-1]
                jz = last_one[1]
                if js is None or realtime_price is None:
                    yjl = 0
                else:
                    yjl = round((realtime_price - jz) / jz * 100, 2)
        else:
            is_realtime = 0
            (yjl, jz) = self.get_fund(table, code)
        return (jz, yjl, is_realtime, code)

    def get_fund(self, table, code):
        if False:
            i = 10
            return i + 15
        query = f'select `折溢价率`,`单位净值` from `{table}` where `基金代码`=%s'
        cursor.execute(query, code)
        ret = cursor.fetchone()
        (yjl, jz) = (ret[0], ret[1])
        yjl = round(yjl, 3)
        jz = round(jz, 3)
        return (yjl, jz)

    def query_fund_data(self, today, order):
        if False:
            i = 10
            return i + 15
        query_sql = "select `基金代码`,`基金简称`,`实时价格`,`实时净值`,`溢价率`,`净值日期` from `{}` where `申购状态`='开放' and `申赎状态`='开放' and `基金简称` not like '%%债%%' and `溢价率` is not null and !(`实时价格`=1 and `涨跌幅`=0 and `成交额-万`=0) order by `溢价率` {} limit 10".format(today, order)
        return self.execute(query_sql, (), conn, self.logger)

    def html_formator(self, ret, html):
        if False:
            print('Hello World!')
        for row in ret:
            html += f"<tr><td>{row[0]}</td><td>{row[1].replace('(LOF)', '')}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td><td>{row[5]}</td></tr>"
        html += '</table></div>'
        return html

    def combine_html(self, html, today):
        if False:
            for i in range(10):
                print('nop')
        body = '<div><table border="1"><tr><th>基金代码</th><th>基金简称</th><th>实时价格</th><th>实时净值</th><th>溢价率</th><th>净值日期</th></tr>'
        html += body
        result_asc = self.query_fund_data(today, 'asc')
        if self.check_content(result_asc):
            html = self.html_formator(result_asc, html)
        html += body
        result_desc = self.query_fund_data(today, 'desc')
        if self.check_content(result_desc):
            html = self.html_formator(result_desc, html)
        return html

    def check_content(self, content):
        if False:
            return 10
        if content is None:
            self.logger.error('获取内容为空')
            return False
        else:
            return True

    def notice_me(self, today):
        if False:
            while True:
                i = 10
        now = datetime.datetime.now()
        if now.hour > NOTIFY_HOUR:
            title = f'{today} 基金折溢价'
            html = ''
            html = self.combine_html(html, TODAY)
            try:
                send_from_aliyun(title, html, types='html')
            except Exception as e:
                self.logger.error(e)
                self.logger.info('发送失败')
            else:
                self.logger.info('发送成功')
if __name__ == '__main__':
    now = datetime.datetime.now()
    TODAY = now.strftime('%Y-%m-%d')
    _time = now.strftime('%H:%M:%S')
    if _time < '11:30:00':
        TODAY += 'morning'
    elif _time < '14:45:00':
        TODAY += 'noon'
    else:
        TODAY += 'close'
    app = TencentFundSpider()
    app.crawl_fund_info_by_code_table()
    app.update_netvalue()