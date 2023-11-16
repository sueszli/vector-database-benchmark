"""
龙虎榜数据
Created on 2015年6月10日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
from pandas.compat import StringIO
from tushare.stock import cons as ct
import numpy as np
import time
import json
import re
import lxml.html
from lxml import etree
from tushare.util import dateu as du
from tushare.stock import ref_vars as rv
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request

def top_list(date=None, retry_count=3, pause=0.001):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取每日龙虎榜列表\n    Parameters\n    --------\n    date:string\n                明细数据日期 format：YYYY-MM-DD 如果为空，返回最近一个交易日的数据\n    retry_count : int, 默认 3\n                 如遇网络等问题重复执行的次数 \n    pause : int, 默认 0\n                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n    \n    Return\n    ------\n    DataFrame\n        code：代码\n        name ：名称\n        pchange：涨跌幅     \n        amount：龙虎榜成交额(万)\n        buy：买入额(万)\n        bratio：占总成交比例\n        sell：卖出额(万)\n        sratio ：占总成交比例\n        reason：上榜原因\n        date  ：日期\n    '
    if date is None:
        if du.get_hour() < 18:
            date = du.last_tddate()
        else:
            date = du.today()
    elif du.is_holiday(date):
        return None
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(rv.LHB_URL % (ct.P_TYPE['http'], ct.DOMAINS['em'], date, date))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            text = text.split('_1=')[1]
            text = eval(text, type('Dummy', (dict,), dict(__getitem__=lambda s, n: n))())
            text = json.dumps(text)
            text = json.loads(text)
            df = pd.DataFrame(text['data'], columns=rv.LHB_TMP_COLS)
            df.columns = rv.LHB_COLS
            df = df.fillna(0)
            df = df.replace('', 0)
            df['buy'] = df['buy'].astype(float)
            df['sell'] = df['sell'].astype(float)
            df['amount'] = df['amount'].astype(float)
            df['Turnover'] = df['Turnover'].astype(float)
            df['bratio'] = df['buy'] / df['Turnover']
            df['sratio'] = df['sell'] / df['Turnover']
            df['bratio'] = df['bratio'].map(ct.FORMAT)
            df['sratio'] = df['sratio'].map(ct.FORMAT)
            df['date'] = date
            for col in ['amount', 'buy', 'sell']:
                df[col] = df[col].astype(float)
                df[col] = df[col] / 10000
                df[col] = df[col].map(ct.FORMAT)
            df = df.drop('Turnover', axis=1)
        except Exception as e:
            print(e)
        else:
            return df
    raise IOError(ct.NETWORK_URL_ERROR_MSG)

def cap_tops(days=5, retry_count=3, pause=0.001):
    if False:
        print('Hello World!')
    '\n    获取个股上榜统计数据\n    Parameters\n    --------\n        days:int\n                  天数，统计n天以来上榜次数，默认为5天，其余是10、30、60\n        retry_count : int, 默认 3\n                     如遇网络等问题重复执行的次数 \n        pause : int, 默认 0\n                    重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n    Return\n    ------\n    DataFrame\n        code：代码\n        name：名称\n        count：上榜次数\n        bamount：累积购买额(万)     \n        samount：累积卖出额(万)\n        net：净额(万)\n        bcount：买入席位数\n        scount：卖出席位数\n    '
    if ct._check_lhb_input(days) is True:
        ct._write_head()
        df = _cap_tops(days, pageNo=1, retry_count=retry_count, pause=pause)
        if df is not None:
            df['code'] = df['code'].map(lambda x: str(x).zfill(6))
            df = df.drop_duplicates('code')
        return df

def _cap_tops(last=5, pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    if False:
        while True:
            i = 10
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(rv.LHB_SINA_URL % (ct.P_TYPE['http'], ct.DOMAINS['vsf'], rv.LHB_KINDS[0], ct.PAGES['fd'], last, pageNo))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            html = lxml.html.parse(StringIO(text))
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>' % sarr
            df = pd.read_html(sarr)[0]
            df.columns = rv.LHB_GGTJ_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall('\\d+', nextPage[0])[0]
                return _cap_tops(last, pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        except Exception as e:
            print(e)

def broker_tops(days=5, retry_count=3, pause=0.001):
    if False:
        print('Hello World!')
    '\n    获取营业部上榜统计数据\n    Parameters\n    --------\n    days:int\n              天数，统计n天以来上榜次数，默认为5天，其余是10、30、60\n    retry_count : int, 默认 3\n                 如遇网络等问题重复执行的次数 \n    pause : int, 默认 0\n                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n    Return\n    ---------\n    broker：营业部名称\n    count：上榜次数\n    bamount：累积购买额(万)\n    bcount：买入席位数\n    samount：累积卖出额(万)\n    scount：卖出席位数\n    top3：买入前三股票\n    '
    if ct._check_lhb_input(days) is True:
        ct._write_head()
        df = _broker_tops(days, pageNo=1, retry_count=retry_count, pause=pause)
        return df

def _broker_tops(last=5, pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    if False:
        i = 10
        return i + 15
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(rv.LHB_SINA_URL % (ct.P_TYPE['http'], ct.DOMAINS['vsf'], rv.LHB_KINDS[1], ct.PAGES['fd'], last, pageNo))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            html = lxml.html.parse(StringIO(text))
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>' % sarr
            df = pd.read_html(sarr)[0]
            df.columns = rv.LHB_YYTJ_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall('\\d+', nextPage[0])[0]
                return _broker_tops(last, pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        except Exception as e:
            print(e)

def inst_tops(days=5, retry_count=3, pause=0.001):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取机构席位追踪统计数据\n    Parameters\n    --------\n    days:int\n              天数，统计n天以来上榜次数，默认为5天，其余是10、30、60\n    retry_count : int, 默认 3\n                 如遇网络等问题重复执行的次数 \n    pause : int, 默认 0\n                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n                \n    Return\n    --------\n    code:代码\n    name:名称\n    bamount:累积买入额(万)\n    bcount:买入次数\n    samount:累积卖出额(万)\n    scount:卖出次数\n    net:净额(万)\n    '
    if ct._check_lhb_input(days) is True:
        ct._write_head()
        df = _inst_tops(days, pageNo=1, retry_count=retry_count, pause=pause)
        df['code'] = df['code'].map(lambda x: str(x).zfill(6))
        return df

def _inst_tops(last=5, pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    if False:
        print('Hello World!')
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(rv.LHB_SINA_URL % (ct.P_TYPE['http'], ct.DOMAINS['vsf'], rv.LHB_KINDS[2], ct.PAGES['fd'], last, pageNo))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            html = lxml.html.parse(StringIO(text))
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>' % sarr
            df = pd.read_html(sarr)[0]
            df = df.drop([2, 3], axis=1)
            df.columns = rv.LHB_JGZZ_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall('\\d+', nextPage[0])[0]
                return _inst_tops(last, pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        except Exception as e:
            print(e)

def inst_detail(retry_count=3, pause=0.001):
    if False:
        print('Hello World!')
    '\n    获取最近一个交易日机构席位成交明细统计数据\n    Parameters\n    --------\n    retry_count : int, 默认 3\n                 如遇网络等问题重复执行的次数 \n    pause : int, 默认 0\n                重复请求数据过程中暂停的秒数，防止请求间隔时间太短出现的问题\n                \n    Return\n    ----------\n    code:股票代码\n    name:股票名称     \n    date:交易日期     \n    bamount:机构席位买入额(万)     \n    samount:机构席位卖出额(万)     \n    type:类型\n    '
    ct._write_head()
    df = _inst_detail(pageNo=1, retry_count=retry_count, pause=pause)
    if len(df) > 0:
        df['code'] = df['code'].map(lambda x: str(x).zfill(6))
    return df

def _inst_detail(pageNo=1, retry_count=3, pause=0.001, dataArr=pd.DataFrame()):
    if False:
        for i in range(10):
            print('nop')
    ct._write_console()
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            request = Request(rv.LHB_SINA_URL % (ct.P_TYPE['http'], ct.DOMAINS['vsf'], rv.LHB_KINDS[3], ct.PAGES['fd'], '', pageNo))
            text = urlopen(request, timeout=10).read()
            text = text.decode('GBK')
            html = lxml.html.parse(StringIO(text))
            res = html.xpath('//table[@id="dataTable"]/tr')
            if ct.PY3:
                sarr = [etree.tostring(node).decode('utf-8') for node in res]
            else:
                sarr = [etree.tostring(node) for node in res]
            sarr = ''.join(sarr)
            sarr = '<table>%s</table>' % sarr
            df = pd.read_html(sarr)[0]
            df.columns = rv.LHB_JGMX_COLS
            dataArr = dataArr.append(df, ignore_index=True)
            nextPage = html.xpath('//div[@class="pages"]/a[last()]/@onclick')
            if len(nextPage) > 0:
                pageNo = re.findall('\\d+', nextPage[0])[0]
                return _inst_detail(pageNo, retry_count, pause, dataArr)
            else:
                return dataArr
        except Exception as e:
            print(e)

def _f_rows(x):
    if False:
        for i in range(10):
            print('nop')
    if '%' in x[3]:
        x[11] = x[6]
        for i in range(6, 11):
            x[i] = x[i - 5]
        for i in range(1, 6):
            x[i] = np.NaN
    return x