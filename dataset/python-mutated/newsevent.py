"""
新闻事件数据接口 
Created on 2015/02/07
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
from tushare.stock import cons as ct
from tushare.stock import news_vars as nv
import pandas as pd
from datetime import datetime
import lxml.html
from lxml import etree
import re
import json
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request

def get_latest_news(top=None, show_content=False):
    if False:
        print('Hello World!')
    '\n        获取即时财经新闻\n    \n    Parameters\n    --------\n        top:数值，显示最新消息的条数，默认为80条\n        show_content:是否显示新闻内容，默认False\n    \n    Return\n    --------\n        DataFrame\n            classify :新闻类别\n            title :新闻标题\n            time :发布时间\n            url :新闻链接\n            content:新闻内容（在show_content为True的情况下出现）\n    '
    top = ct.PAGE_NUM[2] if top is None else top
    try:
        request = Request(nv.LATEST_URL % (ct.P_TYPE['http'], ct.DOMAINS['sina'], ct.PAGES['lnews'], top, _random()))
        data_str = urlopen(request, timeout=10).read()
        data_str = data_str.decode('GBK')
        data_str = data_str.split('=')[1][:-1]
        data_str = eval(data_str, type('Dummy', (dict,), dict(__getitem__=lambda s, n: n))())
        data_str = json.dumps(data_str)
        data_str = json.loads(data_str)
        data_str = data_str['list']
        data = []
        for r in data_str:
            rt = datetime.fromtimestamp(r['time'])
            rtstr = datetime.strftime(rt, '%m-%d %H:%M')
            arow = [r['channel']['title'], r['title'], rtstr, r['url']]
            if show_content:
                arow.append(latest_content(r['url']))
            data.append(arow)
        df = pd.DataFrame(data, columns=nv.LATEST_COLS_C if show_content else nv.LATEST_COLS)
        return df
    except Exception as er:
        print(str(er))

def latest_content(url):
    if False:
        for i in range(10):
            print('nop')
    '\n        获取即时财经新闻内容\n    Parameter\n    --------\n        url:新闻链接\n    \n    Return\n    --------\n        string:返回新闻的文字内容\n    '
    try:
        html = lxml.html.parse(url)
        res = html.xpath('//div[@id="artibody"]/p')
        if ct.PY3:
            sarr = [etree.tostring(node).decode('utf-8') for node in res]
        else:
            sarr = [etree.tostring(node) for node in res]
        sarr = ''.join(sarr).replace('&#12288;', '')
        html_content = lxml.html.fromstring(sarr)
        content = html_content.text_content()
        return content
    except Exception as er:
        print(str(er))

def get_notices(code=None, date=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    个股信息地雷\n    Parameters\n    --------\n        code:股票代码\n        date:信息公布日期\n    \n    Return\n    --------\n        DataFrame，属性列表：\n        title:信息标题\n        type:信息类型\n        date:公告日期\n        url:信息内容URL\n    '
    if code is None:
        return None
    symbol = 'sh' + code if code[:1] == '6' else 'sz' + code
    url = nv.NOTICE_INFO_URL % (ct.P_TYPE['http'], ct.DOMAINS['vsf'], ct.PAGES['ntinfo'], symbol)
    url = url if date is None else '%s&gg_date=%s' % (url, date)
    html = lxml.html.parse(url)
    res = html.xpath('//table[@class="body_table"]/tbody/tr')
    data = []
    for td in res:
        title = td.xpath('th/a/text()')[0]
        type = td.xpath('td[1]/text()')[0]
        date = td.xpath('td[2]/text()')[0]
        url = '%s%s%s' % (ct.P_TYPE['http'], ct.DOMAINS['vsf'], td.xpath('th/a/@href')[0])
        data.append([title, type, date, url])
    df = pd.DataFrame(data, columns=nv.NOTICE_INFO_CLS)
    return df

def notice_content(url):
    if False:
        i = 10
        return i + 15
    '\n        获取信息地雷内容\n    Parameter\n    --------\n        url:内容链接\n    \n    Return\n    --------\n        string:信息内容\n    '
    try:
        html = lxml.html.parse(url)
        res = html.xpath('//div[@id="content"]/pre/text()')[0]
        return res.strip()
    except Exception as er:
        print(str(er))

def guba_sina(show_content=False):
    if False:
        while True:
            i = 10
    '\n       获取sina财经股吧首页的重点消息\n    Parameter\n    --------\n        show_content:是否显示内容，默认False\n    \n    Return\n    --------\n    DataFrame\n        title, 消息标题\n        content, 消息内容（show_content=True的情况下）\n        ptime, 发布时间\n        rcounts,阅读次数\n    '
    from pandas.io.common import urlopen
    try:
        with urlopen(nv.GUBA_SINA_URL % (ct.P_TYPE['http'], ct.DOMAINS['sina'])) as resp:
            lines = resp.read()
        html = lxml.html.document_fromstring(lines)
        res = html.xpath('//ul[@class="list_05"]/li[not (@class)]')
        heads = html.xpath('//div[@class="tit_04"]')
        data = []
        for head in heads:
            title = head.xpath('a/text()')[0]
            url = head.xpath('a/@href')[0]
            ds = [title]
            ds.extend(_guba_content(url))
            data.append(ds)
        for row in res:
            title = row.xpath('a[2]/text()')[0]
            url = row.xpath('a[2]/@href')[0]
            ds = [title]
            ds.extend(_guba_content(url))
            data.append(ds)
        df = pd.DataFrame(data, columns=nv.GUBA_SINA_COLS)
        df['rcounts'] = df['rcounts'].astype(float)
        return df if show_content is True else df.drop('content', axis=1)
    except Exception as er:
        print(str(er))

def _guba_content(url):
    if False:
        i = 10
        return i + 15
    try:
        html = lxml.html.parse(url)
        res = html.xpath('//div[@class="ilt_p"]/p')
        if ct.PY3:
            sarr = [etree.tostring(node).decode('utf-8') for node in res]
        else:
            sarr = [etree.tostring(node) for node in res]
        sarr = ''.join(sarr).replace('&#12288;', '')
        html_content = lxml.html.fromstring(sarr)
        content = html_content.text_content()
        ptime = html.xpath('//div[@class="fl_left iltp_time"]/span/text()')[0]
        rcounts = html.xpath('//div[@class="fl_right iltp_span"]/span[2]/text()')[0]
        reg = re.compile('\\((.*?)\\)')
        rcounts = reg.findall(rcounts)[0]
        return [content, ptime, rcounts]
    except Exception:
        return ['', '', '0']

def _random(n=16):
    if False:
        i = 10
        return i + 15
    from random import randint
    start = 10 ** (n - 1)
    end = 10 ** n - 1
    return str(randint(start, end))