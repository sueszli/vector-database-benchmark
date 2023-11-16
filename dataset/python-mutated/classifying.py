"""
获取股票分类数据接口 
Created on 2015/02/01
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
from tushare.stock import cons as ct
from tushare.stock import ref_vars as rv
import json
import re
from pandas.util.testing import _network_error_classes
import time
import tushare.stock.fundamental as fd
from tushare.util.netbase import Client
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request

def get_industry_classified(standard='sina'):
    if False:
        return 10
    '\n        获取行业分类数据\n    Parameters\n    ----------\n    standard\n    sina:新浪行业 sw：申万 行业\n    \n    Returns\n    -------\n    DataFrame\n        code :股票代码\n        name :股票名称\n        c_name :行业名称\n    '
    if standard == 'sw':
        df = pd.read_csv(ct.TSDATA_CLASS % (ct.P_TYPE['http'], ct.DOMAINS['oss'], 'industry_sw'), dtype={'code': object})
    else:
        df = pd.read_csv(ct.TSDATA_CLASS % (ct.P_TYPE['http'], ct.DOMAINS['oss'], 'industry'), dtype={'code': object})
    return df

def get_concept_classified():
    if False:
        for i in range(10):
            print('nop')
    '\n        获取概念分类数据\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n        c_name :概念名称\n    '
    df = pd.read_csv(ct.TSDATA_CLASS % (ct.P_TYPE['http'], ct.DOMAINS['oss'], 'concept'), dtype={'code': object})
    return df

def concetps():
    if False:
        print('Hello World!')
    ct._write_head()
    df = _get_type_data(ct.SINA_CONCEPTS_INDEX_URL % (ct.P_TYPE['http'], ct.DOMAINS['sf'], ct.PAGES['cpt']))
    data = []
    for row in df.values:
        rowDf = _get_detail(row[0])
        if rowDf is not None:
            rowDf['c_name'] = row[1]
            data.append(rowDf)
    if len(data) > 0:
        data = pd.concat(data, ignore_index=True)
    data.to_csv('d:\\cpt.csv', index=False)

def get_concepts(src='dfcf'):
    if False:
        i = 10
        return i + 15
    '\n        获取概念板块行情数据\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n        c_name :概念名称\n    '
    clt = Client(ct.ET_CONCEPTS_INDEX_URL % (ct.P_TYPE['http'], ct.DOMAINS['dfcf'], _random(15)), ref='')
    content = clt.gvalue()
    content = content.decode('utf-8') if ct.PY3 else content
    js = json.loads(content)
    data = []
    for row in js:
        cols = row.split(',')
        cs = cols[6].split('|')
        arr = [cols[2], cols[3], cs[0], cs[2], cols[7], cols[9]]
        data.append(arr)
    df = pd.DataFrame(data, columns=['concept', 'change', 'up', 'down', 'top_code', 'top_name'])
    return df

def get_area_classified():
    if False:
        for i in range(10):
            print('nop')
    '\n        获取地域分类数据\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n        area :地域名称\n    '
    df = fd.get_stock_basics()
    df = df[['name', 'area']]
    df.reset_index(inplace=True)
    df = df.sort_values('area').reset_index(drop=True)
    return df

def get_gem_classified():
    if False:
        print('Hello World!')
    '\n        获取创业板股票\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n    '
    df = fd.get_stock_basics()
    df.reset_index(inplace=True)
    df = df[ct.FOR_CLASSIFY_COLS]
    df = df.ix[df.code.str[0] == '3']
    df = df.sort_values('code').reset_index(drop=True)
    return df

def get_sme_classified():
    if False:
        i = 10
        return i + 15
    '\n        获取中小板股票\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n    '
    df = fd.get_stock_basics()
    df.reset_index(inplace=True)
    df = df[ct.FOR_CLASSIFY_COLS]
    df = df.ix[df.code.str[0:3] == '002']
    df = df.sort_values('code').reset_index(drop=True)
    return df

def get_st_classified():
    if False:
        print('Hello World!')
    '\n        获取风险警示板股票\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n    '
    df = fd.get_stock_basics()
    df.reset_index(inplace=True)
    df = df[ct.FOR_CLASSIFY_COLS]
    df = df.ix[df.name.str.contains('ST')]
    df = df.sort_values('code').reset_index(drop=True)
    return df

def _get_detail(tag, retry_count=3, pause=0.001):
    if False:
        print('Hello World!')
    dfc = pd.DataFrame()
    p = 0
    num_limit = 100
    while True:
        p = p + 1
        for _ in range(retry_count):
            time.sleep(pause)
            try:
                ct._write_console()
                request = Request(ct.SINA_DATA_DETAIL_URL % (ct.P_TYPE['http'], ct.DOMAINS['vsf'], ct.PAGES['jv'], p, tag))
                text = urlopen(request, timeout=10).read()
                text = text.decode('gbk')
            except _network_error_classes:
                pass
            else:
                break
        reg = re.compile('\\,(.*?)\\:')
        text = reg.sub(',"\\1":', text)
        text = text.replace('"{symbol', '{"symbol')
        text = text.replace('{symbol', '{"symbol"')
        jstr = json.dumps(text)
        js = json.loads(jstr)
        df = pd.DataFrame(pd.read_json(js, dtype={'code': object}), columns=ct.THE_FIELDS)
        df = df[['code', 'name']]
        dfc = pd.concat([dfc, df])
        if df.shape[0] < num_limit:
            return dfc

def _get_type_data(url):
    if False:
        for i in range(10):
            print('nop')
    try:
        request = Request(url)
        data_str = urlopen(request, timeout=10).read()
        data_str = data_str.decode('GBK')
        data_str = data_str.split('=')[1]
        data_json = json.loads(data_str)
        df = pd.DataFrame([[row.split(',')[0], row.split(',')[1]] for row in data_json.values()], columns=['tag', 'name'])
        return df
    except Exception as er:
        print(str(er))

def get_hs300s():
    if False:
        i = 10
        return i + 15
    '\n    获取沪深300当前成份股及所占权重\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n        date :日期\n        weight:权重\n    '
    try:
        wt = pd.read_excel(ct.HS300_CLASSIFY_URL_FTP % (ct.P_TYPE['http'], ct.DOMAINS['idx'], ct.PAGES['hs300w']), usecols=[0, 4, 5, 8])
        wt.columns = ct.FOR_CLASSIFY_W_COLS
        wt['code'] = wt['code'].map(lambda x: str(x).zfill(6))
        return wt
    except Exception as er:
        print(str(er))

def get_sz50s():
    if False:
        return 10
    '\n    获取上证50成份股\n    Return\n    --------\n    DataFrame\n        date :日期\n        code :股票代码\n        name :股票名称\n    '
    try:
        df = pd.read_excel(ct.SZ_CLASSIFY_URL_FTP % (ct.P_TYPE['http'], ct.DOMAINS['idx'], ct.PAGES['sz50b']), parse_cols=[0, 4, 5])
        df.columns = ct.FOR_CLASSIFY_B_COLS
        df['code'] = df['code'].map(lambda x: str(x).zfill(6))
        return df
    except Exception as er:
        print(str(er))

def get_zz500s():
    if False:
        return 10
    '\n    获取中证500成份股\n    Return\n    --------\n    DataFrame\n        date :日期\n        code :股票代码\n        name :股票名称\n        weight : 权重\n    '
    try:
        wt = pd.read_excel(ct.HS300_CLASSIFY_URL_FTP % (ct.P_TYPE['http'], ct.DOMAINS['idx'], ct.PAGES['zz500wt']), usecols=[0, 4, 5, 8])
        wt.columns = ct.FOR_CLASSIFY_W_COLS
        wt['code'] = wt['code'].map(lambda x: str(x).zfill(6))
        return wt
    except Exception as er:
        print(str(er))

def get_terminated():
    if False:
        for i in range(10):
            print('nop')
    '\n    获取终止上市股票列表\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n        oDate:上市日期\n        tDate:终止上市日期 \n    '
    try:
        ref = ct.SSEQ_CQ_REF_URL % (ct.P_TYPE['http'], ct.DOMAINS['sse'])
        clt = Client(rv.TERMINATED_URL % (ct.P_TYPE['http'], ct.DOMAINS['sseq'], ct.PAGES['ssecq'], _random(5), _random()), ref=ref, cookie=rv.MAR_SH_COOKIESTR)
        lines = clt.gvalue()
        lines = lines.decode('utf-8') if ct.PY3 else lines
        lines = lines[19:-1]
        lines = json.loads(lines)
        df = pd.DataFrame(lines['result'], columns=rv.TERMINATED_T_COLS)
        df.columns = rv.TERMINATED_COLS
        return df
    except Exception as er:
        print(str(er))

def get_suspended():
    if False:
        for i in range(10):
            print('nop')
    '\n    获取暂停上市股票列表\n    Return\n    --------\n    DataFrame\n        code :股票代码\n        name :股票名称\n        oDate:上市日期\n        tDate:终止上市日期 \n    '
    try:
        ref = ct.SSEQ_CQ_REF_URL % (ct.P_TYPE['http'], ct.DOMAINS['sse'])
        clt = Client(rv.SUSPENDED_URL % (ct.P_TYPE['http'], ct.DOMAINS['sseq'], ct.PAGES['ssecq'], _random(5), _random()), ref=ref, cookie=rv.MAR_SH_COOKIESTR)
        lines = clt.gvalue()
        lines = lines.decode('utf-8') if ct.PY3 else lines
        lines = lines[19:-1]
        lines = json.loads(lines)
        df = pd.DataFrame(lines['result'], columns=rv.TERMINATED_T_COLS)
        df.columns = rv.TERMINATED_COLS
        return df
    except Exception as er:
        print(str(er))

def _random(n=13):
    if False:
        return 10
    from random import randint
    start = 10 ** (n - 1)
    end = 10 ** n - 1
    return str(randint(start, end))