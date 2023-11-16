"""
数字货币行情数据
Created on 2017年9月9日
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
"""
import pandas as pd
import traceback
import time
import json
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request
URL = {'hb': {'rt': 'http://api.huobi.com/staticmarket/ticker_%s_json.js', 'kline': 'http://api.huobi.com/staticmarket/%s_kline_%s_json.js?length=%s', 'snapshot': 'http://api.huobi.com/staticmarket/depth_%s_%s.js', 'tick': 'http://api.huobi.com/staticmarket/detail_%s_json.js'}, 'ok': {'rt': 'https://www.okcoin.cn/api/v1/ticker.do?symbol=%s_cny', 'kline': 'https://www.okcoin.cn/api/v1/kline.do?symbol=%s_cny&type=%s&size=%s', 'snapshot': 'https://www.okcoin.cn/api/v1/depth.do?symbol=%s_cny&merge=&size=%s', 'tick': 'https://www.okcoin.cn/api/v1/trades.do?symbol=%s_cny'}, 'chbtc': {'rt': 'http://api.chbtc.com/data/v1/ticker?currency=%s_cny', 'kline': 'http://api.chbtc.com/data/v1/kline?currency=%s_cny&type=%s&size=%s', 'snapshot': 'http://api.chbtc.com/data/v1/depth?currency=%s_cny&size=%s&merge=', 'tick': 'http://api.chbtc.com/data/v1/trades?currency=%s_cny'}}
KTYPES = {'D': {'hb': '100', 'ok': '1day', 'chbtc': '1day'}, 'W': {'hb': '200', 'ok': '1week', 'chbtc': '1week'}, 'M': {'hb': '300', 'ok': '', 'chbtc': ''}, '1MIN': {'hb': '001', 'ok': '1min', 'chbtc': '1min'}, '5MIN': {'hb': '005', 'ok': '5min', 'chbtc': '5min'}, '15MIN': {'hb': '015', 'ok': '15min', 'chbtc': '15min'}, '30MIN': {'hb': '030', 'ok': '30min', 'chbtc': '30min'}, '60MIN': {'hb': '060', 'ok': '1hour', 'chbtc': '1hour'}}

def coins_tick(broker='hb', code='btc'):
    if False:
        return 10
    '\n    实时tick行情\n    params:\n    ---------------\n    broker: hb:火币\n            ok:okCoin\n            chbtc:中国比特币\n    code: hb:btc,ltc\n        ----okcoin---\n        btc_cny：比特币    ltc_cny：莱特币    eth_cny :以太坊     etc_cny :以太经典    bcc_cny :比特现金 \n        ----chbtc----\n        btc_cny:BTC/CNY\n        ltc_cny :LTC/CNY\n        eth_cny :以太币/CNY\n        etc_cny :ETC币/CNY\n        bts_cny :BTS币/CNY\n        eos_cny :EOS币/CNY\n        bcc_cny :BCC币/CNY\n        qtum_cny :量子链/CNY\n        hsr_cny :HSR币/CNY\n    return:json\n    ---------------\n    hb:\n    {\n    "time":"1504713534",\n    "ticker":{\n        "symbol":"btccny",\n        "open":26010.90,\n        "last":28789.00,\n        "low":26000.00,\n        "high":28810.00,\n        "vol":17426.2198,\n        "buy":28750.000000,\n        "sell":28789.000000\n        }\n    }\n    ok:\n    {\n    "date":"1504713864",\n    "ticker":{\n        "buy":"28743.0",\n        "high":"28886.99",\n        "last":"28743.0",\n        "low":"26040.0",\n        "sell":"28745.0",\n        "vol":"20767.734"\n        }\n    }\n    chbtc: \n        {\n         u\'date\': u\'1504794151878\',\n         u\'ticker\': {\n             u\'sell\': u\'28859.56\', \n             u\'buy\': u\'28822.89\', \n             u\'last\': u\'28859.56\', \n             u\'vol\': u\'2702.71\', \n             u\'high\': u\'29132\', \n             u\'low\': u\'27929\'\n         }\n        }\n\n        \n    '
    return _get_data(URL[broker]['rt'] % code)

def coins_bar(broker='hb', code='btc', ktype='D', size='2000'):
    if False:
        while True:
            i = 10
    '\n            获取各类k线数据\n    params:\n    broker:hb,ok,chbtc\n    code:btc,ltc,eth,etc,bcc\n    ktype:D,W,M,1min,5min,15min,30min,60min\n    size:<2000\n    return DataFrame: 日期时间，开盘价，最高价，最低价，收盘价，成交量\n    '
    try:
        js = _get_data(URL[broker]['kline'] % (code, KTYPES[ktype.strip().upper()][broker], size))
        if js is None:
            return js
        if broker == 'chbtc':
            js = js['data']
        df = pd.DataFrame(js, columns=['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL'])
        if broker == 'hb':
            if ktype.strip().upper() in ['D', 'W', 'M']:
                df['DATE'] = df['DATE'].apply(lambda x: x[0:8])
            else:
                df['DATE'] = df['DATE'].apply(lambda x: x[0:12])
        else:
            df['DATE'] = df['DATE'].apply(lambda x: int2time(x / 1000))
        if ktype.strip().upper() in ['D', 'W', 'M']:
            df['DATE'] = df['DATE'].apply(lambda x: str(x)[0:10])
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
    except Exception:
        print(traceback.print_exc())

def coins_snapshot(broker='hb', code='btc', size='5'):
    if False:
        print('Hello World!')
    '\n            获取实时快照数据\n    params:\n    broker:hb,ok,chbtc\n    code:btc,ltc,eth,etc,bcc\n    size:<150\n    return Panel: asks,bids\n    '
    try:
        js = _get_data(URL[broker]['snapshot'] % (code, size))
        if js is None:
            return js
        if broker == 'hb':
            timestr = js['ts']
            timestr = int2time(timestr / 1000)
        if broker == 'ok':
            timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        if broker == 'chbtc':
            timestr = js['timestamp']
            timestr = int2time(timestr)
        asks = pd.DataFrame(js['asks'], columns=['price', 'vol'])
        bids = pd.DataFrame(js['bids'], columns=['price', 'vol'])
        asks['time'] = timestr
        bids['time'] = timestr
        djs = {'asks': asks, 'bids': bids}
        pf = pd.Panel(djs)
        return pf
    except Exception:
        print(traceback.print_exc())

def coins_trade(broker='hb', code='btc'):
    if False:
        i = 10
        return i + 15
    "\n    获取实时交易数据\n    params:\n    -------------\n    broker: hb,ok,chbtc\n    code:btc,ltc,eth,etc,bcc\n    \n    return:\n    ---------------\n    DataFrame\n    'tid':order id\n    'datetime', date time \n    'price' : trade price\n    'amount' : trade amount\n    'type' : buy or sell\n    "
    js = _get_data(URL[broker]['tick'] % code)
    if js is None:
        return js
    if broker == 'hb':
        df = pd.DataFrame(js['trades'])
        df = df[['id', 'ts', 'price', 'amount', 'direction']]
        df['ts'] = df['ts'].apply(lambda x: int2time(x / 1000))
    if broker == 'ok':
        df = pd.DataFrame(js)
        df = df[['tid', 'date_ms', 'price', 'amount', 'type']]
        df['date_ms'] = df['date_ms'].apply(lambda x: int2time(x / 1000))
    if broker == 'chbtc':
        df = pd.DataFrame(js)
        df = df[['tid', 'date', 'price', 'amount', 'type']]
        df['date'] = df['date'].apply(lambda x: int2time(x))
    df.columns = ['tid', 'datetime', 'price', 'amount', 'type']
    return df

def _get_data(url):
    if False:
        print('Hello World!')
    try:
        request = Request(url)
        lines = urlopen(request, timeout=10).read()
        if len(lines) < 50:
            return None
        js = json.loads(lines.decode('GBK'))
        return js
    except Exception:
        print(traceback.print_exc())

def int2time(timestamp):
    if False:
        return 10
    value = time.localtime(timestamp)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', value)
    return dt