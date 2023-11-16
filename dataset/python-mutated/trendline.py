"""
股票技术指标接口
Created on 2018/07/26
@author: Wangzili
@group : **
@contact: 446406177@qq.com

所有指标中参数df为通过get_k_data获取的股票数据
"""
import pandas as pd
import numpy as np
import itertools

def ma(df, n=10):
    if False:
        return 10
    '\n    移动平均线 Moving Average\n    MA（N）=（第1日收盘价+第2日收盘价—+……+第N日收盘价）/N\n    '
    pv = pd.DataFrame()
    pv['date'] = df['date']
    pv['v'] = df.close.rolling(n).mean()
    return pv

def _ma(series, n):
    if False:
        while True:
            i = 10
    '\n    移动平均\n    '
    return series.rolling(n).mean()

def md(df, n=10):
    if False:
        for i in range(10):
            print('nop')
    '\n    移动标准差\n    STD=S（CLOSE,N）=[∑（CLOSE-MA(CLOSE，N)）^2/N]^0.5\n    '
    _md = pd.DataFrame()
    _md['date'] = df.date
    _md['md'] = df.close.rolling(n).std(ddof=0)
    return _md

def _md(series, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    标准差MD\n    '
    return series.rolling(n).std(ddof=0)

def ema(df, n=12):
    if False:
        print('Hello World!')
    '\n    指数平均数指标 Exponential Moving Average\n    今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）\n    EMA(X,N)=[2×X+(N-1)×EMA(ref(X),N]/(N+1)\n    '
    _ema = pd.DataFrame()
    _ema['date'] = df['date']
    _ema['ema'] = df.close.ewm(ignore_na=False, span=n, min_periods=0, adjust=False).mean()
    return _ema

def _ema(series, n):
    if False:
        return 10
    '\n    指数平均数\n    '
    return series.ewm(ignore_na=False, span=n, min_periods=0, adjust=False).mean()

def macd(df, n=12, m=26, k=9):
    if False:
        while True:
            i = 10
    '\n    平滑异同移动平均线(Moving Average Convergence Divergence)\n    今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）\n    DIFF= EMA（N1）- EMA（N2）\n    DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)\n    MACD（BAR）=2×（DIF-DEA）\n    return:\n          osc: MACD bar / OSC 差值柱形图 DIFF - DEM\n          diff: 差离值\n          dea: 讯号线\n    '
    _macd = pd.DataFrame()
    _macd['date'] = df['date']
    _macd['diff'] = _ema(df.close, n) - _ema(df.close, m)
    _macd['dea'] = _ema(_macd['diff'], k)
    _macd['macd'] = _macd['diff'] - _macd['dea']
    return _macd

def kdj(df, n=9):
    if False:
        return 10
    '\n    随机指标KDJ\n    N日RSV=（第N日收盘价-N日内最低价）/（N日内最高价-N日内最低价）×100%\n    当日K值=2/3前1日K值+1/3×当日RSV=SMA（RSV,M1）\n    当日D值=2/3前1日D值+1/3×当日K= SMA（K,M2）\n    当日J值=3 ×当日K值-2×当日D值\n    '
    _kdj = pd.DataFrame()
    _kdj['date'] = df['date']
    rsv = (df.close - df.low.rolling(n).min()) / (df.high.rolling(n).max() - df.low.rolling(n).min()) * 100
    _kdj['k'] = sma(rsv, 3)
    _kdj['d'] = sma(_kdj.k, 3)
    _kdj['j'] = 3 * _kdj.k - 2 * _kdj.d
    return _kdj

def rsi(df, n=6):
    if False:
        print('Hello World!')
    '\n    相对强弱指标（Relative Strength Index，简称RSI\n    LC= REF(CLOSE,1)\n    RSI=SMA(MAX(CLOSE-LC,0),N,1)/SMA(ABS(CLOSE-LC),N1,1)×100\n    SMA（C,N,M）=M/N×今日收盘价+(N-M)/N×昨日SMA（N）\n    '
    _rsi = pd.DataFrame()
    _rsi['date'] = df['date']
    px = df.close - df.close.shift(1)
    px[px < 0] = 0
    _rsi['rsi'] = sma(px, n) / sma((df['close'] - df['close'].shift(1)).abs(), n) * 100
    return _rsi

def vrsi(df, n=6):
    if False:
        for i in range(10):
            print('nop')
    '\n    量相对强弱指标\n    VRSI=SMA（最大值（成交量-REF（成交量，1），0），N,1）/SMA（ABS（（成交量-REF（成交量，1），N，1）×100%\n    '
    _vrsi = pd.DataFrame()
    _vrsi['date'] = df['date']
    px = df['volume'] - df['volume'].shift(1)
    px[px < 0] = 0
    _vrsi['vrsi'] = sma(px, n) / sma((df['volume'] - df['volume'].shift(1)).abs(), n) * 100
    return _vrsi

def boll(df, n=26, k=2):
    if False:
        i = 10
        return i + 15
    '\n    布林线指标BOLL boll(26,2)\tMID=MA(N)\n    标准差MD=根号[∑（CLOSE-MA(CLOSE，N)）^2/N]\n    UPPER=MID＋k×MD\n    LOWER=MID－k×MD\n    '
    _boll = pd.DataFrame()
    _boll['date'] = df.date
    _boll['mid'] = _ma(df.close, n)
    _mdd = _md(df.close, n)
    _boll['up'] = _boll.mid + k * _mdd
    _boll['low'] = _boll.mid - k * _mdd
    return _boll

def bbiboll(df, n=10, k=3):
    if False:
        for i in range(10):
            print('nop')
    '\n    BBI多空布林线\tbbiboll(10,3)\n    BBI={MA(3)+ MA(6)+ MA(12)+ MA(24)}/4\n    标准差MD=根号[∑（BBI-MA(BBI，N)）^2/N]\n    UPR= BBI＋k×MD\n    DWN= BBI－k×MD\n    '
    _bbiboll = pd.DataFrame()
    _bbiboll['date'] = df.date
    _bbiboll['bbi'] = (_ma(df.close, 3) + _ma(df.close, 6) + _ma(df.close, 12) + _ma(df.close, 24)) / 4
    _bbiboll['md'] = _md(_bbiboll.bbi, n)
    _bbiboll['upr'] = _bbiboll.bbi + k * _bbiboll.md
    _bbiboll['dwn'] = _bbiboll.bbi - k * _bbiboll.md
    return _bbiboll

def wr(df, n=14):
    if False:
        return 10
    '\n    威廉指标 w&r\n    WR=[最高值（最高价，N）-收盘价]/[最高值（最高价，N）-最低值（最低价，N）]×100%\n    '
    _wr = pd.DataFrame()
    _wr['date'] = df['date']
    higest = df.high.rolling(n).max()
    _wr['wr'] = (higest - df.close) / (higest - df.low.rolling(n).min()) * 100
    return _wr

def bias(df, n=12):
    if False:
        return 10
    '\n    乖离率 bias\n    bias=[(当日收盘价-12日平均价)/12日平均价]×100%\n    '
    _bias = pd.DataFrame()
    _bias['date'] = df.date
    _mav = df.close.rolling(n).mean()
    _bias['bias'] = np.true_divide(df.close - _mav, _mav) * 100
    return _bias

def asi(df, n=5):
    if False:
        print('Hello World!')
    '\n    振动升降指标(累计震动升降因子) ASI  # 同花顺给出的公式不完整就不贴出来了\n    '
    _asi = pd.DataFrame()
    _asi['date'] = df.date
    _m = pd.DataFrame()
    _m['a'] = (df.high - df.close.shift()).abs()
    _m['b'] = (df.low - df.close.shift()).abs()
    _m['c'] = (df.high - df.low.shift()).abs()
    _m['d'] = (df.close.shift() - df.open.shift()).abs()
    _m['r'] = _m.apply(lambda x: x.a + 0.5 * x.b + 0.25 * x.d if max(x.a, x.b, x.c) == x.a else x.b + 0.5 * x.a + 0.25 * x.d if max(x.a, x.b, x.c) == x.b else x.c + 0.25 * x.d, axis=1)
    _m['x'] = df.close - df.close.shift() + 0.5 * (df.close - df.open) + df.close.shift() - df.open.shift()
    _m['k'] = np.maximum(_m.a, _m.b)
    _asi['si'] = 16 * (_m.x / _m.r) * _m.k
    _asi['asi'] = _ma(_asi.si, n)
    return _asi

def vr_rate(df, n=26):
    if False:
        for i in range(10):
            print('nop')
    '\n    成交量变异率 vr or vr_rate\n    VR=（AVS+1/2CVS）/（BVS+1/2CVS）×100\n    其中：\n    AVS：表示N日内股价上涨成交量之和\n    BVS：表示N日内股价下跌成交量之和\n    CVS：表示N日内股价不涨不跌成交量之和\n    '
    _vr = pd.DataFrame()
    _vr['date'] = df['date']
    _m = pd.DataFrame()
    _m['volume'] = df.volume
    _m['cs'] = df.close - df.close.shift(1)
    _m['avs'] = _m.apply(lambda x: x.volume if x.cs > 0 else 0, axis=1)
    _m['bvs'] = _m.apply(lambda x: x.volume if x.cs < 0 else 0, axis=1)
    _m['cvs'] = _m.apply(lambda x: x.volume if x.cs == 0 else 0, axis=1)
    _vr['vr'] = (_m.avs.rolling(n).sum() + 1 / 2 * _m.cvs.rolling(n).sum()) / (_m.bvs.rolling(n).sum() + 1 / 2 * _m.cvs.rolling(n).sum()) * 100
    return _vr

def vr(df, n=5):
    if False:
        for i in range(10):
            print('nop')
    '\n    开市后平均每分钟的成交量与过去5个交易日平均每分钟成交量之比\n    量比:=V/REF(MA(V,5),1);\n    涨幅:=(C-REF(C,1))/REF(C,1)*100;\n    1)量比大于1.8，涨幅小于2%，现价涨幅在0—2%之间，在盘中选股的\n    选股:量比>1.8 AND 涨幅>0 AND 涨幅<2;\n    '
    _vr = pd.DataFrame()
    _vr['date'] = df.date
    _vr['vr'] = df.volume / _ma(df.volume, n).shift(1)
    _vr['rr'] = (df.close - df.close.shift(1)) / df.close.shift(1) * 100
    return _vr

def arbr(df, n=26):
    if False:
        for i in range(10):
            print('nop')
    '\n    人气意愿指标\tarbr(26)\n    N日AR=N日内（H－O）之和除以N日内（O－L）之和\n    其中，H为当日最高价，L为当日最低价，O为当日开盘价，N为设定的时间参数，一般原始参数日设定为26日\n    N日BR=N日内（H－CY）之和除以N日内（CY－L）之和\n    其中，H为当日最高价，L为当日最低价，CY为前一交易日的收盘价，N为设定的时间参数，一般原始参数日设定为26日。\n    '
    _arbr = pd.DataFrame()
    _arbr['date'] = df.date
    _arbr['ar'] = (df.high - df.open).rolling(n).sum() / (df.open - df.low).rolling(n).sum() * 100
    _arbr['br'] = (df.high - df.close.shift(1)).rolling(n).sum() / (df.close.shift() - df.low).rolling(n).sum() * 100
    return _arbr

def dpo(df, n=20, m=6):
    if False:
        return 10
    '\n    区间震荡线指标\tdpo(20,6)\n    DPO=CLOSE-MA（CLOSE, N/2+1）\n    MADPO=MA（DPO,M）\n    '
    _dpo = pd.DataFrame()
    _dpo['date'] = df['date']
    _dpo['dpo'] = df.close - _ma(df.close, int(n / 2 + 1))
    _dpo['dopma'] = _ma(_dpo.dpo, m)
    return _dpo

def trix(df, n=12, m=20):
    if False:
        while True:
            i = 10
    '\n    三重指数平滑平均\tTRIX(12)\n    TR= EMA(EMA(EMA(CLOSE,N),N),N)，即进行三次平滑处理\n    TRIX=(TR-昨日TR)/ 昨日TR×100\n    TRMA=MA（TRIX，M）\n    '
    _trix = pd.DataFrame()
    _trix['date'] = df.date
    tr = _ema(_ema(_ema(df.close, n), n), n)
    _trix['trix'] = (tr - tr.shift()) / tr.shift() * 100
    _trix['trma'] = _ma(_trix.trix, m)
    return _trix

def bbi(df):
    if False:
        return 10
    '\n    多空指数\tBBI(3,6,12,24)\n    BBI=（3日均价+6日均价+12日均价+24日均价）/4\n    '
    _bbi = pd.DataFrame()
    _bbi['date'] = df['date']
    _bbi['bbi'] = (_ma(df.close, 3) + _ma(df.close, 6) + _ma(df.close, 12) + _ma(df.close, 24)) / 4
    return _bbi

def mtm(df, n=6, m=5):
    if False:
        while True:
            i = 10
    '\n    动力指标\tMTM(6,5)\n    MTM（N日）=C-REF(C,N)式中，C=当日的收盘价，REF(C,N)=N日前的收盘价；N日是只计算交易日期，剔除掉节假日。\n    MTMMA（MTM，N1）= MA（MTM，N1）\n    N表示间隔天数，N1表示天数\n    '
    _mtm = pd.DataFrame()
    _mtm['date'] = df.date
    _mtm['mtm'] = df.close - df.close.shift(n)
    _mtm['mtmma'] = _ma(_mtm.mtm, m)
    return _mtm

def obv(df):
    if False:
        for i in range(10):
            print('nop')
    '\n    能量潮  On Balance Volume\n    多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V  # 同花顺貌似用的下面公式\n    主公式：当日OBV=前一日OBV+今日成交量\n    1.基期OBV值为0，即该股上市的第一天，OBV值为0\n    2.若当日收盘价＞上日收盘价，则当日OBV=前一日OBV＋今日成交量\n    3.若当日收盘价＜上日收盘价，则当日OBV=前一日OBV－今日成交量\n    4.若当日收盘价＝上日收盘价，则当日OBV=前一日OBV\n    '
    _obv = pd.DataFrame()
    _obv['date'] = df['date']
    _m = pd.DataFrame()
    _m['date'] = df.date
    _m['cs'] = df.close - df.close.shift()
    _m['v'] = df.volume
    _m['vv'] = _m.apply(lambda x: x.v if x.cs > 0 else -x.v if x.cs < 0 else 0, axis=1)
    _obv['obv'] = _m.vv.expanding(1).sum()
    return _obv

def cci(df, n=14):
    if False:
        while True:
            i = 10
    '\n    顺势指标\n    TYP:=(HIGH+LOW+CLOSE)/3\n    CCI:=(TYP-MA(TYP,N))/(0.015×AVEDEV(TYP,N))\n    '
    _cci = pd.DataFrame()
    _cci['date'] = df['date']
    typ = (df.high + df.low + df.close) / 3
    _cci['cci'] = (typ - typ.rolling(n).mean()) / (0.015 * typ.rolling(min_periods=1, center=False, window=n).apply(lambda x: np.fabs(x - x.mean()).mean()))
    return _cci

def priceosc(df, n=12, m=26):
    if False:
        print('Hello World!')
    '\n    价格振动指数\n    PRICEOSC=(MA(C,12)-MA(C,26))/MA(C,12) * 100\n    '
    _c = pd.DataFrame()
    _c['date'] = df['date']
    man = _ma(df.close, n)
    _c['osc'] = (man - _ma(df.close, m)) / man * 100
    return _c

def sma(a, n, m=1):
    if False:
        i = 10
        return i + 15
    '\n    平滑移动指标 Smooth Moving Average\n    '
    " # 方法一，此方法有缺陷\n    _sma = []\n    for index, value in enumerate(a):\n        if index == 0 or pd.isna(value) or np.isnan(value):\n            tsma = 0\n        else:\n            # Y=(M*X+(N-M)*Y')/N\n            tsma = (m * value + (n - m) * tsma) / n\n        _sma.append(tsma)\n    return pd.Series(_sma)\n    "
    ' # 方法二\n\n    results = np.nan_to_num(a).copy()\n    # FIXME this is very slow\n    for i in range(1, len(a)):\n        results[i] = (m * results[i] + (n - m) * results[i - 1]) / n\n        # results[i] = ((n - 1) * results[i - 1] + results[i]) / n\n    # return results\n    '
    a = a.fillna(0)
    b = a.ewm(min_periods=0, ignore_na=False, adjust=False, alpha=m / n).mean()
    return b

def dbcd(df, n=5, m=16, t=76):
    if False:
        print('Hello World!')
    '\n    异同离差乖离率\tdbcd(5,16,76)\n    BIAS=(C-MA(C,N))/MA(C,N)\n    DIF=(BIAS-REF(BIAS,M))\n    DBCD=SMA(DIF,T,1) =（1-1/T）×SMA(REF(DIF,1),T,1)+ 1/T×DIF\n    MM=MA(DBCD,5)\n    '
    _dbcd = pd.DataFrame()
    _dbcd['date'] = df.date
    man = _ma(df.close, n)
    _bias = (df.close - man) / man
    _dif = _bias - _bias.shift(m)
    _dbcd['dbcd'] = sma(_dif, t)
    _dbcd['mm'] = _ma(_dbcd.dbcd, n)
    return _dbcd

def roc(df, n=12, m=6):
    if False:
        return 10
    '\n    变动速率\troc(12,6)\n    ROC=(今日收盘价-N日前的收盘价)/ N日前的收盘价×100%\n    ROCMA=MA（ROC，M）\n    ROC:(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)×100\n    ROCMA:MA(ROC,M)\n    '
    _roc = pd.DataFrame()
    _roc['date'] = df['date']
    _roc['roc'] = (df.close - df.close.shift(n)) / df.close.shift(n) * 100
    _roc['rocma'] = _ma(_roc.roc, m)
    return _roc

def vroc(df, n=12):
    if False:
        for i in range(10):
            print('nop')
    '\n    量变动速率\n    VROC=(当日成交量-N日前的成交量)/ N日前的成交量×100%\n    '
    _vroc = pd.DataFrame()
    _vroc['date'] = df['date']
    _vroc['vroc'] = (df.volume - df.volume.shift(n)) / df.volume.shift(n) * 100
    return _vroc

def cr(df, n=26):
    if False:
        while True:
            i = 10
    ' 能量指标\n    CR=∑（H-PM）/∑（PM-L）×100\n    PM:上一交易日中价（(最高、最低、收盘价的均值)\n    H：当天最高价\n    L：当天最低价\n    '
    _cr = pd.DataFrame()
    _cr['date'] = df.date
    pm = df[['high', 'low', 'close']].mean(axis=1).shift(1)
    _cr['cr'] = (df.high - pm).rolling(n).sum() / (pm - df.low).rolling(n).sum() * 100
    return _cr

def psy(df, n=12):
    if False:
        return 10
    '\n    心理指标\tPSY(12)\n    PSY=N日内上涨天数/N×100\n    PSY:COUNT(CLOSE>REF(CLOSE,1),N)/N×100\n    MAPSY=PSY的M日简单移动平均\n    '
    _psy = pd.DataFrame()
    _psy['date'] = df.date
    p = df.close - df.close.shift()
    p[p <= 0] = np.nan
    _psy['psy'] = p.rolling(n).count() / n * 100
    return _psy

def wad(df, n=30):
    if False:
        while True:
            i = 10
    '\n    威廉聚散指标\tWAD(30)\n    TRL=昨日收盘价与今日最低价中价格最低者；TRH=昨日收盘价与今日最高价中价格最高者\n    如果今日的收盘价>昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRL\n    如果今日的收盘价<昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRH\n    如果今日的收盘价=昨日的收盘价，则今日的A/D=0\n    WAD=今日的A/D+昨日的WAD；MAWAD=WAD的M日简单移动平均\n    '

    def dmd(x):
        if False:
            return 10
        if x.c > 0:
            y = x.close - x.trl
        elif x.c < 0:
            y = x.close - x.trh
        else:
            y = 0
        return y
    _wad = pd.DataFrame()
    _wad['date'] = df['date']
    _ad = pd.DataFrame()
    _ad['trl'] = np.minimum(df.low, df.close.shift(1))
    _ad['trh'] = np.maximum(df.high, df.close.shift(1))
    _ad['c'] = df.close - df.close.shift()
    _ad['close'] = df.close
    _ad['ad'] = _ad.apply(dmd, axis=1)
    _wad['wad'] = _ad.ad.expanding(1).sum()
    _wad['mawad'] = _ma(_wad.wad, n)
    return _wad

def mfi(df, n=14):
    if False:
        for i in range(10):
            print('nop')
    '\n    资金流向指标\tmfi(14)\n    MF＝TYP×成交量；TYP:当日中价（(最高、最低、收盘价的均值)\n    如果当日TYP>昨日TYP，则将当日的MF值视为当日PMF值。而当日NMF值＝0\n    如果当日TYP<=昨日TYP，则将当日的MF值视为当日NMF值。而当日PMF值=0\n    MR=∑PMF/∑NMF\n    MFI＝100-（100÷(1＋MR)）\n    '
    _mfi = pd.DataFrame()
    _mfi['date'] = df.date
    _m = pd.DataFrame()
    _m['typ'] = df[['high', 'low', 'close']].mean(axis=1)
    _m['mf'] = _m.typ * df.volume
    _m['typ_shift'] = _m.typ - _m.typ.shift(1)
    _m['pmf'] = _m.apply(lambda x: x.mf if x.typ_shift > 0 else 0, axis=1)
    _m['nmf'] = _m.apply(lambda x: x.mf if x.typ_shift <= 0 else 0, axis=1)
    _m['mr'] = _m.pmf.rolling(n).sum() / _m.nmf.rolling(n).sum()
    _mfi['mfi'] = 100 * _m.mr / (1 + _m.mr)
    return _mfi

def pvt(df):
    if False:
        while True:
            i = 10
    '\n    pvt\t量价趋势指标\tpvt\n    如果设x=(今日收盘价—昨日收盘价)/昨日收盘价×当日成交量，\n    那么当日PVT指标值则为从第一个交易日起每日X值的累加。\n    '
    _pvt = pd.DataFrame()
    _pvt['date'] = df.date
    x = (df.close - df.close.shift(1)) / df.close.shift(1) * df.volume
    _pvt['pvt'] = x.expanding(1).sum()
    return _pvt

def wvad(df, n=24, m=6):
    if False:
        return 10
    '  # 算法是对的，同花顺计算wvad用的n=6\n    威廉变异离散量\twvad(24,6)\n    WVAD=N1日的∑ {(当日收盘价－当日开盘价)/(当日最高价－当日最低价)×成交量}\n    MAWVAD=MA（WVAD，N2）\n    '
    _wvad = pd.DataFrame()
    _wvad['date'] = df.date
    _wvad['wvad'] = (np.true_divide(df.close - df.open, df.high - df.low) * df.volume).rolling(n).sum()
    _wvad['mawvad'] = _ma(_wvad.wvad, m)
    return _wvad

def cdp(df):
    if False:
        return 10
    '\n    逆势操作\tcdp\n    CDP=(最高价+最低价+收盘价)/3  # 同花顺实际用的(H+L+2*c)/4\n    AH=CDP+(前日最高价-前日最低价)\n    NH=CDP×2-最低价\n    NL=CDP×2-最高价\n    AL=CDP-(前日最高价-前日最低价)\n    '
    _cdp = pd.DataFrame()
    _cdp['date'] = df.date
    _cdp['cdp'] = df[['high', 'low', 'close', 'close']].shift().mean(axis=1)
    _cdp['ah'] = _cdp.cdp + (df.high.shift(1) - df.low.shift())
    _cdp['al'] = _cdp.cdp - (df.high.shift(1) - df.low.shift())
    _cdp['nh'] = _cdp.cdp * 2 - df.low.shift(1)
    _cdp['nl'] = _cdp.cdp * 2 - df.high.shift(1)
    return _cdp

def env(df, n=14):
    if False:
        return 10
    '\n    ENV指标\tENV(14)\n    Upper=MA(CLOSE，N)×1.06\n    LOWER= MA(CLOSE，N)×0.94\n    '
    _env = pd.DataFrame()
    _env['date'] = df.date
    _env['up'] = df.close.rolling(n).mean() * 1.06
    _env['low'] = df.close.rolling(n).mean() * 0.94
    return _env

def mike(df, n=12):
    if False:
        for i in range(10):
            print('nop')
    '\n    麦克指标\tmike(12)\n    初始价（TYP）=（当日最高价＋当日最低价＋当日收盘价）/3\n    HV=N日内区间最高价\n    LV=N日内区间最低价\n    初级压力线（WR）=TYP×2-LV\n    中级压力线（MR）=TYP+HV-LV\n    强力压力线（SR）=2×HV-LV\n    初级支撑线（WS）=TYP×2-HV\n    中级支撑线（MS）=TYP-HV+LV\n    强力支撑线（SS）=2×LV-HV\n    '
    _mike = pd.DataFrame()
    _mike['date'] = df.date
    typ = df[['high', 'low', 'close']].mean(axis=1)
    hv = df.high.rolling(n).max()
    lv = df.low.rolling(n).min()
    _mike['wr'] = typ * 2 - lv
    _mike['mr'] = typ + hv - lv
    _mike['sr'] = 2 * hv - lv
    _mike['ws'] = typ * 2 - hv
    _mike['ms'] = typ - hv + lv
    _mike['ss'] = 2 * lv - hv
    return _mike

def vma(df, n=5):
    if False:
        for i in range(10):
            print('nop')
    '\n    量简单移动平均\tVMA(5)\tVMA=MA(volume,N)\n    VOLUME表示成交量；N表示天数\n    '
    _vma = pd.DataFrame()
    _vma['date'] = df.date
    _vma['vma'] = _ma(df.volume, n)
    return _vma

def vmacd(df, qn=12, sn=26, m=9):
    if False:
        while True:
            i = 10
    '\n    量指数平滑异同平均\tvmacd(12,26,9)\n    今日EMA（N）=2/（N+1）×今日成交量+(N-1)/（N+1）×昨日EMA（N）\n    DIFF= EMA（N1）- EMA（N2）\n    DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)\n    MACD（BAR）=2×（DIF-DEA）\n    '
    _vmacd = pd.DataFrame()
    _vmacd['date'] = df.date
    _vmacd['diff'] = _ema(df.volume, qn) - _ema(df.volume, sn)
    _vmacd['dea'] = _ema(_vmacd['diff'], m)
    _vmacd['macd'] = _vmacd['diff'] - _vmacd['dea']
    return _vmacd

def vosc(df, n=12, m=26):
    if False:
        i = 10
        return i + 15
    '\n    成交量震荡\tvosc(12,26)\n    VOSC=（MA（VOLUME,SHORT）- MA（VOLUME,LONG））/MA（VOLUME,SHORT）×100\n    '
    _c = pd.DataFrame()
    _c['date'] = df['date']
    _c['osc'] = (_ma(df.volume, n) - _ma(df.volume, m)) / _ma(df.volume, n) * 100
    return _c

def tapi(df, n=6):
    if False:
        return 10
    ' # TODO: 由于get_k_data返回数据中没有amount，可以用get_h_data中amount，算法是正确的\n    加权指数成交值\ttapi(6)\n    TAPI=每日成交总值/当日加权指数=a/PI；A表示每日的成交金额，PI表示当天的股价指数即指收盘价\n    '
    _tapi = pd.DataFrame()
    _tapi['tapi'] = df.amount / df.close
    _tapi['matapi'] = _ma(_tapi.tapi, n)
    return _tapi

def vstd(df, n=10):
    if False:
        return 10
    '\n    成交量标准差\tvstd(10)\n    VSTD=STD（Volume,N）=[∑（Volume-MA(Volume，N)）^2/N]^0.5\n    '
    _vstd = pd.DataFrame()
    _vstd['date'] = df.date
    _vstd['vstd'] = df.volume.rolling(n).std(ddof=1)
    return _vstd

def adtm(df, n=23, m=8):
    if False:
        i = 10
        return i + 15
    '\n    动态买卖气指标\tadtm(23,8)\n    如果开盘价≤昨日开盘价，DTM=0\n    如果开盘价＞昨日开盘价，DTM=(最高价-开盘价)和(开盘价-昨日开盘价)的较大值\n    如果开盘价≥昨日开盘价，DBM=0\n    如果开盘价＜昨日开盘价，DBM=(开盘价-最低价)\n    STM=DTM在N日内的和\n    SBM=DBM在N日内的和\n    如果STM > SBM,ADTM=(STM-SBM)/STM\n    如果STM < SBM , ADTM = (STM-SBM)/SBM\n    如果STM = SBM,ADTM=0\n    ADTMMA=MA(ADTM,M)\n    '
    _adtm = pd.DataFrame()
    _adtm['date'] = df.date
    _m = pd.DataFrame()
    _m['cc'] = df.open - df.open.shift(1)
    _m['ho'] = df.high - df.open
    _m['ol'] = df.open - df.low
    _m['dtm'] = _m.apply(lambda x: max(x.ho, x.cc) if x.cc > 0 else 0, axis=1)
    _m['dbm'] = _m.apply(lambda x: x.ol if x.cc < 0 else 0, axis=1)
    _m['stm'] = _m.dtm.rolling(n).sum()
    _m['sbm'] = _m.dbm.rolling(n).sum()
    _m['ss'] = _m.stm - _m.sbm
    _adtm['adtm'] = _m.apply(lambda x: x.ss / x.stm if x.ss > 0 else x.ss / x.sbm if x.ss < 0 else 0, axis=1)
    _adtm['adtmma'] = _ma(_adtm.adtm, m)
    return _adtm

def mi(df, n=12):
    if False:
        while True:
            i = 10
    '\n    动量指标\tmi(12)\n    A=CLOSE-REF(CLOSE,N)\n    MI=SMA(A,N,1)\n    '
    _mi = pd.DataFrame()
    _mi['date'] = df.date
    _mi['mi'] = sma(df.close - df.close.shift(n), n)
    return _mi

def micd(df, n=3, m=10, k=20):
    if False:
        for i in range(10):
            print('nop')
    '\n    异同离差动力指数\tmicd(3,10,20)\n    MI=CLOSE-ref(CLOSE,1)AMI=SMA(MI,N1,1)\n    DIF=MA(ref(AMI,1),N2)-MA(ref(AMI,1),N3)\n    MICD=SMA(DIF,10,1)\n    '
    _micd = pd.DataFrame()
    _micd['date'] = df.date
    mi = df.close - df.close.shift(1)
    ami = sma(mi, n)
    dif = _ma(ami.shift(1), m) - _ma(ami.shift(1), k)
    _micd['micd'] = sma(dif, m)
    return _micd

def rc(df, n=50):
    if False:
        while True:
            i = 10
    '\n    变化率指数\trc(50)\n    RC=收盘价/REF（收盘价，N）×100\n    ARC=EMA（REF（RC，1），N，1）\n    '
    _rc = pd.DataFrame()
    _rc['date'] = df.date
    _rc['rc'] = df.close / df.close.shift(n) * 100
    _rc['arc'] = sma(_rc.rc.shift(1), n)
    return _rc

def rccd(df, n=59, m=21, k=28):
    if False:
        while True:
            i = 10
    '  # TODO: 计算结果错误和同花顺不同，检查不出来为什么\n    异同离差变化率指数 rate of change convergence divergence\trccd(59,21,28)\n    RC=收盘价/REF（收盘价，N）×100%\n    ARC=EMA(REF(RC,1),N,1)\n    DIF=MA(ref(ARC,1),N1)-MA MA(ref(ARC,1),N2)\n    RCCD=SMA(DIF,N,1)\n    '
    _rccd = pd.DataFrame()
    _rccd['date'] = df.date
    rc = df.close / df.close.shift(n) * 100
    arc = sma(rc.shift(), n)
    dif = _ma(arc.shift(), m) - _ma(arc.shift(), k)
    _rccd['rccd'] = sma(dif, n)
    return _rccd

def srmi(df, n=9):
    if False:
        while True:
            i = 10
    '\n    SRMIMI修正指标\tsrmi(9)\n    如果收盘价>N日前的收盘价，SRMI就等于（收盘价-N日前的收盘价）/收盘价\n    如果收盘价<N日前的收盘价，SRMI就等于（收盘价-N日签的收盘价）/N日前的收盘价\n    如果收盘价=N日前的收盘价，SRMI就等于0\n    '
    _srmi = pd.DataFrame()
    _srmi['date'] = df.date
    _m = pd.DataFrame()
    _m['close'] = df.close
    _m['cp'] = df.close.shift(n)
    _m['cs'] = df.close - df.close.shift(n)
    _srmi['srmi'] = _m.apply(lambda x: x.cs / x.close if x.cs > 0 else x.cs / x.cp if x.cs < 0 else 0, axis=1)
    return _srmi

def dptb(df, n=7):
    if False:
        print('Hello World!')
    '\n    大盘同步指标\tdptb(7)\n    DPTB=（统计N天中个股收盘价>开盘价，且指数收盘价>开盘价的天数或者个股收盘价<开盘价，且指数收盘价<开盘价）/N\n    '
    ind = ts.get_k_data('sh000001', start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index('date', inplace=True)
    ind.set_index('date', inplace=True)
    _dptb = pd.DataFrame(index=df.date)
    q = ind.close - ind.open
    _dptb['p'] = sd.close - sd.open
    _dptb['q'] = q
    _dptb['m'] = _dptb.apply(lambda x: 1 if x.p > 0 and x.q > 0 or (x.p < 0 and x.q < 0) else np.nan, axis=1)
    _dptb['jdrs'] = _dptb.m.rolling(n).count() / n
    _dptb.drop(columns=['p', 'q', 'm'], inplace=True)
    _dptb.reset_index(inplace=True)
    return _dptb

def jdqs(df, n=20):
    if False:
        print('Hello World!')
    '\n    阶段强势指标\tjdqs(20)\n    JDQS=（统计N天中个股收盘价>开盘价，且指数收盘价<开盘价的天数）/（统计N天中指数收盘价<开盘价的天数）\n    '
    ind = ts.get_k_data('sh000001', start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index('date', inplace=True)
    ind.set_index('date', inplace=True)
    _jdrs = pd.DataFrame(index=df.date)
    q = ind.close - ind.open
    _jdrs['p'] = sd.close - sd.open
    _jdrs['q'] = q
    _jdrs['m'] = _jdrs.apply(lambda x: 1 if x.p > 0 and x.q < 0 else np.nan, axis=1)
    q[q > 0] = np.nan
    _jdrs['t'] = q
    _jdrs['jdrs'] = _jdrs.m.rolling(n).count() / _jdrs.t.rolling(n).count()
    _jdrs.drop(columns=['p', 'q', 'm', 't'], inplace=True)
    _jdrs.reset_index(inplace=True)
    return _jdrs

def jdrs(df, n=20):
    if False:
        i = 10
        return i + 15
    '\n    阶段弱势指标\tjdrs(20)\n    JDRS=（统计N天中个股收盘价<开盘价，且指数收盘价>开盘价的天数）/（统计N天中指数收盘价>开盘价的天数）\n    '
    ind = ts.get_k_data('sh000001', start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index('date', inplace=True)
    ind.set_index('date', inplace=True)
    _jdrs = pd.DataFrame(index=df.date)
    q = ind.close - ind.open
    _jdrs['p'] = sd.close - sd.open
    _jdrs['q'] = q
    _jdrs['m'] = _jdrs.apply(lambda x: 1 if x.p < 0 and x.q > 0 else np.nan, axis=1)
    q[q < 0] = np.nan
    _jdrs['t'] = q
    _jdrs['jdrs'] = _jdrs.m.rolling(n).count() / _jdrs.t.rolling(n).count()
    _jdrs.drop(columns=['p', 'q', 'm', 't'], inplace=True)
    _jdrs.reset_index(inplace=True)
    return _jdrs

def zdzb(df, n=125, m=5, k=20):
    if False:
        while True:
            i = 10
    '\n    筑底指标\tzdzb(125,5,20)\n    A=（统计N1日内收盘价>=前收盘价的天数）/（统计N1日内收盘价<前收盘价的天数）\n    B=MA（A,N2）\n    D=MA（A，N3）\n    '
    _zdzb = pd.DataFrame()
    _zdzb['date'] = df.date
    p = df.close - df.close.shift(1)
    q = p.copy()
    p[p < 0] = np.nan
    q[q >= 0] = np.nan
    _zdzb['a'] = p.rolling(n).count() / q.rolling(n).count()
    _zdzb['b'] = _zdzb.a.rolling(m).mean()
    _zdzb['d'] = _zdzb.a.rolling(k).mean()
    return _zdzb

def atr(df, n=14):
    if False:
        i = 10
        return i + 15
    '\n    真实波幅\tatr(14)\n    TR:MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))\n    ATR:MA(TR,N)\n    '
    _atr = pd.DataFrame()
    _atr['date'] = df.date
    _atr['tr'] = np.vstack([df.high - df.low, (df.close.shift(1) - df.high).abs(), (df.close.shift(1) - df.low).abs()]).max(axis=0)
    _atr['atr'] = _atr.tr.rolling(n).mean()
    return _atr

def mass(df, n=9, m=25):
    if False:
        i = 10
        return i + 15
    '\n    梅丝线\tmass(9,25)\n    AHL=MA(（H-L）,N1)\n    BHL= MA（AHL，N1）\n    MASS=SUM（AHL/BHL，N2）\n    H：表示最高价；L：表示最低价\n    '
    _mass = pd.DataFrame()
    _mass['date'] = df.date
    ahl = _ma(df.high - df.low, n)
    bhl = _ma(ahl, n)
    _mass['mass'] = (ahl / bhl).rolling(m).sum()
    return _mass

def vhf(df, n=28):
    if False:
        while True:
            i = 10
    '\n    纵横指标\tvhf(28)\n    VHF=（N日内最大收盘价与N日内最小收盘价之前的差）/（N日收盘价与前收盘价差的绝对值之和）\n    '
    _vhf = pd.DataFrame()
    _vhf['date'] = df.date
    _vhf['vhf'] = (df.close.rolling(n).max() - df.close.rolling(n).min()) / (df.close - df.close.shift(1)).abs().rolling(n).sum()
    return _vhf

def cvlt(df, n=10):
    if False:
        while True:
            i = 10
    '\n    佳庆离散指标\tcvlt(10)\n    cvlt=（最高价与最低价的差的指数移动平均-前N日的最高价与最低价的差的指数移动平均）/前N日的最高价与最低价的差的指数移动平均\n    '
    _cvlt = pd.DataFrame()
    _cvlt['date'] = df.date
    p = _ema(df.high.shift(n) - df.low.shift(n), n)
    _cvlt['cvlt'] = (_ema(df.high - df.low, n) - p) / p * 100
    return _cvlt

def up_n(df):
    if False:
        i = 10
        return i + 15
    '\n    连涨天数\tup_n\t连续上涨天数，当天收盘价大于开盘价即为上涨一天 # 同花顺实际结果用收盘价-前一天收盘价\n    '
    _up = pd.DataFrame()
    _up['date'] = df.date
    p = df.close - df.close.shift()
    p[p > 0] = 1
    p[p < 0] = 0
    m = []
    for (k, g) in itertools.groupby(p):
        t = 0
        for i in g:
            if k == 0:
                m.append(0)
            else:
                t += 1
                m.append(t)
    _up['up'] = m
    return _up

def down_n(df):
    if False:
        while True:
            i = 10
    '\n    连跌天数\tdown_n\t连续下跌天数，当天收盘价小于开盘价即为下跌一天\n    '
    _down = pd.DataFrame()
    _down['date'] = df.date
    p = df.close - df.close.shift()
    p[p > 0] = 0
    p[p < 0] = 1
    m = []
    for (k, g) in itertools.groupby(p):
        t = 0
        for i in g:
            if k == 0:
                m.append(0)
            else:
                t += 1
                m.append(t)
    _down['down'] = m
    return _down

def join_frame(d1, d2, column='date'):
    if False:
        print('Hello World!')
    return d1.join(d2.set_index(column), on=column)
if __name__ == '__main__':
    import tushare as ts
    data = ts.get_k_data('601138', start='2017-05-01')
    print(rccd(data))