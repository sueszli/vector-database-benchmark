__author__ = 'chengzhi'
'\ntqsdk.ta 模块包含了一批常用的技术指标计算函数\n(函数返回值类型保持为 pandas.Dataframe)\n'
import math
import numpy as np
import pandas as pd
import tqsdk.tafunc

def ATR(df, n):
    if False:
        print('Hello World!')
    '\n    平均真实波幅\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 平均真实波幅的周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 分别是"tr"和"atr", 分别代表真实波幅和平均真实波幅\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的平均真实波幅\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import ATR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        atr = ATR(klines, 14)\n        print(atr.tr)  # 真实波幅\n        print(atr.atr)  # 平均真实波幅\n\n        # 预计的输出是这样的:\n        [..., 143.0, 48.0, 80.0, ...]\n        [..., 95.20000000000005, 92.0571428571429, 95.21428571428575, ...]\n    '
    new_df = pd.DataFrame()
    pre_close = df['close'].shift(1)
    new_df['tr'] = np.where(df['high'] - df['low'] > np.absolute(pre_close - df['high']), np.where(df['high'] - df['low'] > np.absolute(pre_close - df['low']), df['high'] - df['low'], np.absolute(pre_close - df['low'])), np.where(np.absolute(pre_close - df['high']) > np.absolute(pre_close - df['low']), np.absolute(pre_close - df['high']), np.absolute(pre_close - df['low'])))
    new_df['atr'] = tqsdk.tafunc.ma(new_df['tr'], n)
    return new_df

def BIAS(df, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    乖离率\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 移动平均的计算周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"bias", 代表计算出来的乖离率值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的乖离率\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import BIAS\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        bias = BIAS(klines, 6)\n        print(list(bias["bias"]))  # 乖离率\n\n        # 预计的输出是这样的:\n        [..., 2.286835533357118, 2.263301549041151, 0.7068445823271412, ...]\n    '
    ma1 = tqsdk.tafunc.ma(df['close'], n)
    new_df = pd.DataFrame(data=list((df['close'] - ma1) / ma1 * 100), columns=['bias'])
    return new_df

def BOLL(df, n, p):
    if False:
        i = 10
        return i + 15
    '\n    布林线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        p (int): 计算参数p\n\n    Returns:\n        pandas.DataFrame: 返回的dataframe包含3列, 分别是"mid", "top"和"bottom", 分别代表布林线的中、上、下轨\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的布林线\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import BOLL\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        boll=BOLL(klines, 26, 2)\n        print(list(boll["mid"]))\n        print(list(boll["top"]))\n        print(list(boll["bottom"]))\n\n        # 预计的输出是这样的:\n        [..., 3401.338461538462, 3425.600000000001, 3452.3230769230777, ...]\n        [..., 3835.083909752222, 3880.677579320277, 3921.885406954584, ...]\n        [..., 2967.593013324702, 2970.5224206797247, 2982.760746891571, ...]\n    '
    new_df = pd.DataFrame()
    mid = tqsdk.tafunc.ma(df['close'], n)
    std = df['close'].rolling(n).std()
    new_df['mid'] = mid
    new_df['top'] = mid + p * std
    new_df['bottom'] = mid - p * std
    return new_df

def DMI(df, n, m):
    if False:
        print('Hello World!')
    '\n    动向指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m (int): 周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含5列, 是"atr", "pdi", "mdi", "adx"和"adxr", 分别代表平均真实波幅, 上升方向线, 下降方向线, 趋向平均值以及评估数值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的动向指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import DMI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        dmi=DMI(klines, 14, 6)\n        print(list(dmi["atr"]))\n        print(list(dmi["pdi"]))\n        print(list(dmi["mdi"]))\n        print(list(dmi["adx"]))\n        print(list(dmi["adxr"]))\n\n        # 预计的输出是这样的:\n        [..., 95.20000000000005, 92.0571428571429, 95.21428571428575, ...]\n        [..., 51.24549819927972, 46.55493482309126, 47.14178544636161, ...]\n        [..., 6.497599039615802, 6.719428926132791, 6.4966241560389655, ...]\n        [..., 78.80507786697127, 76.8773544355082, 75.11662664555287, ...]\n        [..., 70.52493837227118, 73.28531799111778, 74.59341569051983, ...]\n    '
    new_df = pd.DataFrame()
    new_df['atr'] = ATR(df, n)['atr']
    pre_high = df['high'].shift(1)
    pre_low = df['low'].shift(1)
    hd = df['high'] - pre_high
    ld = pre_low - df['low']
    admp = tqsdk.tafunc.ma(pd.Series(np.where((hd > 0) & (hd > ld), hd, 0)), n)
    admm = tqsdk.tafunc.ma(pd.Series(np.where((ld > 0) & (ld > hd), ld, 0)), n)
    new_df['pdi'] = pd.Series(np.where(new_df['atr'] > 0, admp / new_df['atr'] * 100, np.NaN)).ffill()
    new_df['mdi'] = pd.Series(np.where(new_df['atr'] > 0, admm / new_df['atr'] * 100, np.NaN)).ffill()
    ad = pd.Series(np.absolute(new_df['mdi'] - new_df['pdi']) / (new_df['mdi'] + new_df['pdi']) * 100)
    new_df['adx'] = tqsdk.tafunc.ma(ad, m)
    new_df['adxr'] = (new_df['adx'] + new_df['adx'].shift(m)) / 2
    return new_df

def KDJ(df, n, m1, m2):
    if False:
        return 10
    '\n    随机指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m1 (int): 参数m1\n\n        m2 (int): 参数m2\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含3列, 是"k", "d"和"j", 分别代表计算出来的K值, D值和J值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的随机指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import KDJ\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        kdj = KDJ(klines, 9, 3, 3)\n        print(list(kdj["k"]))\n        print(list(kdj["d"]))\n        print(list(kdj["j"]))\n\n        # 预计的输出是这样的:\n        [..., 80.193148635668, 81.83149521546302, 84.60665654726242, ...]\n        [..., 82.33669997171852, 82.16829838630002, 82.98108443995415, ...]\n        [..., 77.8451747299365, 75.90604596356695, 81.15788887378903, ...]\n    '
    new_df = pd.DataFrame()
    hv = df['high'].rolling(n).max()
    lv = df['low'].rolling(n).min()
    rsv = pd.Series(np.where(hv == lv, 0, (df['close'] - lv) / (hv - lv) * 100))
    new_df['k'] = tqsdk.tafunc.sma(rsv, m1, 1)
    new_df['d'] = tqsdk.tafunc.sma(new_df['k'], m2, 1)
    new_df['j'] = 3 * new_df['k'] - 2 * new_df['d']
    return new_df

def MACD(df, short, long, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    异同移动平均线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        short (int): 短周期\n\n        long (int): 长周期\n\n        m (int): 移动平均线的周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含3列, 是"diff", "dea"和"bar", 分别代表离差值, DIFF的指数加权移动平均线, MACD的柱状线\n\n        (注: 因 DataFrame 有diff()函数，因此获取到此指标后："diff"字段使用 macd["diff"] 方式来取值，而非 macd.diff )\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的异同移动平均线\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import MACD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        macd = MACD(klines, 12, 26, 9)\n        print(list(macd["diff"]))\n        print(list(macd["dea"]))\n        print(list(macd["bar"]))\n\n        # 预计的输出是这样的:\n        [..., 149.58313904045826, 155.50790712365142, 160.27622505636737, ...]\n        [..., 121.46944573796466, 128.27713801510203, 134.6769554233551, ...]\n        [..., 56.2273866049872, 54.46153821709879, 51.19853926602451, ...]\n    '
    new_df = pd.DataFrame()
    eshort = tqsdk.tafunc.ema(df['close'], short)
    elong = tqsdk.tafunc.ema(df['close'], long)
    new_df['diff'] = eshort - elong
    new_df['dea'] = tqsdk.tafunc.ema(new_df['diff'], m)
    new_df['bar'] = 2 * (new_df['diff'] - new_df['dea'])
    return new_df

def _sar(open, high, low, close, range_high, range_low, n, step, maximum):
    if False:
        print('Hello World!')
    n = max(np.sum(np.isnan(range_high)), np.sum(np.isnan(range_low))) + 2
    sar = np.empty_like(close)
    sar[:n] = np.NAN
    af = 0
    ep = 0
    trend = 1 if close[n] - open[n] > 0 else -1
    if trend == 1:
        sar[n] = min(range_low[n - 2], low[n - 1])
    else:
        sar[n] = max(range_high[n - 2], high[n - 1])
    for i in range(n, len(sar)):
        if i != n:
            if abs(trend) > 1:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            elif trend == 1:
                sar[i] = min(range_low[i - 2], low[i - 1])
            elif trend == -1:
                sar[i] = max(range_high[i - 2], high[i - 1])
        if trend > 0:
            if sar[i - 1] > low[i]:
                ep = low[i]
                af = step
                trend = -1
            else:
                ep = high[i]
                af = min(af + step, maximum) if ep > range_high[i - 1] else af
                trend += 1
        elif sar[i - 1] < high[i]:
            ep = high[i]
            af = step
            trend = 1
        else:
            ep = low[i]
            af = min(af + step, maximum) if ep < range_low[i - 1] else af
            trend -= 1
    return sar

def SAR(df, n, step, max):
    if False:
        print('Hello World!')
    '\n    抛物线指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): SAR的周期n\n\n        step (float): 步长\n\n        max (float): 极值\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"sar", 代表计算出来的SAR值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的抛物线指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import SAR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        sar=SAR(klines, 4, 0.02, 0.2)\n        print(list(sar["sar"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3742.313604622293, 3764.5708836978342, 3864.4, ...]\n    '
    range_high = df['high'].rolling(n - 1).max()
    range_low = df['low'].rolling(n - 1).min()
    sar = _sar(df['open'].values, df['high'].values, df['low'].values, df['close'].values, range_high.values, range_low.values, n, step, max)
    new_df = pd.DataFrame(data=sar, columns=['sar'])
    return new_df

def WR(df, n):
    if False:
        print('Hello World!')
    '\n    威廉指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"wr", 代表计算出来的威廉指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的威廉指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import WR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        wr = WR(klines, 14)\n        print(list(wr["wr"]))\n\n        # 预计的输出是这样的:\n        [..., -12.843029637760672, -8.488840102451537, -16.381322957198407, ...]\n    '
    hn = df['high'].rolling(n).max()
    ln = df['low'].rolling(n).min()
    new_df = pd.DataFrame(data=list((hn - df['close']) / (hn - ln) * -100), columns=['wr'])
    return new_df

def RSI(df, n):
    if False:
        i = 10
        return i + 15
    '\n    相对强弱指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"rsi", 代表计算出来的相对强弱指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的相对强弱指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import RSI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        rsi = RSI(klines, 7)\n        print(list(rsi["rsi"]))\n\n        # 预计的输出是这样的:\n        [..., 80.21169825630794, 81.57315806032297, 72.34968324924667, ...]\n    '
    lc = df['close'].shift(1)
    rsi = tqsdk.tafunc.sma(pd.Series(np.where(df['close'] - lc > 0, df['close'] - lc, 0)), n, 1) / tqsdk.tafunc.sma(np.absolute(df['close'] - lc), n, 1) * 100
    new_df = pd.DataFrame(data=rsi, columns=['rsi'])
    return new_df

def ASI(df):
    if False:
        while True:
            i = 10
    '\n    振动升降指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"asi", 代表计算出来的振动升降指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的振动升降指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import ASI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        asi = ASI(klines)\n        print(list(asi["asi"]))\n\n\n        # 预计的输出是这样的:\n        [..., -4690.587005986468, -4209.182816350308, -4699.742010304962, ...]\n    '
    lc = df['close'].shift(1)
    aa = np.absolute(df['high'] - lc)
    bb = np.absolute(df['low'] - lc)
    cc = np.absolute(df['high'] - df['low'].shift(1))
    dd = np.absolute(lc - df['open'].shift(1))
    r = np.where((aa > bb) & (aa > cc), aa + bb / 2 + dd / 4, np.where((bb > cc) & (bb > aa), bb + aa / 2 + dd / 4, cc + dd / 4))
    x = df['close'] - lc + (df['close'] - df['open']) / 2 + lc - df['open'].shift(1)
    si = np.where(r == 0, 0, 16 * x / r * np.where(aa > bb, aa, bb))
    new_df = pd.DataFrame(data=list(pd.Series(si).cumsum()), columns=['asi'])
    return new_df

def VR(df, n):
    if False:
        print('Hello World!')
    '\n    VR 容量比率\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"vr", 代表计算出来的VR\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的VR\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import VR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        vr = VR(klines, 26)\n        print(list(vr["vr"]))\n\n\n        # 预计的输出是这样的:\n        [..., 150.1535316212112, 172.2897559521652, 147.04236342791924, ...]\n    '
    lc = df['close'].shift(1)
    vr = pd.Series(np.where(df['close'] > lc, df['volume'], 0)).rolling(n).sum() / pd.Series(np.where(df['close'] <= lc, df['volume'], 0)).rolling(n).sum() * 100
    new_df = pd.DataFrame(data=list(vr), columns=['vr'])
    return new_df

def ARBR(df, n):
    if False:
        return 10
    '\n    人气意愿指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"ar"和"br" , 分别代表人气指标和意愿指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的人气意愿指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import ARBR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        arbr = ARBR(klines, 26)\n        print(list(arbr["ar"]))\n        print(list(arbr["br"]))\n\n\n        # 预计的输出是这样的:\n        [..., 183.5698517817721, 189.98732572877034, 175.08802816901382, ...]\n        [..., 267.78549382716034, 281.567546278062, 251.08041091037902, ...]\n    '
    new_df = pd.DataFrame()
    new_df['ar'] = (df['high'] - df['open']).rolling(n).sum() / (df['open'] - df['low']).rolling(n).sum() * 100
    new_df['br'] = pd.Series(np.where(df['high'] - df['close'].shift(1) > 0, df['high'] - df['close'].shift(1), 0)).rolling(n).sum() / pd.Series(np.where(df['close'].shift(1) - df['low'] > 0, df['close'].shift(1) - df['low'], 0)).rolling(n).sum() * 100
    return new_df

def DMA(df, short, long, m):
    if False:
        print('Hello World!')
    '\n    平均线差\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        short (int): 短周期\n\n        long (int): 长周期\n\n        m (int): 计算周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"ddd"和"ama", 分别代表长短周期均值的差和ddd的简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的平均线差\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import DMA\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        dma = DMA(klines, 10, 50, 10)\n        print(list(dma["ddd"]))\n        print(list(dma["ama"]))\n\n\n        # 预计的输出是这样的:\n        [..., 409.2520000000022, 435.68000000000166, 458.3360000000025, ...]\n        [..., 300.64360000000147, 325.0860000000015, 349.75200000000166, ...]\n    '
    new_df = pd.DataFrame()
    new_df['ddd'] = tqsdk.tafunc.ma(df['close'], short) - tqsdk.tafunc.ma(df['close'], long)
    new_df['ama'] = tqsdk.tafunc.ma(new_df['ddd'], m)
    return new_df

def EXPMA(df, p1, p2):
    if False:
        print('Hello World!')
    '\n    指数加权移动平均线组合\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        p1 (int): 周期1\n\n        p2 (int): 周期2\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"ma1"和"ma2", 分别代表指数加权移动平均线1和指数加权移动平均线2\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的指数加权移动平均线组合\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import EXPMA\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        expma = EXPMA(klines, 5, 10)\n        print(list(expma["ma1"]))\n        print(list(expma["ma2"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3753.679549224137, 3784.6530328160916, 3792.7020218773946, ...]\n        [..., 3672.4492964832566, 3704.113060759028, 3723.1470497119317, ...]\n    '
    new_df = pd.DataFrame()
    new_df['ma1'] = tqsdk.tafunc.ema(df['close'], p1)
    new_df['ma2'] = tqsdk.tafunc.ema(df['close'], p2)
    return new_df

def CR(df, n, m):
    if False:
        return 10
    '\n    CR能量\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m (int): 周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"cr"和"crma", 分别代表CR值和CR值的简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的CR能量\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import CR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        cr = CR(klines, 26, 5)\n        print(list(cr["cr"]))\n        print(list(cr["crma"]))\n\n\n        # 预计的输出是这样的:\n        [..., 291.5751884671343, 316.71058105671943, 299.50578748862046, ...]\n        [..., 316.01257308163747, 319.3545725665982, 311.8275184876805, ...]\n    '
    new_df = pd.DataFrame()
    mid = (df['high'] + df['low'] + df['close']) / 3
    new_df['cr'] = pd.Series(np.where(0 > df['high'] - mid.shift(1), 0, df['high'] - mid.shift(1))).rolling(n).sum() / pd.Series(np.where(0 > mid.shift(1) - df['low'], 0, mid.shift(1) - df['low'])).rolling(n).sum() * 100
    new_df['crma'] = tqsdk.tafunc.ma(new_df['cr'], m).shift(int(m / 2.5 + 1))
    return new_df

def CCI(df, n):
    if False:
        print('Hello World!')
    '\n    顺势指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"cci", 代表计算出来的CCI值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的顺势指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import CCI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        cci = CCI(klines, 14)\n        print(list(cci["cci"]))\n\n\n        # 预计的输出是这样的:\n        [..., 98.13054698810375, 93.57661788413617, 77.8671380173813, ...]\n    '
    typ = (df['high'] + df['low'] + df['close']) / 3
    ma = tqsdk.tafunc.ma(typ, n)

    def mad(x):
        if False:
            return 10
        return np.fabs(x - x.mean()).mean()
    md = typ.rolling(window=n).apply(mad, raw=True)
    new_df = pd.DataFrame(data=list((typ - ma) / (md * 0.015)), columns=['cci'])
    return new_df

def OBV(df):
    if False:
        for i in range(10):
            print('nop')
    '\n    能量潮\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"obv", 代表计算出来的OBV值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的能量潮\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import OBV\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        obv = OBV(klines)\n        print(list(obv["obv"]))\n\n\n        # 预计的输出是这样的:\n        [..., 267209, 360351, 264476, ...]\n    '
    lc = df['close'].shift(1)
    obv = np.where(df['close'] > lc, df['volume'], np.where(df['close'] < lc, -df['volume'], 0)).cumsum()
    new_df = pd.DataFrame(data=obv, columns=['obv'])
    return new_df

def CDP(df, n):
    if False:
        print('Hello World!')
    '\n    逆势操作\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含4列, 是"ah", "al", "nh", "nl", 分别代表最高值, 最低值, 近高值, 近低值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的逆势操作指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import CDP\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        cdp = CDP(klines, 3)\n        print(list(cdp["ah"]))\n        print(list(cdp["al"]))\n        print(list(cdp["nh"]))\n        print(list(cdp["nl"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3828.244444444447, 3871.733333333336, 3904.37777777778, ...]\n        [..., 3656.64444444444, 3698.3999999999955, 3734.9111111111065, ...]\n        [..., 3743.8888888888837, 3792.3999999999946, 3858.822222222217, ...]\n        [..., 3657.2222222222213, 3707.6666666666656, 3789.955555555554, ...]\n    '
    new_df = pd.DataFrame()
    pt = df['high'].shift(1) - df['low'].shift(1)
    cdp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    new_df['ah'] = tqsdk.tafunc.ma(cdp + pt, n)
    new_df['al'] = tqsdk.tafunc.ma(cdp - pt, n)
    new_df['nh'] = tqsdk.tafunc.ma(2 * cdp - df['low'], n)
    new_df['nl'] = tqsdk.tafunc.ma(2 * cdp - df['high'], n)
    return new_df

def HCL(df, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    均线通道\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含3列, 是"mah", "mal", "mac", 分别代表最高价的移动平均线, 最低价的移动平均线以及收盘价的移动平均线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的均线通道指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import HCL\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        hcl = HCL(klines, 10)\n        print(list(hcl["mah"]))\n        print(list(hcl["mal"]))\n        print(list(hcl["mac"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3703.5400000000022, 3743.2800000000025, 3778.300000000002, ...]\n        [..., 3607.339999999999, 3643.079999999999, 3677.579999999999, ...]\n        [..., 3666.1600000000008, 3705.8600000000006, 3741.940000000001, ...]\n    '
    new_df = pd.DataFrame()
    new_df['mah'] = tqsdk.tafunc.ma(df['high'], n)
    new_df['mal'] = tqsdk.tafunc.ma(df['low'], n)
    new_df['mac'] = tqsdk.tafunc.ma(df['close'], n)
    return new_df

def ENV(df, n, k):
    if False:
        return 10
    '\n    包略线 (Envelopes)\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        k (float): 参数k\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"upper", "lower", 分别代表上线和下线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的包略线\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import ENV\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        env = ENV(klines, 14, 6)\n        print(list(env["upper"]))\n        print(list(env["lower"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3842.2122857142863, 3876.7531428571433, 3893.849428571429, ...]\n        [..., 3407.244857142857, 3437.875428571429, 3453.036285714286, ...]\n    '
    new_df = pd.DataFrame()
    new_df['upper'] = tqsdk.tafunc.ma(df['close'], n) * (1 + k / 100)
    new_df['lower'] = tqsdk.tafunc.ma(df['close'], n) * (1 - k / 100)
    return new_df

def MIKE(df, n):
    if False:
        while True:
            i = 10
    '\n    麦克指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含6列, 是"wr", "mr", "sr", "ws", "ms", "ss", 分别代表初级压力价,中级压力,强力压力,初级支撑,中级支撑和强力支撑\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的麦克指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import MIKE\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        mike = MIKE(klines, 12)\n        print(list(mike["wr"]))\n        print(list(mike["mr"]))\n        print(list(mike["sr"]))\n        print(list(mike["ws"]))\n        print(list(mike["ms"]))\n        print(list(mike["ss"]))\n\n\n        # 预计的输出是这样的:\n        [..., 4242.4, 4203.333333333334, 3986.266666666666, ...]\n        [..., 4303.6, 4283.866666666667, 4175.333333333333, ...]\n        [..., 4364.8, 4364.4, 4364.4, ...]\n        [..., 3770.5999999999995, 3731.9333333333343, 3514.866666666666, ...]\n        [..., 3359.9999999999995, 3341.066666666667, 3232.533333333333, ...]\n        [..., 2949.3999999999996, 2950.2, 2950.2, ...]\n    '
    new_df = pd.DataFrame()
    typ = (df['high'] + df['low'] + df['close']) / 3
    ll = df['low'].rolling(n).min()
    hh = df['high'].rolling(n).max()
    new_df['wr'] = typ + (typ - ll)
    new_df['mr'] = typ + (hh - ll)
    new_df['sr'] = 2 * hh - ll
    new_df['ws'] = typ - (hh - typ)
    new_df['ms'] = typ - (hh - ll)
    new_df['ss'] = 2 * ll - hh
    return new_df

def PUBU(df, m):
    if False:
        while True:
            i = 10
    '\n    瀑布线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        m (int): 周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"pb", 代表计算出的瀑布线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的瀑布线\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import PUBU\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        pubu = PUBU(klines, 4)\n        print(list(pubu["pb"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3719.087702972829, 3728.9326217836974, 3715.7537397368856, ...]\n    '
    pb = (tqsdk.tafunc.ema(df['close'], m) + tqsdk.tafunc.ma(df['close'], m * 2) + tqsdk.tafunc.ma(df['close'], m * 4)) / 3
    new_df = pd.DataFrame(data=list(pb), columns=['pb'])
    return new_df

def BBI(df, n1, n2, n3, n4):
    if False:
        for i in range(10):
            print('nop')
    '\n    多空指数\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n1 (int): 周期n1\n\n        n2 (int): 周期n2\n\n        n3 (int): 周期n3\n\n        n4 (int): 周期n4\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"bbi", 代表计算出的多空指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的多空指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import BBI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        bbi = BBI(klines, 3, 6, 12, 24)\n        print(list(bbi["bbi"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3679.841666666668, 3700.9645833333348, 3698.025000000002, ...]\n    '
    bbi = (tqsdk.tafunc.ma(df['close'], n1) + tqsdk.tafunc.ma(df['close'], n2) + tqsdk.tafunc.ma(df['close'], n3) + tqsdk.tafunc.ma(df['close'], n4)) / 4
    new_df = pd.DataFrame(data=list(bbi), columns=['bbi'])
    return new_df

def DKX(df, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    多空线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        m (int): 周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"b", "d", 分别代表计算出来的DKX指标及DKX的m日简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的多空线\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import DKX\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        dkx = DKX(klines, 10)\n        print(list(dkx["b"]))\n        print(list(dkx["d"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3632.081746031746, 3659.4501587301593, 3672.744761904762, ...]\n        [..., 3484.1045714285706, 3516.1797301587294, 3547.44857142857, ...]\n    '
    new_df = pd.DataFrame()
    a = (3 * df['close'] + df['high'] + df['low'] + df['open']) / 6
    new_df['b'] = (20 * a + 19 * a.shift(1) + 18 * a.shift(2) + 17 * a.shift(3) + 16 * a.shift(4) + 15 * a.shift(5) + 14 * a.shift(6) + 13 * a.shift(7) + 12 * a.shift(8) + 11 * a.shift(9) + 10 * a.shift(10) + 9 * a.shift(11) + 8 * a.shift(12) + 7 * a.shift(13) + 6 * a.shift(14) + 5 * a.shift(15) + 4 * a.shift(16) + 3 * a.shift(17) + 2 * a.shift(18) + a.shift(20)) / 210
    new_df['d'] = tqsdk.tafunc.ma(new_df['b'], m)
    return new_df

def BBIBOLL(df, n, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    多空布林线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n        m (int): 参数m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含3列, 是"bbiboll", "upr", "dwn", 分别代表多空布林线, 压力线和支撑线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的多空布林线\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import BBIBOLL\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        bbiboll=BBIBOLL(klines,10,3)\n        print(list(bbiboll["bbiboll"]))\n        print(list(bbiboll["upr"]))\n        print(list(bbiboll["dwn"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3679.841666666668, 3700.9645833333348, 3698.025000000002, ...]\n        [..., 3991.722633271389, 3991.796233444868, 3944.7721466057383, ...]\n        [..., 3367.960700061947, 3410.1329332218015, 3451.2778533942655, ...]\n    '
    new_df = pd.DataFrame()
    new_df['bbiboll'] = (tqsdk.tafunc.ma(df['close'], 3) + tqsdk.tafunc.ma(df['close'], 6) + tqsdk.tafunc.ma(df['close'], 12) + tqsdk.tafunc.ma(df['close'], 24)) / 4
    new_df['upr'] = new_df['bbiboll'] + m * new_df['bbiboll'].rolling(n).std()
    new_df['dwn'] = new_df['bbiboll'] - m * new_df['bbiboll'].rolling(n).std()
    return new_df

def ADTM(df, n, m):
    if False:
        print('Hello World!')
    '\n    动态买卖气指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m (int): 周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"adtm", "adtmma", 分别代表计算出来的ADTM指标及其M日的简单移动平均\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的动态买卖气指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import ADTM\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        adtm = ADTM(klines, 23, 8)\n        print(list(adtm["adtm"]))\n        print(list(adtm["adtmma"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.8404011965511171, 0.837919942816297, 0.8102215868477481, ...]\n        [..., 0.83855483869397, 0.8354743499113684, 0.8257261282040207, ...]\n    '
    new_df = pd.DataFrame()
    dtm = np.where(df['open'] < df['open'].shift(1), 0, np.where(df['high'] - df['open'] > df['open'] - df['open'].shift(1), df['high'] - df['open'], df['open'] - df['open'].shift(1)))
    dbm = np.where(df['open'] >= df['open'].shift(1), 0, np.where(df['open'] - df['low'] > df['open'] - df['open'].shift(1), df['open'] - df['low'], df['open'] - df['open'].shift(1)))
    stm = pd.Series(dtm).rolling(n).sum()
    sbm = pd.Series(dbm).rolling(n).sum()
    new_df['adtm'] = np.where(stm > sbm, (stm - sbm) / stm, np.where(stm == sbm, 0, (stm - sbm) / sbm))
    new_df['adtmma'] = tqsdk.tafunc.ma(new_df['adtm'], m)
    return new_df

def B3612(df):
    if False:
        while True:
            i = 10
    '\n    三减六日乖离率\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"b36", "b612", 分别代表收盘价的3日移动平均线与6日移动平均线的乖离值及收盘价的6日移动平均线与12日移动平均线的乖离值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的三减六日乖离率\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import B3612\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        b3612=B3612(klines)\n        print(list(b3612["b36"]))\n        print(list(b3612["b612"]))\n\n\n        # 预计的输出是这样的:\n        [..., 57.26666666667188, 44.00000000000546, -5.166666666660603, ...]\n        [..., 99.28333333333285, 88.98333333333221, 69.64999999999918, ...]\n    '
    new_df = pd.DataFrame()
    new_df['b36'] = tqsdk.tafunc.ma(df['close'], 3) - tqsdk.tafunc.ma(df['close'], 6)
    new_df['b612'] = tqsdk.tafunc.ma(df['close'], 6) - tqsdk.tafunc.ma(df['close'], 12)
    return new_df

def DBCD(df, n, m, t):
    if False:
        print('Hello World!')
    '\n    异同离差乖离率\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m (int): 参数m\n\n        t (int): 参数t\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"dbcd", "mm", 分别代表离差值及其简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的异同离差乖离率\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import DBCD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        dbcd=DBCD(klines, 5, 16, 76)\n        print(list(dbcd["dbcd"]))\n        print(list(dbcd["mm"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.0038539724453411045, 0.0034209659500908517, 0.0027130669520015094, ...]\n        [..., 0.003998499673401192, 0.003864353204606074, 0.0035925052896395872, ...]\n    '
    new_df = pd.DataFrame()
    bias = (df['close'] - tqsdk.tafunc.ma(df['close'], n)) / tqsdk.tafunc.ma(df['close'], n)
    dif = bias - bias.shift(m)
    new_df['dbcd'] = tqsdk.tafunc.sma(dif, t, 1)
    new_df['mm'] = tqsdk.tafunc.ma(new_df['dbcd'], 5)
    return new_df

def DDI(df, n, n1, m, m1):
    if False:
        while True:
            i = 10
    '\n    方向标准离差指数\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        n1 (int): 参数n1\n\n        m (int): 参数m\n\n        m1 (int): 周期m1\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含3列, 是"ddi", "addi", "ad", 分别代表DIZ与DIF的差值, DDI的加权平均, ADDI的简单移动平均\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的方向标准离差指数\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import DDI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ddi = DDI(klines, 13, 30, 10, 5)\n        print(list(ddi["ddi"]))\n        print(list(ddi["addi"]))\n        print(list(ddi["ad"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.6513560804899388, 0.6129178985672046, 0.40480202190395936, ...]\n        [..., 0.6559570156346113, 0.6416106432788091, 0.5626744361538593, ...]\n        [..., 0.6960565490556135, 0.6765004585407994, 0.6455063893920429, ...]\n    '
    new_df = pd.DataFrame()
    tr = np.where(np.absolute(df['high'] - df['high'].shift(1)) > np.absolute(df['low'] - df['low'].shift(1)), np.absolute(df['high'] - df['high'].shift(1)), np.absolute(df['low'] - df['low'].shift(1)))
    dmz = np.where(df['high'] + df['low'] <= df['high'].shift(1) + df['low'].shift(1), 0, tr)
    dmf = np.where(df['high'] + df['low'] >= df['high'].shift(1) + df['low'].shift(1), 0, tr)
    diz = pd.Series(dmz).rolling(n).sum() / (pd.Series(dmz).rolling(n).sum() + pd.Series(dmf).rolling(n).sum())
    dif = pd.Series(dmf).rolling(n).sum() / (pd.Series(dmf).rolling(n).sum() + pd.Series(dmz).rolling(n).sum())
    new_df['ddi'] = diz - dif
    new_df['addi'] = tqsdk.tafunc.sma(new_df['ddi'], n1, m)
    new_df['ad'] = tqsdk.tafunc.ma(new_df['addi'], m1)
    return new_df

def KD(df, n, m1, m2):
    if False:
        return 10
    '\n    随机指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m1 (int): 参数m1\n\n        m2 (int): 参数m2\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"k", "d", 分别代表计算出来的K值与D值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的随机指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import KD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        kd = KD(klines, 9, 3, 3)\n        print(list(kd["k"]))\n        print(list(kd["d"]))\n\n\n        # 预计的输出是这样的:\n        [..., 84.60665654726242, 80.96145249909222, 57.54863147922147, ...]\n        [..., 82.98108443995415, 82.30787379300017, 74.05479302174061, ...]\n    '
    new_df = pd.DataFrame()
    hv = df['high'].rolling(n).max()
    lv = df['low'].rolling(n).min()
    rsv = pd.Series(np.where(hv == lv, 0, (df['close'] - lv) / (hv - lv) * 100))
    new_df['k'] = tqsdk.tafunc.sma(rsv, m1, 1)
    new_df['d'] = tqsdk.tafunc.sma(new_df['k'], m2, 1)
    return new_df

def LWR(df, n, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    威廉指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m (int): 参数m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"lwr", 代表计算出来的威廉指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的威廉指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import LWR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        lwr = LWR(klines, 9, 3)\n        print(list(lwr["lwr"]))\n\n\n        # 预计的输出是这样的:\n        [..., -15.393343452737565, -19.03854750090778, -42.45136852077853, ...]\n    '
    hv = df['high'].rolling(n).max()
    lv = df['low'].rolling(n).min()
    rsv = pd.Series(np.where(hv == lv, 0, (df['close'] - hv) / (hv - lv) * 100))
    new_df = pd.DataFrame(data=list(tqsdk.tafunc.sma(rsv, m, 1)), columns=['lwr'])
    return new_df

def MASS(df, n1, n2):
    if False:
        i = 10
        return i + 15
    '\n    梅斯线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n1 (int): 周期n1\n\n        n2 (int): 周期n2\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"mass", 代表计算出来的梅斯线指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的梅斯线\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import MASS\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        mass = MASS(klines, 9, 25)\n        print(list(mass["mass"]))\n\n\n        # 预计的输出是这样的:\n        [..., 27.478822053291733, 27.485710830466964, 27.561223922342652, ...]\n    '
    ema1 = tqsdk.tafunc.ema(df['high'] - df['low'], n1)
    ema2 = tqsdk.tafunc.ema(ema1, n1)
    new_df = pd.DataFrame(data=list((ema1 / ema2).rolling(n2).sum()), columns=['mass'])
    return new_df

def MFI(df, n):
    if False:
        return 10
    '\n    资金流量指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"mfi", 代表计算出来的MFI指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的资金流量指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import MFI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        mfi = MFI(klines, 14)\n        print(list(mfi["mfi"]))\n\n\n        # 预计的输出是这样的:\n        [..., 73.47968487105688, 70.2250476611595, 62.950450871062266, ...]\n    '
    typ = (df['high'] + df['low'] + df['close']) / 3
    mr = pd.Series(np.where(typ > typ.shift(1), typ * df['volume'], 0)).rolling(n).sum() / pd.Series(np.where(typ < typ.shift(1), typ * df['volume'], 0)).rolling(n).sum()
    new_df = pd.DataFrame(data=list(100 - 100 / (1 + mr)), columns=['mfi'])
    return new_df

def MI(df, n):
    if False:
        while True:
            i = 10
    '\n    动量指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"a", "mi", 分别代表当日收盘价与N日前收盘价的差值以及MI值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的动量指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import MI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        mi = MI(klines, 12)\n        print(list(mi["a"]))\n        print(list(mi["mi"]))\n\n\n        # 预计的输出是这样的:\n        [..., 399.1999999999998, 370.8000000000002, 223.5999999999999, ...]\n        [..., 293.2089214076506, 299.67484462367975, 293.3352742383731, ...]\n    '
    new_df = pd.DataFrame()
    new_df['a'] = df['close'] - df['close'].shift(n)
    new_df['mi'] = tqsdk.tafunc.sma(new_df['a'], n, 1)
    return new_df

def MICD(df, n, n1, n2):
    if False:
        for i in range(10):
            print('nop')
    '\n    异同离差动力指数\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n        n1 (int): 周期n1\n\n        n2 (int): 周期n2\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"dif", "micd", 代表离差值和MICD指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的异同离差动力指数\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import MICD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        micd = MICD(klines, 3, 10, 20)\n        print(list(micd["dif"]))\n        print(list(micd["micd"]))\n\n\n        # 预计的输出是这样的:\n        [..., 6.801483500680234, 6.700989000453493, 6.527326000302342, ...]\n        [..., 6.2736377238314684, 6.3163728514936714, 6.3374681663745385, ...]\n    '
    new_df = pd.DataFrame()
    mi = df['close'] - df['close'].shift(1)
    ami = tqsdk.tafunc.sma(mi, n, 1)
    new_df['dif'] = tqsdk.tafunc.ma(ami.shift(1), n1) - tqsdk.tafunc.ma(ami.shift(1), n2)
    new_df['micd'] = tqsdk.tafunc.sma(new_df['dif'], 10, 1)
    return new_df

def MTM(df, n, n1):
    if False:
        print('Hello World!')
    '\n    MTM动力指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        n1 (int): 周期n1\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"mtm", "mtmma", 分别代表MTM值和MTM的简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的动力指标\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import MTM\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        mtm = MTM(klines, 6, 6)\n        print(list(mtm["mtm"]))\n        print(list(mtm["mtmma"]))\n\n\n        # 预计的输出是这样的:\n        [..., 144.79999999999973, 123.60000000000036, -4.200000000000273, ...]\n        [..., 198.5666666666667, 177.96666666666678, 139.30000000000004, ...]\n    '
    new_df = pd.DataFrame()
    new_df['mtm'] = df['close'] - df['close'].shift(n)
    new_df['mtmma'] = tqsdk.tafunc.ma(new_df['mtm'], n1)
    return new_df

def PRICEOSC(df, long, short):
    if False:
        print('Hello World!')
    '\n    价格震荡指数 Price Oscillator\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        long (int): 长周期\n\n        short (int): 短周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"priceosc", 代表计算出来的价格震荡指数\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的价格震荡指数\n        from tqsdk import TqApi, TqAuth, TqSim\n        from tqsdk.ta import PRICEOSC\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        priceosc = PRICEOSC(klines, 26, 12)\n        print(list(priceosc["priceosc"]))\n\n\n        # 预计的输出是这样的:\n        [..., 5.730468338384374, 5.826866231225718, 5.776959240989803, ...]\n    '
    ma_s = tqsdk.tafunc.ma(df['close'], short)
    ma_l = tqsdk.tafunc.ma(df['close'], long)
    new_df = pd.DataFrame(data=list((ma_s - ma_l) / ma_s * 100), columns=['priceosc'])
    return new_df

def PSY(df, n, m):
    if False:
        return 10
    '\n    心理线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m (int): 周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"psy", "psyma", 分别代表心理线和心理线的简单移动平均\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的心理线\n        from tqsdk import TqApi, TqSim\n        from tqsdk.ta import PSY\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        psy = PSY(klines, 12, 6)\n        print(list(psy["psy"]))\n        print(list(psy["psyma"]))\n\n\n        # 预计的输出是这样的:\n        [..., 58.333333333333336, 58.333333333333336, 50.0, ...]\n        [..., 54.16666666666671, 54.16666666666671, 54.16666666666671, ...]\n    '
    new_df = pd.DataFrame()
    new_df['psy'] = tqsdk.tafunc.count(df['close'] > df['close'].shift(1), n) / n * 100
    new_df['psyma'] = tqsdk.tafunc.ma(new_df['psy'], m)
    return new_df

def QHLSR(df):
    if False:
        return 10
    '\n    阻力指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"qhl5", "qhl10", 分别代表计算出来的QHL5值和QHL10值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的阻力指标\n        from tqsdk import TqApi, TqSim\n        from tqsdk.ta import QHLSR\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ndf = QHLSR(klines)\n        print(list(ndf["qhl5"]))\n        print(list(ndf["qhl10"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.9512796890171819, 1.0, 0.8061319699743583, 0.36506038490240567, ...]\n        [..., 0.8192641975527878, 0.7851545532504415, 0.5895613967067044, ...]\n    '
    new_df = pd.DataFrame()
    qhl = df['close'] - df['close'].shift(1) - (df['volume'] - df['volume'].shift(1)) * (df['high'].shift(1) - df['low'].shift(1)) / df['volume'].shift(1)
    a = pd.Series(np.where(qhl > 0, qhl, 0)).rolling(5).sum()
    e = pd.Series(np.where(qhl > 0, qhl, 0)).rolling(10).sum()
    b = np.absolute(pd.Series(np.where(qhl < 0, qhl, 0)).rolling(5).sum())
    f = np.absolute(pd.Series(np.where(qhl < 0, qhl, 0)).rolling(10).sum())
    d = a / (a + b)
    g = e / (e + f)
    new_df['qhl5'] = np.where(pd.Series(np.where(qhl > 0, 1, 0)).rolling(5).sum() == 5, 1, np.where(pd.Series(np.where(qhl < 0, 1, 0)).rolling(5).sum() == 5, 0, d))
    new_df['qhl10'] = g
    return new_df

def RC(df, n):
    if False:
        i = 10
        return i + 15
    '\n    变化率指数\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"arc", 代表计算出来的变化率指数\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的变化率指数\n        from tqsdk import TqApi, TqSim\n        from tqsdk.ta import RC\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        rc = RC(klines, 50)\n        print(list(rc["arc"]))\n\n\n        # 预计的输出是这样的:\n        [..., 1.011782057069131, 1.0157160672001329, 1.019680175228899, ...]\n    '
    rc = df['close'] / df['close'].shift(n)
    new_df = pd.DataFrame(data=list(tqsdk.tafunc.sma(rc.shift(1), n, 1)), columns=['arc'])
    return new_df

def RCCD(df, n, n1, n2):
    if False:
        while True:
            i = 10
    '\n    异同离差变化率指数\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n        n1 (int): 周期n1\n\n        n2 (int): 周期n2\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"dif", "rccd", 分别代表离差值和异同离差变化率指数\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的异同离差变化率指数\n        from tqsdk import TqApi, TqSim\n        from tqsdk.ta import RCCD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        rccd = RCCD(klines, 10, 21, 28)\n        print(list(rccd["dif"]))\n        print(list(rccd["rccd"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.007700543190044096, 0.007914865667604465, 0.008297381119103608, ...]\n        [..., 0.007454465277084111, 0.007500505316136147, 0.0075801928964328935, ...]\n    '
    new_df = pd.DataFrame()
    rc = df['close'] / df['close'].shift(n)
    arc = tqsdk.tafunc.sma(rc.shift(1), n, 1)
    new_df['dif'] = tqsdk.tafunc.ma(arc.shift(1), n1) - tqsdk.tafunc.ma(arc.shift(1), n2)
    new_df['rccd'] = tqsdk.tafunc.sma(new_df['dif'], n, 1)
    return new_df

def ROC(df, n, m):
    if False:
        i = 10
        return i + 15
    '\n    变动速率\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n        m (int): 周期m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"roc", "rocma", 分别代表ROC值和ROC的简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的变动速率\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import ROC\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        roc = ROC(klines, 24, 20)\n        print(list(roc["roc"]))\n        print(list(roc["rocma"]))\n\n\n        # 预计的输出是这样的:\n        [..., 21.389800555415288, 19.285937989351712, 15.183443085606768, ...]\n        [..., 14.597071588550435, 15.223202630466648, 15.537530180238516, ...]\n    '
    new_df = pd.DataFrame()
    new_df['roc'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n) * 100
    new_df['rocma'] = tqsdk.tafunc.ma(new_df['roc'], m)
    return new_df

def SLOWKD(df, n, m1, m2, m3):
    if False:
        return 10
    '\n    慢速KD\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 周期n\n\n        m1 (int): 参数m1\n\n        m2 (int): 参数m2\n\n        m3 (int): 参数m3\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"k", "d", 分别代表K值和D值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的慢速KD\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import SLOWKD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        slowkd = SLOWKD(klines, 9, 3, 3, 3)\n        print(list(slowkd["k"]))\n        print(list(slowkd["d"]))\n\n\n        # 预计的输出是这样的:\n        [..., 82.98108443995415, 82.30787379300017, 74.05479302174061, ...]\n        [..., 83.416060393041, 83.04666485969405, 80.0493742470429, ...]\n    '
    new_df = pd.DataFrame()
    rsv = (df['close'] - df['low'].rolling(n).min()) / (df['high'].rolling(n).max() - df['low'].rolling(n).min()) * 100
    fastk = tqsdk.tafunc.sma(rsv, m1, 1)
    new_df['k'] = tqsdk.tafunc.sma(fastk, m2, 1)
    new_df['d'] = tqsdk.tafunc.sma(new_df['k'], m3, 1)
    return new_df

def SRDM(df, n):
    if False:
        return 10
    '\n    动向速度比率\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"srdm", "asrdm", 分别代表计算出来的SRDM值和SRDM值的加权移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的动向速度比率\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import SRDM\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        srdm = SRDM(klines, 30)\n        print(list(srdm["srdm"]))\n        print(list(srdm["asrdm"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.7865067466266866, 0.7570567713288928, 0.5528619528619526, ...]\n        [..., 0.45441550541510667, 0.4645035476122329, 0.4674488277872236, ...]\n    '
    new_df = pd.DataFrame()
    dmz = np.where(df['high'] + df['low'] <= df['high'].shift(1) + df['low'].shift(1), 0, np.where(np.absolute(df['high'] - df['high'].shift(1)) > np.absolute(df['low'] - df['low'].shift(1)), np.absolute(df['high'] - df['high'].shift(1)), np.absolute(df['low'] - df['low'].shift(1))))
    dmf = np.where(df['high'] + df['low'] >= df['high'].shift(1) + df['low'].shift(1), 0, np.where(np.absolute(df['high'] - df['high'].shift(1)) > np.absolute(df['low'] - df['low'].shift(1)), np.absolute(df['high'] - df['high'].shift(1)), np.absolute(df['low'] - df['low'].shift(1))))
    admz = tqsdk.tafunc.ma(pd.Series(dmz), 10)
    admf = tqsdk.tafunc.ma(pd.Series(dmf), 10)
    new_df['srdm'] = np.where(admz > admf, (admz - admf) / admz, np.where(admz == admf, 0, (admz - admf) / admf))
    new_df['asrdm'] = tqsdk.tafunc.sma(new_df['srdm'], n, 1)
    return new_df

def SRMI(df, n):
    if False:
        i = 10
        return i + 15
    '\n    MI修正指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"a", "mi", 分别代表A值和MI值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的MI修正指标\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import SRMI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        srmi = SRMI(klines, 9)\n        print(list(srmi["a"]))\n        print(list(srmi["mi"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.10362397961836425, 0.07062591892459567, -0.03341929372138309, ...]\n        [..., 0.07583104758041452, 0.0752526999519902, 0.06317803398828206, ...]\n    '
    new_df = pd.DataFrame()
    new_df['a'] = np.where(df['close'] < df['close'].shift(n), (df['close'] - df['close'].shift(n)) / df['close'].shift(n), np.where(df['close'] == df['close'].shift(n), 0, (df['close'] - df['close'].shift(n)) / df['close']))
    new_df['mi'] = tqsdk.tafunc.sma(new_df['a'], n, 1)
    return new_df

def ZDZB(df, n1, n2, n3):
    if False:
        i = 10
        return i + 15
    '\n    筑底指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n1 (int): 周期n1\n\n        n2 (int): 周期n2\n\n        n3 (int): 周期n3\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"b", "d", 分别代表A值的n2周期简单移动平均和A值的n3周期简单移动平均\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的筑底指标\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import ZDZB\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        zdzb = ZDZB(klines, 50, 5, 20)\n        print(list(zdzb["b"]))\n        print(list(zdzb["d"]))\n\n\n        # 预计的输出是这样的:\n        [..., 1.119565217391305, 1.1376811594202905, 1.155797101449276, ...]\n        [..., 1.0722350515828771, 1.091644989471076, 1.1077480490523965, ...]\n    '
    new_df = pd.DataFrame()
    a = pd.Series(np.where(df['close'] >= df['close'].shift(1), 1, 0)).rolling(n1).sum() / pd.Series(np.where(df['close'] < df['close'].shift(1), 1, 0)).rolling(n1).sum()
    new_df['b'] = tqsdk.tafunc.ma(a, n2)
    new_df['d'] = tqsdk.tafunc.ma(a, n3)
    return new_df

def DPO(df):
    if False:
        print('Hello World!')
    '\n    区间震荡线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"dpo", 代表计算出来的DPO指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的区间震荡线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import DPO\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        dpo = DPO(klines)\n        print(list(dpo["dpo"]))\n\n\n        # 预计的输出是这样的:\n        [..., 595.4100000000021, 541.8300000000017, 389.7200000000016, ...]\n    '
    dpo = df['close'] - tqsdk.tafunc.ma(df['close'], 20).shift(11)
    new_df = pd.DataFrame(data=list(dpo), columns=['dpo'])
    return new_df

def LON(df):
    if False:
        while True:
            i = 10
    '\n    长线指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"lon", "ma1", 分别代表长线指标和长线指标的10周期简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的长线指标\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import LON\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        lon = LON(klines)\n        print(list(lon["lon"]))\n        print(list(lon["ma1"]))\n\n\n        # 预计的输出是这样的:\n        [..., 6.419941948913239, 6.725451135494827, 6.483546043406369, ...]\n        [..., 4.366625464410439, 4.791685949556344, 5.149808865745246, ...]\n    '
    new_df = pd.DataFrame()
    tb = np.where(df['high'] > df['close'].shift(1), df['high'] - df['close'].shift(1) + df['close'] - df['low'], df['close'] - df['low'])
    ts = np.where(df['close'].shift(1) > df['low'], df['close'].shift(1) - df['low'] + df['high'] - df['close'], df['high'] - df['close'])
    vol1 = (tb - ts) * df['volume'] / (tb + ts) / 10000
    vol10 = vol1.ewm(alpha=0.1, adjust=False).mean()
    vol11 = vol1.ewm(alpha=0.05, adjust=False).mean()
    res1 = vol10 - vol11
    new_df['lon'] = res1.cumsum()
    new_df['ma1'] = tqsdk.tafunc.ma(new_df['lon'], 10)
    return new_df

def SHORT(df):
    if False:
        print('Hello World!')
    '\n    短线指标\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"short", "ma1", 分别代表短线指标和短线指标的10周期简单移动平均值\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的短线指标\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import SHORT\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        short = SHORT(klines)\n        print(list(short["short"]))\n        print(list(short["ma1"]))\n\n\n        # 预计的输出是这样的:\n        [..., 0.6650139934614072, 0.3055091865815881, -0.24190509208845834, ...]\n        [..., 0.41123378999608917, 0.42506048514590444, 0.35812291618890224, ...]\n    '
    new_df = pd.DataFrame()
    tb = np.where(df['high'] > df['close'].shift(1), df['high'] - df['close'].shift(1) + df['close'] - df['low'], df['close'] - df['low'])
    ts = np.where(df['close'].shift(1) > df['low'], df['close'].shift(1) - df['low'] + df['high'] - df['close'], df['high'] - df['close'])
    vol1 = (tb - ts) * df['volume'] / (tb + ts) / 10000
    vol10 = vol1.ewm(alpha=0.1, adjust=False).mean()
    vol11 = vol1.ewm(alpha=0.05, adjust=False).mean()
    new_df['short'] = vol10 - vol11
    new_df['ma1'] = tqsdk.tafunc.ma(new_df['short'], 10)
    return new_df

def MV(df, n, m):
    if False:
        while True:
            i = 10
    '\n    均量线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n        m (int): 参数m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"mv1", "mv2", 分别代表均量线1和均量线2\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的均量线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import MV\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        mv = MV(klines, 10, 20)\n        print(list(mv["mv1"]))\n        print(list(mv["mv2"]))\n\n\n        # 预计的输出是这样的:\n        [..., 69851.39419881169, 72453.75477893051, 75423.57930103746, ...]\n        [..., 49044.75870654942, 51386.27077122195, 53924.557232660845, ...]\n    '
    new_df = pd.DataFrame()
    new_df['mv1'] = tqsdk.tafunc.sma(df['volume'], n, 1)
    new_df['mv2'] = tqsdk.tafunc.sma(df['volume'], m, 1)
    return new_df

def WAD(df, n, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    威廉多空力度线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n        m (int): 参数m\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含3列, 是"a", "b", "e", 分别代表A/D值,A/D值n周期的以1为权重的移动平均, A/D值m周期的以1为权重的移动平均\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的威廉多空力度线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import WAD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        wad = WAD(klines, 10, 30)\n        print(list(wad["a"]))\n        print(list(wad["b"]))\n        print(list(wad["e"]))\n\n\n        # 预计的输出是这样的:\n        [..., 90.0, 134.79999999999973, 270.4000000000001, ...]\n        [..., 344.4265821851701, 323.46392396665306, 318.1575315699878, ...]\n        [..., 498.75825781872277, 486.626315891432, 479.41877202838424, ...]\n    '
    new_df = pd.DataFrame()
    new_df['a'] = np.absolute(np.where(df['close'] > df['close'].shift(1), df['close'] - np.where(df['close'].shift(1) < df['low'], df['close'].shift(1), df['low']), np.where(df['close'] < df['close'].shift(1), df['close'] - np.where(df['close'].shift(1) > df['high'], df['close'].shift(1), df['high']), 0)).cumsum())
    new_df['b'] = tqsdk.tafunc.sma(new_df['a'], n, 1)
    new_df['e'] = tqsdk.tafunc.sma(new_df['a'], m, 1)
    return new_df

def AD(df):
    if False:
        for i in range(10):
            print('nop')
    '\n    累积/派发指标 Accumulation/Distribution\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"ad", 代表计算出来的累积/派发指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的累积/派发指标\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import AD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ad = AD(klines)\n        print(list(ad["ad"]))\n\n\n        # 预计的输出是这样的:\n        [..., 146240.57181105542, 132822.950945916, 49768.15024044845, ...]\n    '
    ad = ((df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']).cumsum()
    new_df = pd.DataFrame(data=list(ad), columns=['ad'])
    return new_df

def CCL(df):
    if False:
        i = 10
        return i + 15
    '\n    持仓异动\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"ccl", 代表计算出来的持仓异动指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的持仓异动指标\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import CCL\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ccl = CCL(klines)\n        print(list(ccl["ccl"]))\n\n\n        # 预计的输出是这样的:\n        [..., \'多头增仓\', \'多头减仓\', \'空头增仓\', ...]\n    '
    ccl = np.where(df['close'] > df['close'].shift(1), np.where(df['close_oi'] > df['close_oi'].shift(1), '多头增仓', '空头减仓'), np.where(df['close_oi'] > df['close_oi'].shift(1), '空头增仓', '多头减仓'))
    new_df = pd.DataFrame(data=list(ccl), columns=['ccl'])
    return new_df

def CJL(df):
    if False:
        print('Hello World!')
    '\n    成交量\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含2列, 是"vol", "opid", 分别代表成交量和持仓量\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的成交量\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import CJL\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ndf = CJL(klines)\n        print(list(ndf["vol"]))\n        print(list(ndf["opid"]))\n\n\n        # 预计的输出是这样的:\n        [..., 93142, 95875, 102152, ...]\n        [..., 69213, 66414, 68379, ...]\n    '
    new_df = pd.DataFrame()
    new_df['vol'] = df['volume']
    new_df['opid'] = df['close_oi']
    return new_df

def OPI(df):
    if False:
        while True:
            i = 10
    '\n    持仓量\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"opi", 代表持仓量\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的持仓量\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import OPI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        opi = OPI(klines)\n        print(list(opi["opi"]))\n\n\n        # 预计的输出是这样的:\n        [..., 69213, 66414, 68379, ...]\n    '
    opi = df['close_oi']
    new_df = pd.DataFrame(data=list(opi), columns=['opi'])
    return new_df

def PVT(df):
    if False:
        while True:
            i = 10
    '\n    价量趋势指数\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"pvt", 代表计算出来的价量趋势指数\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的价量趋势指数\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import PVT\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        pvt = PVT(klines)\n        print(list(pvt["pvt"]))\n\n\n        # 预计的输出是这样的:\n        [..., 13834.536889431965, 12892.3866788564, 9255.595248484618, ...]\n    '
    pvt = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    new_df = pd.DataFrame(data=list(pvt), columns=['pvt'])
    return new_df

def VOSC(df, short, long):
    if False:
        while True:
            i = 10
    '\n    移动平均成交量指标 Volume Oscillator\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        short (int): 短周期\n\n        long (int): 长周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"vosc", 代表计算出来的移动平均成交量指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的移动平均成交量指标\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import VOSC\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        vosc = VOSC(klines, 12, 26)\n        print(list(vosc["vosc"]))\n\n\n        # 预计的输出是这样的:\n        [..., 38.72537848731668, 36.61748077024136, 35.4059127302802, ...]\n    '
    vosc = (tqsdk.tafunc.ma(df['volume'], short) - tqsdk.tafunc.ma(df['volume'], long)) / tqsdk.tafunc.ma(df['volume'], short) * 100
    new_df = pd.DataFrame(data=list(vosc), columns=['vosc'])
    return new_df

def VROC(df, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    量变动速率\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"vroc", 代表计算出来的量变动速率\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的量变动速率\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import VROC\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        vroc = VROC(klines, 12)\n        print(list(vroc["vroc"]))\n\n\n        # 预计的输出是这样的:\n        [..., 41.69905854184833, 74.03274443327598, 3.549394666873177, ...]\n    '
    vroc = (df['volume'] - df['volume'].shift(n)) / df['volume'].shift(n) * 100
    new_df = pd.DataFrame(data=list(vroc), columns=['vroc'])
    return new_df

def VRSI(df, n):
    if False:
        i = 10
        return i + 15
    '\n    量相对强弱\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 参数n\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"vrsi", 代表计算出来的量相对强弱指标\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的量相对强弱\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import VRSI\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        vrsi = VRSI(klines, 6)\n        print(list(vrsi["vrsi"]))\n\n\n        # 预计的输出是这样的:\n        [..., 59.46573277427041, 63.3447660581749, 45.21081537920358, ...]\n    '
    vrsi = tqsdk.tafunc.sma(pd.Series(np.where(df['volume'] - df['volume'].shift(1) > 0, df['volume'] - df['volume'].shift(1), 0)), n, 1) / tqsdk.tafunc.sma(np.absolute(df['volume'] - df['volume'].shift(1)), n, 1) * 100
    new_df = pd.DataFrame(data=list(vrsi), columns=['vrsi'])
    return new_df

def WVAD(df):
    if False:
        print('Hello World!')
    '\n    威廉变异离散量\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"wvad", 代表计算出来的威廉变异离散量\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的威廉变异离散量\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import WVAD\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        wvad = WVAD(klines)\n        print(list(wvad["wvad"]))\n\n\n        # 预计的输出是这样的:\n        [..., -32690.203562340674, -42157.968253968385, 32048.182305630264, ...]\n    '
    wvad = (df['close'] - df['open']) / (df['high'] - df['low']) * df['volume']
    new_df = pd.DataFrame(data=list(wvad), columns=['wvad'])
    return new_df

def MA(df, n):
    if False:
        return 10
    '\n    简单移动平均线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 简单移动平均线的周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"ma", 代表计算出来的简单移动平均线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的简单移动平均线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import MA\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ma = MA(klines, 30)\n        print(list(ma["ma"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3436.300000000001, 3452.8733333333344, 3470.5066666666676, ...]\n    '
    new_df = pd.DataFrame(data=list(tqsdk.tafunc.ma(df['close'], n)), columns=['ma'])
    return new_df

def SMA(df, n, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    扩展指数加权移动平均\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 扩展指数加权移动平均的周期\n\n        m (int): 扩展指数加权移动平均的权重\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"sma", 代表计算出来的扩展指数加权移动平均线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的扩展指数加权移动平均线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import SMA\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        sma = SMA(klines, 5, 2)\n        print(list(sma["sma"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3803.9478653510914, 3751.648719210655, 3739.389231526393, ...]\n    '
    new_df = pd.DataFrame(data=list(tqsdk.tafunc.sma(df['close'], n, m)), columns=['sma'])
    return new_df

def EMA(df, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    指数加权移动平均线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 指数加权移动平均线的周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"ema", 代表计算出来的指数加权移动平均线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的指数加权移动平均线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import EMA\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ema = EMA(klines, 10)\n        print(list(ema["ema"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3723.1470497119317, 3714.065767946126, 3715.3265374104667, ...]\n    '
    new_df = pd.DataFrame(data=list(tqsdk.tafunc.ema(df['close'], n)), columns=['ema'])
    return new_df

def EMA2(df, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    线性加权移动平均 WMA\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 线性加权移动平均的周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"ema2", 代表计算出来的线性加权移动平均线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的线性加权移动平均线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import EMA2\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        ema2 = EMA2(klines, 10)\n        print(list(ema2["ema2"]))\n\n\n        # 预计的输出是这样的:\n        [..., 3775.832727272727, 3763.334545454546, 3757.101818181818, ...]\n    '
    new_df = pd.DataFrame(data=list(tqsdk.tafunc.ema2(df['close'], n)), columns=['ema2'])
    return new_df

def TRMA(df, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    三角移动平均线\n\n    Args:\n        df (pandas.DataFrame): Dataframe格式的K线序列\n\n        n (int): 三角移动平均线的周期\n\n    Returns:\n        pandas.DataFrame: 返回的DataFrame包含1列, 是"trma", 代表计算出来的三角移动平均线\n\n    Example::\n\n        # 获取 CFFEX.IF1903 合约的三角移动平均线\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import TRMA\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1903", 24 * 60 * 60)\n        trma = TRMA(klines, 10)\n        print(list(trma["trma"]))\n\n        # 预计的输出是这样的:\n        [..., 341.366666666669, 3759.160000000002, 3767.7533333333354, ...]\n    '
    new_df = pd.DataFrame(data=list(tqsdk.tafunc.trma(df['close'], n)), columns=['trma'])
    return new_df

def BS_VALUE(df, quote, r=0.025, v=None):
    if False:
        return 10
    '\n    期权 BS 模型理论价格\n\n    Args:\n        df (pandas.DataFrame): 需要计算理论价的期权对应标的合约的 K 线序列，Dataframe 格式\n\n        quote (tqsdk.objs.Quote): 需要计算理论价的期权对象，其标的合约应该是 df 序列对应的合约，否则返回序列值全为 nan\n\n        r (float): [可选]无风险利率\n\n        v (None | float | pandas.Series): [可选]波动率\n\n            * None [默认]: 使用 df 中的 close 序列计算的波动率来计算期权理论价格\n\n            * float: 对于 df 中每一行都使用相同的 v 计算期权理论价格\n\n            * pandas.Series: 其行数应该和 df 行数相同，对于 df 中每一行都使用 v 中对应行的值计算期权理论价格\n\n    Returns:\n        pandas.DataFrame: 返回的 DataFrame 包含 1 列, 是 "bs_price", 代表计算出来的期权理论价格, 与参数 df 行数相同\n\n    Example1::\n\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import BS_VALUE\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006C43000")\n        klines = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 30)\n        bs_serise = BS_VALUE(klines, quote, 0.025)\n        print(list(bs_serise["bs_price"]))\n        api.close()\n\n        # 预计的输出是这样的:\n        [..., 3036.698780158862, 2393.333388624822, 2872.607833620801]\n\n\n    Example2::\n\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import BS_VALUE\n        from tqsdk.tafunc import get_his_volatility\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 30)\n        v = get_his_volatility(ks, api.get_quote("SHFE.cu2006"))\n        print("历史波动率:", v)\n\n        quote = api.get_quote("SHFE.cu2006C43000")\n        bs_serise = BS_VALUE(ks, quote, 0.025, v)\n        print(list(bs_serise["bs_price"]))\n        api.close()\n\n        # 预计的输出是这样的:\n        [..., 3036.698780158862, 2393.333388624822, 2872.607833620801]\n    '
    if not (quote.ins_class.endswith('OPTION') and quote.underlying_symbol == df['symbol'][0]):
        return pd.DataFrame(np.full_like(df['close'], float('nan')), columns=['bs_price'])
    if v is None:
        v = tqsdk.tafunc._get_volatility(df['close'], df['duration'], quote.trading_time)
        if math.isnan(v):
            return pd.DataFrame(np.full_like(df['close'], float('nan')), columns=['bs_price'])
    t = tqsdk.tafunc._get_t_series(df['datetime'], df['duration'], quote.expire_datetime)
    return pd.DataFrame(data=list(tqsdk.tafunc.get_bs_price(df['close'], quote.strike_price, r, v, t, quote.option_class)), columns=['bs_price'])

def OPTION_GREEKS(df, quote, r=0.025, v=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    期权希腊指标\n\n    Args:\n        df (pandas.DataFrame): 期权合约及对应标的合约组成的多 K 线序列, Dataframe 格式\n\n            对于参数 df，需要用 api.get_kline_serial() 获取多 K 线序列，第一个参数为 list 类型，顺序为期权合约在前，对应标的合约在后，否则返回序列值全为 nan。\n\n            例如：api.get_kline_serial(["SHFE.cu2006C44000", "SHFE.cu2006"], 60, 100)\n\n\n        quote (tqsdk.objs.Quote): 期权对象，应该是 df 中多 K 线序列中的期权合约对象，否则返回序列值全为 nan。\n\n            例如：api.get_quote("SHFE.cu2006C44000")\n\n        r (float): [可选]无风险利率\n\n        v (None | float | pandas.Series): [可选]波动率\n\n            * None [默认]: 使用 df 序列计算出的隐含波动率计算希腊指标值\n\n            * float: 对于 df 中每一行都使用相同的 v 计算希腊指标值\n\n            * pandas.Series: 其行数应该和 df 行数相同，对于 df 中每一行都使用 v 中对应行的值计算希腊指标值\n\n    Returns:\n        pandas.DataFrame: 返回的 DataFrame 包含 5 列, 分别是 "delta", "theta", "gamma", "vega", "rho", 与参数 df 行数相同\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import OPTION_GREEKS\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006C44000")\n        klines = api.get_kline_serial(["SHFE.cu2006C44000", "SHFE.cu2006"], 24 * 60 * 60, 30)\n        greeks = OPTION_GREEKS(klines, quote, 0.025)\n        print(list(greeks["delta"]))\n        print(list(greeks["theta"]))\n        print(list(greeks["gamma"]))\n        print(list(greeks["vega"]))\n        print(list(greeks["rho"]))\n        api.close()\n\n    '
    new_df = pd.DataFrame()
    if not (quote.ins_class.endswith('OPTION') and quote.instrument_id == df['symbol'][0] and (quote.underlying_symbol == df['symbol1'][0])):
        new_df['delta'] = pd.Series(np.full_like(df['close1'], float('nan')))
        new_df['theta'] = pd.Series(np.full_like(df['close1'], float('nan')))
        new_df['gamma'] = pd.Series(np.full_like(df['close1'], float('nan')))
        new_df['vega'] = pd.Series(np.full_like(df['close1'], float('nan')))
        new_df['rho'] = pd.Series(np.full_like(df['close1'], float('nan')))
    else:
        t = tqsdk.tafunc._get_t_series(df['datetime'], df['duration'], quote.expire_datetime)
        if v is None:
            v = tqsdk.tafunc.get_impv(df['close1'], df['close'], quote.strike_price, r, 0.3, t, quote.option_class)
        d1 = tqsdk.tafunc._get_d1(df['close1'], quote.strike_price, r, v, t)
        new_df['delta'] = tqsdk.tafunc.get_delta(df['close1'], quote.strike_price, r, v, t, quote.option_class, d1)
        new_df['theta'] = tqsdk.tafunc.get_theta(df['close1'], quote.strike_price, r, v, t, quote.option_class, d1)
        new_df['gamma'] = tqsdk.tafunc.get_gamma(df['close1'], quote.strike_price, r, v, t, d1)
        new_df['vega'] = tqsdk.tafunc.get_vega(df['close1'], quote.strike_price, r, v, t, d1)
        new_df['rho'] = tqsdk.tafunc.get_rho(df['close1'], quote.strike_price, r, v, t, quote.option_class, d1)
    return new_df

def OPTION_VALUE(df, quote):
    if False:
        return 10
    '\n    期权内在价值，时间价值\n\n    Args:\n        df (pandas.DataFrame): 期权合约及对应标的合约组成的多 K 线序列, Dataframe 格式\n\n            对于参数 df，需要用 api.get_kline_serial() 获取多 K 线序列，第一个参数为 list 类型，顺序为期权合约在前，对应标的合约在后，否则返回序列值全为 nan。\n\n            例如：api.get_kline_serial(["SHFE.cu2006C44000", "SHFE.cu2006"], 60, 100)\n\n\n        quote (tqsdk.objs.Quote): 期权对象，应该是 df 中多 K 线序列中的期权合约对象，否则返回序列值全为 nan。\n\n            例如：api.get_quote("SHFE.cu2006C44000")\n\n    Returns:\n        pandas.DataFrame: 返回的 DataFrame 包含 2 列, 是 "intrins" 和 "time", 代表内在价值和时间价值, 与参数 df 行数相同\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import OPTION_VALUE\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006C43000")\n        klines = api.get_kline_serial(["SHFE.cu2006C43000", "SHFE.cu2006"], 24 * 60 * 60, 30)\n        values = OPTION_VALUE(klines, quote)\n        print(list(values["intrins"]))\n        print(list(values["time"]))\n        api.close()\n    '
    new_df = pd.DataFrame()
    if not (quote.ins_class.endswith('OPTION') and quote.instrument_id == df['symbol'][0] and (quote.underlying_symbol == df['symbol1'][0])):
        new_df['intrins'] = pd.Series(np.full_like(df['close1'], float('nan')))
        new_df['time'] = pd.Series(np.full_like(df['close1'], float('nan')))
    else:
        o = 1 if quote.option_class == 'CALL' else -1
        intrins = o * (df['close1'] - quote.strike_price)
        new_df['intrins'] = pd.Series(np.where(intrins > 0.0, intrins, 0.0))
        new_df['time'] = pd.Series(df['close'] - new_df['intrins'])
    return new_df

def OPTION_IMPV(df, quote, r=0.025):
    if False:
        return 10
    '\n    计算期权隐含波动率\n\n    Args:\n        df (pandas.DataFrame): 期权合约及对应标的合约组成的多 K 线序列, Dataframe 格式\n\n            对于参数 df，需要用 api.get_kline_serial() 获取多 K 线序列，第一个参数为 list 类型，顺序为期权合约在前，对应标的合约在后，否则返回序列值全为 nan。\n\n            例如：api.get_kline_serial(["SHFE.cu2006C44000", "SHFE.cu2006"], 60, 100)\n\n\n        quote (tqsdk.objs.Quote): 期权对象，应该是 df 中多 K 线序列中的期权合约对象，否则返回序列值全为 nan。\n\n            例如：api.get_quote("SHFE.cu2006C44000")\n\n        r (float): [可选]无风险利率\n\n    Returns:\n        pandas.DataFrame: 返回的 DataFrame 包含 1 列, 是 "impv", 与参数 df 行数相同\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import OPTION_IMPV\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006C50000")\n        klines = api.get_kline_serial(["SHFE.cu2006C50000", "SHFE.cu2006"], 24 * 60 * 60, 20)\n        impv = OPTION_IMPV(klines, quote, 0.025)\n        print(list(impv["impv"] * 100))\n        api.close()\n    '
    if not (quote.ins_class.endswith('OPTION') and quote.instrument_id == df['symbol'][0] and (quote.underlying_symbol == df['symbol1'][0])):
        return pd.DataFrame(np.full_like(df['close1'], float('nan')), columns=['impv'])
    his_v = tqsdk.tafunc._get_volatility(df['close1'], df['duration'], quote.trading_time)
    his_v = 0.3 if math.isnan(his_v) else his_v
    t = tqsdk.tafunc._get_t_series(df['datetime'], df['duration'], quote.expire_datetime)
    return pd.DataFrame(data=list(tqsdk.tafunc.get_impv(df['close1'], df['close'], quote.strike_price, r, his_v, t, quote.option_class)), columns=['impv'])

def VOLATILITY_CURVE(df: pd.DataFrame, quotes: dict, underlying: str, r=0.025):
    if False:
        i = 10
        return i + 15
    '\n    计算期权隐含波动率曲面\n\n    Args:\n        df (pandas.DataFrame): 期权合约及基础标合约组成的多 K 线序列, Dataframe 格式\n\n        quote (dict): 批量获取合约的行情信息, 存储结构必须为 dict, key 为合约, value 为行情数据\n\n                例如: {\'SHFE.cu2101\':{ ... }, ‘SHFE.cu2101C34000’:{ ... }}\n\n        underlying (str): 基础标的的合约名称, 如 SHFE.cu2101\n\n        r (float): [可选]无风险利率\n\n    Returns:\n        pandas.DataFrame: 返回的 DataFrame\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.ta import VOLATILITY_CURVE\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        underlying = "DCE.m2101"\n        options = api.query_options(underlying_symbol=underlying, option_class="PUT", expired=False)\n        # 批量获取合约的行情信息, 存储结构必须为 dict, key 为合约, value 为行情数据\n        quote = {}\n        for symbol in options:\n            quote[symbol] = api.get_quote(symbol)\n        options.append(underlying)\n\n        klines = api.get_kline_serial(options, 24 * 60 * 60, 20)\n        vc = VOLATILITY_CURVE(klines, quote, underlying, r = 0.025)\n        print(vc)\n        api.close()\n\n        # 预计的输出是这样的:\n                datetime    2450.0    2500.0  ...  3600.0    3650.0\n        0   1.603382e+18  0.336557  0.314832  ...  0.231657  0.237882\n        1   1.603642e+18  0.353507  0.331051  ...  0.231657  0.237882\n\n    '
    symbol_titles = [s for s in df.columns.values if s.startswith('symbol')]
    base_symbol_title = [s for s in symbol_titles if df[s].iloc[0] == underlying]
    if not base_symbol_title:
        raise Exception(f'kline 数据中未包含基础标的合约的K线数据, 请更正')
    base_close_title = f'close{base_symbol_title[0][6:]}'
    res_dict = {}
    pd_columns = []
    for symbol_title in symbol_titles:
        if symbol_title == base_symbol_title[0]:
            continue
        close_title = f'close{symbol_title[6:]}'
        quote = quotes[df[symbol_title].iloc[0]]
        t = tqsdk.tafunc._get_t_series(df['datetime'], df['duration'], quote.expire_datetime)
        res_dict[quote.strike_price] = tqsdk.tafunc.get_impv(df[base_close_title], df[close_title], quote.strike_price, r, 0.5, t, quote.option_class).interpolate(method='linear')
        res_dict['datetime'] = df['datetime']
        pd_columns.append(quote.strike_price)
    pd_columns.sort()
    pd_columns.insert(0, 'datetime')
    return pd.DataFrame(data=res_dict, columns=pd_columns)