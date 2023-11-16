"""
股票技术指标接口
Created on 2018/05/26
@author: Jackie Liao
@group : **
@contact: info@liaocy.net
"""

def ma(data, n=10, val_name='close'):
    if False:
        return 10
    import numpy as np
    '\n    移动平均线 Moving Average\n    Parameters\n    ------\n      data:pandas.DataFrame\n                  通过 get_h_data 取得的股票数据\n      n:int\n                  移动平均线时长，时间单位根据data决定\n      val_name:string\n                  计算哪一列的列名，默认为 close 收盘值\n\n    return\n    -------\n      list\n          移动平均线\n    '
    values = []
    MA = []
    for (index, row) in data.iterrows():
        values.append(row[val_name])
        if len(values) == n:
            del values[0]
        MA.append(np.average(values))
    return np.asarray(MA)

def md(data, n=10, val_name='close'):
    if False:
        for i in range(10):
            print('nop')
    import numpy as np
    '\n    移动标准差\n    Parameters\n    ------\n      data:pandas.DataFrame\n                  通过 get_h_data 取得的股票数据\n      n:int\n                  移动平均线时长，时间单位根据data决定\n      val_name:string\n                  计算哪一列的列名，默认为 close 收盘值\n\n    return\n    -------\n      list\n          移动平均线\n    '
    values = []
    MD = []
    for (index, row) in data.iterrows():
        values.append(row[val_name])
        if len(values) == n:
            del values[0]
        MD.append(np.std(values))
    return np.asarray(MD)

def _get_day_ema(prices, n):
    if False:
        for i in range(10):
            print('nop')
    a = 1 - 2 / (n + 1)
    day_ema = 0
    for (index, price) in enumerate(reversed(prices)):
        day_ema += a ** index * price
    return day_ema

def ema(data, n=12, val_name='close'):
    if False:
        print('Hello World!')
    import numpy as np
    '\n        指数平均数指标 Exponential Moving Average\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n                      移动平均线时长，时间单位根据data决定\n          val_name:string\n                      计算哪一列的列名，默认为 close 收盘值\n\n        return\n        -------\n          EMA:numpy.ndarray<numpy.float64>\n              指数平均数指标\n    '
    prices = []
    EMA = []
    for (index, row) in data.iterrows():
        if index == 0:
            past_ema = row[val_name]
            EMA.append(row[val_name])
        else:
            today_ema = (2 * row[val_name] + (n - 1) * past_ema) / (n + 1)
            past_ema = today_ema
            EMA.append(today_ema)
    return np.asarray(EMA)

def macd(data, quick_n=12, slow_n=26, dem_n=9, val_name='close'):
    if False:
        while True:
            i = 10
    import numpy as np
    '\n        指数平滑异同平均线(MACD: Moving Average Convergence Divergence)\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          quick_n:int\n                      DIFF差离值中快速移动天数\n          slow_n:int\n                      DIFF差离值中慢速移动天数\n          dem_n:int\n                      DEM讯号线的移动天数\n          val_name:string\n                      计算哪一列的列名，默认为 close 收盘值\n\n        return\n        -------\n          OSC:numpy.ndarray<numpy.float64>\n              MACD bar / OSC 差值柱形图 DIFF - DEM\n          DIFF:numpy.ndarray<numpy.float64>\n              差离值\n          DEM:numpy.ndarray<numpy.float64>\n              讯号线\n    '
    ema_quick = np.asarray(ema(data, quick_n, val_name))
    ema_slow = np.asarray(ema(data, slow_n, val_name))
    DIFF = ema_quick - ema_slow
    data['diff'] = DIFF
    DEM = ema(data, dem_n, 'diff')
    OSC = DIFF - DEM
    return (OSC, DIFF, DEM)

def kdj(data):
    if False:
        return 10
    import numpy as np
    '\n        随机指标KDJ\n        Parameters\n        ------\n          data:pandas.DataFrame\n                通过 get_h_data 取得的股票数据\n        return\n        -------\n          K:numpy.ndarray<numpy.float64>\n              K线\n          D:numpy.ndarray<numpy.float64>\n              D线\n          J:numpy.ndarray<numpy.float64>\n              J线\n    '
    (K, D, J) = ([], [], [])
    (last_k, last_d) = (None, None)
    for (index, row) in data.iterrows():
        if last_k is None or last_d is None:
            last_k = 50
            last_d = 50
        (c, l, h) = (row['close'], row['low'], row['high'])
        rsv = (c - l) / (h - l) * 100
        k = 2 / 3 * last_k + 1 / 3 * rsv
        d = 2 / 3 * last_d + 1 / 3 * k
        j = 3 * k - 2 * d
        K.append(k)
        D.append(d)
        J.append(j)
        (last_k, last_d) = (k, d)
    return (np.asarray(K), np.asarray(D), np.asarray(J))

def rsi(data, n=6, val_name='close'):
    if False:
        i = 10
        return i + 15
    import numpy as np
    '\n        相对强弱指标RSI\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n                统计时长，时间单位根据data决定\n        return\n        -------\n          RSI:numpy.ndarray<numpy.float64>\n              RSI线\n        \n    '
    RSI = []
    UP = []
    DOWN = []
    for (index, row) in data.iterrows():
        if index == 0:
            past_value = row[val_name]
            RSI.append(0)
        else:
            diff = row[val_name] - past_value
            if diff > 0:
                UP.append(diff)
                DOWN.append(0)
            else:
                UP.append(0)
                DOWN.append(diff)
            if len(UP) == n:
                del UP[0]
            if len(DOWN) == n:
                del DOWN[0]
            past_value = row[val_name]
            rsi = np.sum(UP) / (-np.sum(DOWN) + np.sum(UP)) * 100
            RSI.append(rsi)
    return np.asarray(RSI)

def boll(data, n=10, val_name='close', k=2):
    if False:
        return 10
    '\n        布林线指标BOLL\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n                统计时长，时间单位根据data决定\n        return\n        -------\n          BOLL:numpy.ndarray<numpy.float64>\n              中轨线\n          UPPER:numpy.ndarray<numpy.float64>\n              D线\n          J:numpy.ndarray<numpy.float64>\n              J线\n    '
    BOLL = ma(data, n, val_name)
    MD = md(data, n, val_name)
    UPPER = BOLL + k * MD
    LOWER = BOLL - k * MD
    return (BOLL, UPPER, LOWER)

def wnr(data, n=14):
    if False:
        while True:
            i = 10
    '\n        威廉指标 w&r\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n                统计时长，时间单位根据data决定\n        return\n        -------\n          WNR:numpy.ndarray<numpy.float64>\n              威廉指标\n    '
    high_prices = []
    low_prices = []
    WNR = []
    for (index, row) in data.iterrows():
        high_prices.append(row['high'])
        if len(high_prices) == n:
            del high_prices[0]
        low_prices.append(row['low'])
        if len(low_prices) == n:
            del low_prices[0]
        highest = max(high_prices)
        lowest = min(low_prices)
        wnr = (highest - row['close']) / (highest - lowest) * 100
        WNR.append(wnr)
    return WNR

def _get_any_ma(arr, n):
    if False:
        return 10
    import numpy as np
    MA = []
    values = []
    for val in arr:
        values.append(val)
        if len(values) == n:
            del values[0]
        MA.append(np.average(values))
    return np.asarray(MA)

def dmi(data, n=14, m=14, k=6):
    if False:
        i = 10
        return i + 15
    import numpy as np
    '\n        动向指标或趋向指标 DMI\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              +-DI(n): DI统计时长，默认14\n          m:int\n              ADX(m): ADX统计时常参数，默认14\n              \n          k:int\n              ADXR(k): ADXR统计k个周期前数据，默认6\n        return\n        -------\n          P_DI:numpy.ndarray<numpy.float64>\n              +DI指标\n          M_DI:numpy.ndarray<numpy.float64>\n              -DI指标\n          ADX:numpy.ndarray<numpy.float64>\n              ADX指标\n          ADXR:numpy.ndarray<numpy.float64>\n              ADXR指标\n        ref.\n        -------\n        https://www.mk-mode.com/octopress/2012/03/03/03002038/\n    '
    P_DM = [0.0]
    M_DM = [0.0]
    TR = [0.0]
    DX = [0.0]
    P_DI = [0.0]
    M_DI = [0.0]
    for (index, row) in data.iterrows():
        if index == 0:
            past_row = row
        else:
            p_dm = row['high'] - past_row['high']
            m_dm = past_row['low'] - row['low']
            if p_dm < 0 and m_dm < 0 or np.isclose(p_dm, m_dm):
                p_dm = 0
                m_dm = 0
            if p_dm > m_dm:
                m_dm = 0
            if m_dm > p_dm:
                p_dm = 0
            P_DM.append(p_dm)
            M_DM.append(m_dm)
            tr = max(row['high'] - past_row['low'], row['high'] - past_row['close'], past_row['close'] - row['low'])
            TR.append(tr)
            if len(P_DM) == n:
                del P_DM[0]
            if len(M_DM) == n:
                del M_DM[0]
            if len(TR) == n:
                del TR[0]
            p_di = np.average(P_DM) / np.average(TR) * 100
            P_DI.append(p_di)
            m_di = np.average(M_DM) / np.average(TR) * 100
            M_DI.append(m_di)
            if p_di + m_di == 0:
                dx = 0
            else:
                dx = abs(p_di - m_di) / (p_di + m_di) * 100
            DX.append(dx)
            past_row = row
    ADX = _get_any_ma(DX, m)
    ADXR = []
    for (index, adx) in enumerate(ADX):
        if index >= k:
            adxr = (adx + ADX[index - k]) / 2
            ADXR.append(adxr)
        else:
            ADXR.append(0)
    return (P_DI, M_DI, ADX, ADXR)

def bias(data, n=5):
    if False:
        print('Hello World!')
    import numpy as np
    '\n        乖离率 bias\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              统计时长，默认5\n        return\n        -------\n          BIAS:numpy.ndarray<numpy.float64>\n              乖离率指标\n\n    '
    MA = ma(data, n)
    CLOSES = data['close']
    BIAS = np.true_divide(CLOSES - MA, MA) * (100 / 100)
    return BIAS

def asi(data, n=5):
    if False:
        return 10
    import numpy as np
    '\n        振动升降指标 ASI\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              统计时长，默认5\n        return\n        -------\n          ASI:numpy.ndarray<numpy.float64>\n              振动升降指标\n\n    '
    SI = []
    for (index, row) in data.iterrows():
        if index == 0:
            last_row = row
            SI.append(0.0)
        else:
            a = abs(row['close'] - last_row['close'])
            b = abs(row['low'] - last_row['close'])
            c = abs(row['high'] - last_row['close'])
            d = abs(last_row['close'] - last_row['open'])
            if b > a and b > c:
                r = b + 1 / 2 * a + 1 / 4 * d
            elif c > a and c > b:
                r = c + 1 / 4 * d
            else:
                r = 0
            e = row['close'] - last_row['close']
            f = row['close'] - last_row['open']
            g = last_row['close'] - last_row['open']
            x = e + 1 / 2 * f + g
            k = max(a, b)
            l = 3
            if np.isclose(r, 0) or np.isclose(l, 0):
                si = 0
            else:
                si = 50 * (x / r) * (k / l)
            SI.append(si)
    ASI = _get_any_ma(SI, n)
    return ASI

def vr(data, n=26):
    if False:
        for i in range(10):
            print('nop')
    import numpy as np
    '\n        Volatility Volume Ratio 成交量变异率\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              统计时长，默认26\n        return\n        -------\n          VR:numpy.ndarray<numpy.float64>\n              成交量变异率\n\n    '
    VR = []
    (AV_volumes, BV_volumes, CV_volumes) = ([], [], [])
    for (index, row) in data.iterrows():
        if row['close'] > row['open']:
            AV_volumes.append(row['volume'])
        elif row['close'] < row['open']:
            BV_volumes.append(row['volume'])
        else:
            CV_volumes.append(row['volume'])
        if len(AV_volumes) == n:
            del AV_volumes[0]
        if len(BV_volumes) == n:
            del BV_volumes[0]
        if len(CV_volumes) == n:
            del CV_volumes[0]
        avs = sum(AV_volumes)
        bvs = sum(BV_volumes)
        cvs = sum(CV_volumes)
        if bvs + 1 / 2 * cvs != 0:
            vr = (avs + 1 / 2 * cvs) / (bvs + 1 / 2 * cvs)
        else:
            vr = 0
        VR.append(vr)
    return np.asarray(VR)

def arbr(data, n=26):
    if False:
        print('Hello World!')
    import numpy as np
    '\n        AR 指标 BR指标\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              统计时长，默认26\n        return\n        -------\n          AR:numpy.ndarray<numpy.float64>\n              AR指标\n          BR:numpy.ndarray<numpy.float64>\n              BR指标\n\n    '
    (H, L, O, PC) = (np.array([0]), np.array([0]), np.array([0]), np.array([0]))
    (AR, BR) = (np.array([0]), np.array([0]))
    for (index, row) in data.iterrows():
        if index == 0:
            last_row = row
        else:
            h = row['high']
            H = np.append(H, [h])
            if len(H) == n:
                H = np.delete(H, 0)
            l = row['low']
            L = np.append(L, [l])
            if len(L) == n:
                L = np.delete(L, 0)
            o = row['open']
            O = np.append(O, [o])
            if len(O) == n:
                O = np.delete(O, 0)
            pc = last_row['close']
            PC = np.append(PC, [pc])
            if len(PC) == n:
                PC = np.delete(PC, 0)
            ar = np.sum(np.asarray(H) - np.asarray(O)) / sum(np.asarray(O) - np.asarray(L)) * 100
            AR = np.append(AR, [ar])
            br = np.sum(np.asarray(H) - np.asarray(PC)) / sum(np.asarray(PC) - np.asarray(L)) * 100
            BR = np.append(BR, [br])
            last_row = row
    return (np.asarray(AR), np.asarray(BR))

def dpo(data, n=20, m=6):
    if False:
        while True:
            i = 10
    '\n        区间震荡线指标 DPO\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              统计时长，默认20\n          m:int\n              MADPO的参数M，默认6\n        return\n        -------\n          DPO:numpy.ndarray<numpy.float64>\n              DPO指标\n          MADPO:numpy.ndarray<numpy.float64>\n              MADPO指标\n\n    '
    CLOSES = data['close']
    DPO = CLOSES - ma(data, int(n / 2 + 1))
    MADPO = _get_any_ma(DPO, m)
    return (DPO, MADPO)

def trix(data, n=12, m=20):
    if False:
        print('Hello World!')
    import numpy as np
    '\n        三重指数平滑平均线 TRIX\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              统计时长，默认12\n          m:int\n              TRMA的参数M，默认20\n        return\n        -------\n          TRIX:numpy.ndarray<numpy.float64>\n              AR指标\n          TRMA:numpy.ndarray<numpy.float64>\n              BR指标\n\n    '
    CLOSES = []
    TRIX = []
    for (index, row) in data.iterrows():
        CLOSES.append(row['close'])
        if len(CLOSES) == n:
            del CLOSES[0]
        tr = np.average(CLOSES)
        if index == 0:
            past_tr = tr
            TRIX.append(0)
        else:
            trix = (tr - past_tr) / past_tr * 100
            TRIX.append(trix)
    TRMA = _get_any_ma(TRIX, m)
    return (TRIX, TRMA)

def bbi(data):
    if False:
        for i in range(10):
            print('nop')
    import numpy as np
    '\n        Bull And Bearlndex 多空指标\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n        return\n        -------\n          BBI:numpy.ndarray<numpy.float64>\n              BBI指标\n\n    '
    CS = []
    BBI = []
    for (index, row) in data.iterrows():
        CS.append(row['close'])
        if len(CS) < 24:
            BBI.append(row['close'])
        else:
            bbi = np.average([np.average(CS[-3:]), np.average(CS[-6:]), np.average(CS[-12:]), np.average(CS[-24:])])
            BBI.append(bbi)
    return np.asarray(BBI)

def mtm(data, n=6):
    if False:
        return 10
    import numpy as np
    '\n        Momentum Index 动量指标\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n          n:int\n              统计时长，默认6\n        return\n        -------\n          MTM:numpy.ndarray<numpy.float64>\n              MTM动量指标\n\n    '
    MTM = []
    CN = []
    for (index, row) in data.iterrows():
        if index < n - 1:
            MTM.append(0.0)
        else:
            mtm = row['close'] - CN[index - n]
            MTM.append(mtm)
        CN.append(row['close'])
    return np.asarray(MTM)

def obv(data):
    if False:
        print('Hello World!')
    import numpy as np
    '\n        On Balance Volume 能量潮指标\n        Parameters\n        ------\n          data:pandas.DataFrame\n                      通过 get_h_data 取得的股票数据\n        return\n        -------\n          OBV:numpy.ndarray<numpy.float64>\n              OBV能量潮指标\n\n    '
    tmp = np.true_divide(data['close'] - data['low'] - (data['high'] - data['close']), data['high'] - data['low'])
    OBV = tmp * data['volume']
    return OBV

def sar(data, n=4):
    if False:
        for i in range(10):
            print('nop')
    raise Exception('Not implemented yet')

def plot_all(data, is_show=True, output=None):
    if False:
        while True:
            i = 10
    import matplotlib.pyplot as plt
    from pylab import rcParams
    import numpy as np
    rcParams['figure.figsize'] = (18, 50)
    plt.figure()
    plt.subplot(20, 1, 1)
    plt.plot(data['date'], data['close'], label='close')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 2)
    MA = ma(data, n=10)
    plt.plot(data['date'], MA, label='MA(n=10)')
    plt.plot(data['date'], data['close'], label='CLOSE PRICE')
    plt.title('MA')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    n = 10
    plt.subplot(20, 1, 3)
    MD = md(data, n)
    plt.plot(data['date'], MD, label='MD(n=10)')
    plt.title('MD')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 4)
    EMA = ema(data, n)
    plt.plot(data['date'], EMA, label='EMA(n=12)')
    plt.title('EMA')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 5)
    (OSC, DIFF, DEM) = macd(data, n)
    plt.plot(data['date'], OSC, label='OSC')
    plt.plot(data['date'], DIFF, label='DIFF')
    plt.plot(data['date'], DEM, label='DEM')
    plt.title('MACD')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 6)
    (K, D, J) = kdj(data)
    plt.plot(data['date'], K, label='K')
    plt.plot(data['date'], D, label='D')
    plt.plot(data['date'], J, label='J')
    plt.title('KDJ')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 7)
    RSI6 = rsi(data, 6)
    RSI12 = rsi(data, 12)
    RSI24 = rsi(data, 24)
    plt.plot(data['date'], RSI6, label='RSI(n=6)')
    plt.plot(data['date'], RSI12, label='RSI(n=12)')
    plt.plot(data['date'], RSI24, label='RSI(n=24)')
    plt.title('RSI')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 8)
    (BOLL, UPPER, LOWER) = boll(data)
    plt.plot(data['date'], BOLL, label='BOLL(n=10)')
    plt.plot(data['date'], UPPER, label='UPPER(n=10)')
    plt.plot(data['date'], LOWER, label='LOWER(n=10)')
    plt.plot(data['date'], data['close'], label='CLOSE PRICE')
    plt.title('BOLL')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 9)
    WNR = wnr(data, n=14)
    plt.plot(data['date'], WNR, label='WNR(n=14)')
    plt.title('WNR')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 10)
    (P_DI, M_DI, ADX, ADXR) = dmi(data)
    plt.plot(data['date'], P_DI, label='+DI(n=14)')
    plt.plot(data['date'], M_DI, label='-DI(n=14)')
    plt.plot(data['date'], ADX, label='ADX(m=14)')
    plt.plot(data['date'], ADXR, label='ADXR(k=6)')
    plt.title('DMI')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 11)
    BIAS = bias(data, n=5)
    plt.plot(data['date'], BIAS, label='BIAS(n=5)')
    plt.title('BIAS')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 12)
    ASI = asi(data, n=5)
    plt.plot(data['date'], ASI, label='ASI(n=5)')
    plt.title('ASI')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 13)
    VR = vr(data, n=26)
    plt.plot(data['date'], VR, label='VR(n=26)')
    plt.title('VR')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 14)
    (AR, BR) = arbr(data, n=26)
    plt.plot(data['date'], AR, label='AR(n=26)')
    plt.plot(data['date'], BR, label='BR(n=26)')
    plt.title('ARBR')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 15)
    (DPO, MADPO) = dpo(data, n=20, m=6)
    plt.plot(data['date'], DPO, label='DPO(n=20)')
    plt.plot(data['date'], MADPO, label='MADPO(m=6)')
    plt.title('DPO')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 16)
    (TRIX, TRMA) = trix(data, n=12, m=20)
    plt.plot(data['date'], TRIX, label='DPO(n=12)')
    plt.plot(data['date'], TRMA, label='MADPO(m=20)')
    plt.title('TRIX')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 17)
    BBI = bbi(data)
    plt.plot(data['date'], BBI, label='BBI(3,6,12,24)')
    plt.title('BBI')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 18)
    MTM = mtm(data, n=6)
    plt.plot(data['date'], MTM, label='MTM(n=6)')
    plt.title('MTM')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.subplot(20, 1, 19)
    OBV = obv(data)
    plt.plot(data['date'], OBV, label='OBV')
    plt.title('OBV')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    if is_show:
        plt.show()
    if output is not None:
        plt.savefig(output)