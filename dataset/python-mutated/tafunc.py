__author__ = 'limin'
from pandas import Series
'\ntqsdk.tafunc 模块包含了一批用于技术指标计算的函数\n(函数基本保持 参数为pandas.Series类型则返回值为pandas.Series类型)\n'
import datetime
import math
from typing import Union
import numpy as np
import pandas as pd
from scipy import stats
from tqsdk.datetime import _get_period_timestamp, _str_to_timestamp_nano, _datetime_to_timestamp_nano

def ref(series, n):
    if False:
        print('Hello World!')
    '\n    简单移动: 求series序列位移n个周期的结果\n\n        注意: 当n为0, 函数返回原序列; 当n为有效值但当前的series序列元素个数不足 n + 1 个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 位移周期\n\n    Returns:\n        pandas.Series: 位移后的序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        pre_close = tafunc.ref(klines.close, 1)  # 将收盘价序列右移一位, 得到昨收盘序列\n        change = klines.close - pre_close        # 收盘价序列 - 昨收盘序列, 得到涨跌序列\n        print(list(change))\n    '
    m = series.shift(n)
    return m

def std(series, n):
    if False:
        i = 10
        return i + 15
    '\n    标准差: 求series序列每n个周期的标准差\n\n        注意: n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 标准差的周期\n\n    Returns:\n        pandas.Series: 标准差序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        std = tafunc.std(klines.close, 5)  # 收盘价序列每5项计算一个标准差\n        print(list(std))\n    '
    m = series.rolling(n).std()
    return m

def ma(series, n):
    if False:
        return 10
    '\n    简单移动平均线: 求series序列n周期的简单移动平均\n\n        计算公式:\n        ma(x, 5) = (x(1) + x(2) + x(3) + x(4) + x(5)) / 5\n\n        注意:\n        1. 简单移动平均线将设定周期内的值取平均值, 其中各元素的权重都相等\n        2. n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 简单移动平均值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        ma = tafunc.ma(klines.close, 5)\n        print(list(ma))\n    '
    ma_data = series.rolling(n).mean()
    return ma_data

def sma(series, n, m):
    if False:
        return 10
    '\n    扩展指数加权移动平均: 求series序列n周期的扩展指数加权移动平均\n    \n        计算公式:\n        sma(x, n, m) = sma(x, n, m).shift(1) * (n - m) / n + x(n) * m / n\n        \n        注意: n必须大于m\n\n    Args:\n        series (pandas.Series): 数据序列\n        \n        n (int): 周期\n        \n        m (int): 权重\n\n    Returns:\n        pandas.Series: 扩展指数加权移动平均序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        sma = tafunc.sma(klines.close, 5, 2)  # 收盘价序列每5项计算一个扩展指数加权移动平均值\n        print(list(sma))\n    '
    sma_data = series.ewm(alpha=m / n, adjust=False).mean()
    return sma_data

def ema(series, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    指数加权移动平均线: 求series序列n周期的指数加权移动平均\n\n        计算公式:\n            ema(x, n) = 2 * x / (n + 1) + (n - 1) * ema(x, n).shift(1) / (n + 1)\n\n        注意:\n            1. n 需大于等于1\n            2. 对距离当前较近的k线赋予了较大的权重\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 指数加权移动平均线序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        ema = tafunc.ema(klines.close, 5)\n        print(list(ema))\n    '
    ema_data = series.ewm(span=n, adjust=False).mean()
    return ema_data

def ema2(series, n):
    if False:
        i = 10
        return i + 15
    '\n    线性加权移动平均: 求series值的n周期线性加权移动平均 (也称WMA)\n\n        计算公式:\n            ema2(x, n) = [n * x(0) + (n - 1) * x(1) + (x - 2) * x(2) + ... + 1 * x(n - 1)] / [n + (n - 1) + (n - 2) + ... + 1]\n\n        注意: 当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 线性加权移动平均线序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        ema2 = tafunc.ema2(klines.close, 5)  # 求收盘价在5个周期的线性加权移动平均值\n        print(list(ema2))\n    '
    weights = list((i for i in range(1, n + 1)))

    def average(elements):
        if False:
            for i in range(10):
                print('nop')
        return np.average(elements, weights=weights)
    ema2 = series.rolling(window=n).apply(average, raw=True)
    return ema2

def crossup(a, b):
    if False:
        while True:
            i = 10
    '\n    向上穿越: 表当a从下方向上穿过b, 成立返回1, 否则返回0\n\n    Args:\n        a (pandas.Series): 数据序列1\n\n        b (pandas.Series): 数据序列2\n\n    Returns:\n        pandas.Series: 上穿标志序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        crossup = tafunc.crossup(tafunc.ma(klines.close, 5), tafunc.ma(klines.close, 10))\n        print(list(crossup))\n    '
    crossup_data = pd.Series(np.where((a > b) & (a.shift(1) <= b.shift(1)), 1, 0))
    return crossup_data

def crossdown(a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    向下穿越: 表示当a从上方向下穿b，成立返回1, 否则返回0\n\n    Args:\n        a (pandas.Series): 数据序列1\n\n        b (pandas.Series): 数据序列2\n\n    Returns:\n        pandas.Series: 下穿标志序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n        \n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        crossdown = tafunc.crossdown(tafunc.ma(klines.close, 5), tafunc.ma(klines.close, 10))\n        print(list(crossdown))\n    '
    crossdown_data = pd.Series(np.where((a < b) & (a.shift(1) >= b.shift(1)), 1, 0))
    return crossdown_data

def count(cond, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    统计n周期中满足cond条件的个数\n\n        注意: 如果n为0, 则从第一个有效值开始统计\n\n    Args:\n        cond (array_like): 条件\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 统计值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        # 统计从申请到的行情数据以来到当前这段时间内, 5周期均线上穿10周期均线的次数:\n        count = tafunc.count(tafunc.crossup(tafunc.ma(klines.close, 5), tafunc.ma(klines.close, 10)), 0)\n        print(list(count))\n    '
    if n == 0:
        count_data = pd.Series(np.where(cond, 1, 0).cumsum())
    else:
        count_data = pd.Series(pd.Series(np.where(cond, 1, 0)).rolling(n).sum())
    return count_data

def trma(series, n):
    if False:
        return 10
    '\n    三角移动平均: 求series的n周期三角移动平均值\n\n        计算方法:\n            三角移动平均线公式, 是采用算数移动平均, 并且对第一个移动平均线再一次应用算数移动平均\n\n        注意: n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 三角移动平均值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        trma = tafunc.trma(klines.close, 10)\n        print(list(trma))\n    '
    if n % 2 == 0:
        n1 = int(n / 2)
        n2 = int(n / 2 + 1)
    else:
        n1 = n2 = int((n + 1) / 2)
    ma_half = ma(series, n1)
    trma_data = ma(ma_half, n2)
    return trma_data

def harmean(series, n):
    if False:
        i = 10
        return i + 15
    '\n    调和平均值: 求series在n个周期内的调和平均值\n\n        计算方法:\n            harmean(x, 5) = 1 / [(1 / x(1) + 1 / x(2) + 1 / x(3) + 1 / x(4) + 1 / x(5)) / 5]\n\n        注意:\n        1. 调和平均值与倒数的简单平均值互为倒数\n        2. 当n为0, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 调和平均值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        harmean = tafunc.harmean(klines.close, 5)  # 求5周期收盘价的调和平均值\n        print(list(harmean))\n    '
    harmean_data = n / (1 / series).rolling(n).sum()
    return harmean_data

def numpow(series, n, m):
    if False:
        while True:
            i = 10
    '\n    自然数幂方和\n\n        计算方法:\n            numpow(x, n, m) = n ^ m * x + (n - 1) ^ m * x.shift(1) + (n - 2) ^ m * x.shift(2) + ... + 2 ^ m * x.shift(n - 2) + 1 ^ m * x.shift(n - 1)\n\n        注意: 当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 自然数\n\n        m (int): 实数\n\n    Returns:\n        pandas.Series: 幂方和序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        numpow = tafunc.numpow(klines.close, 5, 2)\n        print(list(numpow))\n    '
    numpow_data = sum(((n - i) ** m * series.shift(i) for i in range(n)))
    return numpow_data

def abs(series):
    if False:
        for i in range(10):
            print('nop')
    '\n    获取series的绝对值\n\n        注意: 正数的绝对值是它本身, 负数的绝对值是它的相反数, 0的绝对值还是0\n\n    Args:\n        series (pandas.Series): 数据序列\n\n    Returns:\n        pandas.Series: 绝对值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        abs = tafunc.abs(klines.close)\n        print(list(abs))\n    '
    abs_data = pd.Series(np.absolute(series))
    return abs_data

def min(series1, series2):
    if False:
        i = 10
        return i + 15
    '\n    获取series1和series2中的最小值\n\n    Args:\n        series1 (pandas.Series): 数据序列1\n\n        series2 (pandas.Series): 数据序列2\n\n    Returns:\n        pandas.Series: 最小值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        min = tafunc.min(klines.close, klines.open)\n        print(list(min))\n    '
    min_data = np.minimum(series1, series2)
    return min_data

def max(series1, series2):
    if False:
        i = 10
        return i + 15
    '\n    获取series1和series2中的最大值\n\n    Args:\n        series1 (pandas.Series): 数据序列1\n\n        series2 (pandas.Series): 数据序列2\n\n    Returns:\n        pandas.Series: 最大值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        max = tafunc.max(klines.close, klines.open)\n        print(list(max))\n    '
    max_data = np.maximum(series1, series2)
    return max_data

def median(series, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    中位数: 求series在n个周期内居于中间的数值\n    \n        注意:\n            1. 当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n            2. 对n个周期内所有series排序后, 若n为奇数, 则选择第(n + 1) / 2个为中位数, 若n为偶数, 则中位数是(n / 2)以及(n / 2 + 1)的平均数\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 中位数序列\n\n    Example::\n\n        例1:\n            # 假设最近3日的收盘价为2727, 2754, 2748, 那么当前 median(df["close"], 3) 的返回值是2748\n            median3 = tafunc.median(df["close"], 3)\n\n        例2:\n            # 假设最近4日的开盘价为2752, 2743, 2730, 2728, 那么当前 median(df["open"], 4) 的返回值是2736.5\n            median4 = tafunc.median(df["open"], 4)\n    '
    median_data = series.rolling(n).median()
    return median_data

def exist(cond, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    判断n个周期内, 是否有满足cond的条件, 若满足则值为1, 不满足为0\n\n    Args:\n        cond (array_like): 条件\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 判断结果序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        # 判断4个周期中是否存在收盘价大于前一个周期的最高价, 存在返回1, 不存在则返回0\n        exist = tafunc.exist(klines.close > klines.high.shift(1), 4)\n        print(list(exist))\n    '
    exist_data = pd.Series(np.where(pd.Series(np.where(cond, 1, 0)).rolling(n).sum() > 0, 1, 0))
    return exist_data

def every(cond, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    判断n个周期内, 是否一直满足cond条件, 若满足则值为1, 不满足为0\n\n    Args:\n        cond (array_like): 条件\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 判断结果序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        # 判断在4周期内, 3周期的简单移动平均是否一直大于5周期的简单移动平均\n        every = tafunc.every(tafunc.ma(klines.close, 3) > tafunc.ma(klines.close, 5), 4)\n        print(list(every))\n    '
    every_data = pd.Series(np.where(pd.Series(np.where(cond, 1, 0)).rolling(n).sum() == n, 1, 0))
    return every_data

def hhv(series, n):
    if False:
        i = 10
        return i + 15
    '\n    求series在n个周期内的最高值\n\n        注意: n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 最高值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        hhv = tafunc.hhv(klines.high, 4)  # 求4个周期最高价的最大值, 即4周期高点(包含当前k线)\n        print(list(hhv))\n    '
    hhv_data = series.rolling(n).max()
    return hhv_data

def llv(series, n):
    if False:
        while True:
            i = 10
    '\n    求在n个周期内的最小值\n\n        注意: n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 最小值序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        llv = tafunc.llv(klines.low, 5)  # 求5根k线最低点(包含当前k线)\n        print(list(llv))\n    '
    llv_data = series.rolling(n).min()
    return llv_data

def avedev(series, n):
    if False:
        while True:
            i = 10
    '\n    平均绝对偏差: 求series在n周期内的平均绝对偏差\n\n        算法:\n            计算avedev(df["close"],3)在最近一根K线上的值:\n            (abs(df["close"] - (df["close"] + df["close"].shift(1) + df["close"].shift(2)) / 3) + abs(\n            df["close"].shift(1) - (df["close"] + df["close"].shift(1) + df["close"].shift(2)) / 3) + abs(\n            df["close"].shift(2) - (df["close"] + df["close"].shift(1) + df["close"].shift(2)) / 3)) / 3\n\n        注意: n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列\n\n    Args:\n        series (pandas.Series): 数据序列\n\n        n (int): 周期\n\n    Returns:\n        pandas.Series: 平均绝对偏差序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, TqSim, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)\n        # 计算收盘价在5周期内的平均绝对偏差, 表示5个周期内每个周期的收盘价与5周期收盘价的平均值的差的绝对值的平均值, 判断收盘价与其均值的偏离程度:\n        avedev = tafunc.avedev(klines.close, 5)\n        print(list(avedev))\n    '

    def mad(x):
        if False:
            return 10
        return np.fabs(x - x.mean()).mean()
    avedev_data = series.rolling(window=n).apply(mad, raw=True)
    return avedev_data

def _to_ns_timestamp(input_time):
    if False:
        i = 10
        return i + 15
    '\n    辅助函数: 将传入的时间转换为int类型的纳秒级时间戳\n\n    Args:\n    input_time (str/ int/ float/ datetime.datetime): 需要转换的时间:\n        * str: str 类型的时间，如Quote行情时间的datetime字段 (eg. 2019-10-14 14:26:01.000000)\n\n        * int: int 类型纳秒级或秒级时间戳\n\n        * float: float 类型纳秒级或秒级时间戳，如K线或tick的datetime字段 (eg. 1.57103449e+18)\n\n        * datetime.datetime: datetime 模块中 datetime 类型\n\n    Returns:\n        int : int 类型纳秒级时间戳\n    '
    if type(input_time) in {int, float, np.float64, np.float32, np.int64, np.int32}:
        if input_time > 2 ** 32:
            return int(input_time)
        else:
            return int(input_time * 1000000000.0)
    elif isinstance(input_time, str):
        return _str_to_timestamp_nano(input_time)
    elif isinstance(input_time, datetime.datetime):
        return _datetime_to_timestamp_nano(input_time)
    else:
        raise TypeError('暂不支持此类型的转换')

def time_to_ns_timestamp(input_time):
    if False:
        for i in range(10):
            print('nop')
    '\n    将传入的时间转换为int类型的纳秒级时间戳\n\n    Args:\n        input_time (str/ int/ float/ datetime.datetime): 需要转换的时间:\n            * str: str 类型的时间，如Quote行情时间的datetime字段 (eg. 2019-10-14 14:26:01.000000)\n\n            * int: int 类型的纳秒级或秒级时间戳\n\n            * float: float 类型的纳秒级或秒级时间戳，如K线或tick的datetime字段 (eg. 1.57103449e+18)\n\n            * datetime.datetime: datetime 模块中的 datetime 类型时间\n\n    Returns:\n        int : int 类型的纳秒级时间戳\n\n    Example::\n\n        from tqsdk.tafunc import time_to_ns_timestamp\n        print(time_to_ns_timestamp("2019-10-14 14:26:01.000000"))  # 将%Y-%m-%d %H:%M:%S.%f 格式的str类型转为纳秒时间戳\n        print(time_to_ns_timestamp(1571103122))  # 将秒级转为纳秒时间戳\n        print(time_to_ns_timestamp(datetime.datetime(2019, 10, 14, 14, 26, 1)))  # 将datetime.datetime时间转为纳秒时间戳\n    '
    return _to_ns_timestamp(input_time)

def time_to_s_timestamp(input_time):
    if False:
        return 10
    '\n    将传入的时间转换为int类型的秒级时间戳\n\n    Args:\n        input_time (str/ int/ float/ datetime.datetime): 需要转换的时间:\n            * str: str 类型的时间，如Quote行情时间的datetime字段 (eg. 2019-10-14 14:26:01.000000)\n\n            * int: int 类型的纳秒级或秒级时间戳\n\n            * float: float 类型的纳秒级或秒级时间戳，如K线或tick的datetime字段 (eg. 1.57103449e+18)\n\n            * datetime.datetime: datetime 模块中的 datetime 类型时间\n\n    Returns:\n        int : int类型的秒级时间戳\n\n    Example::\n\n        from tqsdk.tafunc import time_to_s_timestamp\n        print(time_to_s_timestamp(1.57103449e+18))  # 将纳秒级时间戳转为秒级时间戳\n        print(time_to_s_timestamp("2019-10-14 14:26:01.000000"))  # 将%Y-%m-%d %H:%M:%S.%f 格式的str类型时间转为秒级时间戳\n        print(time_to_s_timestamp(datetime.datetime(2019, 10, 14, 14, 26, 1)))  # 将datetime.datetime时间转为秒时间戳\n    '
    return int(_to_ns_timestamp(input_time) / 1000000000.0)

def time_to_str(input_time):
    if False:
        print('Hello World!')
    '\n    将传入的时间转换为 %Y-%m-%d %H:%M:%S.%f 格式的 str 类型\n\n    Args:\n        input_time (int/ float/ datetime.datetime): 需要转换的时间:\n\n            * int: int 类型的纳秒级或秒级时间戳\n\n            * float: float 类型的纳秒级或秒级时间戳，如K线或tick的datetime字段 (eg. 1.57103449e+18)\n\n            * datetime.datetime: datetime 模块中的 datetime 类型时间\n\n    Returns:\n        str : %Y-%m-%d %H:%M:%S.%f 格式的 str 类型时间\n\n    Example::\n\n        from tqsdk.tafunc import time_to_str\n        print(time_to_str(1.57103449e+18))  # 将纳秒级时间戳转为%Y-%m-%d %H:%M:%S.%f 格式的str类型时间\n        print(time_to_str(1571103122))  # 将秒级时间戳转为%Y-%m-%d %H:%M:%S.%f 格式的str类型时间\n        print(time_to_str(datetime.datetime(2019, 10, 14, 14, 26, 1)))  # 将datetime.datetime时间转为%Y-%m-%d %H:%M:%S.%f 格式的str类型时间\n    '
    ts = _to_ns_timestamp(input_time) / 1000000000.0
    dt = datetime.datetime.fromtimestamp(ts)
    dt = dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    return dt

def time_to_datetime(input_time):
    if False:
        print('Hello World!')
    '\n    将传入的时间转换为 datetime.datetime 类型\n\n    Args:\n        input_time (int/ float/ str): 需要转换的时间:\n\n            * int: int 类型的纳秒级或秒级时间戳\n\n            * float: float 类型的纳秒级或秒级时间戳，如K线或tick的datetime字段 (eg. 1.57103449e+18)\n\n            * str: str 类型的时间，如Quote行情时间的 datetime 字段 (eg. 2019-10-14 14:26:01.000000)\n\n    Returns:\n        datetime.datetime : datetime 模块中的 datetime 类型时间\n\n    Example::\n\n        from tqsdk.tafunc import time_to_datetime\n        print(time_to_datetime(1.57103449e+18))  # 将纳秒级时间戳转为datetime.datetime时间\n        print(time_to_datetime(1571103122))  # 将秒级时间戳转为datetime.datetime时间\n        print(time_to_datetime("2019-10-14 14:26:01.000000"))  # 将%Y-%m-%d %H:%M:%S.%f 格式的str类型时间转为datetime.datetime时间\n    '
    ts = _to_ns_timestamp(input_time) / 1000000000.0
    dt = datetime.datetime.fromtimestamp(ts)
    return dt

def barlast(cond):
    if False:
        print('Hello World!')
    '\n    返回一个序列，其中每个值表示从上一次条件成立到当前的周期数\n\n    (注： 如果从cond序列第一个值到某个位置之间没有True，则此位置的返回值为 -1； 条件成立的位置上的返回值为0)\n\n\n    Args:\n        cond (pandas.Series): 条件序列(序列中的值需为 True 或 False)\n\n    Returns:\n        pandas.Series : 周期数序列（其长度和 cond 相同；最后一个值即为最后一次条件成立到最新一个数据的周期数）\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth\n        from tqsdk.tafunc import barlast\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        klines = api.get_kline_serial("SHFE.cu1912", 60)\n        # print(list(klines.close))\n        # print(list(klines.open))\n        # print(list(klines.close > klines.open))\n        n = barlast(klines.close > klines.open)  # 获取周期数序列\n        print(list(n))\n        print(n.iloc[-1])  # 获取最后一根k线到上一次满足 "收盘价大于开盘价" 条件的k线的周期数\n        api.close()\n\n    '
    cond = cond.to_numpy()
    v = np.array(~cond, dtype=int)
    c = np.cumsum(v)
    x = c[cond]
    d = np.diff(np.concatenate(([0], x)))
    if len(d) == 0:
        return pd.Series([-1] * len(cond))
    v[cond] = -d
    r = np.cumsum(v)
    r[:x[0]] = -1
    return pd.Series(r)

def _get_t_series(series: pd.Series, dur: int, expire_datetime: int):
    if False:
        print('Hello World!')
    t = pd.Series(pd.to_timedelta(expire_datetime - (series / 1000000000.0 + dur), unit='s'))
    return (t.dt.days * 86400 + t.dt.seconds) / (360 * 86400)

def _get_d1(series: pd.Series, k: float, r: float, v: Union[float, pd.Series], t: Union[float, pd.Series]):
    if False:
        i = 10
        return i + 15
    return pd.Series(np.where((v <= 0) | (t <= 0), np.nan, (np.log(series / k) + (r + 0.5 * np.power(v, 2)) * t) / (v * np.sqrt(t))))

def _get_cdf(series: pd.Series):
    if False:
        i = 10
        return i + 15
    s = series.loc[series.notna()]
    return pd.concat([series.loc[series.isna()], pd.Series(stats.norm.cdf(s), index=s.index)], verify_integrity=True)

def _get_pdf(series: pd.Series):
    if False:
        i = 10
        return i + 15
    s = series.loc[series.notna()]
    return pd.concat([series.loc[series.isna()], pd.Series(stats.norm.pdf(s), index=s.index)], verify_integrity=True)

def _get_options_class(series: pd.Series, option_class: Union[str, pd.Series]):
    if False:
        print('Hello World!')
    "\n    根据价格序列 series，和指定的 option_class\n\n    Args:\n        option_class (str / Series[str]): CALL / PUT / Series(['CALL', 'CALL', 'CALL', 'PUT'])\n\n    Returns:\n        Series[int] :  长度和 series 一致，Series([1, 1, 1, 1]) / Series([-1, -1, -1, -1]) / Series([1, 1, 1, -1]), 对于无效的参数值为 Series([nan, nan, nan, nan])\n\n    "
    if type(option_class) is str and option_class in ['CALL', 'PUT']:
        return Series([1 if option_class == 'CALL' else -1 for _ in range(series.size)])
    elif type(option_class) is Series and series.size == option_class.size:
        return option_class.map({'CALL': 1, 'PUT': -1})
    else:
        return Series([float('nan') for _ in range(series.size)])

def get_t(df, expire_datetime):
    if False:
        while True:
            i = 10
    '\n    计算 K 线序列对应的年化到期时间，主要用于计算期权相关希腊指标时，需要得到计算出序列对应的年化到期时间\n\n    Args:\n        df (pandas.DataFrame): Dataframe 格式的 K 线序列\n\n        expire_datetime (int): 到期日, 秒级时间戳\n\n    Returns:\n        pandas.Series : 返回的 df 对应的年化时间序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote(\'SHFE.cu2006C45000\')\n        klines = api.get_kline_serial([\'SHFE.cu2006C45000\', \'SHFE.cu2006\'], 24 * 60 * 60, 50)\n        t = tafunc.get_t(klines, quote.expire_datetime)\n        print(t)\n        api.close()\n    '
    return pd.Series(_get_t_series(df['datetime'], df['duration'], expire_datetime))

def get_his_volatility(df, quote):
    if False:
        print('Hello World!')
    '\n    计算某个合约的历史波动率\n\n    Args:\n        df (pandas.DataFrame): Dataframe 格式的 K 线序列\n\n        quote (tqsdk.objs.Quote): df 序列对应合约对象\n\n    Returns:\n        float : 返回的 df 对应的历史波动率\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote(\'SHFE.cu2006\')\n        klines = api.get_kline_serial(\'SHFE.cu2006\', 24 * 60 * 60, 50)\n        v = tafunc.get_his_volatility(klines, quote)\n        print(v)\n        api.close()\n    '
    if quote and quote.instrument_id == df['symbol'][0]:
        trading_time = quote.trading_time
    else:
        trading_time = None
    return _get_volatility(df['close'], df['duration'], trading_time)

def _get_volatility(series: pd.Series, dur: Union[pd.Series, int]=86400, trading_time: list=None) -> float:
    if False:
        i = 10
        return i + 15
    series_u = np.log(series.shift(1)[1:] / series[1:])
    series_u = series_u[~np.isnan(series_u)]
    if series_u.size < 2:
        return float('nan')
    seconds_per_day = 24 * 60 * 60
    dur = dur[0] if isinstance(dur, pd.Series) else dur
    if dur < 24 * 60 * 60 and trading_time:
        periods = _get_period_timestamp(0, trading_time.get('day', []) + trading_time.get('night', []))
        seconds_per_day = sum([p[1] - p[0] for p in periods]) / 1000000000.0
    return math.sqrt(250 * seconds_per_day / dur * np.cov(series_u))

def get_bs_price(series, k, r, v, t, option_class):
    if False:
        return 10
    '\n    计算期权 BS 模型理论价格\n\n    Args:\n        series (pandas.Series): 标的价格序列\n\n        k (float): 期权行权价\n\n        r (float): 无风险利率\n\n        v (float / pandas.Series): 波动率\n\n            * float: 对于 series 中每个元素都使用相同的 v 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 v 中对应的值计算理论价\n\n        t (float / pandas.Series): 年化到期时间，例如：还有 100 天到期，则年化到期时间为 100/360\n\n            * float: 对于 series 中每个元素都使用相同的 t 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 t 中对应的值计算理论价\n\n        option_class (str / pandas.Series): 期权方向，必须是两者其一，否则返回的序列值全部为 nan\n\n            * str: "CALL" 或者 "PUT"\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，每个元素的值为 "CALL" 或者 "PUT"\n\n    Returns:\n        pandas.Series: 返回该序列理论价\n\n    Example::\n\n        import pandas as pd\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006")\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 10)\n        v = tafunc.get_his_volatility(ks, quote)  # 历史波动率\n\n        option = api.get_quote("SHFE.cu2006C45000")\n        klines = api.get_kline_serial(["SHFE.cu2006C45000", "SHFE.cu2006"], 24 * 60 * 60, 10)\n        t = tafunc.get_t(klines, option.expire_datetime)\n        bs_price = tafunc.get_bs_price(klines["close1"], 45000, 0.025, v, t, option.option_class)  # 理论价\n        print(list(bs_price.round(2)))\n        api.close()\n    '
    o = _get_options_class(series, option_class=option_class)
    d1 = _get_d1(series, k, r, v, t)
    d2 = pd.Series(np.where(np.isnan(d1), np.nan, d1 - v * np.sqrt(t)))
    return pd.Series(np.where(np.isnan(d1), np.nan, o * (series * _get_cdf(o * d1) - k * np.exp(-r * t) * _get_cdf(o * d2))))

def get_delta(series, k, r, v, t, option_class, d1=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    计算期权希腊指标 delta 值\n\n    Args:\n        series (pandas.Series): 标的价格序列\n\n        k (float): 期权行权价\n\n        r (float): 无风险利率\n\n        v (float / pandas.Series): 波动率\n\n            * float: 对于 series 中每个元素都使用相同的 v 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 v 中对应的值计算理论价\n\n        t (float / pandas.Series): 年化到期时间，例如：还有 100 天到期，则年化到期时间为 100/360\n\n            * float: 对于 series 中每个元素都使用相同的 t 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 t 中对应的值计算理论价\n\n        option_class (str / pandas.Series): 期权方向，必须是两者其一，否则返回的序列值全部为 nan\n\n            * str: "CALL" 或者 "PUT"\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，每个元素的值为 "CALL" 或者 "PUT"\n\n        d1 (None | pandas.Series): [可选] 序列对应的 BS 公式中 b1 值\n\n    Returns:\n        pandas.Series: 该序列的 delta 值\n\n\n    Example::\n\n        import pandas as pd\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006")\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 10)\n        v = tafunc.get_his_volatility(ks, quote)  # 历史波动率\n\n        option = api.get_quote("SHFE.cu2006C45000")\n        klines = api.get_kline_serial(["SHFE.cu2006C45000", "SHFE.cu2006"], 24 * 60 * 60, 10)\n        t = tafunc.get_t(klines, option.expire_datetime)\n        impv = tafunc.get_impv(klines["close1"], klines["close"], 45000, 0.025, v, t, "CALL")\n        delta = tafunc.get_delta(klines["close1"], 45000, 0.025, v, t, "CALL")\n        print("delta", list(delta))\n        api.close()\n\n    '
    o = _get_options_class(series, option_class=option_class)
    if d1 is None:
        d1 = _get_d1(series, k, r, v, t)
    return pd.Series(np.where(np.isnan(d1), np.nan, pd.Series(o * _get_cdf(o * d1))))

def get_gamma(series, k, r, v, t, d1=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    计算期权希腊指标 gamma 值\n\n    Args:\n        series (pandas.Series): 标的价格序列\n\n        k (float): 期权行权价\n\n        r (float): 无风险利率\n\n        v (float / pandas.Series): 波动率\n\n            * float: 对于 series 中每个元素都使用相同的 v 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 v 中对应的值计算理论价\n\n        t (float / pandas.Series): 年化到期时间，例如：还有 100 天到期，则年化到期时间为 100/360\n\n            * float: 对于 series 中每个元素都使用相同的 t 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 t 中对应的值计算理论价\n\n        d1 (None | pandas.Series): [可选] 序列对应的 BS 公式中 b1 值\n\n    Returns:\n        pandas.Series: 该序列的 gamma 值\n\n\n    Example::\n\n        import pandas as pd\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006")\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 10)\n        v = tafunc.get_his_volatility(ks, quote)  # 历史波动率\n\n        option = api.get_quote("SHFE.cu2006C45000")\n        klines = api.get_kline_serial(["SHFE.cu2006C45000", "SHFE.cu2006"], 24 * 60 * 60, 10)\n        t = tafunc.get_t(klines, option.expire_datetime)\n        impv = tafunc.get_impv(klines["close1"], klines["close"], 45000, 0.025, v, t, "CALL")\n        gamma = tafunc.get_gamma(klines["close1"], 45000, 0.025, v, t)\n        print("gamma", list(gamma))\n        api.close()\n\n    '
    if d1 is None:
        d1 = _get_d1(series, k, r, v, t)
    return pd.Series(np.where(np.isnan(d1), np.nan, _get_pdf(d1) / (series * v * np.sqrt(t))))

def get_theta(series, k, r, v, t, option_class, d1=None):
    if False:
        while True:
            i = 10
    '\n    计算期权希腊指标 theta 值\n\n    Args:\n        series (pandas.Series): 标的价格序列\n\n        k (float): 期权行权价\n\n        r (float): 无风险利率\n\n        v (float / pandas.Series): 波动率\n\n            * float: 对于 series 中每个元素都使用相同的 v 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 v 中对应的值计算理论价\n\n        t (float / pandas.Series): 年化到期时间，例如：还有 100 天到期，则年化到期时间为 100/360\n\n            * float: 对于 series 中每个元素都使用相同的 t 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 t 中对应的值计算理论价\n\n        option_class (str / pandas.Series): 期权方向，必须是两者其一，否则返回的序列值全部为 nan\n\n            * str: "CALL" 或者 "PUT"\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，每个元素的值为 "CALL" 或者 "PUT"\n\n        d1 (None | pandas.Series): [可选] 序列对应的 BS 公式中 b1 值\n\n    Returns:\n        pandas.Series: 该序列的 theta 值\n\n\n    Example::\n\n        import pandas as pd\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006")\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 10)\n        v = tafunc.get_his_volatility(ks, quote)  # 历史波动率\n\n        option = api.get_quote("SHFE.cu2006C45000")\n        klines = api.get_kline_serial(["SHFE.cu2006C45000", "SHFE.cu2006"], 24 * 60 * 60, 10)\n        t = tafunc.get_t(klines, option.expire_datetime)\n        impv = tafunc.get_impv(klines["close1"], klines["close"], 45000, 0.025, v, t, "CALL")\n        theta = tafunc.get_theta(klines["close1"], 45000, 0.025, v, t, "CALL")\n        print("theta", list(theta))\n        api.close()\n\n    '
    o = _get_options_class(series, option_class=option_class)
    if d1 is None:
        d1 = _get_d1(series, k, r, v, t)
    d2 = pd.Series(np.where(np.isnan(d1), np.nan, d1 - v * np.sqrt(t)))
    return pd.Series(np.where(np.isnan(d1), np.nan, pd.Series(-v * series * _get_pdf(d1) / (2 * np.sqrt(t)) - o * r * k * np.exp(-r * t) * _get_cdf(o * d2))))

def get_vega(series, k, r, v, t, d1=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    计算期权希腊指标 vega 值\n\n    Args:\n        series (pandas.Series): 标的价格序列\n\n        k (float): 期权行权价\n\n        r (float): 无风险利率\n\n        v (float / pandas.Series): 波动率\n\n            * float: 对于 series 中每个元素都使用相同的 v 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 v 中对应的值计算理论价\n\n        t (float / pandas.Series): 年化到期时间，例如：还有 100 天到期，则年化到期时间为 100/360\n\n            * float: 对于 series 中每个元素都使用相同的 t 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 t 中对应的值计算理论价\n\n        d1 (None | pandas.Series): [可选] 序列对应的 BS 公式中 b1 值\n\n    Returns:\n        pandas.Series: 该序列的 vega 值\n\n\n    Example::\n\n        import pandas as pd\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006")\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 10)\n        v = tafunc.get_his_volatility(ks, quote)  # 历史波动率\n\n        option = api.get_quote("SHFE.cu2006C45000")\n        klines = api.get_kline_serial(["SHFE.cu2006C45000", "SHFE.cu2006"], 24 * 60 * 60, 10)\n        t = tafunc.get_t(klines, option.expire_datetime)\n        impv = tafunc.get_impv(klines["close1"], klines["close"], 45000, 0.025, v, t, "CALL")\n        vega = tafunc.get_vega(klines["close1"], 45000, 0.025, v, t)\n        print("vega", list(vega))\n        api.close()\n\n    '
    if d1 is None:
        d1 = _get_d1(series, k, r, v, t)
    return pd.Series(np.where(np.isnan(d1), np.nan, series * np.sqrt(t) * _get_pdf(d1)))

def get_rho(series, k, r, v, t, option_class, d1=None):
    if False:
        print('Hello World!')
    '\n    计算期权希腊指标 rho 值\n\n    Args:\n        series (pandas.Series): 标的价格序列\n\n        k (float): 期权行权价\n\n        r (float): 无风险利率\n\n        v (float / pandas.Series): 波动率\n\n            * float: 对于 series 中每个元素都使用相同的 v 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 v 中对应的值计算理论价\n\n        t (float / pandas.Series): 年化到期时间，例如：还有 100 天到期，则年化到期时间为 100/360\n\n            * float: 对于 series 中每个元素都使用相同的 t 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 t 中对应的值计算理论价\n\n        option_class (str / pandas.Series): 期权方向，必须是两者其一，否则返回的序列值全部为 nan\n\n            * str: "CALL" 或者 "PUT"\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，每个元素的值为 "CALL" 或者 "PUT"\n\n        d1 (None | pandas.Series): [可选] 序列对应的 BS 公式中 b1 值\n\n    Returns:\n        pandas.Series: 该序列的 rho 值\n\n\n    Example::\n\n        import pandas as pd\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006")\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 10)\n        v = tafunc.get_his_volatility(ks, quote)  # 历史波动率\n\n        option = api.get_quote("SHFE.cu2006C45000")\n        klines = api.get_kline_serial(["SHFE.cu2006C45000", "SHFE.cu2006"], 24 * 60 * 60, 10)\n        t = tafunc.get_t(klines, option.expire_datetime)\n        impv = tafunc.get_impv(klines["close1"], klines["close"], 45000, 0.025, v, t, "CALL")\n        rho = tafunc.get_rho(klines["close1"], 45000, 0.025, v, t, "CALL")\n        print("rho", list(rho))\n        api.close()\n\n    '
    o = _get_options_class(series, option_class=option_class)
    if d1 is None:
        d1 = _get_d1(series, k, r, v, t)
    d2 = pd.Series(np.where(np.isnan(d1), np.nan, d1 - v * np.sqrt(t)))
    return pd.Series(np.where(np.isnan(d1), np.nan, o * k * t * np.exp(-r * t) * _get_cdf(o * d2)))

def get_impv(series, series_option, k, r, init_v, t, option_class):
    if False:
        for i in range(10):
            print('nop')
    '\n    计算期权隐含波动率\n\n    Args:\n        series (pandas.Series): 标的价格序列\n\n        series_option (pandas.Series): 期权价格序列，与 series 长度应该相同\n\n        k (float): 期权行权价\n\n        r (float): 无风险利率\n\n        init_v (float / pandas.Series): 初始波动率，迭代初始值\n\n            * float: 对于 series 中每个元素都使用相同的 init_v 计算隐含波动率\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 init_v 中对应的值计算隐含波动率\n\n        t (float / pandas.Series): 年化到期时间，例如：还有 100 天到期，则年化到期时间为 100/360\n\n            * float: 对于 series 中每个元素都使用相同的 t 计算理论价\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，对于 series 中每个元素都使用 t 中对应的值计算理论价\n\n        option_class (str / pandas.Series): 期权方向，必须是两者其一，否则返回的序列值全部为 nan\n\n            * str: "CALL" 或者 "PUT"\n\n            * pandas.Series: 其元素个数应该和 series 元素个数相同，每个元素的值为 "CALL" 或者 "PUT"\n\n    Returns:\n        pandas.Series: 该序列的隐含波动率\n\n\n    Example::\n\n        import pandas as pd\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        quote = api.get_quote("SHFE.cu2006")\n        ks = api.get_kline_serial("SHFE.cu2006", 24 * 60 * 60, 10)\n        v = tafunc.get_his_volatility(ks, quote)  # 历史波动率\n\n        option = api.get_quote("SHFE.cu2006C45000")\n        klines = api.get_kline_serial(["SHFE.cu2006C45000", "SHFE.cu2006"], 24 * 60 * 60, 10)\n        t = tafunc.get_t(klines, option.expire_datetime)\n        impv = tafunc.get_impv(klines["close1"], klines["close"], 45000, 0.025, v, t, "CALL")\n        print("impv", list((impv * 100).round(2)))\n        api.close()\n    '
    o = _get_options_class(series, option_class=option_class)
    lower_limit = o * (series - k * np.exp(-r * t))
    x = pd.Series(np.where((series_option < lower_limit) | (t <= 0), np.nan, init_v))
    y = pd.Series(np.where(np.isnan(x), np.nan, get_bs_price(series, k, r, x, t, option_class)))
    vega = get_vega(series, k, r, x, t)
    diff_x = pd.Series(np.where(np.isnan(vega) | (vega < 1e-08), np.nan, (series_option - y) / vega))
    while not pd.DataFrame.all((np.abs(series_option - y) < 1e-08) | np.isnan(diff_x)):
        x = pd.Series(np.where(np.isnan(x) | np.isnan(diff_x), x, np.where(x + diff_x < 0, x / 2, np.where(diff_x > x / 2, x * 1.5, x + diff_x))))
        y = pd.Series(np.where(np.isnan(x), np.nan, get_bs_price(series, k, r, x, t, option_class)))
        vega = get_vega(series, k, r, x, t)
        diff_x = pd.Series(np.where(np.isnan(vega) | (vega < 1e-08), np.nan, (series_option - y) / vega))
    return x

def get_ticks_info(df):
    if False:
        print('Hello World!')
    '\n    计算 ticks 开平方向\n\n    Args:\n        df (pandas.DataFrame): Dataframe 格式的 ticks 序列\n\n    Returns:\n        pandas.Series: 返回序列的开平方向序列\n\n    Example::\n\n        from tqsdk import TqApi, TqAuth, tafunc\n\n        api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n        ticks = api.get_tick_serial(\'SHFE.cu2006\', 100)\n        ticksinfo = tafunc.get_ticks_info(ticks)\n        for i, v in ticksinfo.items():\n            print(f"{tafunc.time_to_str(ticks[\'datetime\'][i])[5:21]}  {ticks[\'last_price\'][i]}  {v}")\n        api.close()\n\n        # 预计的输出是这样的:\n        04-27 10:54:11.5   42640.0   多换\n        04-27 10:54:12.0   42640.0   多换\n        04-27 10:54:16.5   42640.0   多换\n        ......\n        04-27 10:55:10.0   42660.0   双平\n        04-27 10:55:10.5   42660.0   双平\n        04-27 10:55:14.0   42670.0   双平\n    '
    if 'open_interest' not in df.keys():
        raise Exception(f'get_ticks_info 参数必须是 ticks，由 api.get_tick_serial 返回的对象。')
    df_pre = df.copy().shift(1)
    df_pre['price_diff'] = df['last_price'] - df_pre['last_price']
    df_pre['oi_diff'] = df['open_interest'] - df_pre['open_interest']
    df_pre['vol_diff'] = df['volume'] - df_pre['volume']
    df_pre['pc'] = np.where(df['last_price'] <= df_pre['bid_price1'], -1, np.where(df['last_price'] >= df_pre['ask_price1'], 1, np.sign(df_pre['price_diff'])))
    pc_g = df_pre['pc'] > 0
    df_pre['info'] = pd.Series(np.where(df_pre['oi_diff'] > 0, np.where(pc_g, '多开', '空开'), np.where(df_pre['oi_diff'] < 0, np.where(pc_g, '空平', '多平'), np.where(df_pre['oi_diff'] == 0, np.where(pc_g, '多换', '空换'), ''))))
    df_pre.loc[df_pre['pc'] == 0, 'info'] = '换手'
    df_pre.loc[(df_pre['oi_diff'] < 0) & (df_pre['oi_diff'] + df_pre['vol_diff'] == 0), 'info'] = '双平'
    df_pre.loc[(df_pre['oi_diff'] > 0) & (df_pre['oi_diff'] == df_pre['vol_diff']), 'info'] = '双开'
    df_pre.loc[df_pre['vol_diff'] == 0, 'info'] = ''
    return df_pre['info']

def get_dividend_df(stock_dividend_ratio, cash_dividend_ratio):
    if False:
        while True:
            i = 10
    '\n    计算复权系数矩阵\n\n    Args:\n        stock_dividend_ratio (list): 除权表（可以由 quote.stock_dividend_ratio 取得）\n\n        cash_dividend_ratio (list): 除息表（可以由 quote.cash_dividend_ratio 取得）\n\n    Returns:\n        pandas.Dataframe: 复权系数矩阵， Dataframe 对象有 ["datetime", "stock_dividend", "cash_dividend"] 三列。\n    '
    stock_dividend_df = pd.DataFrame({'datetime': [_datetime_to_timestamp_nano(datetime.datetime.strptime(s.split(',')[0], '%Y%m%d')) for s in stock_dividend_ratio], 'stock_dividend': np.array([float(s.split(',')[1]) for s in stock_dividend_ratio])})
    cash_dividend_df = pd.DataFrame({'datetime': [_datetime_to_timestamp_nano(datetime.datetime.strptime(s.split(',')[0], '%Y%m%d')) for s in cash_dividend_ratio], 'cash_dividend': [float(s.split(',')[1]) for s in cash_dividend_ratio]})
    dividend_df = pd.merge(stock_dividend_df, cash_dividend_df, on=['datetime'], how='outer', sort=True)
    dividend_df.fillna(0.0, inplace=True)
    return dividend_df

def get_dividend_factor(dividend_df, last_item, item):
    if False:
        while True:
            i = 10
    '\n    返回 item 项对应的复权因子。\n\n    Args:\n        dividend_df (pandas.Dataframe): 除权除息矩阵表\n\n        last_item (dict): 前一个 tickItem / klineItem\n\n        item (dict): 当前 tickItem / klineItem\n\n    Returns:\n        float: 复权因子\n\n    '
    last_dt = last_item['datetime']
    dt = item['datetime']
    if last_dt and dt:
        gt = dividend_df['datetime'].gt(last_dt)
        if gt.any():
            dividend_first = dividend_df[gt].iloc[0]
            if dt >= dividend_first['datetime']:
                c = last_item['close'] if last_item['close'] else last_item['last_price']
                return (1 - dividend_first['cash_dividend'] / c) / (1 + dividend_first['stock_dividend'])
    return 1

def _tq_pstdev(data: Series, mu: float):
    if False:
        for i in range(10):
            print('nop')
    '\n    计算标准差\n    标准库提供的方法 statistics.pstdev 在 py3.6,py3.7 版本下参数 mean 不能设定为指定值，所以这里另外计算。\n    '
    n = data.shape[0]
    assert n >= 1
    return math.sqrt(sum((data - mu) ** 2) / n)

def get_sharp(series, trading_days_of_year=250, r=0.025):
    if False:
        i = 10
        return i + 15
    '\n    年化夏普率\n\n    Args:\n        series (pandas.Series): 每日收益率序列\n\n        trading_days_of_year (int): 年化交易日数量\n\n        r (float): 无风险利率\n\n    Returns:\n        float: 年化夏普率\n    '
    rf = _get_daily_risk_free(trading_days_of_year, r)
    mean = series.mean()
    stddev = _tq_pstdev(series, mu=mean)
    return trading_days_of_year ** (1 / 2) * (mean - rf) / stddev if stddev else float('inf')

def get_sortino(series, trading_days_of_year=250, r=0.025):
    if False:
        return 10
    '\n    年化索提诺比率\n\n    Args:\n        series (pandas.Series): 每日收益率序列\n\n        trading_days_of_year (int): 年化交易日数量\n\n        r (float): 无风险利率\n\n    Returns:\n        float: 年化索提诺比率\n    '
    rf = _get_daily_risk_free(trading_days_of_year, r)
    mean = series.mean()
    left_daily_yield = series.loc[series < rf]
    stddev = _tq_pstdev(left_daily_yield, mu=rf) if left_daily_yield.shape[0] > 0 else 0
    return (trading_days_of_year * left_daily_yield.shape[0] / series.shape[0]) ** (1 / 2) * (mean - rf) / stddev if stddev else float('inf')

def get_calmar(series, max_drawdown, trading_days_of_year=250, r=0.025):
    if False:
        for i in range(10):
            print('nop')
    '\n    年化卡玛比率\n\n    Args:\n        series (pandas.Series): 每日收益率序列\n\n        max_drawdown (float): 最大回撤\n\n        trading_days_of_year (int): 年化交易日数量\n\n        r (float): 无风险利率\n\n    Returns:\n        float: 年化夏普率\n    '
    rf = _get_daily_risk_free(trading_days_of_year, r)
    if max_drawdown and max_drawdown == max_drawdown:
        mean = series.mean()
        return trading_days_of_year ** (1 / 2) * (mean - rf) / max_drawdown
    return float('inf')

def _get_daily_risk_free(trading_days_of_year, r):
    if False:
        return 10
    '日化无风险利率'
    return pow(r + 1, 1 / trading_days_of_year) - 1

def _cum_counts(s: Series):
    if False:
        return 10
    '\n    统计连续为1的个数, 用于计算最大连续盈利/亏损天数\n    input:  [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]\n    output: [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 0]\n    '
    return s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)