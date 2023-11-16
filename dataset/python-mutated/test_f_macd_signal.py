import talib
import numpy as np

def init(context):
    if False:
        i = 10
        return i + 15
    context.s1 = 'IF88'
    context.SHORTPERIOD = 12
    context.LONGPERIOD = 26
    context.SMOOTHPERIOD = 9
    context.OBSERVATION = 50
    subscribe(context.s1)

def handle_bar(context, bar_dict):
    if False:
        for i in range(10):
            print('nop')
    prices = history_bars(context.s1, context.OBSERVATION, '1d', 'close')
    (macd, signal, hist) = talib.MACD(prices, context.SHORTPERIOD, context.LONGPERIOD, context.SMOOTHPERIOD)
    if macd[-1] - signal[-1] > 0 and macd[-2] - signal[-2] < 0:
        sell_qty = context.portfolio.positions[context.s1].sell_quantity
        if sell_qty > 0:
            buy_close(context.s1, 1)
        buy_open(context.s1, 1)
    if macd[-1] - signal[-1] < 0 and macd[-2] - signal[-2] > 0:
        buy_qty = context.portfolio.positions[context.s1].buy_quantity
        if buy_qty > 0:
            sell_close(context.s1, 1)
        sell_open(context.s1, 1)
__config__ = {'base': {'start_date': '2014-09-01', 'end_date': '2016-09-05', 'frequency': '1d', 'matching_type': 'current_bar', 'benchmark': None, 'accounts': {'future': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}, 'sys_simulation': {'signal': True}}}