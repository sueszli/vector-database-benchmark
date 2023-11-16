from rqalpha.apis import *
import talib

def init(context):
    if False:
        print('Hello World!')
    context.s1 = '000001.XSHE'
    context.SHORTPERIOD = 20
    context.LONGPERIOD = 120

def handle_bar(context, bar_dict):
    if False:
        i = 10
        return i + 15
    prices = history_bars(context.s1, context.LONGPERIOD + 1, '1d', 'close')
    short_avg = talib.SMA(prices, context.SHORTPERIOD)
    long_avg = talib.SMA(prices, context.LONGPERIOD)
    plot('short avg', short_avg[-1])
    plot('long avg', long_avg[-1])
    cur_position = get_position(context.s1).quantity
    shares = context.portfolio.cash / bar_dict[context.s1].close
    if short_avg[-1] - long_avg[-1] < 0 and short_avg[-2] - long_avg[-2] > 0 and (cur_position > 0):
        order_target_value(context.s1, 0)
    if short_avg[-1] - long_avg[-1] > 0 and short_avg[-2] - long_avg[-2] < 0:
        order_shares(context.s1, shares)