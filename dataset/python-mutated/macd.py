from rqalpha.apis import *
import talib

def init(context):
    if False:
        while True:
            i = 10
    context.s1 = '000001.XSHE'
    context.SHORTPERIOD = 12
    context.LONGPERIOD = 26
    context.SMOOTHPERIOD = 9
    context.OBSERVATION = 100

def handle_bar(context, bar_dict):
    if False:
        print('Hello World!')
    prices = history_bars(context.s1, context.OBSERVATION, '1d', 'close')
    (macd, signal, hist) = talib.MACD(prices, context.SHORTPERIOD, context.LONGPERIOD, context.SMOOTHPERIOD)
    plot('macd', macd[-1])
    plot('macd signal', signal[-1])
    if macd[-1] - signal[-1] < 0 and macd[-2] - signal[-2] > 0:
        curPosition = get_position(context.s1).quantity
        if curPosition > 0:
            order_target_value(context.s1, 0)
    if macd[-1] - signal[-1] > 0 and macd[-2] - signal[-2] < 0:
        order_target_percent(context.s1, 1)