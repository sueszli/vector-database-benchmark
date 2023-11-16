from rqalpha.apis import *
import talib

def init(context):
    if False:
        while True:
            i = 10
    context.s1 = 'IF1606'
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
        sell_qty = get_position(context.s1, POSITION_DIRECTION.SHORT).quantity
        if sell_qty > 0:
            buy_close(context.s1, 1)
        buy_open(context.s1, 1)
    if macd[-1] - signal[-1] < 0 and macd[-2] - signal[-2] > 0:
        buy_qty = get_position(context.s1, POSITION_DIRECTION.LONG).quantity
        if buy_qty > 0:
            sell_close(context.s1, 1)
        sell_open(context.s1, 1)