from rqalpha.apis import *
import talib

def init(context):
    if False:
        return 10
    context.s1 = '000001.XSHE'
    context.s2 = '601988.XSHG'
    context.s3 = '000068.XSHE'
    context.stocks = [context.s1, context.s2, context.s3]
    context.TIME_PERIOD = 14
    context.HIGH_RSI = 85
    context.LOW_RSI = 30
    context.ORDER_PERCENT = 0.3

def handle_bar(context, bar_dict):
    if False:
        print('Hello World!')
    for stock in context.stocks:
        prices = history_bars(stock, context.TIME_PERIOD + 1, '1d', 'close')
        rsi_data = talib.RSI(prices, timeperiod=context.TIME_PERIOD)[-1]
        cur_position = get_position(stock).quantity
        target_available_cash = context.portfolio.cash * context.ORDER_PERCENT
        if rsi_data > context.HIGH_RSI and cur_position > 0:
            order_target_value(stock, 0)
        if rsi_data < context.LOW_RSI:
            logger.info('target available cash caled: ' + str(target_available_cash))
            order_value(stock, target_available_cash)