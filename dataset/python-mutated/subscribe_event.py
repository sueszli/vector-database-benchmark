from rqalpha.apis import *

def on_trade_handler(event):
    if False:
        i = 10
        return i + 15
    trade = event.trade
    order = event.order
    account = event.account
    logger.info('*' * 10 + 'Trade Handler' + '*' * 10)
    logger.info(trade)
    logger.info(order)
    logger.info(account)

def on_order_handler(event):
    if False:
        i = 10
        return i + 15
    order = event.order
    logger.info('*' * 10 + 'Order Handler' + '*' * 10)
    logger.info(order)

def init(context):
    if False:
        print('Hello World!')
    logger.info('init')
    context.s1 = '000001.XSHE'
    update_universe(context.s1)
    context.fired = False
    subscribe_event(EVENT.TRADE, on_trade_handler)
    subscribe_event(EVENT.ORDER_CREATION_PASS, on_order_handler)

def before_trading(context):
    if False:
        return 10
    pass

def handle_bar(context, bar_dict):
    if False:
        i = 10
        return i + 15
    if not context.fired:
        order_percent(context.s1, 1)
        context.fired = True