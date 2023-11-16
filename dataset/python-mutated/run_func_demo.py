from rqalpha.apis import *
from rqalpha import run_func

def init(context):
    if False:
        print('Hello World!')
    logger.info('init')
    context.s1 = '000001.XSHE'
    update_universe(context.s1)
    context.fired = False

def before_trading(context):
    if False:
        while True:
            i = 10
    pass

def handle_bar(context, bar_dict):
    if False:
        for i in range(10):
            print('nop')
    if not context.fired:
        order_percent(context.s1, 1)
        context.fired = True
config = {'base': {'start_date': '2016-06-01', 'end_date': '2016-12-01', 'benchmark': '000300.XSHG', 'accounts': {'stock': 100000}}, 'extra': {'log_level': 'verbose'}, 'mod': {'sys_analyser': {'enabled': True, 'plot': True}}}
run_func(init=init, before_trading=before_trading, handle_bar=handle_bar, config=config)