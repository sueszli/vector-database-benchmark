def init(context):
    if False:
        for i in range(10):
            print('nop')
    context.s1 = '000001.XSHE'

def before_trading(context):
    if False:
        for i in range(10):
            print('nop')
    pass

def handle_bar(context, bar_dict):
    if False:
        for i in range(10):
            print('nop')
    order_shares(context.s1, 1000)

def after_trading(context):
    if False:
        while True:
            i = 10
    pass
__config__ = {'base': {'start_date': '2015-01-09', 'end_date': '2016-01-12', 'frequency': '1d', 'matching_type': 'current_bar', 'benchmark': '000300.XSHG', 'accounts': {'stock': 1000000}}, 'extra': {'log_level': 'error', 'show': True}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}