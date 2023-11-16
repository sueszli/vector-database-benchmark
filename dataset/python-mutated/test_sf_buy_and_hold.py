def init(context):
    if False:
        i = 10
        return i + 15
    context.counter = 0
    subscribe('IH88')

def before_trading(context):
    if False:
        while True:
            i = 10
    pass

def handle_bar(context, bar_dict):
    if False:
        print('Hello World!')
    context.counter += 1
    if context.counter == 1:
        order_shares('510050.XSHG', 330000)
        sell_open('IH88', 1)

def after_trading(context):
    if False:
        for i in range(10):
            print('nop')
    pass
__config__ = {'base': {'start_date': '2016-06-01', 'end_date': '2016-10-05', 'frequency': '1d', 'matching_type': 'current_bar', 'benchmark': None, 'accounts': {'stock': 1000000, 'future': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}