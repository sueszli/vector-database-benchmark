from rqalpha.apis import *
from rqalpha.environment import Environment
__config__ = {'base': {'start_date': '2022-01-01', 'end_date': '2022-01-30', 'frequency': '1d', 'accounts': {'stock': 1000000, 'future': 1000000}}, 'mod': {'sys_transaction_cost': {'stock_commission_multiplier': 2, 'futures_commission_multiplier': 3}}}

def test_commission_multiplier():
    if False:
        for i in range(10):
            print('nop')

    def init(context):
        if False:
            for i in range(10):
                print('nop')
        context.s1 = '000001.XSHE'
        context.s2 = 'IC2203'
        context.fixed = True

    def handle_bar(context, bar_dict):
        if False:
            return 10
        if context.fixed:
            stock_order = order_percent(context.s1, 1)
            future_order = buy_open(context.s2, 1)
            env = Environment.get_instance()
            future_commission_info = env.data_proxy.get_commission_info(context.s2)
            context.fixed = False
            assert stock_order.transaction_cost == 16.66 * 59900 * 8 / 10000 * 2
            assert future_order.transaction_cost == 7308 * 200 * future_commission_info['open_commission_ratio'] * 3
    return locals()