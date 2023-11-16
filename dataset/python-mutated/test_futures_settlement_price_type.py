from rqalpha.apis import *
__config__ = {'base': {'start_date': '2016-01-01', 'end_date': '2016-01-31', 'frequency': '1d', 'accounts': {'future': 1000000}}, 'mod': {'sys_accounts': {'futures_settlement_price_type': 'settlement'}}}

def test_futures_settlement_price_type():
    if False:
        i = 10
        return i + 15

    def init(context):
        if False:
            return 10
        context.fixed = True
        context.symbol = 'IC1603'
        context.total = 0

    def handle_bar(context, bar_dict):
        if False:
            for i in range(10):
                print('nop')
        if context.fixed:
            buy_open(context.symbol, 1)
            context.fixed = False
        context.total += 1

    def after_trading(context):
        if False:
            print('Hello World!')
        pos = get_position(context.symbol)
        if context.total == 2:
            assert pos.position_pnl == (6364.6 - 6657.0) * 200
        elif context.total == 3:
            assert pos.position_pnl == (6468 - 6351.2) * 200
    return locals()

def test_futures_de_listed():
    if False:
        i = 10
        return i + 15
    ' 期货合约到期交割 '
    __config__ = {'base': {'start_date': '2016-03-17', 'end_date': '2016-03-21', 'frequency': '1d', 'accounts': {'future': 1000000}}, 'mod': {'sys_transaction_cost': {'futures_commission_multiplier': 0}, 'sys_accounts': {'futures_settlement_price_type': 'settlement'}}}

    def init(context):
        if False:
            while True:
                i = 10
        pass

    def before_trading(context):
        if False:
            i = 10
            return i + 15
        if context.now.date() == date(2016, 3, 21):
            assert context.portfolio.total_value == context.total_value + (5944.29 - 5760.0) * 200, '合约交割后的总权益有误'

    def handle_bar(context, bar_dict):
        if False:
            for i in range(10):
                print('nop')
        if context.now.date() == date(2016, 3, 17):
            buy_open('IC1603', 1)
            context.total_value = context.portfolio.total_value
    return locals()