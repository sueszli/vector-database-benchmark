from rqalpha.apis import *
__config__ = {'base': {'start_date': '2016-03-07', 'end_date': '2016-03-08', 'frequency': '1d', 'accounts': {'stock': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}

def test_stock_sellable():
    if False:
        i = 10
        return i + 15

    def init(context):
        if False:
            print('Hello World!')
        context.fired = False
        context.s = '000001.XSHE'

    def handle_bar(context, _):
        if False:
            print('Hello World!')
        if not context.fired:
            order_shares(context.s, 1000)
            sellable = context.portfolio.positions[context.s].sellable
            assert sellable == 0, 'wrong sellable {}, supposed to be {}'.format(sellable, 0)
            context.fired = True
    return locals()

def test_trading_pnl():
    if False:
        return 10
    __config__ = {'base': {'start_date': '2020-01-02', 'end_date': '2020-01-02', 'frequency': '1d', 'accounts': {'future': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}

    def init(context):
        if False:
            return 10
        context.quantity = 2
        context.open_price = None
        context.close_price = None

    def open_auction(context, bar_dict):
        if False:
            print('Hello World!')
        context.open_price = buy_open('IC2001', context.quantity).avg_price

    def handle_bar(context, bar_dict):
        if False:
            for i in range(10):
                print('nop')
        context.close_price = sell_close('IC2001', context.quantity).avg_price

    def after_trading(context):
        if False:
            print('Hello World!')
        pos = get_position('IC2001')
        assert pos.trading_pnl == (5361.8 - 5300.0) * context.quantity * 200
        assert pos.trading_pnl == (context.close_price - context.open_price) * context.quantity * 200
    return locals()