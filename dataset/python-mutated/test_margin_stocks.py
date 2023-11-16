from rqalpha.apis import *
__config__ = {'base': {'start_date': '2016-01-01', 'end_date': '2016-01-31', 'frequency': '1d', 'accounts': {'stock': 1000000}}, 'mod': {'sys_accounts': {'financing_stocks_restriction_enabled': True}}}

def test_margin_stocks():
    if False:
        return 10
    try:
        import rqdatac
    except ImportError:
        print('rqdatac not install, not test margin_stocks')
        return {}

    def init(context):
        if False:
            print('Hello World!')
        context.total = 0
        context.margin_symbol = '000001.XSHE'
        context.not_margin_symbol = '000004.XSHE'

    def handle_bar(context, bar_dict):
        if False:
            return 10
        if context.total == 0:
            order_shares(context.margin_symbol, 100)
            order_shares(context.not_margin_symbol, 100)
        elif context.total == 1:
            assert 100 == get_position(context.not_margin_symbol).quantity
            assert 100 == get_position(context.margin_symbol).quantity
        elif context.total == 2:
            finance(10000)
            order_shares(context.margin_symbol, 100)
            order_shares(context.not_margin_symbol, 100)
        elif context.total == 3:
            assert 100 == get_position(context.not_margin_symbol).quantity
            assert 200 == get_position(context.margin_symbol).quantity
        context.total += 1
    return locals()