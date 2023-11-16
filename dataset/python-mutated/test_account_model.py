from datetime import date
from numpy.testing import assert_equal, assert_almost_equal
from rqalpha.apis import *
__config__ = {'base': {'start_date': '2016-03-07', 'end_date': '2016-03-08', 'frequency': '1d', 'accounts': {'stock': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}

def test_stock_delist():
    if False:
        for i in range(10):
            print('nop')
    import datetime
    __config__ = {'base': {'start_date': '2018-12-25', 'end_date': '2019-01-05'}}

    def init(context):
        if False:
            while True:
                i = 10
        context.s = '000979.XSHE'
        context.fired = False
        context.total_value_before_delisted = None

    def handle_bar(context, _):
        if False:
            for i in range(10):
                print('nop')
        if not context.fired:
            order_shares(context.s, 20000)
            context.fired = True
        if context.now.date() == datetime.date(2018, 12, 27):
            context.total_value_before_delisted = context.portfolio.total_value
        if context.now.date() > datetime.date(2018, 12, 28):
            assert context.portfolio.total_value == context.total_value_before_delisted
    return locals()

def test_stock_dividend():
    if False:
        print('Hello World!')
    __config__ = {'base': {'start_date': '2012-06-04', 'end_date': '2018-07-9'}, 'extra': {'log_level': 'info'}}

    def init(context):
        if False:
            print('Hello World!')
        context.s = '601088.XSHG'
        context.last_cash = None

    def handle_bar(context, _):
        if False:
            i = 10
            return i + 15
        if context.now.date() in (date(2012, 6, 8), date(2017, 7, 7), date(2018, 7, 6)):
            context.last_cash = context.portfolio.cash
        elif context.now.date() == date(2012, 6, 4):
            order_shares(context.s, 1000)
        elif context.now.date() == date(2012, 6, 18):
            assert context.portfolio.cash == context.last_cash + 900
        elif context.now.date() == date(2017, 7, 11):
            assert context.portfolio.cash == context.last_cash + 2970
        elif context.now.date() == date(2018, 7, 9):
            assert context.portfolio.cash == context.last_cash + 910
    return locals()

def test_stock_transform():
    if False:
        while True:
            i = 10
    __config__ = {'base': {'start_date': '2015-05-06', 'end_date': '2015-05-20'}}

    def init(context):
        if False:
            for i in range(10):
                print('nop')
        context.s1 = '601299.XSHG'
        context.s2 = '601766.XSHG'
        context.cash_before_transform = None

    def handle_bar(context, _):
        if False:
            print('Hello World!')
        if context.now.date() == date(2015, 5, 6):
            order_shares(context.s1, 200)
            context.cash_before_transform = context.portfolio.cash
        elif context.now.date() >= date(2015, 5, 20):
            assert int(context.portfolio.positions[context.s2].quantity) == 220
            assert context.portfolio.cash == context.cash_before_transform
    return locals()

def test_stock_split():
    if False:
        i = 10
        return i + 15
    __config__ = {'base': {'start_date': '2016-05-26', 'end_date': '2016-05-27'}}

    def init(context):
        if False:
            print('Hello World!')
        context.s = '000035.XSHE'
        context.counter = 0
        context.cash_before_split = None

    def handle_bar(context, bar_dict):
        if False:
            while True:
                i = 10
        context.counter += 1
        if context.counter == 1:
            order_shares(context.s, 1000)
            assert_equal(get_position(context.s, POSITION_DIRECTION.LONG).quantity, 1000)
            context.cash_before_split = context.portfolio.cash
        elif context.counter == 2:
            position = get_position(context.s, POSITION_DIRECTION.LONG)
            assert_equal(position.quantity, 2000)
            assert_equal(position.trading_pnl, 0)
            assert_almost_equal(position.position_pnl, -140)
            assert_equal(context.portfolio.cash, context.cash_before_split)
    return locals()