from ..utils import assert_order
__config__ = {'base': {'start_date': '2016-03-07', 'end_date': '2016-03-08', 'frequency': '1d', 'accounts': {'stock': 100000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}

def test_order_shares():
    if False:
        i = 10
        return i + 15
    __config__ = {'base': {'start_date': '2016-06-14', 'end_date': '2016-06-19'}}

    def init(context):
        if False:
            return 10
        context.counter = 0
        context.s1 = '000001.XSHE'

    def handle_bar(context, bar_dict):
        if False:
            while True:
                i = 10
        context.counter += 1
        if context.counter == 1:
            order_price = bar_dict[context.s1].limit_up
            o = order_shares(context.s1, 1910, order_price)
            assert_order(o, order_book_id=context.s1, side=SIDE.BUY, quantity=1900, price=order_price)
        elif context.counter == 3:
            assert context.portfolio.positions[context.s1].quantity == 2280
            o = order_shares(context.s1, -1010, bar_dict[context.s1].limit_down)
            assert_order(o, side=SIDE.SELL, quantity=1000, status=ORDER_STATUS.FILLED)
        elif context.counter == 4:
            assert context.portfolio.positions[context.s1].quantity == 1280
            o = order_shares(context.s1, -1280, bar_dict[context.s1].limit_down)
            assert_order(o, quantity=1280, status=ORDER_STATUS.FILLED)
            assert context.portfolio.positions[context.s1].quantity == 0
    return locals()

def test_order_lots():
    if False:
        while True:
            i = 10

    def init(context):
        if False:
            i = 10
            return i + 15
        context.s1 = '000001.XSHE'

    def handle_bar(context, bar_dict):
        if False:
            print('Hello World!')
        order_price = bar_dict[context.s1].limit_up
        o = order_lots(context.s1, 1, order_price)
        assert_order(o, side=SIDE.BUY, order_book_id=context.s1, quantity=100, price=order_price)
    return locals()

def test_order_value():
    if False:
        for i in range(10):
            print('nop')

    def init(context):
        if False:
            while True:
                i = 10
        context.s1 = '000001.XSHE'
        context.amount = 100

    def handle_bar(context, bar_dict):
        if False:
            while True:
                i = 10
        order_price = bar_dict[context.s1].limit_up
        o = order_value(context.s1, context.amount * order_price + 5, order_price)
        assert_order(o, side=SIDE.BUY, order_book_id=context.s1, quantity=context.amount, price=order_price)
    return locals()

def test_order_percent():
    if False:
        return 10

    def init(context):
        if False:
            i = 10
            return i + 15
        context.s1 = '000001.XSHE'

    def handle_bar(context, bar_dict):
        if False:
            print('Hello World!')
        o = order_percent(context.s1, 0.0001, bar_dict[context.s1].limit_up)
        assert_order(o, side=SIDE.BUY, order_book_id=context.s1, price=bar_dict[context.s1].limit_up)
    return locals()

def test_order_target_value():
    if False:
        for i in range(10):
            print('nop')

    def init(context):
        if False:
            for i in range(10):
                print('nop')
        context.order_count = 0
        context.s1 = '000001.XSHE'
        context.amount = 10000

    def handle_bar(context, bar_dict):
        if False:
            i = 10
            return i + 15
        o = order_target_percent(context.s1, 0.02, style=LimitOrder(bar_dict[context.s1].limit_up))
        assert_order(o, side=SIDE.BUY, order_book_id=context.s1, price=bar_dict[context.s1].limit_up)
    return locals()

def test_auto_switch_order_value():
    if False:
        print('Hello World!')
    __config__ = {'base': {'start_date': '2016-03-07', 'end_date': '2016-03-07', 'accounts': {'stock': 2000}}, 'mod': {'sys_accounts': {'auto_switch_order_value': True}}}

    def handle_bar(context, _):
        if False:
            return 10
        order_shares('000001.XSHE', 200)
        assert context.portfolio.positions['000001.XSHE'].quantity == 100
    return locals()

def test_order_target_portfolio():
    if False:
        i = 10
        return i + 15
    __config__ = {'base': {'start_date': '2019-07-30', 'end_date': '2019-08-05', 'accounts': {'stock': 1000000}}}

    def init(context):
        if False:
            while True:
                i = 10
        context.counter = 0

    def handle_bar(context, bar_dict):
        if False:
            for i in range(10):
                print('nop')
        context.counter += 1
        if context.counter == 1:
            order_target_portfolio({'000001.XSHE': 0.1, '000004.XSHE': 0.2})
            assert get_position('000001.XSHE').quantity == 6900
            assert get_position('000004.XSHE').quantity == 10500
        elif context.counter == 2:
            order_target_portfolio({'000004.XSHE': 0.1, '000005.XSHE': 0.2, '600519.XSHG': 0.6}, {'000004.XSHE': (18.5, 18), '000005.XSHE': (2.92,), '600519.XSHG': (970, 980)})
            assert get_position('000001.XSHE').quantity == 0
            assert get_position('000004.XSHE').quantity == 5600
            assert get_position('000005.XSHE').quantity == 68000
            assert get_position('600519.XSHG').quantity == 0
    return locals()

def test_order_target_portfolio_in_signal_mode():
    if False:
        print('Hello World!')
    __config__ = {'base': {'start_date': '2019-07-30', 'end_date': '2019-08-05', 'accounts': {'stock': 1000000}}, 'mod': {'sys_simulation': {'signal': True}}}

    def init(context):
        if False:
            while True:
                i = 10
        context.counter = 0

    def handle_bar(context, handle_bar):
        if False:
            i = 10
            return i + 15
        context.counter += 1
        if context.counter == 1:
            order_target_portfolio({'000001.XSHE': 0.1, '000004.XSHE': 0.2}, {'000001.XSHE': 14, '000004.XSHE': 10})
            assert get_position('000001.XSHE').quantity == 7100
            assert get_position('000004.XSHE').quantity == 0
    return locals()

def test_is_st_stock():
    if False:
        i = 10
        return i + 15
    __config__ = {'base': {'start_date': '2016-03-07', 'end_date': '2016-03-07'}}

    def handle_bar(_, __):
        if False:
            i = 10
            return i + 15
        for (order_book_id, expected_result) in [('600603.XSHG', [True, True]), ('600305.XSHG', [False, False])]:
            result = is_st_stock(order_book_id, 2)
            assert result == expected_result
    return locals()

def test_ksh():
    if False:
        for i in range(10):
            print('nop')
    '科创版买卖最低200股，大于就可以201，202股买卖'
    __config__ = {'base': {'start_date': '2019-07-30', 'end_date': '2019-08-05', 'accounts': {'stock': 1000000}}}

    def init(context):
        if False:
            print('Hello World!')
        context.counter = 0
        context.amount_s1 = 100
        context.amount_s2 = 200
        context.s1 = '688016.XSHG'
        context.s2 = '688010.XSHG'

    def handle_bar(context, bar_dict):
        if False:
            print('Hello World!')
        context.counter += 1
        if context.counter == 1:
            order_shares(context.s1, 201)
            order_shares(context.s2, 199)
            assert context.portfolio.positions[context.s1].quantity == 201
            assert context.portfolio.positions[context.s2].quantity == 0
        if context.counter == 2:
            order_lots(context.s1, 2)
            order_price_s1 = bar_dict[context.s1].close
            order_price_s2 = bar_dict[context.s2].close
            order_value(context.s1, context.amount_s1 * order_price_s1 + 5, order_price_s1)
            order_value(context.s2, context.amount_s2 * order_price_s2 + 5, order_price_s2)
            assert context.portfolio.positions[context.s1].quantity == 201
            assert context.portfolio.positions[context.s2].quantity == 0
    return locals()

def test_finance_repay():
    if False:
        while True:
            i = 10
    ' 测试融资还款接口 '
    financing_rate = 0.1
    money = 10000
    __config__ = {'base': {'start_date': '2016-01-01', 'end_date': '2016-01-31'}, 'mod': {'sys_accounts': {'financing_rate': financing_rate}}}

    def cal_interest(capital, days):
        if False:
            for i in range(10):
                print('nop')
        for i in range(days):
            capital += capital * financing_rate / 365
        return capital

    def init(context):
        if False:
            print('Hello World!')
        context.fixed = True
        context.total = 0

    def handle_bar(context, bar_dict):
        if False:
            i = 10
            return i + 15
        if context.fixed:
            finance(money)
            context.fixed = False
        if context.total == 5:
            assert context.stock_account.cash_liabilities == cal_interest(money, 5)
        elif context.total == 10:
            assert context.stock_account.cash_liabilities == cal_interest(money, 10)
            repay(10100)
        elif context.total == 11:
            assert context.stock_account.total_value == 99999972.5689376
            assert context.stock_account.cash_liabilities == 0
        context.total += 1
    return locals()