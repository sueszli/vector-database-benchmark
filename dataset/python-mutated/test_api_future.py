from ..utils import assert_order
__config__ = {'base': {'start_date': '2016-03-07', 'end_date': '2016-03-08', 'frequency': '1d', 'accounts': {'future': 10000000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}

def test_buy_open():
    if False:
        for i in range(10):
            print('nop')

    def init(context):
        if False:
            return 10
        context.f1 = 'P88'
        subscribe(context.f1)

    def handle_bar(context, _):
        if False:
            for i in range(10):
                print('nop')
        o = buy_open(context.f1, 1)
        assert_order(o, order_book_id=context.f1, quantity=1, status=ORDER_STATUS.FILLED, side=SIDE.BUY, position_effect=POSITION_EFFECT.OPEN)
    return locals()

def test_sell_open():
    if False:
        for i in range(10):
            print('nop')

    def init(context):
        if False:
            for i in range(10):
                print('nop')
        context.f1 = 'P88'
        subscribe(context.f1)

    def handle_bar(context, _):
        if False:
            return 10
        o = sell_open(context.f1, 1)
        assert_order(o, order_book_id=context.f1, quantity=1, status=ORDER_STATUS.FILLED, side=SIDE.SELL, position_effect=POSITION_EFFECT.OPEN)
    return locals()

def test_buy_close():
    if False:
        while True:
            i = 10

    def init(context):
        if False:
            print('Hello World!')
        context.f1 = 'P88'
        subscribe(context.f1)

    def handle_bar(context, _):
        if False:
            while True:
                i = 10
        orders = buy_close(context.f1, 1)
        assert len(orders) == 0
    return locals()

def test_sell_close():
    if False:
        while True:
            i = 10

    def init(context):
        if False:
            while True:
                i = 10
        context.f1 = 'P88'
        subscribe(context.f1)

    def handle_bar(context, _):
        if False:
            print('Hello World!')
        orders = sell_close(context.f1, 1)
        assert len(orders) == 0
    return locals()

def test_close_today():
    if False:
        while True:
            i = 10

    def init(context):
        if False:
            i = 10
            return i + 15
        context.fired = False
        context.f1 = 'P88'
        subscribe(context.f1)

    def handle_bar(context, _):
        if False:
            for i in range(10):
                print('nop')
        if not context.fired:
            buy_open(context.f1, 2)
            sell_close(context.f1, 1, close_today=True)
            assert get_position(context.f1).quantity == 1
            context.fired = True
    return locals()

def test_future_order_to():
    if False:
        print('Hello World!')

    def init(context):
        if False:
            print('Hello World!')
        context.counter = 0
        context.f1 = 'P88'
        subscribe(context.f1)

    def handle_bar(context, _):
        if False:
            while True:
                i = 10
        context.counter += 1
        if context.counter == 1:
            order_to(context.f1, 3)
            assert get_position(context.f1).quantity == 3
            order_to(context.f1, 2)
            assert get_position(context.f1).quantity == 2
        elif context.counter == 2:
            order_to(context.f1, -2)
            assert get_position(context.f1, POSITION_DIRECTION.SHORT).quantity == 2
            order_to(context.f1, 1)
            assert get_position(context.f1, POSITION_DIRECTION.SHORT).quantity == 0
            assert get_position(context.f1, POSITION_DIRECTION.LONG).quantity == 1
    return locals()