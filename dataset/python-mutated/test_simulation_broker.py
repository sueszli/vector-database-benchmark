__config__ = {'base': {'start_date': '2015-04-11', 'end_date': '2015-04-20', 'frequency': '1d', 'accounts': {'stock': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}, 'sys_simulation': {'volume_limit': True, 'volume_percent': 2e-06}}}

def test_open_auction_match():
    if False:
        i = 10
        return i + 15
    __config__ = {'mod': {'sys_simulation': {'volume_limit': True, 'volume_percent': 2e-06}}}

    def init(context):
        if False:
            print('Hello World!')
        context.s = '000001.XSHE'
        context.bar = None
        context.first_day = True

    def open_auction(context, bar_dict):
        if False:
            for i in range(10):
                print('nop')
        bar = bar_dict[context.s]
        if context.first_day:
            order_shares(context.s, 1000, bar.limit_up * 0.99)
            assert get_position(context.s).quantity == 900
            assert get_position(context.s).avg_price == bar.open

    def handle_bar(context, bar_dict):
        if False:
            print('Hello World!')
        if context.first_day:
            bar = bar_dict[context.s]
            assert get_position(context.s).quantity == 1000
            assert get_position(context.s).avg_price == (bar.open * 900 + bar.close * 100) / 1000
            context.first_day = False
    return locals()

def test_vwap_match():
    if False:
        while True:
            i = 10
    __config__ = {'mod': {'sys_simulation': {'volume_limit': True, 'matching_type': 'vwap'}}}

    def init(context):
        if False:
            i = 10
            return i + 15
        context.s = '000001.XSHE'
        context.first_day = True
        context.vwap_price = None

    def handle_bar(context, bar_dict):
        if False:
            for i in range(10):
                print('nop')
        if context.first_day == 1:
            bar = bar_dict[context.s]
            vwap_order = order_shares(context.s, 1000)
            assert bar.total_turnover / bar.volume == vwap_order.avg_price
            context.first_day = False
    return locals()