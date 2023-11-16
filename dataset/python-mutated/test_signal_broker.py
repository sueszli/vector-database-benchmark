__config__ = {'base': {'start_date': '2015-04-10', 'end_date': '2015-04-10', 'frequency': '1d', 'accounts': {'stock': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}, 'sys_simulation': {'signal': True}}}

def test_price_limit():
    if False:
        return 10

    def handle_bar(context, bar_dict):
        if False:
            for i in range(10):
                print('nop')
        stock = '000001.XSHE'
        price = bar_dict[stock].limit_up * 0.99
        order_shares(stock, 100, price)
        assert get_position(stock).quantity == 100
        assert get_position(stock).avg_price == price
        order_shares(stock, 100, bar_dict[stock].limit_up)
        assert get_position(stock).quantity == 100
        assert get_position(stock).avg_price == price
    return locals()