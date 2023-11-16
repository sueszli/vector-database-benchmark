__config__ = {'base': {'start_date': '2015-04-13', 'end_date': '2015-05-10', 'frequency': '1d', 'accounts': {'stock': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}, 'sys_simulation': {'signal': True, 'management_fee': [('stock', 0.05)]}}}

def test_set_management_fee_rate():
    if False:
        print('Hello World!')

    def init(context):
        if False:
            while True:
                i = 10
        context.day_count = 0
        context.equity = 0

    def handle_bar(context, bar_dict):
        if False:
            print('Hello World!')
        context.day_count += 1
        if context.day_count == 1:
            stock = '000001.XSHE'
            order_shares(stock, 100)
            assert context.portfolio.positions[stock].quantity == 100
            context.fired = True
            context.total_value = context.portfolio.accounts['STOCK'].total_value
        if context.day_count == 2:
            assert context.portfolio.accounts['STOCK']._management_fees == context.total_value * 0.05
    return locals()

def test_set_management_function():
    if False:
        for i in range(10):
            print('nop')

    def management_fee_calculator(account, rate):
        if False:
            while True:
                i = 10
        return len(account.positions) * 100

    def init(context):
        if False:
            return 10
        context.day_count = 0
        context.portfolio.accounts['STOCK'].register_management_fee_calculator(management_fee_calculator)

    def handle_bar(context, bar_dict):
        if False:
            i = 10
            return i + 15
        context.day_count += 1
        if context.day_count == 1:
            stock = '000001.XSHE'
            order_shares(stock, 100)
            assert context.portfolio.positions[stock].quantity == 100
            context.fired = True
        if context.day_count == 4:
            assert context.portfolio.accounts['STOCK'].management_fees == 300
    return locals()