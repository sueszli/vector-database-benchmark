def init(context):
    if False:
        i = 10
        return i + 15
    context.s1 = '000905.XSHG'
    subscribe(context.s1)

def handle_bar(context, bar_dict):
    if False:
        for i in range(10):
            print('nop')
    his = history_bars(context.s1, 10, '1d', 'close')
    if his[9] / his[8] < 0.97:
        if len(context.portfolio.positions) > 0:
            for stock in context.portfolio.positions.keys():
                order_target_percent(stock, 0)
        return
    position = context.portfolio.positions[context.s1].quantity
    if position < 100:
        High = history_bars(context.s1, 3, '1d', 'high')
        Low = history_bars(context.s1, 3, '1d', 'low')
        Close = history_bars(context.s1, 3, '1d', 'close')
        Open = history_bars(context.s1, 3, '1d', 'open')
        HH = max(High[:2])
        LC = min(Close[:2])
        HC = max(Close[:2])
        LL = min(Low[:2])
        Openprice = Open[2]
        current_price = Close[2]
        Range = max(HH - LC, HC - LL)
        K1 = 0.9
        BuyLine = Openprice + K1 * Range
        if current_price > BuyLine:
            order_target_percent(context.s1, 1)
    hist = history_bars(context.s1, 3, '1d', 'close')
    case1 = 1 - hist[2] / hist[0] >= 0.06
    case2 = hist[1] / hist[0] <= 0.92
    if case1 or case2:
        order_target_percent(context.s1, 0)
__config__ = {'base': {'start_date': '2013-01-01', 'end_date': '2015-12-29', 'frequency': '1d', 'matching_type': 'current_bar', 'benchmark': '000300.XSHG', 'accounts': {'stock': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}