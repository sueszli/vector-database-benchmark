import numpy as np

def init(context):
    if False:
        for i in range(10):
            print('nop')
    context.s1 = 'AG88'
    context.s2 = 'AU88'
    context.counter = 0
    context.window = 60
    context.ratio = 15
    context.up_cross_up_limit = False
    context.down_cross_down_limit = False
    context.entry_score = 2.5
    subscribe([context.s1, context.s2])

def handle_bar(context, bar_dict):
    if False:
        for i in range(10):
            print('nop')
    context.counter += 1
    position_a = context.portfolio.positions[context.s1]
    position_b = context.portfolio.positions[context.s2]
    if context.counter > context.window:
        price_array_a = history_bars(context.s1, context.window, '1d', 'close')
        price_array_b = history_bars(context.s2, context.window, '1d', 'close')
        spread_array = price_array_a - context.ratio * price_array_b
        std = np.std(spread_array)
        mean = np.mean(spread_array)
        up_limit = mean + context.entry_score * std
        down_limit = mean - context.entry_score * std
        price_a = bar_dict[context.s1].close
        price_b = bar_dict[context.s2].close
        spread = price_a - context.ratio * price_b
        if spread <= down_limit and (not context.down_cross_down_limit):
            qty_a = 1 - position_a.buy_quantity
            qty_b = context.ratio - position_b.sell_quantity
            if qty_a > 0:
                buy_open(context.s1, qty_a)
            if qty_b > 0:
                sell_open(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.down_cross_down_limit = True
        if spread >= mean and context.down_cross_down_limit:
            qty_a = position_a.buy_quantity
            qty_b = position_b.sell_quantity
            if qty_a > 0:
                sell_close(context.s1, qty_a)
            if qty_b > 0:
                buy_close(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.down_cross_down_limit = False
        if spread >= up_limit and (not context.up_cross_up_limit):
            qty_a = 1 - position_a.sell_quantity
            qty_b = context.ratio - position_b.buy_quantity
            if qty_a > 0:
                sell_open(context.s1, qty_a)
            if qty_b > 0:
                buy_open(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.up_cross_up_limit = True
        if spread < mean and context.up_cross_up_limit:
            qty_a = position_a.sell_quantity
            qty_b = position_b.buy_quantity
            if qty_a > 0:
                buy_close(context.s1, qty_a)
            if qty_b > 0:
                sell_close(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.up_cross_up_limit = False
__config__ = {'base': {'start_date': '2014-06-01', 'end_date': '2015-08-01', 'frequency': '1d', 'matching_type': 'next_bar', 'benchmark': None, 'accounts': {'future': 1000000}}, 'extra': {'log_level': 'error'}, 'mod': {'sys_progress': {'enabled': True, 'show': True}}}