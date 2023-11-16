import numpy as np

def init(context):
    if False:
        for i in range(10):
            print('nop')
    context.s1 = 'AG1612'
    context.s2 = 'AU1612'
    context.counter = 0
    context.window = 60
    context.ratio = 15
    context.up_cross_up_limit = False
    context.down_cross_down_limit = False
    context.entry_score = 2
    subscribe([context.s1, context.s2])

def before_trading(context):
    if False:
        print('Hello World!')
    context.counter = 0

def handle_bar(context, bar_dict):
    if False:
        for i in range(10):
            print('nop')
    long_pos_a = get_position(context.s1, POSITION_DIRECTION.LONG)
    short_pos_a = get_position(context.s1, POSITION_DIRECTION.SHORT)
    long_pos_b = get_position(context.s2, POSITION_DIRECTION.LONG)
    short_pos_b = get_position(context.s2, POSITION_DIRECTION.SHORT)
    context.counter += 1
    if context.counter > context.window:
        price_array_a = history_bars(context.s1, context.window, '1m', 'close')
        price_array_b = history_bars(context.s2, context.window, '1m', 'close')
        spread_array = price_array_a - context.ratio * price_array_b
        std = np.std(spread_array)
        mean = np.mean(spread_array)
        up_limit = mean + context.entry_score * std
        down_limit = mean - context.entry_score * std
        price_a = bar_dict[context.s1].close
        price_b = bar_dict[context.s2].close
        spread = price_a - context.ratio * price_b
        if spread <= down_limit and (not context.down_cross_down_limit):
            logger.info('spread: {}, mean: {}, down_limit: {}'.format(spread, mean, down_limit))
            logger.info('创建买入价差中...')
            qty_a = 1 - long_pos_a.quantity
            qty_b = context.ratio - short_pos_b.sell_quantity
            if qty_a > 0:
                buy_open(context.s1, qty_a)
            if qty_b > 0:
                sell_open(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.down_cross_down_limit = True
                logger.info('买入价差仓位创建成功!')
        if spread >= mean and context.down_cross_down_limit:
            logger.info('spread: {}, mean: {}, down_limit: {}'.format(spread, mean, down_limit))
            logger.info('对买入价差仓位进行平仓操作中...')
            qty_a = long_pos_a.quantity
            qty_b = short_pos_b.quantity
            if qty_a > 0:
                sell_close(context.s1, qty_a)
            if qty_b > 0:
                buy_close(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.down_cross_down_limit = False
                logger.info('买入价差仓位平仓成功!')
        if spread >= up_limit and (not context.up_cross_up_limit):
            logger.info('spread: {}, mean: {}, up_limit: {}'.format(spread, mean, up_limit))
            logger.info('创建卖出价差中...')
            qty_a = 1 - short_pos_a.quantity
            qty_b = context.ratio - long_pos_b.quantity
            if qty_a > 0:
                sell_open(context.s1, qty_a)
            if qty_b > 0:
                buy_open(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.up_cross_up_limit = True
                logger.info('卖出价差仓位创建成功')
        if spread < mean and context.up_cross_up_limit:
            logger.info('spread: {}, mean: {}, up_limit: {}'.format(spread, mean, up_limit))
            logger.info('对卖出价差仓位进行平仓操作中...')
            qty_a = short_pos_a.quantity
            qty_b = long_pos_b.quantity
            if qty_a > 0:
                buy_close(context.s1, qty_a)
            if qty_b > 0:
                sell_close(context.s2, qty_b)
            if qty_a == 0 and qty_b == 0:
                context.up_cross_up_limit = False
                logger.info('卖出价差仓位平仓成功!')