__author__ = 'limin'
'\nR-Breaker策略(隔夜留仓) (难度：初级)\n参考: https://www.shinnytech.com/blog/r-breaker\n注: 该示例策略仅用于功能示范, 实盘时请根据自己的策略/经验进行修改\n'
from tqsdk import TqApi, TqAuth, TargetPosTask
SYMBOL = 'SHFE.au2006'
STOP_LOSS_PRICE = 10

def get_index_line(klines):
    if False:
        for i in range(10):
            print('nop')
    '计算指标线'
    high = klines.high.iloc[-2]
    low = klines.low.iloc[-2]
    close = klines.close.iloc[-2]
    pivot = (high + low + close) / 3
    b_break = high + 2 * (pivot - low)
    s_setup = pivot + (high - low)
    s_enter = 2 * pivot - low
    b_enter = 2 * pivot - high
    b_setup = pivot - (high - low)
    s_break = low - 2 * (high - pivot)
    print('已计算新标志线, 枢轴点: %f, 突破买入价: %f, 观察卖出价: %f, 反转卖出价: %f, 反转买入价: %f, 观察买入价: %f, 突破卖出价: %f' % (pivot, b_break, s_setup, s_enter, b_enter, b_setup, s_break))
    return (pivot, b_break, s_setup, s_enter, b_enter, b_setup, s_break)
api = TqApi(auth=TqAuth('快期账户', '账户密码'))
quote = api.get_quote(SYMBOL)
klines = api.get_kline_serial(SYMBOL, 24 * 60 * 60)
position = api.get_position(SYMBOL)
target_pos = TargetPosTask(api, SYMBOL)
target_pos_value = position.pos_long - position.pos_short
open_position_price = position.open_price_long if target_pos_value > 0 else position.open_price_short
(pivot, b_break, s_setup, s_enter, b_enter, b_setup, s_break) = get_index_line(klines)
while True:
    target_pos.set_target_volume(target_pos_value)
    api.wait_update()
    if api.is_changing(klines.iloc[-1], 'datetime'):
        (pivot, b_break, s_setup, s_enter, b_enter, b_setup, s_break) = get_index_line(klines)
    '交易规则'
    if api.is_changing(quote, 'last_price'):
        print('最新价: ', quote.last_price)
        if target_pos_value > 0 and open_position_price - quote.last_price >= STOP_LOSS_PRICE or (target_pos_value < 0 and quote.last_price - open_position_price >= STOP_LOSS_PRICE):
            target_pos_value = 0
        if target_pos_value > 0:
            if quote.highest > s_setup and quote.last_price < s_enter:
                print('多头持仓,当日内最高价超过观察卖出价后跌破反转卖出价: 反手做空')
                target_pos_value = -3
                open_position_price = quote.last_price
        elif target_pos_value < 0:
            if quote.lowest < b_setup and quote.last_price > b_enter:
                print('空头持仓,当日最低价低于观察买入价后超过反转买入价: 反手做多')
                target_pos_value = 3
                open_position_price = quote.last_price
        elif target_pos_value == 0:
            if quote.last_price > b_break:
                print('空仓,盘中价格超过突破买入价: 开仓做多')
                target_pos_value = 3
                open_position_price = quote.last_price
            elif quote.last_price < s_break:
                print('空仓,盘中价格跌破突破卖出价: 开仓做空')
                target_pos_value = -3
                open_position_price = quote.last_price