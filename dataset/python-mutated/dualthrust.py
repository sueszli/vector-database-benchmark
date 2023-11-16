__author__ = 'limin'
'\nDual Thrust策略 (难度：中级)\n参考: https://www.shinnytech.com/blog/dual-thrust\n注: 该示例策略仅用于功能示范, 实盘时请根据自己的策略/经验进行修改\n'
from tqsdk import TqApi, TqAuth, TargetPosTask
SYMBOL = 'DCE.jd2011'
NDAY = 5
K1 = 0.2
K2 = 0.2
api = TqApi(auth=TqAuth('快期账户', '账户密码'))
print('策略开始运行')
quote = api.get_quote(SYMBOL)
klines = api.get_kline_serial(SYMBOL, 24 * 60 * 60)
target_pos = TargetPosTask(api, SYMBOL)

def dual_thrust(quote, klines):
    if False:
        while True:
            i = 10
    current_open = klines.iloc[-1]['open']
    HH = max(klines.high.iloc[-NDAY - 1:-1])
    HC = max(klines.close.iloc[-NDAY - 1:-1])
    LC = min(klines.close.iloc[-NDAY - 1:-1])
    LL = min(klines.low.iloc[-NDAY - 1:-1])
    range = max(HH - LC, HC - LL)
    buy_line = current_open + range * K1
    sell_line = current_open - range * K2
    print('当前开盘价: %f, 上轨: %f, 下轨: %f' % (current_open, buy_line, sell_line))
    return (buy_line, sell_line)
(buy_line, sell_line) = dual_thrust(quote, klines)
while True:
    api.wait_update()
    if api.is_changing(klines.iloc[-1], ['datetime', 'open']):
        (buy_line, sell_line) = dual_thrust(quote, klines)
    if api.is_changing(quote, 'last_price'):
        if quote.last_price > buy_line:
            print('高于上轨,目标持仓 多头3手')
            target_pos.set_target_volume(3)
        elif quote.last_price < sell_line:
            print('低于下轨,目标持仓 空头3手')
            target_pos.set_target_volume(-3)
        else:
            print('未穿越上下轨,不调整持仓')