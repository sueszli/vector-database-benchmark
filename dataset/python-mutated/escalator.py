__author__ = 'Ringo'
'\n自动扶梯 策略 (难度：初级)\n参考: https://www.shinnytech.com/blog/escalator/\n注: 该示例策略仅用于功能示范, 实盘时请根据自己的策略/经验进行修改\n'
from tqsdk import TqApi, TqAuth, TargetPosTask
from tqsdk.ta import MA
SYMBOL = 'SHFE.rb2012'
(MA_SLOW, MA_FAST) = (8, 40)
api = TqApi(auth=TqAuth('快期账户', '账户密码'))
klines = api.get_kline_serial(SYMBOL, 60 * 60 * 24)
quote = api.get_quote(SYMBOL)
position = api.get_position(SYMBOL)
target_pos = TargetPosTask(api, SYMBOL)

def kline_range(num):
    if False:
        print('Hello World!')
    kl_range = (klines.iloc[num].close - klines.iloc[num].low) / (klines.iloc[num].high - klines.iloc[num].low)
    return kl_range

def ma_caculate(klines):
    if False:
        i = 10
        return i + 15
    ma_slow = MA(klines, MA_SLOW).iloc[-1].ma
    ma_fast = MA(klines, MA_FAST).iloc[-1].ma
    return (ma_slow, ma_fast)
(ma_slow, ma_fast) = ma_caculate(klines)
print('慢速均线为%.2f,快速均线为%.2f' % (ma_slow, ma_fast))
while True:
    api.wait_update()
    if api.is_changing(klines.iloc[-1], 'datetime'):
        (ma_slow, ma_fast) = ma_caculate(klines)
        print('慢速均线为%.2f,快速均线为%.2f' % (ma_slow, ma_fast))
    if api.is_changing(quote, 'last_price'):
        if position.pos_long == 0 and position.pos_short == 0:
            kl_range_cur = kline_range(-2)
            kl_range_pre = kline_range(-3)
            if klines.iloc[-2].close > max(ma_slow, ma_fast) and kl_range_pre <= 0.25 and (kl_range_cur >= 0.75):
                print('最新价为:%.2f 开多头' % quote.last_price)
                target_pos.set_target_volume(100)
            elif klines.iloc[-2].close < min(ma_slow, ma_fast) and kl_range_pre >= 0.75 and (kl_range_cur <= 0.25):
                print('最新价为:%.2f 开空头' % quote.last_price)
                target_pos.set_target_volume(-100)
            else:
                print('最新价位:%.2f ，未满足开仓条件' % quote.last_price)
        elif position.pos_long > 0:
            kline_low = min(klines.iloc[-2].low, klines.iloc[-3].low)
            if klines.iloc[-1].close <= kline_low - quote.price_tick:
                print('最新价为:%.2f,进行多头止损' % quote.last_price)
                target_pos.set_target_volume(0)
            else:
                print('多头持仓，当前价格 %.2f,多头离场价格%.2f' % (quote.last_price, kline_low - quote.price_tick))
        elif position.pos_short > 0:
            kline_high = max(klines.iloc[-2].high, klines.iloc[-3].high)
            if klines.iloc[-1].close >= kline_high + quote.price_tick:
                print('最新价为:%.2f 进行空头止损' % quote.last_price)
                target_pos.set_target_volume(0)
            else:
                print('空头持仓，当前价格 %.2f,空头离场价格%.2f' % (quote.last_price, kline_high + quote.price_tick))