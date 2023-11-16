__author__ = 'limin'
'\n网格交易策略 (难度：中级)\n参考: https://www.shinnytech.com/blog/grid-trading/\n注: 该示例策略仅用于功能示范, 实盘时请根据自己的策略/经验进行修改\n'
from functools import reduce
from tqsdk import TqApi, TqAuth, TargetPosTask
SYMBOL = 'DCE.jd2011'
START_PRICE = 4247
GRID_AMOUNT = 10
api = TqApi(auth=TqAuth('快期账户', '账户密码'))
grid_region_long = [0.005] * GRID_AMOUNT
grid_region_short = [0.005] * GRID_AMOUNT
grid_volume_long = [i for i in range(GRID_AMOUNT + 1)]
grid_volume_short = [i for i in range(GRID_AMOUNT + 1)]
grid_prices_long = [reduce(lambda p, r: p * (1 - r), grid_region_long[:i], START_PRICE) for i in range(GRID_AMOUNT + 1)]
grid_prices_short = [reduce(lambda p, r: p * (1 + r), grid_region_short[:i], START_PRICE) for i in range(GRID_AMOUNT + 1)]
print('策略开始运行, 起始价位: %f, 多头每格持仓手数:%s, 多头每格的价位:%s, 空头每格的价位:%s' % (START_PRICE, grid_volume_long, grid_prices_long, grid_prices_short))
quote = api.get_quote(SYMBOL)
target_pos = TargetPosTask(api, SYMBOL)
position = api.get_position(SYMBOL)

def wait_price(layer):
    if False:
        return 10
    '等待行情最新价变动到其他档位,则进入下一档位或回退到上一档位; 如果从下一档位回退到当前档位,则设置为当前对应的持仓手数;\n        layer : 当前所在第几个档位层次; layer>0 表示多头方向, layer<0 表示空头方向\n    '
    if layer > 0 or quote.last_price <= grid_prices_long[1]:
        while True:
            api.wait_update()
            if layer < GRID_AMOUNT and quote.last_price <= grid_prices_long[layer + 1]:
                target_pos.set_target_volume(grid_volume_long[layer + 1])
                print('最新价: %f, 进入: 多头第 %d 档' % (quote.last_price, layer + 1))
                wait_price(layer + 1)
                target_pos.set_target_volume(grid_volume_long[layer + 1])
            if quote.last_price > grid_prices_long[layer]:
                print('最新价: %f, 回退到: 多头第 %d 档' % (quote.last_price, layer))
                return
    elif layer < 0 or quote.last_price >= grid_prices_short[1]:
        layer = -layer
        while True:
            api.wait_update()
            if layer < GRID_AMOUNT and quote.last_price >= grid_prices_short[layer + 1]:
                target_pos.set_target_volume(-grid_volume_short[layer + 1])
                print('最新价: %f, 进入: 空头第 %d 档' % (quote.last_price, layer + 1))
                wait_price(-(layer + 1))
                target_pos.set_target_volume(-grid_volume_short[layer + 1])
            if quote.last_price < grid_prices_short[layer]:
                print('最新价: %f, 回退到: 空头第 %d 档' % (quote.last_price, layer))
                return
while True:
    api.wait_update()
    wait_price(0)
    target_pos.set_target_volume(0)