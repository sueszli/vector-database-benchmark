"""
用于统计数据
"""
import alert
'\n计算收益\nsold 卖价\nbuy 买入价\n'

def percentage(sold, buy):
    if False:
        i = 10
        return i + 15
    x = (sold - buy) * 1.0 / buy * 100
    print(round(x, 2))
    return x
'\n计算买入价\nsold: 卖出的价格\n需要的幅度\n'

def plan_buy_price(sold, percent):
    if False:
        print('Hello World!')
    buy = sold * 1.0 / (1 + percent * 1.0 / 100)
    print(round(buy, 2))
    return buy
a = 1.567
print(round(a, 2))