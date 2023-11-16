"""
Topic: 字典的数据运算
Desc : 
"""

def calc_dict():
    if False:
        i = 10
        return i + 15
    prices = {'ACME': 45.23, 'AAPL': 612.78, 'IBM': 205.55, 'HPQ': 37.2, 'FB': 10.75}
    min_price = min(zip(prices.values(), prices.keys()))
    max_price = max(zip(prices.values(), prices.keys()))
    prices_sorted = sorted(zip(prices.values(), prices.keys()))
    prices_and_names = zip(prices.values(), prices.keys())
    print(min(prices_and_names))
    print(max(prices_and_names))
    min(prices)
    max(prices)
    min(prices, key=lambda k: prices[k])
    max(prices, key=lambda k: prices[k])
    min_value = prices[min(prices, key=lambda k: prices[k])]
    prices = {'AAA': 45.23, 'ZZZ': 45.23}
    min(zip(prices.values(), prices.keys()))
    max(zip(prices.values(), prices.keys()))