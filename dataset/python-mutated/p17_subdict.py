"""
Topic: dict子集
Desc : 
"""

def sub_dict():
    if False:
        return 10
    prices = {'ACME': 45.23, 'AAPL': 612.78, 'IBM': 205.55, 'HPQ': 37.2, 'FB': 10.75}
    p1 = {key: value for (key, value) in prices.items() if value > 200}
    tech_names = {'AAPL', 'IBM', 'HPQ', 'MSFT'}
    p2 = {key: value for (key, value) in prices.items() if key in tech_names}
if __name__ == '__main__':
    sub_dict()