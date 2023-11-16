"""
Topic: 转换并聚集函数
Desc : 
"""
import os

def trans_reduce():
    if False:
        while True:
            i = 10
    nums = [1, 2, 3, 4, 5]
    s = sum((x * x for x in nums))
    print(s)
    files = os.listdir('dirname')
    if any((name.endswith('.py') for name in files)):
        print('There be python!')
    else:
        print('Sorry, no python.')
    s = ('ACME', 50, 123.45)
    print(','.join((str(x) for x in s)))
    portfolio = [{'name': 'GOOG', 'shares': 50}, {'name': 'YHOO', 'shares': 75}, {'name': 'AOL', 'shares': 20}, {'name': 'SCOX', 'shares': 65}]
    min_shares = min((s['shares'] for s in portfolio))
    min_shares = min((s['shares'] for s in portfolio))
    min_shares = min(portfolio, key=lambda s: s['shares'])
if __name__ == '__main__':
    trans_reduce()