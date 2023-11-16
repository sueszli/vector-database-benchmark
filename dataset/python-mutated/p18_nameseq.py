"""
Topic: 通过名称访问序列
Desc : 
"""
from collections import namedtuple

def name_seq():
    if False:
        for i in range(10):
            print('nop')
    subscriber = namedtuple('Subscriber', ['addr', 'joined'])
    sub = subscriber('jonesy@example.com', '2012-10-19')
    print(sub)
    print(sub.addr, sub.joined)
    print(len(sub))
    (addr, joined) = sub
    print(addr, joined)

def compute_cost(records):
    if False:
        i = 10
        return i + 15
    total = 0.0
    for rec in records:
        total += rec[1] * rec[2]
    return total

def compute_cost2(records):
    if False:
        for i in range(10):
            print('nop')
    Stock = namedtuple('SSS', ['name', 'shares', 'price'])
    total = 0.0
    for rec in records:
        st = Stock(*rec)
        total += st.shares * st.price
    s = Stock('ACME', 100, 123.45)
    s = s._replace(shares=75)
    print(s)
    return total
Stock1 = namedtuple('Stock', ['name', 'shares', 'price', 'date', 'time'])
stock_prototype = Stock1('', 0, 0.0, None, None)

def dict_to_stock(s):
    if False:
        while True:
            i = 10
    return stock_prototype._replace(**s)

def default_stock():
    if False:
        print('Hello World!')
    a = {'name': 'ACME', 'shares': 100, 'price': 123.45}
    print(dict_to_stock(a))
    b = {'name': 'ACME', 'shares': 100, 'price': 123.45, 'date': '12/17/2012'}
    print(dict_to_stock(b))
if __name__ == '__main__':
    name_seq()
    rs = [['aa', 12, 33]]
    print(compute_cost2(rs))
    default_stock()