"""
Topic: 四舍五入运算
Desc : 
"""

def round_num():
    if False:
        while True:
            i = 10
    print(round(1.23, 1))
    print(round(1.27, 1))
    print(round(-1.27, 1))
    print(round(1.25361, 3))
    a = 1627731
    print(round(a, -1))
    print(round(a, -2))
    print(round(a, -3))
    x = 1.23456
    print(format(x, '0.2f'))
    print(format(x, '0.3f'))
    print('value is {:0.3f}'.format(x))
    a = 2.1
    b = 4.2
    c = a + b
    print(c)
    c = round(c, 2)
    print(c)
if __name__ == '__main__':
    round_num()