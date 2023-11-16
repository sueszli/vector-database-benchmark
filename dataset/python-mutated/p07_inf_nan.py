"""
Topic: 无穷大与NaN
Desc : 
"""

def inf_nan():
    if False:
        i = 10
        return i + 15
    a = float('inf')
    b = float('-inf')
    c = float('nan')
    print(a + 45)
    print(a + 45 == a)
    print(a * 10 == a)
    print(10 / a)
    print(a / a)
    print(a + b)
    print(c + 23)
    print(c / 2 == c)
if __name__ == '__main__':
    inf_nan()