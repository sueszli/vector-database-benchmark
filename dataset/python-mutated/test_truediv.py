from sympy.core.numbers import Float, Rational
from sympy.core.symbol import Symbol

def test_truediv():
    if False:
        print('Hello World!')
    assert 1 / 2 != 0
    assert Rational(1) / 2 != 0

def dotest(s):
    if False:
        print('Hello World!')
    x = Symbol('x')
    y = Symbol('y')
    l = [Rational(2), Float('1.3'), x, y, pow(x, y) * y, 5, 5.5]
    for x in l:
        for y in l:
            s(x, y)
    return True

def test_basic():
    if False:
        print('Hello World!')

    def s(a, b):
        if False:
            print('Hello World!')
        x = a
        x = +a
        x = -a
        x = a + b
        x = a - b
        x = a * b
        x = a / b
        x = a ** b
        del x
    assert dotest(s)

def test_ibasic():
    if False:
        while True:
            i = 10

    def s(a, b):
        if False:
            for i in range(10):
                print('nop')
        x = a
        x += b
        x = a
        x -= b
        x = a
        x *= b
        x = a
        x /= b
    assert dotest(s)