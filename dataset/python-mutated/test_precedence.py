from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative
from sympy.core.numbers import Integer, Rational, Float, oo
from sympy.core.relational import Rel
from sympy.core.symbol import symbols
from sympy.functions import sin
from sympy.integrals.integrals import Integral
from sympy.series.order import Order
from sympy.printing.precedence import precedence, PRECEDENCE
(x, y) = symbols('x,y')

def test_Add():
    if False:
        for i in range(10):
            print('nop')
    assert precedence(x + y) == PRECEDENCE['Add']
    assert precedence(x * y + 1) == PRECEDENCE['Add']

def test_Function():
    if False:
        for i in range(10):
            print('nop')
    assert precedence(sin(x)) == PRECEDENCE['Func']

def test_Derivative():
    if False:
        for i in range(10):
            print('nop')
    assert precedence(Derivative(x, y)) == PRECEDENCE['Atom']

def test_Integral():
    if False:
        for i in range(10):
            print('nop')
    assert precedence(Integral(x, y)) == PRECEDENCE['Atom']

def test_Mul():
    if False:
        while True:
            i = 10
    assert precedence(x * y) == PRECEDENCE['Mul']
    assert precedence(-x * y) == PRECEDENCE['Add']

def test_Number():
    if False:
        for i in range(10):
            print('nop')
    assert precedence(Integer(0)) == PRECEDENCE['Atom']
    assert precedence(Integer(1)) == PRECEDENCE['Atom']
    assert precedence(Integer(-1)) == PRECEDENCE['Add']
    assert precedence(Integer(10)) == PRECEDENCE['Atom']
    assert precedence(Rational(5, 2)) == PRECEDENCE['Mul']
    assert precedence(Rational(-5, 2)) == PRECEDENCE['Add']
    assert precedence(Float(5)) == PRECEDENCE['Atom']
    assert precedence(Float(-5)) == PRECEDENCE['Add']
    assert precedence(oo) == PRECEDENCE['Atom']
    assert precedence(-oo) == PRECEDENCE['Add']

def test_Order():
    if False:
        i = 10
        return i + 15
    assert precedence(Order(x)) == PRECEDENCE['Atom']

def test_Pow():
    if False:
        return 10
    assert precedence(x ** y) == PRECEDENCE['Pow']
    assert precedence(-x ** y) == PRECEDENCE['Add']
    assert precedence(x ** (-y)) == PRECEDENCE['Pow']

def test_Product():
    if False:
        print('Hello World!')
    assert precedence(Product(x, (x, y, y + 1))) == PRECEDENCE['Atom']

def test_Relational():
    if False:
        i = 10
        return i + 15
    assert precedence(Rel(x + y, y, '<')) == PRECEDENCE['Relational']

def test_Sum():
    if False:
        return 10
    assert precedence(Sum(x, (x, y, y + 1))) == PRECEDENCE['Atom']

def test_Symbol():
    if False:
        for i in range(10):
            print('nop')
    assert precedence(x) == PRECEDENCE['Atom']

def test_And_Or():
    if False:
        i = 10
        return i + 15
    assert precedence(x & y) > precedence(x | y)
    assert precedence(~y) > precedence(x & y)
    assert precedence(x + y) > precedence(x | y)
    assert precedence(x + y) > precedence(x & y)
    assert precedence(x * y) > precedence(x | y)
    assert precedence(x * y) > precedence(x & y)
    assert precedence(~y) > precedence(x * y)
    assert precedence(~y) > precedence(x - y)
    assert precedence(x & y) == PRECEDENCE['And']
    assert precedence(x | y) == PRECEDENCE['Or']
    assert precedence(~y) == PRECEDENCE['Not']