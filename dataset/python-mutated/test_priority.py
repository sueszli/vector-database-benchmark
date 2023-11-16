from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.functions.elementary.integers import floor

class Higher(Integer):
    """
    Integer of value 1 and _op_priority 20

    Operations handled by this class return 1 and reverse operations return 2
    """
    _op_priority = 20.0
    result = 1

    def __new__(cls):
        if False:
            print('Hello World!')
        obj = Expr.__new__(cls)
        obj.p = 1
        return obj

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        return self.result

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return 2 * self.result

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if False:
            print('Hello World!')
        return self.result

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if False:
            return 10
        return 2 * self.result

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        return self.result

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return 2 * self.result

    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if False:
            i = 10
            return i + 15
        return self.result

    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        if False:
            while True:
                i = 10
        return 2 * self.result

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        if False:
            print('Hello World!')
        return self.result

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return 2 * self.result

    @call_highest_priority('__rmod__')
    def __mod__(self, other):
        if False:
            i = 10
            return i + 15
        return self.result

    @call_highest_priority('__mod__')
    def __rmod__(self, other):
        if False:
            print('Hello World!')
        return 2 * self.result

    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other):
        if False:
            print('Hello World!')
        return self.result

    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other):
        if False:
            return 10
        return 2 * self.result

class Lower(Higher):
    """
    Integer of value -1 and _op_priority 5

    Operations handled by this class return -1 and reverse operations return -2
    """
    _op_priority = 5.0
    result = -1

    def __new__(cls):
        if False:
            print('Hello World!')
        obj = Expr.__new__(cls)
        obj.p = -1
        return obj
x = Symbol('x')
h = Higher()
l = Lower()

def test_mul():
    if False:
        for i in range(10):
            print('nop')
    assert h * l == h * x == 1
    assert l * h == x * h == 2
    assert x * l == l * x == -x

def test_add():
    if False:
        i = 10
        return i + 15
    assert h + l == h + x == 1
    assert l + h == x + h == 2
    assert x + l == l + x == x - 1

def test_sub():
    if False:
        return 10
    assert h - l == h - x == 1
    assert l - h == x - h == 2
    assert x - l == -(l - x) == x + 1

def test_pow():
    if False:
        while True:
            i = 10
    assert h ** l == h ** x == 1
    assert l ** h == x ** h == 2
    assert (x ** l).args == (1 / x).args and (x ** l).is_Pow
    assert (l ** x).args == ((-1) ** x).args and (l ** x).is_Pow

def test_div():
    if False:
        for i in range(10):
            print('nop')
    assert h / l == h / x == 1
    assert l / h == x / h == 2
    assert x / l == 1 / (l / x) == -x

def test_mod():
    if False:
        i = 10
        return i + 15
    assert h % l == h % x == 1
    assert l % h == x % h == 2
    assert x % l == Mod(x, -1)
    assert l % x == Mod(-1, x)

def test_floordiv():
    if False:
        i = 10
        return i + 15
    assert h // l == h // x == 1
    assert l // h == x // h == 2
    assert x // l == floor(-x)
    assert l // x == floor(-1 / x)