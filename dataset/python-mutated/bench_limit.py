from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.series.limits import limit
x = Symbol('x')

def timeit_limit_1x():
    if False:
        while True:
            i = 10
    limit(1 / x, x, oo)