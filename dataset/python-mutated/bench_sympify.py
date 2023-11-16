from sympy.core import sympify, Symbol
x = Symbol('x')

def timeit_sympify_1():
    if False:
        i = 10
        return i + 15
    sympify(1)

def timeit_sympify_x():
    if False:
        while True:
            i = 10
    sympify(x)