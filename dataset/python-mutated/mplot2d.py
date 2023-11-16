"""Matplotlib 2D plotting example

Demonstrates plotting with matplotlib.
"""
import sys
from sample import sample
from sympy import sqrt, Symbol
from sympy.utilities.iterables import is_sequence
from sympy.external import import_module

def mplot2d(f, var, *, show=True):
    if False:
        return 10
    '\n    Plot a 2d function using matplotlib/Tk.\n    '
    import warnings
    warnings.filterwarnings('ignore', 'Could not match \\S')
    p = import_module('pylab')
    if not p:
        sys.exit('Matplotlib is required to use mplot2d.')
    if not is_sequence(f):
        f = [f]
    for f_i in f:
        (x, y) = sample(f_i, var)
        p.plot(x, y)
    p.draw()
    if show:
        p.show()

def main():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x')
    mplot2d([sqrt(x), -sqrt(x), sqrt(-x), -sqrt(-x)], (x, -40.0, 40.0, 80))
if __name__ == '__main__':
    main()