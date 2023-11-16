"""
Setup ufuncs for the legendre polynomials
-----------------------------------------

This example demonstrates how you can use the ufuncify utility in SymPy
to create fast, customized universal functions for use with numpy
arrays. An autowrapped sympy expression can be significantly faster than
what you would get by applying a sequence of the ufuncs shipped with
numpy. [0]

You need to have numpy installed to run this example, as well as a
working fortran compiler.


[0]:
http://ojensen.wordpress.com/2010/08/10/fast-ufunc-ish-hydrogen-solutions/
"""
import sys
from sympy.external import import_module
np = import_module('numpy')
if not np:
    sys.exit('Cannot import numpy. Exiting.')
plt = import_module('matplotlib.pyplot')
if not plt:
    sys.exit('Cannot import matplotlib.pyplot. Exiting.')
import mpmath
from sympy.utilities.autowrap import ufuncify
from sympy import symbols, legendre, pprint

def main():
    if False:
        while True:
            i = 10
    print(__doc__)
    x = symbols('x')
    grid = np.linspace(-1, 1, 1000)
    mpmath.mp.dps = 20
    print('Compiling legendre ufuncs and checking results:')
    for n in range(6):
        expr = legendre(n, x)
        print('The polynomial of degree %i is' % n)
        pprint(expr)
        binary_poly = ufuncify(x, expr)
        polyvector = binary_poly(grid)
        maxdiff = 0
        for j in range(len(grid)):
            precise_val = mpmath.legendre(n, grid[j])
            diff = abs(polyvector[j] - precise_val)
            if diff > maxdiff:
                maxdiff = diff
        print('The largest error in applied ufunc was %e' % maxdiff)
        assert maxdiff < 1e-14
        plot1 = plt.pyplot.plot(grid, polyvector, hold=True)
    print("Here's a plot with values calculated by the wrapped binary functions")
    plt.pyplot.show()
if __name__ == '__main__':
    main()