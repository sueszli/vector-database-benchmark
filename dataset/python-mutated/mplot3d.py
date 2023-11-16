"""Matplotlib 3D plotting example

Demonstrates plotting with matplotlib.
"""
import sys
from sample import sample
from sympy import Symbol
from sympy.external import import_module

def mplot3d(f, var1, var2, *, show=True):
    if False:
        return 10
    '\n    Plot a 3d function using matplotlib/Tk.\n    '
    import warnings
    warnings.filterwarnings('ignore', 'Could not match \\S')
    p = import_module('pylab')
    p3 = import_module('mpl_toolkits.mplot3d', import_kwargs={'fromlist': ['something']}) or import_module('matplotlib.axes3d')
    if not p or not p3:
        sys.exit('Matplotlib is required to use mplot3d.')
    (x, y, z) = sample(f, var1, var2)
    fig = p.figure()
    ax = p3.Axes3D(fig)
    ax.plot_wireframe(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if show:
        p.show()

def main():
    if False:
        i = 10
        return i + 15
    x = Symbol('x')
    y = Symbol('y')
    mplot3d(x ** 2 - y ** 2, (x, -10.0, 10.0, 20), (y, -10.0, 10.0, 20))
if __name__ == '__main__':
    main()