"""
Numerical integration with autowrap
-----------------------------------

This example demonstrates how you can use the autowrap module in SymPy
to create fast, numerical integration routines callable from python. See
in the code for detailed explanations of the various steps. An
autowrapped sympy expression can be significantly faster than what you
would get by applying a sequence of the ufuncs shipped with numpy. [0]

We will find the coefficients needed to approximate a quantum mechanical
Hydrogen wave function in terms of harmonic oscillator solutions. For
the sake of demonstration, this will be done by setting up a simple
numerical integration scheme as a SymPy expression, and obtain a binary
implementation with autowrap.

You need to have numpy installed to run this example, as well as a
working fortran compiler. If you have pylab installed, you will be
rewarded with a nice plot in the end.

[0]:
http://ojensen.wordpress.com/2010/08/10/fast-ufunc-ish-hydrogen-solutions/

----
"""
import sys
from sympy.external import import_module
np = import_module('numpy')
if not np:
    sys.exit('Cannot import numpy. Exiting.')
pylab = import_module('pylab', warn_not_installed=True)
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.autowrap import autowrap, ufuncify
from sympy import Idx, IndexedBase, Lambda, pprint, Symbol, oo, Integral, Function
from sympy.physics.sho import R_nl
from sympy.physics.hydrogen import R_nl as hydro_nl
basis_dimension = 5
omega2 = 0.1
orbital_momentum_l = 1
hydrogen_n = 2
rmax = 20
gridsize = 200

def main():
    if False:
        i = 10
        return i + 15
    print(__doc__)
    m = Symbol('m', integer=True)
    i = Idx('i', m)
    A = IndexedBase('A')
    B = IndexedBase('B')
    x = Symbol('x')
    print('Compiling ufuncs for radial harmonic oscillator solutions')
    basis_ho = {}
    for n in range(basis_dimension):
        expr = R_nl(n, orbital_momentum_l, omega2, x)
        expr = expr.evalf(15)
        print('The h.o. wave function with l = %i and n = %i is' % (orbital_momentum_l, n))
        pprint(expr)
        basis_ho[n] = ufuncify(x, expr)
    H_ufunc = ufuncify(x, hydro_nl(hydrogen_n, orbital_momentum_l, 1, x))
    binary_integrator = {}
    for n in range(basis_dimension):
        psi_ho = implemented_function('psi_ho', Lambda(x, R_nl(n, orbital_momentum_l, omega2, x)))
        psi = IndexedBase('psi')
        step = Symbol('step')
        expr = A[i] ** 2 * psi_ho(A[i]) * psi[i] * step
        if n == 0:
            print('Setting up binary integrators for the integral:')
            pprint(Integral(x ** 2 * psi_ho(x) * Function('psi')(x), (x, 0, oo)))
        binary_integrator[n] = autowrap(expr, args=[A.label, psi.label, step, m])
        print('Checking convergence of integrator for n = %i' % n)
        for g in range(3, 8):
            (grid, step) = np.linspace(0, rmax, 2 ** g, retstep=True)
            print('grid dimension %5i, integral = %e' % (2 ** g, binary_integrator[n](grid, H_ufunc(grid), step)))
    print('A binary integrator has been set up for each basis state')
    print('We will now use them to reconstruct a hydrogen solution.')
    (grid, stepsize) = np.linspace(0, rmax, gridsize, retstep=True)
    print('Calculating coefficients with gridsize = %i and stepsize %f' % (len(grid), stepsize))
    coeffs = {}
    for n in range(basis_dimension):
        coeffs[n] = binary_integrator[n](grid, H_ufunc(grid), stepsize)
        print('c(%i) = %e' % (n, coeffs[n]))
    print('Constructing the approximate hydrogen wave')
    hydro_approx = 0
    all_steps = {}
    for n in range(basis_dimension):
        hydro_approx += basis_ho[n](grid) * coeffs[n]
        all_steps[n] = hydro_approx.copy()
        if pylab:
            line = pylab.plot(grid, all_steps[n], ':', label='max n = %i' % n)
    diff = np.max(np.abs(hydro_approx - H_ufunc(grid)))
    print('Error estimate: the element with largest deviation misses by %f' % diff)
    if diff > 0.01:
        print('This is much, try to increase the basis size or adjust omega')
    else:
        print("Ah, that's a pretty good approximation!")
    if pylab:
        print("Here's a plot showing the contribution for each n")
        line[0].set_linestyle('-')
        pylab.plot(grid, H_ufunc(grid), 'r-', label='exact')
        pylab.legend()
        pylab.show()
    print('Note:\n    These binary integrators were specialized to find coefficients for a\n    harmonic oscillator basis, but they can process any wave function as long\n    as it is available as a vector and defined on a grid with equidistant\n    points. That is, on any grid you get from numpy.linspace.\n\n    To make the integrators even more flexible, you can setup the harmonic\n    oscillator solutions with symbolic parameters omega and l.  Then the\n    autowrapped binary routine will take these scalar variables as arguments,\n    so that the integrators can find coefficients for *any* isotropic harmonic\n    oscillator basis.\n\n    ')
if __name__ == '__main__':
    main()