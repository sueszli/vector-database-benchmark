from sympy.core import S, pi, Rational
from sympy.functions import assoc_laguerre, sqrt, exp, factorial, factorial2

def R_nl(n, l, nu, r):
    if False:
        print('Hello World!')
    '\n    Returns the radial wavefunction R_{nl} for a 3d isotropic harmonic\n    oscillator.\n\n    Parameters\n    ==========\n\n    n :\n        The "nodal" quantum number.  Corresponds to the number of nodes in\n        the wavefunction.  ``n >= 0``\n    l :\n        The quantum number for orbital angular momentum.\n    nu :\n        mass-scaled frequency: nu = m*omega/(2*hbar) where `m` is the mass\n        and `omega` the frequency of the oscillator.\n        (in atomic units ``nu == omega/2``)\n    r :\n        Radial coordinate.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.sho import R_nl\n    >>> from sympy.abc import r, nu, l\n    >>> R_nl(0, 0, 1, r)\n    2*2**(3/4)*exp(-r**2)/pi**(1/4)\n    >>> R_nl(1, 0, 1, r)\n    4*2**(1/4)*sqrt(3)*(3/2 - 2*r**2)*exp(-r**2)/(3*pi**(1/4))\n\n    l, nu and r may be symbolic:\n\n    >>> R_nl(0, 0, nu, r)\n    2*2**(3/4)*sqrt(nu**(3/2))*exp(-nu*r**2)/pi**(1/4)\n    >>> R_nl(0, l, 1, r)\n    r**l*sqrt(2**(l + 3/2)*2**(l + 2)/factorial2(2*l + 1))*exp(-r**2)/pi**(1/4)\n\n    The normalization of the radial wavefunction is:\n\n    >>> from sympy import Integral, oo\n    >>> Integral(R_nl(0, 0, 1, r)**2*r**2, (r, 0, oo)).n()\n    1.00000000000000\n    >>> Integral(R_nl(1, 0, 1, r)**2*r**2, (r, 0, oo)).n()\n    1.00000000000000\n    >>> Integral(R_nl(1, 1, 1, r)**2*r**2, (r, 0, oo)).n()\n    1.00000000000000\n\n    '
    (n, l, nu, r) = map(S, [n, l, nu, r])
    n = n + 1
    C = sqrt((2 * nu) ** (l + Rational(3, 2)) * 2 ** (n + l + 1) * factorial(n - 1) / (sqrt(pi) * factorial2(2 * n + 2 * l - 1)))
    return C * r ** l * exp(-nu * r ** 2) * assoc_laguerre(n - 1, l + S.Half, 2 * nu * r ** 2)

def E_nl(n, l, hw):
    if False:
        print('Hello World!')
    '\n    Returns the Energy of an isotropic harmonic oscillator.\n\n    Parameters\n    ==========\n\n    n :\n        The "nodal" quantum number.\n    l :\n        The orbital angular momentum.\n    hw :\n        The harmonic oscillator parameter.\n\n    Notes\n    =====\n\n    The unit of the returned value matches the unit of hw, since the energy is\n    calculated as:\n\n        E_nl = (2*n + l + 3/2)*hw\n\n    Examples\n    ========\n\n    >>> from sympy.physics.sho import E_nl\n    >>> from sympy import symbols\n    >>> x, y, z = symbols(\'x, y, z\')\n    >>> E_nl(x, y, z)\n    z*(2*x + y + 3/2)\n    '
    return (2 * n + l + Rational(3, 2)) * hw