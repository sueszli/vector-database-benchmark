from sympy.core import S, pi, Rational
from sympy.functions import hermite, sqrt, exp, factorial, Abs
from sympy.physics.quantum.constants import hbar

def psi_n(n, x, m, omega):
    if False:
        while True:
            i = 10
    '\n    Returns the wavefunction psi_{n} for the One-dimensional harmonic oscillator.\n\n    Parameters\n    ==========\n\n    n :\n        the "nodal" quantum number.  Corresponds to the number of nodes in the\n        wavefunction.  ``n >= 0``\n    x :\n        x coordinate.\n    m :\n        Mass of the particle.\n    omega :\n        Angular frequency of the oscillator.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.qho_1d import psi_n\n    >>> from sympy.abc import m, x, omega\n    >>> psi_n(0, x, m, omega)\n    (m*omega)**(1/4)*exp(-m*omega*x**2/(2*hbar))/(hbar**(1/4)*pi**(1/4))\n\n    '
    (n, x, m, omega) = map(S, [n, x, m, omega])
    nu = m * omega / hbar
    C = (nu / pi) ** Rational(1, 4) * sqrt(1 / (2 ** n * factorial(n)))
    return C * exp(-nu * x ** 2 / 2) * hermite(n, sqrt(nu) * x)

def E_n(n, omega):
    if False:
        return 10
    '\n    Returns the Energy of the One-dimensional harmonic oscillator.\n\n    Parameters\n    ==========\n\n    n :\n        The "nodal" quantum number.\n    omega :\n        The harmonic oscillator angular frequency.\n\n    Notes\n    =====\n\n    The unit of the returned value matches the unit of hw, since the energy is\n    calculated as:\n\n        E_n = hbar * omega*(n + 1/2)\n\n    Examples\n    ========\n\n    >>> from sympy.physics.qho_1d import E_n\n    >>> from sympy.abc import x, omega\n    >>> E_n(x, omega)\n    hbar*omega*(x + 1/2)\n    '
    return hbar * omega * (n + S.Half)

def coherent_state(n, alpha):
    if False:
        print('Hello World!')
    '\n    Returns <n|alpha> for the coherent states of 1D harmonic oscillator.\n    See https://en.wikipedia.org/wiki/Coherent_states\n\n    Parameters\n    ==========\n\n    n :\n        The "nodal" quantum number.\n    alpha :\n        The eigen value of annihilation operator.\n    '
    return exp(-Abs(alpha) ** 2 / 2) * alpha ** n / sqrt(factorial(n))