from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.polynomials import assoc_laguerre
from sympy.functions.special.spherical_harmonics import Ynm

def R_nl(n, l, r, Z=1):
    if False:
        print('Hello World!')
    '\n    Returns the Hydrogen radial wavefunction R_{nl}.\n\n    Parameters\n    ==========\n\n    n : integer\n        Principal Quantum Number which is\n        an integer with possible values as 1, 2, 3, 4,...\n    l : integer\n        ``l`` is the Angular Momentum Quantum Number with\n        values ranging from 0 to ``n-1``.\n    r :\n        Radial coordinate.\n    Z :\n        Atomic number (1 for Hydrogen, 2 for Helium, ...)\n\n    Everything is in Hartree atomic units.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hydrogen import R_nl\n    >>> from sympy.abc import r, Z\n    >>> R_nl(1, 0, r, Z)\n    2*sqrt(Z**3)*exp(-Z*r)\n    >>> R_nl(2, 0, r, Z)\n    sqrt(2)*(-Z*r + 2)*sqrt(Z**3)*exp(-Z*r/2)/4\n    >>> R_nl(2, 1, r, Z)\n    sqrt(6)*Z*r*sqrt(Z**3)*exp(-Z*r/2)/12\n\n    For Hydrogen atom, you can just use the default value of Z=1:\n\n    >>> R_nl(1, 0, r)\n    2*exp(-r)\n    >>> R_nl(2, 0, r)\n    sqrt(2)*(2 - r)*exp(-r/2)/4\n    >>> R_nl(3, 0, r)\n    2*sqrt(3)*(2*r**2/9 - 2*r + 3)*exp(-r/3)/27\n\n    For Silver atom, you would use Z=47:\n\n    >>> R_nl(1, 0, r, Z=47)\n    94*sqrt(47)*exp(-47*r)\n    >>> R_nl(2, 0, r, Z=47)\n    47*sqrt(94)*(2 - 47*r)*exp(-47*r/2)/4\n    >>> R_nl(3, 0, r, Z=47)\n    94*sqrt(141)*(4418*r**2/9 - 94*r + 3)*exp(-47*r/3)/27\n\n    The normalization of the radial wavefunction is:\n\n    >>> from sympy import integrate, oo\n    >>> integrate(R_nl(1, 0, r)**2 * r**2, (r, 0, oo))\n    1\n    >>> integrate(R_nl(2, 0, r)**2 * r**2, (r, 0, oo))\n    1\n    >>> integrate(R_nl(2, 1, r)**2 * r**2, (r, 0, oo))\n    1\n\n    It holds for any atomic number:\n\n    >>> integrate(R_nl(1, 0, r, Z=2)**2 * r**2, (r, 0, oo))\n    1\n    >>> integrate(R_nl(2, 0, r, Z=3)**2 * r**2, (r, 0, oo))\n    1\n    >>> integrate(R_nl(2, 1, r, Z=4)**2 * r**2, (r, 0, oo))\n    1\n\n    '
    (n, l, r, Z) = map(S, [n, l, r, Z])
    n_r = n - l - 1
    a = 1 / Z
    r0 = 2 * r / (n * a)
    C = sqrt((S(2) / (n * a)) ** 3 * factorial(n_r) / (2 * n * factorial(n + l)))
    return C * r0 ** l * assoc_laguerre(n_r, 2 * l + 1, r0).expand() * exp(-r0 / 2)

def Psi_nlm(n, l, m, r, phi, theta, Z=1):
    if False:
        return 10
    '\n    Returns the Hydrogen wave function psi_{nlm}. It\'s the product of\n    the radial wavefunction R_{nl} and the spherical harmonic Y_{l}^{m}.\n\n    Parameters\n    ==========\n\n    n : integer\n        Principal Quantum Number which is\n        an integer with possible values as 1, 2, 3, 4,...\n    l : integer\n        ``l`` is the Angular Momentum Quantum Number with\n        values ranging from 0 to ``n-1``.\n    m : integer\n        ``m`` is the Magnetic Quantum Number with values\n        ranging from ``-l`` to ``l``.\n    r :\n        radial coordinate\n    phi :\n        azimuthal angle\n    theta :\n        polar angle\n    Z :\n        atomic number (1 for Hydrogen, 2 for Helium, ...)\n\n    Everything is in Hartree atomic units.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hydrogen import Psi_nlm\n    >>> from sympy import Symbol\n    >>> r=Symbol("r", positive=True)\n    >>> phi=Symbol("phi", real=True)\n    >>> theta=Symbol("theta", real=True)\n    >>> Z=Symbol("Z", positive=True, integer=True, nonzero=True)\n    >>> Psi_nlm(1,0,0,r,phi,theta,Z)\n    Z**(3/2)*exp(-Z*r)/sqrt(pi)\n    >>> Psi_nlm(2,1,1,r,phi,theta,Z)\n    -Z**(5/2)*r*exp(I*phi)*exp(-Z*r/2)*sin(theta)/(8*sqrt(pi))\n\n    Integrating the absolute square of a hydrogen wavefunction psi_{nlm}\n    over the whole space leads 1.\n\n    The normalization of the hydrogen wavefunctions Psi_nlm is:\n\n    >>> from sympy import integrate, conjugate, pi, oo, sin\n    >>> wf=Psi_nlm(2,1,1,r,phi,theta,Z)\n    >>> abs_sqrd=wf*conjugate(wf)\n    >>> jacobi=r**2*sin(theta)\n    >>> integrate(abs_sqrd*jacobi, (r,0,oo), (phi,0,2*pi), (theta,0,pi))\n    1\n    '
    (n, l, m, r, phi, theta, Z) = map(S, [n, l, m, r, phi, theta, Z])
    if n.is_integer and n < 1:
        raise ValueError("'n' must be positive integer")
    if l.is_integer and (not n > l):
        raise ValueError("'n' must be greater than 'l'")
    if m.is_integer and (not abs(m) <= l):
        raise ValueError("|'m'| must be less or equal 'l'")
    return R_nl(n, l, r, Z) * Ynm(l, m, theta, phi).expand(func=True)

def E_nl(n, Z=1):
    if False:
        while True:
            i = 10
    '\n    Returns the energy of the state (n, l) in Hartree atomic units.\n\n    The energy does not depend on "l".\n\n    Parameters\n    ==========\n\n    n : integer\n        Principal Quantum Number which is\n        an integer with possible values as 1, 2, 3, 4,...\n    Z :\n        Atomic number (1 for Hydrogen, 2 for Helium, ...)\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hydrogen import E_nl\n    >>> from sympy.abc import n, Z\n    >>> E_nl(n, Z)\n    -Z**2/(2*n**2)\n    >>> E_nl(1)\n    -1/2\n    >>> E_nl(2)\n    -1/8\n    >>> E_nl(3)\n    -1/18\n    >>> E_nl(3, 47)\n    -2209/18\n\n    '
    (n, Z) = (S(n), S(Z))
    if n.is_integer and n < 1:
        raise ValueError("'n' must be positive integer")
    return -Z ** 2 / (2 * n ** 2)

def E_nl_dirac(n, l, spin_up=True, Z=1, c=Float('137.035999037')):
    if False:
        i = 10
        return i + 15
    '\n    Returns the relativistic energy of the state (n, l, spin) in Hartree atomic\n    units.\n\n    The energy is calculated from the Dirac equation. The rest mass energy is\n    *not* included.\n\n    Parameters\n    ==========\n\n    n : integer\n        Principal Quantum Number which is\n        an integer with possible values as 1, 2, 3, 4,...\n    l : integer\n        ``l`` is the Angular Momentum Quantum Number with\n        values ranging from 0 to ``n-1``.\n    spin_up :\n        True if the electron spin is up (default), otherwise down\n    Z :\n        Atomic number (1 for Hydrogen, 2 for Helium, ...)\n    c :\n        Speed of light in atomic units. Default value is 137.035999037,\n        taken from https://arxiv.org/abs/1012.3627\n\n    Examples\n    ========\n\n    >>> from sympy.physics.hydrogen import E_nl_dirac\n    >>> E_nl_dirac(1, 0)\n    -0.500006656595360\n\n    >>> E_nl_dirac(2, 0)\n    -0.125002080189006\n    >>> E_nl_dirac(2, 1)\n    -0.125000416028342\n    >>> E_nl_dirac(2, 1, False)\n    -0.125002080189006\n\n    >>> E_nl_dirac(3, 0)\n    -0.0555562951740285\n    >>> E_nl_dirac(3, 1)\n    -0.0555558020932949\n    >>> E_nl_dirac(3, 1, False)\n    -0.0555562951740285\n    >>> E_nl_dirac(3, 2)\n    -0.0555556377366884\n    >>> E_nl_dirac(3, 2, False)\n    -0.0555558020932949\n\n    '
    (n, l, Z, c) = map(S, [n, l, Z, c])
    if not l >= 0:
        raise ValueError("'l' must be positive or zero")
    if not n > l:
        raise ValueError("'n' must be greater than 'l'")
    if l == 0 and spin_up is False:
        raise ValueError('Spin must be up for l==0.')
    if spin_up:
        skappa = -l - 1
    else:
        skappa = -l
    beta = sqrt(skappa ** 2 - Z ** 2 / c ** 2)
    return c ** 2 / sqrt(1 + Z ** 2 / (n + skappa + beta) ** 2 / c ** 2) - c ** 2