from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar

def wavefunction(n, x):
    if False:
        return 10
    '\n    Returns the wavefunction for particle on ring.\n\n    Parameters\n    ==========\n\n    n : The quantum number.\n        Here ``n`` can be positive as well as negative\n        which can be used to describe the direction of motion of particle.\n    x :\n        The angle.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.pring import wavefunction\n    >>> from sympy import Symbol, integrate, pi\n    >>> x=Symbol("x")\n    >>> wavefunction(1, x)\n    sqrt(2)*exp(I*x)/(2*sqrt(pi))\n    >>> wavefunction(2, x)\n    sqrt(2)*exp(2*I*x)/(2*sqrt(pi))\n    >>> wavefunction(3, x)\n    sqrt(2)*exp(3*I*x)/(2*sqrt(pi))\n\n    The normalization of the wavefunction is:\n\n    >>> integrate(wavefunction(2, x)*wavefunction(-2, x), (x, 0, 2*pi))\n    1\n    >>> integrate(wavefunction(4, x)*wavefunction(-4, x), (x, 0, 2*pi))\n    1\n\n    References\n    ==========\n\n    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum\n           Mechanics (4th ed.).  Pages 71-73.\n\n    '
    (n, x) = (S(n), S(x))
    return exp(n * I * x) / sqrt(2 * pi)

def energy(n, m, r):
    if False:
        return 10
    '\n    Returns the energy of the state corresponding to quantum number ``n``.\n\n    E=(n**2 * (hcross)**2) / (2 * m * r**2)\n\n    Parameters\n    ==========\n\n    n :\n        The quantum number.\n    m :\n        Mass of the particle.\n    r :\n        Radius of circle.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.pring import energy\n    >>> from sympy import Symbol\n    >>> m=Symbol("m")\n    >>> r=Symbol("r")\n    >>> energy(1, m, r)\n    hbar**2/(2*m*r**2)\n    >>> energy(2, m, r)\n    2*hbar**2/(m*r**2)\n    >>> energy(-2, 2.0, 3.0)\n    0.111111111111111*hbar**2\n\n    References\n    ==========\n\n    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum\n           Mechanics (4th ed.).  Pages 71-73.\n\n    '
    (n, m, r) = (S(n), S(m), S(r))
    if n.is_integer:
        return n ** 2 * hbar ** 2 / (2 * m * r ** 2)
    else:
        raise ValueError("'n' must be integer")