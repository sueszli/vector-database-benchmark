"""
Wigner, Clebsch-Gordan, Racah, and Gaunt coefficients

Collection of functions for calculating Wigner 3j, 6j, 9j,
Clebsch-Gordan, Racah as well as Gaunt coefficients exactly, all
evaluating to a rational number times the square root of a rational
number [Rasch03]_.

Please see the description of the individual functions for further
details and examples.

References
==========

.. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
  T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)
.. [Regge59] 'Symmetry Properties of Racah Coefficients',
  T. Regge, Nuovo Cimento, Volume 11, pp. 116 (1959)
.. [Edmonds74] A. R. Edmonds. Angular momentum in quantum mechanics.
  Investigations in physics, 4.; Investigations in physics, no. 4.
  Princeton, N.J., Princeton University Press, 1957.
.. [Rasch03] J. Rasch and A. C. H. Yu, 'Efficient Storage Scheme for
  Pre-calculated Wigner 3j, 6j and Gaunt Coefficients', SIAM
  J. Sci. Comput. Volume 25, Issue 4, pp. 1416-1428 (2003)
.. [Liberatodebrito82] 'FORTRAN program for the integral of three
  spherical harmonics', A. Liberato de Brito,
  Comput. Phys. Commun., Volume 25, pp. 81-85 (1982)
.. [Homeier96] 'Some Properties of the Coupling Coefficients of Real
  Spherical Harmonics and Their Relation to Gaunt Coefficients',
  H. H. H. Homeier and E. O. Steinborn J. Mol. Struct., Volume 368,
  pp. 31-37 (1996)

Credits and Copyright
=====================

This code was taken from Sage with the permission of all authors:

https://groups.google.com/forum/#!topic/sage-devel/M4NZdu-7O38

Authors
=======

- Jens Rasch (2009-03-24): initial version for Sage

- Jens Rasch (2009-05-31): updated to sage-4.0

- Oscar Gerardo Lazo Arjona (2017-06-18): added Wigner D matrices

- Phil Adam LeMaitre (2022-09-19): added real Gaunt coefficient

Copyright (C) 2008 Jens Rasch <jyr2000@gmail.com>

"""
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import int_valued
from sympy.core.function import Function
from sympy.core.numbers import I, Integer, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int
_Factlist = [1]

def _calc_factlist(nn):
    if False:
        return 10
    '\n    Function calculates a list of precomputed factorials in order to\n    massively accelerate future calculations of the various\n    coefficients.\n\n    Parameters\n    ==========\n\n    nn : integer\n        Highest factorial to be computed.\n\n    Returns\n    =======\n\n    list of integers :\n        The list of precomputed factorials.\n\n    Examples\n    ========\n\n    Calculate list of factorials::\n\n        sage: from sage.functions.wigner import _calc_factlist\n        sage: _calc_factlist(10)\n        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]\n    '
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]

def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    if False:
        print('Hello World!')
    '\n    Calculate the Wigner 3j symbol `\\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.\n\n    Parameters\n    ==========\n\n    j_1, j_2, j_3, m_1, m_2, m_3 :\n        Integer or half integer.\n\n    Returns\n    =======\n\n    Rational number times the square root of a rational number.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.wigner import wigner_3j\n    >>> wigner_3j(2, 6, 4, 0, 0, 0)\n    sqrt(715)/143\n    >>> wigner_3j(2, 6, 4, 0, 0, 1)\n    0\n\n    It is an error to have arguments that are not integer or half\n    integer values::\n\n        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)\n        Traceback (most recent call last):\n        ...\n        ValueError: j values must be integer or half integer\n        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)\n        Traceback (most recent call last):\n        ...\n        ValueError: m values must be integer or half integer\n\n    Notes\n    =====\n\n    The Wigner 3j symbol obeys the following symmetry rules:\n\n    - invariant under any permutation of the columns (with the\n      exception of a sign change where `J:=j_1+j_2+j_3`):\n\n      .. math::\n\n         \\begin{aligned}\n         \\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)\n          &=\\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\\\\n          &=\\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\\\\n          &=(-1)^J \\operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\\\\n          &=(-1)^J \\operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\\\\n          &=(-1)^J \\operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)\n         \\end{aligned}\n\n    - invariant under space inflection, i.e.\n\n      .. math::\n\n         \\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)\n         =(-1)^J \\operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)\n\n    - symmetric with respect to the 72 additional symmetries based on\n      the work by [Regge58]_\n\n    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation\n\n    - zero for `m_1 + m_2 + m_3 \\neq 0`\n\n    - zero for violating any one of the conditions\n      `j_1 \\ge |m_1|`,  `j_2 \\ge |m_2|`,  `j_3 \\ge |m_3|`\n\n    Algorithm\n    =========\n\n    This function uses the algorithm of [Edmonds74]_ to calculate the\n    value of the 3j symbol exactly. Note that the formula contains\n    alternating sums over large factorials and is therefore unsuitable\n    for finite precision arithmetic and only useful for a computer\n    algebra system [Rasch03]_.\n\n    Authors\n    =======\n\n    - Jens Rasch (2009-03-24): initial version\n    '
    if int(j_1 * 2) != j_1 * 2 or int(j_2 * 2) != j_2 * 2 or int(j_3 * 2) != j_3 * 2:
        raise ValueError('j values must be integer or half integer')
    if int(m_1 * 2) != m_1 * 2 or int(m_2 * 2) != m_2 * 2 or int(m_3 * 2) != m_3 * 2:
        raise ValueError('m values must be integer or half integer')
    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return S.Zero
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return S.Zero
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return S.Zero
    if abs(m_1) > j_1 or abs(m_2) > j_2 or abs(m_3) > j_3:
        return S.Zero
    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2), j_3 + abs(m_3))
    _calc_factlist(int(maxfact))
    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] * _Factlist[int(j_1 - j_2 + j_3)] * _Factlist[int(-j_1 + j_2 + j_3)] * _Factlist[int(j_1 - m_1)] * _Factlist[int(j_1 + m_1)] * _Factlist[int(j_2 - m_2)] * _Factlist[int(j_2 + m_2)] * _Factlist[int(j_3 - m_3)] * _Factlist[int(j_3 + m_3)]) / _Factlist[int(j_1 + j_2 + j_3 + 1)]
    ressqrt = sqrt(argsqrt)
    if ressqrt.is_complex or ressqrt.is_infinite:
        ressqrt = ressqrt.as_real_imag()[0]
    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[int(ii + j_3 - j_1 - m_2)] * _Factlist[int(j_2 + m_2 - ii)] * _Factlist[int(j_1 - ii - m_1)] * _Factlist[int(ii + j_3 - j_2 + m_1)] * _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den
    res = ressqrt * sumres * prefid
    return res

def clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3):
    if False:
        while True:
            i = 10
    '\n    Calculates the Clebsch-Gordan coefficient.\n    `\\left\\langle j_1 m_1 \\; j_2 m_2 | j_3 m_3 \\right\\rangle`.\n\n    The reference for this function is [Edmonds74]_.\n\n    Parameters\n    ==========\n\n    j_1, j_2, j_3, m_1, m_2, m_3 :\n        Integer or half integer.\n\n    Returns\n    =======\n\n    Rational number times the square root of a rational number.\n\n    Examples\n    ========\n\n    >>> from sympy import S\n    >>> from sympy.physics.wigner import clebsch_gordan\n    >>> clebsch_gordan(S(3)/2, S(1)/2, 2, S(3)/2, S(1)/2, 2)\n    1\n    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, S(3)/2, -S(1)/2, 1)\n    sqrt(3)/2\n    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, -S(1)/2, S(1)/2, 0)\n    -sqrt(2)/2\n\n    Notes\n    =====\n\n    The Clebsch-Gordan coefficient will be evaluated via its relation\n    to Wigner 3j symbols:\n\n    .. math::\n\n        \\left\\langle j_1 m_1 \\; j_2 m_2 | j_3 m_3 \\right\\rangle\n        =(-1)^{j_1-j_2+m_3} \\sqrt{2j_3+1}\n        \\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,-m_3)\n\n    See also the documentation on Wigner 3j symbols which exhibit much\n    higher symmetry relations than the Clebsch-Gordan coefficient.\n\n    Authors\n    =======\n\n    - Jens Rasch (2009-03-24): initial version\n    '
    res = (-1) ** sympify(j_1 - j_2 + m_3) * sqrt(2 * j_3 + 1) * wigner_3j(j_1, j_2, j_3, m_1, m_2, -m_3)
    return res

def _big_delta_coeff(aa, bb, cc, prec=None):
    if False:
        while True:
            i = 10
    '\n    Calculates the Delta coefficient of the 3 angular momenta for\n    Racah symbols. Also checks that the differences are of integer\n    value.\n\n    Parameters\n    ==========\n\n    aa :\n        First angular momentum, integer or half integer.\n    bb :\n        Second angular momentum, integer or half integer.\n    cc :\n        Third angular momentum, integer or half integer.\n    prec :\n        Precision of the ``sqrt()`` calculation.\n\n    Returns\n    =======\n\n    double : Value of the Delta coefficient.\n\n    Examples\n    ========\n\n        sage: from sage.functions.wigner import _big_delta_coeff\n        sage: _big_delta_coeff(1,1,1)\n        1/2*sqrt(1/6)\n    '
    if not int_valued(aa + bb - cc):
        raise ValueError('j values must be integer or half integer and fulfill the triangle relation')
    if not int_valued(aa + cc - bb):
        raise ValueError('j values must be integer or half integer and fulfill the triangle relation')
    if not int_valued(bb + cc - aa):
        raise ValueError('j values must be integer or half integer and fulfill the triangle relation')
    if aa + bb - cc < 0:
        return S.Zero
    if aa + cc - bb < 0:
        return S.Zero
    if bb + cc - aa < 0:
        return S.Zero
    maxfact = max(aa + bb - cc, aa + cc - bb, bb + cc - aa, aa + bb + cc + 1)
    _calc_factlist(maxfact)
    argsqrt = Integer(_Factlist[int(aa + bb - cc)] * _Factlist[int(aa + cc - bb)] * _Factlist[int(bb + cc - aa)]) / Integer(_Factlist[int(aa + bb + cc + 1)])
    ressqrt = sqrt(argsqrt)
    if prec:
        ressqrt = ressqrt.evalf(prec).as_real_imag()[0]
    return ressqrt

def racah(aa, bb, cc, dd, ee, ff, prec=None):
    if False:
        i = 10
        return i + 15
    '\n    Calculate the Racah symbol `W(a,b,c,d;e,f)`.\n\n    Parameters\n    ==========\n\n    a, ..., f :\n        Integer or half integer.\n    prec :\n        Precision, default: ``None``. Providing a precision can\n        drastically speed up the calculation.\n\n    Returns\n    =======\n\n    Rational number times the square root of a rational number\n    (if ``prec=None``), or real number if a precision is given.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.wigner import racah\n    >>> racah(3,3,3,3,3,3)\n    -1/14\n\n    Notes\n    =====\n\n    The Racah symbol is related to the Wigner 6j symbol:\n\n    .. math::\n\n       \\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)\n       =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)\n\n    Please see the 6j symbol for its much richer symmetries and for\n    additional properties.\n\n    Algorithm\n    =========\n\n    This function uses the algorithm of [Edmonds74]_ to calculate the\n    value of the 6j symbol exactly. Note that the formula contains\n    alternating sums over large factorials and is therefore unsuitable\n    for finite precision arithmetic and only useful for a computer\n    algebra system [Rasch03]_.\n\n    Authors\n    =======\n\n    - Jens Rasch (2009-03-24): initial version\n    '
    prefac = _big_delta_coeff(aa, bb, ee, prec) * _big_delta_coeff(cc, dd, ee, prec) * _big_delta_coeff(aa, cc, ff, prec) * _big_delta_coeff(bb, dd, ff, prec)
    if prefac == 0:
        return S.Zero
    imin = max(aa + bb + ee, cc + dd + ee, aa + cc + ff, bb + dd + ff)
    imax = min(aa + bb + cc + dd, aa + dd + ee + ff, bb + cc + ee + ff)
    maxfact = max(imax + 1, aa + bb + cc + dd, aa + dd + ee + ff, bb + cc + ee + ff)
    _calc_factlist(maxfact)
    sumres = 0
    for kk in range(int(imin), int(imax) + 1):
        den = _Factlist[int(kk - aa - bb - ee)] * _Factlist[int(kk - cc - dd - ee)] * _Factlist[int(kk - aa - cc - ff)] * _Factlist[int(kk - bb - dd - ff)] * _Factlist[int(aa + bb + cc + dd - kk)] * _Factlist[int(aa + dd + ee + ff - kk)] * _Factlist[int(bb + cc + ee + ff - kk)]
        sumres = sumres + Integer((-1) ** kk * _Factlist[kk + 1]) / den
    res = prefac * sumres * (-1) ** int(aa + bb + cc + dd)
    return res

def wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6, prec=None):
    if False:
        print('Hello World!')
    "\n    Calculate the Wigner 6j symbol `\\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)`.\n\n    Parameters\n    ==========\n\n    j_1, ..., j_6 :\n        Integer or half integer.\n    prec :\n        Precision, default: ``None``. Providing a precision can\n        drastically speed up the calculation.\n\n    Returns\n    =======\n\n    Rational number times the square root of a rational number\n    (if ``prec=None``), or real number if a precision is given.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.wigner import wigner_6j\n    >>> wigner_6j(3,3,3,3,3,3)\n    -1/14\n    >>> wigner_6j(5,5,5,5,5,5)\n    1/52\n\n    It is an error to have arguments that are not integer or half\n    integer values or do not fulfill the triangle relation::\n\n        sage: wigner_6j(2.5,2.5,2.5,2.5,2.5,2.5)\n        Traceback (most recent call last):\n        ...\n        ValueError: j values must be integer or half integer and fulfill the triangle relation\n        sage: wigner_6j(0.5,0.5,1.1,0.5,0.5,1.1)\n        Traceback (most recent call last):\n        ...\n        ValueError: j values must be integer or half integer and fulfill the triangle relation\n\n    Notes\n    =====\n\n    The Wigner 6j symbol is related to the Racah symbol but exhibits\n    more symmetries as detailed below.\n\n    .. math::\n\n       \\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)\n        =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)\n\n    The Wigner 6j symbol obeys the following symmetry rules:\n\n    - Wigner 6j symbols are left invariant under any permutation of\n      the columns:\n\n      .. math::\n\n         \\begin{aligned}\n         \\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)\n          &=\\operatorname{Wigner6j}(j_3,j_1,j_2,j_6,j_4,j_5) \\\\\n          &=\\operatorname{Wigner6j}(j_2,j_3,j_1,j_5,j_6,j_4) \\\\\n          &=\\operatorname{Wigner6j}(j_3,j_2,j_1,j_6,j_5,j_4) \\\\\n          &=\\operatorname{Wigner6j}(j_1,j_3,j_2,j_4,j_6,j_5) \\\\\n          &=\\operatorname{Wigner6j}(j_2,j_1,j_3,j_5,j_4,j_6)\n         \\end{aligned}\n\n    - They are invariant under the exchange of the upper and lower\n      arguments in each of any two columns, i.e.\n\n      .. math::\n\n         \\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)\n          =\\operatorname{Wigner6j}(j_1,j_5,j_6,j_4,j_2,j_3)\n          =\\operatorname{Wigner6j}(j_4,j_2,j_6,j_1,j_5,j_3)\n          =\\operatorname{Wigner6j}(j_4,j_5,j_3,j_1,j_2,j_6)\n\n    - additional 6 symmetries [Regge59]_ giving rise to 144 symmetries\n      in total\n\n    - only non-zero if any triple of `j`'s fulfill a triangle relation\n\n    Algorithm\n    =========\n\n    This function uses the algorithm of [Edmonds74]_ to calculate the\n    value of the 6j symbol exactly. Note that the formula contains\n    alternating sums over large factorials and is therefore unsuitable\n    for finite precision arithmetic and only useful for a computer\n    algebra system [Rasch03]_.\n\n    "
    res = (-1) ** int(j_1 + j_2 + j_4 + j_5) * racah(j_1, j_2, j_5, j_4, j_3, j_6, prec)
    return res

def wigner_9j(j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9, prec=None):
    if False:
        while True:
            i = 10
    '\n    Calculate the Wigner 9j symbol\n    `\\operatorname{Wigner9j}(j_1,j_2,j_3,j_4,j_5,j_6,j_7,j_8,j_9)`.\n\n    Parameters\n    ==========\n\n    j_1, ..., j_9 :\n        Integer or half integer.\n    prec : precision, default\n        ``None``. Providing a precision can\n        drastically speed up the calculation.\n\n    Returns\n    =======\n\n    Rational number times the square root of a rational number\n    (if ``prec=None``), or real number if a precision is given.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.wigner import wigner_9j\n    >>> wigner_9j(1,1,1, 1,1,1, 1,1,0, prec=64) # ==1/18\n    0.05555555...\n\n    >>> wigner_9j(1/2,1/2,0, 1/2,3/2,1, 0,1,1, prec=64) # ==1/6\n    0.1666666...\n\n    It is an error to have arguments that are not integer or half\n    integer values or do not fulfill the triangle relation::\n\n        sage: wigner_9j(0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5,prec=64)\n        Traceback (most recent call last):\n        ...\n        ValueError: j values must be integer or half integer and fulfill the triangle relation\n        sage: wigner_9j(1,1,1, 0.5,1,1.5, 0.5,1,2.5,prec=64)\n        Traceback (most recent call last):\n        ...\n        ValueError: j values must be integer or half integer and fulfill the triangle relation\n\n    Algorithm\n    =========\n\n    This function uses the algorithm of [Edmonds74]_ to calculate the\n    value of the 3j symbol exactly. Note that the formula contains\n    alternating sums over large factorials and is therefore unsuitable\n    for finite precision arithmetic and only useful for a computer\n    algebra system [Rasch03]_.\n    '
    imax = int(min(j_1 + j_9, j_2 + j_6, j_4 + j_8) * 2)
    imin = imax % 2
    sumres = 0
    for kk in range(imin, int(imax) + 1, 2):
        sumres = sumres + (kk + 1) * racah(j_1, j_2, j_9, j_6, j_3, kk / 2, prec) * racah(j_4, j_6, j_8, j_2, j_5, kk / 2, prec) * racah(j_1, j_4, j_9, j_8, j_7, kk / 2, prec)
    return sumres

def gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the Gaunt coefficient.\n\n    Explanation\n    ===========\n\n    The Gaunt coefficient is defined as the integral over three\n    spherical harmonics:\n\n    .. math::\n\n        \\begin{aligned}\n        \\operatorname{Gaunt}(l_1,l_2,l_3,m_1,m_2,m_3)\n        &=\\int Y_{l_1,m_1}(\\Omega)\n         Y_{l_2,m_2}(\\Omega) Y_{l_3,m_3}(\\Omega) \\,d\\Omega \\\\\n        &=\\sqrt{\\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\\pi}}\n         \\operatorname{Wigner3j}(l_1,l_2,l_3,0,0,0)\n         \\operatorname{Wigner3j}(l_1,l_2,l_3,m_1,m_2,m_3)\n        \\end{aligned}\n\n    Parameters\n    ==========\n\n    l_1, l_2, l_3, m_1, m_2, m_3 :\n        Integer.\n    prec - precision, default: ``None``.\n        Providing a precision can\n        drastically speed up the calculation.\n\n    Returns\n    =======\n\n    Rational number times the square root of a rational number\n    (if ``prec=None``), or real number if a precision is given.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.wigner import gaunt\n    >>> gaunt(1,0,1,1,0,-1)\n    -1/(2*sqrt(pi))\n    >>> gaunt(1000,1000,1200,9,3,-12).n(64)\n    0.00689500421922113448...\n\n    It is an error to use non-integer values for `l` and `m`::\n\n        sage: gaunt(1.2,0,1.2,0,0,0)\n        Traceback (most recent call last):\n        ...\n        ValueError: l values must be integer\n        sage: gaunt(1,0,1,1.1,0,-1.1)\n        Traceback (most recent call last):\n        ...\n        ValueError: m values must be integer\n\n    Notes\n    =====\n\n    The Gaunt coefficient obeys the following symmetry rules:\n\n    - invariant under any permutation of the columns\n\n      .. math::\n        \\begin{aligned}\n          Y(l_1,l_2,l_3,m_1,m_2,m_3)\n          &=Y(l_3,l_1,l_2,m_3,m_1,m_2) \\\\\n          &=Y(l_2,l_3,l_1,m_2,m_3,m_1) \\\\\n          &=Y(l_3,l_2,l_1,m_3,m_2,m_1) \\\\\n          &=Y(l_1,l_3,l_2,m_1,m_3,m_2) \\\\\n          &=Y(l_2,l_1,l_3,m_2,m_1,m_3)\n        \\end{aligned}\n\n    - invariant under space inflection, i.e.\n\n      .. math::\n          Y(l_1,l_2,l_3,m_1,m_2,m_3)\n          =Y(l_1,l_2,l_3,-m_1,-m_2,-m_3)\n\n    - symmetric with respect to the 72 Regge symmetries as inherited\n      for the `3j` symbols [Regge58]_\n\n    - zero for `l_1`, `l_2`, `l_3` not fulfilling triangle relation\n\n    - zero for violating any one of the conditions: `l_1 \\ge |m_1|`,\n      `l_2 \\ge |m_2|`, `l_3 \\ge |m_3|`\n\n    - non-zero only for an even sum of the `l_i`, i.e.\n      `L = l_1 + l_2 + l_3 = 2n` for `n` in `\\mathbb{N}`\n\n    Algorithms\n    ==========\n\n    This function uses the algorithm of [Liberatodebrito82]_ to\n    calculate the value of the Gaunt coefficient exactly. Note that\n    the formula contains alternating sums over large factorials and is\n    therefore unsuitable for finite precision arithmetic and only\n    useful for a computer algebra system [Rasch03]_.\n\n    Authors\n    =======\n\n    Jens Rasch (2009-03-24): initial version for Sage.\n    '
    (l_1, l_2, l_3, m_1, m_2, m_3) = [as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]
    if l_1 + l_2 - l_3 < 0:
        return S.Zero
    if l_1 - l_2 + l_3 < 0:
        return S.Zero
    if -l_1 + l_2 + l_3 < 0:
        return S.Zero
    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    if abs(m_1) > l_1 or abs(m_2) > l_2 or abs(m_3) > l_3:
        return S.Zero
    (bigL, remL) = divmod(l_1 + l_2 + l_3, 2)
    if remL % 2:
        return S.Zero
    imin = max(-l_3 + l_1 + m_2, -l_3 + l_2 - m_1, 0)
    imax = min(l_2 + m_2, l_1 - m_1, l_1 + l_2 - l_3)
    _calc_factlist(max(l_1 + l_2 + l_3 + 1, imax + 1))
    ressqrt = sqrt((2 * l_1 + 1) * (2 * l_2 + 1) * (2 * l_3 + 1) * _Factlist[l_1 - m_1] * _Factlist[l_1 + m_1] * _Factlist[l_2 - m_2] * _Factlist[l_2 + m_2] * _Factlist[l_3 - m_3] * _Factlist[l_3 + m_3] / (4 * pi))
    prefac = Integer(_Factlist[bigL] * _Factlist[l_2 - l_1 + l_3] * _Factlist[l_1 - l_2 + l_3] * _Factlist[l_1 + l_2 - l_3]) / _Factlist[2 * bigL + 1] / (_Factlist[bigL - l_1] * _Factlist[bigL - l_2] * _Factlist[bigL - l_3])
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[ii + l_3 - l_1 - m_2] * _Factlist[l_2 + m_2 - ii] * _Factlist[l_1 - ii - m_1] * _Factlist[ii + l_3 - l_2 + m_1] * _Factlist[l_1 + l_2 - l_3 - ii]
        sumres = sumres + Integer((-1) ** ii) / den
    res = ressqrt * prefac * sumres * Integer((-1) ** (bigL + l_3 + m_1 - m_2))
    if prec is not None:
        res = res.n(prec)
    return res

def real_gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    if False:
        print('Hello World!')
    "\n    Calculate the real Gaunt coefficient.\n\n    Explanation\n    ===========\n\n    The real Gaunt coefficient is defined as the integral over three\n    real spherical harmonics:\n\n    .. math::\n        \\begin{aligned}\n        \\operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)\n        &=\\int Z^{m_1}_{l_1}(\\Omega)\n         Z^{m_2}_{l_2}(\\Omega) Z^{m_3}_{l_3}(\\Omega) \\,d\\Omega \\\\\n        \\end{aligned}\n\n    Alternatively, it can be defined in terms of the standard Gaunt\n    coefficient by relating the real spherical harmonics to the standard\n    spherical harmonics via a unitary transformation `U`, i.e.\n    `Z^{m}_{l}(\\Omega)=\\sum_{m'}U^{m}_{m'}Y^{m'}_{l}(\\Omega)` [Homeier96]_.\n    The real Gaunt coefficient is then defined as\n\n    .. math::\n        \\begin{aligned}\n        \\operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)\n        &=\\int Z^{m_1}_{l_1}(\\Omega)\n         Z^{m_2}_{l_2}(\\Omega) Z^{m_3}_{l_3}(\\Omega) \\,d\\Omega \\\\\n        &=\\sum_{m'_1 m'_2 m'_3} U^{m_1}_{m'_1}U^{m_2}_{m'_2}U^{m_3}_{m'_3}\n         \\operatorname{Gaunt}(l_1,l_2,l_3,m'_1,m'_2,m'_3)\n        \\end{aligned}\n\n    The unitary matrix `U` has components\n\n    .. math::\n        \\begin{aligned}\n        U^m_{m'} = \\delta_{|m||m'|}*(\\delta_{m'0}\\delta_{m0} + \\frac{1}{\\sqrt{2}}\\big[\\Theta(m)\n        \\big(\\delta_{m'm}+(-1)^{m'}\\delta_{m'-m}\\big)+i\\Theta(-m)\\big((-1)^{-m}\n        \\delta_{m'-m}-\\delta_{m'm}*(-1)^{m'-m}\\big)\\big])\n        \\end{aligned}\n\n    where `\\delta_{ij}` is the Kronecker delta symbol and `\\Theta` is a step\n    function defined as\n\n    .. math::\n        \\begin{aligned}\n        \\Theta(x) = \\begin{cases} 1 \\,\\text{for}\\, x > 0 \\\\ 0 \\,\\text{for}\\, x \\leq 0 \\end{cases}\n        \\end{aligned}\n\n    Parameters\n    ==========\n\n    l_1, l_2, l_3, m_1, m_2, m_3 :\n        Integer.\n\n    prec - precision, default: ``None``.\n        Providing a precision can\n        drastically speed up the calculation.\n\n    Returns\n    =======\n\n    Rational number times the square root of a rational number.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.wigner import real_gaunt\n    >>> real_gaunt(2,2,4,-1,-1,0)\n    -2/(7*sqrt(pi))\n    >>> real_gaunt(10,10,20,-9,-9,0).n(64)\n    -0.00002480019791932209313156167...\n\n    It is an error to use non-integer values for `l` and `m`::\n        real_gaunt(2.8,0.5,1.3,0,0,0)\n        Traceback (most recent call last):\n        ...\n        ValueError: l values must be integer\n        real_gaunt(2,2,4,0.7,1,-3.4)\n        Traceback (most recent call last):\n        ...\n        ValueError: m values must be integer\n\n    Notes\n    =====\n\n    The real Gaunt coefficient inherits from the standard Gaunt coefficient,\n    the invariance under any permutation of the pairs `(l_i, m_i)` and the\n    requirement that the sum of the `l_i` be even to yield a non-zero value.\n    It also obeys the following symmetry rules:\n\n    - zero for `l_1`, `l_2`, `l_3` not fulfiling the condition\n      `l_1 \\in \\{l_{\\text{max}}, l_{\\text{max}}-2, \\ldots, l_{\\text{min}}\\}`,\n      where `l_{\\text{max}} = l_2+l_3`,\n\n      .. math::\n          \\begin{aligned}\n          l_{\\text{min}} = \\begin{cases} \\kappa(l_2, l_3, m_2, m_3) & \\text{if}\\,\n          \\kappa(l_2, l_3, m_2, m_3) + l_{\\text{max}}\\, \\text{is even} \\\\\n          \\kappa(l_2, l_3, m_2, m_3)+1 & \\text{if}\\, \\kappa(l_2, l_3, m_2, m_3) +\n          l_{\\text{max}}\\, \\text{is odd}\\end{cases}\n          \\end{aligned}\n\n      and `\\kappa(l_2, l_3, m_2, m_3) = \\max{\\big(|l_2-l_3|, \\min{\\big(|m_2+m_3|,\n      |m_2-m_3|\\big)}\\big)}`\n\n    - zero for an odd number of negative `m_i`\n\n    Algorithms\n    ==========\n\n    This function uses the algorithms of [Homeier96]_ and [Rasch03]_ to\n    calculate the value of the real Gaunt coefficient exactly. Note that\n    the formula used in [Rasch03]_ contains alternating sums over large\n    factorials and is therefore unsuitable for finite precision arithmetic\n    and only useful for a computer algebra system [Rasch03]_. However, this\n    function can in principle use any algorithm that computes the Gaunt\n    coefficient, so it is suitable for finite precision arithmetic in so far\n    as the algorithm which computes the Gaunt coefficient is.\n    "
    (l_1, l_2, l_3, m_1, m_2, m_3) = [as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]
    if sum((1 for i in (m_1, m_2, m_3) if i < 0)) % 2:
        return S.Zero
    if (l_1 + l_2 + l_3) % 2:
        return S.Zero
    lmax = l_2 + l_3
    lmin = max(abs(l_2 - l_3), min(abs(m_2 + m_3), abs(m_2 - m_3)))
    if (lmin + lmax) % 2:
        lmin += 1
    if lmin not in range(lmax, lmin - 2, -2):
        return S.Zero
    kron_del = lambda i, j: 1 if i == j else 0
    s = lambda e: -1 if e % 2 else 1
    A = lambda a, b: -kron_del(a, b) * s(a - b) + kron_del(a, -b) * s(b) if b < 0 else 0
    B = lambda a, b: kron_del(a, b) + kron_del(a, -b) * s(a) if b > 0 else 0
    C = lambda a, b: kron_del(abs(a), abs(b)) * (kron_del(a, 0) * kron_del(b, 0) + (B(a, b) + I * A(a, b)) / sqrt(2))
    ugnt = 0
    for i in range(-l_1, l_1 + 1):
        U1 = C(i, m_1)
        for j in range(-l_2, l_2 + 1):
            U2 = C(j, m_2)
            U3 = C(-i - j, m_3)
            ugnt = ugnt + re(U1 * U2 * U3) * gaunt(l_1, l_2, l_3, i, j, -i - j)
    if prec is not None:
        ugnt = ugnt.n(prec)
    return ugnt

class Wigner3j(Function):

    def doit(self, **hints):
        if False:
            return 10
        if all((obj.is_number for obj in self.args)):
            return wigner_3j(*self.args)
        else:
            return self

def dot_rot_grad_Ynm(j, p, l, m, theta, phi):
    if False:
        print('Hello World!')
    '\n    Returns dot product of rotational gradients of spherical harmonics.\n\n    Explanation\n    ===========\n\n    This function returns the right hand side of the following expression:\n\n    .. math ::\n        \\vec{R}Y{_j^{p}} \\cdot \\vec{R}Y{_l^{m}} = (-1)^{m+p}\n        \\sum\\limits_{k=|l-j|}^{l+j}Y{_k^{m+p}}  * \\alpha_{l,m,j,p,k} *\n        \\frac{1}{2} (k^2-j^2-l^2+k-j-l)\n\n\n    Arguments\n    =========\n\n    j, p, l, m .... indices in spherical harmonics (expressions or integers)\n    theta, phi .... angle arguments in spherical harmonics\n\n    Example\n    =======\n\n    >>> from sympy import symbols\n    >>> from sympy.physics.wigner import dot_rot_grad_Ynm\n    >>> theta, phi = symbols("theta phi")\n    >>> dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit()\n    3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))\n\n    '
    j = sympify(j)
    p = sympify(p)
    l = sympify(l)
    m = sympify(m)
    theta = sympify(theta)
    phi = sympify(phi)
    k = Dummy('k')

    def alpha(l, m, j, p, k):
        if False:
            return 10
        return sqrt((2 * l + 1) * (2 * j + 1) * (2 * k + 1) / (4 * pi)) * Wigner3j(j, l, k, S.Zero, S.Zero, S.Zero) * Wigner3j(j, l, k, p, m, -m - p)
    return S.NegativeOne ** (m + p) * Sum(Ynm(k, m + p, theta, phi) * alpha(l, m, j, p, k) / 2 * (k ** 2 - j ** 2 - l ** 2 + k - j - l), (k, abs(l - j), l + j))

def wigner_d_small(J, beta):
    if False:
        for i in range(10):
            print('nop')
    'Return the small Wigner d matrix for angular momentum J.\n\n    Explanation\n    ===========\n\n    J : An integer, half-integer, or SymPy symbol for the total angular\n        momentum of the angular momentum space being rotated.\n    beta : A real number representing the Euler angle of rotation about\n        the so-called line of nodes. See [Edmonds74]_.\n\n    Returns\n    =======\n\n    A matrix representing the corresponding Euler angle rotation( in the basis\n    of eigenvectors of `J_z`).\n\n    .. math ::\n        \\mathcal{d}_{\\beta} = \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)\n\n    The components are calculated using the general form [Edmonds74]_,\n    equation 4.1.15.\n\n    Examples\n    ========\n\n    >>> from sympy import Integer, symbols, pi, pprint\n    >>> from sympy.physics.wigner import wigner_d_small\n    >>> half = 1/Integer(2)\n    >>> beta = symbols("beta", real=True)\n    >>> pprint(wigner_d_small(half, beta), use_unicode=True)\n    ⎡   ⎛β⎞      ⎛β⎞⎤\n    ⎢cos⎜─⎟   sin⎜─⎟⎥\n    ⎢   ⎝2⎠      ⎝2⎠⎥\n    ⎢               ⎥\n    ⎢    ⎛β⎞     ⎛β⎞⎥\n    ⎢-sin⎜─⎟  cos⎜─⎟⎥\n    ⎣    ⎝2⎠     ⎝2⎠⎦\n\n    >>> pprint(wigner_d_small(2*half, beta), use_unicode=True)\n    ⎡        2⎛β⎞              ⎛β⎞    ⎛β⎞           2⎛β⎞     ⎤\n    ⎢     cos ⎜─⎟        √2⋅sin⎜─⎟⋅cos⎜─⎟        sin ⎜─⎟     ⎥\n    ⎢         ⎝2⎠              ⎝2⎠    ⎝2⎠            ⎝2⎠     ⎥\n    ⎢                                                        ⎥\n    ⎢       ⎛β⎞    ⎛β⎞       2⎛β⎞      2⎛β⎞        ⎛β⎞    ⎛β⎞⎥\n    ⎢-√2⋅sin⎜─⎟⋅cos⎜─⎟  - sin ⎜─⎟ + cos ⎜─⎟  √2⋅sin⎜─⎟⋅cos⎜─⎟⎥\n    ⎢       ⎝2⎠    ⎝2⎠        ⎝2⎠       ⎝2⎠        ⎝2⎠    ⎝2⎠⎥\n    ⎢                                                        ⎥\n    ⎢        2⎛β⎞               ⎛β⎞    ⎛β⎞          2⎛β⎞     ⎥\n    ⎢     sin ⎜─⎟        -√2⋅sin⎜─⎟⋅cos⎜─⎟       cos ⎜─⎟     ⎥\n    ⎣         ⎝2⎠               ⎝2⎠    ⎝2⎠           ⎝2⎠     ⎦\n\n    From table 4 in [Edmonds74]_\n\n    >>> pprint(wigner_d_small(half, beta).subs({beta:pi/2}), use_unicode=True)\n    ⎡ √2   √2⎤\n    ⎢ ──   ──⎥\n    ⎢ 2    2 ⎥\n    ⎢        ⎥\n    ⎢-√2   √2⎥\n    ⎢────  ──⎥\n    ⎣ 2    2 ⎦\n\n    >>> pprint(wigner_d_small(2*half, beta).subs({beta:pi/2}),\n    ... use_unicode=True)\n    ⎡       √2      ⎤\n    ⎢1/2    ──   1/2⎥\n    ⎢       2       ⎥\n    ⎢               ⎥\n    ⎢-√2         √2 ⎥\n    ⎢────   0    ── ⎥\n    ⎢ 2          2  ⎥\n    ⎢               ⎥\n    ⎢      -√2      ⎥\n    ⎢1/2   ────  1/2⎥\n    ⎣       2       ⎦\n\n    >>> pprint(wigner_d_small(3*half, beta).subs({beta:pi/2}),\n    ... use_unicode=True)\n    ⎡ √2    √6    √6   √2⎤\n    ⎢ ──    ──    ──   ──⎥\n    ⎢ 4     4     4    4 ⎥\n    ⎢                    ⎥\n    ⎢-√6   -√2    √2   √6⎥\n    ⎢────  ────   ──   ──⎥\n    ⎢ 4     4     4    4 ⎥\n    ⎢                    ⎥\n    ⎢ √6   -√2   -√2   √6⎥\n    ⎢ ──   ────  ────  ──⎥\n    ⎢ 4     4     4    4 ⎥\n    ⎢                    ⎥\n    ⎢-√2    √6   -√6   √2⎥\n    ⎢────   ──   ────  ──⎥\n    ⎣ 4     4     4    4 ⎦\n\n    >>> pprint(wigner_d_small(4*half, beta).subs({beta:pi/2}),\n    ... use_unicode=True)\n    ⎡             √6            ⎤\n    ⎢1/4   1/2    ──   1/2   1/4⎥\n    ⎢             4             ⎥\n    ⎢                           ⎥\n    ⎢-1/2  -1/2   0    1/2   1/2⎥\n    ⎢                           ⎥\n    ⎢ √6                     √6 ⎥\n    ⎢ ──    0    -1/2   0    ── ⎥\n    ⎢ 4                      4  ⎥\n    ⎢                           ⎥\n    ⎢-1/2  1/2    0    -1/2  1/2⎥\n    ⎢                           ⎥\n    ⎢             √6            ⎥\n    ⎢1/4   -1/2   ──   -1/2  1/4⎥\n    ⎣             4             ⎦\n\n    '
    M = [J - i for i in range(2 * J + 1)]
    d = zeros(2 * J + 1)
    for (i, Mi) in enumerate(M):
        for (j, Mj) in enumerate(M):
            sigmamax = min([J - Mi, J - Mj])
            sigmamin = max([0, -Mi - Mj])
            dij = sqrt(factorial(J + Mi) * factorial(J - Mi) / factorial(J + Mj) / factorial(J - Mj))
            terms = [(-1) ** (J - Mi - s) * binomial(J + Mj, J - Mi - s) * binomial(J - Mj, s) * cos(beta / 2) ** (2 * s + Mi + Mj) * sin(beta / 2) ** (2 * J - 2 * s - Mj - Mi) for s in range(sigmamin, sigmamax + 1)]
            d[i, j] = dij * Add(*terms)
    return ImmutableMatrix(d)

def wigner_d(J, alpha, beta, gamma):
    if False:
        i = 10
        return i + 15
    'Return the Wigner D matrix for angular momentum J.\n\n    Explanation\n    ===========\n\n    J :\n        An integer, half-integer, or SymPy symbol for the total angular\n        momentum of the angular momentum space being rotated.\n    alpha, beta, gamma - Real numbers representing the Euler.\n        Angles of rotation about the so-called vertical, line of nodes, and\n        figure axes. See [Edmonds74]_.\n\n    Returns\n    =======\n\n    A matrix representing the corresponding Euler angle rotation( in the basis\n    of eigenvectors of `J_z`).\n\n    .. math ::\n        \\mathcal{D}_{\\alpha \\beta \\gamma} =\n        \\exp\\big( \\frac{i\\alpha}{\\hbar} J_z\\big)\n        \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)\n        \\exp\\big( \\frac{i\\gamma}{\\hbar} J_z\\big)\n\n    The components are calculated using the general form [Edmonds74]_,\n    equation 4.1.12.\n\n    Examples\n    ========\n\n    The simplest possible example:\n\n    >>> from sympy.physics.wigner import wigner_d\n    >>> from sympy import Integer, symbols, pprint\n    >>> half = 1/Integer(2)\n    >>> alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)\n    >>> pprint(wigner_d(half, alpha, beta, gamma), use_unicode=True)\n    ⎡  ⅈ⋅α  ⅈ⋅γ             ⅈ⋅α  -ⅈ⋅γ         ⎤\n    ⎢  ───  ───             ───  ─────        ⎥\n    ⎢   2    2     ⎛β⎞       2     2      ⎛β⎞ ⎥\n    ⎢ ℯ   ⋅ℯ   ⋅cos⎜─⎟     ℯ   ⋅ℯ     ⋅sin⎜─⎟ ⎥\n    ⎢              ⎝2⎠                    ⎝2⎠ ⎥\n    ⎢                                         ⎥\n    ⎢  -ⅈ⋅α   ⅈ⋅γ          -ⅈ⋅α   -ⅈ⋅γ        ⎥\n    ⎢  ─────  ───          ─────  ─────       ⎥\n    ⎢    2     2     ⎛β⎞     2      2      ⎛β⎞⎥\n    ⎢-ℯ     ⋅ℯ   ⋅sin⎜─⎟  ℯ     ⋅ℯ     ⋅cos⎜─⎟⎥\n    ⎣                ⎝2⎠                   ⎝2⎠⎦\n\n    '
    d = wigner_d_small(J, beta)
    M = [J - i for i in range(2 * J + 1)]
    D = [[exp(I * Mi * alpha) * d[i, j] * exp(I * Mj * gamma) for (j, Mj) in enumerate(M)] for (i, Mi) in enumerate(M)]
    return ImmutableMatrix(D)