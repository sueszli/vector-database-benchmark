"""
The module implements routines to model the polarization of optical fields
and can be used to calculate the effects of polarization optical elements on
the fields.

- Jones vectors.

- Stokes vectors.

- Jones matrices.

- Mueller matrices.

Examples
========

We calculate a generic Jones vector:

>>> from sympy import symbols, pprint, zeros, simplify
>>> from sympy.physics.optics.polarization import (jones_vector, stokes_vector,
...     half_wave_retarder, polarizing_beam_splitter, jones_2_stokes)

>>> psi, chi, p, I0 = symbols("psi, chi, p, I0", real=True)
>>> x0 = jones_vector(psi, chi)
>>> pprint(x0, use_unicode=True)
⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
⎢                                ⎥
⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

And the more general Stokes vector:

>>> s0 = stokes_vector(psi, chi, p, I0)
>>> pprint(s0, use_unicode=True)
⎡          I₀          ⎤
⎢                      ⎥
⎢I₀⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
⎢                      ⎥
⎢I₀⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
⎢                      ⎥
⎣    I₀⋅p⋅sin(2⋅χ)     ⎦

We calculate how the Jones vector is modified by a half-wave plate:

>>> alpha = symbols("alpha", real=True)
>>> HWP = half_wave_retarder(alpha)
>>> x1 = simplify(HWP*x0)

We calculate the very common operation of passing a beam through a half-wave
plate and then through a polarizing beam-splitter. We do this by putting this
Jones vector as the first entry of a two-Jones-vector state that is transformed
by a 4x4 Jones matrix modelling the polarizing beam-splitter to get the
transmitted and reflected Jones vectors:

>>> PBS = polarizing_beam_splitter()
>>> X1 = zeros(4, 1)
>>> X1[:2, :] = x1
>>> X2 = PBS*X1
>>> transmitted_port = X2[:2, :]
>>> reflected_port = X2[2:, :]

This allows us to calculate how the power in both ports depends on the initial
polarization:

>>> transmitted_power = jones_2_stokes(transmitted_port)[0]
>>> reflected_power = jones_2_stokes(reflected_port)[0]
>>> print(transmitted_power)
cos(-2*alpha + chi + psi)**2/2 + cos(2*alpha + chi - psi)**2/2


>>> print(reflected_power)
sin(-2*alpha + chi + psi)**2/2 + sin(2*alpha + chi - psi)**2/2

Please see the description of the individual functions for further
details and examples.

References
==========

.. [1] https://en.wikipedia.org/wiki/Jones_calculus
.. [2] https://en.wikipedia.org/wiki/Mueller_calculus
.. [3] https://en.wikipedia.org/wiki/Stokes_parameters

"""
from sympy.core.numbers import I, pi
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct

def jones_vector(psi, chi):
    if False:
        while True:
            i = 10
    'A Jones vector corresponding to a polarization ellipse with `psi` tilt,\n    and `chi` circularity.\n\n    Parameters\n    ==========\n\n    psi : numeric type or SymPy Symbol\n        The tilt of the polarization relative to the `x` axis.\n\n    chi : numeric type or SymPy Symbol\n        The angle adjacent to the mayor axis of the polarization ellipse.\n\n\n    Returns\n    =======\n\n    Matrix :\n        A Jones vector.\n\n    Examples\n    ========\n\n    The axes on the Poincaré sphere.\n\n    >>> from sympy import pprint, symbols, pi\n    >>> from sympy.physics.optics.polarization import jones_vector\n    >>> psi, chi = symbols("psi, chi", real=True)\n\n    A general Jones vector.\n\n    >>> pprint(jones_vector(psi, chi), use_unicode=True)\n    ⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤\n    ⎢                                ⎥\n    ⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦\n\n    Horizontal polarization.\n\n    >>> pprint(jones_vector(0, 0), use_unicode=True)\n    ⎡1⎤\n    ⎢ ⎥\n    ⎣0⎦\n\n    Vertical polarization.\n\n    >>> pprint(jones_vector(pi/2, 0), use_unicode=True)\n    ⎡0⎤\n    ⎢ ⎥\n    ⎣1⎦\n\n    Diagonal polarization.\n\n    >>> pprint(jones_vector(pi/4, 0), use_unicode=True)\n    ⎡√2⎤\n    ⎢──⎥\n    ⎢2 ⎥\n    ⎢  ⎥\n    ⎢√2⎥\n    ⎢──⎥\n    ⎣2 ⎦\n\n    Anti-diagonal polarization.\n\n    >>> pprint(jones_vector(-pi/4, 0), use_unicode=True)\n    ⎡ √2 ⎤\n    ⎢ ── ⎥\n    ⎢ 2  ⎥\n    ⎢    ⎥\n    ⎢-√2 ⎥\n    ⎢────⎥\n    ⎣ 2  ⎦\n\n    Right-hand circular polarization.\n\n    >>> pprint(jones_vector(0, pi/4), use_unicode=True)\n    ⎡ √2 ⎤\n    ⎢ ── ⎥\n    ⎢ 2  ⎥\n    ⎢    ⎥\n    ⎢√2⋅ⅈ⎥\n    ⎢────⎥\n    ⎣ 2  ⎦\n\n    Left-hand circular polarization.\n\n    >>> pprint(jones_vector(0, -pi/4), use_unicode=True)\n    ⎡  √2  ⎤\n    ⎢  ──  ⎥\n    ⎢  2   ⎥\n    ⎢      ⎥\n    ⎢-√2⋅ⅈ ⎥\n    ⎢──────⎥\n    ⎣  2   ⎦\n\n    '
    return Matrix([-I * sin(chi) * sin(psi) + cos(chi) * cos(psi), I * sin(chi) * cos(psi) + sin(psi) * cos(chi)])

def stokes_vector(psi, chi, p=1, I=1):
    if False:
        print('Hello World!')
    'A Stokes vector corresponding to a polarization ellipse with ``psi``\n    tilt, and ``chi`` circularity.\n\n    Parameters\n    ==========\n\n    psi : numeric type or SymPy Symbol\n        The tilt of the polarization relative to the ``x`` axis.\n    chi : numeric type or SymPy Symbol\n        The angle adjacent to the mayor axis of the polarization ellipse.\n    p : numeric type or SymPy Symbol\n        The degree of polarization.\n    I : numeric type or SymPy Symbol\n        The intensity of the field.\n\n\n    Returns\n    =======\n\n    Matrix :\n        A Stokes vector.\n\n    Examples\n    ========\n\n    The axes on the Poincaré sphere.\n\n    >>> from sympy import pprint, symbols, pi\n    >>> from sympy.physics.optics.polarization import stokes_vector\n    >>> psi, chi, p, I = symbols("psi, chi, p, I", real=True)\n    >>> pprint(stokes_vector(psi, chi, p, I), use_unicode=True)\n    ⎡          I          ⎤\n    ⎢                     ⎥\n    ⎢I⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥\n    ⎢                     ⎥\n    ⎢I⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥\n    ⎢                     ⎥\n    ⎣    I⋅p⋅sin(2⋅χ)     ⎦\n\n\n    Horizontal polarization\n\n    >>> pprint(stokes_vector(0, 0), use_unicode=True)\n    ⎡1⎤\n    ⎢ ⎥\n    ⎢1⎥\n    ⎢ ⎥\n    ⎢0⎥\n    ⎢ ⎥\n    ⎣0⎦\n\n    Vertical polarization\n\n    >>> pprint(stokes_vector(pi/2, 0), use_unicode=True)\n    ⎡1 ⎤\n    ⎢  ⎥\n    ⎢-1⎥\n    ⎢  ⎥\n    ⎢0 ⎥\n    ⎢  ⎥\n    ⎣0 ⎦\n\n    Diagonal polarization\n\n    >>> pprint(stokes_vector(pi/4, 0), use_unicode=True)\n    ⎡1⎤\n    ⎢ ⎥\n    ⎢0⎥\n    ⎢ ⎥\n    ⎢1⎥\n    ⎢ ⎥\n    ⎣0⎦\n\n    Anti-diagonal polarization\n\n    >>> pprint(stokes_vector(-pi/4, 0), use_unicode=True)\n    ⎡1 ⎤\n    ⎢  ⎥\n    ⎢0 ⎥\n    ⎢  ⎥\n    ⎢-1⎥\n    ⎢  ⎥\n    ⎣0 ⎦\n\n    Right-hand circular polarization\n\n    >>> pprint(stokes_vector(0, pi/4), use_unicode=True)\n    ⎡1⎤\n    ⎢ ⎥\n    ⎢0⎥\n    ⎢ ⎥\n    ⎢0⎥\n    ⎢ ⎥\n    ⎣1⎦\n\n    Left-hand circular polarization\n\n    >>> pprint(stokes_vector(0, -pi/4), use_unicode=True)\n    ⎡1 ⎤\n    ⎢  ⎥\n    ⎢0 ⎥\n    ⎢  ⎥\n    ⎢0 ⎥\n    ⎢  ⎥\n    ⎣-1⎦\n\n    Unpolarized light\n\n    >>> pprint(stokes_vector(0, 0, 0), use_unicode=True)\n    ⎡1⎤\n    ⎢ ⎥\n    ⎢0⎥\n    ⎢ ⎥\n    ⎢0⎥\n    ⎢ ⎥\n    ⎣0⎦\n\n    '
    S0 = I
    S1 = I * p * cos(2 * psi) * cos(2 * chi)
    S2 = I * p * sin(2 * psi) * cos(2 * chi)
    S3 = I * p * sin(2 * chi)
    return Matrix([S0, S1, S2, S3])

def jones_2_stokes(e):
    if False:
        return 10
    'Return the Stokes vector for a Jones vector ``e``.\n\n    Parameters\n    ==========\n\n    e : SymPy Matrix\n        A Jones vector.\n\n    Returns\n    =======\n\n    SymPy Matrix\n        A Jones vector.\n\n    Examples\n    ========\n\n    The axes on the Poincaré sphere.\n\n    >>> from sympy import pprint, pi\n    >>> from sympy.physics.optics.polarization import jones_vector\n    >>> from sympy.physics.optics.polarization import jones_2_stokes\n    >>> H = jones_vector(0, 0)\n    >>> V = jones_vector(pi/2, 0)\n    >>> D = jones_vector(pi/4, 0)\n    >>> A = jones_vector(-pi/4, 0)\n    >>> R = jones_vector(0, pi/4)\n    >>> L = jones_vector(0, -pi/4)\n    >>> pprint([jones_2_stokes(e) for e in [H, V, D, A, R, L]],\n    ...         use_unicode=True)\n    ⎡⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤⎤\n    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥\n    ⎢⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥  ⎢0⎥  ⎢0 ⎥⎥\n    ⎢⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥⎥\n    ⎢⎢0⎥  ⎢0 ⎥  ⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥⎥\n    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥\n    ⎣⎣0⎦  ⎣0 ⎦  ⎣0⎦  ⎣0 ⎦  ⎣1⎦  ⎣-1⎦⎦\n\n    '
    (ex, ey) = e
    return Matrix([Abs(ex) ** 2 + Abs(ey) ** 2, Abs(ex) ** 2 - Abs(ey) ** 2, 2 * re(ex * ey.conjugate()), -2 * im(ex * ey.conjugate())])

def linear_polarizer(theta=0):
    if False:
        return 10
    'A linear polarizer Jones matrix with transmission axis at\n    an angle ``theta``.\n\n    Parameters\n    ==========\n\n    theta : numeric type or SymPy Symbol\n        The angle of the transmission axis relative to the horizontal plane.\n\n    Returns\n    =======\n\n    SymPy Matrix\n        A Jones matrix representing the polarizer.\n\n    Examples\n    ========\n\n    A generic polarizer.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import linear_polarizer\n    >>> theta = symbols("theta", real=True)\n    >>> J = linear_polarizer(theta)\n    >>> pprint(J, use_unicode=True)\n    ⎡      2                     ⎤\n    ⎢   cos (θ)     sin(θ)⋅cos(θ)⎥\n    ⎢                            ⎥\n    ⎢                     2      ⎥\n    ⎣sin(θ)⋅cos(θ)     sin (θ)   ⎦\n\n\n    '
    M = Matrix([[cos(theta) ** 2, sin(theta) * cos(theta)], [sin(theta) * cos(theta), sin(theta) ** 2]])
    return M

def phase_retarder(theta=0, delta=0):
    if False:
        return 10
    'A phase retarder Jones matrix with retardance ``delta`` at angle ``theta``.\n\n    Parameters\n    ==========\n\n    theta : numeric type or SymPy Symbol\n        The angle of the fast axis relative to the horizontal plane.\n    delta : numeric type or SymPy Symbol\n        The phase difference between the fast and slow axes of the\n        transmitted light.\n\n    Returns\n    =======\n\n    SymPy Matrix :\n        A Jones matrix representing the retarder.\n\n    Examples\n    ========\n\n    A generic retarder.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import phase_retarder\n    >>> theta, delta = symbols("theta, delta", real=True)\n    >>> R = phase_retarder(theta, delta)\n    >>> pprint(R, use_unicode=True)\n    ⎡                          -ⅈ⋅δ               -ⅈ⋅δ               ⎤\n    ⎢                          ─────              ─────              ⎥\n    ⎢⎛ ⅈ⋅δ    2         2   ⎞    2    ⎛     ⅈ⋅δ⎞    2                ⎥\n    ⎢⎝ℯ   ⋅sin (θ) + cos (θ)⎠⋅ℯ       ⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥\n    ⎢                                                                ⎥\n    ⎢            -ⅈ⋅δ                                           -ⅈ⋅δ ⎥\n    ⎢            ─────                                          ─────⎥\n    ⎢⎛     ⅈ⋅δ⎞    2                  ⎛ ⅈ⋅δ    2         2   ⎞    2  ⎥\n    ⎣⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝ℯ   ⋅cos (θ) + sin (θ)⎠⋅ℯ     ⎦\n\n    '
    R = Matrix([[cos(theta) ** 2 + exp(I * delta) * sin(theta) ** 2, (1 - exp(I * delta)) * cos(theta) * sin(theta)], [(1 - exp(I * delta)) * cos(theta) * sin(theta), sin(theta) ** 2 + exp(I * delta) * cos(theta) ** 2]])
    return R * exp(-I * delta / 2)

def half_wave_retarder(theta):
    if False:
        print('Hello World!')
    'A half-wave retarder Jones matrix at angle ``theta``.\n\n    Parameters\n    ==========\n\n    theta : numeric type or SymPy Symbol\n        The angle of the fast axis relative to the horizontal plane.\n\n    Returns\n    =======\n\n    SymPy Matrix\n        A Jones matrix representing the retarder.\n\n    Examples\n    ========\n\n    A generic half-wave plate.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import half_wave_retarder\n    >>> theta= symbols("theta", real=True)\n    >>> HWP = half_wave_retarder(theta)\n    >>> pprint(HWP, use_unicode=True)\n    ⎡   ⎛     2         2   ⎞                        ⎤\n    ⎢-ⅈ⋅⎝- sin (θ) + cos (θ)⎠    -2⋅ⅈ⋅sin(θ)⋅cos(θ)  ⎥\n    ⎢                                                ⎥\n    ⎢                             ⎛   2         2   ⎞⎥\n    ⎣   -2⋅ⅈ⋅sin(θ)⋅cos(θ)     -ⅈ⋅⎝sin (θ) - cos (θ)⎠⎦\n\n    '
    return phase_retarder(theta, pi)

def quarter_wave_retarder(theta):
    if False:
        return 10
    'A quarter-wave retarder Jones matrix at angle ``theta``.\n\n    Parameters\n    ==========\n\n    theta : numeric type or SymPy Symbol\n        The angle of the fast axis relative to the horizontal plane.\n\n    Returns\n    =======\n\n    SymPy Matrix\n        A Jones matrix representing the retarder.\n\n    Examples\n    ========\n\n    A generic quarter-wave plate.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import quarter_wave_retarder\n    >>> theta= symbols("theta", real=True)\n    >>> QWP = quarter_wave_retarder(theta)\n    >>> pprint(QWP, use_unicode=True)\n    ⎡                       -ⅈ⋅π            -ⅈ⋅π               ⎤\n    ⎢                       ─────           ─────              ⎥\n    ⎢⎛     2         2   ⎞    4               4                ⎥\n    ⎢⎝ⅈ⋅sin (θ) + cos (θ)⎠⋅ℯ       (1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥\n    ⎢                                                          ⎥\n    ⎢         -ⅈ⋅π                                        -ⅈ⋅π ⎥\n    ⎢         ─────                                       ─────⎥\n    ⎢           4                  ⎛   2           2   ⎞    4  ⎥\n    ⎣(1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝sin (θ) + ⅈ⋅cos (θ)⎠⋅ℯ     ⎦\n\n    '
    return phase_retarder(theta, pi / 2)

def transmissive_filter(T):
    if False:
        return 10
    'An attenuator Jones matrix with transmittance ``T``.\n\n    Parameters\n    ==========\n\n    T : numeric type or SymPy Symbol\n        The transmittance of the attenuator.\n\n    Returns\n    =======\n\n    SymPy Matrix\n        A Jones matrix representing the filter.\n\n    Examples\n    ========\n\n    A generic filter.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import transmissive_filter\n    >>> T = symbols("T", real=True)\n    >>> NDF = transmissive_filter(T)\n    >>> pprint(NDF, use_unicode=True)\n    ⎡√T  0 ⎤\n    ⎢      ⎥\n    ⎣0   √T⎦\n\n    '
    return Matrix([[sqrt(T), 0], [0, sqrt(T)]])

def reflective_filter(R):
    if False:
        for i in range(10):
            print('nop')
    'A reflective filter Jones matrix with reflectance ``R``.\n\n    Parameters\n    ==========\n\n    R : numeric type or SymPy Symbol\n        The reflectance of the filter.\n\n    Returns\n    =======\n\n    SymPy Matrix\n        A Jones matrix representing the filter.\n\n    Examples\n    ========\n\n    A generic filter.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import reflective_filter\n    >>> R = symbols("R", real=True)\n    >>> pprint(reflective_filter(R), use_unicode=True)\n    ⎡√R   0 ⎤\n    ⎢       ⎥\n    ⎣0   -√R⎦\n\n    '
    return Matrix([[sqrt(R), 0], [0, -sqrt(R)]])

def mueller_matrix(J):
    if False:
        for i in range(10):
            print('nop')
    'The Mueller matrix corresponding to Jones matrix `J`.\n\n    Parameters\n    ==========\n\n    J : SymPy Matrix\n        A Jones matrix.\n\n    Returns\n    =======\n\n    SymPy Matrix\n        The corresponding Mueller matrix.\n\n    Examples\n    ========\n\n    Generic optical components.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import (mueller_matrix,\n    ...     linear_polarizer, half_wave_retarder, quarter_wave_retarder)\n    >>> theta = symbols("theta", real=True)\n\n    A linear_polarizer\n\n    >>> pprint(mueller_matrix(linear_polarizer(theta)), use_unicode=True)\n    ⎡            cos(2⋅θ)      sin(2⋅θ)     ⎤\n    ⎢  1/2       ────────      ────────    0⎥\n    ⎢               2             2         ⎥\n    ⎢                                       ⎥\n    ⎢cos(2⋅θ)  cos(4⋅θ)   1    sin(4⋅θ)     ⎥\n    ⎢────────  ──────── + ─    ────────    0⎥\n    ⎢   2         4       4       4         ⎥\n    ⎢                                       ⎥\n    ⎢sin(2⋅θ)    sin(4⋅θ)    1   cos(4⋅θ)   ⎥\n    ⎢────────    ────────    ─ - ────────  0⎥\n    ⎢   2           4        4      4       ⎥\n    ⎢                                       ⎥\n    ⎣   0           0             0        0⎦\n\n    A half-wave plate\n\n    >>> pprint(mueller_matrix(half_wave_retarder(theta)), use_unicode=True)\n    ⎡1              0                           0               0 ⎤\n    ⎢                                                             ⎥\n    ⎢        4           2                                        ⎥\n    ⎢0  8⋅sin (θ) - 8⋅sin (θ) + 1           sin(4⋅θ)            0 ⎥\n    ⎢                                                             ⎥\n    ⎢                                     4           2           ⎥\n    ⎢0          sin(4⋅θ)           - 8⋅sin (θ) + 8⋅sin (θ) - 1  0 ⎥\n    ⎢                                                             ⎥\n    ⎣0              0                           0               -1⎦\n\n    A quarter-wave plate\n\n    >>> pprint(mueller_matrix(quarter_wave_retarder(theta)), use_unicode=True)\n    ⎡1       0             0            0    ⎤\n    ⎢                                        ⎥\n    ⎢   cos(4⋅θ)   1    sin(4⋅θ)             ⎥\n    ⎢0  ──────── + ─    ────────    -sin(2⋅θ)⎥\n    ⎢      2       2       2                 ⎥\n    ⎢                                        ⎥\n    ⎢     sin(4⋅θ)    1   cos(4⋅θ)           ⎥\n    ⎢0    ────────    ─ - ────────  cos(2⋅θ) ⎥\n    ⎢        2        2      2               ⎥\n    ⎢                                        ⎥\n    ⎣0    sin(2⋅θ)     -cos(2⋅θ)        0    ⎦\n\n    '
    A = Matrix([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -I, I, 0]])
    return simplify(A * TensorProduct(J, J.conjugate()) * A.inv())

def polarizing_beam_splitter(Tp=1, Rs=1, Ts=0, Rp=0, phia=0, phib=0):
    if False:
        return 10
    'A polarizing beam splitter Jones matrix at angle `theta`.\n\n    Parameters\n    ==========\n\n    J : SymPy Matrix\n        A Jones matrix.\n    Tp : numeric type or SymPy Symbol\n        The transmissivity of the P-polarized component.\n    Rs : numeric type or SymPy Symbol\n        The reflectivity of the S-polarized component.\n    Ts : numeric type or SymPy Symbol\n        The transmissivity of the S-polarized component.\n    Rp : numeric type or SymPy Symbol\n        The reflectivity of the P-polarized component.\n    phia : numeric type or SymPy Symbol\n        The phase difference between transmitted and reflected component for\n        output mode a.\n    phib : numeric type or SymPy Symbol\n        The phase difference between transmitted and reflected component for\n        output mode b.\n\n\n    Returns\n    =======\n\n    SymPy Matrix\n        A 4x4 matrix representing the PBS. This matrix acts on a 4x1 vector\n        whose first two entries are the Jones vector on one of the PBS ports,\n        and the last two entries the Jones vector on the other port.\n\n    Examples\n    ========\n\n    Generic polarizing beam-splitter.\n\n    >>> from sympy import pprint, symbols\n    >>> from sympy.physics.optics.polarization import polarizing_beam_splitter\n    >>> Ts, Rs, Tp, Rp = symbols(r"Ts, Rs, Tp, Rp", positive=True)\n    >>> phia, phib = symbols("phi_a, phi_b", real=True)\n    >>> PBS = polarizing_beam_splitter(Tp, Rs, Ts, Rp, phia, phib)\n    >>> pprint(PBS, use_unicode=False)\n    [   ____                           ____                    ]\n    [ \\/ Tp            0           I*\\/ Rp           0         ]\n    [                                                          ]\n    [                  ____                       ____  I*phi_a]\n    [   0            \\/ Ts            0      -I*\\/ Rs *e       ]\n    [                                                          ]\n    [    ____                         ____                     ]\n    [I*\\/ Rp           0            \\/ Tp            0         ]\n    [                                                          ]\n    [               ____  I*phi_b                    ____      ]\n    [   0      -I*\\/ Rs *e            0            \\/ Ts       ]\n\n    '
    PBS = Matrix([[sqrt(Tp), 0, I * sqrt(Rp), 0], [0, sqrt(Ts), 0, -I * sqrt(Rs) * exp(I * phia)], [I * sqrt(Rp), 0, sqrt(Tp), 0], [0, -I * sqrt(Rs) * exp(I * phib), 0, sqrt(Ts)]])
    return PBS