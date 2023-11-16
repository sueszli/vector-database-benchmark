"""
Test Cython optimize zeros API functions: ``bisect``, ``ridder``, ``brenth``,
and ``brentq`` in `scipy.optimize.cython_optimize`, by finding the roots of a
3rd order polynomial given a sequence of constant terms, ``a0``, and fixed 1st,
2nd, and 3rd order terms in ``args``.

.. math::

    f(x, a0, args) =  ((args[2]*x + args[1])*x + args[0])*x + a0

The 3rd order polynomial function is written in Cython and called in a Python
wrapper named after the zero function. See the private ``_zeros`` Cython module
in `scipy.optimize.cython_optimze` for more information.
"""
import numpy.testing as npt
from scipy.optimize.cython_optimize import _zeros
A0 = tuple((-2.0 - x / 10.0 for x in range(10)))
ARGS = (0.0, 0.0, 1.0)
(XLO, XHI) = (0.0, 2.0)
(XTOL, RTOL, MITR) = (0.001, 0.001, 10)
EXPECTED = [(-a0) ** (1.0 / 3.0) for a0 in A0]

def test_bisect():
    if False:
        for i in range(10):
            print('nop')
    npt.assert_allclose(EXPECTED, list(_zeros.loop_example('bisect', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)), rtol=RTOL, atol=XTOL)

def test_ridder():
    if False:
        i = 10
        return i + 15
    npt.assert_allclose(EXPECTED, list(_zeros.loop_example('ridder', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)), rtol=RTOL, atol=XTOL)

def test_brenth():
    if False:
        i = 10
        return i + 15
    npt.assert_allclose(EXPECTED, list(_zeros.loop_example('brenth', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)), rtol=RTOL, atol=XTOL)

def test_brentq():
    if False:
        return 10
    npt.assert_allclose(EXPECTED, list(_zeros.loop_example('brentq', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)), rtol=RTOL, atol=XTOL)

def test_brentq_full_output():
    if False:
        print('Hello World!')
    output = _zeros.full_output_example((A0[0],) + ARGS, XLO, XHI, XTOL, RTOL, MITR)
    npt.assert_allclose(EXPECTED[0], output['root'], rtol=RTOL, atol=XTOL)
    npt.assert_equal(6, output['iterations'])
    npt.assert_equal(7, output['funcalls'])
    npt.assert_equal(0, output['error_num'])