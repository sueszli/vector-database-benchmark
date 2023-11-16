from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.interpolate import pade

def test_pade_trivial():
    if False:
        print('Hello World!')
    (nump, denomp) = pade([1.0], 0)
    assert_array_equal(nump.c, [1.0])
    assert_array_equal(denomp.c, [1.0])
    (nump, denomp) = pade([1.0], 0, 0)
    assert_array_equal(nump.c, [1.0])
    assert_array_equal(denomp.c, [1.0])

def test_pade_4term_exp():
    if False:
        while True:
            i = 10
    an = [1.0, 1.0, 0.5, 1.0 / 6]
    (nump, denomp) = pade(an, 0)
    assert_array_almost_equal(nump.c, [1.0 / 6, 0.5, 1.0, 1.0])
    assert_array_almost_equal(denomp.c, [1.0])
    (nump, denomp) = pade(an, 1)
    assert_array_almost_equal(nump.c, [1.0 / 6, 2.0 / 3, 1.0])
    assert_array_almost_equal(denomp.c, [-1.0 / 3, 1.0])
    (nump, denomp) = pade(an, 2)
    assert_array_almost_equal(nump.c, [1.0 / 3, 1.0])
    assert_array_almost_equal(denomp.c, [1.0 / 6, -2.0 / 3, 1.0])
    (nump, denomp) = pade(an, 3)
    assert_array_almost_equal(nump.c, [1.0])
    assert_array_almost_equal(denomp.c, [-1.0 / 6, 0.5, -1.0, 1.0])
    (nump, denomp) = pade(an, 0, 3)
    assert_array_almost_equal(nump.c, [1.0 / 6, 0.5, 1.0, 1.0])
    assert_array_almost_equal(denomp.c, [1.0])
    (nump, denomp) = pade(an, 1, 2)
    assert_array_almost_equal(nump.c, [1.0 / 6, 2.0 / 3, 1.0])
    assert_array_almost_equal(denomp.c, [-1.0 / 3, 1.0])
    (nump, denomp) = pade(an, 2, 1)
    assert_array_almost_equal(nump.c, [1.0 / 3, 1.0])
    assert_array_almost_equal(denomp.c, [1.0 / 6, -2.0 / 3, 1.0])
    (nump, denomp) = pade(an, 3, 0)
    assert_array_almost_equal(nump.c, [1.0])
    assert_array_almost_equal(denomp.c, [-1.0 / 6, 0.5, -1.0, 1.0])
    (nump, denomp) = pade(an, 0, 2)
    assert_array_almost_equal(nump.c, [0.5, 1.0, 1.0])
    assert_array_almost_equal(denomp.c, [1.0])
    (nump, denomp) = pade(an, 1, 1)
    assert_array_almost_equal(nump.c, [1.0 / 2, 1.0])
    assert_array_almost_equal(denomp.c, [-1.0 / 2, 1.0])
    (nump, denomp) = pade(an, 2, 0)
    assert_array_almost_equal(nump.c, [1.0])
    assert_array_almost_equal(denomp.c, [1.0 / 2, -1.0, 1.0])

def test_pade_ints():
    if False:
        i = 10
        return i + 15
    an_int = [1, 2, 3, 4]
    an_flt = [1.0, 2.0, 3.0, 4.0]
    for i in range(0, len(an_int)):
        for j in range(0, len(an_int) - i):
            (nump_int, denomp_int) = pade(an_int, i, j)
            (nump_flt, denomp_flt) = pade(an_flt, i, j)
            assert_array_equal(nump_int.c, nump_flt.c)
            assert_array_equal(denomp_int.c, denomp_flt.c)

def test_pade_complex():
    if False:
        print('Hello World!')
    x = 0.2 + 0.6j
    an = [1.0, x, -x * x.conjugate(), x.conjugate() * x ** 2 + x * x.conjugate() ** 2, -x ** 3 * x.conjugate() - 3 * (x * x.conjugate()) ** 2 - x * x.conjugate() ** 3]
    (nump, denomp) = pade(an, 1, 1)
    assert_array_almost_equal(nump.c, [x + x.conjugate(), 1.0])
    assert_array_almost_equal(denomp.c, [x.conjugate(), 1.0])
    (nump, denomp) = pade(an, 1, 2)
    assert_array_almost_equal(nump.c, [x ** 2, 2 * x + x.conjugate(), 1.0])
    assert_array_almost_equal(denomp.c, [x + x.conjugate(), 1.0])
    (nump, denomp) = pade(an, 2, 2)
    assert_array_almost_equal(nump.c, [x ** 2 + x * x.conjugate() + x.conjugate() ** 2, 2 * (x + x.conjugate()), 1.0])
    assert_array_almost_equal(denomp.c, [x.conjugate() ** 2, x + 2 * x.conjugate(), 1.0])