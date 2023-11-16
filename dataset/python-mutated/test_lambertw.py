import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from scipy.special import lambertw
from numpy import nan, inf, pi, e, isnan, log, r_, array, complex128
from scipy.special._testutils import FuncData

def test_values():
    if False:
        print('Hello World!')
    assert_(isnan(lambertw(nan)))
    assert_equal(lambertw(inf, 1).real, inf)
    assert_equal(lambertw(inf, 1).imag, 2 * pi)
    assert_equal(lambertw(-inf, 1).real, inf)
    assert_equal(lambertw(-inf, 1).imag, 3 * pi)
    assert_equal(lambertw(1.0), lambertw(1.0, 0))
    data = [(0, 0, 0), (0 + 0j, 0, 0), (inf, 0, inf), (0, -1, -inf), (0, 1, -inf), (0, 3, -inf), (e, 0, 1), (1, 0, 0.5671432904097838), (-pi / 2, 0, 1j * pi / 2), (-log(2) / 2, 0, -log(2)), (0.25, 0, 0.20388835470224018), (-0.25, 0, -0.3574029561813889), (-1.0 / 10000, 0, -0.00010001000150026672), (-0.25, -1, -2.15329236411035), (0.25, -1, -3.008998009970046 - 4.076529788991597j), (-0.25, -1, -2.15329236411035), (0.25, 1, -3.008998009970046 + 4.076529788991597j), (-0.25, 1, -3.489732284229592 + 7.4140545300960365j), (-4, 0, 0.6788119713209453 + 1.9119507817433994j), (-4, 1, -0.6674310712980098 + 7.768274568027831j), (-4, -1, 0.6788119713209453 - 1.9119507817433994j), (1000, 0, 5.249602852401596), (1000, 1, 4.914922399810545 + 5.4465261597944705j), (1000, -1, 4.914922399810545 - 5.4465261597944705j), (1000, 5, 3.5010625305312892 + 29.961454894118134j), (3 + 4j, 0, 1.281561806123776 + 0.533095222020971j), (-0.4 + 0.4j, 0, -0.10396515323290657 + 0.6189927331517163j), (3 + 4j, 1, -0.11691092896595325 + 5.6188803987128235j), (3 + 4j, -1, 0.2585674068669974 - 3.8521166861614358j), (-0.5, -1, -0.7940236323446894 - 0.7701117505103791j), (-1.0 / 10000, 1, -11.823508372487243 + 6.805460818420021j), (-1.0 / 10000, -1, -11.667114532566355), (-1.0 / 10000, -2, -11.823508372487243 - 6.805460818420021j), (-1.0 / 100000, 4, -14.918689076954054 + 26.185675017878204j), (-1.0 / 100000, 5, -15.093143772637921 + 32.55257212102623j), ((2 + 1j) / 10, 0, 0.17370450376291166 + 0.07178133675283552j), ((2 + 1j) / 10, 1, -3.2174602834982005 + 4.5617543889629255j), ((2 + 1j) / 10, -1, -3.037814050029931 - 3.5394662963350574j), ((2 + 1j) / 10, 4, -4.687850969277325 + 23.83136306976833j), (-(2 + 1j) / 10, 0, -0.22693377251575794 - 0.16498647002015457j), (-(2 + 1j) / 10, 1, -2.4356951704611 + 0.7697406754475629j), (-(2 + 1j) / 10, -1, -3.5485873815198943 - 6.916279218699436j), (-(2 + 1j) / 10, 4, -4.550084692811815 + 20.667298221543465j), (pi, 0, 1.0736581947961492), (-0.5 + 0.002j, 0, -0.7891713813265991 + 0.7674353937999033j), (-0.5 - 0.002j, 0, -0.7891713813265991 - 0.7674353937999033j), (-0.448 + 0.4j, 0, -0.11855133765652383 + 0.6657053431358342j), (-0.448 - 0.4j, 0, -0.11855133765652383 - 0.6657053431358342j)]
    data = array(data, dtype=complex128)

    def w(x, y):
        if False:
            for i in range(10):
                print('nop')
        return lambertw(x, y.real.astype(int))
    with np.errstate(all='ignore'):
        FuncData(w, data, (0, 1), 2, rtol=1e-10, atol=1e-13).check()

def test_ufunc():
    if False:
        for i in range(10):
            print('nop')
    assert_array_almost_equal(lambertw(r_[0.0, e, 1.0]), r_[0.0, 1.0, 0.5671432904097838])

def test_lambertw_ufunc_loop_selection():
    if False:
        for i in range(10):
            print('nop')
    dt = np.dtype(np.complex128)
    assert_equal(lambertw(0, 0, 0).dtype, dt)
    assert_equal(lambertw([0], 0, 0).dtype, dt)
    assert_equal(lambertw(0, [0], 0).dtype, dt)
    assert_equal(lambertw(0, 0, [0]).dtype, dt)
    assert_equal(lambertw([0], [0], [0]).dtype, dt)

@pytest.mark.parametrize('z', [1e-316, -2e-320j, -5e-318 + 1e-320j])
def test_lambertw_subnormal_k0(z):
    if False:
        for i in range(10):
            print('nop')
    w = lambertw(z)
    assert w == z