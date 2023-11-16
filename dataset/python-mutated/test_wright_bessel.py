import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
from scipy.special import rgamma, wright_bessel

@pytest.mark.parametrize('a', [0, 1e-06, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [0, 1e-06, 0.1, 0.5, 1, 10])
def test_wright_bessel_zero(a, b):
    if False:
        for i in range(10):
            print('nop')
    'Test at x = 0.'
    assert_equal(wright_bessel(a, b, 0.0), rgamma(b))

@pytest.mark.parametrize('b', [0, 1e-06, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('x', [0, 1e-06, 0.1, 0.5, 1])
def test_wright_bessel_iv(b, x):
    if False:
        print('Hello World!')
    'Test relation of wright_bessel and modified bessel function iv.\n\n    iv(z) = (1/2*z)**v * Phi(1, v+1; 1/4*z**2).\n    See https://dlmf.nist.gov/10.46.E2\n    '
    if x != 0:
        v = b - 1
        wb = wright_bessel(1, v + 1, x ** 2 / 4.0)
        assert_allclose(np.power(x / 2.0, v) * wb, sc.iv(v, x), rtol=1e-11, atol=1e-11)

@pytest.mark.parametrize('a', [0, 1e-06, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [1, 1 + 0.001, 2, 5, 10])
@pytest.mark.parametrize('x', [0, 1e-06, 0.1, 0.5, 1, 5, 10, 100])
def test_wright_functional(a, b, x):
    if False:
        while True:
            i = 10
    "Test functional relation of wright_bessel.\n\n    Phi(a, b-1, z) = a*z*Phi(a, b+a, z) + (b-1)*Phi(a, b, z)\n\n    Note that d/dx Phi(a, b, x) = Phi(a, b-1, x)\n    See Eq. (22) of\n    B. Stankovic, On the Function of E. M. Wright,\n    Publ. de l' Institut Mathematique, Beograd,\n    Nouvelle S`er. 10 (1970), 113-124.\n    "
    assert_allclose(wright_bessel(a, b - 1, x), a * x * wright_bessel(a, b + a, x) + (b - 1) * wright_bessel(a, b, x), rtol=1e-08, atol=1e-08)
grid_a_b_x_value_acc = np.array([[0.1, 100.0, 709.7827128933841, 8.026353022981087e+34, 2e-08], [0.5, 10.0, 709.7827128933841, 2.680788404494657e+48, 9e-08], [0.5, 10.0, 1000.0, 2.005901980702872e+64, 1e-08], [0.5, 100.0, 1000.0, 3.4112367580445246e-117, 6e-08], [1.0, 20.0, 100000.0, 1.7717158630699857e+225, 3e-11], [1.0, 100.0, 100000.0, 1.0269334596230763e+22, np.nan], [1.0000000000000222, 20.0, 100000.0, 1.7717158630001672e+225, 3e-11], [1.0000000000000222, 100.0, 100000.0, 1.0269334595866202e+22, np.nan], [1.5, 0.0, 500.0, 15648961196.432373, 3e-11], [1.5, 2.220446049250313e-14, 500.0, 15648961196.431465, 3e-11], [1.5, 1e-10, 500.0, 15648961192.344728, 3e-11], [1.5, 1e-05, 500.0, 15648552437.334162, 3e-11], [1.5, 0.1, 500.0, 12049870581.10317, 2e-11], [1.5, 20.0, 100000.0, 7.81930438331405e+43, 3e-09], [1.5, 100.0, 100000.0, 9.653370857459075e-130, np.nan]])

@pytest.mark.xfail
@pytest.mark.parametrize('a, b, x, phi', grid_a_b_x_value_acc[:, :4].tolist())
def test_wright_data_grid_failures(a, b, x, phi):
    if False:
        return 10
    'Test cases of test_data that do not reach relative accuracy of 1e-11'
    assert_allclose(wright_bessel(a, b, x), phi, rtol=1e-11)

@pytest.mark.parametrize('a, b, x, phi, accuracy', grid_a_b_x_value_acc.tolist())
def test_wright_data_grid_less_accurate(a, b, x, phi, accuracy):
    if False:
        i = 10
        return i + 15
    'Test cases of test_data that do not reach relative accuracy of 1e-11\n\n    Here we test for reduced accuracy or even nan.\n    '
    if np.isnan(accuracy):
        assert np.isnan(wright_bessel(a, b, x))
    else:
        assert_allclose(wright_bessel(a, b, x), phi, rtol=accuracy)