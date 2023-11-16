import numpy as np
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import earth_orientation
from astropy.time import Time

@pytest.fixture
def tt_to_test():
    if False:
        while True:
            i = 10
    return Time('2022-08-25', scale='tt')

@pytest.mark.parametrize('algorithm, result', [(2006, 23.43633313804873), (2000, 23.43634457995851), (1980, 23.436346167704045)])
def test_obliquity(tt_to_test, algorithm, result):
    if False:
        print('Hello World!')
    assert_allclose(earth_orientation.obliquity(tt_to_test.jd, algorithm=algorithm), result, rtol=1e-13)

def test_precession_matrix_Capitaine(tt_to_test):
    if False:
        print('Hello World!')
    assert_allclose(earth_orientation.precession_matrix_Capitaine(tt_to_test, tt_to_test + 12.345 * u.yr), np.array([[0.99999547, -0.00276086535, -0.00119936388], [0.00276086537, +0.999996189, -1.64025847e-06], [0.00119936384, -1.67103117e-06, +0.999999281]]), rtol=1e-06)

def test_nutation_components2000B(tt_to_test):
    if False:
        for i in range(10):
            print('nop')
    assert_allclose(earth_orientation.nutation_components2000B(tt_to_test.jd), (0.4090413775522035, -5.4418953539440996e-05, 3.176996651841667e-05), rtol=1e-13)

def test_nutation_matrix(tt_to_test):
    if False:
        i = 10
        return i + 15
    assert_allclose(earth_orientation.nutation_matrix(tt_to_test), np.array([[+0.999999999, +4.99295268e-05, +2.16440489e-05], [-4.99288392e-05, +0.999999998, -3.17705068e-05], [-2.16456351e-05, +3.17694261e-05, +0.999999999]]), rtol=1e-06)