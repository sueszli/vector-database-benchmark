import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates.matrix_utilities import angle_axis, is_O3, is_rotation, matrix_product, rotation_matrix
from astropy.utils.exceptions import AstropyDeprecationWarning

def test_rotation_matrix():
    if False:
        for i in range(10):
            print('nop')
    assert_allclose(rotation_matrix(0 * u.deg, 'x'), np.eye(3))
    assert_allclose(rotation_matrix(90 * u.deg, 'y'), [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-12)
    assert_allclose(rotation_matrix(-90 * u.deg, 'z'), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert_allclose(rotation_matrix(45 * u.deg, 'x'), rotation_matrix(45 * u.deg, [1, 0, 0]))
    assert_allclose(rotation_matrix(125 * u.deg, 'y'), rotation_matrix(125 * u.deg, [0, 1, 0]))
    assert_allclose(rotation_matrix(-30 * u.deg, 'z'), rotation_matrix(-30 * u.deg, [0, 0, 1]))
    assert_allclose(np.dot(rotation_matrix(180 * u.deg, [1, 1, 0]), [1, 0, 0]), [0, 1, 0], atol=1e-12)
    assert_allclose(rotation_matrix(1e-06 * u.deg, 'x'), rotation_matrix(1e-06 * u.deg, [1, 0, 0]))

def test_angle_axis():
    if False:
        i = 10
        return i + 15
    m1 = rotation_matrix(35 * u.deg, 'x')
    (an1, ax1) = angle_axis(m1)
    assert an1 - 35 * u.deg < 1e-10 * u.deg
    assert_allclose(ax1, [1, 0, 0])
    m2 = rotation_matrix(-89 * u.deg, [1, 1, 0])
    (an2, ax2) = angle_axis(m2)
    assert an2 - 89 * u.deg < 1e-10 * u.deg
    assert_allclose(ax2, [-2 ** (-0.5), -2 ** (-0.5), 0])

def test_is_O3():
    if False:
        for i in range(10):
            print('nop')
    'Test the matrix checker ``is_O3``.'
    m1 = rotation_matrix(35 * u.deg, 'x')
    assert is_O3(m1)
    n1 = np.tile(m1, (2, 1, 1))
    assert tuple(is_O3(n1)) == (True, True)
    nn1 = np.tile(0.5 * m1, (2, 1, 1))
    assert tuple(is_O3(nn1)) == (False, False)
    assert tuple(is_O3(nn1, atol=1)) == (True, True)
    m2 = m1.copy()
    m2[0, 0] *= -1
    assert is_O3(m2)
    n2 = np.stack((m1, m2))
    assert tuple(is_O3(n2)) == (True, True)
    m3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert not is_O3(m3)
    n3 = np.stack((m1, m3))
    assert tuple(is_O3(n3)) == (True, False)

def test_is_rotation():
    if False:
        print('Hello World!')
    'Test the rotation matrix checker ``is_rotation``.'
    m1 = rotation_matrix(35 * u.deg, 'x')
    assert is_rotation(m1)
    assert is_rotation(m1, allow_improper=True)
    n1 = np.tile(m1, (2, 1, 1))
    assert tuple(is_rotation(n1)) == (True, True)
    nn1 = np.tile(0.5 * m1, (2, 1, 1))
    assert tuple(is_rotation(nn1)) == (False, False)
    assert tuple(is_rotation(nn1, atol=10)) == (True, True)
    m2 = np.identity(3)
    m2[0, 0] = -1
    assert not is_rotation(m2)
    assert is_rotation(m2, allow_improper=True)
    n2 = np.stack((m1, m2))
    assert tuple(is_rotation(n2)) == (True, False)
    m3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert not is_rotation(m3)
    assert not is_rotation(m3, allow_improper=True)
    n3 = np.stack((m1, m3))
    assert tuple(is_rotation(n3)) == (True, False)

def test_matrix_product_deprecation():
    if False:
        return 10
    with pytest.warns(AstropyDeprecationWarning, match='Use @ instead\\.$'):
        matrix_product(np.eye(2))