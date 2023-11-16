"""
Tests for the projected separation stuff
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle, Distance
from astropy.coordinates.builtin_frames import FK5, ICRS, Galactic
from astropy.tests.helper import assert_quantity_allclose as assert_allclose
coords = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 10, 0), (0, 0, 90, 0), (0, 0, 180, 0), (0, 45, 0, -45), (0, 60, 0, -30), (-135, -15, 45, 15), (100, -89, -80, 89), (0, 0, 0, 0), (0, 0, 1.0 / 60.0, 1.0 / 60.0)]
correct_seps = [1, 1, 1, 1, 10, 90, 180, 90, 90, 180, 180, 0, 0.023570225877234643]
correctness_margin = 2e-10

def test_angsep():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that the angular separation object also behaves correctly.\n    '
    from astropy.coordinates import angular_separation
    for conv in (np.deg2rad, lambda x: u.Quantity(x, 'deg'), lambda x: Angle(x, 'deg')):
        for ((lon1, lat1, lon2, lat2), corrsep) in zip(coords, correct_seps):
            angsep = angular_separation(conv(lon1), conv(lat1), conv(lon2), conv(lat2))
            assert np.fabs(angsep - conv(corrsep)) < conv(correctness_margin)

def test_fk5_seps():
    if False:
        return 10
    '\n    This tests if `separation` works for FK5 objects.\n\n    This is a regression test for github issue #891\n    '
    a = FK5(1.0 * u.deg, 1.0 * u.deg)
    b = FK5(2.0 * u.deg, 2.0 * u.deg)
    a.separation(b)

def test_proj_separations():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test angular separation functionality\n    '
    c1 = ICRS(ra=0 * u.deg, dec=0 * u.deg)
    c2 = ICRS(ra=0 * u.deg, dec=1 * u.deg)
    sep = c2.separation(c1)
    assert isinstance(sep, Angle)
    assert_allclose(sep.degree, 1.0)
    assert_allclose(sep.arcminute, 60.0)
    with pytest.raises(TypeError):
        c1 + c2
    with pytest.raises(TypeError):
        c1 - c2
    ngp = Galactic(l=0 * u.degree, b=90 * u.degree)
    ncp = ICRS(ra=0 * u.degree, dec=90 * u.degree)
    assert_allclose(ncp.separation(ngp.transform_to(ICRS())).degree, ncp.separation(ngp).degree)
    assert_allclose(ncp.separation(ngp.transform_to(ICRS())).degree, 62.87174758503201)

def test_3d_separations():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test 3D separation functionality\n    '
    c1 = ICRS(ra=1 * u.deg, dec=1 * u.deg, distance=9 * u.kpc)
    c2 = ICRS(ra=1 * u.deg, dec=1 * u.deg, distance=10 * u.kpc)
    sep3d = c2.separation_3d(c1)
    assert isinstance(sep3d, Distance)
    assert_allclose(sep3d - 1 * u.kpc, 0 * u.kpc, atol=1e-12 * u.kpc)