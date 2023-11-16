"""
Accuracy tests for Ecliptic coordinate systems.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth, R_sun
from astropy.coordinates import SkyCoord
from astropy.coordinates.builtin_frames import FK5, GCRS, ICRS, BarycentricMeanEcliptic, BarycentricTrueEcliptic, CustomBarycentricEcliptic, GeocentricMeanEcliptic, GeocentricTrueEcliptic, HeliocentricEclipticIAU76, HeliocentricMeanEcliptic, HeliocentricTrueEcliptic
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
from astropy.units import allclose as quantity_allclose

def test_against_pytpm_doc_example():
    if False:
        return 10
    "\n    Check that Astropy's Ecliptic systems give answers consistent with pyTPM\n\n    Currently this is only testing against the example given in the pytpm docs\n    "
    fk5_in = SkyCoord('12h22m54.899s', '15d49m20.57s', frame=FK5(equinox='J2000'))
    pytpm_out = BarycentricMeanEcliptic(lon=178.78256462 * u.deg, lat=16.7597002513 * u.deg, equinox='J2000')
    astropy_out = fk5_in.transform_to(pytpm_out)
    assert pytpm_out.separation(astropy_out) < 1 * u.arcsec

def test_ecliptic_heliobary():
    if False:
        while True:
            i = 10
    '\n    Check that the ecliptic transformations for heliocentric and barycentric\n    at least more or less make sense\n    '
    icrs = ICRS(1 * u.deg, 2 * u.deg, distance=1.5 * R_sun)
    bary = icrs.transform_to(BarycentricMeanEcliptic())
    helio = icrs.transform_to(HeliocentricMeanEcliptic())
    assert np.abs(bary.distance - helio.distance) > 1 * u.km
    helio_in_bary_frame = bary.realize_frame(helio.cartesian)
    assert bary.separation(helio_in_bary_frame) > 1 * u.arcmin

@pytest.mark.parametrize(('trueframe', 'meanframe'), [(BarycentricTrueEcliptic, BarycentricMeanEcliptic), (HeliocentricTrueEcliptic, HeliocentricMeanEcliptic), (GeocentricTrueEcliptic, GeocentricMeanEcliptic), (HeliocentricEclipticIAU76, HeliocentricMeanEcliptic)])
def test_ecliptic_roundtrips(trueframe, meanframe):
    if False:
        print('Hello World!')
    '\n    Check that the various ecliptic transformations at least roundtrip\n    '
    icrs = ICRS(1 * u.deg, 2 * u.deg, distance=1.5 * R_sun)
    truecoo = icrs.transform_to(trueframe())
    meancoo = truecoo.transform_to(meanframe())
    truecoo2 = meancoo.transform_to(trueframe())
    assert not quantity_allclose(truecoo.cartesian.xyz, meancoo.cartesian.xyz)
    assert quantity_allclose(truecoo.cartesian.xyz, truecoo2.cartesian.xyz)

@pytest.mark.parametrize(('trueframe', 'meanframe'), [(BarycentricTrueEcliptic, BarycentricMeanEcliptic), (HeliocentricTrueEcliptic, HeliocentricMeanEcliptic), (GeocentricTrueEcliptic, GeocentricMeanEcliptic)])
def test_ecliptic_true_mean_preserve_latitude(trueframe, meanframe):
    if False:
        i = 10
        return i + 15
    '\n    Check that the ecliptic true/mean transformations preserve latitude\n    '
    truecoo = trueframe(90 * u.deg, 0 * u.deg, distance=1 * u.AU)
    meancoo = truecoo.transform_to(meanframe())
    assert not quantity_allclose(truecoo.lon, meancoo.lon)
    assert quantity_allclose(truecoo.lat, meancoo.lat, atol=1e-10 * u.arcsec)

@pytest.mark.parametrize('frame', [HeliocentricMeanEcliptic, HeliocentricTrueEcliptic, HeliocentricEclipticIAU76])
def test_helioecliptic_induced_velocity(frame):
    if False:
        print('Hello World!')
    time = Time('2021-01-01')
    icrs = ICRS(ra=1 * u.deg, dec=2 * u.deg, distance=3 * u.AU, pm_ra_cosdec=0 * u.deg / u.s, pm_dec=0 * u.deg / u.s, radial_velocity=0 * u.m / u.s)
    transformed = icrs.transform_to(frame(obstime=time))
    (_, vel) = get_body_barycentric_posvel('sun', time)
    assert quantity_allclose(transformed.velocity.norm(), vel.norm())
    back = transformed.transform_to(ICRS())
    assert quantity_allclose(back.velocity.norm(), 0 * u.m / u.s, atol=1e-10 * u.m / u.s)

def test_ecl_geo():
    if False:
        while True:
            i = 10
    '\n    Check that the geocentric version at least gets well away from GCRS.  For a\n    true "accuracy" test we need a comparison dataset that is similar to the\n    geocentric/GCRS comparison we want to do here.  Contributions welcome!\n    '
    gcrs = GCRS(10 * u.deg, 20 * u.deg, distance=1.5 * R_earth)
    gecl = gcrs.transform_to(GeocentricMeanEcliptic())
    assert quantity_allclose(gecl.distance, gcrs.distance)

def test_arraytransforms():
    if False:
        return 10
    '\n    Test that transforms to/from ecliptic coordinates work on array coordinates\n    (not testing for accuracy.)\n    '
    ra = np.ones((4,), dtype=float) * u.deg
    dec = 2 * np.ones((4,), dtype=float) * u.deg
    distance = np.ones((4,), dtype=float) * u.au
    test_icrs = ICRS(ra=ra, dec=dec, distance=distance)
    test_gcrs = GCRS(test_icrs.data)
    bary_arr = test_icrs.transform_to(BarycentricMeanEcliptic())
    assert bary_arr.shape == ra.shape
    helio_arr = test_icrs.transform_to(HeliocentricMeanEcliptic())
    assert helio_arr.shape == ra.shape
    geo_arr = test_gcrs.transform_to(GeocentricMeanEcliptic())
    assert geo_arr.shape == ra.shape
    bary_icrs = bary_arr.transform_to(ICRS())
    assert bary_icrs.shape == test_icrs.shape
    helio_icrs = helio_arr.transform_to(ICRS())
    assert helio_icrs.shape == test_icrs.shape
    geo_gcrs = geo_arr.transform_to(GCRS())
    assert geo_gcrs.shape == test_gcrs.shape

def test_roundtrip_scalar():
    if False:
        return 10
    icrs = ICRS(ra=1 * u.deg, dec=2 * u.deg, distance=3 * u.au)
    gcrs = GCRS(icrs.cartesian)
    bary = icrs.transform_to(BarycentricMeanEcliptic())
    helio = icrs.transform_to(HeliocentricMeanEcliptic())
    geo = gcrs.transform_to(GeocentricMeanEcliptic())
    bary_icrs = bary.transform_to(ICRS())
    helio_icrs = helio.transform_to(ICRS())
    geo_gcrs = geo.transform_to(GCRS())
    assert quantity_allclose(bary_icrs.cartesian.xyz, icrs.cartesian.xyz)
    assert quantity_allclose(helio_icrs.cartesian.xyz, icrs.cartesian.xyz)
    assert quantity_allclose(geo_gcrs.cartesian.xyz, gcrs.cartesian.xyz)

@pytest.mark.parametrize('frame', [HeliocentricMeanEcliptic, HeliocentricTrueEcliptic, GeocentricMeanEcliptic, GeocentricTrueEcliptic, HeliocentricEclipticIAU76])
def test_loopback_obstime(frame):
    if False:
        i = 10
        return i + 15
    from_coo = frame(1 * u.deg, 2 * u.deg, 3 * u.AU, obstime='2001-01-01')
    to_frame = frame(obstime='2001-06-30')
    explicit_coo = from_coo.transform_to(ICRS()).transform_to(to_frame)
    implicit_coo = from_coo.transform_to(to_frame)
    assert not quantity_allclose(explicit_coo.lon, from_coo.lon, rtol=1e-10)
    assert not quantity_allclose(explicit_coo.lat, from_coo.lat, rtol=1e-10)
    assert not quantity_allclose(explicit_coo.distance, from_coo.distance, rtol=1e-10)
    assert quantity_allclose(explicit_coo.lon, implicit_coo.lon, rtol=1e-10)
    assert quantity_allclose(explicit_coo.lat, implicit_coo.lat, rtol=1e-10)
    assert quantity_allclose(explicit_coo.distance, implicit_coo.distance, rtol=1e-10)

@pytest.mark.parametrize('frame', [BarycentricMeanEcliptic, BarycentricTrueEcliptic, HeliocentricMeanEcliptic, HeliocentricTrueEcliptic, GeocentricMeanEcliptic, GeocentricTrueEcliptic])
def test_loopback_equinox(frame):
    if False:
        for i in range(10):
            print('nop')
    from_coo = frame(1 * u.deg, 2 * u.deg, 3 * u.AU, equinox='2001-01-01')
    to_frame = frame(equinox='2001-06-30')
    explicit_coo = from_coo.transform_to(ICRS()).transform_to(to_frame)
    implicit_coo = from_coo.transform_to(to_frame)
    assert not quantity_allclose(explicit_coo.lon, from_coo.lon, rtol=1e-10)
    assert not quantity_allclose(explicit_coo.lat, from_coo.lat, rtol=1e-10)
    assert quantity_allclose(explicit_coo.distance, from_coo.distance, rtol=1e-10)
    assert quantity_allclose(explicit_coo.lon, implicit_coo.lon, rtol=1e-10)
    assert quantity_allclose(explicit_coo.lat, implicit_coo.lat, rtol=1e-10)
    assert quantity_allclose(explicit_coo.distance, implicit_coo.distance, rtol=1e-10)

def test_loopback_obliquity():
    if False:
        while True:
            i = 10
    from_coo = CustomBarycentricEcliptic(1 * u.deg, 2 * u.deg, 3 * u.AU, obliquity=84000 * u.arcsec)
    to_frame = CustomBarycentricEcliptic(obliquity=85000 * u.arcsec)
    explicit_coo = from_coo.transform_to(ICRS()).transform_to(to_frame)
    implicit_coo = from_coo.transform_to(to_frame)
    assert not quantity_allclose(explicit_coo.lon, from_coo.lon, rtol=1e-10)
    assert not quantity_allclose(explicit_coo.lat, from_coo.lat, rtol=1e-10)
    assert quantity_allclose(explicit_coo.distance, from_coo.distance, rtol=1e-10)
    assert quantity_allclose(explicit_coo.lon, implicit_coo.lon, rtol=1e-10)
    assert quantity_allclose(explicit_coo.lat, implicit_coo.lat, rtol=1e-10)
    assert quantity_allclose(explicit_coo.distance, implicit_coo.distance, rtol=1e-10)