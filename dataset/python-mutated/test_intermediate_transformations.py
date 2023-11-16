"""Accuracy tests for GCRS coordinate transformations, primarily to/from AltAz.

"""
import warnings
from importlib import metadata
import erfa
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import CIRS, GCRS, HCRS, ICRS, ITRS, TEME, TETE, AltAz, CartesianDifferential, CartesianRepresentation, EarthLocation, HADec, HeliocentricMeanEcliptic, PrecessedGeocentric, SkyCoord, SphericalRepresentation, UnitSphericalRepresentation, get_sun, golden_spiral_grid, solar_system_ephemeris
from astropy.coordinates.builtin_frames.intermediate_rotation_transforms import cirs_to_itrs_mat, gcrs_to_cirs_mat, get_location_gcrs, tete_to_itrs_mat
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.coordinates.solar_system import get_body
from astropy.tests.helper import CI
from astropy.tests.helper import assert_quantity_allclose as assert_allclose
from astropy.time import Time
from astropy.units import allclose
from astropy.utils import iers
from astropy.utils.compat.optional_deps import HAS_JPLEPHEM
from astropy.utils.exceptions import AstropyWarning

def test_icrs_cirs():
    if False:
        print('Hello World!')
    '\n    Check a few cases of ICRS<->CIRS for consistency.\n\n    Also includes the CIRS<->CIRS transforms at different times, as those go\n    through ICRS\n    '
    usph = golden_spiral_grid(200)
    dist = np.linspace(0.0, 1, len(usph)) * u.pc
    inod = ICRS(usph)
    iwd = ICRS(ra=usph.lon, dec=usph.lat, distance=dist)
    cframe1 = CIRS()
    cirsnod = inod.transform_to(cframe1)
    inod2 = cirsnod.transform_to(ICRS())
    assert_allclose(inod.ra, inod2.ra)
    assert_allclose(inod.dec, inod2.dec)
    cframe2 = CIRS(obstime=Time('J2005'))
    cirsnod2 = inod.transform_to(cframe2)
    assert not allclose(cirsnod.ra, cirsnod2.ra, rtol=1e-08)
    assert not allclose(cirsnod.dec, cirsnod2.dec, rtol=1e-08)
    cirswd = iwd.transform_to(cframe1)
    assert not allclose(cirswd.ra, cirsnod.ra, rtol=1e-08)
    assert not allclose(cirswd.dec, cirsnod.dec, rtol=1e-08)
    assert not allclose(cirswd.distance, iwd.distance, rtol=1e-08)
    cirsnod3 = cirsnod.transform_to(cframe1)
    assert_allclose(cirsnod.ra, cirsnod3.ra)
    assert_allclose(cirsnod.dec, cirsnod3.dec)
    cirsnod4 = cirsnod.transform_to(cframe2)
    assert not allclose(cirsnod4.ra, cirsnod.ra, rtol=1e-08)
    assert not allclose(cirsnod4.dec, cirsnod.dec, rtol=1e-08)
    cirsnod5 = cirsnod4.transform_to(cframe1)
    assert_allclose(cirsnod.ra, cirsnod5.ra)
    assert_allclose(cirsnod.dec, cirsnod5.dec)
usph = golden_spiral_grid(200)
dist = np.linspace(0.5, 1, len(usph)) * u.pc
icrs_coords = [ICRS(usph), ICRS(usph.lon, usph.lat, distance=dist)]
gcrs_frames = [GCRS(), GCRS(obstime=Time('J2005'))]

@pytest.mark.parametrize('icoo', icrs_coords)
def test_icrs_gcrs(icoo):
    if False:
        print('Hello World!')
    '\n    Check ICRS<->GCRS for consistency\n    '
    gcrscoo = icoo.transform_to(gcrs_frames[0])
    icoo2 = gcrscoo.transform_to(ICRS())
    assert_allclose(icoo.distance, icoo2.distance)
    assert_allclose(icoo.ra, icoo2.ra)
    assert_allclose(icoo.dec, icoo2.dec)
    assert isinstance(icoo2.data, icoo.data.__class__)
    gcrscoo2 = icoo.transform_to(gcrs_frames[1])
    assert not allclose(gcrscoo.ra, gcrscoo2.ra, rtol=1e-08, atol=1e-10 * u.deg)
    assert not allclose(gcrscoo.dec, gcrscoo2.dec, rtol=1e-08, atol=1e-10 * u.deg)
    gcrscoo3 = gcrscoo.transform_to(gcrs_frames[0])
    assert_allclose(gcrscoo.ra, gcrscoo3.ra)
    assert_allclose(gcrscoo.dec, gcrscoo3.dec)
    gcrscoo4 = gcrscoo.transform_to(gcrs_frames[1])
    assert not allclose(gcrscoo4.ra, gcrscoo.ra, rtol=1e-08, atol=1e-10 * u.deg)
    assert not allclose(gcrscoo4.dec, gcrscoo.dec, rtol=1e-08, atol=1e-10 * u.deg)
    gcrscoo5 = gcrscoo4.transform_to(gcrs_frames[0])
    assert_allclose(gcrscoo.ra, gcrscoo5.ra, rtol=1e-08, atol=1e-10 * u.deg)
    assert_allclose(gcrscoo.dec, gcrscoo5.dec, rtol=1e-08, atol=1e-10 * u.deg)
    gframe3 = GCRS(obsgeoloc=[385000.0, 0, 0] * u.km, obsgeovel=[1, 0, 0] * u.km / u.s)
    gcrscoo6 = icoo.transform_to(gframe3)
    assert not allclose(gcrscoo.ra, gcrscoo6.ra, rtol=1e-08, atol=1e-10 * u.deg)
    assert not allclose(gcrscoo.dec, gcrscoo6.dec, rtol=1e-08, atol=1e-10 * u.deg)
    icooviag3 = gcrscoo6.transform_to(ICRS())
    assert_allclose(icoo.ra, icooviag3.ra)
    assert_allclose(icoo.dec, icooviag3.dec)

@pytest.mark.parametrize('gframe', gcrs_frames)
def test_icrs_gcrs_dist_diff(gframe):
    if False:
        return 10
    '\n    Check that with and without distance give different ICRS<->GCRS answers\n    '
    gcrsnod = icrs_coords[0].transform_to(gframe)
    gcrswd = icrs_coords[1].transform_to(gframe)
    assert not allclose(gcrswd.ra, gcrsnod.ra, rtol=1e-08, atol=1e-10 * u.deg)
    assert not allclose(gcrswd.dec, gcrsnod.dec, rtol=1e-08, atol=1e-10 * u.deg)
    assert not allclose(gcrswd.distance, icrs_coords[1].distance, rtol=1e-08, atol=1e-10 * u.pc)

def test_cirs_to_altaz():
    if False:
        for i in range(10):
            print('nop')
    '\n    Check the basic CIRS<->AltAz transforms.  More thorough checks implicitly\n    happen in `test_iau_fullstack`\n    '
    from astropy.coordinates import EarthLocation
    usph = golden_spiral_grid(200)
    dist = np.linspace(0.5, 1, len(usph)) * u.pc
    cirs = CIRS(usph, obstime='J2000')
    crepr = SphericalRepresentation(lon=usph.lon, lat=usph.lat, distance=dist)
    cirscart = CIRS(crepr, obstime=cirs.obstime, representation_type=CartesianRepresentation)
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    altazframe = AltAz(location=loc, obstime=Time('J2005'))
    cirs2 = cirs.transform_to(altazframe).transform_to(cirs)
    cirs3 = cirscart.transform_to(altazframe).transform_to(cirs)
    assert_allclose(cirs.ra, cirs2.ra)
    assert_allclose(cirs.dec, cirs2.dec)
    assert_allclose(cirs.ra, cirs3.ra)
    assert_allclose(cirs.dec, cirs3.dec)

def test_cirs_to_hadec():
    if False:
        print('Hello World!')
    '\n    Check the basic CIRS<->HADec transforms.\n    '
    from astropy.coordinates import EarthLocation
    usph = golden_spiral_grid(200)
    dist = np.linspace(0.5, 1, len(usph)) * u.pc
    cirs = CIRS(usph, obstime='J2000')
    crepr = SphericalRepresentation(lon=usph.lon, lat=usph.lat, distance=dist)
    cirscart = CIRS(crepr, obstime=cirs.obstime, representation_type=CartesianRepresentation)
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    hadecframe = HADec(location=loc, obstime=Time('J2005'))
    cirs2 = cirs.transform_to(hadecframe).transform_to(cirs)
    cirs3 = cirscart.transform_to(hadecframe).transform_to(cirs)
    assert_allclose(cirs.ra, cirs2.ra)
    assert_allclose(cirs.dec, cirs2.dec)
    assert_allclose(cirs.ra, cirs3.ra)
    assert_allclose(cirs.dec, cirs3.dec)

def test_itrs_topo_to_altaz_with_refraction():
    if False:
        for i in range(10):
            print('nop')
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    usph = golden_spiral_grid(200)
    dist = np.linspace(1.0, 1000.0, len(usph)) * u.au
    icrs = ICRS(ra=usph.lon, dec=usph.lat, distance=dist)
    altaz_frame1 = AltAz(obstime='J2000', location=loc)
    altaz_frame2 = AltAz(obstime='J2000', location=loc, pressure=1000.0 * u.hPa, relative_humidity=0.5)
    cirs_frame = CIRS(obstime='J2000', location=loc)
    itrs_frame = ITRS(location=loc)
    altaz1 = icrs.transform_to(altaz_frame1)
    altaz2 = icrs.transform_to(altaz_frame2)
    cirs = altaz2.transform_to(cirs_frame)
    altaz3 = cirs.transform_to(altaz_frame1)
    itrs = icrs.transform_to(itrs_frame)
    altaz11 = itrs.transform_to(altaz_frame1)
    assert_allclose(altaz11.az - altaz1.az, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(altaz11.alt - altaz1.alt, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(altaz11.distance - altaz1.distance, 0 * u.cm, atol=10.0 * u.cm)
    itrs11 = altaz11.transform_to(itrs_frame)
    assert_allclose(itrs11.x, itrs.x)
    assert_allclose(itrs11.y, itrs.y)
    assert_allclose(itrs11.z, itrs.z)
    altaz22 = itrs.transform_to(altaz_frame2)
    assert_allclose(altaz22.az - altaz2.az, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(altaz22.alt - altaz2.alt, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(altaz22.distance - altaz2.distance, 0 * u.cm, atol=10.0 * u.cm)
    itrs = altaz22.transform_to(itrs_frame)
    altaz33 = itrs.transform_to(altaz_frame1)
    assert_allclose(altaz33.az - altaz3.az, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(altaz33.alt - altaz3.alt, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(altaz33.distance - altaz3.distance, 0 * u.cm, atol=10.0 * u.cm)

def test_itrs_topo_to_hadec_with_refraction():
    if False:
        while True:
            i = 10
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    usph = golden_spiral_grid(200)
    dist = np.linspace(1.0, 1000.0, len(usph)) * u.au
    icrs = ICRS(ra=usph.lon, dec=usph.lat, distance=dist)
    hadec_frame1 = HADec(obstime='J2000', location=loc)
    hadec_frame2 = HADec(obstime='J2000', location=loc, pressure=1000.0 * u.hPa, relative_humidity=0.5)
    cirs_frame = CIRS(obstime='J2000', location=loc)
    itrs_frame = ITRS(location=loc)
    hadec1 = icrs.transform_to(hadec_frame1)
    hadec2 = icrs.transform_to(hadec_frame2)
    cirs = hadec2.transform_to(cirs_frame)
    hadec3 = cirs.transform_to(hadec_frame1)
    itrs = icrs.transform_to(itrs_frame)
    hadec11 = itrs.transform_to(hadec_frame1)
    assert_allclose(hadec11.ha - hadec1.ha, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(hadec11.dec - hadec1.dec, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(hadec11.distance - hadec1.distance, 0 * u.cm, atol=10.0 * u.cm)
    itrs11 = hadec11.transform_to(itrs_frame)
    assert_allclose(itrs11.x, itrs.x)
    assert_allclose(itrs11.y, itrs.y)
    assert_allclose(itrs11.z, itrs.z)
    hadec22 = itrs.transform_to(hadec_frame2)
    assert_allclose(hadec22.ha - hadec2.ha, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(hadec22.dec - hadec2.dec, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(hadec22.distance - hadec2.distance, 0 * u.cm, atol=10.0 * u.cm)
    itrs = hadec22.transform_to(itrs_frame)
    hadec33 = itrs.transform_to(hadec_frame1)
    assert_allclose(hadec33.ha - hadec3.ha, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(hadec33.dec - hadec3.dec, 0 * u.mas, atol=0.1 * u.mas)
    assert_allclose(hadec33.distance - hadec3.distance, 0 * u.cm, atol=10.0 * u.cm)

def test_gcrs_itrs():
    if False:
        return 10
    '\n    Check basic GCRS<->ITRS transforms for round-tripping.\n    '
    usph = golden_spiral_grid(200)
    gcrs = GCRS(usph, obstime='J2000')
    gcrs6 = GCRS(usph, obstime='J2006')
    gcrs2 = gcrs.transform_to(ITRS()).transform_to(gcrs)
    gcrs6_2 = gcrs6.transform_to(ITRS()).transform_to(gcrs)
    assert_allclose(gcrs.ra, gcrs2.ra)
    assert_allclose(gcrs.dec, gcrs2.dec)
    assert not allclose(gcrs.ra, gcrs6_2.ra, rtol=1e-08)
    assert not allclose(gcrs.dec, gcrs6_2.dec, rtol=1e-08)
    gcrsc = gcrs.realize_frame(gcrs.data)
    gcrsc.representation_type = CartesianRepresentation
    gcrsc2 = gcrsc.transform_to(ITRS()).transform_to(gcrsc)
    assert_allclose(gcrsc.spherical.lon, gcrsc2.ra)
    assert_allclose(gcrsc.spherical.lat, gcrsc2.dec)

def test_cirs_itrs():
    if False:
        while True:
            i = 10
    '\n    Check basic CIRS<->ITRS geocentric transforms for round-tripping.\n    '
    usph = golden_spiral_grid(200)
    cirs = CIRS(usph, obstime='J2000')
    cirs6 = CIRS(usph, obstime='J2006')
    cirs2 = cirs.transform_to(ITRS()).transform_to(cirs)
    cirs6_2 = cirs6.transform_to(ITRS()).transform_to(cirs)
    assert_allclose(cirs.ra, cirs2.ra)
    assert_allclose(cirs.dec, cirs2.dec)
    assert not allclose(cirs.ra, cirs6_2.ra)
    assert not allclose(cirs.dec, cirs6_2.dec)

def test_cirs_itrs_topo():
    if False:
        print('Hello World!')
    '\n    Check basic CIRS<->ITRS topocentric transforms for round-tripping.\n    '
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    usph = golden_spiral_grid(200)
    cirs = CIRS(usph, obstime='J2000', location=loc)
    cirs6 = CIRS(usph, obstime='J2006', location=loc)
    cirs2 = cirs.transform_to(ITRS(location=loc)).transform_to(cirs)
    cirs6_2 = cirs6.transform_to(ITRS(location=loc)).transform_to(cirs)
    assert_allclose(cirs.ra, cirs2.ra)
    assert_allclose(cirs.dec, cirs2.dec)
    assert not allclose(cirs.ra, cirs6_2.ra)
    assert not allclose(cirs.dec, cirs6_2.dec)

def test_gcrs_cirs():
    if False:
        while True:
            i = 10
    "\n    Check GCRS<->CIRS transforms for round-tripping.  More complicated than the\n    above two because it's multi-hop\n    "
    usph = golden_spiral_grid(200)
    gcrs = GCRS(usph, obstime='J2000')
    gcrs6 = GCRS(usph, obstime='J2006')
    gcrs2 = gcrs.transform_to(CIRS()).transform_to(gcrs)
    gcrs6_2 = gcrs6.transform_to(CIRS()).transform_to(gcrs)
    assert_allclose(gcrs.ra, gcrs2.ra)
    assert_allclose(gcrs.dec, gcrs2.dec)
    assert not allclose(gcrs.ra, gcrs6_2.ra, rtol=1e-08)
    assert not allclose(gcrs.dec, gcrs6_2.dec, rtol=1e-08)
    gcrs3 = gcrs.transform_to(ITRS()).transform_to(CIRS()).transform_to(ITRS()).transform_to(gcrs)
    assert_allclose(gcrs.ra, gcrs3.ra)
    assert_allclose(gcrs.dec, gcrs3.dec)
    gcrs4 = gcrs.transform_to(ICRS()).transform_to(CIRS()).transform_to(ICRS()).transform_to(gcrs)
    assert_allclose(gcrs.ra, gcrs4.ra)
    assert_allclose(gcrs.dec, gcrs4.dec)

def test_gcrs_altaz():
    if False:
        i = 10
        return i + 15
    '\n    Check GCRS<->AltAz transforms for round-tripping.  Has multiple paths\n    '
    from astropy.coordinates import EarthLocation
    usph = golden_spiral_grid(128)
    gcrs = GCRS(usph, obstime='J2000')[None]
    times = Time(np.linspace(2456293.25, 2456657.25, 51) * u.day, format='jd')[:, None]
    loc = EarthLocation(lon=10 * u.deg, lat=80.0 * u.deg)
    aaframe = AltAz(obstime=times, location=loc)
    aa1 = gcrs.transform_to(aaframe)
    aa2 = gcrs.transform_to(ICRS()).transform_to(CIRS()).transform_to(aaframe)
    aa3 = gcrs.transform_to(ITRS()).transform_to(CIRS()).transform_to(aaframe)
    assert_allclose(aa1.alt, aa2.alt)
    assert_allclose(aa1.az, aa2.az)
    assert_allclose(aa1.alt, aa3.alt)
    assert_allclose(aa1.az, aa3.az)

def test_gcrs_hadec():
    if False:
        i = 10
        return i + 15
    '\n    Check GCRS<->HADec transforms for round-tripping.  Has multiple paths\n    '
    from astropy.coordinates import EarthLocation
    usph = golden_spiral_grid(128)
    gcrs = GCRS(usph, obstime='J2000')
    times = Time(np.linspace(2456293.25, 2456657.25, 51) * u.day, format='jd')[:, None]
    loc = EarthLocation(lon=10 * u.deg, lat=80.0 * u.deg)
    hdframe = HADec(obstime=times, location=loc)
    hd1 = gcrs.transform_to(hdframe)
    hd2 = gcrs.transform_to(ICRS()).transform_to(CIRS()).transform_to(hdframe)
    hd3 = gcrs.transform_to(ITRS()).transform_to(CIRS()).transform_to(hdframe)
    assert_allclose(hd1.dec, hd2.dec)
    assert_allclose(hd1.ha, hd2.ha)
    assert_allclose(hd1.dec, hd3.dec)
    assert_allclose(hd1.ha, hd3.ha)

def test_precessed_geocentric():
    if False:
        return 10
    assert PrecessedGeocentric().equinox.jd == Time('J2000').jd
    gcrs_coo = GCRS(180 * u.deg, 2 * u.deg, distance=10000 * u.km)
    pgeo_coo = gcrs_coo.transform_to(PrecessedGeocentric())
    assert np.abs(gcrs_coo.ra - pgeo_coo.ra) > 10 * u.marcsec
    assert np.abs(gcrs_coo.dec - pgeo_coo.dec) > 10 * u.marcsec
    assert_allclose(gcrs_coo.distance, pgeo_coo.distance)
    gcrs_roundtrip = pgeo_coo.transform_to(GCRS())
    assert_allclose(gcrs_coo.ra, gcrs_roundtrip.ra)
    assert_allclose(gcrs_coo.dec, gcrs_roundtrip.dec)
    assert_allclose(gcrs_coo.distance, gcrs_roundtrip.distance)
    pgeo_coo2 = gcrs_coo.transform_to(PrecessedGeocentric(equinox='B1850'))
    assert np.abs(gcrs_coo.ra - pgeo_coo2.ra) > 1.5 * u.deg
    assert np.abs(gcrs_coo.dec - pgeo_coo2.dec) > 0.5 * u.deg
    assert_allclose(gcrs_coo.distance, pgeo_coo2.distance)
    gcrs2_roundtrip = pgeo_coo2.transform_to(GCRS())
    assert_allclose(gcrs_coo.ra, gcrs2_roundtrip.ra)
    assert_allclose(gcrs_coo.dec, gcrs2_roundtrip.dec)
    assert_allclose(gcrs_coo.distance, gcrs2_roundtrip.distance)

def test_precessed_geocentric_different_obstime():
    if False:
        return 10
    precessedgeo1 = PrecessedGeocentric(obstime='2021-09-07')
    precessedgeo2 = PrecessedGeocentric(obstime='2021-06-07')
    gcrs_coord = GCRS(10 * u.deg, 20 * u.deg, 3 * u.AU, obstime=precessedgeo1.obstime)
    pg_coord1 = gcrs_coord.transform_to(precessedgeo1)
    pg_coord2 = gcrs_coord.transform_to(precessedgeo2)
    assert not pg_coord1.is_equivalent_frame(pg_coord2)
    assert not allclose(pg_coord1.cartesian.xyz, pg_coord2.cartesian.xyz)
    loopback1 = pg_coord1.transform_to(gcrs_coord)
    loopback2 = pg_coord2.transform_to(gcrs_coord)
    assert loopback1.is_equivalent_frame(gcrs_coord)
    assert loopback2.is_equivalent_frame(gcrs_coord)
    assert_allclose(loopback1.cartesian.xyz, gcrs_coord.cartesian.xyz)
    assert_allclose(loopback2.cartesian.xyz, gcrs_coord.cartesian.xyz)
totest_frames = [AltAz(location=EarthLocation(-90 * u.deg, 65 * u.deg), obstime=Time('J2000')), AltAz(location=EarthLocation(120 * u.deg, -35 * u.deg), obstime=Time('J2000')), AltAz(location=EarthLocation(-90 * u.deg, 65 * u.deg), obstime=Time('2014-01-01 00:00:00')), AltAz(location=EarthLocation(-90 * u.deg, 65 * u.deg), obstime=Time('2014-08-01 08:00:00')), AltAz(location=EarthLocation(120 * u.deg, -35 * u.deg), obstime=Time('2014-01-01 00:00:00'))]
MOONDIST = 385000 * u.km
MOONDIST_CART = CartesianRepresentation(3 ** (-0.5) * MOONDIST, 3 ** (-0.5) * MOONDIST, 3 ** (-0.5) * MOONDIST)
EARTHECC = 0.017 + 0.005

@pytest.mark.parametrize('testframe', totest_frames)
def test_gcrs_altaz_sunish(testframe):
    if False:
        while True:
            i = 10
    '\n    Sanity-check that the sun is at a reasonable distance from any altaz\n    '
    sun = get_sun(testframe.obstime)
    assert sun.frame.name == 'gcrs'
    assert (EARTHECC - 1) * u.au < sun.distance.to(u.au) < (EARTHECC + 1) * u.au
    sunaa = sun.transform_to(testframe)
    assert (EARTHECC - 1) * u.au < sunaa.distance.to(u.au) < (EARTHECC + 1) * u.au

@pytest.mark.parametrize('testframe', totest_frames)
def test_gcrs_altaz_moonish(testframe):
    if False:
        while True:
            i = 10
    '\n    Sanity-check that an object resembling the moon goes to the right place with\n    a GCRS->AltAz transformation\n    '
    moon = GCRS(MOONDIST_CART, obstime=testframe.obstime)
    moonaa = moon.transform_to(testframe)
    assert 1000 * u.km < np.abs(moonaa.distance - moon.distance).to(u.au) < 7000 * u.km
    moon2 = moonaa.transform_to(moon)
    assert_allclose(moon.cartesian.xyz, moon2.cartesian.xyz)

@pytest.mark.parametrize('testframe', totest_frames)
def test_gcrs_altaz_bothroutes(testframe):
    if False:
        while True:
            i = 10
    '\n    Repeat of both the moonish and sunish tests above to make sure the two\n    routes through the coordinate graph are consistent with each other\n    '
    sun = get_sun(testframe.obstime)
    sunaa_viaicrs = sun.transform_to(ICRS()).transform_to(testframe)
    sunaa_viaitrs = sun.transform_to(ITRS(obstime=testframe.obstime)).transform_to(testframe)
    moon = GCRS(MOONDIST_CART, obstime=testframe.obstime)
    moonaa_viaicrs = moon.transform_to(ICRS()).transform_to(testframe)
    moonaa_viaitrs = moon.transform_to(ITRS(obstime=testframe.obstime)).transform_to(testframe)
    assert_allclose(sunaa_viaicrs.cartesian.xyz, sunaa_viaitrs.cartesian.xyz)
    assert_allclose(moonaa_viaicrs.cartesian.xyz, moonaa_viaitrs.cartesian.xyz)

@pytest.mark.parametrize('testframe', totest_frames)
def test_cirs_altaz_moonish(testframe):
    if False:
        while True:
            i = 10
    '\n    Sanity-check that an object resembling the moon goes to the right place with\n    a CIRS<->AltAz transformation\n    '
    moon = CIRS(MOONDIST_CART, obstime=testframe.obstime)
    moonaa = moon.transform_to(testframe)
    assert 1000 * u.km < np.abs(moonaa.distance - moon.distance).to(u.km) < 7000 * u.km
    moon2 = moonaa.transform_to(moon)
    assert_allclose(moon.cartesian.xyz, moon2.cartesian.xyz)

@pytest.mark.parametrize('testframe', totest_frames)
def test_cirs_altaz_nodist(testframe):
    if False:
        i = 10
        return i + 15
    '\n    Check that a UnitSphericalRepresentation coordinate round-trips for the\n    CIRS<->AltAz transformation.\n    '
    coo0 = CIRS(UnitSphericalRepresentation(10 * u.deg, 20 * u.deg), obstime=testframe.obstime)
    coo1 = coo0.transform_to(testframe).transform_to(coo0)
    assert_allclose(coo0.cartesian.xyz, coo1.cartesian.xyz)

@pytest.mark.parametrize('testframe', totest_frames)
def test_cirs_icrs_moonish(testframe):
    if False:
        return 10
    '\n    check that something like the moon goes to about the right distance from the\n    ICRS origin when starting from CIRS\n    '
    moonish = CIRS(MOONDIST_CART, obstime=testframe.obstime)
    moonicrs = moonish.transform_to(ICRS())
    assert 0.97 * u.au < moonicrs.distance < 1.03 * u.au

@pytest.mark.parametrize('testframe', totest_frames)
def test_gcrs_icrs_moonish(testframe):
    if False:
        i = 10
        return i + 15
    '\n    check that something like the moon goes to about the right distance from the\n    ICRS origin when starting from GCRS\n    '
    moonish = GCRS(MOONDIST_CART, obstime=testframe.obstime)
    moonicrs = moonish.transform_to(ICRS())
    assert 0.97 * u.au < moonicrs.distance < 1.03 * u.au

@pytest.mark.parametrize('testframe', totest_frames)
def test_icrs_gcrscirs_sunish(testframe):
    if False:
        i = 10
        return i + 15
    '\n    check that the ICRS barycenter goes to about the right distance from various\n    ~geocentric frames (other than testframe)\n    '
    icrs = ICRS(0 * u.deg, 0 * u.deg, distance=10 * u.km)
    gcrs = icrs.transform_to(GCRS(obstime=testframe.obstime))
    assert (EARTHECC - 1) * u.au < gcrs.distance.to(u.au) < (EARTHECC + 1) * u.au
    cirs = icrs.transform_to(CIRS(obstime=testframe.obstime))
    assert (EARTHECC - 1) * u.au < cirs.distance.to(u.au) < (EARTHECC + 1) * u.au
    itrs = icrs.transform_to(ITRS(obstime=testframe.obstime))
    assert (EARTHECC - 1) * u.au < itrs.spherical.distance.to(u.au) < (EARTHECC + 1) * u.au

@pytest.mark.parametrize('testframe', totest_frames)
def test_icrs_altaz_moonish(testframe):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that something expressed in *ICRS* as being moon-like goes to the\n    right AltAz distance\n    '
    (earth_pv_helio, earth_pv_bary) = erfa.epv00(*get_jd12(testframe.obstime, 'tdb'))
    earth_icrs_xyz = earth_pv_bary[0] * u.au
    moonoffset = [0, 0, MOONDIST.value] * MOONDIST.unit
    moonish_icrs = ICRS(CartesianRepresentation(earth_icrs_xyz + moonoffset))
    moonaa = moonish_icrs.transform_to(testframe)
    assert 1000 * u.km < np.abs(moonaa.distance - MOONDIST).to(u.au) < 7000 * u.km

def test_gcrs_self_transform_closeby():
    if False:
        return 10
    '\n    Tests GCRS self transform for objects which are nearby and thus\n    have reasonable parallax.\n\n    Moon positions were originally created using JPL DE432s ephemeris.\n\n    The two lunar positions (one geocentric, one at a defined location)\n    are created via a transformation from ICRS to two different GCRS frames.\n\n    We test that the GCRS-GCRS self transform can correctly map one GCRS\n    frame onto the other.\n    '
    t = Time('2014-12-25T07:00')
    moon_geocentric = SkyCoord(GCRS(318.10579159 * u.deg, -11.65281165 * u.deg, 365042.64880308 * u.km, obstime=t))
    obsgeoloc = [-5592982.59658935, -63054.1948592, 3059763.90102216] * u.m
    obsgeovel = [4.59798494, -407.84677071, 0.0] * u.m / u.s
    moon_lapalma = SkyCoord(GCRS(318.7048445 * u.deg, -11.98761996 * u.deg, 369722.8231031 * u.km, obstime=t, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel))
    transformed = moon_geocentric.transform_to(moon_lapalma.frame)
    delta = transformed.separation_3d(moon_lapalma)
    assert_allclose(delta, 0.0 * u.m, atol=1 * u.m)

def test_teme_itrf():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test case transform from TEME to ITRF.\n\n    Test case derives from example on appendix C of Vallado, Crawford, Hujsak & Kelso (2006).\n    See https://celestrak.com/publications/AIAA/2006-6753/AIAA-2006-6753-Rev2.pdf\n    '
    v_itrf = CartesianDifferential(-3.22563652, -2.87245145, 5.531924446, unit=u.km / u.s)
    p_itrf = CartesianRepresentation(-1033.479383, 7901.295274, 6380.3565958, unit=u.km, differentials={'s': v_itrf})
    t = Time('2004-04-06T07:51:28.386')
    teme = ITRS(p_itrf, obstime=t).transform_to(TEME(obstime=t))
    v_teme = CartesianDifferential(-4.746131487, 0.785818041, 5.531931288, unit=u.km / u.s)
    p_teme = CartesianRepresentation(5094.1801621, 6127.6446505, 6380.3445327, unit=u.km, differentials={'s': v_teme})
    assert_allclose(teme.cartesian.without_differentials().xyz, p_teme.without_differentials().xyz, atol=30 * u.cm)
    assert_allclose(teme.cartesian.differentials['s'].d_xyz, p_teme.differentials['s'].d_xyz, atol=1.0 * u.cm / u.s)
    itrf = teme.transform_to(ITRS(obstime=t))
    assert_allclose(itrf.cartesian.without_differentials().xyz, p_itrf.without_differentials().xyz, atol=100 * u.cm)
    assert_allclose(itrf.cartesian.differentials['s'].d_xyz, p_itrf.differentials['s'].d_xyz, atol=1 * u.cm / u.s)

def test_precessedgeocentric_loopback():
    if False:
        return 10
    from_coo = PrecessedGeocentric(1 * u.deg, 2 * u.deg, 3 * u.AU, obstime='2001-01-01', equinox='2001-01-01')
    to_frame = PrecessedGeocentric(obstime='2001-06-30', equinox='2001-01-01')
    explicit_coo = from_coo.transform_to(ICRS()).transform_to(to_frame)
    implicit_coo = from_coo.transform_to(to_frame)
    assert not allclose(explicit_coo.ra, from_coo.ra, rtol=1e-10)
    assert not allclose(explicit_coo.dec, from_coo.dec, rtol=1e-10)
    assert not allclose(explicit_coo.distance, from_coo.distance, rtol=1e-10)
    assert_allclose(explicit_coo.ra, implicit_coo.ra, rtol=1e-10)
    assert_allclose(explicit_coo.dec, implicit_coo.dec, rtol=1e-10)
    assert_allclose(explicit_coo.distance, implicit_coo.distance, rtol=1e-10)
    to_frame = PrecessedGeocentric(obstime='2001-01-01', equinox='2001-06-30')
    explicit_coo = from_coo.transform_to(ICRS()).transform_to(to_frame)
    implicit_coo = from_coo.transform_to(to_frame)
    assert not allclose(explicit_coo.ra, from_coo.ra, rtol=1e-10)
    assert not allclose(explicit_coo.dec, from_coo.dec, rtol=1e-10)
    assert allclose(explicit_coo.distance, from_coo.distance, rtol=1e-10)
    assert_allclose(explicit_coo.ra, implicit_coo.ra, rtol=1e-10)
    assert_allclose(explicit_coo.dec, implicit_coo.dec, rtol=1e-10)
    assert_allclose(explicit_coo.distance, implicit_coo.distance, rtol=1e-10)

def test_teme_loopback():
    if False:
        print('Hello World!')
    from_coo = TEME(1 * u.AU, 2 * u.AU, 3 * u.AU, obstime='2001-01-01')
    to_frame = TEME(obstime='2001-06-30')
    explicit_coo = from_coo.transform_to(ICRS()).transform_to(to_frame)
    implicit_coo = from_coo.transform_to(to_frame)
    assert not allclose(explicit_coo.cartesian.xyz, from_coo.cartesian.xyz, rtol=1e-10)
    assert_allclose(explicit_coo.cartesian.xyz, implicit_coo.cartesian.xyz, rtol=1e-10)

@pytest.mark.remote_data
def test_earth_orientation_table(monkeypatch):
    if False:
        return 10
    'Check that we can set the IERS table used as Earth Reference.\n\n    Use the here and now to be sure we get a difference.\n    '
    monkeypatch.setattr('astropy.utils.iers.conf.auto_download', True)
    t = Time.now()
    location = EarthLocation(lat=0 * u.deg, lon=0 * u.deg)
    altaz = AltAz(location=location, obstime=t)
    sc = SkyCoord(1 * u.deg, 2 * u.deg)
    if CI:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*using local IERS-B.*')
            warnings.filterwarnings('ignore', message='.*unclosed.*')
            altaz_auto = sc.transform_to(altaz)
    else:
        altaz_auto = sc.transform_to(altaz)
    with iers.earth_orientation_table.set(iers.IERS_B.open()):
        with pytest.warns(AstropyWarning, match='after IERS data'):
            altaz_b = sc.transform_to(altaz)
    sep_b_auto = altaz_b.separation(altaz_auto)
    assert_allclose(sep_b_auto, 0.0 * u.deg, atol=1 * u.arcsec)
    assert sep_b_auto > 10 * u.microarcsecond
    altaz_auto2 = sc.transform_to(altaz)
    assert_allclose(altaz_auto2.separation(altaz_auto), 0 * u.deg)

@pytest.mark.remote_data
@pytest.mark.skipif(not HAS_JPLEPHEM, reason='requires jplephem')
def test_ephemerides():
    if False:
        for i in range(10):
            print('nop')
    '\n    We test that using different ephemerides gives very similar results\n    for transformations\n    '
    t = Time('2014-12-25T07:00')
    moon = SkyCoord(GCRS(318.10579159 * u.deg, -11.65281165 * u.deg, 365042.64880308 * u.km, obstime=t))
    icrs_frame = ICRS()
    hcrs_frame = HCRS(obstime=t)
    ecl_frame = HeliocentricMeanEcliptic(equinox=t)
    cirs_frame = CIRS(obstime=t)
    moon_icrs_builtin = moon.transform_to(icrs_frame)
    moon_hcrs_builtin = moon.transform_to(hcrs_frame)
    moon_helioecl_builtin = moon.transform_to(ecl_frame)
    moon_cirs_builtin = moon.transform_to(cirs_frame)
    with solar_system_ephemeris.set('jpl'):
        moon_icrs_jpl = moon.transform_to(icrs_frame)
        moon_hcrs_jpl = moon.transform_to(hcrs_frame)
        moon_helioecl_jpl = moon.transform_to(ecl_frame)
        moon_cirs_jpl = moon.transform_to(cirs_frame)
    sep_icrs = moon_icrs_builtin.separation(moon_icrs_jpl)
    sep_hcrs = moon_hcrs_builtin.separation(moon_hcrs_jpl)
    sep_helioecl = moon_helioecl_builtin.separation(moon_helioecl_jpl)
    sep_cirs = moon_cirs_builtin.separation(moon_cirs_jpl)
    assert_allclose([sep_icrs, sep_hcrs, sep_helioecl], 0.0 * u.deg, atol=10 * u.mas)
    assert all((sep > 10 * u.microarcsecond for sep in (sep_icrs, sep_hcrs, sep_helioecl)))
    assert_allclose(sep_cirs, 0.0 * u.deg, atol=1 * u.microarcsecond)

def test_tete_transforms():
    if False:
        for i in range(10):
            print('nop')
    '\n    We test the TETE transforms for proper behaviour here.\n\n    The TETE transforms are tested for accuracy against JPL Horizons in\n    test_solar_system.py. Here we are looking to check for consistency and\n    errors in the self transform.\n    '
    loc = EarthLocation.from_geodetic("-22°57'35.1", "-67°47'14.1", 5186 * u.m)
    time = Time('2020-04-06T00:00')
    (p, v) = loc.get_gcrs_posvel(time)
    gcrs_frame = GCRS(obstime=time, obsgeoloc=p, obsgeovel=v)
    moon = SkyCoord(169.24113968 * u.deg, 10.86086666 * u.deg, 358549.25381755 * u.km, frame=gcrs_frame)
    tete_frame = TETE(obstime=time, location=loc)
    tete_geo = TETE(obstime=time, location=EarthLocation(*[0, 0, 0] * u.km))
    tete_coo1 = moon.transform_to(tete_frame)
    tete_coo2 = moon.transform_to(tete_geo)
    assert_allclose(tete_coo1.separation_3d(tete_coo2), 0 * u.mm, atol=1 * u.mm)
    itrs1 = moon.transform_to(CIRS()).transform_to(ITRS())
    itrs2 = moon.transform_to(TETE()).transform_to(ITRS())
    assert_allclose(itrs1.separation_3d(itrs2), 0 * u.mm, atol=1 * u.mm)
    new_moon = moon.transform_to(TETE()).transform_to(moon)
    assert_allclose(new_moon.separation_3d(moon), 0 * u.mm, atol=1 * u.mm)
    tete_rt = tete_coo1.transform_to(ITRS(obstime=time)).transform_to(tete_coo1)
    assert_allclose(tete_rt.separation_3d(tete_coo1), 0 * u.mm, atol=1 * u.mm)

def test_straight_overhead():
    if False:
        return 10
    "\n    With a precise CIRS<->Observed transformation this should give Alt=90 exactly\n\n    If the CIRS self-transform breaks it won't, due to improper treatment of aberration\n    "
    t = Time('J2010')
    obj = EarthLocation(-1 * u.deg, 52 * u.deg, height=10.0 * u.km)
    home = EarthLocation(-1 * u.deg, 52 * u.deg, height=0.0 * u.km)
    cirs_geo = obj.get_itrs(t).transform_to(CIRS(obstime=t))
    obsrepr = home.get_itrs(t).transform_to(CIRS(obstime=t)).cartesian
    cirs_repr = cirs_geo.cartesian - obsrepr
    topocentric_cirs_frame = CIRS(obstime=t, location=home)
    cirs_topo = topocentric_cirs_frame.realize_frame(cirs_repr)
    aa = cirs_topo.transform_to(AltAz(obstime=t, location=home))
    assert_allclose(aa.alt, 90 * u.deg, atol=1 * u.uas, rtol=0)
    hd = cirs_topo.transform_to(HADec(obstime=t, location=home))
    assert_allclose(hd.ha, 0 * u.hourangle, atol=1 * u.uas, rtol=0)
    assert_allclose(hd.dec, 52 * u.deg, atol=1 * u.uas, rtol=0)

def test_itrs_straight_overhead():
    if False:
        while True:
            i = 10
    '\n    With a precise ITRS<->Observed transformation this should give Alt=90 exactly\n\n    '
    t = Time('J2010')
    obj = EarthLocation(-1 * u.deg, 52 * u.deg, height=10.0 * u.km)
    home = EarthLocation(-1 * u.deg, 52 * u.deg, height=0.0 * u.km)
    aa = obj.get_itrs(t, location=home).transform_to(AltAz(obstime=t, location=home))
    assert_allclose(aa.alt, 90 * u.deg, atol=1 * u.uas, rtol=0)
    hd = obj.get_itrs(t, location=home).transform_to(HADec(obstime=t, location=home))
    assert_allclose(hd.ha, 0 * u.hourangle, atol=1 * u.uas, rtol=0)
    assert_allclose(hd.dec, 52 * u.deg, atol=1 * u.uas, rtol=0)

def jplephem_ge(minversion):
    if False:
        for i in range(10):
            print('nop')
    'Check if jplephem is installed and has version >= minversion.'
    try:
        return HAS_JPLEPHEM and metadata.version('jplephem') >= minversion
    except Exception:
        return False

@pytest.mark.remote_data
@pytest.mark.skipif(not jplephem_ge('2.15'), reason='requires jplephem >= 2.15')
def test_aa_hd_high_precision():
    if False:
        return 10
    'These tests are provided by @mkbrewer - see issue #10356.\n\n    The code that produces them agrees very well (<0.5 mas) with SkyField once Polar motion\n    is turned off, but SkyField does not include polar motion, so a comparison to Skyfield\n    or JPL Horizons will be ~1" off.\n\n    The absence of polar motion within Skyfield and the disagreement between Skyfield and Horizons\n    make high precision comparisons to those codes difficult.\n\n    Updated 2020-11-29, after the comparison between codes became even better,\n    down to 100 nas.\n\n    Updated 2023-02-14, after IERS changes the IERS B format and analysis,\n    causing small deviations.\n\n    NOTE: the agreement reflects consistency in approach between two codes,\n    not necessarily absolute precision.  If this test starts failing, the\n    tolerance can and should be weakened *if* it is clear that the change is\n    due to an improvement (e.g., a new IAU precession model).\n\n    '
    lat = -22.959748 * u.deg
    lon = -67.78726 * u.deg
    elev = 5186 * u.m
    loc = EarthLocation.from_geodetic(lon, lat, elev)
    t = Time('2017-04-06T00:00:00.0', location=loc)
    with solar_system_ephemeris.set('de430'):
        moon = get_body('moon', t, loc)
        moon_aa = moon.transform_to(AltAz(obstime=t, location=loc))
        moon_hd = moon.transform_to(HADec(obstime=t, location=loc))
    (TARGET_AZ, TARGET_EL) = (15.032673662647138 * u.deg, 50.303110087520054 * u.deg)
    TARGET_DISTANCE = 376252.88325051306 * u.km
    assert_allclose(moon_aa.az, TARGET_AZ, atol=0.1 * u.uas, rtol=0)
    assert_allclose(moon_aa.alt, TARGET_EL, atol=0.1 * u.uas, rtol=0)
    assert_allclose(moon_aa.distance, TARGET_DISTANCE, atol=0.1 * u.mm, rtol=0)
    (ha, dec) = erfa.ae2hd(moon_aa.az.to_value(u.radian), moon_aa.alt.to_value(u.radian), lat.to_value(u.radian))
    ha = u.Quantity(ha, u.radian, copy=False)
    dec = u.Quantity(dec, u.radian, copy=False)
    assert_allclose(moon_hd.ha, ha, atol=0.1 * u.uas, rtol=0)
    assert_allclose(moon_hd.dec, dec, atol=0.1 * u.uas, rtol=0)

def test_aa_high_precision_nodata():
    if False:
        return 10
    '\n    These tests are designed to ensure high precision alt-az transforms.\n\n    They are a slight fudge since the target values come from astropy itself. They are generated\n    with a version of the code that passes the tests above, but for the internal solar system\n    ephemerides to avoid the use of remote data.\n    '
    (TARGET_AZ, TARGET_EL) = (15.0323151 * u.deg, 50.30271925 * u.deg)
    lat = -22.959748 * u.deg
    lon = -67.78726 * u.deg
    elev = 5186 * u.m
    loc = EarthLocation.from_geodetic(lon, lat, elev)
    t = Time('2017-04-06T00:00:00.0')
    moon = get_body('moon', t, loc)
    moon_aa = moon.transform_to(AltAz(obstime=t, location=loc))
    assert_allclose(moon_aa.az - TARGET_AZ, 0 * u.mas, atol=0.5 * u.mas)
    assert_allclose(moon_aa.alt - TARGET_EL, 0 * u.mas, atol=0.5 * u.mas)

class TestGetLocationGCRS:

    def setup_class(cls):
        if False:
            return 10
        cls.loc = loc = EarthLocation.from_geodetic(np.linspace(0, 360, 6) * u.deg, np.linspace(-90, 90, 6) * u.deg, 100 * u.m)
        cls.obstime = obstime = Time(np.linspace(2000, 2010, 6), format='jyear')
        loc_itrs = ITRS(loc.x, loc.y, loc.z, obstime=obstime)
        zeros = np.broadcast_to(0.0 * (u.km / u.s), (3,) + loc_itrs.shape, subok=True)
        loc_itrs.data.differentials['s'] = CartesianDifferential(zeros)
        loc_gcrs_cart = loc_itrs.transform_to(GCRS(obstime=obstime)).cartesian
        cls.obsgeoloc = loc_gcrs_cart.without_differentials()
        cls.obsgeovel = loc_gcrs_cart.differentials['s'].to_cartesian()

    def check_obsgeo(self, obsgeoloc, obsgeovel):
        if False:
            return 10
        assert_allclose(obsgeoloc.xyz, self.obsgeoloc.xyz, atol=0.1 * u.um, rtol=0.0)
        assert_allclose(obsgeovel.xyz, self.obsgeovel.xyz, atol=0.1 * u.mm / u.s, rtol=0.0)

    def test_get_gcrs_posvel(self):
        if False:
            print('Hello World!')
        self.check_obsgeo(*self.loc.get_gcrs_posvel(self.obstime))

    def test_tete_quick(self):
        if False:
            i = 10
            return i + 15
        rbpn = erfa.pnm06a(*get_jd12(self.obstime, 'tt'))
        loc_gcrs_frame = get_location_gcrs(self.loc, self.obstime, tete_to_itrs_mat(self.obstime, rbpn=rbpn), rbpn)
        self.check_obsgeo(loc_gcrs_frame.obsgeoloc, loc_gcrs_frame.obsgeovel)

    def test_cirs_quick(self):
        if False:
            i = 10
            return i + 15
        cirs_frame = CIRS(location=self.loc, obstime=self.obstime)
        pmat = gcrs_to_cirs_mat(cirs_frame.obstime)
        loc_gcrs_frame = get_location_gcrs(self.loc, self.obstime, cirs_to_itrs_mat(cirs_frame.obstime), pmat)
        self.check_obsgeo(loc_gcrs_frame.obsgeoloc, loc_gcrs_frame.obsgeovel)