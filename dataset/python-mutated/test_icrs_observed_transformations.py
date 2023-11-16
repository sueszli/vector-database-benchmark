"""Accuracy tests for ICRS transformations, primarily to/from AltAz.

"""
import numpy as np
from astropy import units as u
from astropy.coordinates import CIRS, ICRS, AltAz, EarthLocation, HADec, SkyCoord, frame_transform_graph, golden_spiral_grid
from astropy.tests.helper import assert_quantity_allclose as assert_allclose
from astropy.time import Time

def test_icrs_altaz_consistency():
    if False:
        i = 10
        return i + 15
    '\n    Check ICRS<->AltAz for consistency with ICRS<->CIRS<->AltAz\n\n    The latter is extensively tested in test_intermediate_transformations.py\n    '
    usph = golden_spiral_grid(200)
    dist = np.linspace(0.5, 1, len(usph)) * u.km * 100000.0
    icoo = SkyCoord(ra=usph.lon, dec=usph.lat, distance=dist)
    observer = EarthLocation(28 * u.deg, 23 * u.deg, height=2000.0 * u.km)
    obstime = Time('J2010')
    aa_frame = AltAz(obstime=obstime, location=observer)
    trans = frame_transform_graph.get_transform(ICRS, AltAz).transforms
    assert len(trans) == 1
    aa1 = icoo.transform_to(aa_frame)
    aa2 = icoo.transform_to(CIRS()).transform_to(aa_frame)
    assert_allclose(aa1.separation_3d(aa2), 0 * u.mm, atol=1 * u.mm)
    roundtrip = icoo.transform_to(aa_frame).transform_to(icoo)
    assert_allclose(roundtrip.separation_3d(icoo), 0 * u.mm, atol=1 * u.mm)
    roundtrip = icoo.transform_to(aa_frame).transform_to(CIRS()).transform_to(icoo)
    assert_allclose(roundtrip.separation_3d(icoo), 0 * u.mm, atol=1 * u.mm)

def test_icrs_hadec_consistency():
    if False:
        print('Hello World!')
    '\n    Check ICRS<->HADec for consistency with ICRS<->CIRS<->HADec\n    '
    usph = golden_spiral_grid(200)
    dist = np.linspace(0.5, 1, len(usph)) * u.km * 100000.0
    icoo = SkyCoord(ra=usph.lon, dec=usph.lat, distance=dist)
    observer = EarthLocation(28 * u.deg, 23 * u.deg, height=2000.0 * u.km)
    obstime = Time('J2010')
    hd_frame = HADec(obstime=obstime, location=observer)
    trans = frame_transform_graph.get_transform(ICRS, HADec).transforms
    assert len(trans) == 1
    aa1 = icoo.transform_to(hd_frame)
    aa2 = icoo.transform_to(CIRS()).transform_to(hd_frame)
    assert_allclose(aa1.separation_3d(aa2), 0 * u.mm, atol=1 * u.mm)
    roundtrip = icoo.transform_to(hd_frame).transform_to(icoo)
    assert_allclose(roundtrip.separation_3d(icoo), 0 * u.mm, atol=1 * u.mm)
    roundtrip = icoo.transform_to(hd_frame).transform_to(CIRS()).transform_to(icoo)
    assert_allclose(roundtrip.separation_3d(icoo), 0 * u.mm, atol=1 * u.mm)