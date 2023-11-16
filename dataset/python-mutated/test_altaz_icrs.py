"""Accuracy tests for AltAz to ICRS coordinate transformations.

We use "known good" examples computed with other coordinate libraries.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz
from astropy.time import Time

def test_against_hor2eq():
    if False:
        return 10
    "Check that Astropy gives consistent results with an IDL hor2eq example.\n\n    See : http://idlastro.gsfc.nasa.gov/ftp/pro/astro/hor2eq.pro\n\n    Test is against these run outputs, run at 2000-01-01T12:00:00::\n\n      # NORMAL ATMOSPHERE CASE\n      IDL> hor2eq, ten(37,54,41), ten(264,55,06), 2451545.0d, ra, dec, /verb, obs='kpno', pres=781.0, temp=273.0\n      Latitude = +31 57 48.0   Longitude = *** 36 00.0\n      Julian Date =  2451545.000000\n      Az, El =  17 39 40.4  +37 54 41   (Observer Coords)\n      Az, El =  17 39 40.4  +37 53 40   (Apparent Coords)\n      LMST = +11 15 26.5\n      LAST = +11 15 25.7\n      Hour Angle = +03 38 30.1  (hh:mm:ss)\n      Ra, Dec:  07 36 55.6  +15 25 02   (Apparent Coords)\n      Ra, Dec:  07 36 55.2  +15 25 08   (J2000.0000)\n      Ra, Dec:  07 36 55.2  +15 25 08   (J2000)\n      IDL> print, ra, dec\n             114.23004       15.418818\n\n      # NO PRESSURE CASE\n      IDL> hor2eq, ten(37,54,41), ten(264,55,06), 2451545.0d, ra, dec, /verb, obs='kpno', pres=0.0, temp=273.0\n      Latitude = +31 57 48.0   Longitude = *** 36 00.0\n      Julian Date =  2451545.000000\n      Az, El =  17 39 40.4  +37 54 41   (Observer Coords)\n      Az, El =  17 39 40.4  +37 54 41   (Apparent Coords)\n      LMST = +11 15 26.5\n      LAST = +11 15 25.7\n      Hour Angle = +03 38 26.4  (hh:mm:ss)\n      Ra, Dec:  07 36 59.3  +15 25 31   (Apparent Coords)\n      Ra, Dec:  07 36 58.9  +15 25 37   (J2000.0000)\n      Ra, Dec:  07 36 58.9  +15 25 37   (J2000)\n      IDL> print, ra, dec\n             114.24554       15.427022\n    "
    location = EarthLocation(lon=Angle('-111d36.0m'), lat=Angle('31d57.8m'), height=2120.0 * u.m)
    obstime = Time(2451545.0, format='jd', scale='ut1')
    altaz_frame = AltAz(obstime=obstime, location=location, temperature=0 * u.deg_C, pressure=0.781 * u.bar)
    altaz_frame_noatm = AltAz(obstime=obstime, location=location, temperature=0 * u.deg_C, pressure=0.0 * u.bar)
    altaz = SkyCoord('264d55m06s 37d54m41s', frame=altaz_frame)
    altaz_noatm = SkyCoord('264d55m06s 37d54m41s', frame=altaz_frame_noatm)
    radec_frame = 'icrs'
    radec_actual = altaz.transform_to(radec_frame)
    radec_actual_noatm = altaz_noatm.transform_to(radec_frame)
    radec_expected = SkyCoord('07h36m55.2s +15d25m08s', frame=radec_frame)
    distance = radec_actual.separation(radec_expected).to('arcsec')
    radec_expected_noatm = SkyCoord('07h36m58.9s +15d25m37s', frame=radec_frame)
    distance_noatm = radec_actual_noatm.separation(radec_expected_noatm).to('arcsec')
    assert distance < 5 * u.arcsec
    assert distance_noatm < 0.4 * u.arcsec

def run_pyephem():
    if False:
        for i in range(10):
            print('nop')
    'Test run of pyephem, just in case the numbers below need to be reproduced.'
    import ephem
    observer = ephem.Observer()
    observer.lon = -1 * np.radians(109 + 24 / 60.0 + 53.1 / 60 ** 2)
    observer.lat = np.radians(33 + 41 / 60.0 + 46.0 / 60.0 ** 2)
    observer.elevation = 300
    observer.date = 2455822.868055556 - ephem.julian_date(0)
    (ra, dec) = observer.radec_of(np.radians(6.8927), np.radians(60.7665))
    print(f'EPHEM: {observer.date}: {np.degrees(ra)}, {np.degrees(dec)}')

def test_against_pyephem():
    if False:
        return 10
    'Check that Astropy gives consistent results with one PyEphem example.\n\n    PyEphem: https://rhodesmill.org/pyephem/\n\n    See example input and output here:\n    https://gist.github.com/zonca/1672906\n    https://github.com/phn/pytpm/issues/2#issuecomment-3698679\n    '
    obstime = Time('2011-09-18 08:50:00')
    location = EarthLocation(lon=Angle('-109d24m53.1s'), lat=Angle('33d41m46.0s'), height=300.0 * u.m)
    altaz_frame = AltAz(obstime=obstime, location=location, temperature=15 * u.deg_C, pressure=1.01 * u.bar)
    altaz = SkyCoord('6.8927d +60.7665d', frame=altaz_frame)
    radec_actual = altaz.transform_to('icrs')
    radec_expected = SkyCoord('27.107480889479397d +62.512687777362046d', frame='icrs')
    distance_ephem = radec_actual.separation(radec_expected).to('arcsec')
    assert distance_ephem < 3 * u.arcsec
    radec_expected = SkyCoord('27.10602683d +62.51275391d', frame='icrs')
    distance_astropy = radec_actual.separation(radec_expected).to('arcsec')
    assert distance_astropy < 0.1 * u.arcsec

def test_against_jpl_horizons():
    if False:
        return 10
    'Check that Astropy gives consistent results with the JPL Horizons example.\n\n    The input parameters and reference results are taken from this page:\n    (from the first row of the Results table at the bottom of that page)\n    http://ssd.jpl.nasa.gov/?horizons_tutorial\n    '
    obstime = Time('1998-07-28 03:00')
    location = EarthLocation(lon=Angle('248.405300d'), lat=Angle('31.9585d'), height=2.06 * u.km)
    altaz_frame = AltAz(obstime=obstime, location=location)
    altaz = SkyCoord('143.2970d 2.6223d', frame=altaz_frame)
    radec_actual = altaz.transform_to('icrs')
    radec_expected = SkyCoord('19h24m55.01s -40d56m28.9s', frame='icrs')
    distance = radec_actual.separation(radec_expected).to('arcsec')
    assert distance < 1 * u.arcsec

@pytest.mark.xfail(reason='Current output is completely incorrect')
def test_fk5_equinox_and_epoch_j2000_0_to_topocentric_observed():
    if False:
        while True:
            i = 10
    '\n    http://phn.github.io/pytpm/conversions.html#fk5-equinox-and-epoch-j2000-0-to-topocentric-observed\n    '
    location = EarthLocation(lon=Angle('-111.598333d'), lat=Angle('31.956389d'), height=2093.093 * u.m)
    obstime = Time('2010-01-01 12:00:00')
    altaz_frame = AltAz(obstime=obstime, location=location, temperature=0 * u.deg_C, pressure=0.781 * u.bar)
    radec = SkyCoord('12h22m54.899s 15d49m20.57s', frame='fk5')
    altaz_actual = radec.transform_to(altaz_frame)
    altaz_expected = SkyCoord('264d55m06s 37d54m41s', frame='altaz')
    distance = altaz_actual.separation(altaz_expected)
    'TODO: Current output is completely incorrect ... xfailing this test for now.\n\n    <SkyCoord (AltAz: obstime=2010-01-01 12:00:00.000, location=(-1994497.7199061865, -5037954.447348028, 3357437.2294832403) m, pressure=781.0 hPa, temperature=0.0 deg_C, relative_humidity=0, obswl=1.0 micron):00:00.000, location=(-1994497.7199061865, -5037954.447348028, 3357437.2294832403) m, pressure=781.0 hPa, temperature=0.0 deg_C, relative_humidity=0, obswl=1.0 micron): az=133.4869896371561 deg, alt=67.97857990957701 deg>\n    <SkyCoord (AltAz: obstime=None, location=None, pressure=0.0 hPa, temperature=0.0 deg_C, relative_humidity=0, obswl=1.0 micron): az=264.91833333333335 deg, alt=37.91138888888889 deg>\n    68d02m45.732s\n    '
    assert distance < 1 * u.arcsec