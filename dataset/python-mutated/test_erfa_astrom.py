import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import CIRS, GCRS, AltAz, EarthLocation, SkyCoord
from astropy.coordinates.erfa_astrom import ErfaAstrom, ErfaAstromInterpolator, erfa_astrom
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

def test_science_state():
    if False:
        return 10
    assert erfa_astrom.get().__class__ is ErfaAstrom
    res = 300 * u.s
    with erfa_astrom.set(ErfaAstromInterpolator(res)):
        assert isinstance(erfa_astrom.get(), ErfaAstromInterpolator)
        assert erfa_astrom.get().mjd_resolution == res.to_value(u.day)
    assert erfa_astrom.get().__class__ is ErfaAstrom
    with pytest.raises(TypeError):
        erfa_astrom.set('foo')

def test_warnings():
    if False:
        print('Hello World!')
    with pytest.warns(AstropyWarning):
        with erfa_astrom.set(ErfaAstromInterpolator(9 * u.us)):
            pass

def test_erfa_astrom():
    if False:
        for i in range(10):
            print('nop')
    location = EarthLocation(lon=-17.891105 * u.deg, lat=28.761584 * u.deg, height=2200 * u.m)
    obstime = Time('2020-01-01T18:00') + np.linspace(0, 1, 100) * u.hour
    altaz = AltAz(location=location, obstime=obstime)
    coord = SkyCoord(ra=83.63308333, dec=22.0145, unit=u.deg)
    ref = coord.transform_to(altaz)
    with erfa_astrom.set(ErfaAstromInterpolator(300 * u.s)):
        interp_300s = coord.transform_to(altaz)
    assert np.any(ref.separation(interp_300s) > 0.005 * u.microarcsecond)
    assert np.all(ref.separation(interp_300s) < 1 * u.microarcsecond)

def test_interpolation_nd():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the interpolation also works for nd-arrays\n    '
    fact = EarthLocation(lon=-17.891105 * u.deg, lat=28.761584 * u.deg, height=2200 * u.m)
    interp_provider = ErfaAstromInterpolator(300 * u.s)
    provider = ErfaAstrom()
    for shape in [tuple(), (1,), (10,), (3, 2), (2, 10, 5), (4, 5, 3, 2)]:
        delta_t = np.linspace(0, 12, np.prod(shape, dtype=int)) * u.hour
        obstime = (Time('2020-01-01T18:00') + delta_t).reshape(shape)
        altaz = AltAz(location=fact, obstime=obstime)
        gcrs = GCRS(obstime=obstime)
        cirs = CIRS(obstime=obstime)
        for (frame, tcode) in zip([altaz, cirs, gcrs], ['apio', 'apco', 'apcs']):
            without_interp = getattr(provider, tcode)(frame)
            assert without_interp.shape == shape
            with_interp = getattr(interp_provider, tcode)(frame)
            assert with_interp.shape == shape

def test_interpolation_broadcasting():
    if False:
        i = 10
        return i + 15
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, golden_spiral_grid
    from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
    from astropy.time import Time
    rep = golden_spiral_grid(100)
    coord = SkyCoord(rep)
    times = Time('2020-01-01T20:00') + np.linspace(-0.5, 0.5, 30) * u.hour
    lst1 = EarthLocation(lon=-17.891498 * u.deg, lat=28.761443 * u.deg, height=2200 * u.m)
    aa_frame = AltAz(obstime=times[:, np.newaxis], location=lst1)
    aa_coord = coord.transform_to(aa_frame)
    with erfa_astrom.set(ErfaAstromInterpolator(300 * u.s)):
        aa_coord_interp = coord.transform_to(aa_frame)
    assert aa_coord.shape == aa_coord_interp.shape
    assert np.all(aa_coord.separation(aa_coord_interp) < 1 * u.microarcsecond)