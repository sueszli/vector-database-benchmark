import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import builtin_frames as bf
from astropy.coordinates import galactocentric_frame_defaults
from astropy.coordinates import representation as r
from astropy.coordinates.builtin_frames import CIRS, ICRS, Galactic, Galactocentric
from astropy.coordinates.errors import ConvertError
from astropy.units import allclose as quantity_allclose
POSITION_ON_SKY = {'ra': 37.4 * u.deg, 'dec': -55.8 * u.deg}
DISTANCE = {'distance': 150 * u.pc}
PROPER_MOTION = {'pm_ra_cosdec': -21.2 * u.mas / u.yr, 'pm_dec': 17.1 * u.mas / u.yr}
RADIAL_VELOCITY = {'radial_velocity': 105.7 * u.km / u.s}
CARTESIAN_POSITION = {'x': 1 * u.pc, 'y': 2 * u.pc, 'z': 3 * u.pc, 'representation_type': r.CartesianRepresentation}
CARTESIAN_REPRESENTATION_KEYWORD_STR = {'representation_type': 'cartesian'}
CARTESIAN_VELOCITY = {'v_x': 1 * u.km / u.s, 'v_y': 2 * u.km / u.s, 'v_z': 3 * u.km / u.s, 'differential_type': r.CartesianDifferential}
CARTESIAN_DIFFERENTIAL_KEYWORD_STR = {'differential_type': 'cartesian'}

def test_api():
    if False:
        i = 10
        return i + 15
    with galactocentric_frame_defaults.set('latest'):
        icrs = ICRS(**POSITION_ON_SKY, **DISTANCE, **PROPER_MOTION, **RADIAL_VELOCITY)
        icrs.transform_to(Galactocentric())
        ICRS(**POSITION_ON_SKY, **PROPER_MOTION).transform_to(Galactic())

@pytest.mark.parametrize('kwargs', [POSITION_ON_SKY, {**POSITION_ON_SKY, **DISTANCE}, {**POSITION_ON_SKY, **PROPER_MOTION}, {**POSITION_ON_SKY, **DISTANCE, **PROPER_MOTION}, {**POSITION_ON_SKY, **RADIAL_VELOCITY}, {**POSITION_ON_SKY, **DISTANCE, **RADIAL_VELOCITY}, {**POSITION_ON_SKY, **PROPER_MOTION, **RADIAL_VELOCITY}, {**POSITION_ON_SKY, **DISTANCE, **PROPER_MOTION, **RADIAL_VELOCITY}, CARTESIAN_POSITION, {**CARTESIAN_POSITION, **CARTESIAN_REPRESENTATION_KEYWORD_STR}, {**CARTESIAN_POSITION, **CARTESIAN_VELOCITY}, {**CARTESIAN_POSITION, **CARTESIAN_VELOCITY, **CARTESIAN_DIFFERENTIAL_KEYWORD_STR}])
def test_all_arg_options(kwargs):
    if False:
        print('Hello World!')
    icrs = ICRS(**kwargs)
    gal = icrs.transform_to(Galactic())
    repr_gal = repr(gal)
    for k in kwargs:
        if k == 'differential_type':
            continue
        getattr(icrs, k)
    if 'pm_ra_cosdec' in kwargs:
        assert 'pm_l_cosb' in repr_gal
        assert 'pm_b' in repr_gal
        assert 'mas / yr' in repr_gal
        if 'radial_velocity' not in kwargs:
            assert 'radial_velocity' not in repr_gal
    if 'radial_velocity' in kwargs:
        assert 'radial_velocity' in repr_gal
        assert 'km / s' in repr_gal
        if 'pm_ra_cosdec' not in kwargs:
            assert 'pm_l_cosb' not in repr_gal
            assert 'pm_b' not in repr_gal

@pytest.mark.parametrize('cls,lon,lat', [[bf.ICRS, 'ra', 'dec'], [bf.FK4, 'ra', 'dec'], [bf.FK4NoETerms, 'ra', 'dec'], [bf.FK5, 'ra', 'dec'], [bf.GCRS, 'ra', 'dec'], [bf.HCRS, 'ra', 'dec'], [bf.LSR, 'ra', 'dec'], [bf.CIRS, 'ra', 'dec'], [bf.Galactic, 'l', 'b'], [bf.AltAz, 'az', 'alt'], [bf.Supergalactic, 'sgl', 'sgb'], [bf.GalacticLSR, 'l', 'b'], [bf.HeliocentricMeanEcliptic, 'lon', 'lat'], [bf.GeocentricMeanEcliptic, 'lon', 'lat'], [bf.BarycentricMeanEcliptic, 'lon', 'lat'], [bf.PrecessedGeocentric, 'ra', 'dec']])
def test_expected_arg_names(cls, lon, lat):
    if False:
        return 10
    kwargs = {lon: 37.4 * u.deg, lat: -55.8 * u.deg, f'pm_{lon}_cos{lat}': -21.2 * u.mas / u.yr, f'pm_{lat}': 17.1 * u.mas / u.yr}
    frame = cls(**kwargs, **DISTANCE, **RADIAL_VELOCITY)
_xhip_head = '\n------ ------------ ------------ -------- -------- ------------ ------------ ------- -------- -------- ------- ------ ------ ------\n       R            D            pmRA     pmDE                               Di      pmGLon   pmGLat   RV      U      V      W\nHIP    AJ2000 (deg) EJ2000 (deg) (mas/yr) (mas/yr) GLon (deg)   GLat (deg)   st (pc) (mas/yr) (mas/yr) (km/s)  (km/s) (km/s) (km/s)\n------ ------------ ------------ -------- -------- ------------ ------------ ------- -------- -------- ------- ------ ------ ------\n'.strip()
_xhip_data = '\n    19 000.05331690 +38.30408633    -3.17   -15.37 112.00026470 -23.47789171  247.12    -6.40   -14.33    6.30    7.3    2.0  -17.9\n    20 000.06295067 +23.52928427    36.11   -22.48 108.02779304 -37.85659811   95.90    29.35   -30.78   37.80  -19.3   16.1  -34.2\n    21 000.06623581 +08.00723430    61.48    -0.23 101.69697120 -52.74179515  183.68    58.06   -20.23  -11.72  -45.2  -30.9   -1.3\n 24917 080.09698238 -33.39874984    -4.30    13.40 236.92324669 -32.58047131  107.38   -14.03    -1.15   36.10  -22.4  -21.3  -19.9\n 59207 182.13915108 +65.34963517    18.17     5.49 130.04157185  51.18258601   56.00   -18.98    -0.49    5.70    1.5    6.1    4.4\n 87992 269.60730667 +36.87462906   -89.58    72.46  62.98053142  25.90148234  129.60    45.64   105.79   -4.00  -39.5  -15.8   56.7\n115110 349.72322473 -28.74087144    48.86    -9.25  23.00447250 -69.52799804  116.87    -8.37   -49.02   15.00  -16.8  -12.2  -23.6\n'.strip()

@pytest.mark.parametrize('hip,ra,dec,pmra,pmdec,glon,glat,dist,pmglon,pmglat,rv,U,V,W', [[float(val) for val in row.split()] for row in _xhip_data.split('\n')])
def test_xhip_galactic(hip, ra, dec, pmra, pmdec, glon, glat, dist, pmglon, pmglat, rv, U, V, W):
    if False:
        return 10
    i = ICRS(ra * u.deg, dec * u.deg, dist * u.pc, pm_ra_cosdec=pmra * u.marcsec / u.yr, pm_dec=pmdec * u.marcsec / u.yr, radial_velocity=rv * u.km / u.s)
    g = i.transform_to(Galactic())
    assert quantity_allclose(g.pm_l_cosb, pmglon * u.marcsec / u.yr, atol=0.01 * u.marcsec / u.yr)
    assert quantity_allclose(g.pm_b, pmglat * u.marcsec / u.yr, atol=0.01 * u.marcsec / u.yr)
    uvwg = g.cartesian.differentials['s']
    assert quantity_allclose(uvwg.d_x, U * u.km / u.s, atol=0.1 * u.km / u.s)
    assert quantity_allclose(uvwg.d_y, V * u.km / u.s, atol=0.1 * u.km / u.s)
    assert quantity_allclose(uvwg.d_z, W * u.km / u.s, atol=0.1 * u.km / u.s)

@pytest.mark.parametrize('kwargs,expect_success', ((POSITION_ON_SKY, False), ({**POSITION_ON_SKY, **DISTANCE}, True), ({**POSITION_ON_SKY, **PROPER_MOTION}, False), ({**POSITION_ON_SKY, **RADIAL_VELOCITY}, False), ({**POSITION_ON_SKY, **DISTANCE, **RADIAL_VELOCITY}, False), ({**POSITION_ON_SKY, **PROPER_MOTION, **RADIAL_VELOCITY}, False), ({**POSITION_ON_SKY, **DISTANCE, **PROPER_MOTION, **RADIAL_VELOCITY}, True)))
def test_frame_affinetransform(kwargs, expect_success):
    if False:
        for i in range(10):
            print('nop')
    'There are already tests in test_transformations.py that check that\n    an AffineTransform fails without full-space data, but this just checks that\n    things work as expected at the frame level as well.\n    '
    with galactocentric_frame_defaults.set('latest'):
        icrs = ICRS(**kwargs)
        if expect_success:
            _ = icrs.transform_to(Galactocentric())
        else:
            with pytest.raises(ConvertError):
                icrs.transform_to(Galactocentric())

def test_differential_type_arg():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test passing in an explicit differential class to the initializer or\n    changing the differential class via set_representation_cls\n    '
    icrs = ICRS(**POSITION_ON_SKY, pm_ra=10 * u.mas / u.yr, pm_dec=-11 * u.mas / u.yr, differential_type=r.UnitSphericalDifferential)
    assert icrs.pm_ra == 10 * u.mas / u.yr
    icrs = ICRS(**POSITION_ON_SKY, pm_ra=10 * u.mas / u.yr, pm_dec=-11 * u.mas / u.yr, differential_type={'s': r.UnitSphericalDifferential})
    assert icrs.pm_ra == 10 * u.mas / u.yr
    icrs = ICRS(ra=1 * u.deg, dec=60 * u.deg, pm_ra_cosdec=10 * u.mas / u.yr, pm_dec=-11 * u.mas / u.yr)
    icrs.set_representation_cls(s=r.UnitSphericalDifferential)
    assert quantity_allclose(icrs.pm_ra, 20 * u.mas / u.yr)
    with pytest.raises(TypeError):
        ICRS(**POSITION_ON_SKY, **CARTESIAN_VELOCITY)
    icrs = ICRS(**CARTESIAN_POSITION, **CARTESIAN_VELOCITY)
    assert icrs.x == 1 * u.pc
    assert icrs.y == 2 * u.pc
    assert icrs.z == 3 * u.pc
    assert icrs.v_x == 1 * u.km / u.s
    assert icrs.v_y == 2 * u.km / u.s
    assert icrs.v_z == 3 * u.km / u.s

def test_slicing_preserves_differential():
    if False:
        print('Hello World!')
    icrs = ICRS(**POSITION_ON_SKY, **DISTANCE, **PROPER_MOTION, **RADIAL_VELOCITY)
    icrs2 = icrs.reshape(1, 1)[:1, 0]
    for name in icrs.representation_component_names.keys():
        assert getattr(icrs, name) == getattr(icrs2, name)[0]
    for name in icrs.get_representation_component_names('s').keys():
        assert getattr(icrs, name) == getattr(icrs2, name)[0]

def test_shorthand_attributes():
    if False:
        return 10
    n = 4
    icrs1 = ICRS(ra=np.random.uniform(0, 360, n) * u.deg, dec=np.random.uniform(-90, 90, n) * u.deg, distance=100 * u.pc, pm_ra_cosdec=np.random.normal(0, 100, n) * u.mas / u.yr, pm_dec=np.random.normal(0, 100, n) * u.mas / u.yr, radial_velocity=np.random.normal(0, 100, n) * u.km / u.s)
    v = icrs1.velocity
    pm = icrs1.proper_motion
    assert quantity_allclose(pm[0], icrs1.pm_ra_cosdec)
    assert quantity_allclose(pm[1], icrs1.pm_dec)
    icrs2 = ICRS(**POSITION_ON_SKY, **DISTANCE, **PROPER_MOTION, **RADIAL_VELOCITY)
    v = icrs2.velocity
    pm = icrs2.proper_motion
    assert quantity_allclose(pm[0], icrs2.pm_ra_cosdec)
    assert quantity_allclose(pm[1], icrs2.pm_dec)
    icrs3 = ICRS(**POSITION_ON_SKY, **PROPER_MOTION, **RADIAL_VELOCITY)
    with pytest.raises(ValueError):
        icrs3.velocity
    icrs3.set_representation_cls('cartesian')
    assert hasattr(icrs3, 'radial_velocity')
    assert quantity_allclose(icrs3.radial_velocity, 105.7 * u.km / u.s)
    icrs4 = ICRS(**CARTESIAN_POSITION, **CARTESIAN_VELOCITY)
    icrs4.radial_velocity

@pytest.mark.parametrize('icrs_coords', [POSITION_ON_SKY, {**POSITION_ON_SKY, **PROPER_MOTION}])
def test_negative_distance(icrs_coords):
    if False:
        print('Hello World!')
    'Regression test: #7408\n    Make sure that negative parallaxes turned into distances are handled right\n    '
    c = ICRS(distance=(-10 * u.mas).to(u.pc, u.parallax()), **icrs_coords)
    assert quantity_allclose(c.ra, 37.4 * u.deg)
    assert quantity_allclose(c.dec, -55.8 * u.deg)

def test_velocity_units():
    if False:
        while True:
            i = 10
    'Check that the differential data given has compatible units\n    with the time-derivative of representation data'
    with pytest.raises(ValueError, match='^x has unit "pc" with physical type "length", but v_x has incompatible unit "" with physical type "dimensionless" instead of the expected "speed/velocity".$'):
        ICRS(**CARTESIAN_POSITION, v_x=1, v_y=2, v_z=3, differential_type='cartesian')

def test_frame_with_velocity_without_distance_can_be_transformed():
    if False:
        return 10
    rep = CIRS(**POSITION_ON_SKY, **PROPER_MOTION).transform_to(ICRS())
    assert '<ICRS Coordinate: (ra, dec, distance) in' in repr(rep)