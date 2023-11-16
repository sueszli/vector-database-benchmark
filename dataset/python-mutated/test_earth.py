"""Test initialization of angles not already covered by the API tests"""
import pickle
import numpy as np
import pytest
from astropy import constants
from astropy import units as u
from astropy.coordinates import Latitude, Longitude
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates.name_resolve import NameResolveError
from astropy.coordinates.representation.geodetic import ELLIPSOIDS
from astropy.time import Time
from astropy.units import allclose as quantity_allclose
from astropy.units.tests.test_quantity_erfa_ufuncs import vvd

def allclose_m14(a, b, rtol=1e-14, atol=None):
    if False:
        print('Hello World!')
    if atol is None:
        atol = 1e-14 * getattr(a, 'unit', 1)
    return quantity_allclose(a, b, rtol, atol)

def allclose_m8(a, b, rtol=1e-08, atol=None):
    if False:
        return 10
    if atol is None:
        atol = 1e-08 * getattr(a, 'unit', 1)
    return quantity_allclose(a, b, rtol, atol)

def isclose_m14(val, ref):
    if False:
        while True:
            i = 10
    return np.array([allclose_m14(v, r) for (v, r) in zip(val, ref)])

def isclose_m8(val, ref):
    if False:
        print('Hello World!')
    return np.array([allclose_m8(v, r) for (v, r) in zip(val, ref)])

def test_gc2gd():
    if False:
        return 10
    'Test that we reproduce erfa/src/t_erfa_c.c t_gc2gd'
    (x, y, z) = (2000000.0, 3000000.0, 5244000.0)
    status = 0
    location = EarthLocation.from_geocentric(x, y, z, u.m)
    (e, p, h) = location.to_geodetic('WGS84')
    (e, p, h) = (e.to(u.radian), p.to(u.radian), h.to(u.m))
    vvd(e, 0.982793723247329, 1e-14, 'eraGc2gd', 'e2', status)
    vvd(p, 0.9716018482060785, 1e-14, 'eraGc2gd', 'p2', status)
    vvd(h, 331.41731754844346, 1e-08, 'eraGc2gd', 'h2', status)
    (e, p, h) = location.to_geodetic('GRS80')
    (e, p, h) = (e.to(u.radian), p.to(u.radian), h.to(u.m))
    vvd(e, 0.982793723247329, 1e-14, 'eraGc2gd', 'e2', status)
    vvd(p, 0.9716018482060785, 1e-14, 'eraGc2gd', 'p2', status)
    vvd(h, 331.41731754844346, 1e-08, 'eraGc2gd', 'h2', status)
    (e, p, h) = location.to_geodetic('WGS72')
    (e, p, h) = (e.to(u.radian), p.to(u.radian), h.to(u.m))
    vvd(e, 0.982793723247329, 1e-14, 'eraGc2gd', 'e3', status)
    vvd(p, 0.9716018181101512, 1e-14, 'eraGc2gd', 'p3', status)
    vvd(h, 333.2770726130318, 1e-08, 'eraGc2gd', 'h3', status)

def test_gd2gc():
    if False:
        i = 10
        return i + 15
    'Test that we reproduce erfa/src/t_erfa_c.c t_gd2gc'
    e = 3.1 * u.rad
    p = -0.5 * u.rad
    h = 2500.0 * u.m
    status = 0
    location = EarthLocation.from_geodetic(e, p, h, ellipsoid='WGS84')
    xyz = tuple((v.to(u.m) for v in location.to_geocentric()))
    vvd(xyz[0], -5599000.557704994, 1e-07, 'eraGd2gc', '0/1', status)
    vvd(xyz[1], 233011.67223479203, 1e-07, 'eraGd2gc', '1/1', status)
    vvd(xyz[2], -3040909.470698336, 1e-07, 'eraGd2gc', '2/1', status)
    location = EarthLocation.from_geodetic(e, p, h, ellipsoid='GRS80')
    xyz = tuple((v.to(u.m) for v in location.to_geocentric()))
    vvd(xyz[0], -5599000.557726098, 1e-07, 'eraGd2gc', '0/2', status)
    vvd(xyz[1], 233011.6722356703, 1e-07, 'eraGd2gc', '1/2', status)
    vvd(xyz[2], -3040909.4706095476, 1e-07, 'eraGd2gc', '2/2', status)
    location = EarthLocation.from_geodetic(e, p, h, ellipsoid='WGS72')
    xyz = tuple((v.to(u.m) for v in location.to_geocentric()))
    vvd(xyz[0], -5598998.762630149, 1e-07, 'eraGd2gc', '0/3', status)
    vvd(xyz[1], 233011.5975297822, 1e-07, 'eraGd2gc', '1/3', status)
    vvd(xyz[2], -3040908.686146711, 1e-07, 'eraGd2gc', '2/3', status)

class TestInput:

    def setup_method(self):
        if False:
            return 10
        self.lon = Longitude([0.0, 45.0, 90.0, 135.0, 180.0, -180, -90, -45], u.deg, wrap_angle=180 * u.deg)
        self.lat = Latitude([+0.0, 30.0, 60.0, +90.0, -90.0, -60.0, -30.0, 0.0], u.deg)
        self.h = u.Quantity([0.1, 0.5, 1.0, -0.5, -1.0, +4.2, -11.0, -0.1], u.m)
        self.location = EarthLocation.from_geodetic(self.lon, self.lat, self.h)
        (self.x, self.y, self.z) = self.location.to_geocentric()

    def test_default_ellipsoid(self):
        if False:
            return 10
        assert self.location.ellipsoid == EarthLocation._ellipsoid

    def test_geo_attributes(self):
        if False:
            return 10
        assert all((np.all(_1 == _2) for (_1, _2) in zip(self.location.geodetic, self.location.to_geodetic())))
        assert all((np.all(_1 == _2) for (_1, _2) in zip(self.location.geocentric, self.location.to_geocentric())))

    def test_attribute_classes(self):
        if False:
            while True:
                i = 10
        'Test that attribute classes are correct (and not EarthLocation)'
        assert type(self.location.x) is u.Quantity
        assert type(self.location.y) is u.Quantity
        assert type(self.location.z) is u.Quantity
        assert type(self.location.lon) is Longitude
        assert type(self.location.lat) is Latitude
        assert type(self.location.height) is u.Quantity

    def test_input(self):
        if False:
            return 10
        'Check input is parsed correctly'
        geocentric = EarthLocation(self.x, self.y, self.z)
        assert np.all(geocentric == self.location)
        geocentric2 = EarthLocation(self.x.value, self.y.value, self.z.value, self.x.unit)
        assert np.all(geocentric2 == self.location)
        geodetic = EarthLocation(self.lon, self.lat, self.h)
        assert np.all(geodetic == self.location)
        geodetic2 = EarthLocation(self.lon.to_value(u.degree), self.lat.to_value(u.degree), self.h.to_value(u.m))
        assert np.all(geodetic2 == self.location)
        geodetic3 = EarthLocation(self.lon, self.lat)
        assert allclose_m14(geodetic3.lon.value, self.location.lon.value)
        assert allclose_m14(geodetic3.lat.value, self.location.lat.value)
        assert not np.any(isclose_m14(geodetic3.height.value, self.location.height.value))
        geodetic4 = EarthLocation(self.lon, self.lat, self.h[-1])
        assert allclose_m14(geodetic4.lon.value, self.location.lon.value)
        assert allclose_m14(geodetic4.lat.value, self.location.lat.value)
        assert allclose_m14(geodetic4.height[-1].value, self.location.height[-1].value)
        assert not np.any(isclose_m14(geodetic4.height[:-1].value, self.location.height[:-1].value))
        geocentric5 = EarthLocation(self.x, self.y, self.z, u.pc)
        assert geocentric5.unit is u.pc
        assert geocentric5.x.unit is u.pc
        assert geocentric5.height.unit is u.pc
        assert allclose_m14(geocentric5.x.to_value(self.x.unit), self.x.value)
        geodetic5 = EarthLocation(self.lon, self.lat, self.h.to(u.pc))
        assert geodetic5.unit is u.pc
        assert geodetic5.x.unit is u.pc
        assert geodetic5.height.unit is u.pc
        assert allclose_m14(geodetic5.x.to_value(self.x.unit), self.x.value)

    def test_invalid_input(self):
        if False:
            for i in range(10):
                print('nop')
        'Check invalid input raises exception'
        with pytest.raises(TypeError):
            EarthLocation(self.lon, self.y, self.z)
        with pytest.raises(u.UnitsError, match='should be in units of length'):
            EarthLocation.from_geocentric(self.lon, self.lat, self.lat)
        with pytest.raises(u.UnitsError, match='should all be consistent'):
            EarthLocation.from_geocentric(self.h, self.lon, self.lat)
        with pytest.raises(TypeError):
            EarthLocation.from_geocentric(self.x.value, self.y.value, self.z.value)
        with pytest.raises(ValueError):
            EarthLocation.from_geocentric(self.x, self.y, self.z[:5])
        with pytest.raises(u.UnitsError):
            EarthLocation.from_geodetic(self.x, self.y, self.z)
        with pytest.raises(ValueError):
            EarthLocation.from_geodetic(self.lon, self.lat, self.h[:5])

    def test_slicing(self):
        if False:
            print('Hello World!')
        locwgs72 = EarthLocation.from_geodetic(self.lon, self.lat, self.h, ellipsoid='WGS72')
        loc_slice1 = locwgs72[4]
        assert isinstance(loc_slice1, EarthLocation)
        assert loc_slice1.unit is locwgs72.unit
        assert loc_slice1.ellipsoid == locwgs72.ellipsoid == 'WGS72'
        assert not loc_slice1.shape
        with pytest.raises(TypeError):
            loc_slice1[0]
        with pytest.raises(IndexError):
            len(loc_slice1)
        loc_slice2 = locwgs72[4:6]
        assert isinstance(loc_slice2, EarthLocation)
        assert len(loc_slice2) == 2
        assert loc_slice2.unit is locwgs72.unit
        assert loc_slice2.ellipsoid == locwgs72.ellipsoid
        assert loc_slice2.shape == (2,)
        loc_x = locwgs72['x']
        assert type(loc_x) is u.Quantity
        assert loc_x.shape == locwgs72.shape
        assert loc_x.unit is locwgs72.unit

    def test_invalid_ellipsoid(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            EarthLocation.from_geodetic(self.lon, self.lat, self.h, ellipsoid='foo')
        with pytest.raises(TypeError):
            EarthLocation(self.lon, self.lat, self.h, ellipsoid='foo')
        with pytest.raises(ValueError):
            self.location.ellipsoid = 'foo'
        with pytest.raises(ValueError):
            self.location.to_geodetic('foo')

    @pytest.mark.parametrize('ellipsoid', ELLIPSOIDS)
    def test_ellipsoid(self, ellipsoid):
        if False:
            print('Hello World!')
        'Test that different ellipsoids are understood, and differ'
        (lon, lat, h) = self.location.to_geodetic(ellipsoid)
        if ellipsoid == self.location.ellipsoid:
            assert allclose_m8(h.value, self.h.value)
        else:
            assert not np.all(isclose_m8(h.value, self.h.value))
        location = EarthLocation.from_geodetic(self.lon, self.lat, self.h, ellipsoid=ellipsoid)
        if ellipsoid == self.location.ellipsoid:
            assert allclose_m14(location.z.value, self.z.value)
        else:
            assert not np.all(isclose_m14(location.z.value, self.z.value))

        def test_to_value(self):
            if False:
                while True:
                    i = 10
            loc = self.location
            loc_ndarray = loc.view(np.ndarray)
            assert np.all(loc.value == loc_ndarray)
            loc2 = self.location.to(u.km)
            loc2_ndarray = np.empty_like(loc_ndarray)
            for coo in ('x', 'y', 'z'):
                loc2_ndarray[coo] = loc_ndarray[coo] / 1000.0
            assert np.all(loc2.value == loc2_ndarray)
            loc2_value = self.location.to_value(u.km)
            assert np.all(loc2_value == loc2_ndarray)

def test_pickling():
    if False:
        return 10
    'Regression test against #4304.'
    el = EarthLocation(0.0 * u.m, 6000 * u.km, 6000 * u.km)
    s = pickle.dumps(el)
    el2 = pickle.loads(s)
    assert el == el2

def test_repr_latex():
    if False:
        return 10
    '\n    Regression test for issue #4542\n    '
    somelocation = EarthLocation(lon='149:3:57.9', lat='-31:16:37.3')
    somelocation._repr_latex_()
    somelocation2 = EarthLocation(lon=[1.0, 2.0] * u.deg, lat=[-1.0, 9.0] * u.deg)
    somelocation2._repr_latex_()

@pytest.mark.remote_data
@pytest.mark.parametrize('google_api_key', [None])
def test_of_address(google_api_key):
    if False:
        print('Hello World!')
    NYC_lon = -74.0 * u.deg
    NYC_lat = 40.7 * u.deg
    NYC_tol = 0.1 * u.deg
    try:
        loc = EarthLocation.of_address('New York, NY')
    except NameResolveError as e:
        if 'unknown failure with' not in str(e):
            pytest.xfail(str(e))
    else:
        assert quantity_allclose(loc.lat, NYC_lat, atol=NYC_tol)
        assert quantity_allclose(loc.lon, NYC_lon, atol=NYC_tol)
        assert np.allclose(loc.height.value, 0.0)
    with pytest.raises(NameResolveError):
        EarthLocation.of_address('lkjasdflkja')
    if google_api_key is not None:
        try:
            loc = EarthLocation.of_address('New York, NY', get_height=True)
        except NameResolveError as e:
            pytest.xfail(str(e.value))
        else:
            assert quantity_allclose(loc.lat, NYC_lat, atol=NYC_tol)
            assert quantity_allclose(loc.lon, NYC_lon, atol=NYC_tol)
            assert quantity_allclose(loc.height, 10.438 * u.meter, atol=1.0 * u.cm)

def test_geodetic_tuple():
    if False:
        i = 10
        return i + 15
    lat = 2 * u.deg
    lon = 10 * u.deg
    height = 100 * u.m
    el = EarthLocation.from_geodetic(lat=lat, lon=lon, height=height)
    res1 = el.to_geodetic()
    res2 = el.geodetic
    assert res1.lat == res2.lat and quantity_allclose(res1.lat, lat)
    assert res1.lon == res2.lon and quantity_allclose(res1.lon, lon)
    assert res1.height == res2.height and quantity_allclose(res1.height, height)

def test_gravitational_redshift():
    if False:
        for i in range(10):
            print('nop')
    someloc = EarthLocation(lon=-87.7 * u.deg, lat=37 * u.deg)
    sometime = Time('2017-8-21 18:26:40')
    zg0 = someloc.gravitational_redshift(sometime)
    zg_week = someloc.gravitational_redshift(sometime + 7 * u.day)
    assert 1.0 * u.mm / u.s < abs(zg_week - zg0) < 1 * u.cm / u.s
    zg_halfyear = someloc.gravitational_redshift(sometime + 0.5 * u.yr)
    assert 1 * u.cm / u.s < abs(zg_halfyear - zg0) < 1 * u.dm / u.s
    zg_year = someloc.gravitational_redshift(sometime - 20 * u.year)
    assert 0.1 * u.mm / u.s < abs(zg_year - zg0) < 1 * u.mm / u.s
    masses = {'sun': constants.G * constants.M_sun, 'jupiter': 0 * constants.G * u.kg, 'moon': 0 * constants.G * u.kg}
    zg_moonjup = someloc.gravitational_redshift(sometime, masses=masses)
    assert 0.1 * u.mm / u.s < abs(zg_moonjup - zg0) < 1 * u.mm / u.s
    assert zg_moonjup == someloc.gravitational_redshift(sometime, bodies=('sun',))
    assert zg_moonjup == someloc.gravitational_redshift(sometime, bodies=('earth', 'sun'))
    masses['earth'] = 0 * u.kg
    zg_moonjupearth = someloc.gravitational_redshift(sometime, masses=masses)
    assert 1 * u.dm / u.s < abs(zg_moonjupearth - zg0) < 1 * u.m / u.s
    masses['sun'] = 0 * u.kg
    assert someloc.gravitational_redshift(sometime, masses=masses) == 0
    with pytest.raises(KeyError):
        someloc.gravitational_redshift(sometime, bodies=('saturn',))
    with pytest.raises(u.UnitsError):
        masses = {'sun': constants.G * constants.M_sun, 'jupiter': constants.G * constants.M_jup, 'moon': 1 * u.km, 'earth': constants.G * constants.M_earth}
        someloc.gravitational_redshift(sometime, masses=masses)

def test_read_only_input():
    if False:
        return 10
    lon = np.array([80.0, 440.0]) * u.deg
    lat = np.array([45.0]) * u.deg
    lon.flags.writeable = lat.flags.writeable = False
    loc = EarthLocation.from_geodetic(lon=lon, lat=lat)
    assert quantity_allclose(loc[1].x, loc[0].x)

def test_info():
    if False:
        print('Hello World!')
    EarthLocation._get_site_registry(force_builtin=True)
    greenwich = EarthLocation.of_site('greenwich')
    assert str(greenwich.info).startswith('name = Royal Observatory Greenwich')