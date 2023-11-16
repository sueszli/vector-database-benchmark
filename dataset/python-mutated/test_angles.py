"""Test initialization and other aspects of Angle and subclasses"""
import pickle
import threading
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import astropy.units as u
from astropy.coordinates import Angle, IllegalHourError, IllegalMinuteError, IllegalMinuteWarning, IllegalSecondError, IllegalSecondWarning, Latitude, Longitude
from astropy.utils.compat.numpycompat import NUMPY_LT_2_0

def test_create_angles():
    if False:
        i = 10
        return i + 15
    '\n    Tests creating and accessing Angle objects\n    '
    ' The "angle" is a fundamental object. The internal\n    representation is stored in radians, but this is transparent to the user.\n    Units *must* be specified rather than a default value be assumed. This is\n    as much for self-documenting code as anything else.\n\n    Angle objects simply represent a single angular coordinate. More specific\n    angular coordinates (e.g. Longitude, Latitude) are subclasses of Angle.'
    a1 = Angle(54.12412, unit=u.degree)
    a2 = Angle('54.12412', unit=u.degree)
    a3 = Angle('54:07:26.832', unit=u.degree)
    a4 = Angle('54.12412 deg')
    a5 = Angle('54.12412 degrees')
    a6 = Angle('54.12412°')
    a8 = Angle('54°07\'26.832"')
    a9 = Angle([54, 7, 26.832], unit=u.degree)
    assert_allclose(a9.value, [54, 7, 26.832])
    assert a9.unit is u.degree
    a10 = Angle(3.60827466667, unit=u.hour)
    a11 = Angle('3:36:29.7888000120', unit=u.hour)
    Angle(0.944644098745, unit=u.radian)
    with pytest.raises(u.UnitsError):
        Angle(54.12412)
    with pytest.raises(u.UnitsError):
        Angle(54.12412, unit=u.m)
    with pytest.raises(ValueError):
        Angle(12.34, unit='not a unit')
    a14 = Angle('03h36m29.7888000120')
    a15 = Angle('5h4m3s')
    assert a15.unit == u.hourangle
    a16 = Angle('1 d')
    a17 = Angle('1 degree')
    assert a16.degree == 1
    assert a17.degree == 1
    a18 = Angle('54 07.4472', unit=u.degree)
    a19 = Angle('54:07.4472', unit=u.degree)
    a20 = Angle('54d07.4472m', unit=u.degree)
    a21 = Angle('3h36m', unit=u.hour)
    a22 = Angle('3.6h', unit=u.hour)
    a23 = Angle('- 3h', unit=u.hour)
    a24 = Angle('+ 3h', unit=u.hour)
    a25 = Angle(3.0, unit=u.hour ** 1)
    assert a1 == a2 == a3 == a4 == a5 == a6 == a8 == a18 == a19 == a20
    assert_allclose(a1.radian, a2.radian)
    assert_allclose(a2.degree, a3.degree)
    assert_allclose(a3.radian, a4.radian)
    assert_allclose(a4.radian, a5.radian)
    assert_allclose(a5.radian, a6.radian)
    assert_allclose(a10.degree, a11.degree)
    assert a11 == a14
    assert a21 == a22
    assert a23 == -a24
    assert a24 == a25
    with pytest.raises(IllegalSecondError):
        Angle('12 32 99', unit=u.degree)
    with pytest.raises(IllegalMinuteError):
        Angle('12 99 23', unit=u.degree)
    with pytest.raises(IllegalSecondError):
        Angle('12 32 99', unit=u.hour)
    with pytest.raises(IllegalMinuteError):
        Angle('12 99 23', unit=u.hour)
    with pytest.raises(IllegalHourError):
        Angle('99 25 51.0', unit=u.hour)
    with pytest.raises(ValueError):
        Angle('12 25 51.0xxx', unit=u.hour)
    with pytest.raises(ValueError):
        Angle('12h34321m32.2s')
    assert a1 is not None

def test_angle_from_view():
    if False:
        return 10
    q = np.arange(3.0) * u.deg
    a = q.view(Angle)
    assert type(a) is Angle
    assert a.unit is q.unit
    assert np.all(a == q)
    q2 = np.arange(4) * u.m
    with pytest.raises(u.UnitTypeError):
        q2.view(Angle)

def test_angle_ops():
    if False:
        print('Hello World!')
    '\n    Tests operations on Angle objects\n    '
    a1 = Angle(3.60827466667, unit=u.hour)
    a2 = Angle('54:07:26.832', unit=u.degree)
    a1 + a2
    a1 - a2
    -a1
    assert_allclose((a1 * 2).hour, 2 * 3.6082746666700003)
    assert abs((a1 / 3.123456).hour - 3.60827466667 / 3.123456) < 1e-10
    assert (2 * a1).hour == (a1 * 2).hour
    a3 = Angle(a1)
    assert_allclose(a1.radian, a3.radian)
    assert a1 is not a3
    a4 = abs(-a1)
    assert a4.radian == a1.radian
    a5 = Angle(5.0, unit=u.hour)
    assert a5 > a1
    assert a5 >= a1
    assert a1 < a5
    assert a1 <= a5
    a6 = Angle(45.0, u.degree)
    a7 = a6 * a5
    assert type(a7) is u.Quantity
    a8 = a1 + 1.0 * u.deg
    assert type(a8) is Angle
    a9 = 1.0 * u.deg + a1
    assert type(a9) is Angle
    with pytest.raises(TypeError):
        a6 *= a5
    with pytest.raises(TypeError):
        a6 *= u.m
    with pytest.raises(TypeError):
        np.sin(a6, out=a6)

def test_angle_methods():
    if False:
        while True:
            i = 10
    a = Angle([0.0, 2.0], 'deg')
    a_mean = a.mean()
    assert type(a_mean) is Angle
    assert a_mean == 1.0 * u.degree
    a_std = a.std()
    assert type(a_std) is Angle
    assert a_std == 1.0 * u.degree
    a_var = a.var()
    assert type(a_var) is u.Quantity
    assert a_var == 1.0 * u.degree ** 2
    if NUMPY_LT_2_0:
        a_ptp = a.ptp()
        assert type(a_ptp) is Angle
        assert a_ptp == 2.0 * u.degree
    a_max = a.max()
    assert type(a_max) is Angle
    assert a_max == 2.0 * u.degree
    a_min = a.min()
    assert type(a_min) is Angle
    assert a_min == 0.0 * u.degree

def test_angle_convert():
    if False:
        print('Hello World!')
    '\n    Test unit conversion of Angle objects\n    '
    angle = Angle('54.12412', unit=u.degree)
    assert_allclose(angle.hour, 3.60827466667)
    assert_allclose(angle.radian, 0.944644098745)
    assert_allclose(angle.degree, 54.12412)
    assert len(angle.hms) == 3
    assert isinstance(angle.hms, tuple)
    assert angle.hms[0] == 3
    assert angle.hms[1] == 36
    assert_allclose(angle.hms[2], 29.78879999999947)
    assert angle.hms.h == 3
    assert angle.hms.m == 36
    assert_allclose(angle.hms.s, 29.78879999999947)
    assert len(angle.dms) == 3
    assert isinstance(angle.dms, tuple)
    assert angle.dms[0] == 54
    assert angle.dms[1] == 7
    assert_allclose(angle.dms[2], 26.831999999992036)
    assert angle.dms.d == 54
    assert angle.dms.m == 7
    assert_allclose(angle.dms.s, 26.831999999992036)
    assert isinstance(angle.dms[0], float)
    assert isinstance(angle.hms[0], float)
    negangle = Angle('-54.12412', unit=u.degree)
    assert negangle.dms.d == -54
    assert negangle.dms.m == -7
    assert_allclose(negangle.dms.s, -26.831999999992036)
    assert negangle.signed_dms.sign == -1
    assert negangle.signed_dms.d == 54
    assert negangle.signed_dms.m == 7
    assert_allclose(negangle.signed_dms.s, 26.831999999992036)

def test_angle_formatting():
    if False:
        return 10
    '\n    Tests string formatting for Angle objects\n    '
    '\n    The string method of Angle has this signature:\n    def string(self, unit=DEGREE, decimal=False, sep=" ", precision=5,\n               pad=False):\n\n    The "decimal" parameter defaults to False since if you need to print the\n    Angle as a decimal, there\'s no need to use the "format" method (see\n    above).\n    '
    angle = Angle('54.12412', unit=u.degree)
    assert str(angle) == angle.to_string()
    res = 'Angle as HMS: 3h36m29.7888s'
    assert f'Angle as HMS: {angle.to_string(unit=u.hour)}' == res
    res = 'Angle as HMS: 3:36:29.7888'
    assert f"Angle as HMS: {angle.to_string(unit=u.hour, sep=':')}" == res
    res = 'Angle as HMS: 3:36:29.79'
    assert f"Angle as HMS: {angle.to_string(unit=u.hour, sep=':', precision=2)}" == res
    res = 'Angle as HMS: 3h36m29.7888s'
    assert f"Angle as HMS: {angle.to_string(unit=u.hour, sep=('h', 'm', 's'), precision=4)}" == res
    res = 'Angle as HMS: 3-36|29.7888'
    assert f"Angle as HMS: {angle.to_string(unit=u.hour, sep=['-', '|'], precision=4)}" == res
    res = 'Angle as HMS: 3-36-29.7888'
    assert f"Angle as HMS: {angle.to_string(unit=u.hour, sep='-', precision=4)}" == res
    res = 'Angle as HMS: 03h36m29.7888s'
    assert f'Angle as HMS: {angle.to_string(unit=u.hour, precision=4, pad=True)}' == res
    angle = Angle('3 36 29.78880', unit=u.degree)
    res = 'Angle as DMS: 3d36m29.7888s'
    assert f'Angle as DMS: {angle.to_string(unit=u.degree)}' == res
    res = 'Angle as DMS: 3:36:29.7888'
    assert f"Angle as DMS: {angle.to_string(unit=u.degree, sep=':')}" == res
    res = 'Angle as DMS: 3:36:29.79'
    assert f"Angle as DMS: {angle.to_string(unit=u.degree, sep=':', precision=2)}" == res
    res = 'Angle as DMS: 3d36m29.7888s'
    assert f"Angle as DMS: {angle.to_string(unit=u.deg, sep=('d', 'm', 's'), precision=4)}" == res
    res = 'Angle as DMS: 3-36|29.7888'
    assert f"Angle as DMS: {angle.to_string(unit=u.degree, sep=['-', '|'], precision=4)}" == res
    res = 'Angle as DMS: 3-36-29.7888'
    assert f"Angle as DMS: {angle.to_string(unit=u.degree, sep='-', precision=4)}" == res
    res = 'Angle as DMS: 03d36m29.7888s'
    assert f'Angle as DMS: {angle.to_string(unit=u.degree, precision=4, pad=True)}' == res
    res = 'Angle as rad: 0.0629763 rad'
    assert f'Angle as rad: {angle.to_string(unit=u.radian)}' == res
    res = 'Angle as rad decimal: 0.0629763'
    assert f'Angle as rad decimal: {angle.to_string(unit=u.radian, decimal=True)}' == res
    angle = Angle(-1.23456789, unit=u.degree)
    angle2 = Angle(-1.23456789, unit=u.hour)
    assert angle.to_string() == '-1d14m04.444404s'
    assert angle.to_string(pad=True) == '-01d14m04.444404s'
    assert angle.to_string(unit=u.hour) == '-0h04m56.2962936s'
    assert angle2.to_string(unit=u.hour, pad=True) == '-01h14m04.444404s'
    assert angle.to_string(unit=u.radian, decimal=True) == '-0.0215473'
    assert angle.to_string(unit=u.hour ** 1) == '-0h04m56.2962936s'

def test_to_string_vector():
    if False:
        for i in range(10):
            print('nop')
    assert Angle([1.0 / 7.0, 1.0 / 7.0], unit='deg').to_string()[0] == '0d08m34.28571429s'
    assert Angle([1.0 / 7.0], unit='deg').to_string()[0] == '0d08m34.28571429s'
    assert Angle(1.0 / 7.0, unit='deg').to_string() == '0d08m34.28571429s'

def test_angle_format_roundtripping():
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensures that the string representation of an angle can be used to create a\n    new valid Angle.\n    '
    a1 = Angle(0, unit=u.radian)
    a2 = Angle(10, unit=u.degree)
    a3 = Angle(0.543, unit=u.degree)
    a4 = Angle('1d2m3.4s')
    assert Angle(str(a1)).degree == a1.degree
    assert Angle(str(a2)).degree == a2.degree
    assert Angle(str(a3)).degree == a3.degree
    assert Angle(str(a4)).degree == a4.degree
    ra = Longitude('1h2m3.4s')
    dec = Latitude('1d2m3.4s')
    assert_allclose(Angle(str(ra)).degree, ra.degree)
    assert_allclose(Angle(str(dec)).degree, dec.degree)

def test_radec():
    if False:
        print('Hello World!')
    '\n    Tests creation/operations of Longitude and Latitude objects\n    '
    '\n    Longitude and Latitude are objects that are subclassed from Angle. As with Angle, Longitude\n    and Latitude can parse any unambiguous format (tuples, formatted strings, etc.).\n\n    The intention is not to create an Angle subclass for every possible\n    coordinate object (e.g. galactic l, galactic b). However, equatorial Longitude/Latitude\n    are so prevalent in astronomy that it\'s worth creating ones for these\n    units. They will be noted as "special" in the docs and use of the just the\n    Angle class is to be used for other coordinate systems.\n    '
    with pytest.raises(u.UnitsError):
        Longitude('4:08:15.162342')
    with pytest.raises(u.UnitsError):
        Longitude('-4:08:15.162342')
    with pytest.raises(u.UnitsError):
        Longitude('26:34:15.345634')
    with pytest.raises(u.UnitsError):
        Longitude(68)
    with pytest.raises(u.UnitsError):
        Longitude(12)
    with pytest.raises(ValueError):
        Longitude('garbage containing a d and no units')
    ra = Longitude('12h43m23s')
    assert_allclose(ra.hour, 12.7230555556)
    ra = Longitude('4:08:15.162342', unit=u.hour)
    with pytest.raises(u.UnitsError):
        Latitude('-41:08:15.162342')
    dec = Latitude('-41:08:15.162342', unit=u.degree)

def test_negative_zero_dms():
    if False:
        i = 10
        return i + 15
    a = Angle('-00:00:10', u.deg)
    assert_allclose(a.degree, -10.0 / 3600.0)
    a = Angle('−00:00:10', u.deg)
    assert_allclose(a.degree, -10.0 / 3600.0)

def test_negative_zero_dm():
    if False:
        i = 10
        return i + 15
    a = Angle('-00:10', u.deg)
    assert_allclose(a.degree, -10.0 / 60.0)

def test_negative_zero_hms():
    if False:
        return 10
    a = Angle('-00:00:10', u.hour)
    assert_allclose(a.hour, -10.0 / 3600.0)

def test_negative_zero_hm():
    if False:
        print('Hello World!')
    a = Angle('-00:10', u.hour)
    assert_allclose(a.hour, -10.0 / 60.0)

def test_negative_sixty_hm():
    if False:
        while True:
            i = 10
    with pytest.warns(IllegalMinuteWarning):
        a = Angle('-00:60', u.hour)
    assert_allclose(a.hour, -1.0)

def test_plus_sixty_hm():
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(IllegalMinuteWarning):
        a = Angle('00:60', u.hour)
    assert_allclose(a.hour, 1.0)

def test_negative_fifty_nine_sixty_dms():
    if False:
        return 10
    with pytest.warns(IllegalSecondWarning):
        a = Angle('-00:59:60', u.deg)
    assert_allclose(a.degree, -1.0)

def test_plus_fifty_nine_sixty_dms():
    if False:
        i = 10
        return i + 15
    with pytest.warns(IllegalSecondWarning):
        a = Angle('+00:59:60', u.deg)
    assert_allclose(a.degree, 1.0)

def test_negative_sixty_dms():
    if False:
        i = 10
        return i + 15
    with pytest.warns(IllegalSecondWarning):
        a = Angle('-00:00:60', u.deg)
    assert_allclose(a.degree, -1.0 / 60.0)

def test_plus_sixty_dms():
    if False:
        print('Hello World!')
    with pytest.warns(IllegalSecondWarning):
        a = Angle('+00:00:60', u.deg)
    assert_allclose(a.degree, 1.0 / 60.0)

def test_angle_to_is_angle():
    if False:
        print('Hello World!')
    with pytest.warns(IllegalSecondWarning):
        a = Angle('00:00:60', u.deg)
    assert isinstance(a, Angle)
    assert isinstance(a.to(u.rad), Angle)

def test_angle_to_quantity():
    if False:
        return 10
    with pytest.warns(IllegalSecondWarning):
        a = Angle('00:00:60', u.deg)
    q = u.Quantity(a)
    assert isinstance(q, u.Quantity)
    assert q.unit is u.deg

def test_quantity_to_angle():
    if False:
        i = 10
        return i + 15
    a = Angle(1.0 * u.deg)
    assert isinstance(a, Angle)
    with pytest.raises(u.UnitsError):
        Angle(1.0 * u.meter)
    a = Angle(1.0 * u.hour)
    assert isinstance(a, Angle)
    assert a.unit is u.hourangle
    with pytest.raises(u.UnitsError):
        Angle(1.0 * u.min)

def test_angle_string():
    if False:
        return 10
    with pytest.warns(IllegalSecondWarning):
        a = Angle('00:00:60', u.deg)
    assert str(a) == '0d01m00s'
    a = Angle('00:00:59S', u.deg)
    assert str(a) == '-0d00m59s'
    a = Angle('00:00:59N', u.deg)
    assert str(a) == '0d00m59s'
    a = Angle('00:00:59E', u.deg)
    assert str(a) == '0d00m59s'
    a = Angle('00:00:59W', u.deg)
    assert str(a) == '-0d00m59s'
    a = Angle('-00:00:10', u.hour)
    assert str(a) == '-0h00m10s'
    a = Angle('00:00:59E', u.hour)
    assert str(a) == '0h00m59s'
    a = Angle('00:00:59W', u.hour)
    assert str(a) == '-0h00m59s'
    a = Angle(3.2, u.radian)
    assert str(a) == '3.2 rad'
    a = Angle(4.2, u.microarcsecond)
    assert str(a) == '4.2 uarcsec'
    a = Angle('1.0uarcsec')
    assert a.value == 1.0
    assert a.unit == u.microarcsecond
    a = Angle('1.0uarcsecN')
    assert a.value == 1.0
    assert a.unit == u.microarcsecond
    a = Angle('1.0uarcsecS')
    assert a.value == -1.0
    assert a.unit == u.microarcsecond
    a = Angle('1.0uarcsecE')
    assert a.value == 1.0
    assert a.unit == u.microarcsecond
    a = Angle('1.0uarcsecW')
    assert a.value == -1.0
    assert a.unit == u.microarcsecond
    a = Angle('3d')
    assert_allclose(a.value, 3.0)
    assert a.unit == u.degree
    a = Angle('3dN')
    assert str(a) == '3d00m00s'
    assert a.unit == u.degree
    a = Angle('3dS')
    assert str(a) == '-3d00m00s'
    assert a.unit == u.degree
    a = Angle('3dE')
    assert str(a) == '3d00m00s'
    assert a.unit == u.degree
    a = Angle('3dW')
    assert str(a) == '-3d00m00s'
    assert a.unit == u.degree
    a = Angle('10"')
    assert_allclose(a.value, 10.0)
    assert a.unit == u.arcsecond
    a = Angle("10'N")
    assert_allclose(a.value, 10.0)
    assert a.unit == u.arcminute
    a = Angle("10'S")
    assert_allclose(a.value, -10.0)
    assert a.unit == u.arcminute
    a = Angle("10'E")
    assert_allclose(a.value, 10.0)
    assert a.unit == u.arcminute
    a = Angle("10'W")
    assert_allclose(a.value, -10.0)
    assert a.unit == u.arcminute
    a = Angle('45°55′12″N')
    assert str(a) == '45d55m12s'
    assert_allclose(a.value, 45.92)
    assert a.unit == u.deg
    a = Angle('45°55′12″S')
    assert str(a) == '-45d55m12s'
    assert_allclose(a.value, -45.92)
    assert a.unit == u.deg
    a = Angle('45°55′12″E')
    assert str(a) == '45d55m12s'
    assert_allclose(a.value, 45.92)
    assert a.unit == u.deg
    a = Angle('45°55′12″W')
    assert str(a) == '-45d55m12s'
    assert_allclose(a.value, -45.92)
    assert a.unit == u.deg
    with pytest.raises(ValueError):
        Angle('00h00m10sN')
    with pytest.raises(ValueError):
        Angle('45°55′12″NS')

def test_angle_repr():
    if False:
        for i in range(10):
            print('nop')
    assert 'Angle' in repr(Angle(0, u.deg))
    assert 'Longitude' in repr(Longitude(0, u.deg))
    assert 'Latitude' in repr(Latitude(0, u.deg))
    a = Angle(0, u.deg)
    repr(a)

def test_large_angle_representation():
    if False:
        while True:
            i = 10
    'Test that angles above 360 degrees can be output as strings,\n    in repr, str, and to_string.  (regression test for #1413)'
    a = Angle(350, u.deg) + Angle(350, u.deg)
    a.to_string()
    a.to_string(u.hourangle)
    repr(a)
    repr(a.to(u.hourangle))
    str(a)
    str(a.to(u.hourangle))

def test_wrap_at_inplace():
    if False:
        for i in range(10):
            print('nop')
    a = Angle([-20, 150, 350, 360] * u.deg)
    out = a.wrap_at('180d', inplace=True)
    assert out is None
    assert np.all(a.degree == np.array([-20.0, 150.0, -10.0, 0.0]))

def test_latitude():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        Latitude(['91d', '89d'])
    with pytest.raises(ValueError):
        Latitude('-91d')
    lat = Latitude(['90d', '89d'])
    assert lat[0] == 90 * u.deg
    assert lat[1] == 89 * u.deg
    assert np.all(lat == Angle(['90d', '89d']))
    lat[1] = 45.0 * u.deg
    assert np.all(lat == Angle(['90d', '45d']))
    with pytest.raises(ValueError):
        lat[0] = 90.001 * u.deg
    with pytest.raises(ValueError):
        lat[0] = -90.001 * u.deg
    assert np.all(lat == Angle(['90d', '45d']))
    angle = lat.to('radian')
    assert type(angle) is Latitude
    angle = lat - 190 * u.deg
    assert type(angle) is Angle
    assert angle[0] == -100 * u.deg
    lat = Latitude('80d')
    angle = lat / 2.0
    assert type(angle) is Angle
    assert angle == 40 * u.deg
    angle = lat * 2.0
    assert type(angle) is Angle
    assert angle == 160 * u.deg
    angle = -lat
    assert type(angle) is Angle
    assert angle == -80 * u.deg
    with pytest.raises(TypeError, match='A Latitude angle cannot be created from a Longitude angle'):
        lon = Longitude(10, 'deg')
        Latitude(lon)
    with pytest.raises(TypeError, match='A Longitude angle cannot be assigned to a Latitude angle'):
        lon = Longitude(10, 'deg')
        lat = Latitude([20], 'deg')
        lat[0] = lon
    lon = Longitude(10, 'deg')
    lat = Latitude(Angle(lon))
    assert lat.value == 10.0
    lon = Longitude(10, 'deg')
    lat = Latitude([20], 'deg')
    lat[0] = Angle(lon)
    assert lat.value[0] == 10.0

def test_longitude():
    if False:
        i = 10
        return i + 15
    lon = Longitude(['370d', '88d'])
    assert np.all(lon == Longitude(['10d', '88d']))
    assert np.all(lon == Angle(['10d', '88d']))
    angle = lon.to('hourangle')
    assert type(angle) is Longitude
    assert angle.wrap_angle == lon.wrap_angle
    angle = lon[0]
    assert type(angle) is Longitude
    assert angle.wrap_angle == lon.wrap_angle
    angle = lon[1:]
    assert type(angle) is Longitude
    assert angle.wrap_angle == lon.wrap_angle
    angle = lon / 2.0
    assert np.all(angle == Angle(['5d', '44d']))
    assert type(angle) is Angle
    assert not hasattr(angle, 'wrap_angle')
    angle = lon * 2.0 + 400 * u.deg
    assert np.all(angle == Angle(['420d', '576d']))
    assert type(angle) is Angle
    lon[1] = -10 * u.deg
    assert np.all(lon == Angle(['10d', '350d']))
    lon = Longitude(np.array([0, 0.5, 1.0, 1.5, 2.0]) * np.pi, unit=u.radian)
    assert np.all(lon.degree == np.array([0.0, 90, 180, 270, 0]))
    lon = Longitude(np.array([0, 0.5, 1.0, 1.5, 2.0]) * np.pi, unit=u.radian, wrap_angle='180d')
    assert np.all(lon.degree == np.array([0.0, 90, -180, -90, 0]))
    lon = Longitude(np.array([0, 0.5, 1.0, 1.5, 2.0]) * np.pi, unit=u.radian)
    lon.wrap_angle = '180d'
    assert np.all(lon.degree == np.array([0.0, 90, -180, -90, 0]))
    lon = Longitude('460d')
    assert lon == Angle('100d')
    lon.wrap_angle = '90d'
    assert lon == Angle('-260d')
    lon2 = Longitude(lon)
    assert lon2.wrap_angle == lon.wrap_angle
    lon3 = Longitude(lon, wrap_angle='180d')
    assert lon3.wrap_angle == 180 * u.deg
    lon = Longitude(lon, wrap_angle=Longitude(180 * u.deg))
    assert lon.wrap_angle == 180 * u.deg
    assert lon.wrap_angle.__class__ is Angle
    wrap_angle = 180 * u.deg
    lon = Longitude(lon, wrap_angle=wrap_angle)
    assert lon.wrap_angle == 180 * u.deg
    assert np.may_share_memory(lon.wrap_angle, wrap_angle)
    lon = Longitude(0, u.deg)
    lonstr = lon.to_string()
    assert not lonstr.startswith('-')
    assert Longitude(0, u.deg, dtype=float).dtype == np.dtype(float)
    assert Longitude(0, u.deg, dtype=int).dtype == np.dtype(int)
    with pytest.raises(TypeError, match='A Longitude angle cannot be created from a Latitude angle'):
        lat = Latitude(10, 'deg')
        Longitude(lat)
    with pytest.raises(TypeError, match='A Latitude angle cannot be assigned to a Longitude angle'):
        lat = Latitude(10, 'deg')
        lon = Longitude([20], 'deg')
        lon[0] = lat
    lat = Latitude(10, 'deg')
    lon = Longitude(Angle(lat))
    assert lon.value == 10.0
    lat = Latitude(10, 'deg')
    lon = Longitude([20], 'deg')
    lon[0] = Angle(lat)
    assert lon.value[0] == 10.0

def test_wrap_at():
    if False:
        print('Hello World!')
    a = Angle([-20, 150, 350, 360] * u.deg)
    assert np.all(a.wrap_at(360 * u.deg).degree == np.array([340.0, 150.0, 350.0, 0.0]))
    assert np.all(a.wrap_at(Angle(360, unit=u.deg)).degree == np.array([340.0, 150.0, 350.0, 0.0]))
    assert np.all(a.wrap_at('360d').degree == np.array([340.0, 150.0, 350.0, 0.0]))
    assert np.all(a.wrap_at('180d').degree == np.array([-20.0, 150.0, -10.0, 0.0]))
    assert np.all(a.wrap_at(np.pi * u.rad).degree == np.array([-20.0, 150.0, -10.0, 0.0]))
    a = Angle('190d')
    assert a.wrap_at('180d') == Angle('-170d')
    a = Angle(np.arange(-1000.0, 1000.0, 0.125), unit=u.deg)
    for wrap_angle in (270, 0.2, 0.0, 360.0, 500, -2000.125):
        aw = a.wrap_at(wrap_angle * u.deg)
        assert np.all(aw.degree >= wrap_angle - 360.0)
        assert np.all(aw.degree < wrap_angle)
        aw = a.to(u.rad).wrap_at(wrap_angle * u.deg)
        assert np.all(aw.degree >= wrap_angle - 360.0)
        assert np.all(aw.degree < wrap_angle)

def test_is_within_bounds():
    if False:
        return 10
    a = Angle([-20, 150, 350] * u.deg)
    assert a.is_within_bounds('0d', '360d') is False
    assert a.is_within_bounds(None, '360d') is True
    assert a.is_within_bounds(-30 * u.deg, None) is True
    a = Angle('-20d')
    assert a.is_within_bounds('0d', '360d') is False
    assert a.is_within_bounds(None, '360d') is True
    assert a.is_within_bounds(-30 * u.deg, None) is True

def test_angle_mismatched_unit():
    if False:
        return 10
    a = Angle('+6h7m8s', unit=u.degree)
    assert_allclose(a.value, 91.78333333333332)

def test_regression_formatting_negative():
    if False:
        while True:
            i = 10
    assert Angle(-0.0, unit='deg').to_string() == '-0d00m00s'
    assert Angle(-1.0, unit='deg').to_string() == '-1d00m00s'
    assert Angle(-0.0, unit='hour').to_string() == '-0h00m00s'
    assert Angle(-1.0, unit='hour').to_string() == '-1h00m00s'

def test_regression_formatting_default_precision():
    if False:
        while True:
            i = 10
    assert Angle('10:20:30.12345678d').to_string() == '10d20m30.12345678s'
    assert Angle('10d20m30.123456784564s').to_string() == '10d20m30.12345678s'
    assert Angle('10d20m30.123s').to_string() == '10d20m30.123s'

def test_empty_sep():
    if False:
        while True:
            i = 10
    a = Angle('05h04m31.93830s')
    assert a.to_string(sep='', precision=2, pad=True) == '050431.94'

@pytest.mark.parametrize('angle_class', [Angle, Longitude])
@pytest.mark.parametrize('unit', [u.hourangle, u.hour, None])
def test_create_tuple_fail(angle_class, unit):
    if False:
        while True:
            i = 10
    'Creating an angle from an (h,m,s) tuple should fail.'
    with pytest.raises(TypeError, match='no longer supported'):
        angle_class((12, 14, 52), unit=unit)

def test_list_of_quantities():
    if False:
        for i in range(10):
            print('nop')
    a1 = Angle([1 * u.deg, 1 * u.hourangle])
    assert a1.unit == u.deg
    assert_allclose(a1.value, [1, 15])
    a2 = Angle([1 * u.hourangle, 1 * u.deg], u.deg)
    assert a2.unit == u.deg
    assert_allclose(a2.value, [15, 1])

def test_multiply_divide():
    if False:
        i = 10
        return i + 15
    a1 = Angle([1, 2, 3], u.deg)
    a2 = Angle([4, 5, 6], u.deg)
    a3 = a1 * a2
    assert_allclose(a3.value, [4, 10, 18])
    assert a3.unit == u.deg * u.deg
    a3 = a1 / a2
    assert_allclose(a3.value, [0.25, 0.4, 0.5])
    assert a3.unit == u.dimensionless_unscaled

def test_mixed_string_and_quantity():
    if False:
        while True:
            i = 10
    a1 = Angle(['1d', 1.0 * u.deg])
    assert_array_equal(a1.value, [1.0, 1.0])
    assert a1.unit == u.deg
    a2 = Angle(['1d', 1 * u.rad * np.pi, '3d'])
    assert_array_equal(a2.value, [1.0, 180.0, 3.0])
    assert a2.unit == u.deg

def test_array_angle_tostring():
    if False:
        print('Hello World!')
    aobj = Angle([1, 2], u.deg)
    assert aobj.to_string().dtype.kind == 'U'
    assert np.all(aobj.to_string() == ['1d00m00s', '2d00m00s'])

def test_wrap_at_without_new():
    if False:
        i = 10
        return i + 15
    "\n    Regression test for subtle bugs from situations where an Angle is\n    created via numpy channels that don't do the standard __new__ but instead\n    depend on array_finalize to set state.  Longitude is used because the\n    bug was in its _wrap_angle not getting initialized correctly\n    "
    l1 = Longitude([1] * u.deg)
    l2 = Longitude([2] * u.deg)
    l = np.concatenate([l1, l2])
    assert l._wrap_angle is not None

def test__str__():
    if False:
        return 10
    '\n    Check the __str__ method used in printing the Angle\n    '
    scangle = Angle('10.2345d')
    strscangle = scangle.__str__()
    assert strscangle == '10d14m04.2s'
    arrangle = Angle(['10.2345d', '-20d'])
    strarrangle = arrangle.__str__()
    assert strarrangle == '[10d14m04.2s -20d00m00s]'
    bigarrangle = Angle(np.ones(10000), u.deg)
    assert '...' in bigarrangle.__str__()

def test_repr_latex():
    if False:
        return 10
    '\n    Check the _repr_latex_ method, used primarily by IPython notebooks\n    '
    scangle = Angle(2.1, u.deg)
    rlscangle = scangle._repr_latex_()
    arrangle = Angle([1, 2.1], u.deg)
    rlarrangle = arrangle._repr_latex_()
    assert rlscangle == '$2^\\circ06{}^\\prime00{}^{\\prime\\prime}$'
    assert rlscangle.split('$')[1] in rlarrangle
    bigarrangle = Angle(np.ones(50000) / 50000.0, u.deg)
    assert '...' in bigarrangle._repr_latex_()

def test_angle_with_cds_units_enabled():
    if False:
        i = 10
        return i + 15
    'Regression test for #5350\n\n    Especially the example in\n    https://github.com/astropy/astropy/issues/5350#issuecomment-248770151\n    '
    from astropy.coordinates.angles.formats import _AngleParser
    from astropy.units import cds
    del _AngleParser._thread_local._parser
    with cds.enable():
        Angle('5d')
    del _AngleParser._thread_local._parser
    Angle('5d')

def test_longitude_nan():
    if False:
        while True:
            i = 10
    Longitude([0, np.nan, 1] * u.deg)

def test_latitude_nan():
    if False:
        for i in range(10):
            print('nop')
    Latitude([0, np.nan, 1] * u.deg)

def test_angle_wrap_at_nan():
    if False:
        return 10
    angle = Angle([0, np.nan, 1] * u.deg)
    angle.flags.writeable = False
    angle.wrap_at(180 * u.deg, inplace=True)

def test_angle_multithreading():
    if False:
        for i in range(10):
            print('nop')
    '\n    Regression test for issue #7168\n    '
    angles = ['00:00:00'] * 10000

    def parse_test(i=0):
        if False:
            for i in range(10):
                print('nop')
        Angle(angles, unit='hour')
    for i in range(10):
        threading.Thread(target=parse_test, args=(i,)).start()

@pytest.mark.parametrize('cls', [Angle, Longitude, Latitude])
@pytest.mark.parametrize('input, expstr, exprepr', [(np.nan * u.deg, 'nan', 'nan deg'), ([np.nan, 5, 0] * u.deg, '[nan 5d00m00s 0d00m00s]', '[nan, 5., 0.] deg'), ([6, np.nan, 0] * u.deg, '[6d00m00s nan 0d00m00s]', '[6., nan, 0.] deg'), ([np.nan, np.nan, np.nan] * u.deg, '[nan nan nan]', '[nan, nan, nan] deg'), (np.nan * u.hour, 'nan', 'nan hourangle'), ([np.nan, 5, 0] * u.hour, '[nan 5h00m00s 0h00m00s]', '[nan, 5., 0.] hourangle'), ([6, np.nan, 0] * u.hour, '[6h00m00s nan 0h00m00s]', '[6., nan, 0.] hourangle'), ([np.nan, np.nan, np.nan] * u.hour, '[nan nan nan]', '[nan, nan, nan] hourangle'), (np.nan * u.rad, 'nan', 'nan rad'), ([np.nan, 1, 0] * u.rad, '[nan 1 rad 0 rad]', '[nan, 1., 0.] rad'), ([1.5, np.nan, 0] * u.rad, '[1.5 rad nan 0 rad]', '[1.5, nan, 0.] rad'), ([np.nan, np.nan, np.nan] * u.rad, '[nan nan nan]', '[nan, nan, nan] rad')])
def test_str_repr_angles_nan(cls, input, expstr, exprepr):
    if False:
        print('Hello World!')
    '\n    Regression test for issue #11473\n    '
    q = cls(input)
    assert str(q) == expstr
    assert repr(q).replace(' ', '') == f'<{cls.__name__}{exprepr}>'.replace(' ', '')

@pytest.mark.parametrize('sign', (-1, 1))
@pytest.mark.parametrize('value,expected_value,dtype,expected_dtype', [(np.pi * 2, 0.0, None, np.float64), (np.pi * 2, 0.0, np.float64, np.float64), (np.float32(2 * np.pi), np.float32(0.0), None, np.float32), (np.float32(2 * np.pi), np.float32(0.0), np.float32, np.float32)])
def test_longitude_wrap(value, expected_value, dtype, expected_dtype, sign):
    if False:
        i = 10
        return i + 15
    '\n    Test that the wrapping of the Longitude value range in radians works\n    in both float32 and float64.\n    '
    if sign < 0:
        value = -value
        expected_value = -expected_value
    result = Longitude(value, u.rad, dtype=dtype)
    assert result.value == expected_value
    assert result.dtype == expected_dtype
    assert result.unit == u.rad

@pytest.mark.parametrize('sign', (-1, 1))
@pytest.mark.parametrize('value,expected_value,dtype,expected_dtype', [(np.pi / 2, np.pi / 2, None, np.float64), (np.pi / 2, np.pi / 2, np.float64, np.float64), (np.float32(np.pi / 2), np.float32(np.pi / 2), None, np.float32), (np.float32(np.pi / 2), np.float32(np.pi / 2), np.float32, np.float32)])
def test_latitude_limits(value, expected_value, dtype, expected_dtype, sign):
    if False:
        return 10
    '\n    Test that the validation of the Latitude value range in radians works\n    in both float32 and float64.\n\n    As discussed in issue #13708, before, the float32 representation of pi/2\n    was rejected as invalid because the comparison always used the float64\n    representation.\n    '
    if sign < 0:
        value = -value
        expected_value = -expected_value
    result = Latitude(value, u.rad, dtype=dtype)
    assert result.value == expected_value
    assert result.dtype == expected_dtype
    assert result.unit == u.rad

@pytest.mark.parametrize('value,dtype', [(0.50001 * np.pi, np.float32), (np.float32(0.50001 * np.pi), np.float32), (0.50001 * np.pi, np.float64)])
def test_latitude_out_of_limits(value, dtype):
    if False:
        i = 10
        return i + 15
    '\n    Test that values slightly larger than pi/2 are rejected for different dtypes.\n    Test cases for issue #13708\n    '
    with pytest.raises(ValueError, match='Latitude angle\\(s\\) must be within.*'):
        Latitude(value, u.rad, dtype=dtype)

def test_angle_pickle_to_string():
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that after pickling we can still do to_string on hourangle.\n\n    Regression test for gh-13923.\n    '
    angle = Angle(0.25 * u.hourangle)
    expected = angle.to_string()
    via_pickle = pickle.loads(pickle.dumps(angle))
    via_pickle_string = via_pickle.to_string()
    assert via_pickle_string == expected