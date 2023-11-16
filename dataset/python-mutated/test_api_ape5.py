"""
This is the APE5 coordinates API document re-written to work as a series of test
functions.

Note that new tests for coordinates functionality should generally *not* be
added to this file - instead, add them to other appropriate test modules  in
this package, like ``test_sky_coord.py``, ``test_frames.py``, or
``test_representation.py``.  This file is instead meant mainly to keep track of
deviations from the original APE5 plan.
"""
import numpy as np
import pytest
from numpy import testing as npt
from astropy import coordinates as coords
from astropy import time
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose as assert_allclose
from astropy.units import allclose
from astropy.utils.compat.optional_deps import HAS_SCIPY

def test_representations_api():
    if False:
        return 10
    from astropy.coordinates import Angle, Distance, Latitude, Longitude
    from astropy.coordinates.representation import CartesianRepresentation, PhysicsSphericalRepresentation, SphericalRepresentation, UnitSphericalRepresentation
    UnitSphericalRepresentation(lon=8 * u.hour, lat=5 * u.deg)
    UnitSphericalRepresentation(lon=8 * u.hourangle, lat=5 * u.deg)
    SphericalRepresentation(lon=8 * u.hourangle, lat=5 * u.deg, distance=10 * u.kpc)
    UnitSphericalRepresentation(Longitude(8, u.hour), Latitude(5, u.deg))
    SphericalRepresentation(Longitude(8, u.hour), Latitude(5, u.deg), Distance(10, u.kpc))
    UnitSphericalRepresentation(lon=[8, 9] * u.hourangle, lat=[5, 6] * u.deg)
    UnitSphericalRepresentation(lon=[8, 9] * u.hourangle, lat=[5, 6] * u.deg, copy=False)
    UnitSphericalRepresentation(lon=Angle('2h6m3.3s'), lat=Angle('0.1rad'))
    c1 = SphericalRepresentation(lon=8 * u.hourangle, lat=5 * u.deg, distance=10 * u.kpc)
    c2 = SphericalRepresentation.from_representation(c1)
    SphericalRepresentation(lon=[8, 9] * u.hourangle, lat=[5, 6] * u.deg, distance=[10, 11] * u.kpc)
    c2 = SphericalRepresentation(lon=[8, 9] * u.hourangle, lat=[5, 6] * u.deg, distance=10 * u.kpc)
    assert len(c2.distance) == 2
    with pytest.raises(ValueError):
        c2 = UnitSphericalRepresentation(lon=[8, 9, 10] * u.hourangle, lat=[5, 6] * u.deg)
    c2 = UnitSphericalRepresentation(lon=Angle([8 * u.hourangle, 135 * u.deg]), lat=Angle([5 * u.deg, 6 * np.pi / 180 * u.rad]))
    assert c2.lat.unit == u.deg and c2.lon.unit == u.hourangle
    npt.assert_almost_equal(c2.lon[1].value, 9)
    lon = u.Quantity([120 * u.deg, 135 * u.deg], u.hourangle)
    lat = u.Quantity([5 * np.pi / 180 * u.rad, 0.4 * u.hourangle], u.deg)
    c2 = UnitSphericalRepresentation(lon, lat)
    assert isinstance(c1.lat, Angle)
    assert isinstance(c1.lat, Latitude)
    assert isinstance(c1.distance, Distance)
    with pytest.raises(AttributeError):
        c1.lat = Latitude(5, u.deg)
    c2.lat[:] = [0] * u.deg
    _ = PhysicsSphericalRepresentation(phi=120 * u.deg, theta=85 * u.deg, r=3 * u.kpc)
    c1 = CartesianRepresentation(np.random.randn(3, 100) * u.kpc)
    assert c1.xyz.shape[0] == 3
    assert c1.xyz.unit == u.kpc
    assert c1.x.shape[0] == 100
    assert c1.y.shape[0] == 100
    assert c1.z.shape[0] == 100
    CartesianRepresentation(x=np.random.randn(100) * u.kpc, y=np.random.randn(100) * u.kpc, z=np.random.randn(100) * u.kpc)
    (xarr, yarr, zarr) = np.random.randn(3, 100)
    c1 = CartesianRepresentation(x=xarr * u.kpc, y=yarr * u.kpc, z=zarr * u.kpc)
    c2 = CartesianRepresentation(x=xarr * u.kpc, y=yarr * u.kpc, z=zarr * u.pc)
    assert c1.xyz.unit == c2.xyz.unit == u.kpc
    assert_allclose(c1.z / 1000 - c2.z, 0 * u.kpc, atol=1e-10 * u.kpc)
    srep = SphericalRepresentation(lon=90 * u.deg, lat=0 * u.deg, distance=1 * u.pc)
    crep = srep.represent_as(CartesianRepresentation)
    assert_allclose(crep.x, 0 * u.pc, atol=1e-10 * u.pc)
    assert_allclose(crep.y, 1 * u.pc, atol=1e-10 * u.pc)
    assert_allclose(crep.z, 0 * u.pc, atol=1e-10 * u.pc)

def test_frame_api():
    if False:
        return 10
    from astropy.coordinates.builtin_frames import FK5, ICRS
    from astropy.coordinates.representation import SphericalRepresentation, UnitSphericalRepresentation
    icrs = ICRS(UnitSphericalRepresentation(lon=8 * u.hour, lat=5 * u.deg))
    assert icrs.data.lat == 5 * u.deg
    assert icrs.data.lon == 8 * u.hourangle
    fk5 = FK5(UnitSphericalRepresentation(lon=8 * u.hour, lat=5 * u.deg))
    J2000 = time.Time('J2000')
    fk5_2000 = FK5(UnitSphericalRepresentation(lon=8 * u.hour, lat=5 * u.deg), equinox=J2000)
    assert fk5.equinox == fk5_2000.equinox
    J2001 = time.Time('J2001')
    with pytest.raises(AttributeError):
        fk5.equinox = J2001
    with pytest.raises(AttributeError):
        fk5.data = UnitSphericalRepresentation(lon=8 * u.hour, lat=5 * u.deg)
    assert all((nm in ('equinox', 'obstime') for nm in fk5.frame_attributes))
    assert_allclose(icrs.represent_as(SphericalRepresentation).lat, 5 * u.deg)
    assert_allclose(icrs.spherical.lat, 5 * u.deg)
    assert icrs.cartesian.z.value > 0
    assert_allclose(icrs.dec, 5 * u.deg)
    assert_allclose(fk5.ra, 8 * u.hourangle)
    assert icrs.representation_type == SphericalRepresentation
    icrs_2 = ICRS(ra=8 * u.hour, dec=5 * u.deg, distance=1 * u.kpc)
    assert_allclose(icrs.ra, icrs_2.ra)
    coo1 = ICRS(ra=0 * u.hour, dec=0 * u.deg)
    coo2 = ICRS(ra=0 * u.hour, dec=1 * u.deg)
    assert_allclose(coo1.separation(coo2).degree, 1.0)
    coo3 = ICRS(ra=0 * u.hour, dec=0 * u.deg, distance=1 * u.kpc)
    coo4 = ICRS(ra=0 * u.hour, dec=0 * u.deg, distance=2 * u.kpc)
    assert coo3.separation_3d(coo4).kpc == 1.0
    with pytest.raises(ValueError):
        assert coo1.separation_3d(coo2).kpc == 1.0

def test_transform_api():
    if False:
        print('Hello World!')
    from astropy.coordinates.baseframe import BaseCoordinateFrame, frame_transform_graph
    from astropy.coordinates.builtin_frames import FK5, ICRS
    from astropy.coordinates.representation import UnitSphericalRepresentation
    from astropy.coordinates.transformations import DynamicMatrixTransform
    fk5 = FK5(ra=8 * u.hour, dec=5 * u.deg)
    J2001 = time.Time('J2001')
    fk5_J2001_frame = FK5(equinox=J2001)
    assert repr(fk5_J2001_frame) == '<FK5 Frame (equinox=J2001.000)>'
    srep = UnitSphericalRepresentation(lon=8 * u.hour, lat=5 * u.deg)
    fk5_j2001_with_data = fk5_J2001_frame.realize_frame(srep)
    assert fk5_j2001_with_data.data is not None
    newfk5 = fk5.transform_to(fk5_J2001_frame)
    assert newfk5.equinox == J2001
    fk5_2 = FK5(ra=8 * u.hour, dec=5 * u.deg, equinox=J2001)
    ic_trans = fk5_2.transform_to(ICRS())
    fk5_trans = ic_trans.transform_to(FK5())
    assert not allclose(fk5_2.ra, fk5_trans.ra, rtol=0, atol=1e-10 * u.deg)
    fk5_trans_2 = fk5_2.transform_to(FK5(equinox=J2001))
    assert_allclose(fk5_2.ra, fk5_trans_2.ra, rtol=0, atol=1e-10 * u.deg)
    with pytest.raises(ValueError):
        FK5(equinox=J2001).transform_to(ICRS())

    class SomeNewSystem(BaseCoordinateFrame):
        pass

    @frame_transform_graph.transform(DynamicMatrixTransform, SomeNewSystem, FK5)
    def new_to_fk5(newobj, fk5frame):
        if False:
            while True:
                i = 10
        _ = newobj.obstime
        _ = fk5frame.equinox
        matrix = np.eye(3)
        return matrix

def test_highlevel_api():
    if False:
        while True:
            i = 10
    J2001 = time.Time('J2001')
    sc = coords.SkyCoord(coords.SphericalRepresentation(lon=8 * u.hour, lat=5 * u.deg, distance=1 * u.kpc), frame='icrs')
    sc = coords.SkyCoord(ra=8 * u.hour, dec=5 * u.deg, frame='icrs')
    sc = coords.SkyCoord(l=120 * u.deg, b=5 * u.deg, frame='galactic')
    sc = coords.SkyCoord(coords.ICRS(ra=8 * u.hour, dec=5 * u.deg))
    with pytest.raises(ValueError):
        sc = coords.SkyCoord(coords.FK5(equinox=J2001))
    rscf = repr(sc.frame)
    assert rscf.startswith('<ICRS Coordinate: (ra, dec) in deg')
    rsc = repr(sc)
    assert rsc.startswith('<SkyCoord (ICRS): (ra, dec) in deg')
    sc = coords.SkyCoord('8h00m00s +5d00m00.0s', frame='icrs')
    sc = coords.SkyCoord('8:00:00 +5:00:00.0', unit=(u.hour, u.deg), frame='icrs')
    sc = coords.SkyCoord(['8h 5d', '2°2′3″ 0.3rad'], frame='icrs')
    sc_fk5_j2001 = sc.transform_to(coords.FK5(equinox=J2001))
    assert sc_fk5_j2001.equinox == J2001
    sc1 = coords.SkyCoord(ra=8 * u.hour, dec=5 * u.deg, equinox=J2001, frame='fk5')
    sc2 = sc1.transform_to('icrs')
    assert sc2.equinox == J2001
    sc3 = sc2.transform_to('fk5')
    assert sc3.equinox == J2001
    assert_allclose(sc1.ra, sc3.ra)
    sc = coords.SkyCoord(ra=8 * u.hour, dec=5 * u.deg, frame='icrs')
    scgal = sc.galactic
    assert str(scgal).startswith('<SkyCoord (Galactic): (l, b)')
    if HAS_SCIPY:
        cat1 = coords.SkyCoord(ra=[1, 2] * u.hr, dec=[3, 4.01] * u.deg, distance=[5, 6] * u.kpc, frame='icrs')
        cat2 = coords.SkyCoord(ra=[1, 2, 2.01] * u.hr, dec=[3, 4, 5] * u.deg, distance=[5, 200, 6] * u.kpc, frame='icrs')
        (idx1, sep2d1, dist3d1) = cat1.match_to_catalog_sky(cat2)
        (idx2, sep2d2, dist3d2) = cat1.match_to_catalog_3d(cat2)
        assert np.any(idx1 != idx2)

@pytest.mark.remote_data
def test_highlevel_api_remote():
    if False:
        while True:
            i = 10
    m31icrs = coords.SkyCoord.from_name('M31', frame='icrs')
    m31str = str(m31icrs)
    assert m31str.startswith('<SkyCoord (ICRS): (ra, dec) in deg\n    (')
    assert m31str.endswith(')>')
    assert '10.68' in m31str
    assert '41.26' in m31str
    m31fk4 = coords.SkyCoord.from_name('M31', frame='fk4')
    assert not m31icrs.is_equivalent_frame(m31fk4)
    assert np.abs(m31icrs.ra - m31fk4.ra) > 0.5 * u.deg