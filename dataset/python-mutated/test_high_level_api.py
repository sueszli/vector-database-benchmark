import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs.wcsapi.high_level_api import HighLevelWCSMixin, high_level_objects_to_values, values_to_high_level_objects
from astropy.wcs.wcsapi.low_level_api import BaseLowLevelWCS

class DoubleLowLevelWCS(BaseLowLevelWCS):
    """
    Basic dummy transformation that doubles values.
    """

    def pixel_to_world_values(self, *pixel_arrays):
        if False:
            print('Hello World!')
        return [np.asarray(pix) * 2 for pix in pixel_arrays]

    def world_to_pixel_values(self, *world_arrays):
        if False:
            for i in range(10):
                print('nop')
        return [np.asarray(world) / 2 for world in world_arrays]

class SimpleDuplicateWCS(DoubleLowLevelWCS, HighLevelWCSMixin):
    """
    This example WCS has two of the world coordinates that use the same class,
    which triggers a different path in the high level WCS code.
    """

    @property
    def pixel_n_dim(self):
        if False:
            return 10
        return 2

    @property
    def world_n_dim(self):
        if False:
            return 10
        return 2

    @property
    def world_axis_physical_types(self):
        if False:
            i = 10
            return i + 15
        return ['pos.eq.ra', 'pos.eq.dec']

    @property
    def world_axis_units(self):
        if False:
            while True:
                i = 10
        return ['deg', 'deg']

    @property
    def world_axis_object_components(self):
        if False:
            return 10
        return [('test1', 0, 'value'), ('test2', 0, 'value')]

    @property
    def world_axis_object_classes(self):
        if False:
            while True:
                i = 10
        return {'test1': (Quantity, (), {'unit': 'deg'}), 'test2': (Quantity, (), {'unit': 'deg'})}

def test_simple_duplicate():
    if False:
        return 10
    wcs = SimpleDuplicateWCS()
    (q1, q2) = wcs.pixel_to_world(1, 2)
    assert isinstance(q1, Quantity)
    assert isinstance(q2, Quantity)
    (x, y) = wcs.world_to_pixel(q1, q2)
    assert_allclose(x, 1)
    assert_allclose(y, 2)

class SkyCoordDuplicateWCS(DoubleLowLevelWCS, HighLevelWCSMixin):
    """
    This example WCS returns two SkyCoord objects which, which triggers a
    different path in the high level WCS code.
    """

    @property
    def pixel_n_dim(self):
        if False:
            return 10
        return 4

    @property
    def world_n_dim(self):
        if False:
            for i in range(10):
                print('nop')
        return 4

    @property
    def world_axis_physical_types(self):
        if False:
            return 10
        return ['pos.eq.ra', 'pos.eq.dec', 'pos.galactic.lon', 'pos.galactic.lat']

    @property
    def world_axis_units(self):
        if False:
            for i in range(10):
                print('nop')
        return ['deg', 'deg', 'deg', 'deg']

    @property
    def world_axis_object_components(self):
        if False:
            print('Hello World!')
        return [('test1', 'ra', 'spherical.lon.degree'), ('test1', 'dec', 'spherical.lat.degree'), ('test2', 0, 'spherical.lon.degree'), ('test2', 1, 'spherical.lat.degree')]

    @property
    def world_axis_object_classes(self):
        if False:
            print('Hello World!')
        return {'test1': (SkyCoord, (), {'unit': 'deg'}), 'test2': (SkyCoord, (), {'unit': 'deg', 'frame': 'galactic'})}

def test_skycoord_duplicate():
    if False:
        print('Hello World!')
    wcs = SkyCoordDuplicateWCS()
    (c1, c2) = wcs.pixel_to_world(1, 2, 3, 4)
    assert isinstance(c1, SkyCoord)
    assert isinstance(c2, SkyCoord)
    (x, y, z, a) = wcs.world_to_pixel(c1, c2)
    assert_allclose(x, 1)
    assert_allclose(y, 2)
    assert_allclose(z, 3)
    assert_allclose(a, 4)

class SerializedWCS(DoubleLowLevelWCS, HighLevelWCSMixin):
    """
    WCS with serialized classes
    """

    @property
    def serialized_classes(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def pixel_n_dim(self):
        if False:
            while True:
                i = 10
        return 2

    @property
    def world_n_dim(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

    @property
    def world_axis_physical_types(self):
        if False:
            return 10
        return ['pos.eq.ra', 'pos.eq.dec']

    @property
    def world_axis_units(self):
        if False:
            return 10
        return ['deg', 'deg']

    @property
    def world_axis_object_components(self):
        if False:
            i = 10
            return i + 15
        return [('test', 0, 'value')]

    @property
    def world_axis_object_classes(self):
        if False:
            return 10
        return {'test': ('astropy.units.Quantity', (), {'unit': ('astropy.units.Unit', ('deg',), {})})}

def test_serialized_classes():
    if False:
        return 10
    wcs = SerializedWCS()
    q = wcs.pixel_to_world(1)
    assert isinstance(q, Quantity)
    x = wcs.world_to_pixel(q)
    assert_allclose(x, 1)

def test_objects_to_values():
    if False:
        i = 10
        return i + 15
    wcs = SkyCoordDuplicateWCS()
    (c1, c2) = wcs.pixel_to_world(1, 2, 3, 4)
    values = high_level_objects_to_values(c1, c2, low_level_wcs=wcs)
    assert np.allclose(values, [2, 4, 6, 8])

def test_values_to_objects():
    if False:
        print('Hello World!')
    wcs = SkyCoordDuplicateWCS()
    (c1, c2) = wcs.pixel_to_world(1, 2, 3, 4)
    (c1_out, c2_out) = values_to_high_level_objects(*[2, 4, 6, 8], low_level_wcs=wcs)
    assert c1.ra == c1_out.ra
    assert c2.l == c2_out.l
    assert c1.dec == c1_out.dec
    assert c2.b == c2_out.b

class MinimalHighLevelWCS(HighLevelWCSMixin):

    def __init__(self, low_level_wcs):
        if False:
            return 10
        self._low_level_wcs = low_level_wcs

    @property
    def low_level_wcs(self):
        if False:
            for i in range(10):
                print('nop')
        return self._low_level_wcs

def test_minimal_mixin_subclass():
    if False:
        i = 10
        return i + 15
    fits_wcs = WCS(naxis=2)
    high_level_wcs = MinimalHighLevelWCS(fits_wcs)
    coord = high_level_wcs.pixel_to_world(1, 2)
    pixel = high_level_wcs.world_to_pixel(*coord)
    coord = high_level_wcs.array_index_to_world(1, 2)
    pixel = high_level_wcs.world_to_array_index(*coord)
    assert_allclose(pixel, (1, 2))