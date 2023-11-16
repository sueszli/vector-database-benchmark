import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from astropy.wcs.wcsapi.low_level_api import BaseLowLevelWCS

class CustomLowLevelWCS(BaseLowLevelWCS):

    @property
    def pixel_n_dim(self):
        if False:
            i = 10
            return i + 15
        return 2

    @property
    def world_n_dim(self):
        if False:
            return 10
        return 2

    @property
    def world_axis_physical_types(self):
        if False:
            return 10
        return ['pos.eq.ra', 'pos.eq.dec']

    @property
    def world_axis_units(self):
        if False:
            while True:
                i = 10
        return ['deg', 'deg']

    def pixel_to_world_values(self, *pixel_arrays):
        if False:
            i = 10
            return i + 15
        return [np.asarray(pix) * 2 for pix in pixel_arrays]

    def world_to_pixel_values(self, *world_arrays):
        if False:
            print('Hello World!')
        return [np.asarray(world) / 2 for world in world_arrays]

    @property
    def world_axis_object_components(self):
        if False:
            for i in range(10):
                print('nop')
        return [('test', 0, 'spherical.lon.degree'), ('test', 1, 'spherical.lat.degree')]

    @property
    def world_axis_object_classes(self):
        if False:
            for i in range(10):
                print('nop')
        return {'test': (SkyCoord, (), {'unit': 'deg'})}

def test_wrapper():
    if False:
        i = 10
        return i + 15
    wcs = CustomLowLevelWCS()
    wrapper = HighLevelWCSWrapper(wcs)
    coord = wrapper.pixel_to_world(1, 2)
    assert isinstance(coord, SkyCoord)
    assert coord.isscalar
    (x, y) = wrapper.world_to_pixel(coord)
    assert_allclose(x, 1)
    assert_allclose(y, 2)
    assert wrapper.low_level_wcs is wcs
    assert wrapper.pixel_n_dim == 2
    assert wrapper.world_n_dim == 2
    assert wrapper.world_axis_physical_types == ['pos.eq.ra', 'pos.eq.dec']
    assert wrapper.world_axis_units == ['deg', 'deg']
    assert wrapper.array_shape is None
    assert wrapper.pixel_bounds is None
    assert np.all(wrapper.axis_correlation_matrix)

def test_wrapper_invalid():
    if False:
        print('Hello World!')

    class InvalidCustomLowLevelWCS(CustomLowLevelWCS):

        @property
        def world_axis_object_classes(self):
            if False:
                for i in range(10):
                    print('nop')
            return {}
    wcs = InvalidCustomLowLevelWCS()
    wrapper = HighLevelWCSWrapper(wcs)
    with pytest.raises(KeyError):
        wrapper.pixel_to_world(1, 2)