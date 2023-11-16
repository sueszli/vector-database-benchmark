from .high_level_api import HighLevelWCSMixin
from .low_level_api import BaseLowLevelWCS
from .utils import wcs_info_str
__all__ = ['HighLevelWCSWrapper']

class HighLevelWCSWrapper(HighLevelWCSMixin):
    """
    Wrapper class that can take any :class:`~astropy.wcs.wcsapi.BaseLowLevelWCS`
    object and expose the high-level WCS API.
    """

    def __init__(self, low_level_wcs):
        if False:
            return 10
        if not isinstance(low_level_wcs, BaseLowLevelWCS):
            raise TypeError('Input to a HighLevelWCSWrapper must be a low level WCS object')
        self._low_level_wcs = low_level_wcs

    @property
    def low_level_wcs(self):
        if False:
            for i in range(10):
                print('nop')
        return self._low_level_wcs

    @property
    def pixel_n_dim(self):
        if False:
            i = 10
            return i + 15
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim`.\n        '
        return self.low_level_wcs.pixel_n_dim

    @property
    def world_n_dim(self):
        if False:
            return 10
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim`.\n        '
        return self.low_level_wcs.world_n_dim

    @property
    def world_axis_physical_types(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_physical_types`.\n        '
        return self.low_level_wcs.world_axis_physical_types

    @property
    def world_axis_units(self):
        if False:
            i = 10
            return i + 15
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_units`.\n        '
        return self.low_level_wcs.world_axis_units

    @property
    def array_shape(self):
        if False:
            return 10
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.array_shape`.\n        '
        return self.low_level_wcs.array_shape

    @property
    def pixel_bounds(self):
        if False:
            while True:
                i = 10
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_bounds`.\n        '
        return self.low_level_wcs.pixel_bounds

    @property
    def axis_correlation_matrix(self):
        if False:
            return 10
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.axis_correlation_matrix`.\n        '
        return self.low_level_wcs.axis_correlation_matrix

    def _as_mpl_axes(self):
        if False:
            while True:
                i = 10
        '\n        See `~astropy.wcs.wcsapi.BaseLowLevelWCS._as_mpl_axes`.\n        '
        return self.low_level_wcs._as_mpl_axes()

    def __str__(self):
        if False:
            print('Hello World!')
        return wcs_info_str(self.low_level_wcs)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{object.__repr__(self)}\n{str(self)}'