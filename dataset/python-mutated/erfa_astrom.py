"""
This module contains a helper function to fill erfa.astrom struct and a
ScienceState, which allows to speed up coordinate transformations at the
expense of accuracy.
"""
import warnings
import erfa
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.state import ScienceState
from .builtin_frames.utils import get_cip, get_jd12, get_polar_motion, pav2pv, prepare_earth_position_vel
from .matrix_utilities import rotation_matrix
__all__ = []

class ErfaAstrom:
    """
    The default provider for astrometry values.
    A utility class to extract the necessary arguments for
    erfa functions from frame attributes, call the corresponding
    erfa functions and return the astrom object.
    """

    @staticmethod
    def apco(frame_or_coord):
        if False:
            while True:
                i = 10
        '\n        Wrapper for ``erfa.apco``, used in conversions AltAz <-> ICRS and CIRS <-> ICRS.\n\n        Parameters\n        ----------\n        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``\n            Frame or coordinate instance in the corresponding frame\n            for which to calculate the calculate the astrom values.\n            For this function, an AltAz or CIRS frame is expected.\n        '
        (lon, lat, height) = frame_or_coord.location.to_geodetic('WGS84')
        obstime = frame_or_coord.obstime
        (jd1_tt, jd2_tt) = get_jd12(obstime, 'tt')
        (xp, yp) = get_polar_motion(obstime)
        sp = erfa.sp00(jd1_tt, jd2_tt)
        (x, y, s) = get_cip(jd1_tt, jd2_tt)
        era = erfa.era00(*get_jd12(obstime, 'ut1'))
        (earth_pv, earth_heliocentric) = prepare_earth_position_vel(obstime)
        if hasattr(frame_or_coord, 'pressure'):
            (refa, refb) = erfa.refco(frame_or_coord.pressure.to_value(u.hPa), frame_or_coord.temperature.to_value(u.deg_C), frame_or_coord.relative_humidity.value, frame_or_coord.obswl.to_value(u.micron))
        else:
            (refa, refb) = (0.0, 0.0)
        return erfa.apco(jd1_tt, jd2_tt, earth_pv, earth_heliocentric, x, y, s, era, lon.to_value(u.radian), lat.to_value(u.radian), height.to_value(u.m), xp, yp, sp, refa, refb)

    @staticmethod
    def apcs(frame_or_coord):
        if False:
            return 10
        '\n        Wrapper for ``erfa.apcs``, used in conversions GCRS <-> ICRS.\n\n        Parameters\n        ----------\n        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``\n            Frame or coordinate instance in the corresponding frame\n            for which to calculate the calculate the astrom values.\n            For this function, a GCRS frame is expected.\n        '
        (jd1_tt, jd2_tt) = get_jd12(frame_or_coord.obstime, 'tt')
        obs_pv = pav2pv(frame_or_coord.obsgeoloc.get_xyz(xyz_axis=-1).value, frame_or_coord.obsgeovel.get_xyz(xyz_axis=-1).value)
        (earth_pv, earth_heliocentric) = prepare_earth_position_vel(frame_or_coord.obstime)
        return erfa.apcs(jd1_tt, jd2_tt, obs_pv, earth_pv, earth_heliocentric)

    @staticmethod
    def apio(frame_or_coord):
        if False:
            print('Hello World!')
        '\n        Slightly modified equivalent of ``erfa.apio``, used in conversions AltAz <-> CIRS.\n\n        Since we use a topocentric CIRS frame, we have dropped the steps needed to calculate\n        diurnal aberration.\n\n        Parameters\n        ----------\n        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``\n            Frame or coordinate instance in the corresponding frame\n            for which to calculate the calculate the astrom values.\n            For this function, an AltAz frame is expected.\n        '
        sp = erfa.sp00(*get_jd12(frame_or_coord.obstime, 'tt'))
        theta = erfa.era00(*get_jd12(frame_or_coord.obstime, 'ut1'))
        (lon, lat, height) = frame_or_coord.location.to_geodetic('WGS84')
        elong = lon.to_value(u.radian)
        phi = lat.to_value(u.radian)
        (xp, yp) = get_polar_motion(frame_or_coord.obstime)
        astrom = np.zeros(frame_or_coord.obstime.shape, dtype=erfa.dt_eraASTROM)
        r = rotation_matrix(elong, 'z', unit=u.radian) @ rotation_matrix(-yp, 'x', unit=u.radian) @ rotation_matrix(-xp, 'y', unit=u.radian) @ rotation_matrix(theta + sp, 'z', unit=u.radian)
        a = r[..., 0, 0]
        b = r[..., 0, 1]
        eral = np.arctan2(b, a)
        astrom['eral'] = eral
        c = r[..., 0, 2]
        astrom['xpl'] = np.arctan2(c, np.sqrt(a * a + b * b))
        a = r[..., 1, 2]
        b = r[..., 2, 2]
        astrom['ypl'] = -np.arctan2(a, b)
        astrom['along'] = erfa.anpm(eral - theta)
        astrom['sphi'] = np.sin(phi)
        astrom['cphi'] = np.cos(phi)
        (astrom['refa'], astrom['refb']) = erfa.refco(frame_or_coord.pressure.to_value(u.hPa), frame_or_coord.temperature.to_value(u.deg_C), frame_or_coord.relative_humidity.value, frame_or_coord.obswl.to_value(u.micron))
        return astrom

class ErfaAstromInterpolator(ErfaAstrom):
    """
    A provider for astrometry values that does not call erfa
    for each individual timestamp but interpolates linearly
    between support points.

    For the interpolation, float64 MJD values are used, so time precision
    for the interpolation will be around a microsecond.

    This can dramatically speed up coordinate transformations,
    e.g. between CIRS and ICRS,
    when obstime is an array of many values (factors of 10 to > 100 depending
    on the selected resolution, number of points and the time range of the values).

    The precision of the transformation will still be in the order of microseconds
    for reasonable values of time_resolution, e.g. ``300 * u.s``.

    Users should benchmark performance and accuracy with the default transformation
    for their specific use case and then choose a suitable ``time_resolution``
    from there.

    This class is intended be used together with the ``erfa_astrom`` science state,
    e.g. in a context manager like this

    Example
    -------
    >>> from astropy.coordinates import SkyCoord, CIRS
    >>> from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import numpy as np

    >>> obstime = Time('2010-01-01T20:00:00') + np.linspace(0, 4, 1000) * u.hour
    >>> crab = SkyCoord(ra='05h34m31.94s', dec='22d00m52.2s')
    >>> with erfa_astrom.set(ErfaAstromInterpolator(300 * u.s)):
    ...    cirs = crab.transform_to(CIRS(obstime=obstime))
    """

    @u.quantity_input(time_resolution=u.day)
    def __init__(self, time_resolution):
        if False:
            return 10
        if time_resolution.to_value(u.us) < 10:
            warnings.warn(f'Using {self.__class__.__name__} with `time_resolution` below 10 microseconds might lead to numerical inaccuracies as the MJD-based interpolation is limited by floating point  precision to about a microsecond of precision', AstropyWarning)
        self.mjd_resolution = time_resolution.to_value(u.day)

    def _get_support_points(self, obstime):
        if False:
            print('Hello World!')
        '\n        Calculate support points for the interpolation.\n\n        We divide the MJD by the time resolution (as single float64 values),\n        and calculate ceil and floor.\n        Then we take the unique and sorted values and scale back to MJD.\n        This will create a sparse support for non-regular input obstimes.\n        '
        mjd_scaled = np.ravel(obstime.mjd / self.mjd_resolution)
        mjd_u = np.unique(np.concatenate([np.floor(mjd_scaled), np.ceil(mjd_scaled)]))
        return Time(mjd_u * self.mjd_resolution, format='mjd', scale=obstime.scale)

    @staticmethod
    def _prepare_earth_position_vel(support, obstime):
        if False:
            while True:
                i = 10
        "\n        Calculate Earth's position and velocity.\n\n        Uses the coarser grid ``support`` to do the calculation, and interpolates\n        onto the finer grid ``obstime``.\n        "
        (pv_support, heliocentric_support) = prepare_earth_position_vel(support)
        earth_pv = np.empty(obstime.shape, dtype=erfa.dt_pv)
        earth_heliocentric = np.empty(obstime.shape + (3,))
        for dim in range(3):
            for key in 'pv':
                earth_pv[key][..., dim] = np.interp(obstime.mjd, support.mjd, pv_support[key][..., dim])
            earth_heliocentric[..., dim] = np.interp(obstime.mjd, support.mjd, heliocentric_support[..., dim])
        return (earth_pv, earth_heliocentric)

    @staticmethod
    def _get_c2i(support, obstime):
        if False:
            while True:
                i = 10
        '\n        Calculate the Celestial-to-Intermediate rotation matrix.\n\n        Uses the coarser grid ``support`` to do the calculation, and interpolates\n        onto the finer grid ``obstime``.\n        '
        (jd1_tt_support, jd2_tt_support) = get_jd12(support, 'tt')
        c2i_support = erfa.c2i06a(jd1_tt_support, jd2_tt_support)
        c2i = np.empty(obstime.shape + (3, 3))
        for dim1 in range(3):
            for dim2 in range(3):
                c2i[..., dim1, dim2] = np.interp(obstime.mjd, support.mjd, c2i_support[..., dim1, dim2])
        return c2i

    @staticmethod
    def _get_cip(support, obstime):
        if False:
            i = 10
            return i + 15
        '\n        Find the X, Y coordinates of the CIP and the CIO locator, s.\n\n        Uses the coarser grid ``support`` to do the calculation, and interpolates\n        onto the finer grid ``obstime``.\n        '
        (jd1_tt_support, jd2_tt_support) = get_jd12(support, 'tt')
        cip_support = get_cip(jd1_tt_support, jd2_tt_support)
        return tuple((np.interp(obstime.mjd, support.mjd, cip_component) for cip_component in cip_support))

    @staticmethod
    def _get_polar_motion(support, obstime):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find the two polar motion components in radians.\n\n        Uses the coarser grid ``support`` to do the calculation, and interpolates\n        onto the finer grid ``obstime``.\n        '
        polar_motion_support = get_polar_motion(support)
        return tuple((np.interp(obstime.mjd, support.mjd, polar_motion_component) for polar_motion_component in polar_motion_support))

    def apco(self, frame_or_coord):
        if False:
            i = 10
            return i + 15
        '\n        Wrapper for ``erfa.apco``, used in conversions AltAz <-> ICRS and CIRS <-> ICRS.\n\n        Parameters\n        ----------\n        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``\n            Frame or coordinate instance in the corresponding frame\n            for which to calculate the calculate the astrom values.\n            For this function, an AltAz or CIRS frame is expected.\n        '
        (lon, lat, height) = frame_or_coord.location.to_geodetic('WGS84')
        obstime = frame_or_coord.obstime
        support = self._get_support_points(obstime)
        (jd1_tt, jd2_tt) = get_jd12(obstime, 'tt')
        (earth_pv, earth_heliocentric) = self._prepare_earth_position_vel(support, obstime)
        (xp, yp) = self._get_polar_motion(support, obstime)
        sp = erfa.sp00(jd1_tt, jd2_tt)
        (x, y, s) = self._get_cip(support, obstime)
        era = erfa.era00(*get_jd12(obstime, 'ut1'))
        if hasattr(frame_or_coord, 'pressure'):
            (refa, refb) = erfa.refco(frame_or_coord.pressure.to_value(u.hPa), frame_or_coord.temperature.to_value(u.deg_C), frame_or_coord.relative_humidity.value, frame_or_coord.obswl.to_value(u.micron))
        else:
            (refa, refb) = (0.0, 0.0)
        return erfa.apco(jd1_tt, jd2_tt, earth_pv, earth_heliocentric, x, y, s, era, lon.to_value(u.radian), lat.to_value(u.radian), height.to_value(u.m), xp, yp, sp, refa, refb)

    def apcs(self, frame_or_coord):
        if False:
            print('Hello World!')
        '\n        Wrapper for ``erfa.apci``, used in conversions GCRS <-> ICRS.\n\n        Parameters\n        ----------\n        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``\n            Frame or coordinate instance in the corresponding frame\n            for which to calculate the calculate the astrom values.\n            For this function, a GCRS frame is expected.\n        '
        obstime = frame_or_coord.obstime
        support = self._get_support_points(obstime)
        (earth_pv, earth_heliocentric) = self._prepare_earth_position_vel(support, obstime)
        pv = pav2pv(frame_or_coord.obsgeoloc.get_xyz(xyz_axis=-1).value, frame_or_coord.obsgeovel.get_xyz(xyz_axis=-1).value)
        (jd1_tt, jd2_tt) = get_jd12(obstime, 'tt')
        return erfa.apcs(jd1_tt, jd2_tt, pv, earth_pv, earth_heliocentric)

class erfa_astrom(ScienceState):
    """
    ScienceState to select with astrom provider is used in
    coordinate transformations.
    """
    _value = ErfaAstrom()

    @classmethod
    def validate(cls, value):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, ErfaAstrom):
            raise TypeError(f'Must be an instance of {ErfaAstrom!r}')
        return value