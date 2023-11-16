"""
This module contains convenience functions for retrieving solar system
ephemerides from jplephem.
"""
import os.path
import re
from urllib.parse import urlparse
import erfa
import numpy as np
from astropy import units as u
from astropy.constants import c as speed_of_light
from astropy.utils import indent
from astropy.utils.data import download_file
from astropy.utils.decorators import classproperty, deprecated
from astropy.utils.state import ScienceState
from .builtin_frames import GCRS, ICRS
from .builtin_frames.utils import get_jd12
from .representation import CartesianRepresentation
from .sky_coordinate import SkyCoord
__all__ = ['get_body', 'get_moon', 'get_body_barycentric', 'get_body_barycentric_posvel', 'solar_system_ephemeris']
DEFAULT_JPL_EPHEMERIS = 'de430'
'List of kernel pairs needed to calculate positions of a given object.'
BODY_NAME_TO_KERNEL_SPEC = {'sun': [(0, 10)], 'mercury': [(0, 1), (1, 199)], 'venus': [(0, 2), (2, 299)], 'earth-moon-barycenter': [(0, 3)], 'earth': [(0, 3), (3, 399)], 'moon': [(0, 3), (3, 301)], 'mars': [(0, 4)], 'jupiter': [(0, 5)], 'saturn': [(0, 6)], 'uranus': [(0, 7)], 'neptune': [(0, 8)], 'pluto': [(0, 9)]}
'Indices to the plan94 routine for the given object.'
PLAN94_BODY_NAME_TO_PLANET_INDEX = {'mercury': 1, 'venus': 2, 'earth-moon-barycenter': 3, 'mars': 4, 'jupiter': 5, 'saturn': 6, 'uranus': 7, 'neptune': 8}
_EPHEMERIS_NOTE = "\nYou can either give an explicit ephemeris or use a default, which is normally\na built-in ephemeris that does not require ephemeris files.  To change\nthe default to be the JPL ephemeris::\n\n    >>> from astropy.coordinates import solar_system_ephemeris\n    >>> solar_system_ephemeris.set('jpl')  # doctest: +SKIP\n\nUse of any JPL ephemeris requires the jplephem package\n(https://pypi.org/project/jplephem/).\nIf needed, the ephemeris file will be downloaded (and cached).\n\nOne can check which bodies are covered by a given ephemeris using::\n\n    >>> solar_system_ephemeris.bodies\n    ('earth', 'sun', 'moon', 'mercury', 'venus', 'earth-moon-barycenter', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')\n"[1:-1]

class solar_system_ephemeris(ScienceState):
    """Default ephemerides for calculating positions of Solar-System bodies.

    This can be one of the following:

    - 'builtin': polynomial approximations to the orbital elements.
    - 'dexxx[s]', for a JPL dynamical model, where xxx is the three digit
      version number (e.g. de430), and the 's' is optional to specify the
      'small' version of a kernel. The version number must correspond to an
      ephemeris file available at:
      https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    - 'jpl': Alias for the default JPL ephemeris (currently, 'de430').
    - URL: (str) The url to a SPK ephemeris in SPICE binary (.bsp) format.
    - PATH: (str) File path to a SPK ephemeris in SPICE binary (.bsp) format.
    - `None`: Ensure an Exception is raised without an explicit ephemeris.

    The default is 'builtin', which uses the ``epv00`` and ``plan94``
    routines from the ``erfa`` implementation of the Standards Of Fundamental
    Astronomy library.

    Notes
    -----
    Any file required will be downloaded (and cached) when the state is set.
    The default Satellite Planet Kernel (SPK) file from NASA JPL (de430) is
    ~120MB, and covers years ~1550-2650 CE [1]_.  The smaller de432s file is
    ~10MB, and covers years 1950-2050 [2]_ (and similarly for the newer de440
    and de440s).  Older versions of the JPL ephemerides (such as the widely
    used de200) can be used via their URL [3]_.

    .. [1] https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aareadme_de430-de431.txt
    .. [2] https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aareadme_de432s.txt
    .. [3] https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/
    """
    _value = 'builtin'
    _kernel = None

    @classmethod
    def validate(cls, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return cls._value
        cls.get_kernel(value)
        return value

    @classmethod
    def get_kernel(cls, value):
        if False:
            i = 10
            return i + 15
        if cls._kernel is None or cls._kernel.origin != value:
            if cls._kernel is not None:
                cls._kernel.daf.file.close()
                cls._kernel = None
            kernel = _get_kernel(value)
            if kernel is not None:
                kernel.origin = value
            cls._kernel = kernel
        return cls._kernel

    @classproperty
    def kernel(cls):
        if False:
            while True:
                i = 10
        return cls.get_kernel(cls._value)

    @classproperty
    def bodies(cls):
        if False:
            i = 10
            return i + 15
        if cls._value is None:
            return None
        if cls._value.lower() == 'builtin':
            return ('earth', 'sun', 'moon') + tuple(PLAN94_BODY_NAME_TO_PLANET_INDEX.keys())
        else:
            return tuple(BODY_NAME_TO_KERNEL_SPEC.keys())

def _get_kernel(value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Try importing jplephem, download/retrieve from cache the Satellite Planet\n    Kernel corresponding to the given ephemeris.\n    '
    if value is None or value.lower() == 'builtin':
        return None
    try:
        from jplephem.spk import SPK
    except ImportError:
        raise ImportError('Solar system JPL ephemeris calculations require the jplephem package (https://pypi.org/project/jplephem/)')
    if value.lower() == 'jpl':
        value = DEFAULT_JPL_EPHEMERIS
    if re.compile('de[0-9][0-9][0-9]s?').match(value.lower()):
        value = f'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/{value.lower():s}.bsp'
    elif os.path.isfile(value):
        return SPK.open(value)
    else:
        try:
            urlparse(value)
        except Exception:
            raise ValueError(f'{value} was not one of the standard strings and could not be parsed as a file path or URL')
    return SPK.open(download_file(value, cache=True))

def _get_body_barycentric_posvel(body, time, ephemeris=None, get_velocity=True):
    if False:
        return 10
    'Calculate the barycentric position (and velocity) of a solar system body.\n\n    Parameters\n    ----------\n    body : str or other\n        The solar system body for which to calculate positions.  Can also be a\n        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL\n        kernel.\n    time : `~astropy.time.Time`\n        Time of observation.\n    ephemeris : str, optional\n        Ephemeris to use.  By default, use the one set with\n        ``astropy.coordinates.solar_system_ephemeris.set``\n    get_velocity : bool, optional\n        Whether or not to calculate the velocity as well as the position.\n\n    Returns\n    -------\n    position : `~astropy.coordinates.CartesianRepresentation` or tuple\n        Barycentric (ICRS) position or tuple of position and velocity.\n\n    Notes\n    -----\n    Whether or not velocities are calculated makes little difference for the\n    built-in ephemerides, but for most JPL ephemeris files, the execution time\n    roughly doubles.\n    '
    default_kernel = ephemeris is None or ephemeris is solar_system_ephemeris._value
    kernel = None
    try:
        if default_kernel:
            if solar_system_ephemeris.get() is None:
                raise ValueError(_EPHEMERIS_NOTE)
            kernel = solar_system_ephemeris.kernel
        else:
            kernel = _get_kernel(ephemeris)
        (jd1, jd2) = get_jd12(time, 'tdb')
        if kernel is None:
            body = body.lower()
            (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2)
            if body == 'earth':
                body_pv_bary = earth_pv_bary
            elif body == 'moon':
                moon_pv_geo = erfa.moon98(jd1, jd2)
                body_pv_bary = erfa.pvppv(moon_pv_geo, earth_pv_bary)
            else:
                sun_pv_bary = erfa.pvmpv(earth_pv_bary, earth_pv_helio)
                if body == 'sun':
                    body_pv_bary = sun_pv_bary
                else:
                    try:
                        body_index = PLAN94_BODY_NAME_TO_PLANET_INDEX[body]
                    except KeyError:
                        raise KeyError(f"{body}'s position and velocity cannot be calculated with the '{ephemeris}' ephemeris.")
                    body_pv_helio = erfa.plan94(jd1, jd2, body_index)
                    body_pv_bary = erfa.pvppv(body_pv_helio, sun_pv_bary)
            body_pos_bary = CartesianRepresentation(body_pv_bary['p'], unit=u.au, xyz_axis=-1, copy=False)
            if get_velocity:
                body_vel_bary = CartesianRepresentation(body_pv_bary['v'], unit=u.au / u.day, xyz_axis=-1, copy=False)
        else:
            if isinstance(body, str):
                try:
                    kernel_spec = BODY_NAME_TO_KERNEL_SPEC[body.lower()]
                except KeyError:
                    raise KeyError(f"{body}'s position cannot be calculated with the {ephemeris} ephemeris.")
            else:
                kernel_spec = body
            jd1_shape = getattr(jd1, 'shape', ())
            if len(jd1_shape) > 1:
                (jd1, jd2) = (jd1.ravel(), jd2.ravel())
            body_posvel_bary = np.zeros((2 if get_velocity else 1, 3) + getattr(jd1, 'shape', ()))
            for pair in kernel_spec:
                spk = kernel[pair]
                if spk.data_type == 3:
                    posvel = spk.compute(jd1, jd2)
                    if get_velocity:
                        body_posvel_bary += posvel.reshape(body_posvel_bary.shape)
                    else:
                        body_posvel_bary[0] += posvel[:3]
                else:
                    for (body_p_or_v, p_or_v) in zip(body_posvel_bary, spk.generate(jd1, jd2)):
                        body_p_or_v += p_or_v
            body_posvel_bary.shape = body_posvel_bary.shape[:2] + jd1_shape
            body_pos_bary = CartesianRepresentation(body_posvel_bary[0], unit=u.km, copy=False)
            if get_velocity:
                body_vel_bary = CartesianRepresentation(body_posvel_bary[1], unit=u.km / u.day, copy=False)
        return (body_pos_bary, body_vel_bary) if get_velocity else body_pos_bary
    finally:
        if not default_kernel and kernel is not None:
            kernel.daf.file.close()

def get_body_barycentric_posvel(body, time, ephemeris=None):
    if False:
        while True:
            i = 10
    'Calculate the barycentric position and velocity of a solar system body.\n\n    Parameters\n    ----------\n    body : str or list of tuple\n        The solar system body for which to calculate positions.  Can also be a\n        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL\n        kernel.\n    time : `~astropy.time.Time`\n        Time of observation.\n    ephemeris : str, optional\n        Ephemeris to use.  By default, use the one set with\n        ``astropy.coordinates.solar_system_ephemeris.set``\n\n    Returns\n    -------\n    position, velocity : tuple of `~astropy.coordinates.CartesianRepresentation`\n        Tuple of barycentric (ICRS) position and velocity.\n\n    See Also\n    --------\n    get_body_barycentric : to calculate position only.\n        This is faster by about a factor two for JPL kernels, but has no\n        speed advantage for the built-in ephemeris.\n\n    Notes\n    -----\n    {_EPHEMERIS_NOTE}\n    '
    return _get_body_barycentric_posvel(body, time, ephemeris)

def get_body_barycentric(body, time, ephemeris=None):
    if False:
        i = 10
        return i + 15
    'Calculate the barycentric position of a solar system body.\n\n    Parameters\n    ----------\n    body : str or list of tuple\n        The solar system body for which to calculate positions.  Can also be a\n        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL\n        kernel.\n    time : `~astropy.time.Time`\n        Time of observation.\n    ephemeris : str, optional\n        Ephemeris to use.  By default, use the one set with\n        ``astropy.coordinates.solar_system_ephemeris.set``\n\n    Returns\n    -------\n    position : `~astropy.coordinates.CartesianRepresentation`\n        Barycentric (ICRS) position of the body in cartesian coordinates\n\n    See Also\n    --------\n    get_body_barycentric_posvel : to calculate both position and velocity.\n\n    Notes\n    -----\n    {_EPHEMERIS_NOTE}\n    '
    return _get_body_barycentric_posvel(body, time, ephemeris, get_velocity=False)

def _get_apparent_body_position(body, time, ephemeris, obsgeoloc=None):
    if False:
        while True:
            i = 10
    'Calculate the apparent position of body ``body`` relative to Earth.\n\n    This corrects for the light-travel time to the object.\n\n    Parameters\n    ----------\n    body : str or other\n        The solar system body for which to calculate positions.  Can also be a\n        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL\n        kernel.\n    time : `~astropy.time.Time`\n        Time of observation.\n    ephemeris : str, optional\n        Ephemeris to use.  By default, use the one set with\n        ``~astropy.coordinates.solar_system_ephemeris.set``\n    obsgeoloc : `~astropy.coordinates.CartesianRepresentation`, optional\n        The GCRS position of the observer\n\n    Returns\n    -------\n    cartesian_position : `~astropy.coordinates.CartesianRepresentation`\n        Barycentric (ICRS) apparent position of the body in cartesian coordinates\n\n    Notes\n    -----\n    {_EPHEMERIS_NOTE}\n    '
    if ephemeris is None:
        ephemeris = solar_system_ephemeris.get()
    delta_light_travel_time = 20.0 * u.s
    emitted_time = time
    light_travel_time = 0.0 * u.s
    earth_loc = get_body_barycentric('earth', time, ephemeris)
    if obsgeoloc is not None:
        earth_loc += obsgeoloc
    while np.any(np.fabs(delta_light_travel_time) > 1e-08 * u.s):
        body_loc = get_body_barycentric(body, emitted_time, ephemeris)
        earth_distance = (body_loc - earth_loc).norm()
        delta_light_travel_time = light_travel_time - earth_distance / speed_of_light
        light_travel_time = earth_distance / speed_of_light
        emitted_time = time - light_travel_time
    return get_body_barycentric(body, emitted_time, ephemeris)

def get_body(body, time, location=None, ephemeris=None):
    if False:
        return 10
    "\n    Get a `~astropy.coordinates.SkyCoord` for a solar system body as observed\n    from a location on Earth in the `~astropy.coordinates.GCRS` reference\n    system.\n\n    Parameters\n    ----------\n    body : str or list of tuple\n        The solar system body for which to calculate positions.  Can also be a\n        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL\n        kernel.\n    time : `~astropy.time.Time`\n        Time of observation.\n    location : `~astropy.coordinates.EarthLocation`, optional\n        Location of observer on the Earth.  If not given, will be taken from\n        ``time`` (if not present, a geocentric observer will be assumed).\n    ephemeris : str, optional\n        Ephemeris to use.  If not given, use the one set with\n        ``astropy.coordinates.solar_system_ephemeris.set`` (which is\n        set to 'builtin' by default).\n\n    Returns\n    -------\n    skycoord : `~astropy.coordinates.SkyCoord`\n        GCRS Coordinate for the body\n\n    Notes\n    -----\n    The coordinate returned is the apparent position, which is the position of\n    the body at time *t* minus the light travel time from the *body* to the\n    observing *location*.\n\n    {_EPHEMERIS_NOTE}\n    "
    if location is None:
        location = time.location
    if location is not None:
        (obsgeoloc, obsgeovel) = location.get_gcrs_posvel(time)
    else:
        (obsgeoloc, obsgeovel) = (None, None)
    cartrep = _get_apparent_body_position(body, time, ephemeris, obsgeoloc)
    icrs = ICRS(cartrep)
    gcrs = icrs.transform_to(GCRS(obstime=time, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel))
    return SkyCoord(gcrs)

@deprecated('5.3', alternative='get_body("moon")')
def get_moon(time, location=None, ephemeris=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get a `~astropy.coordinates.SkyCoord` for the Earth's Moon as observed\n    from a location on Earth in the `~astropy.coordinates.GCRS` reference\n    system.\n\n    Parameters\n    ----------\n    time : `~astropy.time.Time`\n        Time of observation\n    location : `~astropy.coordinates.EarthLocation`\n        Location of observer on the Earth. If none is supplied, taken from\n        ``time`` (if not present, a geocentric observer will be assumed).\n    ephemeris : str, optional\n        Ephemeris to use.  If not given, use the one set with\n        ``astropy.coordinates.solar_system_ephemeris.set`` (which is\n        set to 'builtin' by default).\n\n    Returns\n    -------\n    skycoord : `~astropy.coordinates.SkyCoord`\n        GCRS Coordinate for the Moon\n\n    Notes\n    -----\n    The coordinate returned is the apparent position, which is the position of\n    the moon at time *t* minus the light travel time from the moon to the\n    observing *location*.\n\n    {_EPHEMERIS_NOTE}\n    "
    return get_body('moon', time, location=location, ephemeris=ephemeris)
for f in [f for f in locals().values() if callable(f) and f.__doc__ is not None and ('{_EPHEMERIS_NOTE}' in f.__doc__)]:
    f.__doc__ = f.__doc__.format(_EPHEMERIS_NOTE=indent(_EPHEMERIS_NOTE)[4:])