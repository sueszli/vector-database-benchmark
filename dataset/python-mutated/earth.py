import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple
from warnings import warn
import numpy as np
from astropy import constants as consts
from astropy import units as u
from astropy.units.quantity import QuantityInfoBase
from astropy.utils import data
from astropy.utils.exceptions import AstropyUserWarning
from .angles import Angle, Latitude, Longitude
from .errors import UnknownSiteException
from .matrix_utilities import matrix_transpose
from .representation import CartesianDifferential, CartesianRepresentation
from .representation.geodetic import ELLIPSOIDS
__all__ = ['EarthLocation']

class GeodeticLocation(NamedTuple):
    """A namedtuple for geodetic coordinates.

    The longitude is increasing to the east, so west longitudes are negative.
    """
    lon: Longitude
    'The longitude, increasting to the east.'
    lat: Latitude
    'The latitude.'
    height: u.Quantity
    'The height above the reference ellipsoid.'
OMEGA_EARTH = (1.0027378119113546 * u.cycle / u.day).to(1 / u.s, u.dimensionless_angles())
"\nRotational velocity of Earth, following SOFA's pvtob.\n\nIn UT1 seconds, this would be 2 pi / (24 * 3600), but we need the value\nin SI seconds, so multiply by the ratio of stellar to solar day.\nSee Explanatory Supplement to the Astronomical Almanac, ed. P. Kenneth\nSeidelmann (1992), University Science Books. The constant is the\nconventional, exact one (IERS conventions 2003); see\nhttp://hpiers.obspm.fr/eop-pc/index.php?index=constants.\n"

def _check_ellipsoid(ellipsoid=None, default='WGS84'):
    if False:
        while True:
            i = 10
    if ellipsoid is None:
        ellipsoid = default
    if ellipsoid not in ELLIPSOIDS:
        raise ValueError(f'Ellipsoid {ellipsoid} not among known ones ({ELLIPSOIDS})')
    return ellipsoid

def _get_json_result(url, err_str, use_google):
    if False:
        for i in range(10):
            print('nop')
    from .name_resolve import NameResolveError
    try:
        resp = urllib.request.urlopen(url, timeout=data.conf.remote_timeout)
        resp_data = json.loads(resp.read().decode('utf8'))
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise NameResolveError(err_str.format(msg='connection timed out')) from e
        else:
            raise NameResolveError(err_str.format(msg=e.reason)) from e
    except socket.timeout:
        raise NameResolveError(err_str.format(msg='connection timed out'))
    if use_google:
        results = resp_data.get('results', [])
        if resp_data.get('status', None) != 'OK':
            raise NameResolveError(err_str.format(msg='unknown failure with Google API'))
    else:
        results = resp_data
    if not results:
        raise NameResolveError(err_str.format(msg='no results returned'))
    return results

class EarthLocationInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    _represent_as_dict_attrs = ('x', 'y', 'z', 'ellipsoid')

    def _construct_from_dict(self, map):
        if False:
            return 10
        ellipsoid = map.pop('ellipsoid')
        out = self._parent_cls(**map)
        out.ellipsoid = ellipsoid
        return out

    def new_like(self, cols, length, metadata_conflicts='warn', name=None):
        if False:
            return 10
        "\n        Return a new EarthLocation instance which is consistent with the\n        input ``cols`` and has ``length`` rows.\n\n        This is intended for creating an empty column object whose elements can\n        be set in-place for table operations like join or vstack.\n\n        Parameters\n        ----------\n        cols : list\n            List of input columns\n        length : int\n            Length of the output column object\n        metadata_conflicts : str ('warn'|'error'|'silent')\n            How to handle metadata conflicts\n        name : str\n            Output column name\n\n        Returns\n        -------\n        col : EarthLocation (or subclass)\n            Empty instance of this class consistent with ``cols``\n        "
        attrs = self.merge_cols_attributes(cols, metadata_conflicts, name, ('meta', 'format', 'description'))
        attrs.pop('dtype')
        shape = (length,) + attrs.pop('shape')
        data = u.Quantity(np.zeros(shape=shape, dtype=cols[0].dtype), unit=cols[0].unit, copy=False)
        map = {key: data[key] if key in 'xyz' else getattr(cols[-1], key) for key in self._represent_as_dict_attrs}
        out = self._construct_from_dict(map)
        for (attr, value) in attrs.items():
            setattr(out.info, attr, value)
        return out

class EarthLocation(u.Quantity):
    """
    Location on the Earth.

    Initialization is first attempted assuming geocentric (x, y, z) coordinates
    are given; if that fails, another attempt is made assuming geodetic
    coordinates (longitude, latitude, height above a reference ellipsoid).
    When using the geodetic forms, Longitudes are measured increasing to the
    east, so west longitudes are negative. Internally, the coordinates are
    stored as geocentric.

    To ensure a specific type of coordinates is used, use the corresponding
    class methods (`from_geocentric` and `from_geodetic`) or initialize the
    arguments with names (``x``, ``y``, ``z`` for geocentric; ``lon``, ``lat``,
    ``height`` for geodetic).  See the class methods for details.


    Notes
    -----
    This class fits into the coordinates transformation framework in that it
    encodes a position on the `~astropy.coordinates.ITRS` frame.  To get a
    proper `~astropy.coordinates.ITRS` object from this object, use the ``itrs``
    property.
    """
    _ellipsoid = 'WGS84'
    _location_dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': [np.float64] * 3})
    _array_dtype = np.dtype((np.float64, (3,)))
    _site_registry = None
    info = EarthLocationInfo()

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], EarthLocation):
            return args[0].copy()
        try:
            self = cls.from_geocentric(*args, **kwargs)
        except (u.UnitsError, TypeError) as exc_geocentric:
            try:
                self = cls.from_geodetic(*args, **kwargs)
            except Exception as exc_geodetic:
                raise TypeError(f'Coordinates could not be parsed as either geocentric or geodetic, with respective exceptions "{exc_geocentric}" and "{exc_geodetic}"')
        return self

    @classmethod
    def from_geocentric(cls, x, y, z, unit=None):
        if False:
            i = 10
            return i + 15
        '\n        Location on Earth, initialized from geocentric coordinates.\n\n        Parameters\n        ----------\n        x, y, z : `~astropy.units.Quantity` or array-like\n            Cartesian coordinates.  If not quantities, ``unit`` should be given.\n        unit : unit-like or None\n            Physical unit of the coordinate values.  If ``x``, ``y``, and/or\n            ``z`` are quantities, they will be converted to this unit.\n\n        Raises\n        ------\n        astropy.units.UnitsError\n            If the units on ``x``, ``y``, and ``z`` do not match or an invalid\n            unit is given.\n        ValueError\n            If the shapes of ``x``, ``y``, and ``z`` do not match.\n        TypeError\n            If ``x`` is not a `~astropy.units.Quantity` and no unit is given.\n        '
        if unit is None:
            try:
                unit = x.unit
            except AttributeError:
                raise TypeError('Geocentric coordinates should be Quantities unless an explicit unit is given.') from None
        else:
            unit = u.Unit(unit)
        if unit.physical_type != 'length':
            raise u.UnitsError('Geocentric coordinates should be in units of length.')
        try:
            x = u.Quantity(x, unit, copy=False)
            y = u.Quantity(y, unit, copy=False)
            z = u.Quantity(z, unit, copy=False)
        except u.UnitsError:
            raise u.UnitsError('Geocentric coordinate units should all be consistent.')
        (x, y, z) = np.broadcast_arrays(x, y, z, subok=True)
        struc = np.empty_like(x, dtype=cls._location_dtype)
        (struc['x'], struc['y'], struc['z']) = (x, y, z)
        return super().__new__(cls, struc, unit, copy=False)

    @classmethod
    def from_geodetic(cls, lon, lat, height=0.0, ellipsoid=None):
        if False:
            return 10
        "\n        Location on Earth, initialized from geodetic coordinates.\n\n        Parameters\n        ----------\n        lon : `~astropy.coordinates.Longitude` or float\n            Earth East longitude.  Can be anything that initialises an\n            `~astropy.coordinates.Angle` object (if float, in degrees).\n        lat : `~astropy.coordinates.Latitude` or float\n            Earth latitude.  Can be anything that initialises an\n            `~astropy.coordinates.Latitude` object (if float, in degrees).\n        height : `~astropy.units.Quantity` ['length'] or float, optional\n            Height above reference ellipsoid (if float, in meters; default: 0).\n        ellipsoid : str, optional\n            Name of the reference ellipsoid to use (default: 'WGS84').\n            Available ellipsoids are:  'WGS84', 'GRS80', 'WGS72'.\n\n        Raises\n        ------\n        astropy.units.UnitsError\n            If the units on ``lon`` and ``lat`` are inconsistent with angular\n            ones, or that on ``height`` with a length.\n        ValueError\n            If ``lon``, ``lat``, and ``height`` do not have the same shape, or\n            if ``ellipsoid`` is not recognized as among the ones implemented.\n\n        Notes\n        -----\n        For the conversion to geocentric coordinates, the ERFA routine\n        ``gd2gc`` is used.  See https://github.com/liberfa/erfa\n        "
        ellipsoid = _check_ellipsoid(ellipsoid, default=cls._ellipsoid)
        lon = Angle(lon, u.degree, copy=False).wrap_at(180 * u.degree)
        lat = Latitude(lat, u.degree, copy=False)
        if not isinstance(height, u.Quantity):
            height = u.Quantity(height, u.m, copy=False)
        geodetic = ELLIPSOIDS[ellipsoid](lon, lat, height, copy=False)
        xyz = geodetic.to_cartesian().get_xyz(xyz_axis=-1) << height.unit
        self = xyz.view(cls._location_dtype, cls).reshape(geodetic.shape)
        self._ellipsoid = ellipsoid
        return self

    @classmethod
    def of_site(cls, site_name, *, refresh_cache=False):
        if False:
            return 10
        "\n        Return an object of this class for a known observatory/site by name.\n\n        This is intended as a quick convenience function to get basic site\n        information, not a fully-featured exhaustive registry of observatories\n        and all their properties.\n\n        Additional information about the site is stored in the ``.info.meta``\n        dictionary of sites obtained using this method (see the examples below).\n\n        .. note::\n            This function is meant to access the site registry from the astropy\n            data server, which is saved in the user's local cache.  If you would\n            like a site to be added there, issue a pull request to the\n            `astropy-data repository <https://github.com/astropy/astropy-data>`_ .\n            If the cache already exists the function will use it even if the\n            version in the astropy-data repository has been updated unless the\n            ``refresh_cache=True`` option is used.  If there is no cache and the\n            online version cannot be reached, this function falls back on a\n            built-in list, which currently only contains the Greenwich Royal\n            Observatory as an example case.\n\n        Parameters\n        ----------\n        site_name : str\n            Name of the observatory (case-insensitive).\n        refresh_cache : bool, optional\n            If `True`, force replacement of the cached registry with a\n            newly downloaded version.  (Default: `False`)\n\n            .. versionadded:: 5.3\n\n        Returns\n        -------\n        site : `~astropy.coordinates.EarthLocation` (or subclass) instance\n            The location of the observatory. The returned class will be the same\n            as this class.\n\n        Examples\n        --------\n        >>> from astropy.coordinates import EarthLocation\n        >>> keck = EarthLocation.of_site('Keck Observatory')  # doctest: +REMOTE_DATA\n        >>> keck.geodetic  # doctest: +REMOTE_DATA +FLOAT_CMP\n        GeodeticLocation(lon=<Longitude -155.47833333 deg>, lat=<Latitude 19.82833333 deg>, height=<Quantity 4160. m>)\n        >>> keck.info  # doctest: +REMOTE_DATA\n        name = W. M. Keck Observatory\n        dtype = (float64, float64, float64)\n        unit = m\n        class = EarthLocation\n        n_bad = 0\n        >>> keck.info.meta  # doctest: +REMOTE_DATA\n        {'source': 'IRAF Observatory Database', 'timezone': 'US/Hawaii'}\n\n        See Also\n        --------\n        get_site_names : the list of sites that this function can access\n        "
        registry = cls._get_site_registry(force_download=refresh_cache)
        try:
            el = registry[site_name]
        except UnknownSiteException as e:
            raise UnknownSiteException(e.site, 'EarthLocation.get_site_names', close_names=e.close_names) from e
        if cls is el.__class__:
            return el
        else:
            newel = cls.from_geodetic(*el.to_geodetic())
            newel.info.name = el.info.name
            return newel

    @classmethod
    def of_address(cls, address, get_height=False, google_api_key=None):
        if False:
            while True:
                i = 10
        "\n        Return an object of this class for a given address by querying either\n        the OpenStreetMap Nominatim tool [1]_ (default) or the Google geocoding\n        API [2]_, which requires a specified API key.\n\n        This is intended as a quick convenience function to get easy access to\n        locations. If you need to specify a precise location, you should use the\n        initializer directly and pass in a longitude, latitude, and elevation.\n\n        In the background, this just issues a web query to either of\n        the APIs noted above. This is not meant to be abused! Both\n        OpenStreetMap and Google use IP-based query limiting and will ban your\n        IP if you send more than a few thousand queries per hour [2]_.\n\n        .. warning::\n            If the query returns more than one location (e.g., searching on\n            ``address='springfield'``), this function will use the **first**\n            returned location.\n\n        Parameters\n        ----------\n        address : str\n            The address to get the location for. As per the Google maps API,\n            this can be a fully specified street address (e.g., 123 Main St.,\n            New York, NY) or a city name (e.g., Danbury, CT), or etc.\n        get_height : bool, optional\n            This only works when using the Google API! See the ``google_api_key``\n            block below. Use the retrieved location to perform a second query to\n            the Google maps elevation API to retrieve the height of the input\n            address [3]_.\n        google_api_key : str, optional\n            A Google API key with the Geocoding API and (optionally) the\n            elevation API enabled. See [4]_ for more information.\n\n        Returns\n        -------\n        location : `~astropy.coordinates.EarthLocation` (or subclass) instance\n            The location of the input address.\n            Will be type(this class)\n\n        References\n        ----------\n        .. [1] https://nominatim.openstreetmap.org/\n        .. [2] https://developers.google.com/maps/documentation/geocoding/start\n        .. [3] https://developers.google.com/maps/documentation/elevation/start\n        .. [4] https://developers.google.com/maps/documentation/geocoding/get-api-key\n\n        "
        use_google = google_api_key is not None
        if not use_google and get_height:
            raise ValueError('Currently, `get_height` only works when using the Google geocoding API, which requires passing a Google API key with `google_api_key`. See: https://developers.google.com/maps/documentation/geocoding/get-api-key for information on obtaining an API key.')
        if use_google:
            pars = urllib.parse.urlencode({'address': address, 'key': google_api_key})
            geo_url = f'https://maps.googleapis.com/maps/api/geocode/json?{pars}'
        else:
            pars = urllib.parse.urlencode({'q': address, 'format': 'json'})
            geo_url = f'https://nominatim.openstreetmap.org/search?{pars}'
        err_str = f"Unable to retrieve coordinates for address '{address}'; {{msg}}"
        geo_result = _get_json_result(geo_url, err_str=err_str, use_google=use_google)
        if use_google:
            loc = geo_result[0]['geometry']['location']
            lat = loc['lat']
            lon = loc['lng']
        else:
            loc = geo_result[0]
            lat = float(loc['lat'])
            lon = float(loc['lon'])
        if get_height:
            pars = {'locations': f'{lat:.8f},{lon:.8f}', 'key': google_api_key}
            pars = urllib.parse.urlencode(pars)
            ele_url = f'https://maps.googleapis.com/maps/api/elevation/json?{pars}'
            err_str = f"Unable to retrieve elevation for address '{address}'; {{msg}}"
            ele_result = _get_json_result(ele_url, err_str=err_str, use_google=use_google)
            height = ele_result[0]['elevation'] * u.meter
        else:
            height = 0.0
        return cls.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=height)

    @classmethod
    def get_site_names(cls, *, refresh_cache=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get list of names of observatories for use with\n        `~astropy.coordinates.EarthLocation.of_site`.\n\n        .. note::\n            This function is meant to access the site registry from the astropy\n            data server, which is saved in the user's local cache.  If you would\n            like a site to be added there, issue a pull request to the\n            `astropy-data repository <https://github.com/astropy/astropy-data>`_ .\n            If the cache already exists the function will use it even if the\n            version in the astropy-data repository has been updated unless the\n            ``refresh_cache=True`` option is used.  If there is no cache and the\n            online version cannot be reached, this function falls back on a\n            built-in list, which currently only contains the Greenwich Royal\n            Observatory as an example case.\n\n        Parameters\n        ----------\n        refresh_cache : bool, optional\n            If `True`, force replacement of the cached registry with a\n            newly downloaded version.  (Default: `False`)\n\n            .. versionadded:: 5.3\n\n        Returns\n        -------\n        names : list of str\n            List of valid observatory names\n\n        See Also\n        --------\n        of_site : Gets the actual location object for one of the sites names\n            this returns.\n        "
        return cls._get_site_registry(force_download=refresh_cache).names

    @classmethod
    def _get_site_registry(cls, force_download=False, force_builtin=False):
        if False:
            print('Hello World!')
        '\n        Gets the site registry.  The first time this either downloads or loads\n        from the data file packaged with astropy.  Subsequent calls will use the\n        cached version unless explicitly overridden.\n\n        Parameters\n        ----------\n        force_download : bool or str\n            If not False, force replacement of the cached registry with a\n            downloaded version. If a str, that will be used as the URL to\n            download from (if just True, the default URL will be used).\n        force_builtin : bool\n            If True, load from the data file bundled with astropy and set the\n            cache to that.\n\n        Returns\n        -------\n        reg : astropy.coordinates.sites.SiteRegistry\n        '
        from .sites import get_builtin_sites, get_downloaded_sites
        if force_builtin and force_download:
            raise ValueError('Cannot have both force_builtin and force_download True')
        if force_builtin:
            cls._site_registry = get_builtin_sites()
        elif force_download or not cls._site_registry:
            try:
                if isinstance(force_download, str):
                    cls._site_registry = get_downloaded_sites(force_download)
                else:
                    cls._site_registry = get_downloaded_sites()
            except OSError:
                if force_download:
                    raise
                msg = "Could not access the main site list. Falling back on the built-in version, which is rather limited. If you want to retry the download, use the option 'refresh_cache=True'."
                warn(msg, AstropyUserWarning)
                cls._site_registry = get_builtin_sites()
        return cls._site_registry

    @property
    def ellipsoid(self):
        if False:
            return 10
        'The default ellipsoid used to convert to geodetic coordinates.'
        return self._ellipsoid

    @ellipsoid.setter
    def ellipsoid(self, ellipsoid):
        if False:
            for i in range(10):
                print('nop')
        self._ellipsoid = _check_ellipsoid(ellipsoid)

    @property
    def geodetic(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert to geodetic coordinates for the default ellipsoid.'
        return self.to_geodetic()

    def to_geodetic(self, ellipsoid=None):
        if False:
            for i in range(10):
                print('nop')
        "Convert to geodetic coordinates.\n\n        Parameters\n        ----------\n        ellipsoid : str, optional\n            Reference ellipsoid to use.  Default is the one the coordinates\n            were initialized with.  Available are: 'WGS84', 'GRS80', 'WGS72'\n\n        Returns\n        -------\n        lon, lat, height : `~astropy.units.Quantity`\n            The tuple is a ``GeodeticLocation`` namedtuple and is comprised of\n            instances of `~astropy.coordinates.Longitude`,\n            `~astropy.coordinates.Latitude`, and `~astropy.units.Quantity`.\n\n        Raises\n        ------\n        ValueError\n            if ``ellipsoid`` is not recognized as among the ones implemented.\n\n        Notes\n        -----\n        For the conversion to geodetic coordinates, the ERFA routine\n        ``gc2gd`` is used.  See https://github.com/liberfa/erfa\n        "
        ellipsoid = _check_ellipsoid(ellipsoid, default=self.ellipsoid)
        xyz = self.view(self._array_dtype, u.Quantity)
        llh = CartesianRepresentation(xyz, xyz_axis=-1, copy=False).represent_as(ELLIPSOIDS[ellipsoid])
        return GeodeticLocation(Longitude(llh.lon, u.deg, wrap_angle=180 * u.deg, copy=False), llh.lat << u.deg, llh.height << self.unit)

    @property
    def lon(self):
        if False:
            for i in range(10):
                print('nop')
        'Longitude of the location, for the default ellipsoid.'
        return self.geodetic[0]

    @property
    def lat(self):
        if False:
            return 10
        'Latitude of the location, for the default ellipsoid.'
        return self.geodetic[1]

    @property
    def height(self):
        if False:
            return 10
        'Height of the location, for the default ellipsoid.'
        return self.geodetic[2]

    @property
    def geocentric(self):
        if False:
            i = 10
            return i + 15
        'Convert to a tuple with X, Y, and Z as quantities.'
        return self.to_geocentric()

    def to_geocentric(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert to a tuple with X, Y, and Z as quantities.'
        return (self.x, self.y, self.z)

    def get_itrs(self, obstime=None, location=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Generates an `~astropy.coordinates.ITRS` object with the location of\n        this object at the requested ``obstime``, either geocentric, or\n        topocentric relative to a given ``location``.\n\n        Parameters\n        ----------\n        obstime : `~astropy.time.Time` or None\n            The ``obstime`` to apply to the new `~astropy.coordinates.ITRS`, or\n            if None, the default ``obstime`` will be used.\n        location : `~astropy.coordinates.EarthLocation` or None\n            A possible observer's location, for a topocentric ITRS position.\n            If not given (default), a geocentric ITRS object will be created.\n\n        Returns\n        -------\n        itrs : `~astropy.coordinates.ITRS`\n            The new object in the ITRS frame, either geocentric or topocentric\n            relative to the given ``location``.\n        "
        if obstime and self.size == 1 and obstime.shape:
            self = np.broadcast_to(self, obstime.shape, subok=True)
        from .builtin_frames import ITRS
        if location is None:
            return ITRS(x=self.x, y=self.y, z=self.z, obstime=obstime)
        else:
            return ITRS(self.x - location.x, self.y - location.y, self.z - location.z, copy=False, obstime=obstime, location=location)
    itrs = property(get_itrs, doc='An `~astropy.coordinates.ITRS` object\n               for the location of this object at the\n               default ``obstime``.')

    def get_gcrs(self, obstime):
        if False:
            i = 10
            return i + 15
        'GCRS position with velocity at ``obstime`` as a GCRS coordinate.\n\n        Parameters\n        ----------\n        obstime : `~astropy.time.Time`\n            The ``obstime`` to calculate the GCRS position/velocity at.\n\n        Returns\n        -------\n        gcrs : `~astropy.coordinates.GCRS` instance\n            With velocity included.\n        '
        from .builtin_frames import GCRS
        (loc, vel) = self.get_gcrs_posvel(obstime)
        loc.differentials['s'] = CartesianDifferential.from_cartesian(vel)
        return GCRS(loc, obstime=obstime)

    def _get_gcrs_posvel(self, obstime, ref_to_itrs, gcrs_to_ref):
        if False:
            print('Hello World!')
        'Calculate GCRS position and velocity given transformation matrices.\n\n        The reference frame z axis must point to the Celestial Intermediate Pole\n        (as is the case for CIRS and TETE).\n\n        This private method is used in intermediate_rotation_transforms,\n        where some of the matrices are already available for the coordinate\n        transformation.\n\n        The method is faster by an order of magnitude than just adding a zero\n        velocity to ITRS and transforming to GCRS, because it avoids calculating\n        the velocity via finite differencing of the results of the transformation\n        at three separate times.\n        '
        ref_to_gcrs = matrix_transpose(gcrs_to_ref)
        itrs_to_gcrs = ref_to_gcrs @ matrix_transpose(ref_to_itrs)
        rot_vec_gcrs = CartesianRepresentation(ref_to_gcrs[..., 2] * OMEGA_EARTH, xyz_axis=-1, copy=False)
        itrs_cart = CartesianRepresentation(self.x, self.y, self.z, copy=False)
        pos = itrs_cart.transform(itrs_to_gcrs)
        vel = rot_vec_gcrs.cross(pos)
        return (pos, vel)

    def get_gcrs_posvel(self, obstime):
        if False:
            while True:
                i = 10
        '\n        Calculate the GCRS position and velocity of this object at the\n        requested ``obstime``.\n\n        Parameters\n        ----------\n        obstime : `~astropy.time.Time`\n            The ``obstime`` to calculate the GCRS position/velocity at.\n\n        Returns\n        -------\n        obsgeoloc : `~astropy.coordinates.CartesianRepresentation`\n            The GCRS position of the object\n        obsgeovel : `~astropy.coordinates.CartesianRepresentation`\n            The GCRS velocity of the object\n        '
        from .builtin_frames.intermediate_rotation_transforms import cirs_to_itrs_mat, gcrs_to_cirs_mat
        return self._get_gcrs_posvel(obstime, cirs_to_itrs_mat(obstime), gcrs_to_cirs_mat(obstime))

    def gravitational_redshift(self, obstime, bodies=['sun', 'jupiter', 'moon'], masses={}):
        if False:
            return 10
        'Return the gravitational redshift at this EarthLocation.\n\n        Calculates the gravitational redshift, of order 3 m/s, due to the\n        requested solar system bodies.\n\n        Parameters\n        ----------\n        obstime : `~astropy.time.Time`\n            The ``obstime`` to calculate the redshift at.\n\n        bodies : iterable, optional\n            The bodies (other than the Earth) to include in the redshift\n            calculation.  List elements should be any body name\n            `get_body_barycentric` accepts.  Defaults to Jupiter, the Sun, and\n            the Moon.  Earth is always included (because the class represents\n            an *Earth* location).\n\n        masses : dict[str, `~astropy.units.Quantity`], optional\n            The mass or gravitational parameters (G * mass) to assume for the\n            bodies requested in ``bodies``. Can be used to override the\n            defaults for the Sun, Jupiter, the Moon, and the Earth, or to\n            pass in masses for other bodies.\n\n        Returns\n        -------\n        redshift : `~astropy.units.Quantity`\n            Gravitational redshift in velocity units at given obstime.\n        '
        from .solar_system import get_body_barycentric
        bodies = list(bodies)
        if 'earth' in bodies:
            bodies.remove('earth')
        bodies.append('earth')
        _masses = {'sun': consts.GM_sun, 'jupiter': consts.GM_jup, 'moon': consts.G * 7.34767309e+22 * u.kg, 'earth': consts.GM_earth}
        _masses.update(masses)
        GMs = []
        M_GM_equivalency = (u.kg, u.Unit(consts.G * u.kg))
        for body in bodies:
            try:
                GMs.append(_masses[body].to(u.m ** 3 / u.s ** 2, [M_GM_equivalency]))
            except KeyError as err:
                raise KeyError(f'body "{body}" does not have a mass.') from err
            except u.UnitsError as exc:
                exc.args += ('"masses" argument values must be masses or gravitational parameters.',)
                raise
        positions = [get_body_barycentric(name, obstime) for name in bodies]
        distances = [(pos - positions[-1]).norm() for pos in positions[:-1]]
        distances.append(CartesianRepresentation(self.geocentric).norm())
        redshifts = [-GM / consts.c / distance for (GM, distance) in zip(GMs, distances)]
        return sum(redshifts[::-1])

    @property
    def x(self):
        if False:
            print('Hello World!')
        'The X component of the geocentric coordinates.'
        return self['x']

    @property
    def y(self):
        if False:
            print('Hello World!')
        'The Y component of the geocentric coordinates.'
        return self['y']

    @property
    def z(self):
        if False:
            print('Hello World!')
        'The Z component of the geocentric coordinates.'
        return self['z']

    def __getitem__(self, item):
        if False:
            return 10
        result = super().__getitem__(item)
        if result.dtype is self.dtype:
            return result.view(self.__class__)
        else:
            return result.view(u.Quantity)

    def __array_finalize__(self, obj):
        if False:
            i = 10
            return i + 15
        super().__array_finalize__(obj)
        if hasattr(obj, '_ellipsoid'):
            self._ellipsoid = obj._ellipsoid

    def __len__(self):
        if False:
            return 10
        if self.shape == ():
            raise IndexError('0-d EarthLocation arrays cannot be indexed')
        else:
            return super().__len__()

    def _to_value(self, unit, equivalencies=[]):
        if False:
            for i in range(10):
                print('nop')
        'Helper method for to and to_value.'
        array_view = self.view(self._array_dtype, np.ndarray)
        if equivalencies == []:
            equivalencies = self._equivalencies
        new_array = self.unit.to(unit, array_view, equivalencies=equivalencies)
        return new_array.view(self.dtype).reshape(self.shape)