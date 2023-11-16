import copy
import operator
import re
import warnings
import erfa
import numpy as np
from astropy import units as u
from astropy.constants import c as speed_of_light
from astropy.table import QTable
from astropy.time import Time
from astropy.utils import ShapedLikeNDArray
from astropy.utils.data_info import MixinInfo
from astropy.utils.exceptions import AstropyUserWarning
from .angles import Angle
from .baseframe import BaseCoordinateFrame, GenericFrame, frame_transform_graph
from .distances import Distance
from .representation import RadialDifferential, SphericalDifferential, SphericalRepresentation, UnitSphericalCosLatDifferential, UnitSphericalDifferential, UnitSphericalRepresentation
from .sky_coordinate_parsers import _get_frame_class, _get_frame_without_data, _parse_coordinate_data
__all__ = ['SkyCoord', 'SkyCoordInfo']

class SkyCoordInfo(MixinInfo):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    attrs_from_parent = {'unit'}
    _supports_indexing = False

    @staticmethod
    def default_format(val):
        if False:
            for i in range(10):
                print('nop')
        repr_data = val.info._repr_data
        formats = ['{0.' + compname + '.value:}' for compname in repr_data.components]
        return ','.join(formats).format(repr_data)

    @property
    def unit(self):
        if False:
            while True:
                i = 10
        repr_data = self._repr_data
        unit = ','.join((str(getattr(repr_data, comp).unit) or 'None' for comp in repr_data.components))
        return unit

    @property
    def _repr_data(self):
        if False:
            while True:
                i = 10
        if self._parent is None:
            return None
        sc = self._parent
        if issubclass(sc.representation_type, SphericalRepresentation) and isinstance(sc.data, UnitSphericalRepresentation):
            repr_data = sc.represent_as(sc.data.__class__, in_frame_units=True)
        else:
            repr_data = sc.represent_as(sc.representation_type, in_frame_units=True)
        return repr_data

    def _represent_as_dict(self):
        if False:
            for i in range(10):
                print('nop')
        sc = self._parent
        attrs = list(sc.representation_component_names)
        if isinstance(sc.data, UnitSphericalRepresentation):
            attrs = attrs[:-1]
        diff = sc.data.differentials.get('s')
        if diff is not None:
            diff_attrs = list(sc.get_representation_component_names('s'))
            if isinstance(diff, RadialDifferential):
                diff_attrs = diff_attrs[2:]
            elif isinstance(diff, (UnitSphericalDifferential, UnitSphericalCosLatDifferential)):
                diff_attrs = diff_attrs[:-1]
            attrs.extend(diff_attrs)
        attrs.extend(frame_transform_graph.frame_attributes.keys())
        out = super()._represent_as_dict(attrs)
        out['representation_type'] = sc.representation_type.get_name()
        out['frame'] = sc.frame.name
        return out

    def new_like(self, skycoords, length, metadata_conflicts='warn', name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a new SkyCoord instance which is consistent with the input\n        SkyCoord objects ``skycoords`` and has ``length`` rows.  Being\n        "consistent" is defined as being able to set an item from one to each of\n        the rest without any exception being raised.\n\n        This is intended for creating a new SkyCoord instance whose elements can\n        be set in-place for table operations like join or vstack.  This is used\n        when a SkyCoord object is used as a mixin column in an astropy Table.\n\n        The data values are not predictable and it is expected that the consumer\n        of the object will fill in all values.\n\n        Parameters\n        ----------\n        skycoords : list\n            List of input SkyCoord objects\n        length : int\n            Length of the output skycoord object\n        metadata_conflicts : str (\'warn\'|\'error\'|\'silent\')\n            How to handle metadata conflicts\n        name : str\n            Output name (sets output skycoord.info.name)\n\n        Returns\n        -------\n        skycoord : |SkyCoord| (or subclass)\n            Instance of this class consistent with ``skycoords``\n\n        '
        attrs = self.merge_cols_attributes(skycoords, metadata_conflicts, name, ('meta', 'description'))
        skycoord0 = skycoords[0]
        indexes = np.zeros(length, dtype=np.int64)
        out = skycoord0[indexes]
        for skycoord in skycoords[1:]:
            try:
                out[0] = skycoord[0]
            except Exception as err:
                raise ValueError('Input skycoords are inconsistent.') from err
        for attr in ('name', 'meta', 'description'):
            if attr in attrs:
                setattr(out.info, attr, attrs[attr])
        return out

class SkyCoord(ShapedLikeNDArray):
    """High-level object providing a flexible interface for celestial coordinate
    representation, manipulation, and transformation between systems.

    The |SkyCoord| class accepts a wide variety of inputs for initialization. At
    a minimum these must provide one or more celestial coordinate values with
    unambiguous units.  Inputs may be scalars or lists/tuples/arrays, yielding
    scalar or array coordinates (can be checked via ``SkyCoord.isscalar``).
    Typically one also specifies the coordinate frame, though this is not
    required. The general pattern for spherical representations is::

      SkyCoord(COORD, [FRAME], keyword_args ...)
      SkyCoord(LON, LAT, [FRAME], keyword_args ...)
      SkyCoord(LON, LAT, [DISTANCE], frame=FRAME, unit=UNIT, keyword_args ...)
      SkyCoord([FRAME], <lon_attr>=LON, <lat_attr>=LAT, keyword_args ...)

    It is also possible to input coordinate values in other representations
    such as cartesian or cylindrical.  In this case one includes the keyword
    argument ``representation_type='cartesian'`` (for example) along with data
    in ``x``, ``y``, and ``z``.

    See also: https://docs.astropy.org/en/stable/coordinates/

    Examples
    --------
    The examples below illustrate common ways of initializing a |SkyCoord|
    object.  For a complete description of the allowed syntax see the
    full coordinates documentation.  First some imports::

      >>> from astropy.coordinates import SkyCoord  # High-level coordinates
      >>> from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
      >>> from astropy.coordinates import Angle, Latitude, Longitude  # Angles
      >>> import astropy.units as u

    The coordinate values and frame specification can now be provided using
    positional and keyword arguments::

      >>> c = SkyCoord(10, 20, unit="deg")  # defaults to ICRS frame
      >>> c = SkyCoord([1, 2, 3], [-30, 45, 8], frame="icrs", unit="deg")  # 3 coords

      >>> coords = ["1:12:43.2 +31:12:43", "1 12 43.2 +31 12 43"]
      >>> c = SkyCoord(coords, frame=FK4, unit=(u.hourangle, u.deg), obstime="J1992.21")

      >>> c = SkyCoord("1h12m43.2s +1d12m43s", frame=Galactic)  # Units from string
      >>> c = SkyCoord(frame="galactic", l="1h12m43.2s", b="+1d12m43s")

      >>> ra = Longitude([1, 2, 3], unit=u.deg)  # Could also use Angle
      >>> dec = np.array([4.5, 5.2, 6.3]) * u.deg  # Astropy Quantity
      >>> c = SkyCoord(ra, dec, frame='icrs')
      >>> c = SkyCoord(frame=ICRS, ra=ra, dec=dec, obstime='2001-01-02T12:34:56')

      >>> c = FK4(1 * u.deg, 2 * u.deg)  # Uses defaults for obstime, equinox
      >>> c = SkyCoord(c, obstime='J2010.11', equinox='B1965')  # Override defaults

      >>> c = SkyCoord(w=0, u=1, v=2, unit='kpc', frame='galactic',
      ...              representation_type='cartesian')

      >>> c = SkyCoord([ICRS(ra=1*u.deg, dec=2*u.deg), ICRS(ra=3*u.deg, dec=4*u.deg)])

    Velocity components (proper motions or radial velocities) can also be
    provided in a similar manner::

      >>> c = SkyCoord(ra=1*u.deg, dec=2*u.deg, radial_velocity=10*u.km/u.s)

      >>> c = SkyCoord(ra=1*u.deg, dec=2*u.deg, pm_ra_cosdec=2*u.mas/u.yr, pm_dec=1*u.mas/u.yr)

    As shown, the frame can be a `~astropy.coordinates.BaseCoordinateFrame`
    class or the corresponding string alias -- lower-case versions of the
    class name that allow for creating a |SkyCoord| object and transforming
    frames without explicitly importing the frame classes.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame` class or string, optional
        Type of coordinate frame this |SkyCoord| should represent. Defaults to
        to ICRS if not given or given as None.
    unit : `~astropy.units.Unit`, string, or tuple of :class:`~astropy.units.Unit` or str, optional
        Units for supplied coordinate values.
        If only one unit is supplied then it applies to all values.
        Note that passing only one unit might lead to unit conversion errors
        if the coordinate values are expected to have mixed physical meanings
        (e.g., angles and distances).
    obstime : time-like, optional
        Time(s) of observation.
    equinox : time-like, optional
        Coordinate frame equinox time.
    representation_type : str or Representation class
        Specifies the representation, e.g. 'spherical', 'cartesian', or
        'cylindrical'.  This affects the positional args and other keyword args
        which must correspond to the given representation.
    copy : bool, optional
        If `True` (default), a copy of any coordinate data is made.  This
        argument can only be passed in as a keyword argument.
    **keyword_args
        Other keyword arguments as applicable for user-defined coordinate frames.
        Common options include:

        ra, dec : angle-like, optional
            RA and Dec for frames where ``ra`` and ``dec`` are keys in the
            frame's ``representation_component_names``, including ``ICRS``,
            ``FK5``, ``FK4``, and ``FK4NoETerms``.
        pm_ra_cosdec, pm_dec  : `~astropy.units.Quantity` ['angular speed'], optional
            Proper motion components, in angle per time units.
        l, b : angle-like, optional
            Galactic ``l`` and ``b`` for for frames where ``l`` and ``b`` are
            keys in the frame's ``representation_component_names``, including
            the ``Galactic`` frame.
        pm_l_cosb, pm_b : `~astropy.units.Quantity` ['angular speed'], optional
            Proper motion components in the `~astropy.coordinates.Galactic` frame,
            in angle per time units.
        x, y, z : float or `~astropy.units.Quantity` ['length'], optional
            Cartesian coordinates values
        u, v, w : float or `~astropy.units.Quantity` ['length'], optional
            Cartesian coordinates values for the Galactic frame.
        radial_velocity : `~astropy.units.Quantity` ['speed'], optional
            The component of the velocity along the line-of-sight (i.e., the
            radial direction), in velocity units.
    """
    info = SkyCoordInfo()

    def __init__(self, *args, copy=True, **kwargs):
        if False:
            print('Hello World!')
        self._extra_frameattr_names = set()
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (BaseCoordinateFrame, SkyCoord)):
            coords = args[0]
            if isinstance(coords, SkyCoord):
                self._extra_frameattr_names = coords._extra_frameattr_names
                self.info = coords.info
                for attr_name in self._extra_frameattr_names:
                    setattr(self, attr_name, getattr(coords, attr_name))
                coords = coords.frame
            if not coords.has_data:
                raise ValueError('Cannot initialize from a coordinate frame instance without coordinate data')
            if copy:
                self._sky_coord_frame = coords.copy()
            else:
                self._sky_coord_frame = coords
        else:
            (frame_cls, frame_kwargs) = _get_frame_without_data(args, kwargs)
            args = list(args)
            (skycoord_kwargs, components, info) = _parse_coordinate_data(frame_cls(**frame_kwargs), args, kwargs)
            for attr in skycoord_kwargs:
                setattr(self, attr, skycoord_kwargs[attr])
            if info is not None:
                self.info = info
            frame_kwargs.update(components)
            self._sky_coord_frame = frame_cls(copy=copy, **frame_kwargs)
            if not self._sky_coord_frame.has_data:
                raise ValueError('Cannot create a SkyCoord without data')

    @property
    def frame(self):
        if False:
            i = 10
            return i + 15
        return self._sky_coord_frame

    @property
    def representation_type(self):
        if False:
            print('Hello World!')
        return self.frame.representation_type

    @representation_type.setter
    def representation_type(self, value):
        if False:
            return 10
        self.frame.representation_type = value

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        return self.frame.shape

    def __eq__(self, value):
        if False:
            i = 10
            return i + 15
        'Equality operator for SkyCoord.\n\n        This implements strict equality and requires that the frames are\n        equivalent, extra frame attributes are equivalent, and that the\n        representation data are exactly equal.\n        '
        if isinstance(value, BaseCoordinateFrame):
            if value._data is None:
                raise ValueError('Can only compare SkyCoord to Frame with data')
            return self.frame == value
        if not isinstance(value, SkyCoord):
            return NotImplemented
        for attr in self._extra_frameattr_names | value._extra_frameattr_names:
            if not self.frame._frameattr_equiv(getattr(self, attr), getattr(value, attr)):
                raise ValueError(f"cannot compare: extra frame attribute '{attr}' is not equivalent (perhaps compare the frames directly to avoid this exception)")
        return self._sky_coord_frame == value._sky_coord_frame

    def __ne__(self, value):
        if False:
            return 10
        return np.logical_not(self == value)

    def _apply(self, method, *args, **kwargs):
        if False:
            return 10
        'Create a new instance, applying a method to the underlying data.\n\n        In typical usage, the method is any of the shape-changing methods for\n        `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those\n        picking particular elements (``__getitem__``, ``take``, etc.), which\n        are all defined in `~astropy.utils.shapes.ShapedLikeNDArray`. It will be\n        applied to the underlying arrays in the representation (e.g., ``x``,\n        ``y``, and ``z`` for `~astropy.coordinates.CartesianRepresentation`),\n        as well as to any frame attributes that have a shape, with the results\n        used to create a new instance.\n\n        Internally, it is also used to apply functions to the above parts\n        (in particular, `~numpy.broadcast_to`).\n\n        Parameters\n        ----------\n        method : str or callable\n            If str, it is the name of a method that is applied to the internal\n            ``components``. If callable, the function is applied.\n        *args\n            Any positional arguments for ``method``.\n        **kwargs : dict\n            Any keyword arguments for ``method``.\n        '

        def apply_method(value):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(value, ShapedLikeNDArray):
                return value._apply(method, *args, **kwargs)
            elif callable(method):
                return method(value, *args, **kwargs)
            else:
                return getattr(value, method)(*args, **kwargs)
        new = super().__new__(self.__class__)
        new._sky_coord_frame = self._sky_coord_frame._apply(method, *args, **kwargs)
        new._extra_frameattr_names = self._extra_frameattr_names.copy()
        for attr in self._extra_frameattr_names:
            value = getattr(self, attr)
            if getattr(value, 'shape', ()):
                value = apply_method(value)
            elif method == 'copy' or method == 'flatten':
                value = copy.copy(value)
            setattr(new, '_' + attr, value)
        if 'info' in self.__dict__:
            new.info = self.info
        return new

    def __setitem__(self, item, value):
        if False:
            while True:
                i = 10
        'Implement self[item] = value for SkyCoord.\n\n        The right hand ``value`` must be strictly consistent with self:\n        - Identical class\n        - Equivalent frames\n        - Identical representation_types\n        - Identical representation differentials keys\n        - Identical frame attributes\n        - Identical "extra" frame attributes (e.g. obstime for an ICRS coord)\n\n        With these caveats the setitem ends up as effectively a setitem on\n        the representation data.\n\n          self.frame.data[item] = value.frame.data\n        '
        if self.__class__ is not value.__class__:
            raise TypeError(f'can only set from object of same class: {self.__class__.__name__} vs. {value.__class__.__name__}')
        for attr in self._extra_frameattr_names | value._extra_frameattr_names:
            if not self.frame._frameattr_equiv(getattr(self, attr), getattr(value, attr)):
                raise ValueError(f'attribute {attr} is not equivalent')
        self._sky_coord_frame[item] = value._sky_coord_frame

    def insert(self, obj, values, axis=0):
        if False:
            print('Hello World!')
        '\n        Insert coordinate values before the given indices in the object and\n        return a new Frame object.\n\n        The values to be inserted must conform to the rules for in-place setting\n        of |SkyCoord| objects.\n\n        The API signature matches the ``np.insert`` API, but is more limited.\n        The specification of insert index ``obj`` must be a single integer,\n        and the ``axis`` must be ``0`` for simple insertion before the index.\n\n        Parameters\n        ----------\n        obj : int\n            Integer index before which ``values`` is inserted.\n        values : array-like\n            Value(s) to insert.  If the type of ``values`` is different\n            from that of quantity, ``values`` is converted to the matching type.\n        axis : int, optional\n            Axis along which to insert ``values``.  Default is 0, which is the\n            only allowed value and will insert a row.\n\n        Returns\n        -------\n        out : `~astropy.coordinates.SkyCoord` instance\n            New coordinate object with inserted value(s)\n\n        '
        try:
            idx0 = operator.index(obj)
        except TypeError:
            raise TypeError('obj arg must be an integer')
        if axis != 0:
            raise ValueError('axis must be 0')
        if not self.shape:
            raise TypeError(f'cannot insert into scalar {self.__class__.__name__} object')
        if abs(idx0) > len(self):
            raise IndexError(f'index {idx0} is out of bounds for axis 0 with size {len(self)}')
        if idx0 < 0:
            idx0 = len(self) + idx0
        n_values = len(values) if values.shape else 1
        out = self.__class__.info.new_like([self], len(self) + n_values, name=self.info.name)
        out[:idx0] = self[:idx0]
        out[idx0:idx0 + n_values] = values
        out[idx0 + n_values:] = self[idx0:]
        return out

    def is_transformable_to(self, new_frame):
        if False:
            i = 10
            return i + 15
        "\n        Determines if this coordinate frame can be transformed to another\n        given frame.\n\n        Parameters\n        ----------\n        new_frame : frame class, frame object, or str\n            The proposed frame to transform into.\n\n        Returns\n        -------\n        transformable : bool or str\n            `True` if this can be transformed to ``new_frame``, `False` if\n            not, or the string 'same' if ``new_frame`` is the same system as\n            this object but no transformation is defined.\n\n        Notes\n        -----\n        A return value of 'same' means the transformation will work, but it will\n        just give back a copy of this object.  The intended usage is::\n\n            if coord.is_transformable_to(some_unknown_frame):\n                coord2 = coord.transform_to(some_unknown_frame)\n\n        This will work even if ``some_unknown_frame``  turns out to be the same\n        frame class as ``coord``.  This is intended for cases where the frame\n        is the same regardless of the frame attributes (e.g. ICRS), but be\n        aware that it *might* also indicate that someone forgot to define the\n        transformation between two objects of the same frame class but with\n        different attributes.\n        "
        new_frame = _get_frame_class(new_frame) if isinstance(new_frame, str) else new_frame
        return self.frame.is_transformable_to(new_frame)

    def transform_to(self, frame, merge_attributes=True):
        if False:
            i = 10
            return i + 15
        "Transform this coordinate to a new frame.\n\n        The precise frame transformed to depends on ``merge_attributes``.\n        If `False`, the destination frame is used exactly as passed in.\n        But this is often not quite what one wants.  E.g., suppose one wants to\n        transform an ICRS coordinate that has an obstime attribute to FK4; in\n        this case, one likely would want to use this information. Thus, the\n        default for ``merge_attributes`` is `True`, in which the precedence is\n        as follows: (1) explicitly set (i.e., non-default) values in the\n        destination frame; (2) explicitly set values in the source; (3) default\n        value in the destination frame.\n\n        Note that in either case, any explicitly set attributes on the source\n        |SkyCoord| that are not part of the destination frame's definition are\n        kept (stored on the resulting |SkyCoord|), and thus one can round-trip\n        (e.g., from FK4 to ICRS to FK4 without losing obstime).\n\n        Parameters\n        ----------\n        frame : str, `~astropy.coordinates.BaseCoordinateFrame` class or instance, or |SkyCoord| instance\n            The frame to transform this coordinate into.  If a |SkyCoord|, the\n            underlying frame is extracted, and all other information ignored.\n        merge_attributes : bool, optional\n            Whether the default attributes in the destination frame are allowed\n            to be overridden by explicitly set attributes in the source\n            (see note above; default: `True`).\n\n        Returns\n        -------\n        coord : |SkyCoord|\n            A new object with this coordinate represented in the `frame` frame.\n\n        Raises\n        ------\n        ValueError\n            If there is no possible transformation route.\n\n        "
        from astropy.coordinates.errors import ConvertError
        frame_kwargs = {}
        try:
            frame = _get_frame_class(frame)()
        except Exception:
            pass
        if isinstance(frame, SkyCoord):
            frame = frame.frame
        if isinstance(frame, BaseCoordinateFrame):
            new_frame_cls = frame.__class__
            for attr in frame_transform_graph.frame_attributes:
                self_val = getattr(self, attr, None)
                frame_val = getattr(frame, attr, None)
                if frame_val is not None and (not (merge_attributes and frame.is_frame_attr_default(attr))):
                    frame_kwargs[attr] = frame_val
                elif self_val is not None and (not self.is_frame_attr_default(attr)):
                    frame_kwargs[attr] = self_val
                elif frame_val is not None:
                    frame_kwargs[attr] = frame_val
        else:
            raise ValueError('Transform `frame` must be a frame name, class, or instance')
        trans = frame_transform_graph.get_transform(self.frame.__class__, new_frame_cls)
        if trans is None:
            raise ConvertError(f'Cannot transform from {self.frame.__class__} to {new_frame_cls}')
        generic_frame = GenericFrame(frame_kwargs)
        new_coord = trans(self.frame, generic_frame)
        for attr in set(new_coord.frame_attributes) & set(frame_kwargs.keys()):
            frame_kwargs.pop(attr)
        frame_kwargs.pop('origin', None)
        return self.__class__(new_coord, **frame_kwargs)

    def apply_space_motion(self, new_obstime=None, dt=None):
        if False:
            i = 10
            return i + 15
        'Compute the position to a new time using the velocities.\n\n        Compute the position of the source represented by this coordinate object\n        to a new time using the velocities stored in this object and assuming\n        linear space motion (including relativistic corrections). This is\n        sometimes referred to as an "epoch transformation".\n\n        The initial time before the evolution is taken from the ``obstime``\n        attribute of this coordinate.  Note that this method currently does not\n        support evolving coordinates where the *frame* has an ``obstime`` frame\n        attribute, so the ``obstime`` is only used for storing the before and\n        after times, not actually as an attribute of the frame. Alternatively,\n        if ``dt`` is given, an ``obstime`` need not be provided at all.\n\n        Parameters\n        ----------\n        new_obstime : `~astropy.time.Time`, optional\n            The time at which to evolve the position to. Requires that the\n            ``obstime`` attribute be present on this frame.\n        dt : `~astropy.units.Quantity`, `~astropy.time.TimeDelta`, optional\n            An amount of time to evolve the position of the source. Cannot be\n            given at the same time as ``new_obstime``.\n\n        Returns\n        -------\n        new_coord : |SkyCoord|\n            A new coordinate object with the evolved location of this coordinate\n            at the new time.  ``obstime`` will be set on this object to the new\n            time only if ``self`` also has ``obstime``.\n        '
        from .builtin_frames.icrs import ICRS
        if (new_obstime is None) == (dt is None):
            raise ValueError('You must specify one of `new_obstime` or `dt`, but not both.')
        if 's' not in self.frame.data.differentials:
            raise ValueError('SkyCoord requires velocity data to evolve the position.')
        if 'obstime' in self.frame.frame_attributes:
            raise NotImplementedError('Updating the coordinates in a frame with explicit time dependence is currently not supported. If you would like this functionality, please open an issue on github:\nhttps://github.com/astropy/astropy')
        if new_obstime is not None and self.obstime is None:
            raise ValueError('This object has no associated `obstime`. apply_space_motion() must receive a time difference, `dt`, and not a new obstime.')
        t1 = self.obstime
        if dt is None:
            t2 = new_obstime
        elif t1 is None:
            t1 = Time('J2000')
            new_obstime = None
            t2 = t1 + dt
        else:
            t2 = t1 + dt
            new_obstime = t2
        t1 = t1.tdb
        t2 = t2.tdb
        icrsrep = self.icrs.represent_as(SphericalRepresentation, SphericalDifferential)
        icrsvel = icrsrep.differentials['s']
        parallax_zero = False
        try:
            plx = icrsrep.distance.to_value(u.arcsecond, u.parallax())
        except u.UnitConversionError:
            plx = 0.0
            parallax_zero = True
        try:
            rv = icrsvel.d_distance.to_value(u.km / u.s)
        except u.UnitConversionError:
            rv = 0.0
        starpm = erfa.pmsafe(icrsrep.lon.radian, icrsrep.lat.radian, icrsvel.d_lon.to_value(u.radian / u.yr), icrsvel.d_lat.to_value(u.radian / u.yr), plx, rv, t1.jd1, t1.jd2, t2.jd1, t2.jd2)
        if parallax_zero:
            new_distance = None
        else:
            new_distance = Distance(parallax=starpm[4] << u.arcsec)
        icrs2 = ICRS(ra=u.Quantity(starpm[0], u.radian, copy=False), dec=u.Quantity(starpm[1], u.radian, copy=False), pm_ra=u.Quantity(starpm[2], u.radian / u.yr, copy=False), pm_dec=u.Quantity(starpm[3], u.radian / u.yr, copy=False), distance=new_distance, radial_velocity=u.Quantity(starpm[5], u.km / u.s, copy=False), differential_type=SphericalDifferential)
        frattrs = {attrnm: getattr(self, attrnm) for attrnm in self._extra_frameattr_names}
        frattrs['obstime'] = new_obstime
        result = self.__class__(icrs2, **frattrs).transform_to(self.frame)
        result.differential_type = self.differential_type
        return result

    def _is_name(self, string):
        if False:
            while True:
                i = 10
        '\n        Returns whether a string is one of the aliases for the frame.\n        '
        return self.frame.name == string or (isinstance(self.frame.name, list) and string in self.frame.name)

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        '\n        Overrides getattr to return coordinates that this can be transformed\n        to, based on the alias attr in the primary transform graph.\n        '
        if '_sky_coord_frame' in self.__dict__:
            if self._is_name(attr):
                return self
            if attr in frame_transform_graph.frame_attributes:
                if attr in self.frame.frame_attributes:
                    return getattr(self.frame, attr)
                else:
                    return getattr(self, '_' + attr, None)
            if not attr.startswith('_') and hasattr(self._sky_coord_frame, attr):
                return getattr(self._sky_coord_frame, attr)
            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                return self.transform_to(attr)
        return self.__getattribute__(attr)

    def __setattr__(self, attr, val):
        if False:
            i = 10
            return i + 15
        if '_sky_coord_frame' in self.__dict__:
            if self._is_name(attr):
                raise AttributeError(f"'{attr}' is immutable")
            if not attr.startswith('_') and hasattr(self._sky_coord_frame, attr):
                setattr(self._sky_coord_frame, attr, val)
                return
            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                raise AttributeError(f"'{attr}' is immutable")
        if attr in frame_transform_graph.frame_attributes:
            super().__setattr__('_' + attr, val)
            frame_transform_graph.frame_attributes[attr].__get__(self)
            self._extra_frameattr_names |= {attr}
        else:
            super().__setattr__(attr, val)

    def __delattr__(self, attr):
        if False:
            while True:
                i = 10
        if '_sky_coord_frame' in self.__dict__:
            if self._is_name(attr):
                raise AttributeError(f"'{attr}' is immutable")
            if not attr.startswith('_') and hasattr(self._sky_coord_frame, attr):
                delattr(self._sky_coord_frame, attr)
                return
            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self.frame.is_transformable_to(frame_cls):
                raise AttributeError(f"'{attr}' is immutable")
        if attr in frame_transform_graph.frame_attributes:
            super().__delattr__('_' + attr)
            self._extra_frameattr_names -= {attr}
        else:
            super().__delattr__(attr)

    def __dir__(self):
        if False:
            return 10
        'Original dir() behavior, plus frame attributes and transforms.\n\n        This dir includes:\n        - All attributes of the SkyCoord class\n        - Coordinate transforms available by aliases\n        - Attribute / methods of the underlying self.frame objects\n        '
        dir_values = set(super().__dir__())
        for name in frame_transform_graph.get_names():
            frame_cls = frame_transform_graph.lookup_name(name)
            if self.frame.is_transformable_to(frame_cls):
                dir_values.add(name)
        dir_values.update({attr for attr in dir(self.frame) if not attr.startswith('_')})
        dir_values.update(frame_transform_graph.frame_attributes.keys())
        return sorted(dir_values)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        clsnm = self.__class__.__name__
        coonm = self.frame.__class__.__name__
        frameattrs = self.frame._frame_attrs_repr()
        if frameattrs:
            frameattrs = ': ' + frameattrs
        data = self.frame._data_repr()
        if data:
            data = ': ' + data
        return f'<{clsnm} ({coonm}{frameattrs}){data}>'

    def to_string(self, style='decimal', **kwargs):
        if False:
            print('Hello World!')
        '\n        A string representation of the coordinates.\n\n        The default styles definitions are::\n\n          \'decimal\': \'lat\': {\'decimal\': True, \'unit\': "deg"}\n                     \'lon\': {\'decimal\': True, \'unit\': "deg"}\n          \'dms\': \'lat\': {\'unit\': "deg"}\n                 \'lon\': {\'unit\': "deg"}\n          \'hmsdms\': \'lat\': {\'alwayssign\': True, \'pad\': True, \'unit\': "deg"}\n                    \'lon\': {\'pad\': True, \'unit\': "hour"}\n\n        See :meth:`~astropy.coordinates.Angle.to_string` for details and\n        keyword arguments (the two angles forming the coordinates are are\n        both :class:`~astropy.coordinates.Angle` instances). Keyword\n        arguments have precedence over the style defaults and are passed\n        to :meth:`~astropy.coordinates.Angle.to_string`.\n\n        Parameters\n        ----------\n        style : {\'hmsdms\', \'dms\', \'decimal\'}\n            The formatting specification to use. These encode the three most\n            common ways to represent coordinates. The default is `decimal`.\n        **kwargs\n            Keyword args passed to :meth:`~astropy.coordinates.Angle.to_string`.\n        '
        sph_coord = self.frame.represent_as(SphericalRepresentation)
        styles = {'hmsdms': {'lonargs': {'unit': u.hour, 'pad': True}, 'latargs': {'unit': u.degree, 'pad': True, 'alwayssign': True}}, 'dms': {'lonargs': {'unit': u.degree}, 'latargs': {'unit': u.degree}}, 'decimal': {'lonargs': {'unit': u.degree, 'decimal': True}, 'latargs': {'unit': u.degree, 'decimal': True}}}
        lonargs = {}
        latargs = {}
        if style in styles:
            lonargs.update(styles[style]['lonargs'])
            latargs.update(styles[style]['latargs'])
        else:
            raise ValueError(f"Invalid style.  Valid options are: {','.join(styles)}")
        lonargs.update(kwargs)
        latargs.update(kwargs)
        if np.isscalar(sph_coord.lon.value):
            coord_string = f'{sph_coord.lon.to_string(**lonargs)} {sph_coord.lat.to_string(**latargs)}'
        else:
            coord_string = []
            for (lonangle, latangle) in zip(sph_coord.lon.ravel(), sph_coord.lat.ravel()):
                coord_string += [f'{lonangle.to_string(**lonargs)} {latangle.to_string(**latargs)}']
            if len(sph_coord.shape) > 1:
                coord_string = np.array(coord_string).reshape(sph_coord.shape)
        return coord_string

    def to_table(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Convert this |SkyCoord| to a |QTable|.\n\n        Any attributes that have the same length as the |SkyCoord| will be\n        converted to columns of the |QTable|. All other attributes will be\n        recorded as metadata.\n\n        Returns\n        -------\n        `~astropy.table.QTable`\n            A |QTable| containing the data of this |SkyCoord|.\n\n        Examples\n        --------\n        >>> sc = SkyCoord(ra=[40, 70]*u.deg, dec=[0, -20]*u.deg,\n        ...               obstime=Time([2000, 2010], format='jyear'))\n        >>> t =  sc.to_table()\n        >>> t\n        <QTable length=2>\n           ra     dec   obstime\n          deg     deg\n        float64 float64   Time\n        ------- ------- -------\n           40.0     0.0  2000.0\n           70.0   -20.0  2010.0\n        >>> t.meta\n        {'representation_type': 'spherical', 'frame': 'icrs'}\n        "
        self_as_dict = self.info._represent_as_dict()
        tabledata = {}
        metadata = {}
        for (key, value) in self_as_dict.items():
            if getattr(value, 'shape', ())[:1] == (len(self),):
                tabledata[key] = value
            else:
                metadata[key] = value
        return QTable(tabledata, meta=metadata)

    def is_equivalent_frame(self, other):
        if False:
            print('Hello World!')
        "\n        Checks if this object's frame is the same as that of the ``other``\n        object.\n\n        To be the same frame, two objects must be the same frame class and have\n        the same frame attributes. For two |SkyCoord| objects, *all* of the\n        frame attributes have to match, not just those relevant for the object's\n        frame.\n\n        Parameters\n        ----------\n        other : SkyCoord or BaseCoordinateFrame\n            The other object to check.\n\n        Returns\n        -------\n        isequiv : bool\n            True if the frames are the same, False if not.\n\n        Raises\n        ------\n        TypeError\n            If ``other`` isn't a |SkyCoord| or a subclass of\n            `~astropy.coordinates.BaseCoordinateFrame`.\n        "
        if isinstance(other, BaseCoordinateFrame):
            return self.frame.is_equivalent_frame(other)
        elif isinstance(other, SkyCoord):
            if other.frame.name != self.frame.name:
                return False
            for fattrnm in frame_transform_graph.frame_attributes:
                if not BaseCoordinateFrame._frameattr_equiv(getattr(self, fattrnm), getattr(other, fattrnm)):
                    return False
            return True
        else:
            raise TypeError("Tried to do is_equivalent_frame on something that isn't frame-like")

    def separation(self, other):
        if False:
            return 10
        '\n        Computes on-sky separation between this coordinate and another.\n\n        .. note::\n\n            If the ``other`` coordinate object is in a different frame, it is\n            first transformed to the frame of this object. This can lead to\n            unintuitive behavior if not accounted for. Particularly of note is\n            that ``self.separation(other)`` and ``other.separation(self)`` may\n            not give the same answer in this case.\n\n        For more on how to use this (and related) functionality, see the\n        examples in :doc:`astropy:/coordinates/matchsep`.\n\n        Parameters\n        ----------\n        other : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`\n            The coordinate to get the separation to.\n\n        Returns\n        -------\n        sep : `~astropy.coordinates.Angle`\n            The on-sky separation between this and the ``other`` coordinate.\n\n        Notes\n        -----\n        The separation is calculated using the Vincenty formula, which\n        is stable at all locations, including poles and antipodes [1]_.\n\n        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance\n\n        '
        from .angles import Angle, angular_separation
        if not self.is_equivalent_frame(other):
            try:
                kwargs = {'merge_attributes': False} if isinstance(other, SkyCoord) else {}
                other = other.transform_to(self, **kwargs)
            except TypeError:
                raise TypeError('Can only get separation to another SkyCoord or a coordinate frame with data')
        lon1 = self.spherical.lon
        lat1 = self.spherical.lat
        lon2 = other.spherical.lon
        lat2 = other.spherical.lat
        sep = angular_separation(lon1, lat1, lon2, lat2)
        return Angle(sep, unit=u.degree)

    def separation_3d(self, other):
        if False:
            print('Hello World!')
        '\n        Computes three dimensional separation between this coordinate\n        and another.\n\n        For more on how to use this (and related) functionality, see the\n        examples in :doc:`astropy:/coordinates/matchsep`.\n\n        Parameters\n        ----------\n        other : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`\n            The coordinate to get the separation to.\n\n        Returns\n        -------\n        sep : `~astropy.coordinates.Distance`\n            The real-space distance between these two coordinates.\n\n        Raises\n        ------\n        ValueError\n            If this or the other coordinate do not have distances.\n        '
        if not self.is_equivalent_frame(other):
            try:
                kwargs = {'merge_attributes': False} if isinstance(other, SkyCoord) else {}
                other = other.transform_to(self, **kwargs)
            except TypeError:
                raise TypeError('Can only get separation to another SkyCoord or a coordinate frame with data')
        if issubclass(self.data.__class__, UnitSphericalRepresentation):
            raise ValueError('This object does not have a distance; cannot compute 3d separation.')
        if issubclass(other.data.__class__, UnitSphericalRepresentation):
            raise ValueError('The other object does not have a distance; cannot compute 3d separation.')
        c1 = self.cartesian.without_differentials()
        c2 = other.cartesian.without_differentials()
        return Distance((c1 - c2).norm())

    def spherical_offsets_to(self, tocoord):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes angular offsets to go *from* this coordinate *to* another.\n\n        Parameters\n        ----------\n        tocoord : `~astropy.coordinates.BaseCoordinateFrame`\n            The coordinate to find the offset to.\n\n        Returns\n        -------\n        lon_offset : `~astropy.coordinates.Angle`\n            The angular offset in the longitude direction. The definition of\n            "longitude" depends on this coordinate\'s frame (e.g., RA for\n            equatorial coordinates).\n        lat_offset : `~astropy.coordinates.Angle`\n            The angular offset in the latitude direction. The definition of\n            "latitude" depends on this coordinate\'s frame (e.g., Dec for\n            equatorial coordinates).\n\n        Raises\n        ------\n        ValueError\n            If the ``tocoord`` is not in the same frame as this one. This is\n            different from the behavior of the `separation`/`separation_3d`\n            methods because the offset components depend critically on the\n            specific choice of frame.\n\n        Notes\n        -----\n        This uses the sky offset frame machinery, and hence will produce a new\n        sky offset frame if one does not already exist for this object\'s frame\n        class.\n\n        See Also\n        --------\n        separation :\n            for the *total* angular offset (not broken out into components).\n        position_angle :\n            for the direction of the offset.\n\n        '
        if not self.is_equivalent_frame(tocoord):
            raise ValueError('Tried to use spherical_offsets_to with two non-matching frames!')
        aframe = self.skyoffset_frame()
        acoord = tocoord.transform_to(aframe)
        dlon = acoord.spherical.lon.view(Angle)
        dlat = acoord.spherical.lat.view(Angle)
        return (dlon, dlat)

    def spherical_offsets_by(self, d_lon, d_lat):
        if False:
            i = 10
            return i + 15
        '\n        Computes the coordinate that is a specified pair of angular offsets away\n        from this coordinate.\n\n        Parameters\n        ----------\n        d_lon : angle-like\n            The angular offset in the longitude direction. The definition of\n            "longitude" depends on this coordinate\'s frame (e.g., RA for\n            equatorial coordinates).\n        d_lat : angle-like\n            The angular offset in the latitude direction. The definition of\n            "latitude" depends on this coordinate\'s frame (e.g., Dec for\n            equatorial coordinates).\n\n        Returns\n        -------\n        newcoord : `~astropy.coordinates.SkyCoord`\n            The coordinates for the location that corresponds to offsetting by\n            ``d_lat`` in the latitude direction and ``d_lon`` in the longitude\n            direction.\n\n        Notes\n        -----\n        This internally uses `~astropy.coordinates.SkyOffsetFrame` to do the\n        transformation. For a more complete set of transform offsets, use\n        `~astropy.coordinates.SkyOffsetFrame` or `~astropy.wcs.WCS` manually.\n        This specific method can be reproduced by doing\n        ``SkyCoord(SkyOffsetFrame(d_lon, d_lat, origin=self.frame).transform_to(self))``.\n\n        See Also\n        --------\n        spherical_offsets_to : compute the angular offsets to another coordinate\n        directional_offset_by : offset a coordinate by an angle in a direction\n        '
        from .builtin_frames.skyoffset import SkyOffsetFrame
        return self.__class__(SkyOffsetFrame(d_lon, d_lat, origin=self.frame).transform_to(self))

    def directional_offset_by(self, position_angle, separation):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes coordinates at the given offset from this coordinate.\n\n        Parameters\n        ----------\n        position_angle : `~astropy.coordinates.Angle`\n            position_angle of offset\n        separation : `~astropy.coordinates.Angle`\n            offset angular separation\n\n        Returns\n        -------\n        newpoints : `~astropy.coordinates.SkyCoord`\n            The coordinates for the location that corresponds to offsetting by\n            the given `position_angle` and `separation`.\n\n        Notes\n        -----\n        Returned SkyCoord frame retains only the frame attributes that are for\n        the resulting frame type.  (e.g. if the input frame is\n        `~astropy.coordinates.ICRS`, an ``equinox`` value will be retained, but\n        an ``obstime`` will not.)\n\n        For a more complete set of transform offsets, use `~astropy.wcs.WCS`.\n        `~astropy.coordinates.SkyCoord.skyoffset_frame()` can also be used to\n        create a spherical frame with (lat=0, lon=0) at a reference point,\n        approximating an xy cartesian system for small offsets. This method\n        is distinct in that it is accurate on the sphere.\n\n        See Also\n        --------\n        position_angle : inverse operation for the ``position_angle`` component\n        separation : inverse operation for the ``separation`` component\n\n        '
        from .angles import offset_by
        slat = self.represent_as(UnitSphericalRepresentation).lat
        slon = self.represent_as(UnitSphericalRepresentation).lon
        (newlon, newlat) = offset_by(lon=slon, lat=slat, posang=position_angle, distance=separation)
        return SkyCoord(newlon, newlat, frame=self.frame)

    def match_to_catalog_sky(self, catalogcoord, nthneighbor=1):
        if False:
            print('Hello World!')
        "\n        Finds the nearest on-sky matches of this coordinate in a set of\n        catalog coordinates.\n\n        For more on how to use this (and related) functionality, see the\n        examples in :doc:`astropy:/coordinates/matchsep`.\n\n        Parameters\n        ----------\n        catalogcoord : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`\n            The base catalog in which to search for matches. Typically this\n            will be a coordinate object that is an array (i.e.,\n            ``catalogcoord.isscalar == False``)\n        nthneighbor : int, optional\n            Which closest neighbor to search for.  Typically ``1`` is\n            desired here, as that is correct for matching one set of\n            coordinates to another. The next likely use case is ``2``,\n            for matching a coordinate catalog against *itself* (``1``\n            is inappropriate because each point will find itself as the\n            closest match).\n\n        Returns\n        -------\n        idx : int array\n            Indices into ``catalogcoord`` to get the matched points for\n            each of this object's coordinates. Shape matches this\n            object.\n        sep2d : `~astropy.coordinates.Angle`\n            The on-sky separation between the closest match for each\n            element in this object in ``catalogcoord``. Shape matches\n            this object.\n        dist3d : `~astropy.units.Quantity` ['length']\n            The 3D distance between the closest match for each element\n            in this object in ``catalogcoord``. Shape matches this\n            object. Unless both this and ``catalogcoord`` have associated\n            distances, this quantity assumes that all sources are at a\n            distance of 1 (dimensionless).\n\n        Notes\n        -----\n        This method requires `SciPy <https://www.scipy.org/>`_ to be\n        installed or it will fail.\n\n        See Also\n        --------\n        astropy.coordinates.match_coordinates_sky\n        SkyCoord.match_to_catalog_3d\n        "
        from .matching import match_coordinates_sky
        if not (isinstance(catalogcoord, (SkyCoord, BaseCoordinateFrame)) and catalogcoord.has_data):
            raise TypeError('Can only get separation to another SkyCoord or a coordinate frame with data')
        res = match_coordinates_sky(self, catalogcoord, nthneighbor=nthneighbor, storekdtree='_kdtree_sky')
        return res

    def match_to_catalog_3d(self, catalogcoord, nthneighbor=1):
        if False:
            print('Hello World!')
        "\n        Finds the nearest 3-dimensional matches of this coordinate to a set\n        of catalog coordinates.\n\n        This finds the 3-dimensional closest neighbor, which is only different\n        from the on-sky distance if ``distance`` is set in this object or the\n        ``catalogcoord`` object.\n\n        For more on how to use this (and related) functionality, see the\n        examples in :doc:`astropy:/coordinates/matchsep`.\n\n        Parameters\n        ----------\n        catalogcoord : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`\n            The base catalog in which to search for matches. Typically this\n            will be a coordinate object that is an array (i.e.,\n            ``catalogcoord.isscalar == False``)\n        nthneighbor : int, optional\n            Which closest neighbor to search for.  Typically ``1`` is\n            desired here, as that is correct for matching one set of\n            coordinates to another.  The next likely use case is\n            ``2``, for matching a coordinate catalog against *itself*\n            (``1`` is inappropriate because each point will find\n            itself as the closest match).\n\n        Returns\n        -------\n        idx : int array\n            Indices into ``catalogcoord`` to get the matched points for\n            each of this object's coordinates. Shape matches this\n            object.\n        sep2d : `~astropy.coordinates.Angle`\n            The on-sky separation between the closest match for each\n            element in this object in ``catalogcoord``. Shape matches\n            this object.\n        dist3d : `~astropy.units.Quantity` ['length']\n            The 3D distance between the closest match for each element\n            in this object in ``catalogcoord``. Shape matches this\n            object.\n\n        Notes\n        -----\n        This method requires `SciPy <https://www.scipy.org/>`_ to be\n        installed or it will fail.\n\n        See Also\n        --------\n        astropy.coordinates.match_coordinates_3d\n        SkyCoord.match_to_catalog_sky\n        "
        from .matching import match_coordinates_3d
        if not (isinstance(catalogcoord, (SkyCoord, BaseCoordinateFrame)) and catalogcoord.has_data):
            raise TypeError('Can only get separation to another SkyCoord or a coordinate frame with data')
        res = match_coordinates_3d(self, catalogcoord, nthneighbor=nthneighbor, storekdtree='_kdtree_3d')
        return res

    def search_around_sky(self, searcharoundcoords, seplimit):
        if False:
            return 10
        "\n        Searches for all coordinates in this object around a supplied set of\n        points within a given on-sky separation.\n\n        This is intended for use on `~astropy.coordinates.SkyCoord` objects\n        with coordinate arrays, rather than a scalar coordinate.  For a scalar\n        coordinate, it is better to use\n        `~astropy.coordinates.SkyCoord.separation`.\n\n        For more on how to use this (and related) functionality, see the\n        examples in :doc:`astropy:/coordinates/matchsep`.\n\n        Parameters\n        ----------\n        searcharoundcoords : coordinate-like\n            The coordinates to search around to try to find matching points in\n            this |SkyCoord|. This should be an object with array coordinates,\n            not a scalar coordinate object.\n        seplimit : `~astropy.units.Quantity` ['angle']\n            The on-sky separation to search within.\n\n        Returns\n        -------\n        idxsearcharound : int array\n            Indices into ``searcharoundcoords`` that match the\n            corresponding elements of ``idxself``. Shape matches\n            ``idxself``.\n        idxself : int array\n            Indices into ``self`` that match the\n            corresponding elements of ``idxsearcharound``. Shape matches\n            ``idxsearcharound``.\n        sep2d : `~astropy.coordinates.Angle`\n            The on-sky separation between the coordinates. Shape matches\n            ``idxsearcharound`` and ``idxself``.\n        dist3d : `~astropy.units.Quantity` ['length']\n            The 3D distance between the coordinates. Shape matches\n            ``idxsearcharound`` and ``idxself``.\n\n        Notes\n        -----\n        This method requires `SciPy <https://www.scipy.org/>`_ to be\n        installed or it will fail.\n\n        In the current implementation, the return values are always sorted in\n        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is\n        in ascending order).  This is considered an implementation detail,\n        though, so it could change in a future release.\n\n        See Also\n        --------\n        astropy.coordinates.search_around_sky\n        SkyCoord.search_around_3d\n        "
        from .matching import search_around_sky
        return search_around_sky(searcharoundcoords, self, seplimit, storekdtree='_kdtree_sky')

    def search_around_3d(self, searcharoundcoords, distlimit):
        if False:
            return 10
        "\n        Searches for all coordinates in this object around a supplied set of\n        points within a given 3D radius.\n\n        This is intended for use on `~astropy.coordinates.SkyCoord` objects\n        with coordinate arrays, rather than a scalar coordinate.  For a scalar\n        coordinate, it is better to use\n        `~astropy.coordinates.SkyCoord.separation_3d`.\n\n        For more on how to use this (and related) functionality, see the\n        examples in :doc:`astropy:/coordinates/matchsep`.\n\n        Parameters\n        ----------\n        searcharoundcoords : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`\n            The coordinates to search around to try to find matching points in\n            this |SkyCoord|. This should be an object with array coordinates,\n            not a scalar coordinate object.\n        distlimit : `~astropy.units.Quantity` ['length']\n            The physical radius to search within.\n\n        Returns\n        -------\n        idxsearcharound : int array\n            Indices into ``searcharoundcoords`` that match the\n            corresponding elements of ``idxself``. Shape matches\n            ``idxself``.\n        idxself : int array\n            Indices into ``self`` that match the\n            corresponding elements of ``idxsearcharound``. Shape matches\n            ``idxsearcharound``.\n        sep2d : `~astropy.coordinates.Angle`\n            The on-sky separation between the coordinates. Shape matches\n            ``idxsearcharound`` and ``idxself``.\n        dist3d : `~astropy.units.Quantity` ['length']\n            The 3D distance between the coordinates. Shape matches\n            ``idxsearcharound`` and ``idxself``.\n\n        Notes\n        -----\n        This method requires `SciPy <https://www.scipy.org/>`_ to be\n        installed or it will fail.\n\n        In the current implementation, the return values are always sorted in\n        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is\n        in ascending order).  This is considered an implementation detail,\n        though, so it could change in a future release.\n\n        See Also\n        --------\n        astropy.coordinates.search_around_3d\n        SkyCoord.search_around_sky\n        "
        from .matching import search_around_3d
        return search_around_3d(searcharoundcoords, self, distlimit, storekdtree='_kdtree_3d')

    def position_angle(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Computes the on-sky position angle (East of North) between this\n        SkyCoord and another.\n\n        Parameters\n        ----------\n        other : |SkyCoord|\n            The other coordinate to compute the position angle to.  It is\n            treated as the "head" of the vector of the position angle.\n\n        Returns\n        -------\n        pa : `~astropy.coordinates.Angle`\n            The (positive) position angle of the vector pointing from ``self``\n            to ``other``.  If either ``self`` or ``other`` contain arrays, this\n            will be an array following the appropriate `numpy` broadcasting\n            rules.\n\n        Examples\n        --------\n        >>> c1 = SkyCoord(0*u.deg, 0*u.deg)\n        >>> c2 = SkyCoord(1*u.deg, 0*u.deg)\n        >>> c1.position_angle(c2).degree\n        90.0\n        >>> c3 = SkyCoord(1*u.deg, 1*u.deg)\n        >>> c1.position_angle(c3).degree  # doctest: +FLOAT_CMP\n        44.995636455344844\n        '
        from .angles import position_angle
        if not self.is_equivalent_frame(other):
            try:
                other = other.transform_to(self, merge_attributes=False)
            except TypeError:
                raise TypeError('Can only get position_angle to another SkyCoord or a coordinate frame with data')
        slat = self.represent_as(UnitSphericalRepresentation).lat
        slon = self.represent_as(UnitSphericalRepresentation).lon
        olat = other.represent_as(UnitSphericalRepresentation).lat
        olon = other.represent_as(UnitSphericalRepresentation).lon
        return position_angle(slon, slat, olon, olat)

    def skyoffset_frame(self, rotation=None):
        if False:
            while True:
                i = 10
        '\n        Returns the sky offset frame with this SkyCoord at the origin.\n\n        Parameters\n        ----------\n        rotation : angle-like\n            The final rotation of the frame about the ``origin``. The sign of\n            the rotation is the left-hand rule. That is, an object at a\n            particular position angle in the un-rotated system will be sent to\n            the positive latitude (z) direction in the final frame.\n\n        Returns\n        -------\n        astrframe : `~astropy.coordinates.SkyOffsetFrame`\n            A sky offset frame of the same type as this |SkyCoord| (e.g., if\n            this object has an ICRS coordinate, the resulting frame is\n            SkyOffsetICRS, with the origin set to this object)\n        '
        from .builtin_frames.skyoffset import SkyOffsetFrame
        return SkyOffsetFrame(origin=self, rotation=rotation)

    def get_constellation(self, short_name=False, constellation_list='iau'):
        if False:
            while True:
                i = 10
        '\n        Determines the constellation(s) of the coordinates this SkyCoord contains.\n\n        Parameters\n        ----------\n        short_name : bool\n            If True, the returned names are the IAU-sanctioned abbreviated\n            names.  Otherwise, full names for the constellations are used.\n        constellation_list : str\n            The set of constellations to use.  Currently only ``\'iau\'`` is\n            supported, meaning the 88 "modern" constellations endorsed by the IAU.\n\n        Returns\n        -------\n        constellation : str or string array\n            If this is a scalar coordinate, returns the name of the\n            constellation.  If it is an array |SkyCoord|, it returns an array of\n            names.\n\n        Notes\n        -----\n        To determine which constellation a point on the sky is in, this first\n        precesses to B1875, and then uses the Delporte boundaries of the 88\n        modern constellations, as tabulated by\n        `Roman 1987 <https://cdsarc.cds.unistra.fr/viz-bin/Cat?VI/42>`_.\n\n        See Also\n        --------\n        astropy.coordinates.get_constellation\n        '
        from .funcs import get_constellation
        extra_frameattrs = {nm: getattr(self, nm) for nm in self._extra_frameattr_names}
        novel = SkyCoord(self.realize_frame(self.data.without_differentials()), **extra_frameattrs)
        return get_constellation(novel, short_name, constellation_list)

    def to_pixel(self, wcs, origin=0, mode='all'):
        if False:
            print('Hello World!')
        "\n        Convert this coordinate to pixel coordinates using a `~astropy.wcs.WCS`\n        object.\n\n        Parameters\n        ----------\n        wcs : `~astropy.wcs.WCS`\n            The WCS to use for convert\n        origin : int\n            Whether to return 0 or 1-based pixel coordinates.\n        mode : 'all' or 'wcs'\n            Whether to do the transformation including distortions (``'all'``) or\n            only including only the core WCS transformation (``'wcs'``).\n\n        Returns\n        -------\n        xp, yp : `numpy.ndarray`\n            The pixel coordinates\n\n        See Also\n        --------\n        astropy.wcs.utils.skycoord_to_pixel : the implementation of this method\n        "
        from astropy.wcs.utils import skycoord_to_pixel
        return skycoord_to_pixel(self, wcs=wcs, origin=origin, mode=mode)

    @classmethod
    def from_pixel(cls, xp, yp, wcs, origin=0, mode='all'):
        if False:
            i = 10
            return i + 15
        "\n        Create a new SkyCoord from pixel coordinates using a World Coordinate System.\n\n        Parameters\n        ----------\n        xp, yp : float or ndarray\n            The coordinates to convert.\n        wcs : `~astropy.wcs.WCS`\n            The WCS to use for convert\n        origin : int\n            Whether to return 0 or 1-based pixel coordinates.\n        mode : 'all' or 'wcs'\n            Whether to do the transformation including distortions (``'all'``) or\n            only including only the core WCS transformation (``'wcs'``).\n\n        Returns\n        -------\n        coord : `~astropy.coordinates.SkyCoord`\n            A new object with sky coordinates corresponding to the input ``xp``\n            and ``yp``.\n\n        See Also\n        --------\n        to_pixel : to do the inverse operation\n        astropy.wcs.utils.pixel_to_skycoord : the implementation of this method\n        "
        from astropy.wcs.utils import pixel_to_skycoord
        return pixel_to_skycoord(xp, yp, wcs=wcs, origin=origin, mode=mode, cls=cls)

    def contained_by(self, wcs, image=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Determines if the SkyCoord is contained in the given wcs footprint.\n\n        Parameters\n        ----------\n        wcs : `~astropy.wcs.WCS`\n            The coordinate to check if it is within the wcs coordinate.\n        image : array\n            Optional.  The image associated with the wcs object that the coordinate\n            is being checked against. If not given the naxis keywords will be used\n            to determine if the coordinate falls within the wcs footprint.\n        **kwargs\n            Additional arguments to pass to `~astropy.coordinates.SkyCoord.to_pixel`\n\n        Returns\n        -------\n        response : bool\n            True means the WCS footprint contains the coordinate, False means it does not.\n        '
        if image is not None:
            (ymax, xmax) = image.shape
        else:
            (xmax, ymax) = wcs._naxis
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                (x, y) = self.to_pixel(wcs, **kwargs)
            except Exception:
                return False
        return (x < xmax) & (x > 0) & (y < ymax) & (y > 0)

    def radial_velocity_correction(self, kind='barycentric', obstime=None, location=None):
        if False:
            while True:
                i = 10
        "\n        Compute the correction required to convert a radial velocity at a given\n        time and place on the Earth's Surface to a barycentric or heliocentric\n        velocity.\n\n        Parameters\n        ----------\n        kind : str\n            The kind of velocity correction.  Must be 'barycentric' or\n            'heliocentric'.\n        obstime : `~astropy.time.Time` or None, optional\n            The time at which to compute the correction.  If `None`, the\n            ``obstime`` frame attribute on the |SkyCoord| will be used.\n        location : `~astropy.coordinates.EarthLocation` or None, optional\n            The observer location at which to compute the correction.  If\n            `None`, the  ``location`` frame attribute on the passed-in\n            ``obstime`` will be used, and if that is None, the ``location``\n            frame attribute on the |SkyCoord| will be used.\n\n        Raises\n        ------\n        ValueError\n            If either ``obstime`` or ``location`` are passed in (not ``None``)\n            when the frame attribute is already set on this |SkyCoord|.\n        TypeError\n            If ``obstime`` or ``location`` aren't provided, either as arguments\n            or as frame attributes.\n\n        Returns\n        -------\n        vcorr : `~astropy.units.Quantity` ['speed']\n            The  correction with a positive sign.  I.e., *add* this\n            to an observed radial velocity to get the barycentric (or\n            heliocentric) velocity. If m/s precision or better is needed,\n            see the notes below.\n\n        Notes\n        -----\n        The barycentric correction is calculated to higher precision than the\n        heliocentric correction and includes additional physics (e.g time dilation).\n        Use barycentric corrections if m/s precision is required.\n\n        The algorithm here is sufficient to perform corrections at the mm/s level, but\n        care is needed in application. The barycentric correction returned uses the optical\n        approximation v = z * c. Strictly speaking, the barycentric correction is\n        multiplicative and should be applied as::\n\n          >>> from astropy.time import Time\n          >>> from astropy.coordinates import SkyCoord, EarthLocation\n          >>> from astropy.constants import c\n          >>> t = Time(56370.5, format='mjd', scale='utc')\n          >>> loc = EarthLocation('149d33m00.5s','-30d18m46.385s',236.87*u.m)\n          >>> sc = SkyCoord(1*u.deg, 2*u.deg)\n          >>> vcorr = sc.radial_velocity_correction(kind='barycentric', obstime=t, location=loc)  # doctest: +REMOTE_DATA\n          >>> rv = rv + vcorr + rv * vcorr / c  # doctest: +SKIP\n\n        Also note that this method returns the correction velocity in the so-called\n        *optical convention*::\n\n          >>> vcorr = zb * c  # doctest: +SKIP\n\n        where ``zb`` is the barycentric correction redshift as defined in section 3\n        of Wright & Eastman (2014). The application formula given above follows from their\n        equation (11) under assumption that the radial velocity ``rv`` has also been defined\n        using the same optical convention. Note, this can be regarded as a matter of\n        velocity definition and does not by itself imply any loss of accuracy, provided\n        sufficient care has been taken during interpretation of the results. If you need\n        the barycentric correction expressed as the full relativistic velocity (e.g., to provide\n        it as the input to another software which performs the application), the\n        following recipe can be used::\n\n          >>> zb = vcorr / c  # doctest: +REMOTE_DATA\n          >>> zb_plus_one_squared = (zb + 1) ** 2  # doctest: +REMOTE_DATA\n          >>> vcorr_rel = c * (zb_plus_one_squared - 1) / (zb_plus_one_squared + 1)  # doctest: +REMOTE_DATA\n\n        or alternatively using just equivalencies::\n\n          >>> vcorr_rel = vcorr.to(u.Hz, u.doppler_optical(1*u.Hz)).to(vcorr.unit, u.doppler_relativistic(1*u.Hz))  # doctest: +REMOTE_DATA\n\n        See also `~astropy.units.equivalencies.doppler_optical`,\n        `~astropy.units.equivalencies.doppler_radio`, and\n        `~astropy.units.equivalencies.doppler_relativistic` for more information on\n        the velocity conventions.\n\n        The default is for this method to use the builtin ephemeris for\n        computing the sun and earth location.  Other ephemerides can be chosen\n        by setting the `~astropy.coordinates.solar_system_ephemeris` variable,\n        either directly or via ``with`` statement.  For example, to use the JPL\n        ephemeris, do::\n\n          >>> from astropy.coordinates import solar_system_ephemeris\n          >>> sc = SkyCoord(1*u.deg, 2*u.deg)\n          >>> with solar_system_ephemeris.set('jpl'):  # doctest: +REMOTE_DATA\n          ...     rv += sc.radial_velocity_correction(obstime=t, location=loc)  # doctest: +SKIP\n\n        "
        from .solar_system import get_body_barycentric_posvel
        timeloc = getattr(obstime, 'location', None)
        if location is None:
            if self.location is not None:
                location = self.location
                if timeloc is not None:
                    raise ValueError('`location` cannot be in both the passed-in `obstime` and this `SkyCoord` because it is ambiguous which is meant for the radial_velocity_correction.')
            elif timeloc is not None:
                location = timeloc
            else:
                raise TypeError('Must provide a `location` to radial_velocity_correction, either as a SkyCoord frame attribute, as an attribute on the passed in `obstime`, or in the method call.')
        elif self.location is not None or timeloc is not None:
            raise ValueError('Cannot compute radial velocity correction if `location` argument is passed in and there is also a  `location` attribute on this SkyCoord or the passed-in `obstime`.')
        coo_at_rv_obstime = self
        if obstime is None:
            obstime = self.obstime
            if obstime is None:
                raise TypeError('Must provide an `obstime` to radial_velocity_correction, either as a SkyCoord frame attribute or in the method call.')
        elif self.obstime is not None and self.frame.data.differentials:
            coo_at_rv_obstime = self.apply_space_motion(obstime)
        elif self.obstime is None:
            if 's' in self.data.differentials:
                warnings.warn('SkyCoord has space motion, and therefore the specified position of the SkyCoord may not be the same as the `obstime` for the radial velocity measurement. This may affect the rv correction at the order of km/sfor very high proper motions sources. If you wish to apply space motion of the SkyCoord to correct for thisthe `obstime` attribute of the SkyCoord must be set', AstropyUserWarning)
        (pos_earth, v_earth) = get_body_barycentric_posvel('earth', obstime)
        if kind == 'barycentric':
            v_origin_to_earth = v_earth
        elif kind == 'heliocentric':
            v_sun = get_body_barycentric_posvel('sun', obstime)[1]
            v_origin_to_earth = v_earth - v_sun
        else:
            raise ValueError(f"`kind` argument to radial_velocity_correction must be 'barycentric' or 'heliocentric', but got '{kind}'")
        (gcrs_p, gcrs_v) = location.get_gcrs_posvel(obstime)
        icrs_cart = coo_at_rv_obstime.icrs.cartesian
        icrs_cart_novel = icrs_cart.without_differentials()
        if self.data.__class__ is UnitSphericalRepresentation:
            targcart = icrs_cart_novel
        else:
            obs_icrs_cart = pos_earth + gcrs_p
            targcart = icrs_cart_novel - obs_icrs_cart
            targcart /= targcart.norm()
        if kind == 'barycentric':
            beta_obs = (v_origin_to_earth + gcrs_v) / speed_of_light
            gamma_obs = 1 / np.sqrt(1 - beta_obs.norm() ** 2)
            gr = location.gravitational_redshift(obstime)
            zb = gamma_obs * (1 + beta_obs.dot(targcart)) / (1 + gr / speed_of_light)
            if icrs_cart.differentials:
                try:
                    ro = self.icrs.cartesian
                    beta_star = ro.differentials['s'].to_cartesian() / speed_of_light
                    ro = ro.without_differentials()
                    ro /= ro.norm()
                    zb *= (1 + beta_star.dot(ro)) / (1 + beta_star.dot(targcart))
                except u.UnitConversionError:
                    warnings.warn('SkyCoord contains some velocity information, but not enough to calculate the full space motion of the source, and so this has been ignored for the purposes of calculating the radial velocity correction. This can lead to errors on the order of metres/second.', AstropyUserWarning)
            zb = zb - 1
            return zb * speed_of_light
        else:
            return targcart.dot(v_origin_to_earth + gcrs_v)

    @classmethod
    def guess_from_table(cls, table, **coord_kwargs):
        if False:
            print('Hello World!')
        '\n        A convenience method to create and return a new SkyCoord from the data\n        in an astropy Table.\n\n        This method matches table columns that start with the case-insensitive\n        names of the components of the requested frames (including\n        differentials), if they are also followed by a non-alphanumeric\n        character. It will also match columns that *end* with the component name\n        if a non-alphanumeric character is *before* it.\n\n        For example, the first rule means columns with names like\n        ``\'RA[J2000]\'`` or ``\'ra\'`` will be interpreted as ``ra`` attributes for\n        `~astropy.coordinates.ICRS` frames, but ``\'RAJ2000\'`` or ``\'radius\'``\n        are *not*. Similarly, the second rule applied to the\n        `~astropy.coordinates.Galactic` frame means that a column named\n        ``\'gal_l\'`` will be used as the ``l`` component, but ``gall`` or\n        ``\'fill\'`` will not.\n\n        The definition of alphanumeric here is based on Unicode\'s definition\n        of alphanumeric, except without ``_`` (which is normally considered\n        alphanumeric).  So for ASCII, this means the non-alphanumeric characters\n        are ``<space>_!"#$%&\'()*+,-./\\:;<=>?@[]^`{|}~``).\n\n        Parameters\n        ----------\n        table : `~astropy.table.Table` or subclass\n            The table to load data from.\n        **coord_kwargs\n            Any additional keyword arguments are passed directly to this class\'s\n            constructor.\n\n        Returns\n        -------\n        newsc : `~astropy.coordinates.SkyCoord` or subclass\n            The new instance.\n\n        Raises\n        ------\n        ValueError\n            If more than one match is found in the table for a component,\n            unless the additional matches are also valid frame component names.\n            If a "coord_kwargs" is provided for a value also found in the table.\n\n        '
        (_frame_cls, _frame_kwargs) = _get_frame_without_data([], coord_kwargs)
        frame = _frame_cls(**_frame_kwargs)
        coord_kwargs['frame'] = coord_kwargs.get('frame', frame)
        representation_component_names = set(frame.get_representation_component_names()).union(set(frame.get_representation_component_names('s')))
        comp_kwargs = {}
        for comp_name in representation_component_names:
            starts_with_comp = comp_name + '(\\W|\\b|_)'
            ends_with_comp = '.*(\\W|\\b|_)' + comp_name + '\\b'
            rex = re.compile(f'({starts_with_comp})|({ends_with_comp})', re.IGNORECASE | re.UNICODE)
            matches = {col_name for col_name in table.colnames if rex.match(col_name)}
            if len(matches) == 0:
                continue
            elif len(matches) == 1:
                col_name = matches.pop()
            else:
                matches -= representation_component_names - {comp_name}
                if len(matches) == 1:
                    col_name = matches.pop()
                else:
                    raise ValueError(f'Found at least two matches for component "{comp_name}": "{matches}". Cannot guess coordinates from a table with this ambiguity.')
            comp_kwargs[comp_name] = table[col_name]
        for (k, v) in comp_kwargs.items():
            if k in coord_kwargs:
                raise ValueError(f'Found column "{v.name}" in table, but it was already provided as "{{k}}" keyword to guess_from_table function.')
            else:
                coord_kwargs[k] = v
        return cls(**coord_kwargs)

    @classmethod
    def from_name(cls, name, frame='icrs', parse=False, cache=True):
        if False:
            while True:
                i = 10
        "\n        Given a name, query the CDS name resolver to attempt to retrieve\n        coordinate information for that object. The search database, sesame\n        url, and  query timeout can be set through configuration items in\n        ``astropy.coordinates.name_resolve`` -- see docstring for\n        `~astropy.coordinates.get_icrs_coordinates` for more\n        information.\n\n        Parameters\n        ----------\n        name : str\n            The name of the object to get coordinates for, e.g. ``'M42'``.\n        frame : str or `BaseCoordinateFrame` class or instance\n            The frame to transform the object to.\n        parse : bool\n            Whether to attempt extracting the coordinates from the name by\n            parsing with a regex. For objects catalog names that have\n            J-coordinates embedded in their names, e.g.,\n            'CRTS SSS100805 J194428-420209', this may be much faster than a\n            Sesame query for the same object name. The coordinates extracted\n            in this way may differ from the database coordinates by a few\n            deci-arcseconds, so only use this option if you do not need\n            sub-arcsecond accuracy for coordinates.\n        cache : bool, optional\n            Determines whether to cache the results or not. To update or\n            overwrite an existing value, pass ``cache='update'``.\n\n        Returns\n        -------\n        coord : SkyCoord\n            Instance of the SkyCoord class.\n        "
        from .name_resolve import get_icrs_coordinates
        icrs_coord = get_icrs_coordinates(name, parse, cache=cache)
        icrs_sky_coord = cls(icrs_coord)
        if frame in ('icrs', icrs_coord.__class__):
            return icrs_sky_coord
        else:
            return icrs_sky_coord.transform_to(frame)