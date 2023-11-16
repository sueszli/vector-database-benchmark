"""
Framework and base classes for coordinate frames/"low-level" coordinate
classes.
"""
from __future__ import annotations
__all__ = ['BaseCoordinateFrame', 'frame_transform_graph', 'GenericFrame', 'RepresentationMapping']
import copy
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from astropy import units as u
from astropy.utils import ShapedLikeNDArray, check_broadcast
from astropy.utils.decorators import deprecated, format_doc, lazyproperty
from astropy.utils.exceptions import AstropyWarning
from . import representation as r
from .angles import Angle
from .attributes import Attribute
from .transformations import TransformGraph
if TYPE_CHECKING:
    from astropy.units import Unit
frame_transform_graph = TransformGraph()

def _get_repr_cls(value):
    if False:
        return 10
    '\n    Return a valid representation class from ``value`` or raise exception.\n    '
    if value in r.REPRESENTATION_CLASSES:
        value = r.REPRESENTATION_CLASSES[value]
    elif not isinstance(value, type) or not issubclass(value, r.BaseRepresentation):
        raise ValueError(f'Representation is {value!r} but must be a BaseRepresentation class or one of the string aliases {list(r.REPRESENTATION_CLASSES)}')
    return value

def _get_diff_cls(value):
    if False:
        print('Hello World!')
    '\n    Return a valid differential class from ``value`` or raise exception.\n\n    As originally created, this is only used in the SkyCoord initializer, so if\n    that is refactored, this function my no longer be necessary.\n    '
    if value in r.DIFFERENTIAL_CLASSES:
        value = r.DIFFERENTIAL_CLASSES[value]
    elif not isinstance(value, type) or not issubclass(value, r.BaseDifferential):
        raise ValueError(f'Differential is {value!r} but must be a BaseDifferential class or one of the string aliases {list(r.DIFFERENTIAL_CLASSES)}')
    return value

def _get_repr_classes(base, **differentials):
    if False:
        i = 10
        return i + 15
    "Get valid representation and differential classes.\n\n    Parameters\n    ----------\n    base : str or `~astropy.coordinates.BaseRepresentation` subclass\n        class for the representation of the base coordinates.  If a string,\n        it is looked up among the known representation classes.\n    **differentials : dict of str or `~astropy.coordinates.BaseDifferentials`\n        Keys are like for normal differentials, i.e., 's' for a first\n        derivative in time, etc.  If an item is set to `None`, it will be\n        guessed from the base class.\n\n    Returns\n    -------\n    repr_classes : dict of subclasses\n        The base class is keyed by 'base'; the others by the keys of\n        ``diffferentials``.\n    "
    base = _get_repr_cls(base)
    repr_classes = {'base': base}
    for (name, differential_type) in differentials.items():
        if differential_type == 'base':
            differential_type = r.DIFFERENTIAL_CLASSES.get(base.get_name(), None)
        elif differential_type in r.DIFFERENTIAL_CLASSES:
            differential_type = r.DIFFERENTIAL_CLASSES[differential_type]
        elif differential_type is not None and (not isinstance(differential_type, type) or not issubclass(differential_type, r.BaseDifferential)):
            raise ValueError(f'Differential is {{differential_type!r}} but must be a BaseDifferential class or one of the string aliases {list(r.DIFFERENTIAL_CLASSES)}')
        repr_classes[name] = differential_type
    return repr_classes

class RepresentationMapping(NamedTuple):
    """
    This :class:`~typing.NamedTuple` is used with the
    ``frame_specific_representation_info`` attribute to tell frames what
    attribute names (and default units) to use for a particular representation.
    ``reprname`` and ``framename`` should be strings, while ``defaultunit`` can
    be either an astropy unit, the string ``'recommended'`` (which is degrees
    for Angles, nothing otherwise), or None (to indicate that no unit mapping
    should be done).
    """
    reprname: str
    framename: str
    defaultunit: str | Unit = 'recommended'
base_doc = "{__doc__}\n    Parameters\n    ----------\n    data : `~astropy.coordinates.BaseRepresentation` subclass instance\n        A representation object or ``None`` to have no data (or use the\n        coordinate component arguments, see below).\n    {components}\n    representation_type : `~astropy.coordinates.BaseRepresentation` subclass, str, optional\n        A representation class or string name of a representation class. This\n        sets the expected input representation class, thereby changing the\n        expected keyword arguments for the data passed in. For example, passing\n        ``representation_type='cartesian'`` will make the classes expect\n        position data with cartesian names, i.e. ``x, y, z`` in most cases\n        unless overridden via ``frame_specific_representation_info``. To see this\n        frame's names, check out ``<this frame>().representation_info``.\n    differential_type : `~astropy.coordinates.BaseDifferential` subclass, str, dict, optional\n        A differential class or dictionary of differential classes (currently\n        only a velocity differential with key 's' is supported). This sets the\n        expected input differential class, thereby changing the expected keyword\n        arguments of the data passed in. For example, passing\n        ``differential_type='cartesian'`` will make the classes expect velocity\n        data with the argument names ``v_x, v_y, v_z`` unless overridden via\n        ``frame_specific_representation_info``. To see this frame's names,\n        check out ``<this frame>().representation_info``.\n    copy : bool, optional\n        If `True` (default), make copies of the input coordinate arrays.\n        Can only be passed in as a keyword argument.\n    {footer}\n"
_components = '\n    *args, **kwargs\n        Coordinate components, with names that depend on the subclass.\n'

@format_doc(base_doc, components=_components, footer='')
class BaseCoordinateFrame(ShapedLikeNDArray):
    """
    The base class for coordinate frames.

    This class is intended to be subclassed to create instances of specific
    systems.  Subclasses can implement the following attributes:

    * `default_representation`
        A subclass of `~astropy.coordinates.BaseRepresentation` that will be
        treated as the default representation of this frame.  This is the
        representation assumed by default when the frame is created.

    * `default_differential`
        A subclass of `~astropy.coordinates.BaseDifferential` that will be
        treated as the default differential class of this frame.  This is the
        differential class assumed by default when the frame is created.

    * `~astropy.coordinates.Attribute` class attributes
       Frame attributes such as ``FK4.equinox`` or ``FK4.obstime`` are defined
       using a descriptor class.  See the narrative documentation or
       built-in classes code for details.

    * `frame_specific_representation_info`
        A dictionary mapping the name or class of a representation to a list of
        `~astropy.coordinates.RepresentationMapping` objects that tell what
        names and default units should be used on this frame for the components
        of that representation.

    Unless overridden via `frame_specific_representation_info`, velocity name
    defaults are:

      * ``pm_{lon}_cos{lat}``, ``pm_{lat}`` for `~astropy.coordinates.SphericalCosLatDifferential` velocity components
      * ``pm_{lon}``, ``pm_{lat}`` for `~astropy.coordinates.SphericalDifferential` velocity components
      * ``radial_velocity`` for any ``d_distance`` component
      * ``v_{x,y,z}`` for `~astropy.coordinates.CartesianDifferential` velocity components

    where ``{lon}`` and ``{lat}`` are the frame names of the angular components.
    """
    default_representation = None
    default_differential = None
    frame_specific_representation_info = {}
    frame_attributes = {}

    def __init_subclass__(cls, **kwargs):
        if False:
            i = 10
            return i + 15
        default_repr = getattr(cls, 'default_representation', None)
        default_diff = getattr(cls, 'default_differential', None)
        repr_info = getattr(cls, 'frame_specific_representation_info', None)
        if default_repr is None or isinstance(default_repr, property):
            default_repr = getattr(cls, '_default_representation', None)
        if default_diff is None or isinstance(default_diff, property):
            default_diff = getattr(cls, '_default_differential', None)
        if repr_info is None or isinstance(repr_info, property):
            repr_info = getattr(cls, '_frame_specific_representation_info', None)
        repr_info = cls._infer_repr_info(repr_info)
        cls._create_readonly_property('default_representation', default_repr, 'Default representation for position data')
        cls._create_readonly_property('default_differential', default_diff, 'Default representation for differential data (e.g., velocity)')
        cls._create_readonly_property('frame_specific_representation_info', copy.deepcopy(repr_info), 'Mapping for frame-specific component names')
        frame_attrs = {}
        for basecls in reversed(cls.__bases__):
            if issubclass(basecls, BaseCoordinateFrame):
                frame_attrs.update(basecls.frame_attributes)
        for (k, v) in cls.__dict__.items():
            if isinstance(v, Attribute):
                frame_attrs[k] = v
        cls.frame_attributes = frame_attrs
        if not hasattr(cls, 'name'):
            cls.name = cls.__name__.lower()
        elif BaseCoordinateFrame not in cls.__bases__ and cls.name in [getattr(base, 'name', None) for base in cls.__bases__]:
            cls.name = cls.__name__.lower()
        cls._frame_class_cache = {}
        super().__init_subclass__(**kwargs)
        cls.get_frame_attr_defaults()

    def __init__(self, *args, copy=True, representation_type=None, differential_type=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._attr_names_with_defaults = []
        self._representation = self._infer_representation(representation_type, differential_type)
        data = self._infer_data(args, copy, kwargs)
        shapes = [] if data is None else [data.shape]
        values = {}
        for (fnm, fdefault) in self.get_frame_attr_defaults().items():
            if fnm in kwargs:
                value = kwargs.pop(fnm)
                setattr(self, '_' + fnm, value)
                values[fnm] = value = getattr(self, fnm)
                shapes.append(getattr(value, 'shape', ()))
            else:
                setattr(self, '_' + fnm, fdefault)
                self._attr_names_with_defaults.append(fnm)
        if kwargs:
            raise TypeError(f'Coordinate frame {self.__class__.__name__} got unexpected keywords: {list(kwargs)}')
        try:
            self._shape = check_broadcast(*shapes)
        except ValueError as err:
            raise ValueError(f'non-scalar data and/or attributes with inconsistent shapes: {shapes}') from err
        if data is not None and data.shape != self._shape:
            data = data._apply(np.broadcast_to, shape=self._shape, subok=True)
        self._data = data
        for key in values:
            getattr(self, key)
        if self.has_data:
            if 's' in self._data.differentials:
                key = (self._data.__class__.__name__, self._data.differentials['s'].__class__.__name__, False)
            else:
                key = (self._data.__class__.__name__, False)
            self.cache['representation'][key] = self._data

    def _infer_representation(self, representation_type, differential_type):
        if False:
            i = 10
            return i + 15
        if representation_type is None and differential_type is None:
            return {'base': self.default_representation, 's': self.default_differential}
        if representation_type is None:
            representation_type = self.default_representation
        if isinstance(differential_type, type) and issubclass(differential_type, r.BaseDifferential):
            differential_type = {'s': differential_type}
        elif isinstance(differential_type, str):
            diff_cls = r.DIFFERENTIAL_CLASSES[differential_type]
            differential_type = {'s': diff_cls}
        elif differential_type is None:
            if representation_type == self.default_representation:
                differential_type = {'s': self.default_differential}
            else:
                differential_type = {'s': 'base'}
        return _get_repr_classes(representation_type, **differential_type)

    def _infer_data(self, args, copy, kwargs):
        if False:
            while True:
                i = 10
        representation_data = None
        differential_data = None
        args = list(args)
        if args and (isinstance(args[0], r.BaseRepresentation) or args[0] is None):
            representation_data = args.pop(0)
            if len(args) > 0:
                raise TypeError('Cannot create a frame with both a representation object and other positional arguments')
            if representation_data is not None:
                diffs = representation_data.differentials
                differential_data = diffs.get('s', None)
                if differential_data is None and len(diffs) > 0 or (differential_data is not None and len(diffs) > 1):
                    raise ValueError(f'Multiple differentials are associated with the representation object passed in to the frame initializer. Only a single velocity differential is supported. Got: {diffs}')
        else:
            representation_cls = self.get_representation_cls()
            repr_kwargs = {}
            for (nmkw, nmrep) in self.representation_component_names.items():
                if len(args) > 0:
                    repr_kwargs[nmrep] = args.pop(0)
                elif nmkw in kwargs:
                    repr_kwargs[nmrep] = kwargs.pop(nmkw)
            if repr_kwargs:
                if repr_kwargs.get('distance', True) is None:
                    del repr_kwargs['distance']
                if issubclass(representation_cls, r.SphericalRepresentation) and 'distance' not in repr_kwargs:
                    representation_cls = representation_cls._unit_representation
                try:
                    representation_data = representation_cls(copy=copy, **repr_kwargs)
                except TypeError as e:
                    try:
                        representation_data = representation_cls._unit_representation(copy=copy, **repr_kwargs)
                    except Exception:
                        msg = str(e)
                        names = self.get_representation_component_names()
                        for (frame_name, repr_name) in names.items():
                            msg = msg.replace(repr_name, frame_name)
                        msg = msg.replace('__init__()', f'{self.__class__.__name__}()')
                        e.args = (msg,)
                        raise e
            differential_cls = self.get_representation_cls('s')
            diff_component_names = self.get_representation_component_names('s')
            diff_kwargs = {}
            for (nmkw, nmrep) in diff_component_names.items():
                if len(args) > 0:
                    diff_kwargs[nmrep] = args.pop(0)
                elif nmkw in kwargs:
                    diff_kwargs[nmrep] = kwargs.pop(nmkw)
            if diff_kwargs:
                if hasattr(differential_cls, '_unit_differential') and 'd_distance' not in diff_kwargs:
                    differential_cls = differential_cls._unit_differential
                elif len(diff_kwargs) == 1 and 'd_distance' in diff_kwargs:
                    differential_cls = r.RadialDifferential
                try:
                    differential_data = differential_cls(copy=copy, **diff_kwargs)
                except TypeError as e:
                    msg = str(e)
                    names = self.get_representation_component_names('s')
                    for (frame_name, repr_name) in names.items():
                        msg = msg.replace(repr_name, frame_name)
                    msg = msg.replace('__init__()', f'{self.__class__.__name__}()')
                    e.args = (msg,)
                    raise
        if len(args) > 0:
            raise TypeError('{}.__init__ had {} remaining unhandled arguments'.format(self.__class__.__name__, len(args)))
        if representation_data is None and differential_data is not None:
            raise ValueError('Cannot pass in differential component data without positional (representation) data.')
        if differential_data:
            for comp in representation_data.components:
                if (diff_comp := f'd_{comp}') in differential_data.components:
                    current_repr_unit = representation_data._units[comp]
                    current_diff_unit = differential_data._units[diff_comp]
                    expected_unit = current_repr_unit / u.s
                    if not current_diff_unit.is_equivalent(expected_unit):
                        for (key, val) in self.get_representation_component_names().items():
                            if val == comp:
                                current_repr_name = key
                                break
                        for (key, val) in self.get_representation_component_names('s').items():
                            if val == diff_comp:
                                current_diff_name = key
                                break
                        raise ValueError(f'{current_repr_name} has unit "{current_repr_unit}" with physical type "{current_repr_unit.physical_type}", but {current_diff_name} has incompatible unit "{current_diff_unit}" with physical type "{current_diff_unit.physical_type}" instead of the expected "{expected_unit.physical_type}".')
            representation_data = representation_data.with_differentials({'s': differential_data})
        return representation_data

    @classmethod
    def _infer_repr_info(cls, repr_info):
        if False:
            print('Hello World!')
        if repr_info is None:
            repr_info = {}
        for cls_or_name in tuple(repr_info.keys()):
            if isinstance(cls_or_name, str):
                _cls = _get_repr_cls(cls_or_name)
                repr_info[_cls] = repr_info.pop(cls_or_name)
        repr_info.setdefault(r.SphericalRepresentation, [RepresentationMapping('lon', 'lon'), RepresentationMapping('lat', 'lat')])
        sph_component_map = {m.reprname: m.framename for m in repr_info[r.SphericalRepresentation]}
        repr_info.setdefault(r.SphericalCosLatDifferential, [RepresentationMapping('d_lon_coslat', 'pm_{lon}_cos{lat}'.format(**sph_component_map), u.mas / u.yr), RepresentationMapping('d_lat', 'pm_{lat}'.format(**sph_component_map), u.mas / u.yr), RepresentationMapping('d_distance', 'radial_velocity', u.km / u.s)])
        repr_info.setdefault(r.SphericalDifferential, [RepresentationMapping('d_lon', 'pm_{lon}'.format(**sph_component_map), u.mas / u.yr), RepresentationMapping('d_lat', 'pm_{lat}'.format(**sph_component_map), u.mas / u.yr), RepresentationMapping('d_distance', 'radial_velocity', u.km / u.s)])
        repr_info.setdefault(r.CartesianDifferential, [RepresentationMapping('d_x', 'v_x', u.km / u.s), RepresentationMapping('d_y', 'v_y', u.km / u.s), RepresentationMapping('d_z', 'v_z', u.km / u.s)])
        repr_info.setdefault(r.UnitSphericalRepresentation, repr_info[r.SphericalRepresentation])
        repr_info.setdefault(r.UnitSphericalCosLatDifferential, repr_info[r.SphericalCosLatDifferential])
        repr_info.setdefault(r.UnitSphericalDifferential, repr_info[r.SphericalDifferential])
        return repr_info

    @classmethod
    def _create_readonly_property(cls, attr_name, value, doc=None):
        if False:
            print('Hello World!')
        private_attr = '_' + attr_name

        def getter(self):
            if False:
                while True:
                    i = 10
            return getattr(self, private_attr)
        setattr(cls, private_attr, value)
        setattr(cls, attr_name, property(getter, doc=doc))

    @lazyproperty
    def cache(self):
        if False:
            for i in range(10):
                print('nop')
        "Cache for this frame, a dict.\n\n        It stores anything that should be computed from the coordinate data (*not* from\n        the frame attributes). This can be used in functions to store anything that\n        might be expensive to compute but might be re-used by some other function.\n        E.g.::\n\n            if 'user_data' in myframe.cache:\n                data = myframe.cache['user_data']\n            else:\n                myframe.cache['user_data'] = data = expensive_func(myframe.lat)\n\n        If in-place modifications are made to the frame data, the cache should\n        be cleared::\n\n            myframe.cache.clear()\n\n        "
        return defaultdict(dict)

    @property
    def data(self):
        if False:
            i = 10
            return i + 15
        '\n        The coordinate data for this object.  If this frame has no data, an\n        `ValueError` will be raised.  Use `has_data` to\n        check if data is present on this frame object.\n        '
        if self._data is None:
            raise ValueError(f'The frame object "{self!r}" does not have associated data')
        return self._data

    @property
    def has_data(self):
        if False:
            while True:
                i = 10
        '\n        True if this frame has `data`, False otherwise.\n        '
        return self._data is not None

    @property
    def shape(self):
        if False:
            print('Hello World!')
        return self._shape

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.data)

    def __bool__(self):
        if False:
            return 10
        return self.has_data and self.size > 0

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        return self.data.size

    @property
    def isscalar(self):
        if False:
            print('Hello World!')
        return self.has_data and self.data.isscalar

    @classmethod
    def get_frame_attr_defaults(cls):
        if False:
            i = 10
            return i + 15
        'Return a dict with the defaults for each frame attribute.'
        return {name: getattr(cls, name) for name in cls.frame_attributes}

    @deprecated('5.2', alternative='get_frame_attr_defaults', message='The {func}() {obj_type} is deprecated and may be removed in a future version. Use {alternative}() to obtain a dict of frame attribute names and default values. The fastest way to obtain the names is frame_attributes.keys()')
    @classmethod
    def get_frame_attr_names(cls):
        if False:
            print('Hello World!')
        'Return a dict with the defaults for each frame attribute.'
        return cls.get_frame_attr_defaults()

    def get_representation_cls(self, which='base'):
        if False:
            for i in range(10):
                print('nop')
        "The class used for part of this frame's data.\n\n        Parameters\n        ----------\n        which : ('base', 's', `None`)\n            The class of which part to return.  'base' means the class used to\n            represent the coordinates; 's' the first derivative to time, i.e.,\n            the class representing the proper motion and/or radial velocity.\n            If `None`, return a dict with both.\n\n        Returns\n        -------\n        representation : `~astropy.coordinates.BaseRepresentation` or `~astropy.coordinates.BaseDifferential`.\n        "
        if which is not None:
            return self._representation[which]
        else:
            return self._representation

    def set_representation_cls(self, base=None, s='base'):
        if False:
            while True:
                i = 10
        "Set representation and/or differential class for this frame's data.\n\n        Parameters\n        ----------\n        base : str, `~astropy.coordinates.BaseRepresentation` subclass, optional\n            The name or subclass to use to represent the coordinate data.\n        s : `~astropy.coordinates.BaseDifferential` subclass, optional\n            The differential subclass to use to represent any velocities,\n            such as proper motion and radial velocity.  If equal to 'base',\n            which is the default, it will be inferred from the representation.\n            If `None`, the representation will drop any differentials.\n        "
        if base is None:
            base = self._representation['base']
        self._representation = _get_repr_classes(base=base, s=s)
    representation_type = property(fget=get_representation_cls, fset=set_representation_cls, doc="The representation class used for this frame's data.\n\n        This will be a subclass from `~astropy.coordinates.BaseRepresentation`.\n        Can also be *set* using the string name of the representation. If you\n        wish to set an explicit differential class (rather than have it be\n        inferred), use the ``set_representation_cls`` method.\n        ")

    @property
    def differential_type(self):
        if False:
            return 10
        "\n        The differential used for this frame's data.\n\n        This will be a subclass from `~astropy.coordinates.BaseDifferential`.\n        For simultaneous setting of representation and differentials, see the\n        ``set_representation_cls`` method.\n        "
        return self.get_representation_cls('s')

    @differential_type.setter
    def differential_type(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.set_representation_cls(s=value)

    @classmethod
    def _get_representation_info(cls):
        if False:
            i = 10
            return i + 15
        if cls._frame_class_cache.get('last_reprdiff_hash', None) != r.get_reprdiff_cls_hash():
            repr_attrs = {}
            for repr_diff_cls in list(r.REPRESENTATION_CLASSES.values()) + list(r.DIFFERENTIAL_CLASSES.values()):
                repr_attrs[repr_diff_cls] = {'names': [], 'units': []}
                for (c, c_cls) in repr_diff_cls.attr_classes.items():
                    repr_attrs[repr_diff_cls]['names'].append(c)
                    rec_unit = u.deg if issubclass(c_cls, Angle) else None
                    repr_attrs[repr_diff_cls]['units'].append(rec_unit)
            for (repr_diff_cls, mappings) in cls._frame_specific_representation_info.items():
                nms = repr_attrs[repr_diff_cls]['names']
                uns = repr_attrs[repr_diff_cls]['units']
                comptomap = {m.reprname: m for m in mappings}
                for (i, c) in enumerate(repr_diff_cls.attr_classes.keys()):
                    if c in comptomap:
                        mapp = comptomap[c]
                        nms[i] = mapp.framename
                        if not (isinstance(mapp.defaultunit, str) and mapp.defaultunit == 'recommended'):
                            uns[i] = mapp.defaultunit
                repr_attrs[repr_diff_cls]['names'] = tuple(nms)
                repr_attrs[repr_diff_cls]['units'] = tuple(uns)
            cls._frame_class_cache['representation_info'] = repr_attrs
            cls._frame_class_cache['last_reprdiff_hash'] = r.get_reprdiff_cls_hash()
        return cls._frame_class_cache['representation_info']

    @lazyproperty
    def representation_info(self):
        if False:
            while True:
                i = 10
        '\n        A dictionary with the information of what attribute names for this frame\n        apply to particular representations.\n        '
        return self._get_representation_info()

    def get_representation_component_names(self, which='base'):
        if False:
            for i in range(10):
                print('nop')
        out = {}
        repr_or_diff_cls = self.get_representation_cls(which)
        if repr_or_diff_cls is None:
            return out
        data_names = repr_or_diff_cls.attr_classes.keys()
        repr_names = self.representation_info[repr_or_diff_cls]['names']
        for (repr_name, data_name) in zip(repr_names, data_names):
            out[repr_name] = data_name
        return out

    def get_representation_component_units(self, which='base'):
        if False:
            for i in range(10):
                print('nop')
        out = {}
        repr_or_diff_cls = self.get_representation_cls(which)
        if repr_or_diff_cls is None:
            return out
        repr_attrs = self.representation_info[repr_or_diff_cls]
        repr_names = repr_attrs['names']
        repr_units = repr_attrs['units']
        for (repr_name, repr_unit) in zip(repr_names, repr_units):
            if repr_unit:
                out[repr_name] = repr_unit
        return out
    representation_component_names = property(get_representation_component_names)
    representation_component_units = property(get_representation_component_units)

    def _replicate(self, data, copy=False, **kwargs):
        if False:
            while True:
                i = 10
        'Base for replicating a frame, with possibly different attributes.\n\n        Produces a new instance of the frame using the attributes of the old\n        frame (unless overridden) and with the data given.\n\n        Parameters\n        ----------\n        data : `~astropy.coordinates.BaseRepresentation` or None\n            Data to use in the new frame instance.  If `None`, it will be\n            a data-less frame.\n        copy : bool, optional\n            Whether data and the attributes on the old frame should be copied\n            (default), or passed on by reference.\n        **kwargs\n            Any attributes that should be overridden.\n        '
        if isinstance(data, type):
            raise TypeError('Class passed as data instead of a representation instance. If you called frame.representation, this returns the representation class. frame.data returns the instantiated object - you may want to  use this instead.')
        if copy and data is not None:
            data = data.copy()
        for attr in self.frame_attributes:
            if attr not in self._attr_names_with_defaults and attr not in kwargs:
                value = getattr(self, attr)
                if copy:
                    value = value.copy()
                kwargs[attr] = value
        return self.__class__(data, copy=False, **kwargs)

    def replicate(self, copy=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a replica of the frame, optionally with new frame attributes.\n\n        The replica is a new frame object that has the same data as this frame\n        object and with frame attributes overridden if they are provided as extra\n        keyword arguments to this method. If ``copy`` is set to `True` then a\n        copy of the internal arrays will be made.  Otherwise the replica will\n        use a reference to the original arrays when possible to save memory. The\n        internal arrays are normally not changeable by the user so in most cases\n        it should not be necessary to set ``copy`` to `True`.\n\n        Parameters\n        ----------\n        copy : bool, optional\n            If True, the resulting object is a copy of the data.  When False,\n            references are used where  possible. This rule also applies to the\n            frame attributes.\n        **kwargs\n            Any additional keywords are treated as frame attributes to be set on the\n            new frame object.\n\n        Returns\n        -------\n        frameobj : `~astropy.coordinates.BaseCoordinateFrame` subclass instance\n            Replica of this object, but possibly with new frame attributes.\n        '
        return self._replicate(self.data, copy=copy, **kwargs)

    def replicate_without_data(self, copy=False, **kwargs):
        if False:
            return 10
        '\n        Return a replica without data, optionally with new frame attributes.\n\n        The replica is a new frame object without data but with the same frame\n        attributes as this object, except where overridden by extra keyword\n        arguments to this method.  The ``copy`` keyword determines if the frame\n        attributes are truly copied vs being references (which saves memory for\n        cases where frame attributes are large).\n\n        This method is essentially the converse of `realize_frame`.\n\n        Parameters\n        ----------\n        copy : bool, optional\n            If True, the resulting object has copies of the frame attributes.\n            When False, references are used where  possible.\n        **kwargs\n            Any additional keywords are treated as frame attributes to be set on the\n            new frame object.\n\n        Returns\n        -------\n        frameobj : `~astropy.coordinates.BaseCoordinateFrame` subclass instance\n            Replica of this object, but without data and possibly with new frame\n            attributes.\n        '
        return self._replicate(None, copy=copy, **kwargs)

    def realize_frame(self, data, **kwargs):
        if False:
            return 10
        '\n        Generates a new frame with new data from another frame (which may or\n        may not have data). Roughly speaking, the converse of\n        `replicate_without_data`.\n\n        Parameters\n        ----------\n        data : `~astropy.coordinates.BaseRepresentation`\n            The representation to use as the data for the new frame.\n        **kwargs\n            Any additional keywords are treated as frame attributes to be set on the\n            new frame object. In particular, `representation_type` can be specified.\n\n        Returns\n        -------\n        frameobj : `~astropy.coordinates.BaseCoordinateFrame` subclass instance\n            A new object in *this* frame, with the same frame attributes as\n            this one, but with the ``data`` as the coordinate data.\n\n        '
        return self._replicate(data, **kwargs)

    def represent_as(self, base, s='base', in_frame_units=False):
        if False:
            while True:
                i = 10
        "\n        Generate and return a new representation of this frame's `data`\n        as a Representation object.\n\n        Note: In order to make an in-place change of the representation\n        of a Frame or SkyCoord object, set the ``representation``\n        attribute of that object to the desired new representation, or\n        use the ``set_representation_cls`` method to also set the differential.\n\n        Parameters\n        ----------\n        base : subclass of BaseRepresentation or string\n            The type of representation to generate.  Must be a *class*\n            (not an instance), or the string name of the representation\n            class.\n        s : subclass of `~astropy.coordinates.BaseDifferential`, str, optional\n            Class in which any velocities should be represented. Must be\n            a *class* (not an instance), or the string name of the\n            differential class.  If equal to 'base' (default), inferred from\n            the base class.  If `None`, all velocity information is dropped.\n        in_frame_units : bool, keyword-only\n            Force the representation units to match the specified units\n            particular to this frame\n\n        Returns\n        -------\n        newrep : BaseRepresentation-derived object\n            A new representation object of this frame's `data`.\n\n        Raises\n        ------\n        AttributeError\n            If this object had no `data`\n\n        Examples\n        --------\n        >>> from astropy import units as u\n        >>> from astropy.coordinates import SkyCoord, CartesianRepresentation\n        >>> coord = SkyCoord(0*u.deg, 0*u.deg)\n        >>> coord.represent_as(CartesianRepresentation)  # doctest: +FLOAT_CMP\n        <CartesianRepresentation (x, y, z) [dimensionless]\n                (1., 0., 0.)>\n\n        >>> coord.representation_type = CartesianRepresentation\n        >>> coord  # doctest: +FLOAT_CMP\n        <SkyCoord (ICRS): (x, y, z) [dimensionless]\n            (1., 0., 0.)>\n        "
        if isinstance(s, bool):
            warnings.warn('The argument position for `in_frame_units` in `represent_as` has changed. Use as a keyword argument if needed.', AstropyWarning)
            in_frame_units = s
            s = 'base'
        repr_classes = _get_repr_classes(base=base, s=s)
        representation_cls = repr_classes['base']
        if 's' in self.data.differentials:
            if s == 'base' and self.data.differentials['s'].__class__ in representation_cls._compatible_differentials:
                differential_cls = self.data.differentials['s'].__class__
            else:
                differential_cls = repr_classes['s']
        elif s is None or s == 'base':
            differential_cls = None
        else:
            raise TypeError('Frame data has no associated differentials (i.e. the frame has no velocity data) - represent_as() only accepts a new representation.')
        if differential_cls:
            cache_key = (representation_cls.__name__, differential_cls.__name__, in_frame_units)
        else:
            cache_key = (representation_cls.__name__, in_frame_units)
        cached_repr = self.cache['representation'].get(cache_key)
        if not cached_repr:
            if differential_cls:
                if isinstance(self.data, r.UnitSphericalRepresentation) and issubclass(representation_cls, r.CartesianRepresentation) and (not isinstance(self.data.differentials['s'], (r.UnitSphericalDifferential, r.UnitSphericalCosLatDifferential, r.RadialDifferential))):
                    raise u.UnitConversionError('need a distance to retrieve a cartesian representation when both radial velocity and proper motion are present, since otherwise the units cannot match.')
                data = self.data.represent_as(representation_cls, differential_cls)
                diff = data.differentials['s']
            else:
                data = self.data.represent_as(representation_cls)
            new_attrs = self.representation_info.get(representation_cls)
            if new_attrs and in_frame_units:
                datakwargs = {comp: getattr(data, comp) for comp in data.components}
                for (comp, new_attr_unit) in zip(data.components, new_attrs['units']):
                    if new_attr_unit:
                        datakwargs[comp] = datakwargs[comp].to(new_attr_unit)
                data = data.__class__(copy=False, **datakwargs)
            if differential_cls:
                data_diff = self.data.differentials['s']
                new_attrs = self.representation_info.get(differential_cls)
                if new_attrs and in_frame_units:
                    diffkwargs = {comp: getattr(diff, comp) for comp in diff.components}
                    for (comp, new_attr_unit) in zip(diff.components, new_attrs['units']):
                        if isinstance(data_diff, (r.UnitSphericalDifferential, r.UnitSphericalCosLatDifferential)) and comp not in data_diff.__class__.attr_classes:
                            continue
                        elif isinstance(data_diff, r.RadialDifferential) and comp not in data_diff.__class__.attr_classes:
                            continue
                        if new_attr_unit and hasattr(diff, comp):
                            try:
                                diffkwargs[comp] = diffkwargs[comp].to(new_attr_unit)
                            except Exception:
                                pass
                    diff = diff.__class__(copy=False, **diffkwargs)
                    data._differentials.update({'s': diff})
            self.cache['representation'][cache_key] = data
        return self.cache['representation'][cache_key]

    def transform_to(self, new_frame):
        if False:
            print('Hello World!')
        "\n        Transform this object's coordinate data to a new frame.\n\n        Parameters\n        ----------\n        new_frame : coordinate-like\n            The frame to transform this coordinate frame into.\n\n        Returns\n        -------\n        transframe : coordinate-like\n            A new object with the coordinate data represented in the\n            ``newframe`` system.\n\n        Raises\n        ------\n        ValueError\n            If there is no possible transformation route.\n        "
        from .errors import ConvertError
        if self._data is None:
            raise ValueError('Cannot transform a frame with no data')
        if getattr(self.data, 'differentials', None) and hasattr(self, 'obstime') and hasattr(new_frame, 'obstime') and np.any(self.obstime != new_frame.obstime):
            raise NotImplementedError('You cannot transform a frame that has velocities to another frame at a different obstime. If you think this should (or should not) be possible, please comment at https://github.com/astropy/astropy/issues/6280')
        if hasattr(new_frame, '_sky_coord_frame'):
            new_frame = new_frame._sky_coord_frame
        trans = frame_transform_graph.get_transform(self.__class__, new_frame.__class__)
        if trans is None:
            if new_frame is self.__class__:
                return new_frame.realize_frame(self.data)
            msg = 'Cannot transform from {0} to {1}'
            raise ConvertError(msg.format(self.__class__, new_frame.__class__))
        return trans(self, new_frame)

    def is_transformable_to(self, new_frame):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines if this coordinate frame can be transformed to another\n        given frame.\n\n        Parameters\n        ----------\n        new_frame : `~astropy.coordinates.BaseCoordinateFrame` subclass or instance\n            The proposed frame to transform into.\n\n        Returns\n        -------\n        transformable : bool or str\n            `True` if this can be transformed to ``new_frame``, `False` if\n            not, or the string 'same' if ``new_frame`` is the same system as\n            this object but no transformation is defined.\n\n        Notes\n        -----\n        A return value of 'same' means the transformation will work, but it will\n        just give back a copy of this object.  The intended usage is::\n\n            if coord.is_transformable_to(some_unknown_frame):\n                coord2 = coord.transform_to(some_unknown_frame)\n\n        This will work even if ``some_unknown_frame``  turns out to be the same\n        frame class as ``coord``.  This is intended for cases where the frame\n        is the same regardless of the frame attributes (e.g. ICRS), but be\n        aware that it *might* also indicate that someone forgot to define the\n        transformation between two objects of the same frame class but with\n        different attributes.\n        "
        new_frame_cls = new_frame if isinstance(new_frame, type) else type(new_frame)
        trans = frame_transform_graph.get_transform(self.__class__, new_frame_cls)
        if trans is None:
            if new_frame_cls is self.__class__:
                return 'same'
            else:
                return False
        else:
            return True

    def is_frame_attr_default(self, attrnm):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determine whether or not a frame attribute has its value because it's\n        the default value, or because this frame was created with that value\n        explicitly requested.\n\n        Parameters\n        ----------\n        attrnm : str\n            The name of the attribute to check.\n\n        Returns\n        -------\n        isdefault : bool\n            True if the attribute ``attrnm`` has its value by default, False if\n            it was specified at creation of this frame.\n        "
        return attrnm in self._attr_names_with_defaults

    @staticmethod
    def _frameattr_equiv(left_fattr, right_fattr):
        if False:
            print('Hello World!')
        '\n        Determine if two frame attributes are equivalent.  Implemented as a\n        staticmethod mainly as a convenient location, although conceivable it\n        might be desirable for subclasses to override this behavior.\n\n        Primary purpose is to check for equality of representations.  This\n        aspect can actually be simplified/removed now that representations have\n        equality defined.\n\n        Secondary purpose is to check for equality of coordinate attributes,\n        which first checks whether they themselves are in equivalent frames\n        before checking for equality in the normal fashion.  This is because\n        checking for equality with non-equivalent frames raises an error.\n        '
        if left_fattr is right_fattr:
            return True
        elif left_fattr is None or right_fattr is None:
            return False
        left_is_repr = isinstance(left_fattr, r.BaseRepresentationOrDifferential)
        right_is_repr = isinstance(right_fattr, r.BaseRepresentationOrDifferential)
        if left_is_repr and right_is_repr:
            if getattr(left_fattr, 'differentials', False) or getattr(right_fattr, 'differentials', False):
                warnings.warn('Two representation frame attributes were checked for equivalence when at least one of them has differentials.  This yields False even if the underlying representations are equivalent (although this may change in future versions of Astropy)', AstropyWarning)
                return False
            if isinstance(right_fattr, left_fattr.__class__):
                return np.all([getattr(left_fattr, comp) == getattr(right_fattr, comp) for comp in left_fattr.components])
            else:
                return np.all(left_fattr.to_cartesian().xyz == right_fattr.to_cartesian().xyz)
        elif left_is_repr or right_is_repr:
            return False
        left_is_coord = isinstance(left_fattr, BaseCoordinateFrame)
        right_is_coord = isinstance(right_fattr, BaseCoordinateFrame)
        if left_is_coord and right_is_coord:
            if left_fattr.is_equivalent_frame(right_fattr):
                return np.all(left_fattr == right_fattr)
            else:
                return False
        elif left_is_coord or right_is_coord:
            return False
        return np.all(left_fattr == right_fattr)

    def is_equivalent_frame(self, other):
        if False:
            return 10
        "\n        Checks if this object is the same frame as the ``other`` object.\n\n        To be the same frame, two objects must be the same frame class and have\n        the same frame attributes.  Note that it does *not* matter what, if any,\n        data either object has.\n\n        Parameters\n        ----------\n        other : :class:`~astropy.coordinates.BaseCoordinateFrame`\n            the other frame to check\n\n        Returns\n        -------\n        isequiv : bool\n            True if the frames are the same, False if not.\n\n        Raises\n        ------\n        TypeError\n            If ``other`` isn't a `~astropy.coordinates.BaseCoordinateFrame` or subclass.\n        "
        if self.__class__ == other.__class__:
            for frame_attr_name in self.frame_attributes:
                if not self._frameattr_equiv(getattr(self, frame_attr_name), getattr(other, frame_attr_name)):
                    return False
            return True
        elif not isinstance(other, BaseCoordinateFrame):
            raise TypeError("Tried to do is_equivalent_frame on something that isn't a frame")
        else:
            return False

    def __repr__(self):
        if False:
            while True:
                i = 10
        frameattrs = self._frame_attrs_repr()
        data_repr = self._data_repr()
        if frameattrs:
            frameattrs = f' ({frameattrs})'
        if data_repr:
            return f'<{self.__class__.__name__} Coordinate{frameattrs}: {data_repr}>'
        else:
            return f'<{self.__class__.__name__} Frame{frameattrs}>'

    def _data_repr(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a string representation of the coordinate data.'
        if not self.has_data:
            return ''
        if self.representation_type:
            if hasattr(self.representation_type, '_unit_representation') and isinstance(self.data, self.representation_type._unit_representation):
                rep_cls = self.data.__class__
            else:
                rep_cls = self.representation_type
            if 's' in self.data.differentials:
                dif_cls = self.get_representation_cls('s')
                dif_data = self.data.differentials['s']
                if isinstance(dif_data, (r.UnitSphericalDifferential, r.UnitSphericalCosLatDifferential, r.RadialDifferential)):
                    dif_cls = dif_data.__class__
            else:
                dif_cls = None
            data = self.represent_as(rep_cls, dif_cls, in_frame_units=True)
            data_repr = repr(data)
            (part1, _, remainder) = data_repr.partition('(')
            if remainder != '':
                (comp_str, _, part2) = remainder.partition(')')
                comp_names = comp_str.split(', ')
                invnames = {nmrepr: nmpref for (nmpref, nmrepr) in self.representation_component_names.items()}
                for (i, name) in enumerate(comp_names):
                    comp_names[i] = invnames.get(name, name)
                data_repr = part1 + '(' + ', '.join(comp_names) + ')' + part2
        else:
            data = self.data
            data_repr = repr(self.data)
        if data_repr.startswith('<' + data.__class__.__name__):
            data_repr = data_repr[len(data.__class__.__name__) + 2:-1]
        else:
            data_repr = 'Data:\n' + data_repr
        if 's' in self.data.differentials:
            data_repr_spl = data_repr.split('\n')
            if 'has differentials' in data_repr_spl[-1]:
                diffrepr = repr(data.differentials['s']).split('\n')
                if diffrepr[0].startswith('<'):
                    diffrepr[0] = ' ' + ' '.join(diffrepr[0].split(' ')[1:])
                for (frm_nm, rep_nm) in self.get_representation_component_names('s').items():
                    diffrepr[0] = diffrepr[0].replace(rep_nm, frm_nm)
                if diffrepr[-1].endswith('>'):
                    diffrepr[-1] = diffrepr[-1][:-1]
                data_repr_spl[-1] = '\n'.join(diffrepr)
            data_repr = '\n'.join(data_repr_spl)
        return data_repr

    def _frame_attrs_repr(self):
        if False:
            print('Hello World!')
        "\n        Returns a string representation of the frame's attributes, if any.\n        "
        attr_strs = []
        for attribute_name in self.frame_attributes:
            attr = getattr(self, attribute_name)
            if hasattr(attr, '_astropy_repr_in_frame'):
                attrstr = attr._astropy_repr_in_frame()
            else:
                attrstr = str(attr)
            attr_strs.append(f'{attribute_name}={attrstr}')
        return ', '.join(attr_strs)

    def _apply(self, method, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create a new instance, applying a method to the underlying data.\n\n        In typical usage, the method is any of the shape-changing methods for\n        `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those\n        picking particular elements (``__getitem__``, ``take``, etc.), which\n        are all defined in `~astropy.utils.shapes.ShapedLikeNDArray`. It will be\n        applied to the underlying arrays in the representation (e.g., ``x``,\n        ``y``, and ``z`` for `~astropy.coordinates.CartesianRepresentation`),\n        as well as to any frame attributes that have a shape, with the results\n        used to create a new instance.\n\n        Internally, it is also used to apply functions to the above parts\n        (in particular, `~numpy.broadcast_to`).\n\n        Parameters\n        ----------\n        method : str or callable\n            If str, it is the name of a method that is applied to the internal\n            ``components``. If callable, the function is applied.\n        *args : tuple\n            Any positional arguments for ``method``.\n        **kwargs : dict\n            Any keyword arguments for ``method``.\n        '

        def apply_method(value):
            if False:
                return 10
            if isinstance(value, ShapedLikeNDArray):
                return value._apply(method, *args, **kwargs)
            elif callable(method):
                return method(value, *args, **kwargs)
            else:
                return getattr(value, method)(*args, **kwargs)
        new = super().__new__(self.__class__)
        if hasattr(self, '_representation'):
            new._representation = self._representation.copy()
        new._attr_names_with_defaults = self._attr_names_with_defaults.copy()
        new_shape = ()
        for attr in self.frame_attributes:
            _attr = '_' + attr
            if attr in self._attr_names_with_defaults:
                setattr(new, _attr, getattr(self, _attr))
            else:
                value = getattr(self, _attr)
                if getattr(value, 'shape', ()):
                    value = apply_method(value)
                    new_shape = new_shape or value.shape
                elif method == 'copy' or method == 'flatten':
                    value = copy.copy(value)
                setattr(new, _attr, value)
        if self.has_data:
            new._data = apply_method(self.data)
            new_shape = new_shape or new._data.shape
        else:
            new._data = None
        new._shape = new_shape
        return new

    def __setitem__(self, item, value):
        if False:
            i = 10
            return i + 15
        if self.__class__ is not value.__class__:
            raise TypeError(f'can only set from object of same class: {self.__class__.__name__} vs. {value.__class__.__name__}')
        if not self.is_equivalent_frame(value):
            raise ValueError('can only set frame item from an equivalent frame')
        if value._data is None:
            raise ValueError('can only set frame with value that has data')
        if self._data is None:
            raise ValueError('cannot set frame which has no data')
        if self.shape == ():
            raise TypeError(f"scalar '{self.__class__.__name__}' frame object does not support item assignment")
        if self._data is None:
            raise ValueError('can only set frame if it has data')
        if self._data.__class__ is not value._data.__class__:
            raise TypeError(f'can only set from object of same class: {self._data.__class__.__name__} vs. {value._data.__class__.__name__}')
        if self._data._differentials:
            if self._data._differentials.keys() != value._data._differentials.keys():
                raise ValueError('setitem value must have same differentials')
            for (key, self_diff) in self._data._differentials.items():
                if self_diff.__class__ is not value._data._differentials[key].__class__:
                    raise TypeError(f'can only set from object of same class: {self_diff.__class__.__name__} vs. {value._data._differentials[key].__class__.__name__}')
        self._data[item] = value._data
        self.cache.clear()

    def __dir__(self):
        if False:
            while True:
                i = 10
        '\n        Override the builtin `dir` behavior to include representation\n        names.\n\n        TODO: dynamic representation transforms (i.e. include cylindrical et al.).\n        '
        return sorted(set(super().__dir__()) | set(self.representation_component_names) | set(self.get_representation_component_names('s')))

    def __getattr__(self, attr):
        if False:
            return 10
        '\n        Allow access to attributes on the representation and differential as\n        found via ``self.get_representation_component_names``.\n\n        TODO: We should handle dynamic representation transforms here (e.g.,\n        `.cylindrical`) instead of defining properties as below.\n        '
        if attr.startswith('_'):
            return self.__getattribute__(attr)
        repr_names = self.representation_component_names
        if attr in repr_names:
            if self._data is None:
                self.data
            rep = self.represent_as(self.representation_type, in_frame_units=True)
            val = getattr(rep, repr_names[attr])
            return val
        diff_names = self.get_representation_component_names('s')
        if attr in diff_names:
            if self._data is None:
                self.data
            rep = self.represent_as(in_frame_units=True, **self.get_representation_cls(None))
            val = getattr(rep.differentials['s'], diff_names[attr])
            return val
        return self.__getattribute__(attr)

    def __setattr__(self, attr, value):
        if False:
            return 10
        if not attr.startswith('_'):
            if hasattr(self, 'representation_info'):
                repr_attr_names = set()
                for representation_attr in self.representation_info.values():
                    repr_attr_names.update(representation_attr['names'])
                if attr in repr_attr_names:
                    raise AttributeError(f'Cannot set any frame attribute {attr}')
        super().__setattr__(attr, value)

    def __eq__(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Equality operator for frame.\n\n        This implements strict equality and requires that the frames are\n        equivalent and that the representation data are exactly equal.\n        '
        if not isinstance(value, BaseCoordinateFrame):
            return NotImplemented
        is_equiv = self.is_equivalent_frame(value)
        if self._data is None and value._data is None:
            return is_equiv
        if not is_equiv:
            raise TypeError(f'cannot compare: objects must have equivalent frames: {self.replicate_without_data()} vs. {value.replicate_without_data()}')
        if (value._data is None) != (self._data is None):
            raise ValueError('cannot compare: one frame has data and the other does not')
        return self._data == value._data

    def __ne__(self, value):
        if False:
            print('Hello World!')
        return np.logical_not(self == value)

    def separation(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes on-sky separation between this coordinate and another.\n\n        .. note::\n\n            If the ``other`` coordinate object is in a different frame, it is\n            first transformed to the frame of this object. This can lead to\n            unintuitive behavior if not accounted for. Particularly of note is\n            that ``self.separation(other)`` and ``other.separation(self)`` may\n            not give the same answer in this case.\n\n        Parameters\n        ----------\n        other : `~astropy.coordinates.BaseCoordinateFrame`\n            The coordinate to get the separation to.\n\n        Returns\n        -------\n        sep : `~astropy.coordinates.Angle`\n            The on-sky separation between this and the ``other`` coordinate.\n\n        Notes\n        -----\n        The separation is calculated using the Vincenty formula, which\n        is stable at all locations, including poles and antipodes [1]_.\n\n        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance\n\n        '
        from .angles import Angle, angular_separation
        self_unit_sph = self.represent_as(r.UnitSphericalRepresentation)
        other_transformed = other.transform_to(self)
        other_unit_sph = other_transformed.represent_as(r.UnitSphericalRepresentation)
        sep = angular_separation(self_unit_sph.lon, self_unit_sph.lat, other_unit_sph.lon, other_unit_sph.lat)
        return Angle(sep, unit=u.degree)

    def separation_3d(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Computes three dimensional separation between this coordinate\n        and another.\n\n        Parameters\n        ----------\n        other : `~astropy.coordinates.BaseCoordinateFrame`\n            The coordinate system to get the distance to.\n\n        Returns\n        -------\n        sep : `~astropy.coordinates.Distance`\n            The real-space distance between these two coordinates.\n\n        Raises\n        ------\n        ValueError\n            If this or the other coordinate do not have distances.\n        '
        from .distances import Distance
        if issubclass(self.data.__class__, r.UnitSphericalRepresentation):
            raise ValueError('This object does not have a distance; cannot compute 3d separation.')
        other_in_self_system = other.transform_to(self)
        if issubclass(other_in_self_system.__class__, r.UnitSphericalRepresentation):
            raise ValueError('The other object does not have a distance; cannot compute 3d separation.')
        self_car = self.data.without_differentials().represent_as(r.CartesianRepresentation)
        other_car = other_in_self_system.data.without_differentials().represent_as(r.CartesianRepresentation)
        dist = (self_car - other_car).norm()
        if dist.unit == u.one:
            return dist
        else:
            return Distance(dist)

    @property
    def cartesian(self):
        if False:
            return 10
        '\n        Shorthand for a cartesian representation of the coordinates in this\n        object.\n        '
        return self.represent_as('cartesian', in_frame_units=True)

    @property
    def cylindrical(self):
        if False:
            while True:
                i = 10
        '\n        Shorthand for a cylindrical representation of the coordinates in this\n        object.\n        '
        return self.represent_as('cylindrical', in_frame_units=True)

    @property
    def spherical(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Shorthand for a spherical representation of the coordinates in this\n        object.\n        '
        return self.represent_as('spherical', in_frame_units=True)

    @property
    def sphericalcoslat(self):
        if False:
            return 10
        '\n        Shorthand for a spherical representation of the positional data and a\n        `~astropy.coordinates.SphericalCosLatDifferential` for the velocity\n        data in this object.\n        '
        return self.represent_as('spherical', 'sphericalcoslat', in_frame_units=True)

    @property
    def velocity(self):
        if False:
            i = 10
            return i + 15
        "\n        Shorthand for retrieving the Cartesian space-motion as a\n        `~astropy.coordinates.CartesianDifferential` object.\n\n        This is equivalent to calling ``self.cartesian.differentials['s']``.\n        "
        if 's' not in self.data.differentials:
            raise ValueError('Frame has no associated velocity (Differential) data information.')
        return self.cartesian.differentials['s']

    @property
    def proper_motion(self):
        if False:
            return 10
        '\n        Shorthand for the two-dimensional proper motion as a\n        `~astropy.units.Quantity` object with angular velocity units. In the\n        returned `~astropy.units.Quantity`, ``axis=0`` is the longitude/latitude\n        dimension so that ``.proper_motion[0]`` is the longitudinal proper\n        motion and ``.proper_motion[1]`` is latitudinal. The longitudinal proper\n        motion already includes the cos(latitude) term.\n        '
        if 's' not in self.data.differentials:
            raise ValueError('Frame has no associated velocity (Differential) data information.')
        sph = self.represent_as('spherical', 'sphericalcoslat', in_frame_units=True)
        pm_lon = sph.differentials['s'].d_lon_coslat
        pm_lat = sph.differentials['s'].d_lat
        return np.stack((pm_lon.value, pm_lat.to(pm_lon.unit).value), axis=0) * pm_lon.unit

    @property
    def radial_velocity(self):
        if False:
            return 10
        '\n        Shorthand for the radial or line-of-sight velocity as a\n        `~astropy.units.Quantity` object.\n        '
        if 's' not in self.data.differentials:
            raise ValueError('Frame has no associated velocity (Differential) data information.')
        sph = self.represent_as('spherical', in_frame_units=True)
        return sph.differentials['s'].d_distance

class GenericFrame(BaseCoordinateFrame):
    """
    A frame object that can't store data but can hold any arbitrary frame
    attributes. Mostly useful as a utility for the high-level class to store
    intermediate frame attributes.

    Parameters
    ----------
    frame_attrs : dict
        A dictionary of attributes to be used as the frame attributes for this
        frame.
    """
    name = None

    def __init__(self, frame_attrs):
        if False:
            while True:
                i = 10
        self.frame_attributes = {}
        for (name, default) in frame_attrs.items():
            self.frame_attributes[name] = Attribute(default)
            setattr(self, '_' + name, default)
        super().__init__(None)

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if '_' + name in self.__dict__:
            return getattr(self, '_' + name)
        else:
            raise AttributeError(f'no {name}')

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        if name in self.frame_attributes:
            raise AttributeError(f"can't set frame attribute '{name}'")
        else:
            super().__setattr__(name, value)