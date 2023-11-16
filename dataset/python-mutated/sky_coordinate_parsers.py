import re
from collections.abc import Sequence
import numpy as np
from astropy import units as u
from astropy.units import IrreducibleUnit, Unit
from .baseframe import BaseCoordinateFrame, _get_diff_cls, _get_repr_cls, frame_transform_graph
from .representation import BaseRepresentation, SphericalRepresentation, UnitSphericalRepresentation
'\nThis module contains utility functions to make the SkyCoord initializer more modular\nand maintainable. No functionality here should be in the public API, but rather used as\npart of creating SkyCoord objects.\n'
PLUS_MINUS_RE = re.compile('(\\+|\\-)')
J_PREFIXED_RA_DEC_RE = re.compile('J                              # J prefix\n    ([0-9]{6,7}\\.?[0-9]{0,2})          # RA as HHMMSS.ss or DDDMMSS.ss, optional decimal digits\n    ([\\+\\-][0-9]{6}\\.?[0-9]{0,2})\\s*$  # Dec as DDMMSS.ss, optional decimal digits\n    ', re.VERBOSE)

def _get_frame_class(frame):
    if False:
        print('Hello World!')
    '\n    Get a frame class from the input `frame`, which could be a frame name\n    string, or frame class.\n    '
    if isinstance(frame, str):
        frame_names = frame_transform_graph.get_names()
        if frame not in frame_names:
            raise ValueError(f'Coordinate frame name "{frame}" is not a known coordinate frame ({sorted(frame_names)})')
        frame_cls = frame_transform_graph.lookup_name(frame)
    elif isinstance(frame, type) and issubclass(frame, BaseCoordinateFrame):
        frame_cls = frame
    else:
        raise ValueError(f"Coordinate frame must be a frame name or frame class, not a '{frame.__class__.__name__}'")
    return frame_cls
_conflict_err_msg = "Coordinate attribute '{0}'={1!r} conflicts with keyword argument '{0}'={2!r}. This usually means an attribute was set on one of the input objects and also in the keyword arguments to {3}"

def _get_frame_without_data(args, kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Determines the coordinate frame from input SkyCoord args and kwargs.\n\n    This function extracts (removes) all frame attributes from the kwargs and\n    determines the frame class either using the kwargs, or using the first\n    element in the args (if a single frame object is passed in, for example).\n    This function allows a frame to be specified as a string like 'icrs' or a\n    frame class like ICRS, or an instance ICRS(), as long as the instance frame\n    attributes don't conflict with kwargs passed in (which could require a\n    three-way merge with the coordinate data possibly specified via the args).\n    "
    from .sky_coordinate import SkyCoord
    frame_cls = None
    frame_cls_kwargs = {}
    frame = kwargs.pop('frame', None)
    if frame is not None:
        if isinstance(frame, SkyCoord):
            for attr in frame._extra_frameattr_names:
                if attr in kwargs and np.any(getattr(frame, attr) != kwargs[attr]):
                    raise ValueError(_conflict_err_msg.format(attr, getattr(frame, attr), kwargs[attr], 'SkyCoord'))
                else:
                    kwargs[attr] = getattr(frame, attr)
            frame = frame.frame
        if isinstance(frame, BaseCoordinateFrame):
            for attr in frame.frame_attributes:
                if attr in kwargs:
                    raise ValueError(f"Cannot specify frame attribute '{attr}' directly as an argument to SkyCoord because a frame instance was passed in. Either pass a frame class, or modify the frame attributes of the input frame instance.")
                elif not frame.is_frame_attr_default(attr):
                    kwargs[attr] = getattr(frame, attr)
            frame_cls = frame.__class__
            kwargs.setdefault('representation_type', frame.representation_type)
            kwargs.setdefault('differential_type', frame.differential_type)
        if frame_cls is None:
            frame_cls = _get_frame_class(frame)
    for arg in args:
        if isinstance(arg, (Sequence, np.ndarray)) and len(args) == 1 and (len(arg) > 0):
            arg = arg[0]
        coord_frame_obj = coord_frame_cls = None
        if isinstance(arg, BaseCoordinateFrame):
            coord_frame_obj = arg
        elif isinstance(arg, SkyCoord):
            coord_frame_obj = arg.frame
        if coord_frame_obj is not None:
            coord_frame_cls = coord_frame_obj.__class__
            frame_diff = coord_frame_obj.get_representation_cls('s')
            if frame_diff is not None:
                kwargs.setdefault('differential_type', frame_diff)
            for attr in coord_frame_obj.frame_attributes:
                if attr in kwargs and (not coord_frame_obj.is_frame_attr_default(attr)) and np.any(kwargs[attr] != getattr(coord_frame_obj, attr)):
                    raise ValueError(f"Frame attribute '{attr}' has conflicting values between the input coordinate data and either keyword arguments or the frame specification (frame=...): {getattr(coord_frame_obj, attr)} =/= {kwargs[attr]}")
                elif attr not in kwargs and (not coord_frame_obj.is_frame_attr_default(attr)):
                    kwargs[attr] = getattr(coord_frame_obj, attr)
        if coord_frame_cls is not None:
            if frame_cls is None:
                frame_cls = coord_frame_cls
            elif frame_cls is not coord_frame_cls:
                raise ValueError(f"Cannot override frame='{coord_frame_cls.__name__}' of input coordinate with new frame='{frame_cls.__name__}'. Instead, transform the coordinate.")
    if frame_cls is None:
        from .builtin_frames import ICRS
        frame_cls = ICRS
    if not issubclass(frame_cls, BaseCoordinateFrame):
        raise ValueError(f'Frame class has unexpected type: {frame_cls.__name__}')
    for attr in frame_cls.frame_attributes:
        if attr in kwargs:
            frame_cls_kwargs[attr] = kwargs.pop(attr)
    if 'representation_type' in kwargs:
        frame_cls_kwargs['representation_type'] = _get_repr_cls(kwargs.pop('representation_type'))
    differential_type = kwargs.pop('differential_type', None)
    if differential_type is not None:
        frame_cls_kwargs['differential_type'] = _get_diff_cls(differential_type)
    return (frame_cls, frame_cls_kwargs)

def _parse_coordinate_data(frame, args, kwargs):
    if False:
        print('Hello World!')
    '\n    Extract coordinate data from the args and kwargs passed to SkyCoord.\n\n    By this point, we assume that all of the frame attributes have been\n    extracted from kwargs (see _get_frame_without_data()), so all that are left\n    are (1) extra SkyCoord attributes, and (2) the coordinate data, specified in\n    any of the valid ways.\n    '
    valid_skycoord_kwargs = {}
    valid_components = {}
    info = None
    attr_names = list(kwargs.keys())
    for attr in attr_names:
        if attr in frame_transform_graph.frame_attributes:
            valid_skycoord_kwargs[attr] = kwargs.pop(attr)
    units = _get_representation_component_units(args, kwargs)
    valid_components.update(_get_representation_attrs(frame, units, kwargs))
    if kwargs:
        pm_message = ''
        if frame.representation_type == SphericalRepresentation:
            frame_names = list(frame.get_representation_component_names().keys())
            lon_name = frame_names[0]
            lat_name = frame_names[1]
            if f'pm_{lon_name}' in list(kwargs.keys()):
                pm_message = f'\n\n By default, most frame classes expect the longitudinal proper motion to include the cos(latitude) term, named `pm_{lon_name}_cos{lat_name}`. Did you mean to pass in this component?'
        raise ValueError('Unrecognized keyword argument(s) {}{}'.format(', '.join((f"'{key}'" for key in kwargs)), pm_message))
    if args:
        if len(args) == 1:
            (_skycoord_kwargs, _components) = _parse_coordinate_arg(args[0], frame, units, kwargs)
            if 'info' in getattr(args[0], '__dict__', ()):
                info = args[0].info
        elif len(args) <= 3:
            _skycoord_kwargs = {}
            _components = {}
            frame_attr_names = frame.representation_component_names.keys()
            repr_attr_names = frame.representation_component_names.values()
            for (arg, frame_attr_name, repr_attr_name, unit) in zip(args, frame_attr_names, repr_attr_names, units):
                attr_class = frame.representation_type.attr_classes[repr_attr_name]
                _components[frame_attr_name] = attr_class(arg, unit=unit)
        else:
            raise ValueError(f'Must supply no more than three positional arguments, got {len(args)}')
        for (attr, coord_value) in _components.items():
            if attr in valid_components:
                raise ValueError(_conflict_err_msg.format(attr, coord_value, valid_components[attr], 'SkyCoord'))
            valid_components[attr] = coord_value
        for (attr, value) in _skycoord_kwargs.items():
            if attr in valid_skycoord_kwargs and np.any(valid_skycoord_kwargs[attr] != value):
                raise ValueError(_conflict_err_msg.format(attr, value, valid_skycoord_kwargs[attr], 'SkyCoord'))
            valid_skycoord_kwargs[attr] = value
    return (valid_skycoord_kwargs, valid_components, info)

def _get_representation_component_units(args, kwargs):
    if False:
        while True:
            i = 10
    '\n    Get the unit from kwargs for the *representation* components (not the\n    differentials).\n    '
    if 'unit' not in kwargs:
        units = [None, None, None]
    else:
        units = kwargs.pop('unit')
        if isinstance(units, str):
            units = [x.strip() for x in units.split(',')]
            if len(units) == 1:
                units = [units[0], units[0], units[0]]
        elif isinstance(units, (Unit, IrreducibleUnit)):
            units = [units, units, units]
        try:
            units = [Unit(x) if x else None for x in units]
            units.extend((None for x in range(3 - len(units))))
            if len(units) > 3:
                raise ValueError()
        except Exception as err:
            raise ValueError('Unit keyword must have one to three unit values as tuple or comma-separated string.') from err
    return units

def _parse_coordinate_arg(coords, frame, units, init_kwargs):
    if False:
        print('Hello World!')
    '\n    Single unnamed arg supplied.  This must be:\n    - Coordinate frame with data\n    - Representation\n    - SkyCoord\n    - List or tuple of:\n      - String which splits into two values\n      - Iterable with two values\n      - SkyCoord, frame, or representation objects.\n\n    Returns a dict mapping coordinate attribute names to values (or lists of\n    values)\n    '
    from .sky_coordinate import SkyCoord
    is_scalar = False
    components = {}
    skycoord_kwargs = {}
    frame_attr_names = list(frame.representation_component_names.keys())
    repr_attr_names = list(frame.representation_component_names.values())
    repr_attr_classes = list(frame.representation_type.attr_classes.values())
    n_attr_names = len(repr_attr_names)
    if isinstance(coords, str):
        is_scalar = True
        coords = [coords]
    if isinstance(coords, (SkyCoord, BaseCoordinateFrame)):
        if not coords.has_data:
            raise ValueError('Cannot initialize from a frame without coordinate data')
        data = coords.data.represent_as(frame.representation_type)
        values = []
        repr_attr_name_to_drop = []
        for repr_attr_name in repr_attr_names:
            if isinstance(coords.data, UnitSphericalRepresentation) and repr_attr_name == 'distance':
                repr_attr_name_to_drop.append(repr_attr_name)
                continue
            values.append(getattr(data, repr_attr_name))
        for nametodrop in repr_attr_name_to_drop:
            nameidx = repr_attr_names.index(nametodrop)
            del repr_attr_names[nameidx]
            del units[nameidx]
            del frame_attr_names[nameidx]
            del repr_attr_classes[nameidx]
        if coords.data.differentials and 's' in coords.data.differentials:
            orig_vel = coords.data.differentials['s']
            vel = coords.data.represent_as(frame.representation_type, frame.get_representation_cls('s')).differentials['s']
            for (frname, reprname) in frame.get_representation_component_names('s').items():
                if reprname == 'd_distance' and (not hasattr(orig_vel, reprname)) and ('unit' in orig_vel.get_name()):
                    continue
                values.append(getattr(vel, reprname))
                units.append(None)
                frame_attr_names.append(frname)
                repr_attr_names.append(reprname)
                repr_attr_classes.append(vel.attr_classes[reprname])
        for attr in frame_transform_graph.frame_attributes:
            value = getattr(coords, attr, None)
            use_value = isinstance(coords, SkyCoord) or attr not in coords.frame_attributes
            if use_value and value is not None:
                skycoord_kwargs[attr] = value
    elif isinstance(coords, BaseRepresentation):
        if coords.differentials and 's' in coords.differentials:
            diffs = frame.get_representation_cls('s')
            data = coords.represent_as(frame.representation_type, diffs)
            values = [getattr(data, repr_attr_name) for repr_attr_name in repr_attr_names]
            for (frname, reprname) in frame.get_representation_component_names('s').items():
                values.append(getattr(data.differentials['s'], reprname))
                units.append(None)
                frame_attr_names.append(frname)
                repr_attr_names.append(reprname)
                repr_attr_classes.append(data.differentials['s'].attr_classes[reprname])
        else:
            data = coords.represent_as(frame.representation_type)
            values = [getattr(data, repr_attr_name) for repr_attr_name in repr_attr_names]
    elif isinstance(coords, np.ndarray) and coords.dtype.kind in 'if' and (coords.ndim == 2) and (coords.shape[1] <= 3):
        values = coords.transpose()
    elif isinstance(coords, (Sequence, np.ndarray)):
        vals = []
        is_ra_dec_representation = 'ra' in frame.representation_component_names and 'dec' in frame.representation_component_names
        coord_types = (SkyCoord, BaseCoordinateFrame, BaseRepresentation)
        if any((isinstance(coord, coord_types) for coord in coords)):
            scs = [SkyCoord(coord, **init_kwargs) for coord in coords]
            for sc in scs[1:]:
                if not sc.is_equivalent_frame(scs[0]):
                    raise ValueError(f"List of inputs don't have equivalent frames: {sc} != {scs[0]}")
            not_unit_sphere = not isinstance(scs[0].data, UnitSphericalRepresentation)
            for fattrnm in scs[0].frame.frame_attributes:
                skycoord_kwargs[fattrnm] = getattr(scs[0].frame, fattrnm)
            for fattrnm in scs[0]._extra_frameattr_names:
                skycoord_kwargs[fattrnm] = getattr(scs[0], fattrnm)
            values = [np.concatenate([np.atleast_1d(getattr(sc, data_attr)) for sc in scs]) for (data_attr, repr_attr) in zip(frame_attr_names, repr_attr_names) if not_unit_sphere or repr_attr != 'distance']
        else:
            for coord in coords:
                if isinstance(coord, str):
                    coord1 = coord.split()
                    if len(coord1) == 6:
                        coord = (' '.join(coord1[:3]), ' '.join(coord1[3:]))
                    elif is_ra_dec_representation:
                        coord = _parse_ra_dec(coord)
                    else:
                        coord = coord1
                vals.append(coord)
            try:
                n_coords = sorted({len(x) for x in vals})
            except Exception as err:
                raise ValueError('One or more elements of input sequence does not have a length.') from err
            if len(n_coords) > 1:
                raise ValueError(f'Input coordinate values must have same number of elements, found {n_coords}')
            n_coords = n_coords[0]
            if n_coords > n_attr_names:
                raise ValueError(f'Input coordinates have {n_coords} values but representation {frame.representation_type.get_name()} only accepts {n_attr_names}')
            values = [list(x) for x in zip(*vals)]
            if is_scalar:
                values = [x[0] for x in values]
    else:
        raise ValueError('Cannot parse coordinates from first argument')
    try:
        for (frame_attr_name, repr_attr_class, value, unit) in zip(frame_attr_names, repr_attr_classes, values, units):
            components[frame_attr_name] = repr_attr_class(value, unit=unit, copy=False)
    except Exception as err:
        raise ValueError(f'Cannot parse first argument data "{value}" for attribute {frame_attr_name}') from err
    return (skycoord_kwargs, components)

def _get_representation_attrs(frame, units, kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find instances of the "representation attributes" for specifying data\n    for this frame.  Pop them off of kwargs, run through the appropriate class\n    constructor (to validate and apply unit), and put into the output\n    valid_kwargs.  "Representation attributes" are the frame-specific aliases\n    for the underlying data values in the representation, e.g. "ra" for "lon"\n    for many equatorial spherical representations, or "w" for "x" in the\n    cartesian representation of Galactic.\n\n    This also gets any *differential* kwargs, because they go into the same\n    frame initializer later on.\n    '
    frame_attr_names = frame.representation_component_names.keys()
    repr_attr_classes = frame.representation_type.attr_classes.values()
    valid_kwargs = {}
    for (frame_attr_name, repr_attr_class, unit) in zip(frame_attr_names, repr_attr_classes, units):
        value = kwargs.pop(frame_attr_name, None)
        if value is not None:
            try:
                valid_kwargs[frame_attr_name] = repr_attr_class(value, unit=unit)
            except u.UnitConversionError as err:
                error_message = f"Unit '{unit}' ({unit.physical_type}) could not be applied to '{frame_attr_name}'. This can occur when passing units for some coordinate components when other components are specified as Quantity objects. Either pass a list of units for all components (and unit-less coordinate data), or pass Quantities for all components."
                raise u.UnitConversionError(error_message) from err
    differential_type = frame.differential_type
    if differential_type is not None:
        for (frame_name, repr_name) in frame.get_representation_component_names('s').items():
            diff_attr_class = differential_type.attr_classes[repr_name]
            value = kwargs.pop(frame_name, None)
            if value is not None:
                valid_kwargs[frame_name] = diff_attr_class(value)
    return valid_kwargs

def _parse_ra_dec(coord_str):
    if False:
        while True:
            i = 10
    'Parse RA and Dec values from a coordinate string.\n\n    Currently the following formats are supported:\n\n     * space separated 6-value format\n     * space separated <6-value format, this requires a plus or minus sign\n       separation between RA and Dec\n     * sign separated format\n     * JHHMMSS.ss+DDMMSS.ss format, with up to two optional decimal digits\n     * JDDDMMSS.ss+DDMMSS.ss format, with up to two optional decimal digits\n\n    Parameters\n    ----------\n    coord_str : str\n        Coordinate string to parse.\n\n    Returns\n    -------\n    coord : str or list of str\n        Parsed coordinate values.\n    '
    if isinstance(coord_str, str):
        coord1 = coord_str.split()
    else:
        raise TypeError('coord_str must be a single str')
    if len(coord1) == 6:
        coord = (' '.join(coord1[:3]), ' '.join(coord1[3:]))
    elif len(coord1) > 2:
        coord = PLUS_MINUS_RE.split(coord_str)
        coord = (coord[0], ' '.join(coord[1:]))
    elif len(coord1) == 1:
        match_j = J_PREFIXED_RA_DEC_RE.match(coord_str)
        if match_j:
            coord = match_j.groups()
            if len(coord[0].split('.')[0]) == 7:
                coord = (f'{coord[0][0:3]} {coord[0][3:5]} {coord[0][5:]}', f'{coord[1][0:3]} {coord[1][3:5]} {coord[1][5:]}')
            else:
                coord = (f'{coord[0][0:2]} {coord[0][2:4]} {coord[0][4:]}', f'{coord[1][0:3]} {coord[1][3:5]} {coord[1][5:]}')
        else:
            coord = PLUS_MINUS_RE.split(coord_str)
            coord = (coord[0], ' '.join(coord[1:]))
    else:
        coord = coord1
    return coord