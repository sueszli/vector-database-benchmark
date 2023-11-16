from __future__ import annotations
import copy
from collections.abc import MappingView
from types import MappingProxyType
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import representation as r
from astropy.coordinates.attributes import CoordinateAttribute, DifferentialAttribute, QuantityAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, base_doc, frame_transform_graph
from astropy.coordinates.errors import ConvertError
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.coordinates.transformations import AffineTransform
from astropy.utils.decorators import classproperty, format_doc
from astropy.utils.state import ScienceState
from .icrs import ICRS
__all__ = ['Galactocentric']
_ROLL0 = Angle(58.5986320306 * u.degree)

class _StateProxy(MappingView):
    """
    `~collections.abc.MappingView` with a read-only ``getitem`` through
    `~types.MappingProxyType`.

    """

    def __init__(self, mapping):
        if False:
            i = 10
            return i + 15
        super().__init__(mapping)
        self._mappingproxy = MappingProxyType(self._mapping)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        'Read-only ``getitem``.'
        return self._mappingproxy[key]

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        return copy.deepcopy(self._mapping, memo=memo)

class galactocentric_frame_defaults(ScienceState):
    """Global setting of default values for the frame attributes in the `~astropy.coordinates.Galactocentric` frame.

    These constancts may be updated in future versions of ``astropy``. Note
    that when using `~astropy.coordinates.Galactocentric`, changing values
    here will not affect any attributes that are set explicitly by passing
    values in to the `~astropy.coordinates.Galactocentric`
    initializer. Modifying these defaults will only affect the frame attribute
    values when using the frame as, e.g., ``Galactocentric`` or
    ``Galactocentric()`` with no explicit arguments.

    This class controls the parameter settings by specifying a string name,
    with the following pre-specified options:

    - 'pre-v4.0': The current default value, which sets the default frame
      attribute values to their original (pre-astropy-v4.0) values.
    - 'v4.0': The attribute values as updated in Astropy version 4.0.
    - 'latest': An alias of the most recent parameter set (currently: 'v4.0')

    Alternatively, user-defined parameter settings may be registered, with
    :meth:`~astropy.coordinates.galactocentric_frame_defaults.register`,
    and used identically as pre-specified parameter sets. At minimum,
    registrations must have unique names and a dictionary of parameters
    with keys "galcen_coord", "galcen_distance", "galcen_v_sun", "z_sun",
    "roll". See examples below.

    This class also tracks the references for all parameter values in the
    attribute ``references``, as well as any further information the registry.
    The pre-specified options can be extended to include similar
    state information as user-defined parameter settings -- for example, to add
    parameter uncertainties.

    The preferred method for getting a parameter set and metadata, by name, is
    :meth:`~astropy.coordinates.galactocentric_frame_defaults.get_from_registry`
    since it ensures the immutability of the registry.

    See :ref:`astropy:astropy-coordinates-galactocentric-defaults` for more
    information.

    Examples
    --------
    The default `~astropy.coordinates.Galactocentric` frame parameters can be
    modified globally::

        >>> from astropy.coordinates import galactocentric_frame_defaults
        >>> _ = galactocentric_frame_defaults.set('v4.0') # doctest: +SKIP
        >>> Galactocentric() # doctest: +SKIP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>
        >>> _ = galactocentric_frame_defaults.set('pre-v4.0') # doctest: +SKIP
        >>> Galactocentric() # doctest: +SKIP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.3 kpc, galcen_v_sun=(11.1, 232.24, 7.25) km / s, z_sun=27.0 pc, roll=0.0 deg)>

    The default parameters can also be updated by using this class as a context
    manager::

        >>> with galactocentric_frame_defaults.set('pre-v4.0'):
        ...     print(Galactocentric()) # doctest: +FLOAT_CMP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.3 kpc, galcen_v_sun=(11.1, 232.24, 7.25) km / s, z_sun=27.0 pc, roll=0.0 deg)>

    Again, changing the default parameter values will not affect frame
    attributes that are explicitly specified::

        >>> import astropy.units as u
        >>> with galactocentric_frame_defaults.set('pre-v4.0'):
        ...     print(Galactocentric(galcen_distance=8.0*u.kpc)) # doctest: +FLOAT_CMP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.0 kpc, galcen_v_sun=(11.1, 232.24, 7.25) km / s, z_sun=27.0 pc, roll=0.0 deg)>

    Additional parameter sets may be registered, for instance to use the
    Dehnen & Binney (1998) measurements of the solar motion. We can also
    add metadata, such as the 1-sigma errors. In this example we will modify
    the required key "parameters", change the recommended key "references" to
    match "parameters", and add the extra key "error" (any key can be added)::

        >>> state = galactocentric_frame_defaults.get_from_registry("v4.0")
        >>> state["parameters"]["galcen_v_sun"] = (10.00, 225.25, 7.17) * (u.km / u.s)
        >>> state["references"]["galcen_v_sun"] = "https://ui.adsabs.harvard.edu/full/1998MNRAS.298..387D"
        >>> state["error"] = {"galcen_v_sun": (0.36, 0.62, 0.38) * (u.km / u.s)}
        >>> galactocentric_frame_defaults.register(name="DB1998", **state)

    Just as in the previous examples, the new parameter set can be retrieved with::

        >>> state = galactocentric_frame_defaults.get_from_registry("DB1998")
        >>> print(state["error"]["galcen_v_sun"])  # doctest: +FLOAT_CMP
        [0.36 0.62 0.38] km / s

    """
    _latest_value = 'v4.0'
    _value = None
    _references = None
    _state = dict()
    _registry = {'v4.0': {'parameters': _StateProxy({'galcen_coord': ICRS(ra=266.4051 * u.degree, dec=-28.936175 * u.degree), 'galcen_distance': 8.122 * u.kpc, 'galcen_v_sun': r.CartesianDifferential([12.9, 245.6, 7.78] * (u.km / u.s)), 'z_sun': 20.8 * u.pc, 'roll': 0 * u.deg}), 'references': _StateProxy({'galcen_coord': 'https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R', 'galcen_distance': 'https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G', 'galcen_v_sun': ['https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..210D', 'https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G', 'https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R'], 'z_sun': 'https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1417B', 'roll': None})}, 'pre-v4.0': {'parameters': _StateProxy({'galcen_coord': ICRS(ra=266.4051 * u.degree, dec=-28.936175 * u.degree), 'galcen_distance': 8.3 * u.kpc, 'galcen_v_sun': r.CartesianDifferential([11.1, 220 + 12.24, 7.25] * (u.km / u.s)), 'z_sun': 27.0 * u.pc, 'roll': 0 * u.deg}), 'references': _StateProxy({'galcen_coord': 'https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R', 'galcen_distance': 'https://ui.adsabs.harvard.edu/#abs/2009ApJ...692.1075G', 'galcen_v_sun': ['https://ui.adsabs.harvard.edu/#abs/2010MNRAS.403.1829S', 'https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B'], 'z_sun': 'https://ui.adsabs.harvard.edu/#abs/2001ApJ...553..184C', 'roll': None})}}

    @classproperty
    def parameters(cls):
        if False:
            i = 10
            return i + 15
        return cls._value

    @classproperty
    def references(cls):
        if False:
            while True:
                i = 10
        return cls._references

    @classmethod
    def get_from_registry(cls, name: str) -> dict[str, dict]:
        if False:
            return 10
        '\n        Return Galactocentric solar parameters and metadata given string names\n        for the parameter sets. This method ensures the returned state is a\n        mutable copy, so any changes made do not affect the registry state.\n\n        Returns\n        -------\n        state : dict\n            Copy of the registry for the string name.\n            Should contain, at minimum:\n\n            - "parameters": dict\n                Galactocentric solar parameters\n            - "references" : Dict[str, Union[str, Sequence[str]]]\n                References for "parameters".\n                Fields are str or sequence of str.\n\n        Raises\n        ------\n        KeyError\n            If invalid string input to registry\n            to retrieve solar parameters for Galactocentric frame.\n\n        '
        if name == 'latest':
            name = cls._latest_value
        state = copy.deepcopy(cls._registry[name])
        return state

    @classmethod
    def validate(cls, value):
        if False:
            return 10
        if value is None:
            value = cls._latest_value
        if isinstance(value, str):
            state = cls.get_from_registry(value)
            cls._references = state['references']
            cls._state = state
            parameters = state['parameters']
        elif isinstance(value, dict):
            parameters = value
        elif isinstance(value, Galactocentric):
            parameters = dict()
            for k in value.frame_attributes:
                parameters[k] = getattr(value, k)
            cls._references = value.frame_attribute_references.copy()
            cls._state = dict(parameters=parameters, references=cls._references)
        else:
            raise ValueError('Invalid input to retrieve solar parameters for Galactocentric frame: input must be a string, dict, or Galactocentric instance')
        return parameters

    @classmethod
    def register(cls, name: str, parameters: dict, references=None, **meta: dict) -> None:
        if False:
            i = 10
            return i + 15
        'Register a set of parameters.\n\n        Parameters\n        ----------\n        name : str\n            The registration name for the parameter and metadata set.\n        parameters : dict\n            The solar parameters for Galactocentric frame.\n        references : dict or None, optional\n            References for contents of `parameters`.\n            None becomes empty dict.\n        **meta : dict, optional\n            Any other properties to register.\n\n        '
        must_have = {'galcen_coord', 'galcen_distance', 'galcen_v_sun', 'z_sun', 'roll'}
        missing = must_have.difference(parameters)
        if missing:
            raise ValueError(f'Missing parameters: {missing}')
        references = references or {}
        state = dict(parameters=parameters, references=references)
        state.update(meta)
        cls._registry[name] = state
doc_components = '\n    x : `~astropy.units.Quantity`, optional\n        Cartesian, Galactocentric :math:`x` position component.\n    y : `~astropy.units.Quantity`, optional\n        Cartesian, Galactocentric :math:`y` position component.\n    z : `~astropy.units.Quantity`, optional\n        Cartesian, Galactocentric :math:`z` position component.\n\n    v_x : `~astropy.units.Quantity`, optional\n        Cartesian, Galactocentric :math:`v_x` velocity component.\n    v_y : `~astropy.units.Quantity`, optional\n        Cartesian, Galactocentric :math:`v_y` velocity component.\n    v_z : `~astropy.units.Quantity`, optional\n        Cartesian, Galactocentric :math:`v_z` velocity component.\n'
doc_footer = "\n    Other parameters\n    ----------------\n    galcen_coord : `~astropy.coordinates.ICRS`, optional, keyword-only\n        The ICRS coordinates of the Galactic center.\n    galcen_distance : `~astropy.units.Quantity`, optional, keyword-only\n        The distance from the sun to the Galactic center.\n    galcen_v_sun : `~astropy.coordinates.CartesianDifferential`, `~astropy.units.Quantity` ['speed'], optional, keyword-only\n        The velocity of the sun *in the Galactocentric frame* as Cartesian\n        velocity components.\n    z_sun : `~astropy.units.Quantity` ['length'], optional, keyword-only\n        The distance from the sun to the Galactic midplane.\n    roll : `~astropy.coordinates.Angle`, optional, keyword-only\n        The angle to rotate about the final x-axis, relative to the\n        orientation for Galactic. For example, if this roll angle is 0,\n        the final x-z plane will align with the Galactic coordinates x-z\n        plane. Unless you really know what this means, you probably should\n        not change this!\n\n    Examples\n    --------\n\n    To transform to the Galactocentric frame with the default\n    frame attributes, pass the uninstantiated class name to the\n    ``transform_to()`` method of a `~astropy.coordinates.SkyCoord` object::\n\n        >>> import astropy.units as u\n        >>> import astropy.coordinates as coord\n        >>> c = coord.SkyCoord(ra=[158.3122, 24.5] * u.degree,\n        ...                    dec=[-17.3, 81.52] * u.degree,\n        ...                    distance=[11.5, 24.12] * u.kpc,\n        ...                    frame='icrs')\n        >>> c.transform_to(coord.Galactocentric) # doctest: +FLOAT_CMP\n        <SkyCoord (Galactocentric: galcen_coord=<ICRS Coordinate: (ra, dec) in deg\n            (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg): (x, y, z) in kpc\n            [( -9.43489286, -9.40062188, 6.51345359),\n             (-21.11044918, 18.76334013, 7.83175149)]>\n\n\n    To specify a custom set of parameters, you have to include extra keyword\n    arguments when initializing the Galactocentric frame object::\n\n        >>> c.transform_to(coord.Galactocentric(galcen_distance=8.1*u.kpc)) # doctest: +FLOAT_CMP\n        <SkyCoord (Galactocentric: galcen_coord=<ICRS Coordinate: (ra, dec) in deg\n            (266.4051, -28.936175)>, galcen_distance=8.1 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg): (x, y, z) in kpc\n            [( -9.41284763, -9.40062188, 6.51346272),\n             (-21.08839478, 18.76334013, 7.83184184)]>\n\n    Similarly, transforming from the Galactocentric frame to another coordinate frame::\n\n        >>> c = coord.SkyCoord(x=[-8.3, 4.5] * u.kpc,\n        ...                    y=[0., 81.52] * u.kpc,\n        ...                    z=[0.027, 24.12] * u.kpc,\n        ...                    frame=coord.Galactocentric)\n        >>> c.transform_to(coord.ICRS) # doctest: +FLOAT_CMP\n        <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, kpc)\n            [( 88.22423301, 29.88672864,  0.17813456),\n             (289.72864549, 49.9865043 , 85.93949064)]>\n\n    Or, with custom specification of the Galactic center::\n\n        >>> c = coord.SkyCoord(x=[-8.0, 4.5] * u.kpc,\n        ...                    y=[0., 81.52] * u.kpc,\n        ...                    z=[21.0, 24120.0] * u.pc,\n        ...                    frame=coord.Galactocentric,\n        ...                    z_sun=21 * u.pc, galcen_distance=8. * u.kpc)\n        >>> c.transform_to(coord.ICRS) # doctest: +FLOAT_CMP\n        <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, kpc)\n            [( 86.2585249 , 28.85773187, 2.75625475e-05),\n             (289.77285255, 50.06290457, 8.59216010e+01)]>\n\n"

@format_doc(base_doc, components=doc_components, footer=doc_footer)
class Galactocentric(BaseCoordinateFrame):
    """
    A coordinate or frame in the Galactocentric system.

    This frame allows specifying the Sun-Galactic center distance, the height of
    the Sun above the Galactic midplane, and the solar motion relative to the
    Galactic center. However, as there is no modern standard definition of a
    Galactocentric reference frame, it is important to pay attention to the
    default values used in this class if precision is important in your code.
    The default values of the parameters of this frame are taken from the
    original definition of the frame in 2014. As such, the defaults are somewhat
    out of date relative to recent measurements made possible by, e.g., Gaia.
    The defaults can, however, be changed at runtime by setting the parameter
    set name in `~astropy.coordinates.galactocentric_frame_defaults`.

    The current default parameter set is ``"pre-v4.0"``, indicating that the
    parameters were adopted before ``astropy`` version 4.0. A regularly-updated
    parameter set can instead be used by setting
    ``galactocentric_frame_defaults.set ('latest')``, and other parameter set
    names may be added in future versions. To find out the scientific papers
    that the current default parameters are derived from, use
    ``galcen.frame_attribute_references`` (where ``galcen`` is an instance of
    this frame), which will update even if the default parameter set is changed.

    The position of the Sun is assumed to be on the x axis of the final,
    right-handed system. That is, the x axis points from the position of
    the Sun projected to the Galactic midplane to the Galactic center --
    roughly towards :math:`(l,b) = (0^\\circ,0^\\circ)`. For the default
    transformation (:math:`{\\rm roll}=0^\\circ`), the y axis points roughly
    towards Galactic longitude :math:`l=90^\\circ`, and the z axis points
    roughly towards the North Galactic Pole (:math:`b=90^\\circ`).

    For a more detailed look at the math behind this transformation, see
    the document :ref:`astropy:coordinates-galactocentric`.

    The frame attributes are listed under **Other Parameters**.
    """
    default_representation = r.CartesianRepresentation
    default_differential = r.CartesianDifferential
    galcen_coord = CoordinateAttribute(frame=ICRS)
    galcen_distance = QuantityAttribute(unit=u.kpc)
    galcen_v_sun = DifferentialAttribute(allowed_classes=[r.CartesianDifferential])
    z_sun = QuantityAttribute(unit=u.pc)
    roll = QuantityAttribute(unit=u.deg)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        default_params = galactocentric_frame_defaults.get()
        self.frame_attribute_references = galactocentric_frame_defaults.references.copy()
        for k in default_params:
            if k in kwargs:
                self.frame_attribute_references.pop(k, None)
            kwargs[k] = kwargs.get(k, default_params[k])
        super().__init__(*args, **kwargs)

    @classmethod
    def get_roll0(cls):
        if False:
            i = 10
            return i + 15
        "The additional roll angle (about the final x axis) necessary to align the\n        final z axis to match the Galactic yz-plane.  Setting the ``roll``\n        frame attribute to -this method's return value removes this rotation,\n        allowing the use of the `~astropy.coordinates.Galactocentric` frame\n        in more general contexts.\n\n        "
        return _ROLL0

def get_matrix_vectors(galactocentric_frame, inverse=False):
    if False:
        return 10
    '\n    Use the ``inverse`` argument to get the inverse transformation, matrix and\n    offsets to go from Galactocentric to ICRS.\n    '
    gcf = galactocentric_frame
    mat1 = rotation_matrix(-gcf.galcen_coord.dec, 'y')
    mat2 = rotation_matrix(gcf.galcen_coord.ra, 'z')
    mat0 = rotation_matrix(gcf.get_roll0() - gcf.roll, 'x')
    R = mat0 @ mat1 @ mat2
    translation = r.CartesianRepresentation(gcf.galcen_distance * [1.0, 0.0, 0.0])
    z_d = gcf.z_sun / gcf.galcen_distance
    H = rotation_matrix(-np.arcsin(z_d), 'y')
    A = H @ R
    offset = -translation.transform(H)
    if inverse:
        A = matrix_transpose(A)
        offset = (-offset).transform(A)
        offset_v = r.CartesianDifferential.from_cartesian((-gcf.galcen_v_sun).to_cartesian().transform(A))
        offset = offset.with_differentials(offset_v)
    else:
        offset = offset.with_differentials(gcf.galcen_v_sun)
    return (A, offset)

def _check_coord_repr_diff_types(c):
    if False:
        return 10
    if isinstance(c.data, r.UnitSphericalRepresentation):
        raise ConvertError('Transforming to/from a Galactocentric frame requires a 3D coordinate, e.g. (angle, angle, distance) or (x, y, z).')
    if 's' in c.data.differentials and isinstance(c.data.differentials['s'], (r.UnitSphericalDifferential, r.UnitSphericalCosLatDifferential, r.RadialDifferential)):
        raise ConvertError('Transforming to/from a Galactocentric frame requires a 3D velocity, e.g., proper motion components and radial velocity.')

@frame_transform_graph.transform(AffineTransform, ICRS, Galactocentric)
def icrs_to_galactocentric(icrs_coord, galactocentric_frame):
    if False:
        while True:
            i = 10
    _check_coord_repr_diff_types(icrs_coord)
    return get_matrix_vectors(galactocentric_frame)

@frame_transform_graph.transform(AffineTransform, Galactocentric, ICRS)
def galactocentric_to_icrs(galactocentric_coord, icrs_frame):
    if False:
        for i in range(10):
            print('nop')
    _check_coord_repr_diff_types(galactocentric_coord)
    return get_matrix_vectors(galactocentric_coord, inverse=True)
frame_transform_graph._add_merged_transform(Galactocentric, ICRS, Galactocentric)