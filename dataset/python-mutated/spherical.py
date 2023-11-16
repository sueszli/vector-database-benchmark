"""Spherical representations and differentials."""
import operator
import numpy as np
from erfa import ufunc as erfa_ufunc
import astropy.units as u
from astropy.coordinates.angles import Angle, Latitude, Longitude
from astropy.coordinates.distances import Distance
from astropy.coordinates.matrix_utilities import is_O3
from astropy.utils import classproperty
from .base import BaseDifferential, BaseRepresentation
from .cartesian import CartesianRepresentation

class UnitSphericalRepresentation(BaseRepresentation):
    """
    Representation of points on a unit sphere.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity` ['angle'] or str
        The longitude and latitude of the point(s), in angular units. The
        latitude should be between -90 and 90 degrees, and the longitude will
        be wrapped to an angle between 0 and 360 degrees. These can also be
        instances of `~astropy.coordinates.Angle`,
        `~astropy.coordinates.Longitude`, or `~astropy.coordinates.Latitude`.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single `~astropy.coordinates.BaseDifferential`
        instance (see `._compatible_differentials` for valid types), or a
        dictionary of of differential instances with keys set to a string
        representation of the SI unit with which the differential (derivative)
        is taken. For example, for a velocity differential on a positional
        representation, the key would be ``'s'`` for seconds, indicating that
        the derivative is a time derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes = {'lon': Longitude, 'lat': Latitude}

    @classproperty
    def _dimensional_representation(cls):
        if False:
            return 10
        return SphericalRepresentation

    def __init__(self, lon, lat=None, differentials=None, copy=True):
        if False:
            return 10
        super().__init__(lon, lat, differentials=differentials, copy=copy)

    @classproperty
    def _compatible_differentials(cls):
        if False:
            i = 10
            return i + 15
        return [UnitSphericalDifferential, UnitSphericalCosLatDifferential, SphericalDifferential, SphericalCosLatDifferential, RadialDifferential]

    @property
    def lon(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The longitude of the point(s).\n        '
        return self._lon

    @property
    def lat(self):
        if False:
            print('Hello World!')
        '\n        The latitude of the point(s).\n        '
        return self._lat

    def unit_vectors(self):
        if False:
            print('Hello World!')
        (sinlon, coslon) = (np.sin(self.lon), np.cos(self.lon))
        (sinlat, coslat) = (np.sin(self.lat), np.cos(self.lat))
        return {'lon': CartesianRepresentation(-sinlon, coslon, 0.0, copy=False), 'lat': CartesianRepresentation(-sinlat * coslon, -sinlat * sinlon, coslat, copy=False)}

    def scale_factors(self, omit_coslat=False):
        if False:
            i = 10
            return i + 15
        sf_lat = np.broadcast_to(1.0 / u.radian, self.shape, subok=True)
        sf_lon = sf_lat if omit_coslat else np.cos(self.lat) / u.radian
        return {'lon': sf_lon, 'lat': sf_lat}

    def to_cartesian(self):
        if False:
            while True:
                i = 10
        '\n        Converts spherical polar coordinates to 3D rectangular cartesian\n        coordinates.\n        '
        p = erfa_ufunc.s2c(self.lon, self.lat)
        return CartesianRepresentation(p, xyz_axis=-1, copy=False)

    @classmethod
    def from_cartesian(cls, cart):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts 3D rectangular cartesian coordinates to spherical polar\n        coordinates.\n        '
        p = cart.get_xyz(xyz_axis=-1)
        return cls(*erfa_ufunc.c2s(p), copy=False)

    def represent_as(self, other_class, differential_class=None):
        if False:
            print('Hello World!')
        if isinstance(other_class, type) and (not differential_class):
            if issubclass(other_class, PhysicsSphericalRepresentation):
                return other_class(phi=self.lon, theta=90 * u.deg - self.lat, r=1.0, copy=False)
            elif issubclass(other_class, SphericalRepresentation):
                return other_class(lon=self.lon, lat=self.lat, distance=1.0, copy=False)
        return super().represent_as(other_class, differential_class)

    def transform(self, matrix):
        if False:
            return 10
        'Transform the unit-spherical coordinates using a 3x3 matrix.\n\n        This returns a new representation and does not modify the original one.\n        Any differentials attached to this representation will also be\n        transformed.\n\n        Parameters\n        ----------\n        matrix : (3,3) array-like\n            A 3x3 matrix, such as a rotation matrix (or a stack of matrices).\n\n        Returns\n        -------\n        `~astropy.coordinates.UnitSphericalRepresentation` or `~astropy.coordinates.SphericalRepresentation`\n            If ``matrix`` is O(3) -- :math:`M \\dot M^T = I` -- like a rotation,\n            then the result is a `~astropy.coordinates.UnitSphericalRepresentation`.\n            All other matrices will change the distance, so the dimensional\n            representation is used instead.\n\n        '
        if np.all(is_O3(matrix)):
            xyz = erfa_ufunc.s2c(self.lon, self.lat)
            p = erfa_ufunc.rxp(matrix, xyz)
            (lon, lat) = erfa_ufunc.c2s(p)
            rep = self.__class__(lon=lon, lat=lat)
            new_diffs = {k: d.transform(matrix, self, rep) for (k, d) in self.differentials.items()}
            rep = rep.with_differentials(new_diffs)
        else:
            rep = self._dimensional_representation(lon=self.lon, lat=self.lat, distance=1, differentials=self.differentials).transform(matrix)
        return rep

    def _scale_operation(self, op, *args):
        if False:
            return 10
        return self._dimensional_representation(lon=self.lon, lat=self.lat, distance=1.0, differentials=self.differentials)._scale_operation(op, *args)

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        if any((differential.base_representation is not self.__class__ for differential in self.differentials.values())):
            return super().__neg__()
        result = self.__class__(self.lon + 180.0 * u.deg, -self.lat, copy=False)
        for (key, differential) in self.differentials.items():
            new_comps = (op(getattr(differential, comp)) for (op, comp) in zip((operator.pos, operator.neg), differential.components))
            result.differentials[key] = differential.__class__(*new_comps, copy=False)
        return result

    def norm(self):
        if False:
            for i in range(10):
                print('nop')
        "Vector norm.\n\n        The norm is the standard Frobenius norm, i.e., the square root of the\n        sum of the squares of all components with non-angular units, which is\n        always unity for vectors on the unit sphere.\n\n        Returns\n        -------\n        norm : `~astropy.units.Quantity` ['dimensionless']\n            Dimensionless ones, with the same shape as the representation.\n        "
        return u.Quantity(np.ones(self.shape), u.dimensionless_unscaled, copy=False)

    def _combine_operation(self, op, other, reverse=False):
        if False:
            for i in range(10):
                print('nop')
        self._raise_if_has_differentials(op.__name__)
        result = self.to_cartesian()._combine_operation(op, other, reverse)
        if result is NotImplemented:
            return NotImplemented
        else:
            return self._dimensional_representation.from_cartesian(result)

    def mean(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Vector mean.\n\n        The representation is converted to cartesian, the means of the x, y,\n        and z components are calculated, and the result is converted to a\n        `~astropy.coordinates.SphericalRepresentation`.\n\n        Refer to `~numpy.mean` for full documentation of the arguments, noting\n        that ``axis`` is the entry in the ``shape`` of the representation, and\n        that the ``out`` argument cannot be used.\n        '
        self._raise_if_has_differentials('mean')
        return self._dimensional_representation.from_cartesian(self.to_cartesian().mean(*args, **kwargs))

    def sum(self, *args, **kwargs):
        if False:
            return 10
        'Vector sum.\n\n        The representation is converted to cartesian, the sums of the x, y,\n        and z components are calculated, and the result is converted to a\n        `~astropy.coordinates.SphericalRepresentation`.\n\n        Refer to `~numpy.sum` for full documentation of the arguments, noting\n        that ``axis`` is the entry in the ``shape`` of the representation, and\n        that the ``out`` argument cannot be used.\n        '
        self._raise_if_has_differentials('sum')
        return self._dimensional_representation.from_cartesian(self.to_cartesian().sum(*args, **kwargs))

    def cross(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Cross product of two representations.\n\n        The calculation is done by converting both ``self`` and ``other``\n        to `~astropy.coordinates.CartesianRepresentation`, and converting the\n        result back to `~astropy.coordinates.SphericalRepresentation`.\n\n        Parameters\n        ----------\n        other : `~astropy.coordinates.BaseRepresentation` subclass instance\n            The representation to take the cross product with.\n\n        Returns\n        -------\n        cross_product : `~astropy.coordinates.SphericalRepresentation`\n            With vectors perpendicular to both ``self`` and ``other``.\n        '
        self._raise_if_has_differentials('cross')
        return self._dimensional_representation.from_cartesian(self.to_cartesian().cross(other))

class RadialRepresentation(BaseRepresentation):
    """
    Representation of the distance of points from the origin.

    Note that this is mostly intended as an internal helper representation.
    It can do little else but being used as a scale in multiplication.

    Parameters
    ----------
    distance : `~astropy.units.Quantity` ['length']
        The distance of the point(s) from the origin.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single `~astropy.coordinates.BaseDifferential`
        instance (see `._compatible_differentials` for valid types), or a
        dictionary of of differential instances with keys set to a string
        representation of the SI unit with which the differential (derivative)
        is taken. For example, for a velocity differential on a positional
        representation, the key would be ``'s'`` for seconds, indicating that
        the derivative is a time derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes = {'distance': u.Quantity}

    def __init__(self, distance, differentials=None, copy=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(distance, differentials=differentials, copy=copy)

    @property
    def distance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The distance from the origin to the point(s).\n        '
        return self._distance

    def unit_vectors(self):
        if False:
            for i in range(10):
                print('nop')
        'Cartesian unit vectors are undefined for radial representation.'
        raise NotImplementedError(f'Cartesian unit vectors are undefined for {self.__class__} instances')

    def scale_factors(self):
        if False:
            print('Hello World!')
        l = np.broadcast_to(1.0 * u.one, self.shape, subok=True)
        return {'distance': l}

    def to_cartesian(self):
        if False:
            i = 10
            return i + 15
        'Cannot convert radial representation to cartesian.'
        raise NotImplementedError(f'cannot convert {self.__class__} instance to cartesian.')

    @classmethod
    def from_cartesian(cls, cart):
        if False:
            return 10
        '\n        Converts 3D rectangular cartesian coordinates to radial coordinate.\n        '
        return cls(distance=cart.norm(), copy=False)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, BaseRepresentation):
            return self.distance * other
        else:
            return super().__mul__(other)

    def norm(self):
        if False:
            while True:
                i = 10
        "Vector norm.\n\n        Just the distance itself.\n\n        Returns\n        -------\n        norm : `~astropy.units.Quantity` ['dimensionless']\n            Dimensionless ones, with the same shape as the representation.\n        "
        return self.distance

    def _combine_operation(self, op, other, reverse=False):
        if False:
            return 10
        return NotImplemented

    def transform(self, matrix):
        if False:
            print('Hello World!')
        'Radial representations cannot be transformed by a Cartesian matrix.\n\n        Parameters\n        ----------\n        matrix : array-like\n            The transformation matrix in a Cartesian basis.\n            Must be a multiplication: a diagonal matrix with identical elements.\n            Must have shape (..., 3, 3), where the last 2 indices are for the\n            matrix on each other axis. Make sure that the matrix shape is\n            compatible with the shape of this representation.\n\n        Raises\n        ------\n        ValueError\n            If the matrix is not a multiplication.\n\n        '
        scl = matrix[..., 0, 0]
        if np.any(matrix != scl[..., np.newaxis, np.newaxis] * np.identity(3)):
            raise ValueError('Radial representations can only be transformed by a scaled identity matrix')
        return self * scl

def _spherical_op_funcs(op, *args):
    if False:
        while True:
            i = 10
    'For given operator, return functions that adjust lon, lat, distance.'
    if op is operator.neg:
        return (lambda x: x + 180 * u.deg, operator.neg, operator.pos)
    try:
        scale_sign = np.sign(args[0])
    except Exception:
        return (operator.pos, operator.pos, lambda x: op(x, *args))
    scale = abs(args[0])
    return (lambda x: x + 180 * u.deg * np.signbit(scale_sign), lambda x: x * scale_sign, lambda x: op(x, scale))

class SphericalRepresentation(BaseRepresentation):
    """
    Representation of points in 3D spherical coordinates.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity` ['angle']
        The longitude and latitude of the point(s), in angular units. The
        latitude should be between -90 and 90 degrees, and the longitude will
        be wrapped to an angle between 0 and 360 degrees. These can also be
        instances of `~astropy.coordinates.Angle`,
        `~astropy.coordinates.Longitude`, or `~astropy.coordinates.Latitude`.

    distance : `~astropy.units.Quantity` ['length']
        The distance to the point(s). If the distance is a length, it is
        passed to the :class:`~astropy.coordinates.Distance` class, otherwise
        it is passed to the :class:`~astropy.units.Quantity` class.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single `~astropy.coordinates.BaseDifferential`
        instance (see `._compatible_differentials` for valid types), or a
        dictionary of of differential instances with keys set to a string
        representation of the SI unit with which the differential (derivative)
        is taken. For example, for a velocity differential on a positional
        representation, the key would be ``'s'`` for seconds, indicating that
        the derivative is a time derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes = {'lon': Longitude, 'lat': Latitude, 'distance': u.Quantity}
    _unit_representation = UnitSphericalRepresentation

    def __init__(self, lon, lat=None, distance=None, differentials=None, copy=True):
        if False:
            while True:
                i = 10
        super().__init__(lon, lat, distance, copy=copy, differentials=differentials)
        if not isinstance(self._distance, Distance) and self._distance.unit.physical_type == 'length':
            try:
                self._distance = Distance(self._distance, copy=False)
            except ValueError as e:
                if e.args[0].startswith('distance must be >= 0'):
                    raise ValueError("Distance must be >= 0. To allow negative distance values, you must explicitly pass in a `Distance` object with the argument 'allow_negative=True'.") from e
                else:
                    raise

    @classproperty
    def _compatible_differentials(cls):
        if False:
            while True:
                i = 10
        return [UnitSphericalDifferential, UnitSphericalCosLatDifferential, SphericalDifferential, SphericalCosLatDifferential, RadialDifferential]

    @property
    def lon(self):
        if False:
            while True:
                i = 10
        '\n        The longitude of the point(s).\n        '
        return self._lon

    @property
    def lat(self):
        if False:
            while True:
                i = 10
        '\n        The latitude of the point(s).\n        '
        return self._lat

    @property
    def distance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The distance from the origin to the point(s).\n        '
        return self._distance

    def unit_vectors(self):
        if False:
            while True:
                i = 10
        (sinlon, coslon) = (np.sin(self.lon), np.cos(self.lon))
        (sinlat, coslat) = (np.sin(self.lat), np.cos(self.lat))
        return {'lon': CartesianRepresentation(-sinlon, coslon, 0.0, copy=False), 'lat': CartesianRepresentation(-sinlat * coslon, -sinlat * sinlon, coslat, copy=False), 'distance': CartesianRepresentation(coslat * coslon, coslat * sinlon, sinlat, copy=False)}

    def scale_factors(self, omit_coslat=False):
        if False:
            for i in range(10):
                print('nop')
        sf_lat = self.distance / u.radian
        sf_lon = sf_lat if omit_coslat else sf_lat * np.cos(self.lat)
        sf_distance = np.broadcast_to(1.0 * u.one, self.shape, subok=True)
        return {'lon': sf_lon, 'lat': sf_lat, 'distance': sf_distance}

    def represent_as(self, other_class, differential_class=None):
        if False:
            return 10
        if isinstance(other_class, type):
            if issubclass(other_class, PhysicsSphericalRepresentation):
                diffs = self._re_represent_differentials(other_class, differential_class)
                return other_class(phi=self.lon, theta=90 * u.deg - self.lat, r=self.distance, differentials=diffs, copy=False)
            elif issubclass(other_class, UnitSphericalRepresentation):
                diffs = self._re_represent_differentials(other_class, differential_class)
                return other_class(lon=self.lon, lat=self.lat, differentials=diffs, copy=False)
        return super().represent_as(other_class, differential_class)

    def to_cartesian(self):
        if False:
            i = 10
            return i + 15
        '\n        Converts spherical polar coordinates to 3D rectangular cartesian\n        coordinates.\n        '
        if isinstance(self.distance, Distance):
            d = self.distance.view(u.Quantity)
        else:
            d = self.distance
        p = erfa_ufunc.s2p(self.lon, self.lat, d)
        return CartesianRepresentation(p, xyz_axis=-1, copy=False)

    @classmethod
    def from_cartesian(cls, cart):
        if False:
            while True:
                i = 10
        '\n        Converts 3D rectangular cartesian coordinates to spherical polar\n        coordinates.\n        '
        p = cart.get_xyz(xyz_axis=-1)
        return cls(*erfa_ufunc.p2s(p), copy=False)

    def transform(self, matrix):
        if False:
            i = 10
            return i + 15
        'Transform the spherical coordinates using a 3x3 matrix.\n\n        This returns a new representation and does not modify the original one.\n        Any differentials attached to this representation will also be\n        transformed.\n\n        Parameters\n        ----------\n        matrix : (3,3) array-like\n            A 3x3 matrix, such as a rotation matrix (or a stack of matrices).\n\n        '
        xyz = erfa_ufunc.s2c(self.lon, self.lat)
        p = erfa_ufunc.rxp(matrix, xyz)
        (lon, lat, ur) = erfa_ufunc.p2s(p)
        rep = self.__class__(lon=lon, lat=lat, distance=self.distance * ur)
        new_diffs = {k: d.transform(matrix, self, rep) for (k, d) in self.differentials.items()}
        return rep.with_differentials(new_diffs)

    def norm(self):
        if False:
            i = 10
            return i + 15
        'Vector norm.\n\n        The norm is the standard Frobenius norm, i.e., the square root of the\n        sum of the squares of all components with non-angular units.  For\n        spherical coordinates, this is just the absolute value of the distance.\n\n        Returns\n        -------\n        norm : `astropy.units.Quantity`\n            Vector norm, with the same shape as the representation.\n        '
        return np.abs(self.distance)

    def _scale_operation(self, op, *args):
        if False:
            return 10
        if any((differential.base_representation is not self.__class__ for differential in self.differentials.values())):
            return super()._scale_operation(op, *args)
        (lon_op, lat_op, distance_op) = _spherical_op_funcs(op, *args)
        result = self.__class__(lon_op(self.lon), lat_op(self.lat), distance_op(self.distance), copy=False)
        for (key, differential) in self.differentials.items():
            new_comps = (op(getattr(differential, comp)) for (op, comp) in zip((operator.pos, lat_op, distance_op), differential.components))
            result.differentials[key] = differential.__class__(*new_comps, copy=False)
        return result

class PhysicsSphericalRepresentation(BaseRepresentation):
    """
    Representation of points in 3D spherical coordinates (using the physics
    convention of using ``phi`` and ``theta`` for azimuth and inclination
    from the pole).

    Parameters
    ----------
    phi, theta : `~astropy.units.Quantity` or str
        The azimuth and inclination of the point(s), in angular units. The
        inclination should be between 0 and 180 degrees, and the azimuth will
        be wrapped to an angle between 0 and 360 degrees. These can also be
        instances of `~astropy.coordinates.Angle`.  If ``copy`` is False, `phi`
        will be changed inplace if it is not between 0 and 360 degrees.

    r : `~astropy.units.Quantity`
        The distance to the point(s). If the distance is a length, it is
        passed to the :class:`~astropy.coordinates.Distance` class, otherwise
        it is passed to the :class:`~astropy.units.Quantity` class.

    differentials : dict, `~astropy.coordinates.PhysicsSphericalDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single
        `~astropy.coordinates.PhysicsSphericalDifferential` instance, or a dictionary of of
        differential instances with keys set to a string representation of the
        SI unit with which the differential (derivative) is taken. For example,
        for a velocity differential on a positional representation, the key
        would be ``'s'`` for seconds, indicating that the derivative is a time
        derivative.

    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    attr_classes = {'phi': Angle, 'theta': Angle, 'r': u.Quantity}

    def __init__(self, phi, theta=None, r=None, differentials=None, copy=True):
        if False:
            return 10
        super().__init__(phi, theta, r, copy=copy, differentials=differentials)
        self._phi.wrap_at(360 * u.deg, inplace=True)
        if np.any(self._theta < 0.0 * u.deg) or np.any(self._theta > 180.0 * u.deg):
            raise ValueError(f'Inclination angle(s) must be within 0 deg <= angle <= 180 deg, got {theta.to(u.degree)}')
        if self._r.unit.physical_type == 'length':
            self._r = self._r.view(Distance)

    @property
    def phi(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The azimuth of the point(s).\n        '
        return self._phi

    @property
    def theta(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The elevation of the point(s).\n        '
        return self._theta

    @property
    def r(self):
        if False:
            while True:
                i = 10
        '\n        The distance from the origin to the point(s).\n        '
        return self._r

    def unit_vectors(self):
        if False:
            print('Hello World!')
        (sinphi, cosphi) = (np.sin(self.phi), np.cos(self.phi))
        (sintheta, costheta) = (np.sin(self.theta), np.cos(self.theta))
        return {'phi': CartesianRepresentation(-sinphi, cosphi, 0.0, copy=False), 'theta': CartesianRepresentation(costheta * cosphi, costheta * sinphi, -sintheta, copy=False), 'r': CartesianRepresentation(sintheta * cosphi, sintheta * sinphi, costheta, copy=False)}

    def scale_factors(self):
        if False:
            i = 10
            return i + 15
        r = self.r / u.radian
        sintheta = np.sin(self.theta)
        l = np.broadcast_to(1.0 * u.one, self.shape, subok=True)
        return {'phi': r * sintheta, 'theta': r, 'r': l}

    def represent_as(self, other_class, differential_class=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other_class, type):
            if issubclass(other_class, SphericalRepresentation):
                diffs = self._re_represent_differentials(other_class, differential_class)
                return other_class(lon=self.phi, lat=90 * u.deg - self.theta, distance=self.r, differentials=diffs, copy=False)
            elif issubclass(other_class, UnitSphericalRepresentation):
                diffs = self._re_represent_differentials(other_class, differential_class)
                return other_class(lon=self.phi, lat=90 * u.deg - self.theta, differentials=diffs, copy=False)
        return super().represent_as(other_class, differential_class)

    def to_cartesian(self):
        if False:
            print('Hello World!')
        '\n        Converts spherical polar coordinates to 3D rectangular cartesian\n        coordinates.\n        '
        if isinstance(self.r, Distance):
            d = self.r.view(u.Quantity)
        else:
            d = self.r
        x = d * np.sin(self.theta) * np.cos(self.phi)
        y = d * np.sin(self.theta) * np.sin(self.phi)
        z = d * np.cos(self.theta)
        return CartesianRepresentation(x=x, y=y, z=z, copy=False)

    @classmethod
    def from_cartesian(cls, cart):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts 3D rectangular cartesian coordinates to spherical polar\n        coordinates.\n        '
        s = np.hypot(cart.x, cart.y)
        r = np.hypot(s, cart.z)
        phi = np.arctan2(cart.y, cart.x)
        theta = np.arctan2(s, cart.z)
        return cls(phi=phi, theta=theta, r=r, copy=False)

    def transform(self, matrix):
        if False:
            while True:
                i = 10
        'Transform the spherical coordinates using a 3x3 matrix.\n\n        This returns a new representation and does not modify the original one.\n        Any differentials attached to this representation will also be\n        transformed.\n\n        Parameters\n        ----------\n        matrix : (3,3) array-like\n            A 3x3 matrix, such as a rotation matrix (or a stack of matrices).\n\n        '
        xyz = erfa_ufunc.s2c(self.phi, 90 * u.deg - self.theta)
        p = erfa_ufunc.rxp(matrix, xyz)
        (lon, lat, ur) = erfa_ufunc.p2s(p)
        rep = self.__class__(phi=lon, theta=90 * u.deg - lat, r=self.r * ur)
        new_diffs = {k: d.transform(matrix, self, rep) for (k, d) in self.differentials.items()}
        return rep.with_differentials(new_diffs)

    def norm(self):
        if False:
            i = 10
            return i + 15
        'Vector norm.\n\n        The norm is the standard Frobenius norm, i.e., the square root of the\n        sum of the squares of all components with non-angular units.  For\n        spherical coordinates, this is just the absolute value of the radius.\n\n        Returns\n        -------\n        norm : `astropy.units.Quantity`\n            Vector norm, with the same shape as the representation.\n        '
        return np.abs(self.r)

    def _scale_operation(self, op, *args):
        if False:
            for i in range(10):
                print('nop')
        if any((differential.base_representation is not self.__class__ for differential in self.differentials.values())):
            return super()._scale_operation(op, *args)
        (phi_op, adjust_theta_sign, r_op) = _spherical_op_funcs(op, *args)
        result = self.__class__(phi_op(self.phi), phi_op(adjust_theta_sign(self.theta)), r_op(self.r), copy=False)
        for (key, differential) in self.differentials.items():
            new_comps = (op(getattr(differential, comp)) for (op, comp) in zip((operator.pos, adjust_theta_sign, r_op), differential.components))
            result.differentials[key] = differential.__class__(*new_comps, copy=False)
        return result

class BaseSphericalDifferential(BaseDifferential):

    def _d_lon_coslat(self, base):
        if False:
            while True:
                i = 10
        'Convert longitude differential d_lon to d_lon_coslat.\n\n        Parameters\n        ----------\n        base : instance of ``cls.base_representation``\n            The base from which the latitude will be taken.\n        '
        self._check_base(base)
        return self.d_lon * np.cos(base.lat)

    @classmethod
    def _get_d_lon(cls, d_lon_coslat, base):
        if False:
            return 10
        'Convert longitude differential d_lon_coslat to d_lon.\n\n        Parameters\n        ----------\n        d_lon_coslat : `~astropy.units.Quantity`\n            Longitude differential that includes ``cos(lat)``.\n        base : instance of ``cls.base_representation``\n            The base from which the latitude will be taken.\n        '
        cls._check_base(base)
        return d_lon_coslat / np.cos(base.lat)

    def _combine_operation(self, op, other, reverse=False):
        if False:
            while True:
                i = 10
        'Combine two differentials, or a differential with a representation.\n\n        If ``other`` is of the same differential type as ``self``, the\n        components will simply be combined.  If both are different parts of\n        a `~astropy.coordinates.SphericalDifferential` (e.g., a\n        `~astropy.coordinates.UnitSphericalDifferential` and a\n        `~astropy.coordinates.RadialDifferential`), they will combined\n        appropriately.\n\n        If ``other`` is a representation, it will be used as a base for which\n        to evaluate the differential, and the result is a new representation.\n\n        Parameters\n        ----------\n        op : `~operator` callable\n            Operator to apply (e.g., `~operator.add`, `~operator.sub`, etc.\n        other : `~astropy.coordinates.BaseRepresentation` subclass instance\n            The other differential or representation.\n        reverse : bool\n            Whether the operands should be reversed (e.g., as we got here via\n            ``self.__rsub__`` because ``self`` is a subclass of ``other``).\n        '
        if isinstance(other, BaseSphericalDifferential) and (not isinstance(self, type(other))) or isinstance(other, RadialDifferential):
            all_components = set(self.components) | set(other.components)
            (first, second) = (self, other) if not reverse else (other, self)
            result_args = {c: op(getattr(first, c, 0.0), getattr(second, c, 0.0)) for c in all_components}
            return SphericalDifferential(**result_args)
        return super()._combine_operation(op, other, reverse)

class UnitSphericalDifferential(BaseSphericalDifferential):
    """Differential(s) of points on a unit sphere.

    Parameters
    ----------
    d_lon, d_lat : `~astropy.units.Quantity`
        The longitude and latitude of the differentials.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = UnitSphericalRepresentation

    @classproperty
    def _dimensional_differential(cls):
        if False:
            for i in range(10):
                print('nop')
        return SphericalDifferential

    def __init__(self, d_lon, d_lat=None, copy=True):
        if False:
            print('Hello World!')
        super().__init__(d_lon, d_lat, copy=copy)
        if not self._d_lon.unit.is_equivalent(self._d_lat.unit):
            raise u.UnitsError('d_lon and d_lat should have equivalent units.')

    @classmethod
    def from_cartesian(cls, other, base):
        if False:
            return 10
        dimensional = cls._dimensional_differential.from_cartesian(other, base)
        return dimensional.represent_as(cls)

    def to_cartesian(self, base):
        if False:
            return 10
        if isinstance(base, SphericalRepresentation):
            scale = base.distance
        elif isinstance(base, PhysicsSphericalRepresentation):
            scale = base.r
        else:
            return super().to_cartesian(base)
        base = base.represent_as(UnitSphericalRepresentation)
        return scale * super().to_cartesian(base)

    def represent_as(self, other_class, base=None):
        if False:
            return 10
        if issubclass(other_class, UnitSphericalCosLatDifferential):
            return other_class(self._d_lon_coslat(base), self.d_lat)
        return super().represent_as(other_class, base)

    @classmethod
    def from_representation(cls, representation, base=None):
        if False:
            return 10
        if isinstance(representation, SphericalDifferential):
            return cls(representation.d_lon, representation.d_lat)
        elif isinstance(representation, (SphericalCosLatDifferential, UnitSphericalCosLatDifferential)):
            d_lon = cls._get_d_lon(representation.d_lon_coslat, base)
            return cls(d_lon, representation.d_lat)
        elif isinstance(representation, PhysicsSphericalDifferential):
            return cls(representation.d_phi, -representation.d_theta)
        return super().from_representation(representation, base)

    def transform(self, matrix, base, transformed_base):
        if False:
            i = 10
            return i + 15
        'Transform differential using a 3x3 matrix in a Cartesian basis.\n\n        This returns a new differential and does not modify the original one.\n\n        Parameters\n        ----------\n        matrix : (3,3) array-like\n            A 3x3 (or stack thereof) matrix, such as a rotation matrix.\n        base : instance of ``cls.base_representation``\n            Base relative to which the differentials are defined.  If the other\n            class is a differential representation, the base will be converted\n            to its ``base_representation``.\n        transformed_base : instance of ``cls.base_representation``\n            Base relative to which the transformed differentials are defined.\n            If the other class is a differential representation, the base will\n            be converted to its ``base_representation``.\n        '
        if np.all(is_O3(matrix)):
            diff = super().transform(matrix, base, transformed_base)
        else:
            du = self.d_lon.unit / base.lon.unit
            diff = self._dimensional_differential(d_lon=self.d_lon, d_lat=self.d_lat, d_distance=0 * du).transform(matrix, base, transformed_base)
        return diff

    def _scale_operation(self, op, *args, scaled_base=False):
        if False:
            for i in range(10):
                print('nop')
        if scaled_base:
            return self.copy()
        else:
            return super()._scale_operation(op, *args)

class SphericalDifferential(BaseSphericalDifferential):
    """Differential(s) of points in 3D spherical coordinates.

    Parameters
    ----------
    d_lon, d_lat : `~astropy.units.Quantity`
        The differential longitude and latitude.
    d_distance : `~astropy.units.Quantity`
        The differential distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = SphericalRepresentation
    _unit_differential = UnitSphericalDifferential

    def __init__(self, d_lon, d_lat=None, d_distance=None, copy=True):
        if False:
            return 10
        super().__init__(d_lon, d_lat, d_distance, copy=copy)
        if not self._d_lon.unit.is_equivalent(self._d_lat.unit):
            raise u.UnitsError('d_lon and d_lat should have equivalent units.')

    def represent_as(self, other_class, base=None):
        if False:
            print('Hello World!')
        if issubclass(other_class, UnitSphericalDifferential):
            return other_class(self.d_lon, self.d_lat)
        elif issubclass(other_class, RadialDifferential):
            return other_class(self.d_distance)
        elif issubclass(other_class, SphericalCosLatDifferential):
            return other_class(self._d_lon_coslat(base), self.d_lat, self.d_distance)
        elif issubclass(other_class, UnitSphericalCosLatDifferential):
            return other_class(self._d_lon_coslat(base), self.d_lat)
        elif issubclass(other_class, PhysicsSphericalDifferential):
            return other_class(self.d_lon, -self.d_lat, self.d_distance)
        else:
            return super().represent_as(other_class, base)

    @classmethod
    def from_representation(cls, representation, base=None):
        if False:
            print('Hello World!')
        if isinstance(representation, SphericalCosLatDifferential):
            d_lon = cls._get_d_lon(representation.d_lon_coslat, base)
            return cls(d_lon, representation.d_lat, representation.d_distance)
        elif isinstance(representation, PhysicsSphericalDifferential):
            return cls(representation.d_phi, -representation.d_theta, representation.d_r)
        return super().from_representation(representation, base)

    def _scale_operation(self, op, *args, scaled_base=False):
        if False:
            while True:
                i = 10
        if scaled_base:
            return self.__class__(self.d_lon, self.d_lat, op(self.d_distance, *args))
        else:
            return super()._scale_operation(op, *args)

class BaseSphericalCosLatDifferential(BaseDifferential):
    """Differentials from points on a spherical base representation.

    With cos(lat) assumed to be included in the longitude differential.
    """

    @classmethod
    def _get_base_vectors(cls, base):
        if False:
            print('Hello World!')
        'Get unit vectors and scale factors from (unit)spherical base.\n\n        Parameters\n        ----------\n        base : instance of ``self.base_representation``\n            The points for which the unit vectors and scale factors should be\n            retrieved.\n\n        Returns\n        -------\n        unit_vectors : dict of `~astropy.coordinates.CartesianRepresentation`\n            In the directions of the coordinates of base.\n        scale_factors : dict of `~astropy.units.Quantity`\n            Scale factors for each of the coordinates.  The scale factor for\n            longitude does not include the cos(lat) factor.\n\n        Raises\n        ------\n        TypeError : if the base is not of the correct type\n        '
        cls._check_base(base)
        return (base.unit_vectors(), base.scale_factors(omit_coslat=True))

    def _d_lon(self, base):
        if False:
            for i in range(10):
                print('nop')
        'Convert longitude differential with cos(lat) to one without.\n\n        Parameters\n        ----------\n        base : instance of ``cls.base_representation``\n            The base from which the latitude will be taken.\n        '
        self._check_base(base)
        return self.d_lon_coslat / np.cos(base.lat)

    @classmethod
    def _get_d_lon_coslat(cls, d_lon, base):
        if False:
            return 10
        'Convert longitude differential d_lon to d_lon_coslat.\n\n        Parameters\n        ----------\n        d_lon : `~astropy.units.Quantity`\n            Value of the longitude differential without ``cos(lat)``.\n        base : instance of ``cls.base_representation``\n            The base from which the latitude will be taken.\n        '
        cls._check_base(base)
        return d_lon * np.cos(base.lat)

    def _combine_operation(self, op, other, reverse=False):
        if False:
            return 10
        'Combine two differentials, or a differential with a representation.\n\n        If ``other`` is of the same differential type as ``self``, the\n        components will simply be combined.  If both are different parts of\n        a `~astropy.coordinates.SphericalDifferential` (e.g., a\n        `~astropy.coordinates.UnitSphericalDifferential` and a\n        `~astropy.coordinates.RadialDifferential`), they will combined\n        appropriately.\n\n        If ``other`` is a representation, it will be used as a base for which\n        to evaluate the differential, and the result is a new representation.\n\n        Parameters\n        ----------\n        op : `~operator` callable\n            Operator to apply (e.g., `~operator.add`, `~operator.sub`, etc.\n        other : `~astropy.coordinates.BaseRepresentation` subclass instance\n            The other differential or representation.\n        reverse : bool\n            Whether the operands should be reversed (e.g., as we got here via\n            ``self.__rsub__`` because ``self`` is a subclass of ``other``).\n        '
        if isinstance(other, BaseSphericalCosLatDifferential) and (not isinstance(self, type(other))) or isinstance(other, RadialDifferential):
            all_components = set(self.components) | set(other.components)
            (first, second) = (self, other) if not reverse else (other, self)
            result_args = {c: op(getattr(first, c, 0.0), getattr(second, c, 0.0)) for c in all_components}
            return SphericalCosLatDifferential(**result_args)
        return super()._combine_operation(op, other, reverse)

class UnitSphericalCosLatDifferential(BaseSphericalCosLatDifferential):
    """Differential(s) of points on a unit sphere.

    Parameters
    ----------
    d_lon_coslat, d_lat : `~astropy.units.Quantity`
        The longitude and latitude of the differentials.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = UnitSphericalRepresentation
    attr_classes = {'d_lon_coslat': u.Quantity, 'd_lat': u.Quantity}

    @classproperty
    def _dimensional_differential(cls):
        if False:
            return 10
        return SphericalCosLatDifferential

    def __init__(self, d_lon_coslat, d_lat=None, copy=True):
        if False:
            return 10
        super().__init__(d_lon_coslat, d_lat, copy=copy)
        if not self._d_lon_coslat.unit.is_equivalent(self._d_lat.unit):
            raise u.UnitsError('d_lon_coslat and d_lat should have equivalent units.')

    @classmethod
    def from_cartesian(cls, other, base):
        if False:
            i = 10
            return i + 15
        dimensional = cls._dimensional_differential.from_cartesian(other, base)
        return dimensional.represent_as(cls)

    def to_cartesian(self, base):
        if False:
            i = 10
            return i + 15
        if isinstance(base, SphericalRepresentation):
            scale = base.distance
        elif isinstance(base, PhysicsSphericalRepresentation):
            scale = base.r
        else:
            return super().to_cartesian(base)
        base = base.represent_as(UnitSphericalRepresentation)
        return scale * super().to_cartesian(base)

    def represent_as(self, other_class, base=None):
        if False:
            while True:
                i = 10
        if issubclass(other_class, UnitSphericalDifferential):
            return other_class(self._d_lon(base), self.d_lat)
        return super().represent_as(other_class, base)

    @classmethod
    def from_representation(cls, representation, base=None):
        if False:
            while True:
                i = 10
        if isinstance(representation, SphericalCosLatDifferential):
            return cls(representation.d_lon_coslat, representation.d_lat)
        elif isinstance(representation, (SphericalDifferential, UnitSphericalDifferential)):
            d_lon_coslat = cls._get_d_lon_coslat(representation.d_lon, base)
            return cls(d_lon_coslat, representation.d_lat)
        elif isinstance(representation, PhysicsSphericalDifferential):
            d_lon_coslat = cls._get_d_lon_coslat(representation.d_phi, base)
            return cls(d_lon_coslat, -representation.d_theta)
        return super().from_representation(representation, base)

    def transform(self, matrix, base, transformed_base):
        if False:
            print('Hello World!')
        'Transform differential using a 3x3 matrix in a Cartesian basis.\n\n        This returns a new differential and does not modify the original one.\n\n        Parameters\n        ----------\n        matrix : (3,3) array-like\n            A 3x3 (or stack thereof) matrix, such as a rotation matrix.\n        base : instance of ``cls.base_representation``\n            Base relative to which the differentials are defined.  If the other\n            class is a differential representation, the base will be converted\n            to its ``base_representation``.\n        transformed_base : instance of ``cls.base_representation``\n            Base relative to which the transformed differentials are defined.\n            If the other class is a differential representation, the base will\n            be converted to its ``base_representation``.\n        '
        if np.all(is_O3(matrix)):
            diff = super().transform(matrix, base, transformed_base)
        else:
            du = self.d_lat.unit / base.lat.unit
            diff = self._dimensional_differential(d_lon_coslat=self.d_lon_coslat, d_lat=self.d_lat, d_distance=0 * du).transform(matrix, base, transformed_base)
        return diff

    def _scale_operation(self, op, *args, scaled_base=False):
        if False:
            while True:
                i = 10
        if scaled_base:
            return self.copy()
        else:
            return super()._scale_operation(op, *args)

class SphericalCosLatDifferential(BaseSphericalCosLatDifferential):
    """Differential(s) of points in 3D spherical coordinates.

    Parameters
    ----------
    d_lon_coslat, d_lat : `~astropy.units.Quantity`
        The differential longitude (with cos(lat) included) and latitude.
    d_distance : `~astropy.units.Quantity`
        The differential distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = SphericalRepresentation
    _unit_differential = UnitSphericalCosLatDifferential
    attr_classes = {'d_lon_coslat': u.Quantity, 'd_lat': u.Quantity, 'd_distance': u.Quantity}

    def __init__(self, d_lon_coslat, d_lat=None, d_distance=None, copy=True):
        if False:
            print('Hello World!')
        super().__init__(d_lon_coslat, d_lat, d_distance, copy=copy)
        if not self._d_lon_coslat.unit.is_equivalent(self._d_lat.unit):
            raise u.UnitsError('d_lon_coslat and d_lat should have equivalent units.')

    def represent_as(self, other_class, base=None):
        if False:
            print('Hello World!')
        if issubclass(other_class, UnitSphericalCosLatDifferential):
            return other_class(self.d_lon_coslat, self.d_lat)
        elif issubclass(other_class, RadialDifferential):
            return other_class(self.d_distance)
        elif issubclass(other_class, SphericalDifferential):
            return other_class(self._d_lon(base), self.d_lat, self.d_distance)
        elif issubclass(other_class, UnitSphericalDifferential):
            return other_class(self._d_lon(base), self.d_lat)
        elif issubclass(other_class, PhysicsSphericalDifferential):
            return other_class(self._d_lon(base), -self.d_lat, self.d_distance)
        return super().represent_as(other_class, base)

    @classmethod
    def from_representation(cls, representation, base=None):
        if False:
            return 10
        if isinstance(representation, SphericalDifferential):
            d_lon_coslat = cls._get_d_lon_coslat(representation.d_lon, base)
            return cls(d_lon_coslat, representation.d_lat, representation.d_distance)
        elif isinstance(representation, PhysicsSphericalDifferential):
            d_lon_coslat = cls._get_d_lon_coslat(representation.d_phi, base)
            return cls(d_lon_coslat, -representation.d_theta, representation.d_r)
        return super().from_representation(representation, base)

    def _scale_operation(self, op, *args, scaled_base=False):
        if False:
            for i in range(10):
                print('nop')
        if scaled_base:
            return self.__class__(self.d_lon_coslat, self.d_lat, op(self.d_distance, *args))
        else:
            return super()._scale_operation(op, *args)

class RadialDifferential(BaseDifferential):
    """Differential(s) of radial distances.

    Parameters
    ----------
    d_distance : `~astropy.units.Quantity`
        The differential distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = RadialRepresentation

    def to_cartesian(self, base):
        if False:
            i = 10
            return i + 15
        unit_vec = base.represent_as(UnitSphericalRepresentation).to_cartesian()
        return self.d_distance * unit_vec

    def norm(self, base=None):
        if False:
            for i in range(10):
                print('nop')
        return self.d_distance

    @classmethod
    def from_cartesian(cls, other, base):
        if False:
            for i in range(10):
                print('nop')
        return cls(other.dot(base.represent_as(UnitSphericalRepresentation)), copy=False)

    @classmethod
    def from_representation(cls, representation, base=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(representation, (SphericalDifferential, SphericalCosLatDifferential)):
            return cls(representation.d_distance)
        elif isinstance(representation, PhysicsSphericalDifferential):
            return cls(representation.d_r)
        else:
            return super().from_representation(representation, base)

    def _combine_operation(self, op, other, reverse=False):
        if False:
            return 10
        if isinstance(other, self.base_representation):
            if reverse:
                (first, second) = (other.distance, self.d_distance)
            else:
                (first, second) = (self.d_distance, other.distance)
            return other.__class__(op(first, second), copy=False)
        elif isinstance(other, (BaseSphericalDifferential, BaseSphericalCosLatDifferential)):
            all_components = set(self.components) | set(other.components)
            (first, second) = (self, other) if not reverse else (other, self)
            result_args = {c: op(getattr(first, c, 0.0), getattr(second, c, 0.0)) for c in all_components}
            return SphericalDifferential(**result_args)
        else:
            return super()._combine_operation(op, other, reverse)

class PhysicsSphericalDifferential(BaseDifferential):
    """Differential(s) of 3D spherical coordinates using physics convention.

    Parameters
    ----------
    d_phi, d_theta : `~astropy.units.Quantity`
        The differential azimuth and inclination.
    d_r : `~astropy.units.Quantity`
        The differential radial distance.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """
    base_representation = PhysicsSphericalRepresentation

    def __init__(self, d_phi, d_theta=None, d_r=None, copy=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(d_phi, d_theta, d_r, copy=copy)
        if not self._d_phi.unit.is_equivalent(self._d_theta.unit):
            raise u.UnitsError('d_phi and d_theta should have equivalent units.')

    def represent_as(self, other_class, base=None):
        if False:
            print('Hello World!')
        if issubclass(other_class, SphericalDifferential):
            return other_class(self.d_phi, -self.d_theta, self.d_r)
        elif issubclass(other_class, UnitSphericalDifferential):
            return other_class(self.d_phi, -self.d_theta)
        elif issubclass(other_class, SphericalCosLatDifferential):
            self._check_base(base)
            d_lon_coslat = self.d_phi * np.sin(base.theta)
            return other_class(d_lon_coslat, -self.d_theta, self.d_r)
        elif issubclass(other_class, UnitSphericalCosLatDifferential):
            self._check_base(base)
            d_lon_coslat = self.d_phi * np.sin(base.theta)
            return other_class(d_lon_coslat, -self.d_theta)
        elif issubclass(other_class, RadialDifferential):
            return other_class(self.d_r)
        return super().represent_as(other_class, base)

    @classmethod
    def from_representation(cls, representation, base=None):
        if False:
            print('Hello World!')
        if isinstance(representation, SphericalDifferential):
            return cls(representation.d_lon, -representation.d_lat, representation.d_distance)
        elif isinstance(representation, SphericalCosLatDifferential):
            cls._check_base(base)
            d_phi = representation.d_lon_coslat / np.sin(base.theta)
            return cls(d_phi, -representation.d_lat, representation.d_distance)
        return super().from_representation(representation, base)

    def _scale_operation(self, op, *args, scaled_base=False):
        if False:
            i = 10
            return i + 15
        if scaled_base:
            return self.__class__(self.d_phi, self.d_theta, op(self.d_r, *args))
        else:
            return super()._scale_operation(op, *args)