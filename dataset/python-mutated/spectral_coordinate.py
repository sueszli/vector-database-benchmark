import warnings
from textwrap import indent
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.coordinates import ICRS, CartesianDifferential, CartesianRepresentation, SkyCoord
from astropy.coordinates.baseframe import BaseCoordinateFrame, frame_transform_graph
from astropy.coordinates.spectral_quantity import SpectralQuantity
from astropy.utils.exceptions import AstropyUserWarning
__all__ = ['SpectralCoord']

class NoVelocityWarning(AstropyUserWarning):
    pass

class NoDistanceWarning(AstropyUserWarning):
    pass
KMS = u.km / u.s
ZERO_VELOCITIES = CartesianDifferential([0, 0, 0] * KMS)
DEFAULT_DISTANCE = 1000000.0 * u.kpc
__doctest_skip__ = ['SpectralCoord.*']

def _apply_relativistic_doppler_shift(scoord, velocity):
    if False:
        print('Hello World!')
    '\n    Given a `SpectralQuantity` and a velocity, return a new `SpectralQuantity`\n    that is Doppler shifted by this amount.\n\n    Note that the Doppler shift applied is the full relativistic one, so\n    `SpectralQuantity` currently expressed in velocity and not using the\n    relativistic convention will temporarily be converted to use the\n    relativistic convention while the shift is applied.\n\n    Positive velocities are assumed to redshift the spectral quantity,\n    while negative velocities blueshift the spectral quantity.\n    '
    squantity = scoord.view(SpectralQuantity)
    beta = velocity / c
    doppler_factor = np.sqrt((1 + beta) / (1 - beta))
    if squantity.unit.is_equivalent(u.m):
        return squantity * doppler_factor
    elif squantity.unit.is_equivalent(u.Hz) or squantity.unit.is_equivalent(u.eV) or squantity.unit.is_equivalent(1 / u.m):
        return squantity / doppler_factor
    elif squantity.unit.is_equivalent(KMS):
        return (squantity.to(u.Hz) / doppler_factor).to(squantity.unit)
    else:
        raise RuntimeError(f'Unexpected units in velocity shift: {squantity.unit}. This should not happen, so please report this in the astropy issue tracker!')

def update_differentials_to_match(original, velocity_reference, preserve_observer_frame=False):
    if False:
        return 10
    '\n    Given an original coordinate object, update the differentials so that\n    the final coordinate is at the same location as the original coordinate\n    but co-moving with the velocity reference object.\n\n    If preserve_original_frame is set to True, the resulting object will be in\n    the frame of the original coordinate, otherwise it will be in the frame of\n    the velocity reference.\n    '
    if not velocity_reference.data.differentials:
        raise ValueError('Reference frame has no velocities')
    if 'obstime' in velocity_reference.frame_attributes and hasattr(original, 'obstime'):
        velocity_reference = velocity_reference.replicate(obstime=original.obstime)
    original_icrs = original.transform_to(ICRS())
    velocity_reference_icrs = velocity_reference.transform_to(ICRS())
    differentials = velocity_reference_icrs.data.represent_as(CartesianRepresentation, CartesianDifferential).differentials
    data_with_differentials = original_icrs.data.represent_as(CartesianRepresentation).with_differentials(differentials)
    final_icrs = original_icrs.realize_frame(data_with_differentials)
    if preserve_observer_frame:
        final = final_icrs.transform_to(original)
    else:
        final = final_icrs.transform_to(velocity_reference)
    return final.replicate(representation_type=CartesianRepresentation, differential_type=CartesianDifferential)

def attach_zero_velocities(coord):
    if False:
        return 10
    '\n    Set the differentials to be stationary on a coordinate object.\n    '
    new_data = coord.cartesian.with_differentials(ZERO_VELOCITIES)
    return coord.realize_frame(new_data)

def _get_velocities(coord):
    if False:
        print('Hello World!')
    if 's' in coord.data.differentials:
        return coord.velocity
    else:
        return ZERO_VELOCITIES

class SpectralCoord(SpectralQuantity):
    """
    A spectral coordinate with its corresponding unit.

    .. note:: The |SpectralCoord| class is new in Astropy v4.1 and should be
              considered experimental at this time. Note that we do not fully
              support cases where the observer and target are moving
              relativistically relative to each other, so care should be taken
              in those cases. It is possible that there will be API changes in
              future versions of Astropy based on user feedback. If you have
              specific ideas for how it might be improved, please  let us know
              on the `astropy-dev mailing list`_ or at
              http://feedback.astropy.org.

    Parameters
    ----------
    value : ndarray or `~astropy.units.Quantity` or `SpectralCoord`
        Spectral values, which should be either wavelength, frequency,
        energy, wavenumber, or velocity values.
    unit : unit-like
        Unit for the given spectral values.
    observer : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional
        The coordinate (position and velocity) of observer. If no velocities
        are present on this object, the observer is assumed to be stationary
        relative to the frame origin.
    target : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional
        The coordinate (position and velocity) of target. If no velocities
        are present on this object, the target is assumed to be stationary
        relative to the frame origin.
    radial_velocity : `~astropy.units.Quantity` ['speed'], optional
        The radial velocity of the target with respect to the observer. This
        can only be specified if ``redshift`` is not specified.
    redshift : float, optional
        The relativistic redshift of the target with respect to the observer.
        This can only be specified if ``radial_velocity`` cannot be specified.
    doppler_rest : `~astropy.units.Quantity`, optional
        The rest value to use when expressing the spectral value as a velocity.
    doppler_convention : str, optional
        The Doppler convention to use when expressing the spectral value as a velocity.
    """

    @u.quantity_input(radial_velocity=u.km / u.s)
    def __new__(cls, value, unit=None, observer=None, target=None, radial_velocity=None, redshift=None, **kwargs):
        if False:
            while True:
                i = 10
        obj = super().__new__(cls, value, unit=unit, **kwargs)
        if target is not None and observer is not None:
            if radial_velocity is not None or redshift is not None:
                raise ValueError('Cannot specify radial velocity or redshift if both target and observer are specified')
        if redshift is not None:
            if radial_velocity is not None:
                raise ValueError('Cannot set both a radial velocity and redshift')
            redshift = u.Quantity(redshift)
            if not redshift.unit.is_equivalent(u.one):
                raise u.UnitsError('redshift should be dimensionless')
            radial_velocity = redshift.to(u.km / u.s, u.doppler_redshift())
        if observer is None:
            observer = getattr(value, 'observer', None)
        if target is None:
            target = getattr(value, 'target', None)
        if observer is None or target is None:
            if radial_velocity is None:
                radial_velocity = getattr(value, 'radial_velocity', None)
        obj._radial_velocity = radial_velocity
        obj._observer = cls._validate_coordinate(observer, label='observer')
        obj._target = cls._validate_coordinate(target, label='target')
        return obj

    def __array_finalize__(self, obj):
        if False:
            while True:
                i = 10
        super().__array_finalize__(obj)
        self._radial_velocity = getattr(obj, '_radial_velocity', None)
        self._observer = getattr(obj, '_observer', None)
        self._target = getattr(obj, '_target', None)

    @staticmethod
    def _validate_coordinate(coord, label=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks the type of the frame and whether a velocity differential and a\n        distance has been defined on the frame object.\n\n        If no distance is defined, the target is assumed to be "really far\n        away", and the observer is assumed to be "in the solar system".\n\n        Parameters\n        ----------\n        coord : `~astropy.coordinates.BaseCoordinateFrame`\n            The new frame to be used for target or observer.\n        label : str, optional\n            The name of the object being validated (e.g. \'target\' or \'observer\'),\n            which is then used in error messages.\n        '
        if coord is None:
            return
        if not issubclass(coord.__class__, BaseCoordinateFrame):
            if isinstance(coord, SkyCoord):
                coord = coord.frame
            else:
                raise TypeError(f'{label} must be a SkyCoord or coordinate frame instance')
        with np.errstate(all='ignore'):
            distance = getattr(coord, 'distance', None)
        if distance is not None and distance.unit.physical_type == 'dimensionless':
            coord = SkyCoord(coord, distance=DEFAULT_DISTANCE)
            warnings.warn(f'Distance on coordinate object is dimensionless, an arbitrary distance value of {DEFAULT_DISTANCE} will be set instead.', NoDistanceWarning)
        if 's' not in coord.data.differentials:
            warnings.warn(f'No velocity defined on frame, assuming {ZERO_VELOCITIES}.', NoVelocityWarning)
            coord = attach_zero_velocities(coord)
        return coord

    def replicate(self, value=None, unit=None, observer=None, target=None, radial_velocity=None, redshift=None, doppler_convention=None, doppler_rest=None, copy=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a replica of the `SpectralCoord`, optionally changing the\n        values or attributes.\n\n        Note that no conversion is carried out by this method - this keeps\n        all the values and attributes the same, except for the ones explicitly\n        passed to this method which are changed.\n\n        If ``copy`` is set to `True` then a full copy of the internal arrays\n        will be made.  By default the replica will use a reference to the\n        original arrays when possible to save memory.\n\n        Parameters\n        ----------\n        value : ndarray or `~astropy.units.Quantity` or `SpectralCoord`, optional\n            Spectral values, which should be either wavelength, frequency,\n            energy, wavenumber, or velocity values.\n        unit : unit-like\n            Unit for the given spectral values.\n        observer : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional\n            The coordinate (position and velocity) of observer.\n        target : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional\n            The coordinate (position and velocity) of target.\n        radial_velocity : `~astropy.units.Quantity` ['speed'], optional\n            The radial velocity of the target with respect to the observer.\n        redshift : float, optional\n            The relativistic redshift of the target with respect to the observer.\n        doppler_rest : `~astropy.units.Quantity`, optional\n            The rest value to use when expressing the spectral value as a velocity.\n        doppler_convention : str, optional\n            The Doppler convention to use when expressing the spectral value as a velocity.\n        copy : bool, optional\n            If `True`, and ``value`` is not specified, the values are copied to\n            the new `SkyCoord` - otherwise a reference to the same values is used.\n\n        Returns\n        -------\n        sc : `SpectralCoord` object\n            Replica of this object\n        "
        if isinstance(value, u.Quantity):
            if unit is not None:
                raise ValueError('Cannot specify value as a Quantity and also specify unit')
            else:
                (value, unit) = (value.value, value.unit)
        value = value if value is not None else self.value
        unit = unit or self.unit
        observer = self._validate_coordinate(observer) or self.observer
        target = self._validate_coordinate(target) or self.target
        doppler_convention = doppler_convention or self.doppler_convention
        doppler_rest = doppler_rest or self.doppler_rest
        if copy:
            value = value.copy()
        if (self.observer is None or self.target is None) and radial_velocity is None and (redshift is None):
            radial_velocity = self.radial_velocity
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NoVelocityWarning)
            return self.__class__(value=value, unit=unit, observer=observer, target=target, radial_velocity=radial_velocity, redshift=redshift, doppler_convention=doppler_convention, doppler_rest=doppler_rest, copy=False)

    @property
    def quantity(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert the ``SpectralCoord`` to a `~astropy.units.Quantity`.\n        Equivalent to ``self.view(u.Quantity)``.\n\n        Returns\n        -------\n        `~astropy.units.Quantity`\n            This object viewed as a `~astropy.units.Quantity`.\n\n        '
        return self.view(u.Quantity)

    @property
    def observer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The coordinates of the observer.\n\n        If set, and a target is set as well, this will override any explicit\n        radial velocity passed in.\n\n        Returns\n        -------\n        `~astropy.coordinates.BaseCoordinateFrame`\n            The astropy coordinate frame representing the observation.\n        '
        return self._observer

    @observer.setter
    def observer(self, value):
        if False:
            while True:
                i = 10
        if self.observer is not None:
            raise ValueError('observer has already been set')
        self._observer = self._validate_coordinate(value, label='observer')
        if self._target is not None:
            self._radial_velocity = None

    @property
    def target(self):
        if False:
            i = 10
            return i + 15
        '\n        The coordinates of the target being observed.\n\n        If set, and an observer is set as well, this will override any explicit\n        radial velocity passed in.\n\n        Returns\n        -------\n        `~astropy.coordinates.BaseCoordinateFrame`\n            The astropy coordinate frame representing the target.\n        '
        return self._target

    @target.setter
    def target(self, value):
        if False:
            while True:
                i = 10
        if self.target is not None:
            raise ValueError('target has already been set')
        self._target = self._validate_coordinate(value, label='target')
        if self._observer is not None:
            self._radial_velocity = None

    @property
    def radial_velocity(self):
        if False:
            while True:
                i = 10
        "\n        Radial velocity of target relative to the observer.\n\n        Returns\n        -------\n        `~astropy.units.Quantity` ['speed']\n            Radial velocity of target.\n\n        Notes\n        -----\n        This is different from the ``.radial_velocity`` property of a\n        coordinate frame in that this calculates the radial velocity with\n        respect to the *observer*, not the origin of the frame.\n        "
        if self._observer is None or self._target is None:
            if self._radial_velocity is None:
                return 0 * KMS
            else:
                return self._radial_velocity
        else:
            return self._calculate_radial_velocity(self._observer, self._target, as_scalar=True)

    @property
    def redshift(self):
        if False:
            print('Hello World!')
        '\n        Redshift of target relative to observer. Calculated from the radial\n        velocity.\n\n        Returns\n        -------\n        `astropy.units.Quantity`\n            Redshift of target.\n        '
        return self.radial_velocity.to(u.dimensionless_unscaled, u.doppler_redshift())

    @staticmethod
    def _calculate_radial_velocity(observer, target, as_scalar=False):
        if False:
            while True:
                i = 10
        "\n        Compute the line-of-sight velocity from the observer to the target.\n\n        Parameters\n        ----------\n        observer : `~astropy.coordinates.BaseCoordinateFrame`\n            The frame of the observer.\n        target : `~astropy.coordinates.BaseCoordinateFrame`\n            The frame of the target.\n        as_scalar : bool\n            If `True`, the magnitude of the velocity vector will be returned,\n            otherwise the full vector will be returned.\n\n        Returns\n        -------\n        `~astropy.units.Quantity` ['speed']\n            The radial velocity of the target with respect to the observer.\n        "
        observer_icrs = observer.transform_to(ICRS())
        target_icrs = target.transform_to(ICRS())
        pos_hat = SpectralCoord._normalized_position_vector(observer_icrs, target_icrs)
        d_vel = target_icrs.velocity - observer_icrs.velocity
        vel_mag = pos_hat.dot(d_vel)
        if as_scalar:
            return vel_mag
        else:
            return vel_mag * pos_hat

    @staticmethod
    def _normalized_position_vector(observer, target):
        if False:
            i = 10
            return i + 15
        '\n        Calculate the normalized position vector between two frames.\n\n        Parameters\n        ----------\n        observer : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`\n            The observation frame or coordinate.\n        target : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`\n            The target frame or coordinate.\n\n        Returns\n        -------\n        pos_hat : `BaseRepresentation`\n            Position representation.\n        '
        d_pos = target.cartesian.without_differentials() - observer.cartesian.without_differentials()
        dp_norm = d_pos.norm()
        dp_norm[dp_norm == 0] = 1 * dp_norm.unit
        pos_hat = d_pos / dp_norm
        return pos_hat

    @u.quantity_input(velocity=u.km / u.s)
    def with_observer_stationary_relative_to(self, frame, velocity=None, preserve_observer_frame=False):
        if False:
            print('Hello World!')
        "\n        A new  `SpectralCoord` with the velocity of the observer altered,\n        but not the position.\n\n        If a coordinate frame is specified, the observer velocities will be\n        modified to be stationary in the specified frame. If a coordinate\n        instance is specified, optionally with non-zero velocities, the\n        observer velocities will be updated so that the observer is co-moving\n        with the specified coordinates.\n\n        Parameters\n        ----------\n        frame : str, `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`\n            The observation frame in which the observer will be stationary. This\n            can be the name of a frame (e.g. 'icrs'), a frame class, frame instance\n            with no data, or instance with data. This can optionally include\n            velocities.\n        velocity : `~astropy.units.Quantity` or `~astropy.coordinates.CartesianDifferential`, optional\n            If ``frame`` does not contain velocities, these can be specified as\n            a 3-element `~astropy.units.Quantity`. In the case where this is\n            also not specified, the velocities default to zero.\n        preserve_observer_frame : bool\n            If `True`, the final observer frame class will be the same as the\n            original one, and if `False` it will be the frame of the velocity\n            reference class.\n\n        Returns\n        -------\n        new_coord : `SpectralCoord`\n            The new coordinate object representing the spectral data\n            transformed based on the observer's new velocity frame.\n        "
        if self.observer is None or self.target is None:
            raise ValueError('This method can only be used if both observer and target are defined on the SpectralCoord.')
        if isinstance(frame, SkyCoord):
            frame = frame.frame
        if isinstance(frame, BaseCoordinateFrame):
            if not frame.has_data:
                frame = frame.realize_frame(CartesianRepresentation(0 * u.km, 0 * u.km, 0 * u.km))
            if frame.data.differentials:
                if velocity is not None:
                    raise ValueError('frame already has differentials, cannot also specify velocity')
            else:
                if velocity is None:
                    differentials = ZERO_VELOCITIES
                else:
                    differentials = CartesianDifferential(velocity)
                frame = frame.realize_frame(frame.data.with_differentials(differentials))
        if isinstance(frame, (type, str)):
            if isinstance(frame, type):
                frame_cls = frame
            elif isinstance(frame, str):
                frame_cls = frame_transform_graph.lookup_name(frame)
            if velocity is None:
                velocity = (0 * u.m / u.s, 0 * u.m / u.s, 0 * u.m / u.s)
            elif velocity.shape != (3,):
                raise ValueError('velocity should be a Quantity vector with 3 elements')
            frame = frame_cls(0 * u.m, 0 * u.m, 0 * u.m, *velocity, representation_type='cartesian', differential_type='cartesian')
        observer = update_differentials_to_match(self.observer, frame, preserve_observer_frame=preserve_observer_frame)
        init_obs_vel = self._calculate_radial_velocity(self.observer, self.target, as_scalar=True)
        fin_obs_vel = self._calculate_radial_velocity(observer, self.target, as_scalar=True)
        new_data = _apply_relativistic_doppler_shift(self, fin_obs_vel - init_obs_vel)
        new_coord = self.replicate(value=new_data, observer=observer)
        return new_coord

    def with_radial_velocity_shift(self, target_shift=None, observer_shift=None):
        if False:
            while True:
                i = 10
        "\n        Apply a velocity shift to this spectral coordinate.\n\n        The shift can be provided as a redshift (float value) or radial\n        velocity (`~astropy.units.Quantity` with physical type of 'speed').\n\n        Parameters\n        ----------\n        target_shift : float or `~astropy.units.Quantity` ['speed']\n            Shift value to apply to current target.\n        observer_shift : float or `~astropy.units.Quantity` ['speed']\n            Shift value to apply to current observer.\n\n        Returns\n        -------\n        `SpectralCoord`\n            New spectral coordinate with the target/observer velocity changed\n            to incorporate the shift. This is always a new object even if\n            ``target_shift`` and ``observer_shift`` are both `None`.\n        "
        if observer_shift is not None and (self.target is None or self.observer is None):
            raise ValueError('Both an observer and target must be defined before applying a velocity shift.')
        for arg in [x for x in [target_shift, observer_shift] if x is not None]:
            if isinstance(arg, u.Quantity) and (not arg.unit.is_equivalent((u.one, KMS))):
                raise u.UnitsError("Argument must have unit physical type 'speed' for radial velocty or 'dimensionless' for redshift.")
        if target_shift is None:
            if self._observer is None or self._target is None:
                return self.replicate()
            target_shift = 0 * KMS
        else:
            target_shift = u.Quantity(target_shift)
            if target_shift.unit.physical_type == 'dimensionless':
                target_shift = target_shift.to(u.km / u.s, u.doppler_redshift())
            if self._observer is None or self._target is None:
                return self.replicate(value=_apply_relativistic_doppler_shift(self, target_shift), radial_velocity=self.radial_velocity + target_shift)
        if observer_shift is None:
            observer_shift = 0 * KMS
        else:
            observer_shift = u.Quantity(observer_shift)
            if observer_shift.unit.physical_type == 'dimensionless':
                observer_shift = observer_shift.to(u.km / u.s, u.doppler_redshift())
        target_icrs = self._target.transform_to(ICRS())
        observer_icrs = self._observer.transform_to(ICRS())
        pos_hat = SpectralCoord._normalized_position_vector(observer_icrs, target_icrs)
        target_velocity = _get_velocities(target_icrs) + target_shift * pos_hat
        observer_velocity = _get_velocities(observer_icrs) + observer_shift * pos_hat
        target_velocity = CartesianDifferential(target_velocity.xyz)
        observer_velocity = CartesianDifferential(observer_velocity.xyz)
        new_target = target_icrs.realize_frame(target_icrs.cartesian.with_differentials(target_velocity)).transform_to(self._target)
        new_observer = observer_icrs.realize_frame(observer_icrs.cartesian.with_differentials(observer_velocity)).transform_to(self._observer)
        init_obs_vel = self._calculate_radial_velocity(observer_icrs, target_icrs, as_scalar=True)
        fin_obs_vel = self._calculate_radial_velocity(new_observer, new_target, as_scalar=True)
        new_data = _apply_relativistic_doppler_shift(self, fin_obs_vel - init_obs_vel)
        return self.replicate(value=new_data, observer=new_observer, target=new_target)

    def to_rest(self):
        if False:
            print('Hello World!')
        '\n        Transforms the spectral axis to the rest frame.\n        '
        if self.observer is not None and self.target is not None:
            return self.with_observer_stationary_relative_to(self.target)
        result = _apply_relativistic_doppler_shift(self, -self.radial_velocity)
        return self.replicate(value=result, radial_velocity=0.0 * KMS, redshift=None)

    def __repr__(self):
        if False:
            while True:
                i = 10
        prefixstr = '<' + self.__class__.__name__ + ' '
        try:
            radial_velocity = self.radial_velocity
            redshift = self.redshift
        except ValueError:
            radial_velocity = redshift = 'Undefined'
        repr_items = [f'{prefixstr}']
        if self.observer is not None:
            observer_repr = indent(repr(self.observer), 14 * ' ').lstrip()
            repr_items.append(f'    observer: {observer_repr}')
        if self.target is not None:
            target_repr = indent(repr(self.target), 12 * ' ').lstrip()
            repr_items.append(f'    target: {target_repr}')
        if self._observer is not None and self._target is not None or self._radial_velocity is not None:
            if self.observer is not None and self.target is not None:
                repr_items.append('    observer to target (computed from above):')
            else:
                repr_items.append('    observer to target:')
            repr_items.append(f'      radial_velocity={radial_velocity}')
            repr_items.append(f'      redshift={redshift}')
        if self.doppler_rest is not None or self.doppler_convention is not None:
            repr_items.append(f'    doppler_rest={self.doppler_rest}')
            repr_items.append(f'    doppler_convention={self.doppler_convention}')
        arrstr = np.array2string(self.view(np.ndarray), separator=', ', prefix='  ')
        if len(repr_items) == 1:
            repr_items[0] += f'{arrstr}{self._unitstr:s}'
        else:
            repr_items[1] = '   (' + repr_items[1].lstrip()
            repr_items[-1] += ')'
            repr_items.append(f'  {arrstr}{self._unitstr:s}')
        return '\n'.join(repr_items) + '>'