import numpy as np
from astropy.units import Unit, si
from astropy.units import equivalencies as eq
from astropy.units.decorators import quantity_input
from astropy.units.quantity import Quantity, SpecificTypeQuantity
__all__ = ['SpectralQuantity']
__doctest_skip__ = ['SpectralQuantity.*']
KMS = si.km / si.s
SPECTRAL_UNITS = (si.Hz, si.m, si.J, si.m ** (-1), KMS)
DOPPLER_CONVENTIONS = {'radio': eq.doppler_radio, 'optical': eq.doppler_optical, 'relativistic': eq.doppler_relativistic}

class SpectralQuantity(SpecificTypeQuantity):
    """
    One or more value(s) with spectral units.

    The spectral units should be those for frequencies, wavelengths, energies,
    wavenumbers, or velocities (interpreted as Doppler velocities relative to a
    rest spectral value). The advantage of using this class over the regular
    `~astropy.units.Quantity` class is that in `SpectralQuantity`, the
    ``u.spectral`` equivalency is enabled by default (allowing automatic
    conversion between spectral units), and a preferred Doppler rest value and
    convention can be stored for easy conversion to/from velocities.

    Parameters
    ----------
    value : ndarray or `~astropy.units.Quantity` or `SpectralQuantity`
        Spectral axis data values.
    unit : unit-like
        Unit for the given data.
    doppler_rest : `~astropy.units.Quantity` ['speed'], optional
        The rest value to use for conversions from/to velocities
    doppler_convention : str, optional
        The convention to use when converting the spectral data to/from
        velocities.
    """
    _equivalent_unit = SPECTRAL_UNITS
    _include_easy_conversion_members = True

    def __new__(cls, value, unit=None, doppler_rest=None, doppler_convention=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        obj = super().__new__(cls, value, unit=unit, **kwargs)
        if doppler_rest is None:
            doppler_rest = getattr(value, 'doppler_rest', None)
        if doppler_convention is None:
            doppler_convention = getattr(value, 'doppler_convention', None)
        obj._doppler_rest = doppler_rest
        obj._doppler_convention = doppler_convention
        return obj

    def __array_finalize__(self, obj):
        if False:
            i = 10
            return i + 15
        super().__array_finalize__(obj)
        self._doppler_rest = getattr(obj, '_doppler_rest', None)
        self._doppler_convention = getattr(obj, '_doppler_convention', None)

    def __quantity_subclass__(self, unit):
        if False:
            for i in range(10):
                print('nop')
        if unit is self.unit:
            return (SpectralQuantity, True)
        else:
            return (Quantity, False)

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        if (function is np.multiply or (function is np.true_divide and inputs[0] is self)) and result.unit == self.unit or (function in (np.minimum, np.maximum, np.fmax, np.fmin) and method in ('reduce', 'reduceat')):
            result = result.view(self.__class__)
            result.__array_finalize__(self)
        else:
            if result is self:
                raise TypeError(f'Cannot store the result of this operation in {self.__class__.__name__}')
            if result.dtype.kind == 'b':
                result = result.view(np.ndarray)
            else:
                result = result.view(Quantity)
        return result

    @property
    def doppler_rest(self):
        if False:
            print('Hello World!')
        "\n        The rest value of the spectrum used for transformations to/from\n        velocity space.\n\n        Returns\n        -------\n        `~astropy.units.Quantity` ['speed']\n            Rest value as an astropy `~astropy.units.Quantity` object.\n        "
        return self._doppler_rest

    @doppler_rest.setter
    @quantity_input(value=SPECTRAL_UNITS)
    def doppler_rest(self, value):
        if False:
            for i in range(10):
                print('nop')
        "\n        New rest value needed for velocity-space conversions.\n\n        Parameters\n        ----------\n        value : `~astropy.units.Quantity` ['speed']\n            Rest value.\n        "
        if self._doppler_rest is not None:
            raise AttributeError('doppler_rest has already been set, and cannot be changed. Use the ``to`` method to convert the spectral values(s) to use a different rest value')
        self._doppler_rest = value

    @property
    def doppler_convention(self):
        if False:
            while True:
                i = 10
        "\n        The defined convention for conversions to/from velocity space.\n\n        Returns\n        -------\n        str\n            One of 'optical', 'radio', or 'relativistic' representing the\n            equivalency used in the unit conversions.\n        "
        return self._doppler_convention

    @doppler_convention.setter
    def doppler_convention(self, value):
        if False:
            i = 10
            return i + 15
        '\n        New velocity convention used for velocity space conversions.\n\n        Parameters\n        ----------\n        value\n\n        Notes\n        -----\n        More information on the equations dictating the transformations can be\n        found in the astropy documentation [1]_.\n\n        References\n        ----------\n        .. [1] Astropy documentation: https://docs.astropy.org/en/stable/units/equivalencies.html#spectral-doppler-equivalencies\n\n        '
        if self._doppler_convention is not None:
            raise AttributeError('doppler_convention has already been set, and cannot be changed. Use the ``to`` method to convert the spectral values(s) to use a different convention')
        if value is not None and value not in DOPPLER_CONVENTIONS:
            raise ValueError(f"doppler_convention should be one of {'/'.join(sorted(DOPPLER_CONVENTIONS))}")
        self._doppler_convention = value

    @quantity_input(doppler_rest=SPECTRAL_UNITS)
    def to(self, unit, equivalencies=[], doppler_rest=None, doppler_convention=None):
        if False:
            i = 10
            return i + 15
        "\n        Return a new `~astropy.coordinates.SpectralQuantity` object with the specified unit.\n\n        By default, the ``spectral`` equivalency will be enabled, as well as\n        one of the Doppler equivalencies if converting to/from velocities.\n\n        Parameters\n        ----------\n        unit : unit-like\n            An object that represents the unit to convert to. Must be\n            an `~astropy.units.UnitBase` object or a string parseable\n            by the `~astropy.units` package, and should be a spectral unit.\n        equivalencies : list of `~astropy.units.equivalencies.Equivalency`, optional\n            A list of equivalence pairs to try if the units are not\n            directly convertible (along with spectral).\n            See :ref:`astropy:unit_equivalencies`.\n            If not provided or ``[]``, spectral equivalencies will be used.\n            If `None`, no equivalencies will be applied at all, not even any\n            set globally or within a context.\n        doppler_rest : `~astropy.units.Quantity` ['speed'], optional\n            The rest value used when converting to/from velocities. This will\n            also be set at an attribute on the output\n            `~astropy.coordinates.SpectralQuantity`.\n        doppler_convention : {'relativistic', 'optical', 'radio'}, optional\n            The Doppler convention used when converting to/from velocities.\n            This will also be set at an attribute on the output\n            `~astropy.coordinates.SpectralQuantity`.\n\n        Returns\n        -------\n        `SpectralQuantity`\n            New spectral coordinate object with data converted to the new unit.\n        "
        unit = Unit(unit)
        if equivalencies is None:
            result = super().to(unit, equivalencies=None)
            result = result.view(self.__class__)
            result.__array_finalize__(self)
            return result
        if doppler_rest is None:
            doppler_rest = self._doppler_rest
        if doppler_convention is None:
            doppler_convention = self._doppler_convention
        elif doppler_convention not in DOPPLER_CONVENTIONS:
            raise ValueError(f"doppler_convention should be one of {'/'.join(sorted(DOPPLER_CONVENTIONS))}")
        if self.unit.is_equivalent(KMS) and unit.is_equivalent(KMS):
            if doppler_convention is not None and self._doppler_convention is None:
                raise ValueError('Original doppler_convention not set')
            if doppler_rest is not None and self._doppler_rest is None:
                raise ValueError('Original doppler_rest not set')
            if doppler_rest is None and doppler_convention is None:
                result = super().to(unit, equivalencies=equivalencies)
                result = result.view(self.__class__)
                result.__array_finalize__(self)
                return result
            elif (doppler_rest is None) is not (doppler_convention is None):
                raise ValueError('Either both or neither doppler_rest and doppler_convention should be defined for velocity conversions')
            vel_equiv1 = DOPPLER_CONVENTIONS[self._doppler_convention](self._doppler_rest)
            freq = super().to(si.Hz, equivalencies=equivalencies + vel_equiv1)
            vel_equiv2 = DOPPLER_CONVENTIONS[doppler_convention](doppler_rest)
            result = freq.to(unit, equivalencies=equivalencies + vel_equiv2)
        else:
            additional_equivalencies = eq.spectral()
            if self.unit.is_equivalent(KMS) or unit.is_equivalent(KMS):
                if doppler_convention is None:
                    raise ValueError('doppler_convention not set, cannot convert to/from velocities')
                if doppler_rest is None:
                    raise ValueError('doppler_rest not set, cannot convert to/from velocities')
                additional_equivalencies = additional_equivalencies + DOPPLER_CONVENTIONS[doppler_convention](doppler_rest)
            result = super().to(unit, equivalencies=equivalencies + additional_equivalencies)
        result = result.view(self.__class__)
        result.__array_finalize__(self)
        result._doppler_convention = doppler_convention
        result._doppler_rest = doppler_rest
        return result

    def to_value(self, unit=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if unit is None:
            return self.view(np.ndarray)
        return self.to(unit, *args, **kwargs).value