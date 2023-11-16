import numbers
import numpy as np
from astropy.units import CompositeUnit, Unit, UnitConversionError, UnitsError, UnitTypeError, dimensionless_unscaled, photometric
from astropy.utils.compat.numpycompat import NUMPY_LT_2_0
from .core import FunctionQuantity, FunctionUnitBase
from .units import dB, dex, mag
__all__ = ['LogUnit', 'MagUnit', 'DexUnit', 'DecibelUnit', 'LogQuantity', 'Magnitude', 'Decibel', 'Dex', 'STmag', 'ABmag', 'M_bol', 'm_bol']

class LogUnit(FunctionUnitBase):
    """Logarithmic unit containing a physical one.

    Usually, logarithmic units are instantiated via specific subclasses
    such `~astropy.units.MagUnit`, `~astropy.units.DecibelUnit`, and
    `~astropy.units.DexUnit`.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the logarithmic function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, the same as the logarithmic unit set by the subclass.

    """

    @property
    def _default_function_unit(self):
        if False:
            i = 10
            return i + 15
        return dex

    @property
    def _quantity_class(self):
        if False:
            i = 10
            return i + 15
        return LogQuantity

    def from_physical(self, x):
        if False:
            print('Hello World!')
        'Transformation from value in physical to value in logarithmic units.\n        Used in equivalency.\n        '
        return dex.to(self._function_unit, np.log10(x))

    def to_physical(self, x):
        if False:
            i = 10
            return i + 15
        'Transformation from value in logarithmic to value in physical units.\n        Used in equivalency.\n        '
        return 10 ** self._function_unit.to(dex, x)

    def _add_and_adjust_physical_unit(self, other, sign_self, sign_other):
        if False:
            for i in range(10):
                print('nop')
        'Add/subtract LogUnit to/from another unit, and adjust physical unit.\n\n        self and other are multiplied by sign_self and sign_other, resp.\n\n        We wish to do:   ±lu_1 + ±lu_2  -> lu_f          (lu=logarithmic unit)\n                  and     pu_1^(±1) * pu_2^(±1) -> pu_f  (pu=physical unit)\n\n        Raises\n        ------\n        UnitsError\n            If function units are not equivalent.\n        '
        try:
            getattr(other, 'function_unit', other)._to(self._function_unit)
        except AttributeError:
            return NotImplemented
        except UnitsError:
            raise UnitsError('Can only add/subtract logarithmic units of compatible type.')
        other_physical_unit = getattr(other, 'physical_unit', dimensionless_unscaled)
        physical_unit = CompositeUnit(1, [self._physical_unit, other_physical_unit], [sign_self, sign_other])
        return self._copy(physical_unit)

    def __neg__(self):
        if False:
            return 10
        return self._copy(self.physical_unit ** (-1))

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self._add_and_adjust_physical_unit(other, +1, +1)

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return self._add_and_adjust_physical_unit(other, +1, +1)

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        return self._add_and_adjust_physical_unit(other, +1, -1)

    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._add_and_adjust_physical_unit(other, -1, +1)

class MagUnit(LogUnit):
    """Logarithmic physical units expressed in magnitudes.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the magnitude function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, this is ``mag``, but this allows one to use an equivalent
        unit such as ``2 mag``.
    """

    @property
    def _default_function_unit(self):
        if False:
            while True:
                i = 10
        return mag

    @property
    def _quantity_class(self):
        if False:
            i = 10
            return i + 15
        return Magnitude

class DexUnit(LogUnit):
    """Logarithmic physical units expressed in magnitudes.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the magnitude function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, this is ``dex``, but this allows one to use an equivalent
        unit such as ``0.5 dex``.
    """

    @property
    def _default_function_unit(self):
        if False:
            i = 10
            return i + 15
        return dex

    @property
    def _quantity_class(self):
        if False:
            return 10
        return Dex

    def to_string(self, format='generic'):
        if False:
            for i in range(10):
                print('nop')
        if format == 'cds':
            if self.physical_unit == dimensionless_unscaled:
                return '[-]'
            else:
                return f'[{self.physical_unit.to_string(format=format)}]'
        else:
            return super().to_string()

class DecibelUnit(LogUnit):
    """Logarithmic physical units expressed in dB.

    Parameters
    ----------
    physical_unit : `~astropy.units.Unit` or `string`
        Unit that is encapsulated within the decibel function unit.
        If not given, dimensionless.

    function_unit :  `~astropy.units.Unit` or `string`
        By default, this is ``dB``, but this allows one to use an equivalent
        unit such as ``2 dB``.
    """

    @property
    def _default_function_unit(self):
        if False:
            for i in range(10):
                print('nop')
        return dB

    @property
    def _quantity_class(self):
        if False:
            for i in range(10):
                print('nop')
        return Decibel

class LogQuantity(FunctionQuantity):
    """A representation of a (scaled) logarithm of a number with a unit.

    Parameters
    ----------
    value : number, `~astropy.units.Quantity`, `~astropy.units.LogQuantity`, or sequence of quantity-like.
        The numerical value of the logarithmic quantity. If a number or
        a `~astropy.units.Quantity` with a logarithmic unit, it will be
        converted to ``unit`` and the physical unit will be inferred from
        ``unit``.  If a `~astropy.units.Quantity` with just a physical unit,
        it will converted to the logarithmic unit, after, if necessary,
        converting it to the physical unit inferred from ``unit``.

    unit : str, `~astropy.units.UnitBase`, or `~astropy.units.FunctionUnitBase`, optional
        For an `~astropy.units.FunctionUnitBase` instance, the
        physical unit will be taken from it; for other input, it will be
        inferred from ``value``. By default, ``unit`` is set by the subclass.

    dtype : `~numpy.dtype`, optional
        The ``dtype`` of the resulting Numpy array or scalar that will
        hold the value.  If not provided, is is determined automatically
        from the input value.

    copy : bool, optional
        If `True` (default), then the value is copied.  Otherwise, a copy will
        only be made if ``__array__`` returns a copy, if value is a nested
        sequence, or if a copy is needed to satisfy an explicitly given
        ``dtype``.  (The `False` option is intended mostly for internal use,
        to speed up initialization where a copy is known to have been made.
        Use with care.)

    Examples
    --------
    Typically, use is made of an `~astropy.units.FunctionQuantity`
    subclass, as in::

        >>> import astropy.units as u
        >>> u.Magnitude(-2.5)
        <Magnitude -2.5 mag>
        >>> u.Magnitude(10.*u.count/u.second)
        <Magnitude -2.5 mag(ct / s)>
        >>> u.Decibel(1.*u.W, u.DecibelUnit(u.mW))  # doctest: +FLOAT_CMP
        <Decibel 30. dB(mW)>

    """
    _unit_class = LogUnit

    def __add__(self, other):
        if False:
            return 10
        new_unit = self.unit + getattr(other, 'unit', dimensionless_unscaled)
        result = self._function_view + getattr(other, '_function_view', other)
        return self._new_view(result, new_unit)

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__add__(other)

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        new_unit = self.unit + getattr(other, 'unit', dimensionless_unscaled)
        function_view = self._function_view
        function_view += getattr(other, '_function_view', other)
        self._set_unit(new_unit)
        return self

    def __sub__(self, other):
        if False:
            return 10
        new_unit = self.unit - getattr(other, 'unit', dimensionless_unscaled)
        result = self._function_view - getattr(other, '_function_view', other)
        return self._new_view(result, new_unit)

    def __rsub__(self, other):
        if False:
            i = 10
            return i + 15
        new_unit = self.unit.__rsub__(getattr(other, 'unit', dimensionless_unscaled))
        result = self._function_view.__rsub__(getattr(other, '_function_view', other))
        result = result.to(new_unit.function_unit)
        return self._new_view(result, new_unit)

    def __isub__(self, other):
        if False:
            i = 10
            return i + 15
        new_unit = self.unit - getattr(other, 'unit', dimensionless_unscaled)
        function_view = self._function_view
        function_view -= getattr(other, '_function_view', other)
        self._set_unit(new_unit)
        return self

    def __mul__(self, other):
        if False:
            return 10
        if isinstance(other, numbers.Number):
            new_physical_unit = self.unit.physical_unit ** other
            result = self.view(np.ndarray) * other
            return self._new_view(result, self.unit._copy(new_physical_unit))
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        if False:
            return 10
        return self.__mul__(other)

    def __imul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, numbers.Number):
            new_physical_unit = self.unit.physical_unit ** other
            function_view = self._function_view
            function_view *= other
            self._set_unit(self.unit._copy(new_physical_unit))
            return self
        else:
            return super().__imul__(other)

    def __truediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, numbers.Number):
            new_physical_unit = self.unit.physical_unit ** (1 / other)
            result = self.view(np.ndarray) / other
            return self._new_view(result, self.unit._copy(new_physical_unit))
        else:
            return super().__truediv__(other)

    def __itruediv__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, numbers.Number):
            new_physical_unit = self.unit.physical_unit ** (1 / other)
            function_view = self._function_view
            function_view /= other
            self._set_unit(self.unit._copy(new_physical_unit))
            return self
        else:
            return super().__itruediv__(other)

    def __pow__(self, other):
        if False:
            print('Hello World!')
        try:
            other = float(other)
        except TypeError:
            return NotImplemented
        new_unit = self.unit ** other
        new_value = self.view(np.ndarray) ** other
        return self._new_view(new_value, new_unit)

    def __ilshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        try:
            other = Unit(other)
        except UnitTypeError:
            return NotImplemented
        if not isinstance(other, self._unit_class):
            return NotImplemented
        try:
            factor = self.unit.physical_unit._to(other.physical_unit)
        except UnitConversionError:
            try:
                value = self._to_value(other)
            except UnitConversionError:
                return NotImplemented
            self.view(np.ndarray)[...] = value
        else:
            self.view(np.ndarray)[...] += self.unit.from_physical(factor)
        self._set_unit(other)
        return self

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        if False:
            i = 10
            return i + 15
        unit = self.unit.function_unit ** 2
        return self._wrap_function(np.var, axis, dtype, out=out, ddof=ddof, unit=unit)

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        if False:
            for i in range(10):
                print('nop')
        unit = self.unit._copy(dimensionless_unscaled)
        return self._wrap_function(np.std, axis, dtype, out=out, ddof=ddof, unit=unit)
    if NUMPY_LT_2_0:

        def ptp(self, axis=None, out=None):
            if False:
                print('Hello World!')
            unit = self.unit._copy(dimensionless_unscaled)
            return self._wrap_function(np.ptp, axis, out=out, unit=unit)
    else:

        def __array_function__(self, function, types, args, kwargs):
            if False:
                i = 10
                return i + 15
            if function is np.ptp:
                unit = self.unit._copy(dimensionless_unscaled)
                return self._wrap_function(np.ptp, *args[1:], unit=unit, **kwargs)
            else:
                return super().__array_function__(function, types, args, kwargs)

    def diff(self, n=1, axis=-1):
        if False:
            for i in range(10):
                print('nop')
        unit = self.unit._copy(dimensionless_unscaled)
        return self._wrap_function(np.diff, n, axis, unit=unit)

    def ediff1d(self, to_end=None, to_begin=None):
        if False:
            i = 10
            return i + 15
        unit = self.unit._copy(dimensionless_unscaled)
        return self._wrap_function(np.ediff1d, to_end, to_begin, unit=unit)
    _supported_functions = FunctionQuantity._supported_functions | {getattr(np, function) for function in ('var', 'std', 'ptp', 'diff', 'ediff1d')}

class Dex(LogQuantity):
    _unit_class = DexUnit

class Decibel(LogQuantity):
    _unit_class = DecibelUnit

class Magnitude(LogQuantity):
    _unit_class = MagUnit
dex._function_unit_class = DexUnit
dB._function_unit_class = DecibelUnit
mag._function_unit_class = MagUnit
STmag = MagUnit(photometric.STflux)
STmag.__doc__ = 'ST magnitude: STmag=-21.1 corresponds to 1 erg/s/cm2/A'
ABmag = MagUnit(photometric.ABflux)
ABmag.__doc__ = 'AB magnitude: ABmag=-48.6 corresponds to 1 erg/s/cm2/Hz'
M_bol = MagUnit(photometric.Bol)
M_bol.__doc__ = f'Absolute bolometric magnitude: M_bol=0 corresponds to L_bol0={photometric.Bol.si}'
m_bol = MagUnit(photometric.bol)
m_bol.__doc__ = f'Apparent bolometric magnitude: m_bol=0 corresponds to f_bol0={photometric.bol.si}'