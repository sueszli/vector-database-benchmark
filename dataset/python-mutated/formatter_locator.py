import re
import warnings
import numpy as np
from matplotlib import rcParams
from astropy import units as u
from astropy.coordinates import Angle
from astropy.units import UnitsError
DMS_RE = re.compile('^dd(:mm(:ss(.(s)+)?)?)?$')
HMS_RE = re.compile('^hh(:mm(:ss(.(s)+)?)?)?$')
DDEC_RE = re.compile('^d(.(d)+)?$')
DMIN_RE = re.compile('^m(.(m)+)?$')
DSEC_RE = re.compile('^s(.(s)+)?$')
SCAL_RE = re.compile('^x(.(x)+)?$')
CUSTOM_UNITS = {u.degree: u.def_unit('custom_degree', represents=u.degree, format={'generic': '°', 'latex': '^\\circ', 'unicode': '°'}), u.arcmin: u.def_unit('custom_arcmin', represents=u.arcmin, format={'generic': "'", 'latex': '^\\prime', 'unicode': '′'}), u.arcsec: u.def_unit('custom_arcsec', represents=u.arcsec, format={'generic': '"', 'latex': '^{\\prime\\prime}', 'unicode': '″'}), u.hourangle: u.def_unit('custom_hourangle', represents=u.hourangle, format={'generic': 'h', 'latex': '^{\\mathrm{h}}', 'unicode': '$\\mathregular{^h}$'})}

class BaseFormatterLocator:
    """
    A joint formatter/locator.
    """

    def __init__(self, values=None, number=None, spacing=None, format=None, unit=None, format_unit=None):
        if False:
            print('Hello World!')
        if len([x for x in (values, number, spacing) if x is None]) < 2:
            raise ValueError('At most one of values/number/spacing can be specified')
        self._unit = unit
        self._format_unit = format_unit or unit
        if values is not None:
            self.values = values
        elif number is not None:
            self.number = number
        elif spacing is not None:
            self.spacing = spacing
        else:
            self.number = 5
        self.format = format

    @property
    def values(self):
        if False:
            for i in range(10):
                print('nop')
        return self._values

    @values.setter
    def values(self, values):
        if False:
            return 10
        if not isinstance(values, u.Quantity) or not values.ndim == 1:
            raise TypeError('values should be an astropy.units.Quantity array')
        if not values.unit.is_equivalent(self._unit):
            raise UnitsError(f'value should be in units compatible with coordinate units ({self._unit}) but found {values.unit}')
        self._number = None
        self._spacing = None
        self._values = values

    @property
    def number(self):
        if False:
            while True:
                i = 10
        return self._number

    @number.setter
    def number(self, number):
        if False:
            for i in range(10):
                print('nop')
        self._number = number
        self._spacing = None
        self._values = None

    @property
    def spacing(self):
        if False:
            for i in range(10):
                print('nop')
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if False:
            for i in range(10):
                print('nop')
        self._number = None
        self._spacing = spacing
        self._values = None

    def minor_locator(self, spacing, frequency, value_min, value_max):
        if False:
            return 10
        if self.values is not None:
            return [] * self._unit
        minor_spacing = spacing.value / frequency
        values = self._locate_values(value_min, value_max, minor_spacing)
        index = np.where(values % frequency == 0)
        index = index[0][0]
        values = np.delete(values, np.s_[index::frequency])
        return values * minor_spacing * self._unit

    @property
    def format_unit(self):
        if False:
            return 10
        return self._format_unit

    @format_unit.setter
    def format_unit(self, unit):
        if False:
            print('Hello World!')
        self._format_unit = u.Unit(unit)

    @staticmethod
    def _locate_values(value_min, value_max, spacing):
        if False:
            return 10
        imin = np.ceil(value_min / spacing)
        imax = np.floor(value_max / spacing)
        values = np.arange(imin, imax + 1, dtype=int)
        return values

class AngleFormatterLocator(BaseFormatterLocator):
    """
    A joint formatter/locator.

    Parameters
    ----------
    number : int, optional
        Number of ticks.
    """

    def __init__(self, values=None, number=None, spacing=None, format=None, unit=None, decimal=None, format_unit=None, show_decimal_unit=True):
        if False:
            print('Hello World!')
        if unit is None:
            unit = u.degree
        if format_unit is None:
            format_unit = unit
        if format_unit not in (u.degree, u.hourangle, u.hour):
            if decimal is False:
                raise UnitsError('Units should be degrees or hours when using non-decimal (sexagesimal) mode')
        self._decimal = decimal
        self._sep = None
        self.show_decimal_unit = show_decimal_unit
        super().__init__(values=values, number=number, spacing=spacing, format=format, unit=unit, format_unit=format_unit)

    @property
    def decimal(self):
        if False:
            print('Hello World!')
        decimal = self._decimal
        if self.format_unit not in (u.degree, u.hourangle, u.hour):
            if self._decimal is None:
                decimal = True
            elif self._decimal is False:
                raise UnitsError('Units should be degrees or hours when using non-decimal (sexagesimal) mode')
        elif self._decimal is None:
            decimal = False
        return decimal

    @decimal.setter
    def decimal(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._decimal = value

    @property
    def spacing(self):
        if False:
            return 10
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if False:
            while True:
                i = 10
        if spacing is not None and (not isinstance(spacing, u.Quantity) or spacing.unit.physical_type != 'angle'):
            raise TypeError('spacing should be an astropy.units.Quantity instance with units of angle')
        self._number = None
        self._spacing = spacing
        self._values = None

    @property
    def sep(self):
        if False:
            for i in range(10):
                print('nop')
        return self._sep

    @sep.setter
    def sep(self, separator):
        if False:
            print('Hello World!')
        self._sep = separator

    @property
    def format(self):
        if False:
            while True:
                i = 10
        return self._format

    @format.setter
    def format(self, value):
        if False:
            return 10
        self._format = value
        if value is None:
            return
        if DMS_RE.match(value) is not None:
            self._decimal = False
            self._format_unit = u.degree
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
                self._fields = 3
            else:
                self._precision = 0
                self._fields = value.count(':') + 1
        elif HMS_RE.match(value) is not None:
            self._decimal = False
            self._format_unit = u.hourangle
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
                self._fields = 3
            else:
                self._precision = 0
                self._fields = value.count(':') + 1
        elif DDEC_RE.match(value) is not None:
            self._decimal = True
            self._format_unit = u.degree
            self._fields = 1
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0
        elif DMIN_RE.match(value) is not None:
            self._decimal = True
            self._format_unit = u.arcmin
            self._fields = 1
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0
        elif DSEC_RE.match(value) is not None:
            self._decimal = True
            self._format_unit = u.arcsec
            self._fields = 1
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0
        else:
            raise ValueError(f'Invalid format: {value}')
        if self.spacing is not None and self.spacing < self.base_spacing:
            warnings.warn('Spacing is too small - resetting spacing to match format')
            self.spacing = self.base_spacing
        if self.spacing is not None:
            ratio = (self.spacing / self.base_spacing).decompose().value
            remainder = ratio - np.round(ratio)
            if abs(remainder) > 1e-10:
                warnings.warn('Spacing is not a multiple of base spacing - resetting spacing to match format')
                self.spacing = self.base_spacing * max(1, round(ratio))

    @property
    def base_spacing(self):
        if False:
            for i in range(10):
                print('nop')
        if self.decimal:
            spacing = self._format_unit / 10.0 ** self._precision
        elif self._fields == 1:
            spacing = 1.0 * u.degree
        elif self._fields == 2:
            spacing = 1.0 * u.arcmin
        elif self._fields == 3:
            if self._precision == 0:
                spacing = 1.0 * u.arcsec
            else:
                spacing = u.arcsec / 10.0 ** self._precision
        if self._format_unit is u.hourangle:
            spacing *= 15
        return spacing

    def locator(self, value_min, value_max):
        if False:
            for i in range(10):
                print('nop')
        if self.values is not None:
            return (self.values, 1.1 * u.arcsec)
        else:
            if value_min == value_max:
                return ([] * self._unit, 1 * u.arcsec)
            if self.spacing is not None:
                spacing_value = self.spacing.to_value(self._unit)
            elif self.number == 0:
                return ([] * self._unit, np.nan * self._unit)
            elif self.number is not None:
                dv = abs(float(value_max - value_min)) / self.number * self._unit
                if self.format is not None and dv < self.base_spacing:
                    spacing_value = self.base_spacing.to_value(self._unit)
                elif self.decimal:
                    from .utils import select_step_scalar
                    spacing_value = select_step_scalar(dv.to_value(self._format_unit)) * self._format_unit.to(self._unit)
                elif self._format_unit is u.degree:
                    from .utils import select_step_degree
                    spacing_value = select_step_degree(dv).to_value(self._unit)
                else:
                    from .utils import select_step_hour
                    spacing_value = select_step_hour(dv).to_value(self._unit)
            values = self._locate_values(value_min, value_max, spacing_value)
            return (values * spacing_value * self._unit, spacing_value * self._unit)

    def formatter(self, values, spacing, format='auto'):
        if False:
            while True:
                i = 10
        if not isinstance(values, u.Quantity) and values is not None:
            raise TypeError('values should be a Quantities array')
        if len(values) > 0:
            decimal = self.decimal
            unit = self._format_unit
            if unit is u.hour:
                unit = u.hourangle
            if self.format is None:
                if decimal:
                    spacing = spacing.to_value(unit)
                    fields = 0
                    precision = len(f'{spacing:.10f}'.replace('0', ' ').strip().split('.', 1)[1])
                else:
                    spacing = spacing.to_value(unit / 3600)
                    if spacing >= 3600:
                        fields = 1
                        precision = 0
                    elif spacing >= 60:
                        fields = 2
                        precision = 0
                    elif spacing >= 1:
                        fields = 3
                        precision = 0
                    else:
                        fields = 3
                        precision = -int(np.floor(np.log10(spacing)))
            else:
                fields = self._fields
                precision = self._precision
            is_latex = format == 'latex' or (format == 'auto' and rcParams['text.usetex'])
            if decimal:
                if self.show_decimal_unit:
                    sep = 'fromunit'
                    if is_latex:
                        fmt = 'latex'
                    elif unit is u.hourangle:
                        fmt = 'unicode'
                    else:
                        fmt = 'generic'
                    unit = CUSTOM_UNITS.get(unit, unit)
                else:
                    sep = 'fromunit'
                    fmt = None
            elif self.sep is not None:
                sep = self.sep
                fmt = None
            else:
                sep = 'fromunit'
                if unit == u.degree:
                    if is_latex:
                        fmt = 'latex'
                    else:
                        sep = ('°', "'", '"')
                        fmt = None
                elif format == 'ascii':
                    fmt = None
                elif is_latex:
                    fmt = 'latex'
                else:
                    sep = ('$\\mathregular{^h}$', '$\\mathregular{^m}$', '$\\mathregular{^s}$')
                    fmt = None
            angles = Angle(values)
            string = angles.to_string(unit=unit, precision=precision, decimal=decimal, fields=fields, sep=sep, format=fmt).tolist()
            return string
        else:
            return []

class ScalarFormatterLocator(BaseFormatterLocator):
    """
    A joint formatter/locator.
    """

    def __init__(self, values=None, number=None, spacing=None, format=None, unit=None, format_unit=None):
        if False:
            print('Hello World!')
        if unit is None:
            if spacing is not None:
                unit = spacing.unit
            elif values is not None:
                unit = values.unit
        format_unit = format_unit or unit
        super().__init__(values=values, number=number, spacing=spacing, format=format, unit=unit, format_unit=format_unit)

    @property
    def spacing(self):
        if False:
            i = 10
            return i + 15
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if False:
            print('Hello World!')
        if spacing is not None and (not isinstance(spacing, u.Quantity)):
            raise TypeError('spacing should be an astropy.units.Quantity instance')
        self._number = None
        self._spacing = spacing
        self._values = None

    @property
    def format(self):
        if False:
            print('Hello World!')
        return self._format

    @format.setter
    def format(self, value):
        if False:
            return 10
        self._format = value
        if value is None:
            return
        if SCAL_RE.match(value) is not None:
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0
            if self.spacing is not None and self.spacing < self.base_spacing:
                warnings.warn('Spacing is too small - resetting spacing to match format')
                self.spacing = self.base_spacing
            if self.spacing is not None:
                ratio = (self.spacing / self.base_spacing).decompose().value
                remainder = ratio - np.round(ratio)
                if abs(remainder) > 1e-10:
                    warnings.warn('Spacing is not a multiple of base spacing - resetting spacing to match format')
                    self.spacing = self.base_spacing * max(1, round(ratio))
        elif not value.startswith('%'):
            raise ValueError(f'Invalid format: {value}')

    @property
    def base_spacing(self):
        if False:
            for i in range(10):
                print('nop')
        return self._format_unit / 10.0 ** self._precision

    def locator(self, value_min, value_max):
        if False:
            i = 10
            return i + 15
        if self.values is not None:
            return (self.values, 1.1 * self._unit)
        else:
            if value_min == value_max:
                return ([] * self._unit, 0 * self._unit)
            if self.spacing is not None:
                spacing = self.spacing.to_value(self._unit)
            elif self.number is not None:
                dv = abs(float(value_max - value_min)) / self.number * self._unit
                if self.format is not None and (not self.format.startswith('%')) and (dv < self.base_spacing):
                    spacing = self.base_spacing.to_value(self._unit)
                else:
                    from .utils import select_step_scalar
                    spacing = select_step_scalar(dv.to_value(self._format_unit)) * self._format_unit.to(self._unit)
            values = self._locate_values(value_min, value_max, spacing)
            return (values * spacing * self._unit, spacing * self._unit)

    def formatter(self, values, spacing, format='auto'):
        if False:
            i = 10
            return i + 15
        if len(values) > 0:
            if self.format is None:
                if spacing.value < 1.0:
                    precision = -int(np.floor(np.log10(spacing.value)))
                else:
                    precision = 0
            elif self.format.startswith('%'):
                return [self.format % x.value for x in values]
            else:
                precision = self._precision
            return [('{0:.' + str(precision) + 'f}').format(x.to_value(self._format_unit)) for x in values]
        else:
            return []