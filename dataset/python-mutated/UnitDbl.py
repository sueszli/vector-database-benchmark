"""UnitDbl module."""
import functools
import operator
from matplotlib import _api

class UnitDbl:
    """Class UnitDbl in development."""
    allowed = {'m': (0.001, 'km'), 'km': (1, 'km'), 'mile': (1.609344, 'km'), 'rad': (1, 'rad'), 'deg': (0.0174532925199433, 'rad'), 'sec': (1, 'sec'), 'min': (60.0, 'sec'), 'hour': (3600, 'sec')}
    _types = {'km': 'distance', 'rad': 'angle', 'sec': 'time'}

    def __init__(self, value, units):
        if False:
            print('Hello World!')
        '\n        Create a new UnitDbl object.\n\n        Units are internally converted to km, rad, and sec.  The only\n        valid inputs for units are [m, km, mile, rad, deg, sec, min, hour].\n\n        The field UnitDbl.value will contain the converted value.  Use\n        the convert() method to get a specific type of units back.\n\n        = ERROR CONDITIONS\n        - If the input units are not in the allowed list, an error is thrown.\n\n        = INPUT VARIABLES\n        - value     The numeric value of the UnitDbl.\n        - units     The string name of the units the value is in.\n        '
        data = _api.check_getitem(self.allowed, units=units)
        self._value = float(value * data[0])
        self._units = data[1]

    def convert(self, units):
        if False:
            return 10
        '\n        Convert the UnitDbl to a specific set of units.\n\n        = ERROR CONDITIONS\n        - If the input units are not in the allowed list, an error is thrown.\n\n        = INPUT VARIABLES\n        - units     The string name of the units to convert to.\n\n        = RETURN VALUE\n        - Returns the value of the UnitDbl in the requested units as a floating\n          point number.\n        '
        if self._units == units:
            return self._value
        data = _api.check_getitem(self.allowed, units=units)
        if self._units != data[1]:
            raise ValueError(f'Error trying to convert to different units.\n    Invalid conversion requested.\n    UnitDbl: {self}\n    Units:   {units}\n')
        return self._value / data[0]

    def __abs__(self):
        if False:
            return 10
        'Return the absolute value of this UnitDbl.'
        return UnitDbl(abs(self._value), self._units)

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the negative value of this UnitDbl.'
        return UnitDbl(-self._value, self._units)

    def __bool__(self):
        if False:
            return 10
        'Return the truth value of a UnitDbl.'
        return bool(self._value)

    def _cmp(self, op, rhs):
        if False:
            return 10
        'Check that *self* and *rhs* share units; compare them using *op*.'
        self.checkSameUnits(rhs, 'compare')
        return op(self._value, rhs._value)
    __eq__ = functools.partialmethod(_cmp, operator.eq)
    __ne__ = functools.partialmethod(_cmp, operator.ne)
    __lt__ = functools.partialmethod(_cmp, operator.lt)
    __le__ = functools.partialmethod(_cmp, operator.le)
    __gt__ = functools.partialmethod(_cmp, operator.gt)
    __ge__ = functools.partialmethod(_cmp, operator.ge)

    def _binop_unit_unit(self, op, rhs):
        if False:
            return 10
        'Check that *self* and *rhs* share units; combine them using *op*.'
        self.checkSameUnits(rhs, op.__name__)
        return UnitDbl(op(self._value, rhs._value), self._units)
    __add__ = functools.partialmethod(_binop_unit_unit, operator.add)
    __sub__ = functools.partialmethod(_binop_unit_unit, operator.sub)

    def _binop_unit_scalar(self, op, scalar):
        if False:
            for i in range(10):
                print('nop')
        'Combine *self* and *scalar* using *op*.'
        return UnitDbl(op(self._value, scalar), self._units)
    __mul__ = functools.partialmethod(_binop_unit_scalar, operator.mul)
    __rmul__ = functools.partialmethod(_binop_unit_scalar, operator.mul)

    def __str__(self):
        if False:
            while True:
                i = 10
        'Print the UnitDbl.'
        return f'{self._value:g} *{self._units}'

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Print the UnitDbl.'
        return f"UnitDbl({self._value:g}, '{self._units}')"

    def type(self):
        if False:
            print('Hello World!')
        'Return the type of UnitDbl data.'
        return self._types[self._units]

    @staticmethod
    def range(start, stop, step=None):
        if False:
            print('Hello World!')
        '\n        Generate a range of UnitDbl objects.\n\n        Similar to the Python range() method.  Returns the range [\n        start, stop) at the requested step.  Each element will be a\n        UnitDbl object.\n\n        = INPUT VARIABLES\n        - start     The starting value of the range.\n        - stop      The stop value of the range.\n        - step      Optional step to use.  If set to None, then a UnitDbl of\n                      value 1 w/ the units of the start is used.\n\n        = RETURN VALUE\n        - Returns a list containing the requested UnitDbl values.\n        '
        if step is None:
            step = UnitDbl(1, start._units)
        elems = []
        i = 0
        while True:
            d = start + i * step
            if d >= stop:
                break
            elems.append(d)
            i += 1
        return elems

    def checkSameUnits(self, rhs, func):
        if False:
            return 10
        '\n        Check to see if units are the same.\n\n        = ERROR CONDITIONS\n        - If the units of the rhs UnitDbl are not the same as our units,\n          an error is thrown.\n\n        = INPUT VARIABLES\n        - rhs     The UnitDbl to check for the same units\n        - func    The name of the function doing the check.\n        '
        if self._units != rhs._units:
            raise ValueError(f'Cannot {func} units of different types.\nLHS: {self._units}\nRHS: {rhs._units}')